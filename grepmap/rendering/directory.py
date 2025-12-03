"""
Directory overview renderer for GrepMap.

Renders a hierarchical symbol overview showing files with their classes,
methods, fields, and top-level functions. This topology-preserving format
maximizes orientation signal for navigating large codebases.
"""

from typing import List, Set, Optional, Callable, Dict
from collections import defaultdict
from pathlib import Path
from io import StringIO
import shutil

from rich.console import Console
from rich.text import Text

from grepmap.core.types import RankedTag, Tag, DetailLevel


class DirectoryRenderer:
    """Renderer that displays hierarchical symbol overview.

    Shows files with their symbols organized by type and hierarchy:
    - Classes with their fields, properties, and methods indented
    - Top-level functions
    - Constants and variables

    Supports three detail levels:
    - LOW: Symbol names only
    - MEDIUM: Names with simplified signatures
    - HIGH: Full type annotations and signatures

    Includes a low-resolution "also in scope" section for overflow tags,
    extending orientation beyond the detailed view.
    """

    def __init__(
        self,
        root: Path,
        token_counter: Callable[[str], int],
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize DirectoryRenderer.

        Args:
            root: Repository root path for resolving relative filenames
            token_counter: Function to count tokens in rendered output
            verbose: Whether to output verbose logging
            output_handler: Function for info messages (default: print)
        """
        self.root = root
        self.token_counter = token_counter
        self.verbose = verbose
        self.output_handler = output_handler or print

    def render(
        self,
        tags: List[RankedTag],
        chat_files: Set[str],
        detail: DetailLevel,
        overflow_tags: Optional[List[RankedTag]] = None
    ) -> str:
        """Render ranked tags as hierarchical directory overview.

        Args:
            tags: Primary ranked tags for detailed display
            chat_files: Files currently in chat context
            detail: Level of detail for rendering (LOW/MEDIUM/HIGH)
            overflow_tags: Additional tags for low-resolution summaries

        Returns:
            Formatted directory overview with hierarchical symbol structure
        """
        if not tags:
            return ""

        string_io = StringIO()
        console = Console(file=string_io, force_terminal=True, width=999999)

        # Per-file seen pattern tracking for smart deduplication
        seen_patterns: Dict[str, set] = defaultdict(set)

        # Group tags by file
        file_tags: Dict[str, List[RankedTag]] = defaultdict(list)
        for rt in tags:
            if rt.tag.kind == 'def':
                file_tags[rt.tag.rel_fname].append(rt)

        # Sort files by importance
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rt.rank for rt in x[1]),
            reverse=True
        )

        term_width = shutil.get_terminal_size().columns

        for rel_fname, file_tag_list in sorted_files:
            if self.verbose:
                max_rank = max(rt.rank for rt in file_tag_list)
                self.output_handler(f"  {rel_fname}: rank={max_rank:.4f}")

            file_seen = seen_patterns[rel_fname]

            # Separate tags into categories
            classes: List[Tag] = []
            methods_by_class: Dict[str, List[Tag]] = defaultdict(list)
            top_level_funcs: List[Tag] = []
            constants: List[Tag] = []

            for rt in file_tag_list:
                tag = rt.tag
                if tag.node_type == 'class':
                    classes.append(tag)
                elif tag.node_type == 'function':
                    if tag.parent_name:
                        # Nested function or method - group under parent
                        methods_by_class[tag.parent_name].append(tag)
                    else:
                        top_level_funcs.append(tag)
                elif tag.node_type == 'method':
                    if tag.parent_name:
                        methods_by_class[tag.parent_name].append(tag)
                    else:
                        top_level_funcs.append(tag)
                elif tag.node_type in ('constant', 'variable'):
                    constants.append(tag)

            # File header
            text = Text()
            text.append(f"{rel_fname}:", style="bold blue")
            console.print(text, no_wrap=True)

            # Render classes with their fields and methods
            for class_tag in classes:
                class_display = self._render_symbol(class_tag, detail, file_seen)
                line = Text()
                line.append("  class ", style="magenta")
                line.append(class_display, style="bold cyan")

                # Get fields and methods for this class
                class_fields = class_tag.fields or ()
                all_class_methods = methods_by_class.get(class_tag.name, [])

                # Separate properties from regular methods
                # A property has 'property' in its signature decorators
                class_properties = []
                class_methods = []
                for m in all_class_methods:
                    if m.signature and 'property' in m.signature.decorators:
                        class_properties.append(m)
                    else:
                        class_methods.append(m)

                has_content = class_fields or class_properties or class_methods
                if has_content:
                    line.append(":", style="dim white")
                console.print(line, no_wrap=True)

                # Render fields indented under the class
                if class_fields:
                    field_names = [f.render(detail) for f in class_fields]
                    self._render_labeled_list(
                        console, field_names, indent="    ", label="fields",
                        term_width=term_width, label_color="dim magenta", item_color="bright_cyan"
                    )

                # Render properties indented under the class
                if class_properties:
                    self._render_symbol_list(
                        console, class_properties, detail, file_seen,
                        indent="    ", label="props", term_width=term_width,
                        label_color="dim magenta", item_color="bright_cyan"
                    )

                # Render methods indented under the class
                if class_methods:
                    self._render_symbol_list(
                        console, class_methods, detail, file_seen,
                        indent="    ", label="def", term_width=term_width,
                        label_color="dim magenta", item_color="yellow"
                    )

            # Render top-level functions
            if top_level_funcs:
                self._render_symbol_list(
                    console, top_level_funcs, detail, file_seen,
                    indent="  ", label="def", term_width=term_width,
                    label_color="magenta", item_color="green"
                )

            # Render constants
            if constants:
                self._render_symbol_list(
                    console, constants, detail, file_seen,
                    indent="  ", label="const", term_width=term_width,
                    label_color="magenta", item_color="bright_green"
                )

        # Low-resolution summary: show overflow tags (files beyond the detailed view)
        # This extends orientation at reduced fidelity
        if overflow_tags:
            shown_files = set(rel_fname for rel_fname, _ in sorted_files)

            # Collect ALL definitions from overflow, organized by file
            overflow_by_file: Dict[str, Dict[str, List[str]]] = defaultdict(
                lambda: {'classes': [], 'funcs': [], 'methods': [], 'const': []}
            )
            for rt in overflow_tags:
                tag = rt.tag
                if tag.kind == 'def' and tag.rel_fname not in shown_files:
                    if tag.node_type == 'class':
                        overflow_by_file[tag.rel_fname]['classes'].append(tag.name)
                    elif tag.node_type == 'function':
                        if tag.parent_name:
                            overflow_by_file[tag.rel_fname]['methods'].append(tag.name)
                        else:
                            overflow_by_file[tag.rel_fname]['funcs'].append(tag.name)
                    elif tag.node_type in ('constant', 'variable'):
                        overflow_by_file[tag.rel_fname]['const'].append(tag.name)

            if overflow_by_file:
                console.print("")  # Blank line
                console.print(Text("--- Also in scope ---", style="dim yellow"))

                # Sort by total symbols, limit display
                sorted_overflow = sorted(
                    overflow_by_file.items(),
                    key=lambda x: (
                        len(x[1]['classes']) * 3 +  # Weight classes highest
                        len(x[1]['funcs']) * 2 +
                        len(x[1]['methods']) +
                        len(x[1]['const'])
                    ),
                    reverse=True
                )[:30]

                for rel_fname, symbols in sorted_overflow:
                    line = Text("  ")
                    line.append(rel_fname, style="dim cyan")
                    line.append(": ", style="dim")

                    parts = []
                    if symbols['classes']:
                        classes = sorted(symbols['classes'])
                        if len(classes) <= 3:
                            parts.append(", ".join(classes))
                        else:
                            parts.append(f"{', '.join(classes[:3])} +{len(classes)-3}")

                    # Summarize other symbols
                    counts = []
                    if symbols['funcs']:
                        counts.append(f"{len(symbols['funcs'])}f")
                    if symbols['methods']:
                        counts.append(f"{len(symbols['methods'])}m")
                    if symbols['const']:
                        counts.append(f"{len(symbols['const'])}c")
                    if counts:
                        parts.append(" ".join(counts))

                    line.append(", ".join(parts) if parts else "...", style="dim white")
                    console.print(line, no_wrap=True)

        return string_io.getvalue().rstrip()

    def estimate_tokens(self, output: str) -> int:
        """Estimate token count for rendered output.

        Args:
            output: Rendered directory view string

        Returns:
            Estimated token count using configured token counter
        """
        return self.token_counter(output)

    def _render_symbol(
        self,
        tag: Tag,
        detail_level: DetailLevel,
        seen_patterns: Optional[set] = None
    ) -> str:
        """Render a symbol name at the specified detail level.

        Args:
            tag: The tag to render
            detail_level: LOW (name only), MEDIUM (with params), HIGH (full sig)
            seen_patterns: For HIGH detail, tracks seen param:type patterns for dedup

        Returns:
            Rendered symbol string (e.g., "connect", "connect(hobo, remote)",
            or "connect(hobo: HoboWindow) -> bool")
        """
        name = tag.name

        if detail_level == DetailLevel.LOW:
            # Just the name - fields are shown separately in hierarchy
            return name

        # MEDIUM or HIGH: include signature for functions
        if tag.node_type in ("function", "method") and tag.signature:
            sig_str = tag.signature.render(detail_level, seen_patterns)
            return f"{name}{sig_str}"

        # MEDIUM or HIGH: include fields for classes
        if tag.node_type == "class" and tag.fields:
            if detail_level == DetailLevel.MEDIUM:
                # Simplified: just field names
                field_names = ", ".join(f.name for f in tag.fields[:5])
                if len(tag.fields) > 5:
                    field_names += f", +{len(tag.fields) - 5}"
                return f"{name}({field_names})"
            else:  # HIGH
                # Full field types
                field_strs = [f.render(detail_level) for f in tag.fields[:5]]
                if len(tag.fields) > 5:
                    field_strs.append(f"+{len(tag.fields) - 5}")
                return f"{name}({', '.join(field_strs)})"

        return name

    def _render_labeled_list(
        self,
        console: Console,
        items: List[str],
        indent: str,
        label: str,
        term_width: int,
        label_color: str,
        item_color: str
    ) -> None:
        """Render a labeled list with inline label and aligned continuations.

        Output format:
            {indent}{label}: item1, item2, item3, ...
            {indent}        continuation aligned here
        """
        if not items:
            return

        # First line starts with label
        line = Text()
        line.append(indent, style="dim white")
        line.append(f"{label}: ", style=label_color)
        label_width = len(indent) + len(label) + 2  # +2 for ": "
        continuation_prefix = " " * label_width
        current_length = label_width

        for i, item in enumerate(items):
            sep = ", " if i > 0 else ""
            item_len = len(sep) + len(item)

            # Wrap if needed
            if i > 0 and current_length + item_len > term_width - 5:
                console.print(line, no_wrap=True)
                line = Text()
                line.append(continuation_prefix, style="dim white")
                current_length = label_width
                sep = ""

            if sep:
                line.append(sep, style="dim white")
            line.append(item, style=item_color)
            current_length += item_len

        if line:
            console.print(line, no_wrap=True)

    def _render_symbol_list(
        self,
        console: Console,
        tags_list: List[Tag],
        detail_level: DetailLevel,
        seen_patterns: set,
        indent: str,
        label: str,
        term_width: int,
        label_color: str,
        item_color: str
    ) -> None:
        """Render a labeled list of symbols with wrapping."""
        if not tags_list:
            return
        items = [self._render_symbol(tag, detail_level, seen_patterns) for tag in tags_list]
        self._render_labeled_list(console, items, indent, label, term_width, label_color, item_color)
