"""
Tree view renderer for GrepMap.

Renders code snippets with specific lines of interest, using tree-sitter for
granular syntax highlighting. This view is optimized for showing actual code
context with proper indentation and semantic coloring.
"""

from typing import List, Set, Optional, Callable, Dict
from collections import defaultdict
from pathlib import Path
from io import StringIO

from rich.console import Console
from rich.text import Text
from grep_ast import TreeContext

from grepmap.core.types import RankedTag, Tag, DetailLevel
from grepmap.rendering.syntax import get_token_color


class TreeRenderer:
    """Renderer that displays code snippets with syntax highlighting.

    Uses tree-sitter for token-level coloring and grep_ast's TreeContext
    for fallback rendering. Shows actual code lines with proper indentation
    and visual hierarchy based on nesting depth.

    This renderer is ideal for showing concrete code examples and implementation
    details, complementing the higher-level DirectoryRenderer.
    """

    def __init__(
        self,
        root: Path,
        token_counter: Callable[[str], int],
        file_reader: Callable[[str], Optional[str]],
        color: bool = True
    ):
        """Initialize TreeRenderer.

        Args:
            root: Repository root path for resolving relative filenames
            token_counter: Function to count tokens in rendered output
            file_reader: Function to read file contents by absolute path
            color: Whether to apply syntax highlighting (default: True)
        """
        self.root = root
        self.token_counter = token_counter
        self.file_reader = file_reader
        self.color = color
        self.tree_context_cache: Dict[str, TreeContext] = {}

    def render(
        self,
        tags: List[RankedTag],
        chat_files: Set[str],
        detail: DetailLevel,
        overflow_tags: Optional[List[RankedTag]] = None
    ) -> str:
        """Render ranked tags as formatted code tree.

        Groups tags by file, sorts by importance, and renders code snippets
        with syntax highlighting at the specified detail level.

        Args:
            tags: Ranked tags to render
            chat_files: Files currently in chat context (for highlighting)
            detail: Detail level (unused in tree view - formatting is fixed)
            overflow_tags: Additional tags (unused in tree view)

        Returns:
            Formatted code snippets with file headers and rank annotations
        """
        if not tags:
            return ""

        # Group tags by file
        file_tags = defaultdict(list)
        for rt in tags:
            file_tags[rt.tag.rel_fname].append(rt)

        # Sort files by importance (max rank of their tags)
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rt.rank for rt in x[1]),
            reverse=True
        )

        tree_parts = []
        # Only show rank when multiple files (single file is always 1.0, no info)
        show_rank = len(sorted_files) > 1

        for rel_fname, file_tag_list in sorted_files:
            # Get lines of interest and tags
            lois = [rt.tag.line for rt in file_tag_list]
            file_tags_only = [rt.tag for rt in file_tag_list]

            # Find absolute filename
            abs_fname = str(self.root / rel_fname)

            # Get the max rank for the file
            max_rank = max(rt.rank for rt in file_tag_list)

            # Render the tree for this file (pass tags for semantic coloring)
            rendered = self._render_tree(abs_fname, rel_fname, lois, file_tags_only)
            if rendered:
                rendered_lines = rendered.splitlines()
                first_line = rendered_lines[0]
                code_lines = rendered_lines[1:]

                if show_rank:
                    tree_parts.append(
                        f"{first_line}\n"
                        f"(Rank: {max_rank:.4f})\n\n"
                        + "\n".join(code_lines)
                    )
                else:
                    tree_parts.append(
                        f"{first_line}\n"
                        + "\n".join(code_lines)
                    )

        return "\n\n".join(tree_parts)

    def estimate_tokens(self, output: str) -> int:
        """Estimate token count for rendered output.

        Args:
            output: Rendered tree view string

        Returns:
            Estimated token count using configured token counter
        """
        return self.token_counter(output)

    def _render_tree(
        self,
        abs_fname: str,
        rel_fname: str,
        lois: List[int],
        tags: Optional[List[Tag]] = None
    ) -> str:
        """Render a code snippet with specific lines of interest.

        Args:
            abs_fname: Absolute file path
            rel_fname: Relative file path
            lois: Lines of interest (line numbers)
            tags: Optional list of Tag objects for semantic coloring

        Returns:
            Formatted string with code lines and syntax highlighting
        """
        code = self.file_reader(abs_fname)
        if not code:
            return ""

        # Build a mapping from line number to tag for semantic coloring
        line_to_tag = {}
        if tags:
            for tag in tags:
                if tag.line in lois:
                    line_to_tag[tag.line] = tag

        # Use Rich for colored output with tree-sitter token-level coloring
        if self.color and tags:
            string_io = StringIO()
            console = Console(file=string_io, force_terminal=True, width=120)
            lines = code.splitlines()

            # Header
            header_text = Text(f"{rel_fname}:", style="bold blue")
            console.print(header_text)

            # Parse the file with tree-sitter for token-level coloring
            try:
                from grep_ast import filename_to_lang
                from grep_ast.tsl import get_parser

                lang_name = filename_to_lang(abs_fname)
                if lang_name:
                    parser = get_parser(lang_name)
                    tree = parser.parse(bytes(code, "utf-8"))
                else:
                    tree = None
            except Exception:
                # Fallback to simple coloring if tree-sitter fails
                tree = None

            # Build hierarchy depth map for indentation
            # Calculate proper nesting depth by tracing parent chains
            depth_map = {}
            line_to_full_tag = {tag.line: tag for tag in tags if tag.kind == 'def'}

            def calculate_depth(tag: Tag, visited: Optional[set] = None) -> int:
                """Recursively calculate nesting depth by tracing parent chain."""
                if visited is None:
                    visited = set()

                # Avoid infinite loops
                if tag.line in visited:
                    return 0
                visited.add(tag.line)

                # If no parent, depth is 0
                if not tag.parent_name or not tag.parent_line:
                    return 0

                # Find parent tag and recurse
                parent_tag = line_to_full_tag.get(tag.parent_line)
                if parent_tag:
                    return 1 + calculate_depth(parent_tag, visited)

                # Parent not in our tag list, assume depth 1
                return 1

            # Calculate depth for each tag
            for tag in tags:
                if tag.kind == 'def':
                    depth_map[tag.line] = calculate_depth(tag)

            # Render each line of interest with tree-sitter token coloring
            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    line_text = lines[loi-1].lstrip()  # Strip indent from source
                    indent_level = depth_map.get(loi, 0)
                    indent = "    " * indent_level  # 4 spaces per level

                    # Check if we should strip trailing colon for this line
                    tag = line_to_tag.get(loi)
                    should_strip_colon = tag and tag.node_type in ('function', 'method', 'class')

                    # Handle multi-line signatures: synthesize complete one-line signature
                    # Works for C-style languages (Python, JS, Java, C++, Rust, Go, etc.)
                    if tag and tag.node_type in ('function', 'method') and tag.signature:
                        # Check if signature is incomplete (unbalanced parens)
                        open_parens = line_text.count('(')
                        close_parens = line_text.count(')')
                        if open_parens > 0 and open_parens > close_parens:
                            # Multi-line signature detected - synthesize from SignatureInfo
                            sig_rendered = tag.signature.render(DetailLevel.HIGH)
                            # Preserve language keyword (def, func, fn, function, etc.)
                            keyword_match = line_text.split(tag.name)[0].strip()
                            line_text = f"{keyword_match} {tag.name}{sig_rendered}"

                    # Handle markdown headers: append section preview snippet
                    # Markdown headers (h1-h6) store section preview in signature.raw
                    markdown_preview = None
                    if tag and tag.node_type.startswith('h') and tag.signature and tag.signature.raw:
                        markdown_preview = tag.signature.render(DetailLevel.HIGH)

                    if tree:
                        # Use tree-sitter for granular token coloring
                        rich_line = self._colorize_line_with_tree_sitter(
                            loi, line_text, tree, code, indent
                        )

                        # Strip trailing colon from function/class definitions to save tokens
                        # Must be done AFTER colorization since colorizer uses original source
                        if should_strip_colon:
                            rich_line = self._strip_trailing_colon(rich_line)

                        # Append markdown section preview in dim style
                        if markdown_preview:
                            rich_line.append(f" {markdown_preview}", style="dim")

                        console.print(rich_line)
                    else:
                        # Fallback to simple coloring
                        if should_strip_colon:
                            line_text = line_text.rstrip(':').rstrip()
                        line_output = f"{loi:4d}: {indent}{line_text}"
                        if markdown_preview:
                            line_output += f" {markdown_preview}"
                        console.print(line_output, style="white")

            # Get the rendered output
            return string_io.getvalue().rstrip()

        # Use TreeContext for non-colored rendering (fallback)
        try:
            # TreeContext API changed - use simple fallback instead
            lines = code.splitlines()
            result_lines = [f"{rel_fname}:"]

            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    result_lines.append(f"{loi:4d}: {lines[loi-1]}")

            return "\n".join(result_lines)

        except Exception:
            # Fallback to simple line extraction
            lines = code.splitlines()
            result_lines = [f"{rel_fname}:"]

            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    result_lines.append(f"{loi:4d}: {lines[loi-1]}")

            return "\n".join(result_lines)

    def _strip_trailing_colon(self, rich_text: Text) -> Text:
        """Remove trailing colon and whitespace from a Rich Text object.

        Args:
            rich_text: Rich Text object with potential trailing colon

        Returns:
            New Rich Text object with trailing colon removed
        """
        # Convert to plain text to check for trailing colon
        plain = rich_text.plain
        if not plain.rstrip().endswith(':'):
            return rich_text

        # Find the last non-whitespace character position
        stripped_plain = plain.rstrip()
        if not stripped_plain.endswith(':'):
            return rich_text

        # Build new text without the trailing colon
        new_text = Text()
        target_len = len(stripped_plain) - 1  # Remove the colon

        # Walk through spans and copy up to target length
        for span in rich_text.spans:
            span_start = span.start
            span_end = span.end
            span_text = plain[span_start:span_end]

            if span_end <= target_len:
                # Entire span is before the cutoff
                new_text.append(span_text, style=span.style)
            elif span_start < target_len:
                # Span crosses the cutoff - truncate
                chars_to_keep = target_len - span_start
                new_text.append(span_text[:chars_to_keep], style=span.style)
                break
            else:
                # Span is after cutoff - skip
                break

        return new_text

    def _colorize_line_with_tree_sitter(
        self,
        line_num: int,
        line_text: str,
        tree,
        code_full: str,
        indent: str = ""
    ) -> Text:
        """Colorize a line using tree-sitter tokens for granular syntax highlighting.

        Args:
            line_num: Line number (1-indexed)
            line_text: The stripped code text for this line (without original indentation)
            tree: Tree-sitter parse tree
            code_full: The full file content (needed for byte positions)
            indent: Indentation string to prepend

        Returns:
            Rich Text object with token-level coloring
        """
        text = Text()

        # Line number and indent in dim cyan
        text.append(f"{line_num:4d}: {indent}", style="dim cyan")

        # Calculate byte offset for start of this line in the full file
        lines_before = code_full.splitlines()[:line_num-1]
        line_start_byte = sum(len(line.encode('utf-8')) + 1 for line in lines_before)  # +1 for newline

        # Get the actual line from the full code (with original indentation)
        actual_line = code_full.splitlines()[line_num-1] if line_num <= len(code_full.splitlines()) else line_text
        line_end_byte = line_start_byte + len(actual_line.encode('utf-8'))

        # Collect all leaf nodes that appear on this line
        tokens = []

        def collect_tokens(node):
            """Recursively collect all leaf tokens on the target line."""
            node_start_line = node.start_point[0] + 1
            node_end_line = node.end_point[0] + 1

            # Only process nodes on our target line
            if node_start_line == line_num == node_end_line:
                if not node.children or len(node.children) == 0:
                    # Leaf node - add as token
                    tokens.append((node.start_byte, node.end_byte, node.type))
                else:
                    # Non-leaf - recurse
                    for child in node.children:
                        collect_tokens(child)
            elif node_start_line <= line_num <= node_end_line:
                # Multi-line node - check children
                for child in node.children:
                    collect_tokens(child)

        collect_tokens(tree.root_node)

        # Sort tokens by start position
        tokens.sort(key=lambda x: x[0])

        # Build colored text from tokens
        # We need to map from original line (with indent) to stripped line
        original_indent_len = len(actual_line) - len(actual_line.lstrip())

        current_char_pos = 0  # Position in stripped line_text

        for start_byte, end_byte, node_type in tokens:
            # Skip if token is outside our line's byte range
            if end_byte <= line_start_byte or start_byte >= line_end_byte:
                continue

            # Calculate character position in the original line
            actual_line_bytes = actual_line.encode('utf-8')
            byte_offset = start_byte - line_start_byte
            char_pos_in_orig = len(actual_line_bytes[:byte_offset].decode('utf-8', errors='ignore'))
            token_text_bytes = actual_line_bytes[byte_offset:end_byte - line_start_byte]
            token_text = token_text_bytes.decode('utf-8', errors='ignore')

            # Adjust for stripped indentation
            char_pos_in_stripped = char_pos_in_orig - original_indent_len

            if char_pos_in_stripped < 0:
                # Token is in the indentation we stripped
                continue

            # Add any whitespace/text before this token
            if char_pos_in_stripped > current_char_pos:
                gap_text = line_text[current_char_pos:char_pos_in_stripped]
                text.append(gap_text, style="white")

            # Add the colored token
            color = get_token_color(node_type)
            text.append(token_text, style=color)

            current_char_pos = char_pos_in_stripped + len(token_text)

        # Add any remaining text
        if current_char_pos < len(line_text):
            text.append(line_text[current_char_pos:], style="white")

        return text
