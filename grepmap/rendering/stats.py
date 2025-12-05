"""
Stats/diagnostics renderer for GrepMap.

Renders a compact tree view showing file statistics (LOC, definition counts)
instead of code content. This mode packs significantly more files into the
token budget, ideal for rapid codebase complexity assessment.

Example output:
    grepmap/
      core/
        types.py                   275 LOC │  8 defs
        config.py                  142 LOC │  0 defs
      ranking/
        pagerank.py                231 LOC │  5 defs
        boosts.py                  167 LOC │  4 defs
"""

from typing import List, Set, Optional, Callable, Dict, Tuple
from collections import defaultdict
from pathlib import Path
from io import StringIO

from rich.console import Console
from rich.text import Text

from grepmap.core.types import RankedTag
from grepmap.core.config import STATS_MAX_FILES


class StatsRenderer:
    """Renderer that displays file statistics in a tree structure.

    Shows LOC (lines of code) and definition counts per file, organized
    hierarchically by directory. Designed for maximum file coverage within
    token budget - useful for understanding codebase scale and complexity.

    Key metrics per file:
    - LOC: Lines of code (non-empty lines)
    - defs: Definition count (functions, classes, methods)

    Directory aggregates show totals for quick complexity assessment.
    """

    def __init__(
        self,
        root: Path,
        file_reader: Callable[[str], Optional[str]],
        token_counter: Callable[[str], int],
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize StatsRenderer.

        Args:
            root: Repository root path for resolving relative filenames
            file_reader: Function to read file contents (for LOC counting)
            token_counter: Function to count tokens in rendered output
            verbose: Whether to output verbose logging
            output_handler: Function for info messages (default: print)
        """
        self.root = root
        self.file_reader = file_reader
        self.token_counter = token_counter
        self.verbose = verbose
        self.output_handler = output_handler or print

    def render(
        self,
        tags: List[RankedTag],
        chat_files: Set[str],
        file_stats: Optional[Dict[str, Tuple[int, int]]] = None,
        tree_view: bool = False
    ) -> str:
        """Render file statistics.

        Args:
            tags: Ranked tags (used for definition counting and file ordering)
            chat_files: Files currently in chat context (highlighted)
            file_stats: Optional pre-computed stats dict {rel_fname: (loc, defs)}
                       If not provided, will be computed from tags
            tree_view: If True, render as hierarchical tree; otherwise flat paths

        Returns:
            Formatted stats view with LOC and definition counts
        """
        if not tags:
            return ""

        # Compute stats from tags if not provided
        if file_stats is None:
            file_stats = self._compute_stats_from_tags(tags)

        # Sort files by rank (using max tag rank per file)
        file_ranks: Dict[str, float] = defaultdict(float)
        for rt in tags:
            if rt.tag.kind == 'def':
                file_ranks[rt.tag.rel_fname] = max(
                    file_ranks[rt.tag.rel_fname], rt.rank
                )

        sorted_files = sorted(
            file_stats.keys(),
            key=lambda f: file_ranks.get(f, 0),
            reverse=True
        )[:STATS_MAX_FILES]

        string_io = StringIO()
        console = Console(file=string_io, force_terminal=True, width=999999)

        if tree_view:
            # Hierarchical tree view
            tree = self._build_tree(sorted_files, file_stats)
            self._render_tree(console, tree, chat_files, indent=0)
        else:
            # Flat view with relative paths, sorted by importance
            self._render_flat(console, sorted_files, file_stats, chat_files)

        return string_io.getvalue().rstrip()

    def _compute_stats_from_tags(
        self,
        tags: List[RankedTag]
    ) -> Dict[str, Tuple[int, int]]:
        """Compute file statistics from tags.

        Args:
            tags: List of ranked tags

        Returns:
            Dict mapping rel_fname to (loc, def_count)
        """
        # Count definitions per file
        def_counts: Dict[str, int] = defaultdict(int)
        files_seen: Set[str] = set()

        for rt in tags:
            tag = rt.tag
            files_seen.add(tag.rel_fname)
            if tag.kind == 'def':
                def_counts[tag.rel_fname] += 1

        # Count LOC for each file
        stats: Dict[str, Tuple[int, int]] = {}
        for rel_fname in files_seen:
            abs_fname = str(self.root / rel_fname)
            loc = self._count_loc(abs_fname)
            stats[rel_fname] = (loc, def_counts[rel_fname])

        return stats

    def _count_loc(self, abs_fname: str) -> int:
        """Count non-empty lines in a file.

        Args:
            abs_fname: Absolute file path

        Returns:
            Number of non-empty lines (LOC)
        """
        content = self.file_reader(abs_fname)
        if not content:
            return 0
        return sum(1 for line in content.splitlines() if line.strip())

    def _build_tree(
        self,
        files: List[str],
        stats: Dict[str, Tuple[int, int]]
    ) -> Dict:
        """Build nested directory tree from file list.

        Args:
            files: List of relative file paths
            stats: Dict mapping rel_fname to (loc, def_count)

        Returns:
            Nested dict representing directory structure with stats
        """
        tree: Dict = {}

        for rel_fname in files:
            parts = Path(rel_fname).parts
            current = tree

            # Navigate/create directory path
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {'__is_dir__': True, '__children__': {}}
                current = current[part]['__children__']

            # Add file entry
            fname = parts[-1]
            loc, defs = stats.get(rel_fname, (0, 0))
            current[fname] = {
                '__is_dir__': False,
                '__loc__': loc,
                '__defs__': defs,
                '__path__': rel_fname
            }

        # Compute directory aggregates
        self._compute_aggregates(tree)

        return tree

    def _compute_aggregates(self, tree: Dict) -> Tuple[int, int, int]:
        """Recursively compute aggregate stats for directories.

        Args:
            tree: Directory tree dict

        Returns:
            Tuple of (total_loc, total_defs, file_count)
        """
        total_loc = 0
        total_defs = 0
        file_count = 0

        for name, entry in tree.items():
            if entry.get('__is_dir__'):
                loc, defs, count = self._compute_aggregates(entry['__children__'])
                entry['__loc__'] = loc
                entry['__defs__'] = defs
                entry['__file_count__'] = count
                total_loc += loc
                total_defs += defs
                file_count += count
            else:
                total_loc += entry.get('__loc__', 0)
                total_defs += entry.get('__defs__', 0)
                file_count += 1

        return total_loc, total_defs, file_count

    def _render_tree(
        self,
        console: Console,
        tree: Dict,
        chat_files: Set[str],
        indent: int = 0
    ) -> None:
        """Recursively render directory tree.

        Args:
            console: Rich console for output
            tree: Directory tree dict
            chat_files: Files in chat context (for highlighting)
            indent: Current indentation level
        """
        # Sort entries: directories first, then files, alphabetically within each
        dirs = []
        files = []

        for name, entry in tree.items():
            if entry.get('__is_dir__'):
                dirs.append((name, entry))
            else:
                files.append((name, entry))

        dirs.sort(key=lambda x: x[0].lower())
        files.sort(key=lambda x: x[0].lower())

        # Render directories
        for name, entry in dirs:
            loc = entry.get('__loc__', 0)
            defs = entry.get('__defs__', 0)
            file_count = entry.get('__file_count__', 0)

            line = Text()
            line.append("  " * indent)
            line.append(f"{name}/", style="bold blue")
            line.append(f" {file_count} files; {loc} loc; {defs} def", style="dim white")

            console.print(line, no_wrap=True)

            # Recurse into children
            self._render_tree(
                console, entry['__children__'], chat_files, indent + 1
            )

        # Render files
        for name, entry in files:
            loc = entry.get('__loc__', 0)
            defs = entry.get('__defs__', 0)
            path = entry.get('__path__', '')

            line = Text()
            line.append("  " * indent)

            # Highlight chat files
            if path in chat_files:
                line.append(name, style="bold green")
            else:
                line.append(name, style="cyan")

            line.append(f" {loc} loc; {defs} def", style="dim white")

            console.print(line, no_wrap=True)

    def _render_flat(
        self,
        console: Console,
        sorted_files: List[str],
        file_stats: Dict[str, Tuple[int, int]],
        chat_files: Set[str]
    ) -> None:
        """Render flat file list sorted by rank.

        Simple flat view with relative paths, no tree structure.
        Files sorted by PageRank importance. Stats first for easy scanning.

        Args:
            console: Rich console for output
            sorted_files: Files pre-sorted by rank
            file_stats: Dict mapping rel_fname to (loc, defs)
            chat_files: Files in chat context (for highlighting)
        """
        for rel_fname in sorted_files:
            loc, defs = file_stats.get(rel_fname, (0, 0))

            line = Text()
            line.append(f"{loc:>5} loc; {defs:>3} def  ", style="dim white")

            if rel_fname in chat_files:
                line.append(rel_fname, style="bold green")
            else:
                line.append(rel_fname, style="cyan")

            console.print(line, no_wrap=True)
