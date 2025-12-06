#!/usr/bin/env python3
"""
Standalone GrepMap Tool

A command-line tool that generates a "map" of a software repository,
highlighting important files and definitions based on their relevance.
Uses Tree-sitter for parsing and PageRank for ranking importance.
"""

import argparse
import sys
from pathlib import Path

from utils import count_tokens, read_text
from grepmap_class import GrepMap
from grepmap.discovery import find_source_files


def tool_output(*messages):
    """Print informational messages."""
    print(*messages, file=sys.stdout)


def tool_warning(message):
    """Print warning messages."""
    print(f"Warning: {message}", file=sys.stderr)


def tool_error(message):
    """Print error messages."""
    print(f"Error: {message}", file=sys.stderr)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a grep map showing important code structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                    # Map current directory
  %(prog)s src/ --map-tokens 2048  # Map src/ with 2048 token limit
  %(prog)s file1.py file2.py    # Map specific files
  %(prog)s --chat-files main.py --other-files src/  # Specify chat vs other files
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to include in the map"
    )
    
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--map-tokens",
        type=int,
        default=24576,
        help="Maximum tokens for the generated map (default: 24576)"
    )
    
    parser.add_argument(
        "--focus", "-f",
        nargs="*",
        help="Focus targets: file paths or search queries. Files get highest priority. "
             "Queries like 'authentication' match symbol names across the codebase."
    )

    # Backwards compatibility alias for --focus
    parser.add_argument(
        "--chat-files",
        nargs="*",
        dest="chat_files_compat",
        help="[Deprecated: use --focus] Files currently being edited"
    )

    parser.add_argument(
        "--other-files",
        nargs="*",
        help="Other files to consider for the map"
    )

    parser.add_argument(
        "--mentioned-files",
        nargs="*",
        help="Files explicitly mentioned (given higher priority)"
    )

    parser.add_argument(
        "--mentioned-idents",
        nargs="*",
        help="Identifiers explicitly mentioned (given higher priority)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name for token counting (default: gpt-4)"
    )
    
    parser.add_argument(
        "--max-context-window",
        type=int,
        help="Maximum context window size"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of caches"
    )

    parser.add_argument(
        "--exclude-unranked",
        action="store_true",
        help="Exclude files with Page Rank 0 from the map"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable semantic color highlighting (enabled by default)"
    )

    parser.add_argument(
        "--tree",
        action="store_true",
        help="Use detailed tree view instead of compact directory overview"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the tags cache before running (useful for testing or fixing corrupted cache)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show compact diagnostics (LOC, def counts) instead of code content. "
             "Packs many more files into token budget for rapid complexity assessment."
    )

    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive detail levels: focus files get HIGH detail, "
             "top-ranked get HIGH, mid-ranked get MEDIUM, low-ranked get LOW. "
             "Maximizes information density by varying resolution based on importance."
    )

    parser.add_argument(
        "--no-symbol-rank",
        action="store_true",
        help="Disable symbol-level PageRank (use file-level only). "
             "Symbol ranking surfaces individual important functions even from large files. "
             "Disable if you prefer uniform file-based ranking."
    )

    parser.add_argument(
        "--git-weight",
        action="store_true",
        help="Enable git-based weighting to boost recently modified files and hotspots. "
             "Uses recency (exponential decay) and churn (commit frequency) signals. "
             "Adds temporal awareness to favor actively developed code."
    )

    args = parser.parse_args()
    
    # Set up token counter with specified model
    def token_counter(text: str) -> int:
        return count_tokens(text, args.model)
    
    # Set up output handlers
    output_handlers = {
        'info': tool_output,
        'warning': tool_warning,
        'error': tool_error
    }
    
    # Process focus targets (combine --focus and deprecated --chat-files)
    # Focus targets can be file paths OR search queries for symbol matching
    focus_targets = []
    if args.focus:
        focus_targets.extend(args.focus)
    if args.chat_files_compat:
        focus_targets.extend(args.chat_files_compat)

    # Determine the list of unresolved path specifications that will form the 'other_files'
    # These can be files or directories. find_source_files will expand them.
    unresolved_paths_for_other_files_specs = []
    if args.other_files:  # If --other-files is explicitly provided, it's the source
        unresolved_paths_for_other_files_specs.extend(args.other_files)
    elif args.paths:  # Else, if positional paths are given, they are the source
        unresolved_paths_for_other_files_specs.extend(args.paths)
    # If neither, unresolved_paths_for_other_files_specs remains empty.

    # Now, expand all directory paths in unresolved_paths_for_other_files_specs into actual file lists
    # and collect all file paths. find_source_files handles both files and directories.
    effective_other_files_unresolved = []
    for path_spec_str in unresolved_paths_for_other_files_specs:
        effective_other_files_unresolved.extend(find_source_files(path_spec_str))

    other_files = [str(Path(f).resolve()) for f in effective_other_files_unresolved]

    if args.verbose:
        tool_output(f"Found {len(focus_targets)} focus targets, {len(other_files)} other files")

    # Auto-detect root if not explicitly provided and files are from outside current directory
    # This ensures that when you run "grepmap /some/other/path/file.py" it automatically
    # uses the correct repository root instead of the current directory
    if args.root == "." and other_files:
        # Try to find git repository root
        file_paths = [Path(f) for f in other_files]
        # Start with the first file's parent and walk up looking for .git
        search_path = file_paths[0].parent
        git_root = None
        # Walk up to find .git directory (max 20 levels to avoid infinite loop)
        for _ in range(20):
            if (search_path / ".git").exists():
                git_root = search_path
                break
            if search_path.parent == search_path:
                break  # Reached filesystem root
            search_path = search_path.parent

        # If we found a git root, use it; otherwise find common ancestor
        if git_root:
            root_path = git_root
        else:
            # Fall back to finding common ancestor directory
            common_root = file_paths[0].parent
            while not all(common_root in p.parents or common_root == p.parent for p in file_paths):
                if common_root.parent == common_root:
                    common_root = Path(".").resolve()
                    break
                common_root = common_root.parent
            root_path = common_root
    else:
        root_path = Path(args.root).resolve()

    if args.verbose:
        print(f"Root path: {root_path}")
        if focus_targets:
            print(f"Focus targets: {focus_targets}")

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        from grepmap_class import CACHE_VERSION
        cache_dir = root_path / f".repomap.tags.cache.v{CACHE_VERSION}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            tool_output(f"Cleared cache: {cache_dir}")
        else:
            tool_output(f"No cache to clear at: {cache_dir}")

    # Convert mentioned files to sets
    mentioned_fnames = set(args.mentioned_files) if args.mentioned_files else None
    mentioned_idents = set(args.mentioned_idents) if args.mentioned_idents else None

    # Auto-detect mode: use tree view for single files, directory view for multiple files/directories
    # User can override with --tree flag
    use_directory_mode = not args.tree  # Default from flag
    if not args.tree:  # Only auto-detect if user didn't explicitly request tree mode
        # Use tree mode (not directory mode) if analyzing a single file
        if len(other_files) == 1:
            use_directory_mode = False

    # Create GrepMap instance
    grep_map = GrepMap(
        map_tokens=args.map_tokens,
        root=str(root_path),
        token_counter_func=token_counter,
        file_reader_func=read_text,
        output_handler_funcs=output_handlers,
        verbose=args.verbose,
        max_context_window=args.max_context_window,
        exclude_unranked=args.exclude_unranked,
        color=not args.no_color,
        directory_mode=use_directory_mode,
        stats_mode=args.stats,
        adaptive_mode=args.adaptive,
        symbol_rank=not args.no_symbol_rank,
        git_weight=args.git_weight
    )
    
    # Generate the map
    try:
        map_content, file_report = grep_map.get_grep_map(
            focus_targets=focus_targets,
            other_files=other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            force_refresh=args.force_refresh
        )

        if map_content:
            if args.verbose:
                tokens = grep_map.token_count(map_content)
                tool_output(f"Generated map: {len(map_content)} chars, ~{tokens} tokens")
            
            print(map_content)
        else:
            # Provide helpful context about why no map was generated
            if file_report.excluded:
                tool_error("No grep map generated. Files were excluded:")
                for fname, reason in list(file_report.excluded.items())[:5]:
                    tool_error(f"  {fname}: {reason}")
                if len(file_report.excluded) > 5:
                    tool_error(f"  ... and {len(file_report.excluded) - 5} more")
            elif file_report.total_files_considered == 0:
                tool_error("No grep map generated: No files provided to analyze")
            elif file_report.definition_matches == 0:
                tool_error("No grep map generated: No code definitions found in files")
            else:
                tool_error("No grep map generated (check verbose output for details)")
            
    except KeyboardInterrupt:
        tool_error("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        tool_error(f"Error generating grep map: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
