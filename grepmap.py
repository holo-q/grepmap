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
        default=8192,
        help="Maximum tokens for the generated map (default: 8192)"
    )
    
    parser.add_argument(
        "--chat-files",
        nargs="*",
        help="Files currently being edited (given higher priority)"
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
    
    # Process file arguments
    chat_files_from_args = args.chat_files or [] # These are the paths as strings from the CLI
    
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
    
    # Convert to absolute paths
    chat_files = [str(Path(f).resolve()) for f in chat_files_from_args]
    other_files = [str(Path(f).resolve()) for f in effective_other_files_unresolved]

    if args.verbose:
        tool_output(f"Found {len(chat_files)} chat files, {len(other_files)} other files")

    # Auto-detect root if not explicitly provided and files are from outside current directory
    # This ensures that when you run "grepmap /some/other/path/file.py" it automatically
    # uses the correct repository root instead of the current directory
    if args.root == "." and (chat_files or other_files):
        all_files = chat_files + other_files
        # Try to find git repository root
        if all_files:
            file_paths = [Path(f) for f in all_files]
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
    else:
        root_path = Path(args.root).resolve()

    if args.verbose:
        print(f"Root path: {root_path}")
        if chat_files:
            print(f"Chat files: {chat_files}")

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
        total_files = len(chat_files) + len(other_files)
        if total_files == 1:
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
        stats_mode=args.stats
    )
    
    # Generate the map
    try:
        map_content, file_report = grep_map.get_grep_map(
            chat_files=chat_files,
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
