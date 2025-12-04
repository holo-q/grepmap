import asyncio
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, settings
from grepmap_class import GrepMap
from utils import count_tokens, read_text

# Helper function from your CLI, useful to have here
def find_src_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return [directory] if os.path.isfile(directory) else []
    src_files = []
    for r, d, f_list in os.walk(directory):
        d[:] = [d_name for d_name in d if not d_name.startswith('.') and d_name not in {'node_modules', '__pycache__', 'venv', 'env'}]
        for f in f_list:
            if not f.startswith('.'):
                src_files.append(os.path.join(r, f))
    return src_files

# Configure logging - only show errors
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

# Create console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Suppress FastMCP logs
fastmcp_logger = logging.getLogger('fastmcp')
fastmcp_logger.setLevel(logging.ERROR)
# Suppress server startup message
server_logger = logging.getLogger('fastmcp.server')
server_logger.setLevel(logging.ERROR)

log = logging.getLogger(__name__)

# Set global stateless_http setting
settings.stateless_http = True

# Create MCP server
mcp = FastMCP("GrepMapServer")

@mcp.tool()
async def grep_map(
    project_root: str,
    chat_files: Optional[List[str]] = None,
    other_files: Optional[List[str]] = None,
    token_limit: Any = 8192,  # Accept any type to handle empty strings
    exclude_unranked: bool = False,
    force_refresh: bool = False,
    mentioned_files: Optional[List[str]] = None,
    mentioned_idents: Optional[List[str]] = None,
    verbose: bool = False,
    max_context_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a ranked map of a codebase showing important files and symbols.

    This is your GPS for navigating code. It uses PageRank to identify the "main characters" -
    files that other code depends on most. Use this BEFORE diving into grep searches to orient
    yourself and understand where the action is.

    **Workflow pattern:**
    1. Call grep_map first to see the landscape (which files matter)
    2. Identify relevant files from the rankings
    3. Call again with those files in chat_files for more detail
    4. Use search_identifiers or grep for specific symbols
    5. Read files directly for final verification

    **When to use this vs grep:**
    - grep_map: "Where does session management happen?" → shows you session.py is a VIP
    - grep: "What's on line 42 of session.py?" → exact content lookup

    The map shows classes, functions, and methods organized by directory topology,
    with high-PageRank files promoted and vendor/deep-nested files demoted.

    :param project_root: Absolute path to the project root directory.
    :param chat_files: Files you're actively working on (relative paths). Get highest ranking boost.
    :param other_files: Additional files to consider. If omitted, scans entire project.
    :param token_limit: Token budget for the map output (default: 8192). Increase for more detail.
    :param exclude_unranked: Hide files with PageRank of 0 (peripheral files).
    :param force_refresh: Bypass cache and reparse everything.
    :param mentioned_files: Files mentioned in conversation (mid-level ranking boost).
    :param mentioned_idents: Identifiers to boost (function/class names you're looking for).
    :param verbose: Show ranking details and debug info.
    :param max_context_window: Max context window for token calculation.
    :returns: {map: string, report: {excluded, definition_matches, reference_matches, total_files_considered}}
    """
    if not os.path.isdir(project_root):
        return {"error": f"Project root directory not found: {project_root}"}

    # 1. Handle and validate parameters
    # Convert token_limit to integer with fallback
    try:
        token_limit = int(token_limit) if token_limit else 8192
    except (TypeError, ValueError):
        token_limit = 8192
    
    # Ensure token_limit is positive
    if token_limit <= 0:
        token_limit = 8192
    
    chat_files_list = chat_files or []
    mentioned_fnames_set = set(mentioned_files) if mentioned_files else None
    mentioned_idents_set = set(mentioned_idents) if mentioned_idents else None

    # 2. If a specific list of other_files isn't provided, scan the whole root directory.
    # This should happen regardless of whether chat_files are present.
    effective_other_files = []
    if other_files:
        effective_other_files = other_files
    else:
        log.info("No other_files provided, scanning root directory for context...")
        effective_other_files = find_src_files(project_root)

    # Add a print statement for debugging so you can see what the tool is working with.
    log.debug(f"Chat files: {chat_files_list}")
    log.debug(f"Effective other_files count: {len(effective_other_files)}")

    # If after all that we have no files, we can exit early.
    if not chat_files_list and not effective_other_files:
        log.info("No files to process.")
        return {"map": "No files found to generate a map."}

    # 3. Resolve paths relative to project root
    root_path = Path(project_root).resolve()
    abs_chat_files = [str(root_path / f) for f in chat_files_list]
    abs_other_files = [str(root_path / f) for f in effective_other_files]
    
    # Remove any chat files from the other_files list to avoid duplication
    abs_chat_files_set = set(abs_chat_files)
    abs_other_files = [f for f in abs_other_files if f not in abs_chat_files_set]

    # 4. Instantiate and run RepoMap
    try:
        grep_mapper = GrepMap(
            map_tokens=token_limit,
            root=str(root_path),
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=verbose,
            exclude_unranked=exclude_unranked,
            max_context_window=max_context_window
        )
    except Exception as e:
        log.exception(f"Failed to initialize RepoMap for project '{project_root}': {e}")
        return {"error": f"Failed to initialize RepoMap: {str(e)}"}

    try:
        map_content, file_report = await asyncio.to_thread(
            grep_mapper.get_grep_map,
            chat_files=abs_chat_files,
            other_files=abs_other_files,
            mentioned_fnames=mentioned_fnames_set,
            mentioned_idents=mentioned_idents_set,
            force_refresh=force_refresh
        )
        
        # Convert FileReport to dictionary for JSON serialization
        report_dict = {
            "excluded": file_report.excluded,
            "definition_matches": file_report.definition_matches,
            "reference_matches": file_report.reference_matches,
            "total_files_considered": file_report.total_files_considered
        }
        
        return {
            "map": map_content or "No grep map could be generated.",
            "report": report_dict
        }
    except Exception as e:
        log.exception(f"Error generating grep map for project '{project_root}': {e}")
        return {"error": f"Error generating grep map: {str(e)}"}
    
@mcp.tool()
async def search_identifiers(
    project_root: str,
    query: str,
    max_results: int = 50,
    context_lines: int = 2,
    include_definitions: bool = True,
    include_references: bool = True
) -> Dict[str, Any]:
    """Search for symbols (functions, classes, variables) across the codebase.

    This is your microscope for finding specific symbols. Use it AFTER grep_map has oriented you,
    or when you know exactly what identifier you're looking for.

    **When to use this:**
    - You know the function/class name but not which file it's in
    - You want to find all definitions AND usages of a symbol
    - You need AST-aware search (understands code structure, not just text)

    **When to use grep instead:**
    - Searching for arbitrary strings or patterns
    - Looking for comments, strings, or non-identifier text
    - Need regex matching

    The search is case-insensitive and matches partial names. Results are sorted with
    definitions first, then references.

    :param project_root: Absolute path to the project root directory.
    :param query: Identifier name to search for (e.g., "Session", "get_frame", "fps").
    :param max_results: Maximum results to return (default: 50).
    :param context_lines: Lines of context around each match (default: 2).
    :param include_definitions: Include where symbols are defined (default: true).
    :param include_references: Include where symbols are used (default: true).
    :returns: {results: [{file, line, name, kind (def/ref), context}]}
    """
    if not os.path.isdir(project_root):
        return {"error": f"Project root directory not found: {project_root}"}

    try:
        # Initialize GrepMap with search-specific settings
        grep_map = GrepMap(
            root=project_root,
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=False,
            exclude_unranked=True
        )

        # Find all source files in the project
        all_files = find_src_files(project_root)

        # Get all tags (definitions and references) for all files
        all_tags = []
        for file_path in all_files:
            rel_path = str(Path(file_path).relative_to(project_root))
            tags = grep_map.get_tags(file_path, rel_path)
            all_tags.extend(tags)

        # Filter tags based on search query and options
        matching_tags = []
        query_lower = query.lower()
        
        for tag in all_tags:
            if query_lower in tag.name.lower():
                if (tag.kind == "def" and include_definitions) or \
                   (tag.kind == "ref" and include_references):
                    matching_tags.append(tag)

        # Sort by relevance (definitions first, then references)
        matching_tags.sort(key=lambda x: (x.kind != "def", x.name.lower().find(query_lower)))

        # Limit results
        matching_tags = matching_tags[:max_results]

        # Format results with context
        results = []
        for tag in matching_tags:
            file_path = str(Path(project_root) / tag.rel_fname)
            
            # Calculate context range based on context_lines parameter
            start_line = max(1, tag.line - context_lines)
            end_line = tag.line + context_lines
            context_range = list(range(start_line, end_line + 1))

            context = grep_map.render_tree(
                file_path,
                tag.rel_fname,
                context_range
            )
            
            if context:
                results.append({
                    "file": tag.rel_fname,
                    "line": tag.line,
                    "name": tag.name,
                    "kind": tag.kind,
                    "context": context
                })

        return {"results": results}

    except Exception as e:
        log.exception(f"Error searching identifiers in project '{project_root}': {e}")
        return {"error": f"Error searching identifiers: {str(e)}"}    

# --- Main Entry Point ---
def main():
    # Run the MCP server
    log.debug("Starting FastMCP server...")
    mcp.run()

if __name__ == "__main__":
    main()
