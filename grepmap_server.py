import asyncio
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, settings
from grepmap_class import GrepMap
from grepmap.discovery import find_source_files
from utils import count_tokens, read_text


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
    """Generate a topology-aware structural map using PageRank over the dependency graph.

    **What this provides:**
    NOT alphabetical file lists. YES graph-theoretic importance analysis.
    - Parses all code with tree-sitter (functions, classes, imports, references)
    - Builds dependency graph: files as nodes, symbol references as edges
    - Runs PageRank with depth-aware personalization (root=1.0x, vendor=0.01x)
    - Binary-searches token budget to maximize information density

    **Topology preservation:**
    Output maintains directory hierarchy and class structure:
    - Directory nesting shows architectural layers
    - Classes display fields/properties/methods grouped hierarchically
    - Multi-line signatures collapsed to one line with full type info
    - Colon stripping and type deduplication for token efficiency

    **Causality model:**
    High-ranked files are *dependencies* of many others (causal anchors).
    If session.py has high PageRank, it's because many files import from it.
    This is transitive importance, not file size or alphabetical proximity.

    **Workflow pattern:**
    1. Call grep_map FIRST to learn graph topology (which files are central)
    2. Hypothesize: "session management probably in high-ranked session.py"
    3. Call again with chat_files to boost specific files for deeper detail
    4. Verify with search_identifiers or grep for exact content
    5. Read files directly for final confirmation

    **When to use this vs grep:**
    - grep_map: "WHERE does session management happen?" → surfaces session.py as VIP
    - grep: "HOW is session.start() implemented?" → exact line-by-line content

    The map shows WHAT exists and WHERE it matters (ranked). Grep shows HOW it works.

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
        effective_other_files = find_source_files(project_root)

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
    """AST-aware symbol search across the dependency graph (microscope after GPS).

    **What this provides:**
    NOT text search. YES tree-sitter structural analysis.
    - Extracts ALL symbols from parsed ASTs (functions, classes, variables, methods)
    - Distinguishes definitions (where declared) from references (where used)
    - Returns code context with syntax highlighting for each match
    - Case-insensitive partial matching with relevance sorting

    **Structural awareness:**
    Unlike grep, this understands code semantics:
    - Finds symbol definitions (def MyClass:, def my_function():)
    - Finds symbol references (calls, imports, attribute access)
    - Knows the difference between "Session" the class vs "session" the variable
    - Provides context lines showing actual usage in situ

    **When to use this:**
    - After grep_map identifies important files, drill down to specific symbols
    - "Find all definitions and usages of 'Session'"
    - "Where is get_frame() defined and called?"
    - AST-level accuracy (not fooled by comments or strings)

    **When to use grep instead:**
    - Arbitrary string/regex patterns ("TODO|FIXME")
    - Text in comments, docstrings, or string literals
    - Non-identifier searches (URLs, config values, etc.)

    **Result ordering:**
    Definitions first (most relevant), then references, sorted by match quality.
    Each result includes file path, line number, symbol kind, and surrounding context.

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
        all_files = find_source_files(project_root)

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
