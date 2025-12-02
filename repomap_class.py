"""
RepoMap class for generating repository maps.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Callable
import shutil
import sqlite3
from utils import Tag
from dataclasses import dataclass
import diskcache
import networkx as nx
from grep_ast import TreeContext
from utils import count_tokens, read_text
from scm import get_scm_fname
from rich.console import Console
from rich.text import Text


@dataclass
class FileReport:
    excluded: Dict[str, str]        # File -> exclusion reason with status
    definition_matches: int         # Total definition tags
    reference_matches: int          # Total reference tags
    total_files_considered: int     # Total files provided as input



# Constants
CACHE_VERSION = 1

TAGS_CACHE_DIR = os.path.join(os.getcwd(), f".repomap.tags.cache.v{CACHE_VERSION}")
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)

# Tag is imported from utils.py and includes parent scope information


class RepoMap:
    """Main class for generating repository maps."""
    
    def __init__(
        self,
        map_tokens: int = 1024,
        root: Optional[str] = None,
        token_counter_func: Callable[[str], int] = count_tokens,
        file_reader_func: Callable[[str], Optional[str]] = read_text,
        output_handler_funcs: Optional[Dict[str, Callable]] = None,
        repo_content_prefix: Optional[str] = None,
        verbose: bool = False,
        max_context_window: Optional[int] = None,
        map_mul_no_files: int = 8,
        refresh: str = "auto",
        exclude_unranked: bool = False,
        color: bool = True
    ):
        """Initialize RepoMap instance."""
        self.map_tokens = map_tokens
        self.max_map_tokens = map_tokens
        self.root = Path(root or os.getcwd()).resolve()
        self.token_count_func_internal = token_counter_func
        self.read_text_func_internal = file_reader_func
        self.repo_content_prefix = repo_content_prefix
        self.verbose = verbose
        self.max_context_window = max_context_window
        self.map_mul_no_files = map_mul_no_files
        self.refresh = refresh
        self.exclude_unranked = exclude_unranked
        self.color = color
        
        # Set up output handlers
        if output_handler_funcs is None:
            output_handler_funcs = {
                'info': print,
                'warning': print,
                'error': print
            }
        self.output_handlers = output_handler_funcs
        
        # Initialize caches
        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        
        # Load persistent tags cache
        self.load_tags_cache()
    
    def load_tags_cache(self):
        """Load the persistent tags cache."""
        cache_dir = self.root / TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = diskcache.Cache(str(cache_dir))
        except Exception as e:
            self.output_handlers['warning'](f"Failed to load tags cache: {e}")
            self.TAGS_CACHE = {}
    
    def save_tags_cache(self):
        """Save the tags cache (no-op as diskcache handles persistence)."""
        pass
    
    def tags_cache_error(self):
        """Handle tags cache errors."""
        try:
            cache_dir = self.root / TAGS_CACHE_DIR
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            self.load_tags_cache()
        except Exception:
            self.output_handlers['warning']("Failed to recreate tags cache, using in-memory cache")
            self.TAGS_CACHE = {}
    
    def token_count(self, text: str) -> int:
        """Count tokens in text with sampling optimization for long texts."""
        if not text:
            return 0
        
        len_text = len(text)
        if len_text < 200:
            return self.token_count_func_internal(text)
        
        # Sample for longer texts
        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        
        step = max(1, num_lines // 100)
        sampled_lines = lines[::step]
        sample_text = "".join(sampled_lines)
        
        if not sample_text:
            return self.token_count_func_internal(text)
        
        sample_tokens = self.token_count_func_internal(sample_text)
        
        if len(sample_text) == 0:
            return self.token_count_func_internal(text)
        
        est_tokens = (sample_tokens / len(sample_text)) * len_text
        return int(est_tokens)
    
    def get_rel_fname(self, fname: str) -> str:
        """Get relative filename from absolute path."""
        try:
            return str(Path(fname).relative_to(self.root))
        except ValueError:
            return fname
    
    def get_mtime(self, fname: str) -> Optional[float]:
        """Get file modification time."""
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.output_handlers['warning'](f"File not found: {fname}")
            return None

    def get_language(self, fname: str) -> str:
        """Get the language name for a file."""
        from grep_ast import filename_to_lang
        return filename_to_lang(fname) or "python"

    def get_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Get tags for a file, using cache when possible."""
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        
        try:
            # Handle both diskcache Cache and in-memory dict
            if isinstance(self.TAGS_CACHE, dict):
                cached_entry = self.TAGS_CACHE.get(fname)
            else:
                cached_entry = self.TAGS_CACHE.get(fname)
                
            if cached_entry and cached_entry.get("mtime") == file_mtime:
                return cached_entry["data"]
        except SQLITE_ERRORS:
            self.tags_cache_error()
        
        # Cache miss or file changed
        tags = self.get_tags_raw(fname, rel_fname)
        
        try:
            self.TAGS_CACHE[fname] = {"mtime": file_mtime, "data": tags}
        except SQLITE_ERRORS:
            self.tags_cache_error()
        
        return tags
    
    def get_tags_raw(self, fname: str, rel_fname: str) -> List[Tag]:
        """Parse file to extract tags using Tree-sitter."""
        try:
            from grep_ast import filename_to_lang
            from grep_ast.tsl import get_language, get_parser
            from tree_sitter import QueryCursor
        except ImportError:
            print("Error: grep-ast is required. Install with: pip install grep-ast")
            sys.exit(1)
            
        lang = filename_to_lang(fname)
        if not lang:
            return []
        
        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            self.output_handlers['error'](f"Skipping file {fname}: {err}")
            return []
        
        scm_fname = get_scm_fname(lang)
        if not scm_fname:
            return []
        
        code = self.read_text_func_internal(fname)
        if not code:
            return []
        
        try:
            tree = parser.parse(bytes(code, "utf-8"))
            
            # Load query from SCM file
            query_text = read_text(scm_fname, silent=True)
            if not query_text:
                return []
            
            query = language.query(query_text)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            
            tags = []
            # Process captures as a dictionary
            for capture_name, nodes in captures.items():
                for node in nodes:
                    if "name.definition" in capture_name:
                        kind = "def"
                    elif "name.reference" in capture_name:
                        kind = "ref"
                    else:
                        # Skip other capture types like 'reference.call' if not needed for tagging
                        continue

                    # Extract semantic node type from capture name
                    # e.g., "name.definition.function" -> "function"
                    # e.g., "name.definition.class" -> "class"
                    parts = capture_name.split('.')
                    node_type = parts[-1] if len(parts) > 2 else "unknown"

                    line_num = node.start_point[0] + 1
                    # Handle potential None value
                    name = node.text.decode('utf-8') if node.text else ""

                    # Find parent scope (class/function containing this definition)
                    # Walk up the tree to find enclosing class or function
                    parent_name = None
                    parent_line = None
                    current = node.parent

                    # First, we need to get to the actual definition node that contains this name
                    # The node we have is the identifier, we need to find its containing definition first
                    my_definition = current
                    while my_definition and my_definition.type not in ('class_definition', 'function_definition', 'method_definition'):
                        my_definition = my_definition.parent

                    # Now search for the parent definition (skip our own definition)
                    if my_definition:
                        search_node = my_definition.parent
                        while search_node:
                            if search_node.type in ('class_definition', 'function_definition', 'method_definition'):
                                # Try to get the 'name' field first (more reliable)
                                name_node = None
                                for child in search_node.children:
                                    if child.type == 'identifier':
                                        name_node = child
                                        break

                                if name_node:
                                    parent_name = name_node.text.decode('utf-8') if name_node.text else None
                                    parent_line = name_node.start_point[0] + 1
                                    break
                            search_node = search_node.parent

                    tags.append(Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        line=line_num,
                        name=name,
                        kind=kind,
                        node_type=node_type,
                        parent_name=parent_name,
                        parent_line=parent_line
                    ))
            
            return tags
            
        except Exception as e:
            self.output_handlers['error'](f"Error parsing {fname}: {e}")
            return []
    
    def get_ranked_tags(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[List[Tuple[float, Tag]], FileReport]:
        """Get ranked tags using PageRank algorithm with file report."""
        # Return empty list and empty report if no files
        if not chat_fnames and not other_fnames:
            return [], FileReport({}, 0, 0, 0)
            
        # Initialize file report early
        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0
        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()
        
        # Normalize paths to absolute
        def normalize_path(path):
            return str(Path(path).resolve())
        
        chat_fnames = [normalize_path(f) for f in chat_fnames]
        other_fnames = [normalize_path(f) for f in other_fnames]
        
        # Initialize file report
        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0
        
        # Collect all tags
        defines = defaultdict(set)
        references = defaultdict(set)
        definitions = defaultdict(set)
        
        personalization = {}
        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)
        
        all_fnames = list(set(chat_fnames + other_fnames))
        
        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            
            if not os.path.exists(fname):
                reason = "File not found"
                excluded[fname] = reason
                self.output_handlers['warning'](f"Repo-map can't include {fname}: {reason}")
                continue
                
            included.append(fname)
            
            tags = self.get_tags(fname, rel_fname)
            
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    definitions[rel_fname].add(tag.name)
                    total_definitions += 1
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)
                    total_references += 1
            
            # Set personalization for chat files
            if fname in chat_fnames:
                personalization[rel_fname] = 100.0
        
        # Build graph
        G = nx.MultiDiGraph()
        
        # Add nodes
        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            G.add_node(rel_fname)
        
        # Add edges based on references
        for name, ref_fnames in references.items():
            def_fnames = defines.get(name, set())
            for ref_fname in ref_fnames:
                for def_fname in def_fnames:
                    if ref_fname != def_fname:
                        G.add_edge(ref_fname, def_fname, name=name)
        
        if not G.nodes():
            # Create empty file report for this edge case
            return [], FileReport(excluded, total_definitions, total_references, len(chat_fnames) + len(other_fnames))
        
        # Run PageRank
        try:
            if personalization:
                ranks = nx.pagerank(G, personalization=personalization, alpha=0.85)
            else:
                ranks = {node: 1.0 for node in G.nodes()}
        except Exception:
            # Fallback to uniform ranking
            ranks = {node: 1.0 for node in G.nodes()}
        
        # Update excluded dictionary with status information
        for fname in set(chat_fnames + other_fnames):
            if fname in excluded:
                # Add status prefix to existing exclusion reason
                excluded[fname] = f"[EXCLUDED] {excluded[fname]}"
            elif fname not in included:
                excluded[fname] = "[NOT PROCESSED] File not included in final processing"
        
        # Create file report
        file_report = FileReport(
            excluded=excluded,
            definition_matches=total_definitions,
            reference_matches=total_references,
            total_files_considered=len(all_fnames)
        )
        
        # Collect and rank tags
        ranked_tags = []
        
        for fname in included:
            rel_fname = self.get_rel_fname(fname)
            file_rank = ranks.get(rel_fname, 0.0)

            # Exclude files with low Page Rank if exclude_unranked is True
            if self.exclude_unranked and file_rank <= 0.0001:  # Use a small threshold to exclude near-zero ranks
                continue
            
            tags = self.get_tags(fname, rel_fname)
            for tag in tags:
                if tag.kind == "def":
                    # Boost for mentioned identifiers
                    boost = 1.0
                    if tag.name in mentioned_idents:
                        boost *= 10.0
                    if rel_fname in mentioned_fnames:
                        boost *= 5.0
                    if rel_fname in chat_rel_fnames:
                        boost *= 20.0
                    
                    final_rank = file_rank * boost
                    ranked_tags.append((final_rank, tag))
        
        # Sort by rank (descending)
        ranked_tags.sort(key=lambda x: x[0], reverse=True)

        return ranked_tags, file_report

    def get_token_color(self, node_type: str) -> str:
        """Map tree-sitter node type to Rich color/style for token-level coloring.

        Returns a Rich color/style string for granular syntax highlighting.
        """
        color_map = {
            # Keywords
            'def': 'bold magenta',
            'class': 'bold magenta',
            'async': 'bold magenta',
            'await': 'bold magenta',
            'return': 'magenta',
            'if': 'magenta',
            'else': 'magenta',
            'elif': 'magenta',
            'for': 'magenta',
            'while': 'magenta',
            'import': 'magenta',
            'from': 'magenta',
            'as': 'magenta',
            'with': 'magenta',
            'try': 'magenta',
            'except': 'magenta',
            'finally': 'magenta',
            'raise': 'magenta',
            'pass': 'magenta',
            'break': 'magenta',
            'continue': 'magenta',
            'lambda': 'magenta',
            'yield': 'magenta',
            'assert': 'magenta',
            'del': 'magenta',
            'global': 'magenta',
            'nonlocal': 'magenta',
            'in': 'magenta',
            'is': 'magenta',
            'not': 'magenta',
            'and': 'magenta',
            'or': 'magenta',

            # Types and classes
            'type': 'cyan',
            'type_identifier': 'cyan',
            'class_definition': 'bold cyan',

            # Functions
            'function_definition': 'bold yellow',
            'call': 'yellow',
            'identifier': 'white',

            # Strings and literals
            'string': 'green',
            'string_content': 'green',
            'integer': 'blue',
            'float': 'blue',
            'true': 'blue',
            'false': 'blue',
            'none': 'blue',

            # Comments
            'comment': 'dim white',

            # Operators and punctuation
            'operator': 'red',
            ':': 'red',
            '=': 'red',
            '->': 'red',
            '(': 'dim white',
            ')': 'dim white',
            '[': 'dim white',
            ']': 'dim white',
            '{': 'dim white',
            '}': 'dim white',
            ',': 'dim white',
            '.': 'red',

            # Decorators
            'decorator': 'bold blue',
            '@': 'bold blue',

            # Default
            'unknown': 'white'
        }
        return color_map.get(node_type, 'white')

    def colorize_line_with_tree_sitter(self, line_num: int, line_text: str, tree, code_full: str, indent: str = "") -> Text:
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
            color = self.get_token_color(node_type)
            text.append(token_text, style=color)

            current_char_pos = char_pos_in_stripped + len(token_text)

        # Add any remaining text
        if current_char_pos < len(line_text):
            text.append(line_text[current_char_pos:], style="white")

        return text

    def render_tree(self, abs_fname: str, rel_fname: str, lois: List[int], tags: Optional[List[Tag]] = None) -> str:
        """Render a code snippet with specific lines of interest.

        Args:
            abs_fname: Absolute file path
            rel_fname: Relative file path
            lois: Lines of interest (line numbers)
            tags: Optional list of Tag objects for semantic coloring

        Returns:
            Formatted string with code lines
        """
        code = self.read_text_func_internal(abs_fname)
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
            from io import StringIO

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

                    if tree:
                        # Use tree-sitter for granular token coloring
                        rich_line = self.colorize_line_with_tree_sitter(loi, line_text, tree, code, indent)
                        console.print(rich_line)
                    else:
                        # Fallback to simple coloring
                        console.print(f"{loi:4d}: {indent}{line_text}", style="white")

            # Get the rendered output
            return string_io.getvalue().rstrip()

        # Use TreeContext for non-colored rendering
        try:
            if rel_fname not in self.tree_context_cache:
                self.tree_context_cache[rel_fname] = TreeContext(
                    rel_fname,
                    code,
                    color=False
                )

            tree_context = self.tree_context_cache[rel_fname]
            return tree_context.format(lois)

        except Exception:
            # Fallback to simple line extraction
            lines = code.splitlines()
            result_lines = [f"{rel_fname}:"]

            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    result_lines.append(f"{loi:4d}: {lines[loi-1]}")

            return "\n".join(result_lines)
    
    def to_tree(self, tags: List[Tuple[float, Tag]], chat_rel_fnames: Set[str]) -> str:
        """Convert ranked tags to formatted tree output."""
        if not tags:
            return ""
        
        # Group tags by file
        file_tags = defaultdict(list)
        for rank, tag in tags:
            file_tags[tag.rel_fname].append((rank, tag))
        
        # Sort files by importance (max rank of their tags)
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rank for rank, tag in x[1]),
            reverse=True
        )
        
        tree_parts = []
        
        for rel_fname, file_tag_list in sorted_files:
            # Get lines of interest and tags
            lois = [tag.line for rank, tag in file_tag_list]
            file_tags_only = [tag for rank, tag in file_tag_list]

            # Find absolute filename
            abs_fname = str(self.root / rel_fname)

            # Get the max rank for the file
            max_rank = max(rank for rank, tag in file_tag_list)

            # Render the tree for this file (pass tags for semantic coloring)
            rendered = self.render_tree(abs_fname, rel_fname, lois, file_tags_only)
            if rendered:
                # Add rank value to the output
                rendered_lines = rendered.splitlines()
                first_line = rendered_lines[0]
                code_lines = rendered_lines[1:]
                
                tree_parts.append(
                    f"{first_line}\n"
                    f"(Rank value: {max_rank:.4f})\n\n" # Added an extra newline here
                    + "\n".join(code_lines)
                )
        
        return "\n\n".join(tree_parts)
    
    def get_ranked_tags_map(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Get the ranked tags map with caching."""
        cache_key = (
            tuple(sorted(chat_fnames)),
            tuple(sorted(other_fnames)),
            max_map_tokens,
            tuple(sorted(mentioned_fnames or [])),
            tuple(sorted(mentioned_idents or [])),
        )
        
        if not force_refresh and cache_key in self.map_cache:
            return self.map_cache[cache_key]
        
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens,
            mentioned_fnames, mentioned_idents
        )
        
        self.map_cache[cache_key] = result
        return result
    
    def get_ranked_tags_map_uncached(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the ranked tags map without caching."""
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )
        
        if not ranked_tags:
            return None, file_report

        # Binary search to find the right number of tags
        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)
        
        def try_tags(num_tags: int) -> Tuple[Optional[str], int]:
            if num_tags <= 0:
                return None, 0
            
            selected_tags = ranked_tags[:num_tags]
            tree_output = self.to_tree(selected_tags, chat_rel_fnames)
            
            if not tree_output:
                return None, 0
            
            tokens = self.token_count(tree_output)
            return tree_output, tokens
        
        # Binary search for optimal number of tags
        left, right = 0, len(ranked_tags)
        best_tree = None
        
        while left <= right:
            mid = (left + right) // 2
            tree_output, tokens = try_tags(mid)
            
            if tree_output and tokens <= max_map_tokens:
                best_tree = tree_output
                left = mid + 1
            else:
                right = mid - 1
        
        return best_tree, file_report
    
    def get_repo_map(
        self,
        chat_files: Optional[List[str]] = None,
        other_files: Optional[List[str]] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the repository map with file report."""
        if chat_files is None:
            chat_files = []
        if other_files is None:
            other_files = []
            
        # Create empty report for error cases
        empty_report = FileReport({}, 0, 0, 0)
        
        if self.max_map_tokens <= 0 or not other_files:
            return None, empty_report
        
        # Adjust max_map_tokens if no chat files
        max_map_tokens = self.max_map_tokens
        if not chat_files and self.max_context_window:
            padding = 1024
            available = self.max_context_window - padding
            max_map_tokens = min(
                max_map_tokens * self.map_mul_no_files,
                available
            )
        
        try:
            # get_ranked_tags_map returns (map_string, file_report)
            map_string, file_report = self.get_ranked_tags_map(
                chat_files, other_files, max_map_tokens,
                mentioned_fnames, mentioned_idents, force_refresh
            )
        except RecursionError:
            self.output_handlers['error']("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return None, FileReport({}, 0, 0, 0)  # Ensure consistent return type
        
        if map_string is None:
            print("map_string is None")
            return None, file_report
        
        if self.verbose:
            tokens = self.token_count(map_string)
            self.output_handlers['info'](f"Repo-map: {tokens / 1024:.1f} k-tokens")
        
        # Format final output
        other = "other " if chat_files else ""
        
        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""
        
        repo_content += map_string
        
        return repo_content, file_report
