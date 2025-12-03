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
from utils import Tag, DetailLevel, SignatureInfo, FieldInfo, RenderConfig
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
CACHE_VERSION = 3  # Bumped for Tag format change (added signature, fields for multi-detail rendering)

TAGS_CACHE_DIR = f".repomap.tags.cache.v{CACHE_VERSION}"
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
        color: bool = True,
        directory_mode: bool = True
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
        self.directory_mode = directory_mode

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

    def extract_signature_info(self, func_node, code_bytes: bytes) -> Optional[SignatureInfo]:
        """Extract function signature from tree-sitter function_definition node.

        Parses parameters with type annotations and return type for multi-detail rendering.
        Used during tag extraction to populate Tag.signature field.
        """
        if func_node is None:
            return None

        parameters = []
        return_type = None
        decorators = []

        for child in func_node.children:
            # Extract decorators (appear before the function definition)
            if child.type == 'decorator':
                # Get the decorator name (skip the @ symbol)
                for deco_child in child.children:
                    if deco_child.type == 'identifier':
                        decorators.append(deco_child.text.decode('utf-8') if deco_child.text else '')
                        break
                    elif deco_child.type == 'call':
                        # Decorator with arguments like @dataclass(frozen=True)
                        for call_child in deco_child.children:
                            if call_child.type == 'identifier':
                                decorators.append(call_child.text.decode('utf-8') if call_child.text else '')
                                break

            # Extract parameters
            elif child.type == 'parameters':
                for param in child.children:
                    if param.type == 'identifier':
                        # Simple parameter without type annotation
                        param_name = param.text.decode('utf-8') if param.text else ''
                        if param_name:
                            parameters.append((param_name, None))

                    elif param.type == 'typed_parameter':
                        # Parameter with type annotation: name: Type
                        param_name = None
                        param_type = None
                        for typed_child in param.children:
                            if typed_child.type == 'identifier':
                                param_name = typed_child.text.decode('utf-8') if typed_child.text else ''
                            elif typed_child.type == 'type':
                                param_type = typed_child.text.decode('utf-8') if typed_child.text else ''
                        if param_name:
                            parameters.append((param_name, param_type))

                    elif param.type == 'default_parameter':
                        # Parameter with default value: name=value
                        for default_child in param.children:
                            if default_child.type == 'identifier':
                                param_name = default_child.text.decode('utf-8') if default_child.text else ''
                                parameters.append((param_name, None))
                                break

                    elif param.type == 'typed_default_parameter':
                        # Parameter with type and default: name: Type = value
                        param_name = None
                        param_type = None
                        for typed_child in param.children:
                            if typed_child.type == 'identifier':
                                param_name = typed_child.text.decode('utf-8') if typed_child.text else ''
                            elif typed_child.type == 'type':
                                param_type = typed_child.text.decode('utf-8') if typed_child.text else ''
                        if param_name:
                            parameters.append((param_name, param_type))

                    elif param.type == 'list_splat_pattern':
                        # *args
                        for splat_child in param.children:
                            if splat_child.type == 'identifier':
                                param_name = '*' + (splat_child.text.decode('utf-8') if splat_child.text else '')
                                parameters.append((param_name, None))
                                break

                    elif param.type == 'dictionary_splat_pattern':
                        # **kwargs
                        for splat_child in param.children:
                            if splat_child.type == 'identifier':
                                param_name = '**' + (splat_child.text.decode('utf-8') if splat_child.text else '')
                                parameters.append((param_name, None))
                                break

            # Extract return type annotation
            elif child.type == 'type':
                return_type = child.text.decode('utf-8') if child.text else None

        # Also check for decorated_definition wrapper
        if func_node.parent and func_node.parent.type == 'decorated_definition':
            for child in func_node.parent.children:
                if child.type == 'decorator':
                    for deco_child in child.children:
                        if deco_child.type == 'identifier':
                            deco_name = deco_child.text.decode('utf-8') if deco_child.text else ''
                            if deco_name and deco_name not in decorators:
                                decorators.append(deco_name)
                            break
                        elif deco_child.type == 'call':
                            for call_child in deco_child.children:
                                if call_child.type == 'identifier':
                                    deco_name = call_child.text.decode('utf-8') if call_child.text else ''
                                    if deco_name and deco_name not in decorators:
                                        decorators.append(deco_name)
                                    break

        return SignatureInfo(
            parameters=tuple(parameters),
            return_type=return_type,
            decorators=tuple(decorators)
        )

    def extract_class_fields(self, class_node, code_bytes: bytes) -> Optional[Tuple[FieldInfo, ...]]:
        """Extract class fields from tree-sitter class_definition node.

        Captures annotated assignments in class body for dataclass-style display.
        Returns up to 10 fields to keep output concise.
        """
        if class_node is None:
            return None

        fields = []

        # Find the block (class body)
        body = None
        for child in class_node.children:
            if child.type == 'block':
                body = child
                break

        if not body:
            return None

        # Walk through statements in the class body
        for stmt in body.children:
            if len(fields) >= 10:  # Limit to 10 fields
                break

            # Look for annotated assignments: field: Type or field: Type = value
            if stmt.type == 'expression_statement':
                expr = stmt.children[0] if stmt.children else None
                if expr and expr.type == 'assignment':
                    # Check if left side has type annotation
                    # Pattern: (identifier or attribute) : type = value
                    # or just (identifier or attribute) : type
                    left = None
                    type_ann = None

                    for i, child in enumerate(expr.children):
                        if child.type == 'identifier':
                            left = child.text.decode('utf-8') if child.text else None
                        elif child.type == 'type':
                            type_ann = child.text.decode('utf-8') if child.text else None

                    if left and type_ann:
                        fields.append(FieldInfo(name=left, type_annotation=type_ann))

            # Also handle standalone type annotations: field: Type
            elif stmt.type == 'type_alias_statement' or (stmt.type == 'expression_statement' and
                    stmt.children and stmt.children[0].type == 'type'):
                # This is a type annotation without assignment
                pass

        # Alternative: look for annotated_assignment nodes directly
        if not fields:
            for stmt in body.children:
                if len(fields) >= 10:
                    break

                # Handle both expression_statement containing type or direct patterns
                if stmt.type == 'expression_statement':
                    for child in stmt.children:
                        if child.type == 'assignment':
                            # Check for pattern: name: type = value
                            name = None
                            type_ann = None
                            for sub in child.children:
                                if sub.type == 'identifier' and name is None:
                                    name = sub.text.decode('utf-8') if sub.text else None
                                elif sub.type == 'type':
                                    type_ann = sub.text.decode('utf-8') if sub.text else None
                            if name and type_ann:
                                fields.append(FieldInfo(name=name, type_annotation=type_ann))

        return tuple(fields) if fields else None

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
            code_bytes = bytes(code, "utf-8")
            tree = parser.parse(code_bytes)

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

                    # Extract signature for functions or fields for classes
                    # Only for definitions, not references
                    signature = None
                    fields = None
                    if kind == "def" and my_definition:
                        if node_type == "function":
                            signature = self.extract_signature_info(my_definition, code_bytes)
                        elif node_type == "class":
                            fields = self.extract_class_fields(my_definition, code_bytes)

                    tags.append(Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        line=line_num,
                        name=name,
                        kind=kind,
                        node_type=node_type,
                        parent_name=parent_name,
                        parent_line=parent_line,
                        signature=signature,
                        fields=fields
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

        if self.verbose:
            self.output_handlers['info'](f"Processing {len(all_fnames)} files for tag extraction...")

        for idx, fname in enumerate(all_fnames):
            rel_fname = self.get_rel_fname(fname)

            # Show progress every 100 files or at key milestones
            if self.verbose and (idx % 100 == 0 or idx == len(all_fnames) - 1):
                self.output_handlers['info'](f"  [{idx + 1}/{len(all_fnames)}] {rel_fname}")

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

        if self.verbose:
            self.output_handlers['info'](f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Build depth-based personalization for PageRank
        # Root files get higher bias, but truly important deep files can still rank high
        # if they're heavily interconnected with root files (graph structure wins)
        depth_personalization = {}
        for node in G.nodes():
            depth = node.count('/')

            # Check for vendor/third-party patterns
            vendor_patterns = ['node_modules', 'vendor', 'third_party', 'torchhub', '__pycache__', 'site-packages']
            is_vendor = any(pattern in node for pattern in vendor_patterns)

            if is_vendor:
                # Strong bias against vendor code
                depth_personalization[node] = 0.01
            elif depth <= 2:
                # Strong bias for root/shallow files
                depth_personalization[node] = 1.0
            elif depth <= 4:
                # Moderate bias
                depth_personalization[node] = 0.5
            else:
                # Weak bias for deep files (but graph can override)
                depth_personalization[node] = 0.1

        # Combine with chat file personalization if present
        if personalization:
            # Merge: multiply chat boost with depth bias
            for node in depth_personalization:
                if node in personalization:
                    depth_personalization[node] *= personalization[node]

        # Run PageRank with depth-aware personalization
        try:
            ranks = nx.pagerank(G, personalization=depth_personalization, alpha=0.85)

            if self.verbose and ranks:
                rank_values = list(ranks.values())
                self.output_handlers['info'](
                    f"PageRank scores (depth-aware) - min: {min(rank_values):.6f}, "
                    f"max: {max(rank_values):.6f}, "
                    f"avg: {sum(rank_values)/len(rank_values):.6f}"
                )

                # Show top files and their referrers for debugging
                sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
                self.output_handlers['info']("Top 5 files by PageRank:")
                for i, (node, rank) in enumerate(sorted_ranks[:5]):
                    # Count incoming edges (references TO this file)
                    in_edges = list(G.in_edges(node))
                    referrers = set(edge[0] for edge in in_edges)
                    self.output_handlers['info'](
                        f"  {i+1}. {node}: rank={rank:.6f}, "
                        f"referenced_by={len(referrers)} files, "
                        f"in_edges={len(in_edges)}"
                    )
        except Exception as e:
            # Fallback to uniform ranking if PageRank fails
            if self.verbose:
                self.output_handlers['warning'](f"PageRank failed: {e}, using uniform ranking")
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

    def get_symbol_icon(self, node_type: str) -> str:
        """Get icon/emoji for different symbol types."""
        icon_map = {
            'class': '󰠱 ',      # Nerd font class icon
            'function': ' ',    # Function icon
            'method': '󰊕 ',     # Method icon
            'variable': '󰀫 ',   # Variable icon
            'constant': '󰏿 ',   # Constant icon
            'interface': '󰜰 ',  # Interface icon
            'enum': ' ',       # Enum icon
            'module': ' ',     # Module/package icon
        }
        return icon_map.get(node_type, '• ')

    def get_symbol_color(self, node_type: str) -> str:
        """Get color for symbol types in directory view."""
        color_map = {
            'class': 'bold cyan',
            'function': 'bold yellow',
            'method': 'yellow',
            'variable': 'green',
            'constant': 'bold green',
            'interface': 'cyan',
            'enum': 'cyan',
            'module': 'blue',
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
                        line_output = f"{loi:4d}: {indent}{line_text}"
                        console.print(line_output, style="white")

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

    def render_symbol(
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
            # Just the name for functions, add fields preview for classes
            if tag.node_type == "class" and tag.fields:
                # Show class with field count preview
                return f"{name}({len(tag.fields)} fields)"
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

    def to_directory_overview(
        self,
        tags: List[Tuple[float, Tag]],
        chat_rel_fnames: Set[str],
        detail_level: DetailLevel = DetailLevel.LOW
    ) -> str:
        """Convert ranked tags to compact directory overview format.

        Shows files with their symbols in a condensed format using icons.
        Uses heuristics to choose between single-line, multi-line, or single-class format.

        Detail levels control how much signature info is shown:
        - LOW: Symbol names only (default, most compact)
        - MEDIUM: Names with simplified parameter names
        - HIGH: Full signatures with types

        Smart deduplication: At HIGH detail, repeated parameter patterns like
        `hobo: HoboWindow` are shown once, then just `hobo` for subsequent uses.
        """
        if not tags:
            return ""

        from io import StringIO
        string_io = StringIO()
        # Use very large width to prevent Rich's automatic wrapping - we handle wrapping manually
        console = Console(file=string_io, force_terminal=True, width=999999)

        # Per-file seen pattern tracking for smart deduplication
        # Key: rel_fname, Value: set of "param_name:type" patterns already shown
        seen_patterns: Dict[str, set] = defaultdict(set)

        # Group tags by file
        file_tags = defaultdict(list)
        for rank, tag in tags:
            if tag.kind == 'def':  # Only include definitions
                file_tags[tag.rel_fname].append((rank, tag))

        # Sort files by importance
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rank for rank, tag in x[1]),
            reverse=True
        )

        # Get terminal width for line-fitting heuristic
        import shutil
        term_width = shutil.get_terminal_size().columns

        for rel_fname, file_tag_list in sorted_files:
            # Calculate max rank for this file for verbose logging
            max_rank = max(rank for rank, tag in file_tag_list)

            if self.verbose:
                self.output_handlers['info'](f"  {rel_fname}: rank={max_rank:.4f}")

            # Get per-file seen patterns for dedup
            file_seen = seen_patterns[rel_fname]

            # Group tags by type (keep full tag for signature rendering)
            grouped: Dict[str, List[Tag]] = defaultdict(list)
            for rank, tag in file_tag_list:
                grouped[tag.node_type].append(tag)

            # Apply heuristics to choose display format
            total_symbols = sum(len(tags_list) for tags_list in grouped.values())
            num_classes = len(grouped.get('class', []))
            num_methods = len(grouped.get('method', []))

            # Calculate estimated line length for single-line format
            # At LOW detail, just use names; at higher detail, estimate longer strings
            all_tags_flat = [t for tags_list in grouped.values() for t in tags_list]
            if detail_level == DetailLevel.LOW:
                estimated_length = len(rel_fname) + 2 + sum(len(t.name) for t in all_tags_flat) + (len(all_tags_flat) - 1) * 2
            else:
                # Higher detail levels produce longer output - estimate conservatively
                estimated_length = term_width  # Force multi-line for detail

            # Heuristic 1: Single class with methods -> special format
            if num_classes == 1 and num_methods > 0 and total_symbols < 15:
                class_tag = grouped['class'][0]
                method_tags = grouped.get('method', [])

                text = Text()
                text.append(f"{rel_fname}: ", style="bold blue")
                text.append("class ", style="bold magenta")
                class_display = self.render_symbol(class_tag, detail_level, file_seen)
                text.append(class_display, style="bold cyan")
                text.append(": ", style="dim white")

                for i, method_tag in enumerate(method_tags):
                    if i > 0:
                        text.append(", ", style="dim white")
                    method_display = self.render_symbol(method_tag, detail_level, file_seen)
                    text.append(method_display, style="yellow")

                console.print(text, no_wrap=True)

            # Heuristic 2: Fits on one line -> single line format
            elif estimated_length < term_width * 0.8:  # Use 80% of terminal width as threshold
                all_symbols = []
                for node_type, tags_list in sorted(grouped.items()):
                    for tag in tags_list:
                        all_symbols.append((node_type, tag))

                # Build the output with manual wrapping and indentation
                prefix = f"{rel_fname}: "
                indent = " " * len(prefix)

                text = Text()
                text.append(prefix, style="bold blue")

                current_line_length = len(prefix)
                for i, (node_type, tag) in enumerate(all_symbols):
                    display_name = self.render_symbol(tag, detail_level, file_seen)
                    separator = ", " if i > 0 else ""
                    item_length = len(separator) + len(display_name)

                    # Check if we need to wrap
                    if i > 0 and current_line_length + item_length > term_width - 5:
                        text.append("\n" + indent)
                        current_line_length = len(indent)
                        separator = ""

                    if separator:
                        text.append(separator, style="dim white")
                    color = self.get_symbol_color(node_type)
                    text.append(display_name, style=color)
                    current_line_length += item_length

                console.print(text, no_wrap=True)

            # Heuristic 3: Many symbols -> multi-line grouped
            else:
                # File name header
                text = Text()
                text.append(f"{rel_fname}:", style="bold blue")
                console.print(text, no_wrap=True)

                # Group and display by type
                for node_type in ['class', 'function', 'method', 'variable', 'constant']:
                    if node_type not in grouped:
                        continue

                    tags_list = grouped[node_type]
                    icon = self.get_symbol_icon(node_type)
                    color = self.get_symbol_color(node_type)

                    # Calculate indentation for wrapped lines
                    prefix = f"  {icon}"
                    indent = " " * len(prefix)

                    line = Text()
                    line.append(prefix, style=color)

                    current_line_length = len(prefix)
                    for i, tag in enumerate(tags_list):
                        display_name = self.render_symbol(tag, detail_level, file_seen)
                        separator = ", " if i > 0 else ""
                        item_length = len(separator) + len(display_name)

                        # Check if we need to wrap
                        if i > 0 and current_line_length + item_length > term_width - 5:
                            line.append("\n" + indent)
                            current_line_length = len(indent)
                            separator = ""

                        if separator:
                            line.append(separator, style="dim white")
                        line.append(display_name, style=color)
                        current_line_length += item_length

                    console.print(line, no_wrap=True)

        # Add summary section: other files and classes in the project
        shown_files = set(rel_fname for rel_fname, _ in sorted_files)
        all_files = set(tag.rel_fname for _, tag in tags)
        other_files = sorted(all_files - shown_files)

        # Collect all classes
        # TODO: Future enhancement - extract inheritance hierarchies and type annotations
        # from tree-sitter to build class hierarchy graph and show parent/child relationships
        all_classes = set()
        class_to_file = {}
        for _, tag in tags:
            if tag.node_type == 'class' and tag.kind == 'def':
                all_classes.add(tag.name)
                if tag.name not in class_to_file:
                    class_to_file[tag.name] = tag.rel_fname

        if other_files or all_classes:
            console.print("")  # Blank line
            console.print(Text("═" * 80, style="dim"))

        # Show other files (not detailed above)
        if other_files:
            console.print(Text("\nOther files in project:", style="bold yellow"))
            # Show in columns
            files_per_line = 3
            for i in range(0, len(other_files), files_per_line):
                line_files = other_files[i:i + files_per_line]
                text = Text("  ")
                for j, fname in enumerate(line_files):
                    if j > 0:
                        text.append(" │ ", style="dim")
                    text.append(fname, style="cyan")
                console.print(text, no_wrap=True)

        # Show all classes in a tree structure organized by directory hierarchy
        # Clean indentation without tree line characters to maximize signal
        if all_classes:
            console.print(Text("\nClasses in project:", style="bold yellow"))

            # Build a tree structure: path components -> file -> classes
            file_to_classes = defaultdict(list)
            for cls in all_classes:
                file_to_classes[class_to_file[cls]].append(cls)

            # Build directory tree
            tree = {}
            for file_path in sorted(file_to_classes.keys()):
                parts = file_path.split('/')
                tree_key = '/'.join(parts[:-1]) if len(parts) > 1 else ''
                filename = parts[-1]
                if tree_key not in tree:
                    tree[tree_key] = {}
                tree[tree_key][filename] = sorted(file_to_classes[file_path])

            # Build directory hierarchy
            all_dirs = set()
            for file_path in file_to_classes.keys():
                parts = file_path.split('/')
                if len(parts) > 1:
                    for i in range(1, len(parts)):
                        all_dirs.add('/'.join(parts[:i]))

            sorted_dirs = sorted(all_dirs, key=lambda d: (d.count('/'), d))
            rendered = set()

            def render_directory_tree(dir_path, depth=0):
                """Render a directory and all its subdirectories/files with clean indentation."""
                if dir_path in rendered:
                    return
                rendered.add(dir_path)

                parts = dir_path.split('/')
                dir_name = parts[-1]
                indent = "  " * depth

                # Find immediate children (subdirs and files)
                children_dirs = []
                for d in sorted_dirs:
                    if d.startswith(dir_path + '/') and d.count('/') == dir_path.count('/') + 1:
                        children_dirs.append(d)

                # Show directory name
                dir_text = Text(indent)
                dir_text.append(dir_name + "/", style="bold blue")
                console.print(dir_text, no_wrap=True)

                # Render files in this directory
                if dir_path in tree:
                    files = tree[dir_path]
                    for filename, classes in sorted(files.items()):
                        file_text = Text("  " * (depth + 1))
                        file_text.append(filename, style="yellow")
                        file_text.append(": ", style="dim")

                        # Format classes
                        if len(classes) <= 3:
                            file_text.append(", ".join(classes), style="cyan")
                        else:
                            file_text.append(", ".join(classes[:3]), style="cyan")
                            file_text.append(f", +{len(classes)-3} more", style="dim")

                        console.print(file_text, no_wrap=True)

                # Recursively render subdirectories
                for subdir in children_dirs:
                    render_directory_tree(subdir, depth + 1)

            # Render root-level files first
            if '' in tree:
                for filename, classes in sorted(tree[''].items()):
                    file_text = Text("  ")
                    file_text.append(filename, style="yellow")
                    file_text.append(": ", style="dim")
                    if len(classes) <= 3:
                        file_text.append(", ".join(classes), style="cyan")
                    else:
                        file_text.append(", ".join(classes[:3]), style="cyan")
                        file_text.append(f", +{len(classes)-3} more", style="dim")
                    console.print(file_text, no_wrap=True)

            # Render top-level directories
            top_level_dirs = [d for d in sorted_dirs if '/' not in d]
            for top_dir in sorted(top_level_dirs):
                render_directory_tree(top_dir, depth=1)

        return string_io.getvalue().rstrip()

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
        """Generate the ranked tags map without caching.

        Uses multi-configuration optimization to find the best combination of
        tag count and detail level that fits within the token budget while
        maximizing information content.

        The optimizer tries configs in descending score order (coverage * 10 + detail)
        which prioritizes more tags over higher detail levels.
        """
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        if not ranked_tags:
            return None, file_report

        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)
        n = len(ranked_tags)

        # Generate configuration space: various tag counts × detail levels
        # More granular tag counts for better optimization
        tag_counts = sorted(set([
            n,
            int(n * 0.9),
            int(n * 0.75),
            int(n * 0.5),
            int(n * 0.25),
            min(n, 50),
            min(n, 25),
            min(n, 10)
        ]), reverse=True)
        tag_counts = [c for c in tag_counts if c > 0]

        # For non-directory mode, only use LOW detail (tree view handles its own formatting)
        if self.directory_mode:
            detail_levels = [DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.LOW]
        else:
            detail_levels = [DetailLevel.LOW]

        # Create all configs and sort by score (descending)
        configs = [RenderConfig(num_tags=c, detail_level=d)
                   for c in tag_counts for d in detail_levels]
        configs.sort(key=lambda x: x.score, reverse=True)

        if self.verbose:
            self.output_handlers['info'](
                f"Optimizing: {len(configs)} configs, {n} total tags, {max_map_tokens} token budget"
            )

        # Try each config in score order, return first that fits
        for config in configs:
            selected_tags = ranked_tags[:config.num_tags]

            if self.directory_mode:
                tree_output = self.to_directory_overview(
                    selected_tags, chat_rel_fnames, config.detail_level
                )
            else:
                tree_output = self.to_tree(selected_tags, chat_rel_fnames)

            if not tree_output:
                continue

            tokens = self.token_count(tree_output)
            if tokens <= max_map_tokens:
                if self.verbose:
                    detail_name = config.detail_level.name
                    self.output_handlers['info'](
                        f"Selected: {config.num_tags} tags, {detail_name} detail, {tokens} tokens"
                    )
                return tree_output, file_report

        # Fallback: minimal output if nothing fits
        if self.verbose:
            self.output_handlers['info']("Using minimal fallback output")

        minimal_tags = ranked_tags[:min(10, n)]
        if self.directory_mode:
            return self.to_directory_overview(
                minimal_tags, chat_rel_fnames, DetailLevel.LOW
            ), file_report
        else:
            return self.to_tree(minimal_tags, chat_rel_fnames), file_report
    
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
