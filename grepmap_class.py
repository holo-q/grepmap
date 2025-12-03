"""
GrepMap class for generating grep maps.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Callable
import shutil
import sqlite3
from utils import Tag, DetailLevel, SignatureInfo, FieldInfo
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
CACHE_VERSION = 4  # Bumped for decorator extraction fix (properties now properly detected)

TAGS_CACHE_DIR = f".grepmap.tags.cache.v{CACHE_VERSION}"
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)

# Tag is imported from utils.py and includes parent scope information


class GrepMap:
    """Main class for generating grep maps."""
    
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
        """Initialize GrepMap instance."""
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

        # Decorators are in a parent decorated_definition node, not children of function_definition
        # Structure: decorated_definition -> [decorator, decorator, ..., function_definition]
        parent = func_node.parent
        if parent and parent.type == 'decorated_definition':
            for sibling in parent.children:
                if sibling.type == 'decorator':
                    # Get the decorator name (skip the @ symbol)
                    for deco_child in sibling.children:
                        if deco_child.type == 'identifier':
                            decorators.append(deco_child.text.decode('utf-8') if deco_child.text else '')
                            break
                        elif deco_child.type == 'call':
                            # Decorator with arguments like @dataclass(frozen=True)
                            for call_child in deco_child.children:
                                if call_child.type == 'identifier':
                                    decorators.append(call_child.text.decode('utf-8') if call_child.text else '')
                                    break
                        elif deco_child.type == 'attribute':
                            # Decorator like @functools.wraps
                            attr_text = deco_child.text.decode('utf-8') if deco_child.text else ''
                            decorators.append(attr_text)
                            break

        for child in func_node.children:
            # Extract parameters
            if child.type == 'parameters':
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
        items = [self.render_symbol(tag, detail_level, seen_patterns) for tag in tags_list]
        self._render_labeled_list(console, items, indent, label, term_width, label_color, item_color)

    def to_directory_overview(
        self,
        tags: List[Tuple[float, Tag]],
        chat_rel_fnames: Set[str],
        detail_level: DetailLevel = DetailLevel.LOW,
        overflow_tags: Optional[List[Tuple[float, Tag]]] = None
    ) -> str:
        """Convert ranked tags to hierarchical directory overview format.

        Shows files with their symbols in a topology-preserving format:
        - Classes shown with their methods indented underneath
        - Top-level functions shown separately
        - Constants/variables shown at the end

        Args:
            tags: Primary ranked tags for detailed display
            chat_rel_fnames: Files currently in chat context
            detail_level: How much signature detail to show
            overflow_tags: Additional tags beyond the detailed view, shown at lower resolution

        This maximizes topological signal for orientation in the codebase.
        """
        if not tags:
            return ""

        from io import StringIO
        string_io = StringIO()
        console = Console(file=string_io, force_terminal=True, width=999999)

        # Per-file seen pattern tracking for smart deduplication
        seen_patterns: Dict[str, set] = defaultdict(set)

        # Group tags by file
        file_tags: Dict[str, List[Tuple[float, Tag]]] = defaultdict(list)
        for rank, tag in tags:
            if tag.kind == 'def':
                file_tags[tag.rel_fname].append((rank, tag))

        # Sort files by importance
        sorted_files = sorted(
            file_tags.items(),
            key=lambda x: max(rank for rank, tag in x[1]),
            reverse=True
        )

        import shutil
        term_width = shutil.get_terminal_size().columns

        for rel_fname, file_tag_list in sorted_files:
            if self.verbose:
                max_rank = max(rank for rank, tag in file_tag_list)
                self.output_handlers['info'](f"  {rel_fname}: rank={max_rank:.4f}")

            file_seen = seen_patterns[rel_fname]

            # Separate tags into categories
            classes: List[Tag] = []
            methods_by_class: Dict[str, List[Tag]] = defaultdict(list)
            top_level_funcs: List[Tag] = []
            constants: List[Tag] = []

            for rank, tag in file_tag_list:
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
                class_display = self.render_symbol(class_tag, detail_level, file_seen)
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
                    field_names = [f.render(detail_level) for f in class_fields]
                    self._render_labeled_list(
                        console, field_names, indent="    ", label="fields",
                        term_width=term_width, label_color="dim magenta", item_color="bright_cyan"
                    )

                # Render properties indented under the class
                if class_properties:
                    self._render_symbol_list(
                        console, class_properties, detail_level, file_seen,
                        indent="    ", label="props", term_width=term_width,
                        label_color="dim magenta", item_color="bright_cyan"
                    )

                # Render methods indented under the class
                if class_methods:
                    self._render_symbol_list(
                        console, class_methods, detail_level, file_seen,
                        indent="    ", label="def", term_width=term_width,
                        label_color="dim magenta", item_color="yellow"
                    )

            # Render top-level functions
            if top_level_funcs:
                self._render_symbol_list(
                    console, top_level_funcs, detail_level, file_seen,
                    indent="  ", label="def", term_width=term_width,
                    label_color="magenta", item_color="green"
                )

            # Render constants
            if constants:
                self._render_symbol_list(
                    console, constants, detail_level, file_seen,
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
            for _, tag in overflow_tags:
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
                console.print(Text("── Also in scope ──", style="dim yellow"))

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

        Uses binary search per detail level to find optimal tag count, then
        picks the configuration with highest score (coverage * 10 + detail).
        This ensures we maximize file coverage before adding signature detail.
        """
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        if not ranked_tags:
            return None, file_report

        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)
        n = len(ranked_tags)

        # For non-directory mode, only use LOW detail (tree view handles its own formatting)
        if self.directory_mode:
            detail_levels = [DetailLevel.LOW, DetailLevel.MEDIUM, DetailLevel.HIGH]
        else:
            detail_levels = [DetailLevel.LOW]

        def try_render(num_tags: int, detail: DetailLevel) -> Tuple[Optional[str], int]:
            """Try rendering with given config, return (output, tokens)."""
            if num_tags <= 0:
                return None, 0
            selected = ranked_tags[:num_tags]
            if self.directory_mode:
                output = self.to_directory_overview(selected, chat_rel_fnames, detail)
            else:
                output = self.to_tree(selected, chat_rel_fnames)
            if not output:
                return None, 0
            return output, self.token_count(output)

        # Binary search for each detail level to find max tags that fit
        best_configs: List[Tuple[int, DetailLevel, str, int]] = []  # (num_tags, detail, output, tokens)

        for detail in detail_levels:
            left, right = 1, n
            best_for_detail = None

            while left <= right:
                mid = (left + right) // 2
                output, tokens = try_render(mid, detail)

                if output and tokens <= max_map_tokens:
                    best_for_detail = (mid, detail, output, tokens)
                    left = mid + 1  # Try more tags
                else:
                    right = mid - 1  # Try fewer tags

            if best_for_detail:
                best_configs.append(best_for_detail)

        if not best_configs:
            # Fallback: minimal output
            if self.verbose:
                self.output_handlers['info']("Using minimal fallback output")
            minimal_tags = ranked_tags[:min(10, n)]
            if self.directory_mode:
                return self.to_directory_overview(
                    minimal_tags, chat_rel_fnames, DetailLevel.LOW
                ), file_report
            else:
                return self.to_tree(minimal_tags, chat_rel_fnames), file_report

        # Pick config with highest score: num_tags * 10 + detail_level
        # This strongly favors coverage over detail
        best = max(best_configs, key=lambda x: x[0] * 10 + x[1].value)
        num_tags, detail, output, tokens = best

        if self.verbose:
            self.output_handlers['info'](
                f"Selected: {num_tags} tags, {detail.name} detail, {tokens} tokens "
                f"(from {len(best_configs)} candidates)"
            )

        # Re-render with overflow tags for the low-res "also in scope" section
        # Take a large slice of remaining tags to find ~30 additional files
        if self.directory_mode and num_tags < n:
            # Use remaining tags up to 2000 more, or all if less
            overflow_count = min(2000, n - num_tags)
            overflow = ranked_tags[num_tags:num_tags + overflow_count]
            selected = ranked_tags[:num_tags]
            output = self.to_directory_overview(selected, chat_rel_fnames, detail, overflow_tags=overflow)

        return output, file_report
    
    def get_grep_map(
        self,
        chat_files: Optional[List[str]] = None,
        other_files: Optional[List[str]] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the grep map with file report."""
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
