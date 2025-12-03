"""
Tag parser for extracting code definitions and references from source files.

This module provides the core TagParser class that uses tree-sitter to parse
source code and extract structured tag information (definitions and references)
with full metadata including parent scopes, signatures, and fields.

The parser integrates with tree-sitter query files (.scm) to extract semantic
information from ASTs in a language-agnostic manner.
"""

from typing import List, Optional, Callable

from grepmap.core.types import Tag
from grepmap.extraction.signatures import extract_signature_info
from grepmap.extraction.fields import extract_class_fields


class TagParser:
    """Parser for extracting tags from source code using tree-sitter.

    This class encapsulates the logic for parsing source files and extracting
    structured tag information including definitions, references, parent scopes,
    function signatures, and class fields.

    The parser uses tree-sitter queries (defined in .scm files) to identify
    relevant nodes in the AST and extracts metadata suitable for code mapping
    and navigation.

    Attributes:
        language: Tree-sitter Language object for parsing
        parser: Tree-sitter Parser instance
        query: Pre-compiled tree-sitter Query for tag extraction
        lang_name: Language name (e.g., "python", "javascript")
    """

    def __init__(self, language, parser, query, lang_name: str):
        """Initialize TagParser with tree-sitter components.

        Args:
            language: Tree-sitter Language object
            parser: Tree-sitter Parser instance
            query: Pre-compiled Query object with tag extraction patterns
            lang_name: Language name for context
        """
        self.language = language
        self.parser = parser
        self.query = query
        self.lang_name = lang_name

    def parse_tags(
        self,
        code: str,
        fname: str,
        rel_fname: str,
        error_handler: Optional[Callable[[str], None]] = None
    ) -> List[Tag]:
        """Parse source code and extract all tags (definitions and references).

        This is the main entry point for tag extraction. It parses the source code,
        runs tree-sitter queries to identify symbols, and constructs Tag objects
        with complete metadata including parent scopes and signature/field info.

        The extraction process:
        1. Parse code into tree-sitter AST
        2. Run pre-compiled query to find definitions and references
        3. For each captured node, determine its kind (def/ref) and node_type
        4. Walk up the AST to find parent scope (enclosing class/function)
        5. Extract signature info for functions or field info for classes
        6. Construct and return Tag objects

        Args:
            code: Source code text to parse
            fname: Absolute file path for tag metadata
            rel_fname: Relative file path for display
            error_handler: Optional callback for reporting errors

        Returns:
            List of Tag objects representing all extracted definitions and references.
            Returns empty list if parsing fails or no tags found.
        """
        try:
            code_bytes = bytes(code, "utf-8")
            tree = self.parser.parse(code_bytes)

            # Import QueryCursor here to avoid top-level import issues
            from tree_sitter import QueryCursor
            cursor = QueryCursor(self.query)
            captures = cursor.captures(tree.root_node)

            tags = []

            # Process captures as a dictionary of {capture_name: [nodes]}
            for capture_name, nodes in captures.items():
                for node in nodes:
                    # Determine tag kind (def or ref) from capture name
                    if "name.definition" in capture_name:
                        kind = "def"
                    elif "name.reference" in capture_name:
                        kind = "ref"
                    else:
                        # Skip other capture types not relevant for tagging
                        continue

                    # Extract semantic node type from capture name
                    # e.g., "name.definition.function" -> "function"
                    # e.g., "name.definition.class" -> "class"
                    parts = capture_name.split('.')
                    node_type = parts[-1] if len(parts) > 2 else "unknown"

                    line_num = node.start_point[0] + 1
                    name = node.text.decode('utf-8') if node.text else ""

                    # Find parent scope (class/function containing this definition)
                    # Walk up the AST to find enclosing class or function
                    parent_name, parent_line = self._find_parent_scope(node)

                    # Find the definition node containing this name identifier
                    # This is needed to extract signature/field information
                    my_definition = self._find_containing_definition(node)

                    # Extract signature for functions or fields for classes
                    # Only for definitions, not references
                    signature = None
                    fields = None
                    if kind == "def" and my_definition:
                        if node_type == "function":
                            signature = extract_signature_info(my_definition, code_bytes)
                        elif node_type == "class":
                            fields = extract_class_fields(my_definition, code_bytes)

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
            if error_handler:
                error_handler(f"Error parsing {fname}: {e}")
            return []

    def _find_containing_definition(self, node):
        """Find the definition node (class/function) containing the given identifier.

        This walks up from the identifier node to find its enclosing definition,
        which is needed to extract signature or field information.

        Args:
            node: Tree-sitter node (typically an identifier)

        Returns:
            Tree-sitter node representing the containing definition, or None
        """
        current = node.parent
        while current and current.type not in (
            'class_definition',
            'function_definition',
            'method_definition'
        ):
            current = current.parent
        return current

    def _find_parent_scope(self, node):
        """Find the parent scope (class/function) containing this node.

        Walks up the AST from the node to find the first enclosing class or
        function definition. Skips over the node's own definition to find
        the parent scope (e.g., a method's containing class).

        This enables hierarchical rendering where methods are shown nested
        under their classes.

        Args:
            node: Tree-sitter node to find parent for

        Returns:
            Tuple of (parent_name, parent_line) or (None, None) if top-level
        """
        # First, get to the actual definition node that contains this name
        my_definition = self._find_containing_definition(node)

        # Now search for the parent definition (skip our own definition)
        if my_definition:
            search_node = my_definition.parent
            while search_node:
                if search_node.type in (
                    'class_definition',
                    'function_definition',
                    'method_definition'
                ):
                    # Try to get the 'name' field (more reliable than first identifier)
                    name_node = None
                    for child in search_node.children:
                        if child.type == 'identifier':
                            name_node = child
                            break

                    if name_node:
                        parent_name = name_node.text.decode('utf-8') if name_node.text else None
                        parent_line = name_node.start_point[0] + 1
                        return parent_name, parent_line

                search_node = search_node.parent

        return None, None


def create_tag_parser(
    fname: str,
    scm_fname: Optional[str] = None,
    error_handler: Optional[Callable[[str], None]] = None
) -> Optional[TagParser]:
    """Factory function to create a TagParser for a given file.

    This function handles all the setup required to create a TagParser:
    1. Determines the language from the filename
    2. Loads the tree-sitter language and parser
    3. Loads and compiles the tree-sitter query from .scm file
    4. Returns a configured TagParser instance

    This is the recommended way to create TagParser instances, as it
    encapsulates all the tree-sitter setup complexity.

    Args:
        fname: File path to parse (used to determine language)
        scm_fname: Optional path to .scm query file (auto-detected if None)
        error_handler: Optional callback for reporting errors

    Returns:
        Configured TagParser instance, or None if setup fails
    """
    try:
        from grep_ast import filename_to_lang
        from grep_ast.tsl import get_language, get_parser
        from scm import get_scm_fname
        from utils import read_text
    except ImportError as e:
        if error_handler:
            error_handler(f"Required dependencies not available: {e}")
        return None

    # Determine language from filename
    lang = filename_to_lang(fname)
    if not lang:
        return None

    try:
        language = get_language(lang)
        parser = get_parser(lang)
    except Exception as err:
        if error_handler:
            error_handler(f"Failed to load tree-sitter for {lang}: {err}")
        return None

    # Load query file
    if scm_fname is None:
        scm_fname = get_scm_fname(lang)

    if not scm_fname:
        return None

    query_text = read_text(scm_fname, silent=True)
    if not query_text:
        return None

    try:
        query = language.query(query_text)
    except Exception as err:
        if error_handler:
            error_handler(f"Failed to compile query for {lang}: {err}")
        return None

    return TagParser(language, parser, query, lang)


def get_tags_raw(
    fname: str,
    rel_fname: str,
    read_text_func: Callable[[str], Optional[str]],
    error_handler: Optional[Callable[[str], None]] = None
) -> List[Tag]:
    """Parse file to extract tags using Tree-sitter.

    This is a convenience function that combines TagParser creation and
    tag extraction in a single call. It's the main entry point for
    extracting tags from a file.

    The function:
    1. Creates a TagParser for the file's language
    2. Reads the file content
    3. Parses and extracts tags
    4. Returns the tag list

    Args:
        fname: Absolute file path to parse
        rel_fname: Relative file path for tag metadata
        read_text_func: Function to read file content
        error_handler: Optional callback for reporting errors

    Returns:
        List of extracted Tag objects, or empty list if parsing fails
    """
    # Create parser for this file's language
    parser = create_tag_parser(fname, error_handler=error_handler)
    if not parser:
        return []

    # Read file content
    code = read_text_func(fname)
    if not code:
        return []

    # Parse and extract tags
    return parser.parse_tags(code, fname, rel_fname, error_handler)
