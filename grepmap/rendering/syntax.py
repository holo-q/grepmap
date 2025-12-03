"""
Syntax highlighting utilities for GrepMap rendering.

Provides color mapping and icon/emoji utilities for rich terminal output.
These functions are used across renderers to maintain consistent styling.
"""


def get_token_color(node_type: str) -> str:
    """Map tree-sitter node type to Rich color/style for token-level coloring.

    Returns Rich color/style strings for granular syntax highlighting within
    code snippets. Supports Python node types from tree-sitter parsing.

    Args:
        node_type: Tree-sitter node type string (e.g., "def", "class", "string")

    Returns:
        Rich color/style string (e.g., "bold magenta", "dim white")

    Example:
        >>> get_token_color("def")
        'bold magenta'
        >>> get_token_color("string")
        'green'
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


def get_symbol_icon(node_type: str) -> str:
    """Get icon/emoji for different symbol types in directory views.

    Returns simple text icons for visual distinction between symbol types.
    Enhances readability in hierarchical directory overviews.

    Args:
        node_type: Symbol type (e.g., "class", "function", "method")

    Returns:
        Icon string with trailing space

    Note:
        Uses simple ASCII-compatible symbols for maximum compatibility.
    """
    icon_map = {
        'class': 'C ',
        'function': 'f ',
        'method': 'm ',
        'variable': 'v ',
        'constant': 'K ',
        'interface': 'I ',
        'enum': 'E ',
        'module': 'M ',
    }
    return icon_map.get(node_type, '• ')


def get_symbol_color(node_type: str) -> str:
    """Get Rich color for symbol types in directory view.

    Provides consistent color coding for different symbol types across
    the directory overview renderer. Colors convey semantic meaning.

    Args:
        node_type: Symbol type (e.g., "class", "function", "variable")

    Returns:
        Rich color/style string

    Example:
        >>> get_symbol_color("class")
        'bold cyan'
        >>> get_symbol_color("function")
        'bold yellow'
    """
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
