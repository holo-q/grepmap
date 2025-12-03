"""
Function signature extraction from tree-sitter AST nodes.

This module provides utilities for extracting detailed function signature
information from tree-sitter function_definition nodes, including:
- Parameter names and type annotations
- Return type annotations
- Decorators (e.g., @staticmethod, @property)

The extracted signature information enables multi-detail rendering where
signatures can be shown at different verbosity levels (name only, with params,
or with full type annotations).
"""

from typing import Optional
from grepmap.core.types import SignatureInfo


def extract_signature_info(func_node, code_bytes: bytes) -> Optional[SignatureInfo]:
    """Extract function signature from tree-sitter function_definition node.

    Parses parameters with type annotations and return type for multi-detail rendering.
    This function walks the AST of a function definition to extract:
    - All parameters (including typed, default, *args, **kwargs variants)
    - Type annotations for parameters
    - Return type annotation
    - Decorators applied to the function

    The extracted information is used during tag extraction to populate the
    Tag.signature field, enabling variable-detail rendering without re-parsing.

    Tree-sitter structure context:
    - function_definition contains 'parameters' and 'type' (return) nodes
    - decorated_definition wraps function_definition and contains 'decorator' nodes
    - Decorators are siblings of the function_definition, not children

    Args:
        func_node: Tree-sitter node of type 'function_definition' or 'method_definition'
        code_bytes: Source code as bytes (needed for decoding node text)

    Returns:
        SignatureInfo with parameters, return_type, and decorators, or None if parsing fails
    """
    if func_node is None:
        return None

    parameters = []
    return_type = None
    decorators = []

    # Extract decorators from parent decorated_definition node
    # Structure: decorated_definition -> [decorator, decorator, ..., function_definition]
    parent = func_node.parent
    if parent and parent.type == 'decorated_definition':
        for sibling in parent.children:
            if sibling.type == 'decorator':
                # Get the decorator name (skip the @ symbol)
                decorator_name = _extract_decorator_name(sibling)
                if decorator_name:
                    decorators.append(decorator_name)

    # Extract parameters and return type from function_definition children
    for child in func_node.children:
        # Extract parameters block
        if child.type == 'parameters':
            for param in child.children:
                param_info = _extract_parameter(param)
                if param_info:
                    parameters.append(param_info)

        # Extract return type annotation
        elif child.type == 'type':
            return_type = child.text.decode('utf-8') if child.text else None

    return SignatureInfo(
        parameters=tuple(parameters),
        return_type=return_type,
        decorators=tuple(decorators)
    )


def _extract_decorator_name(decorator_node) -> Optional[str]:
    """Extract decorator name from a decorator node.

    Handles different decorator patterns:
    - Simple: @property -> "property"
    - With args: @dataclass(frozen=True) -> "dataclass"
    - Attribute: @functools.wraps -> "functools.wraps"

    Args:
        decorator_node: Tree-sitter node of type 'decorator'

    Returns:
        Decorator name as string, or None if extraction fails
    """
    for child in decorator_node.children:
        if child.type == 'identifier':
            # Simple decorator: @property
            return child.text.decode('utf-8') if child.text else None

        elif child.type == 'call':
            # Decorator with arguments: @dataclass(frozen=True)
            for call_child in child.children:
                if call_child.type == 'identifier':
                    return call_child.text.decode('utf-8') if call_child.text else None

        elif child.type == 'attribute':
            # Decorator with module: @functools.wraps
            return child.text.decode('utf-8') if child.text else None

    return None


def _extract_parameter(param_node) -> Optional[tuple]:
    """Extract parameter information from a parameter node.

    Handles all Python parameter variants:
    - Simple: param
    - Typed: param: Type
    - Default: param=value
    - Typed with default: param: Type = value
    - Variable positional: *args
    - Variable keyword: **kwargs

    Args:
        param_node: Tree-sitter node representing a parameter

    Returns:
        Tuple of (name, type_annotation) where type_annotation may be None,
        or None if this node doesn't represent a parameter
    """
    param_type = param_node.type

    if param_type == 'identifier':
        # Simple parameter without type annotation
        param_name = param_node.text.decode('utf-8') if param_node.text else ''
        return (param_name, None) if param_name else None

    elif param_type == 'typed_parameter':
        # Parameter with type annotation: name: Type
        param_name = None
        param_type_str = None
        for child in param_node.children:
            if child.type == 'identifier':
                param_name = child.text.decode('utf-8') if child.text else ''
            elif child.type == 'type':
                param_type_str = child.text.decode('utf-8') if child.text else ''
        return (param_name, param_type_str) if param_name else None

    elif param_type == 'default_parameter':
        # Parameter with default value: name=value
        for child in param_node.children:
            if child.type == 'identifier':
                param_name = child.text.decode('utf-8') if child.text else ''
                return (param_name, None) if param_name else None

    elif param_type == 'typed_default_parameter':
        # Parameter with type and default: name: Type = value
        param_name = None
        param_type_str = None
        for child in param_node.children:
            if child.type == 'identifier':
                param_name = child.text.decode('utf-8') if child.text else ''
            elif child.type == 'type':
                param_type_str = child.text.decode('utf-8') if child.text else ''
        return (param_name, param_type_str) if param_name else None

    elif param_type == 'list_splat_pattern':
        # *args
        for child in param_node.children:
            if child.type == 'identifier':
                param_name = '*' + (child.text.decode('utf-8') if child.text else '')
                return (param_name, None) if param_name else None

    elif param_type == 'dictionary_splat_pattern':
        # **kwargs
        for child in param_node.children:
            if child.type == 'identifier':
                param_name = '**' + (child.text.decode('utf-8') if child.text else '')
                return (param_name, None) if param_name else None

    return None
