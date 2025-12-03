"""
Class field extraction from tree-sitter AST nodes.

This module provides utilities for extracting class field information from
tree-sitter class_definition nodes. It captures annotated assignments in
class bodies to enable dataclass-style display of class structure.

The extracted field information supports multi-detail rendering where class
definitions can be shown with varying levels of type annotation detail.
"""

from typing import Optional, Tuple
from grepmap.core.types import FieldInfo


def extract_class_fields(class_node, code_bytes: bytes) -> Optional[Tuple[FieldInfo, ...]]:
    """Extract class fields from tree-sitter class_definition node.

    Captures annotated assignments in the class body (not in methods) to provide
    a dataclass-style view of class structure. This is particularly useful for
    data classes, config classes, and other structured types.

    The extraction focuses on annotated assignments:
    - field: Type
    - field: Type = value

    Instance variables assigned in __init__ are NOT captured, as they're not
    part of the class definition itself. Only class-level annotated assignments
    are extracted.

    Returns up to 10 fields to keep output concise and avoid overwhelming the
    display with large classes.

    Tree-sitter structure:
    - class_definition -> block -> [statements]
    - Look for expression_statement -> assignment with type annotation

    Args:
        class_node: Tree-sitter node of type 'class_definition'
        code_bytes: Source code as bytes (needed for decoding node text)

    Returns:
        Tuple of FieldInfo objects (up to 10 fields), or None if no fields found
    """
    if class_node is None:
        return None

    fields = []

    # Find the block (class body) within the class definition
    body = None
    for child in class_node.children:
        if child.type == 'block':
            body = child
            break

    if not body:
        return None

    # Walk through statements in the class body
    for stmt in body.children:
        if len(fields) >= 10:  # Limit to 10 fields for concise output
            break

        # Look for annotated assignments: field: Type or field: Type = value
        field_info = _extract_field_from_statement(stmt)
        if field_info:
            fields.append(field_info)

    return tuple(fields) if fields else None


def _extract_field_from_statement(stmt) -> Optional[FieldInfo]:
    """Extract field information from a statement node.

    Looks for patterns indicating class field definitions:
    - expression_statement -> assignment with type annotation
    - Assignment with identifier and type children

    Args:
        stmt: Tree-sitter statement node

    Returns:
        FieldInfo object if field found, None otherwise
    """
    # Handle expression_statement containing assignment
    if stmt.type == 'expression_statement':
        expr = stmt.children[0] if stmt.children else None
        if expr and expr.type == 'assignment':
            return _extract_field_from_assignment(expr)

    return None


def _extract_field_from_assignment(assignment_node) -> Optional[FieldInfo]:
    """Extract field information from an assignment node.

    Parses assignment to find annotated field definitions.
    Pattern: identifier : type (= value)?

    Args:
        assignment_node: Tree-sitter node of type 'assignment'

    Returns:
        FieldInfo if annotated assignment found, None otherwise
    """
    field_name = None
    type_annotation = None

    # Walk children to find identifier (field name) and type annotation
    for child in assignment_node.children:
        if child.type == 'identifier' and field_name is None:
            # First identifier is the field name
            field_name = child.text.decode('utf-8') if child.text else None

        elif child.type == 'type':
            # Type annotation
            type_annotation = child.text.decode('utf-8') if child.text else None

    # Only return FieldInfo if we found both name and type
    # (unannotated assignments are not considered fields for our purposes)
    if field_name and type_annotation:
        return FieldInfo(name=field_name, type_annotation=type_annotation)

    return None
