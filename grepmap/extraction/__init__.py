"""
Tag extraction module for RepoMapper.

This module provides focused, self-contained utilities for extracting code
symbols (tags) from source files using tree-sitter. The extraction logic
is split into specialized modules:

- parser: Main TagParser class and tag extraction orchestration
- signatures: Function signature extraction from AST nodes
- fields: Class field extraction from AST nodes

Public API:
- TagParser: Main parser class for extracting tags
- create_tag_parser: Factory function for creating configured parsers
- get_tags_raw: Convenience function for one-shot tag extraction
- extract_signature_info: Extract function signature from AST node
- extract_class_fields: Extract class fields from AST node
"""

from grepmap.extraction.parser import (
    TagParser,
    create_tag_parser,
    get_tags_raw
)
from grepmap.extraction.signatures import extract_signature_info
from grepmap.extraction.fields import extract_class_fields

__all__ = [
    'TagParser',
    'create_tag_parser',
    'get_tags_raw',
    'extract_signature_info',
    'extract_class_fields',
]
