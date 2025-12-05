"""
Tag extraction module for grepmap.

This module provides focused, self-contained utilities for extracting code
symbols (tags) from source files using tree-sitter. The extraction logic
is split into specialized modules:

- parser: Main TagParser class and tag extraction orchestration
- signatures: Function signature extraction from AST nodes
- fields: Class field extraction from AST nodes
- markdown: Regex-based markdown header extraction

Public API:
- TagParser: Main parser class for extracting tags
- create_tag_parser: Factory function for creating configured parsers
- get_tags_raw: Convenience function for one-shot tag extraction
- extract_signature_info: Extract function signature from AST node
- extract_class_fields: Extract class fields from AST node
- parse_markdown_tags: Extract headers from markdown files
- is_markdown_file: Check if a file is markdown by extension
"""

from grepmap.extraction.parser import (
    TagParser,
    create_tag_parser,
    get_tags_raw
)
from grepmap.extraction.signatures import extract_signature_info
from grepmap.extraction.fields import extract_class_fields
from grepmap.extraction.markdown import (
    parse_markdown_tags,
    is_markdown_file
)

__all__ = [
    'TagParser',
    'create_tag_parser',
    'get_tags_raw',
    'extract_signature_info',
    'extract_class_fields',
    'parse_markdown_tags',
    'is_markdown_file',
]
