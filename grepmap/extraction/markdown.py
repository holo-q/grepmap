"""
Markdown parser for extracting headers and section previews.

This module provides a simple regex-based parser for markdown files,
extracting headers as Tag objects with section preview snippets.
Unlike code files, markdown doesn't use tree-sitter - just regex.

Output format mirrors code tags:
- kind="def" for all headers (they define sections)
- node_type="h1", "h2", "h3", etc. based on header level
- name=header text
- signature.raw=truncated preview of section content
"""

import re
from typing import List, Optional, Tuple

from grepmap.core.types import Tag, SignatureInfo


# Match markdown headers: # Header, ## Header, etc.
# Captures: group(1)=hashes, group(2)=header text
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*#*)?$', re.MULTILINE)

# Match code blocks (fenced with ``` or ~~~)
CODE_BLOCK_PATTERN = re.compile(r'^(`{3,}|~{3,}).*?^\1', re.MULTILINE | re.DOTALL)


def _find_code_block_ranges(code: str) -> List[Tuple[int, int]]:
    """Find all code block ranges (start, end) positions in the markdown."""
    ranges = []
    for match in CODE_BLOCK_PATTERN.finditer(code):
        ranges.append((match.start(), match.end()))
    return ranges


def _is_inside_code_block(pos: int, code_blocks: List[Tuple[int, int]]) -> bool:
    """Check if a position is inside any code block."""
    for start, end in code_blocks:
        if start <= pos < end:
            return True
    return False


def parse_markdown_tags(
    code: str,
    fname: str,
    rel_fname: str,
    snippet_length: int = 80
) -> List[Tag]:
    """Parse markdown file and extract headers as tags.

    Each header becomes a Tag with:
    - kind="def" (headers define sections)
    - node_type="h1", "h2", etc.
    - name=header text (cleaned)
    - signature=preview snippet of section content

    Parent hierarchy is tracked so h2 under h1 shows nesting.

    Args:
        code: Markdown file content
        fname: Absolute file path
        rel_fname: Relative file path for display
        snippet_length: Max chars for section preview (default 80)

    Returns:
        List of Tag objects for each header
    """
    if not code:
        return []

    # Find code block ranges to skip headers inside them
    code_blocks = _find_code_block_ranges(code)

    tags = []

    # Track parent headers at each level for nesting
    # parent_stack[level] = (name, line_number)
    parent_stack: dict[int, tuple[str, int]] = {}

    for match in HEADER_PATTERN.finditer(code):
        # Skip headers inside code blocks
        if _is_inside_code_block(match.start(), code_blocks):
            continue
        hashes = match.group(1)
        header_text = match.group(2).strip()
        level = len(hashes)

        # Calculate line number (1-indexed)
        line_num = code[:match.start()].count('\n') + 1

        # Find parent: nearest header with lower level
        parent_name = None
        parent_line = None
        for check_level in range(level - 1, 0, -1):
            if check_level in parent_stack:
                parent_name, parent_line = parent_stack[check_level]
                break

        # Update parent stack: this header is now the parent for deeper levels
        parent_stack[level] = (header_text, line_num)
        # Clear any deeper levels (they're no longer valid parents)
        for deeper in list(parent_stack.keys()):
            if deeper > level:
                del parent_stack[deeper]

        # Extract section content preview (text until next header or EOF)
        snippet = _extract_section_snippet(code, match.end(), snippet_length, code_blocks)

        # Create signature info with the snippet as "raw" representation
        signature = None
        if snippet:
            signature = SignatureInfo(
                parameters=(),
                return_type=None,
                decorators=(),
                raw=snippet
            )

        tags.append(Tag(
            rel_fname=rel_fname,
            fname=fname,
            line=line_num,
            name=header_text,
            kind="def",
            node_type=f"h{level}",
            parent_name=parent_name,
            parent_line=parent_line,
            signature=signature,
            fields=None
        ))

    return tags


def _extract_section_snippet(
    code: str,
    start_pos: int,
    max_length: int,
    code_blocks: List[Tuple[int, int]]
) -> Optional[str]:
    """Extract a preview snippet of section content after a header.

    Skips code blocks and finds actual prose text for the preview.
    Collects text from start_pos until the next real header (not inside code block) or EOF.

    Args:
        code: Full markdown content
        start_pos: Position after the header line
        max_length: Maximum snippet length
        code_blocks: List of (start, end) positions of code blocks

    Returns:
        Cleaned, truncated snippet or None if section is empty/only code
    """
    # Find the next header that's NOT inside a code block
    end_pos = len(code)
    search_pos = start_pos
    while True:
        next_header = HEADER_PATTERN.search(code, search_pos)
        if not next_header:
            break
        if not _is_inside_code_block(next_header.start(), code_blocks):
            end_pos = next_header.start()
            break
        # Header is inside code block, keep searching after it
        search_pos = next_header.end()

    # Extract section content
    section = code[start_pos:end_pos]

    if not section.strip():
        return None

    # Remove code blocks entirely to find prose text
    # This handles both ``` and ~~~ fenced blocks
    section_no_code = CODE_BLOCK_PATTERN.sub('', section)

    # Also remove inline code spans for cleaner preview
    section_no_code = re.sub(r'`[^`]+`', '', section_no_code)

    # Remove common markdown formatting for cleaner preview
    # - Links: [text](url) -> text
    section_no_code = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', section_no_code)
    # - Bold/italic markers
    section_no_code = re.sub(r'\*+([^*]+)\*+', r'\1', section_no_code)
    section_no_code = re.sub(r'_+([^_]+)_+', r'\1', section_no_code)

    # Collapse whitespace and clean up
    section_no_code = re.sub(r'\s+', ' ', section_no_code)
    section_no_code = section_no_code.strip()

    if not section_no_code:
        return None

    # Truncate with head [...] tail format if too long
    if len(section_no_code) > max_length:
        # Split budget: ~60% head, ~40% tail
        head_len = int(max_length * 0.6)
        tail_len = max_length - head_len - 6  # 6 chars for " [...] "

        # Extract head, break at word boundary
        head = section_no_code[:head_len]
        last_space = head.rfind(' ')
        if last_space > head_len // 2:
            head = head[:last_space]
        head = head.rstrip('.,;:')

        # Extract tail, break at word boundary
        tail = section_no_code[-tail_len:]
        first_space = tail.find(' ')
        if first_space > 0 and first_space < tail_len // 2:
            tail = tail[first_space + 1:]
        tail = tail.lstrip('.,;:')

        section_no_code = f"{head} [...] {tail}"

    return section_no_code


def is_markdown_file(fname: str) -> bool:
    """Check if a file is a markdown file by extension."""
    return fname.lower().endswith(('.md', '.markdown', '.mdown', '.mkd'))
