"""
Unit tests for markdown header extraction.

Tests cover:
- Basic header extraction (h1-h6)
- Code block filtering (headers inside ``` blocks should be skipped)
- Parent hierarchy tracking (h2 under h1, h3 under h2, etc.)
- Section snippet extraction (head [...] tail format)
- Edge cases (empty files, only code blocks, malformed markdown)
"""

from grepmap.extraction.markdown import (
    parse_markdown_tags,
    is_markdown_file,
    _find_code_block_ranges,
    _is_inside_code_block,
)


class TestIsMarkdownFile:
    """Test markdown file extension detection."""

    def test_standard_md_extension(self):
        assert is_markdown_file("README.md") is True
        assert is_markdown_file("docs/guide.md") is True

    def test_alternative_extensions(self):
        assert is_markdown_file("file.markdown") is True
        assert is_markdown_file("file.mdown") is True
        assert is_markdown_file("file.mkd") is True

    def test_case_insensitive(self):
        assert is_markdown_file("README.MD") is True
        assert is_markdown_file("file.Markdown") is True

    def test_non_markdown_files(self):
        assert is_markdown_file("script.py") is False
        assert is_markdown_file("style.css") is False
        assert is_markdown_file("readme.txt") is False


class TestCodeBlockDetection:
    """Test code block range finding and position checking."""

    def test_finds_backtick_blocks(self):
        code = "text\n```python\ncode\n```\nmore text"
        ranges = _find_code_block_ranges(code)
        assert len(ranges) == 1
        # Block should span from ``` to closing ```
        start, end = ranges[0]
        assert code[start:end] == "```python\ncode\n```"

    def test_finds_tilde_blocks(self):
        code = "text\n~~~\ncode\n~~~\nmore text"
        ranges = _find_code_block_ranges(code)
        assert len(ranges) == 1

    def test_finds_multiple_blocks(self):
        code = "```\na\n```\ntext\n```\nb\n```"
        ranges = _find_code_block_ranges(code)
        assert len(ranges) == 2

    def test_position_inside_code_block(self):
        code = "text\n```\ncode\n```\nmore"
        ranges = _find_code_block_ranges(code)
        # Position inside code block
        code_pos = code.find("code")
        assert _is_inside_code_block(code_pos, ranges) is True
        # Position outside code block
        more_pos = code.find("more")
        assert _is_inside_code_block(more_pos, ranges) is False


class TestBasicHeaderExtraction:
    """Test extraction of markdown headers."""

    def test_extracts_h1_header(self):
        code = "# Main Title\n\nSome content"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) == 1
        assert tags[0].name == "Main Title"
        assert tags[0].node_type == "h1"
        assert tags[0].line == 1

    def test_extracts_multiple_header_levels(self):
        code = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) == 6
        for i, tag in enumerate(tags):
            assert tag.node_type == f"h{i+1}"

    def test_header_with_trailing_hashes(self):
        """Headers can have trailing hashes: # Title #####"""
        code = "# Title #####\n"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert tags[0].name == "Title"

    def test_empty_file_returns_no_tags(self):
        tags = parse_markdown_tags("", "/path/file.md", "file.md")
        assert tags == []

    def test_file_without_headers(self):
        code = "Just some text\nwithout any headers"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert tags == []


class TestCodeBlockFiltering:
    """Test that headers inside code blocks are skipped."""

    def test_skips_header_in_backtick_block(self):
        code = """# Real Header

```bash
# This is a comment, not a header
echo "hello"
```

## Another Real Header
"""
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        names = [t.name for t in tags]
        assert "Real Header" in names
        assert "Another Real Header" in names
        assert "This is a comment, not a header" not in names

    def test_skips_header_in_tilde_block(self):
        code = """# Real Header

~~~python
# Comment in code
print("hello")
~~~
"""
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) == 1
        assert tags[0].name == "Real Header"

    def test_multiple_code_blocks(self):
        code = """# Start

```
# In block 1
```

## Middle

```
# In block 2
```

### End
"""
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        names = [t.name for t in tags]
        assert names == ["Start", "Middle", "End"]


class TestParentHierarchy:
    """Test parent-child relationships between headers."""

    def test_h2_has_h1_parent(self):
        code = "# Parent\n## Child"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        h2 = [t for t in tags if t.node_type == "h2"][0]
        assert h2.parent_name == "Parent"
        assert h2.parent_line == 1

    def test_h3_has_h2_parent_not_h1(self):
        code = "# H1\n## H2\n### H3"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        h3 = [t for t in tags if t.node_type == "h3"][0]
        assert h3.parent_name == "H2"  # Direct parent is H2, not H1

    def test_h1_has_no_parent(self):
        code = "# Top Level"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert tags[0].parent_name is None
        assert tags[0].parent_line is None

    def test_sibling_headers_share_parent(self):
        code = "# Parent\n## Child 1\n## Child 2"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        children = [t for t in tags if t.node_type == "h2"]
        assert all(c.parent_name == "Parent" for c in children)

    def test_new_h1_resets_hierarchy(self):
        code = "# First Section\n## Sub 1\n# Second Section\n## Sub 2"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        sub2 = [t for t in tags if t.name == "Sub 2"][0]
        assert sub2.parent_name == "Second Section"


class TestSectionSnippets:
    """Test section content preview extraction."""

    def test_extracts_snippet_from_prose(self):
        code = "# Title\n\nThis is the section content that should appear in the snippet."
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert tags[0].signature is not None
        assert "section content" in tags[0].signature.raw

    def test_snippet_truncates_long_content(self):
        code = "# Title\n\n" + "word " * 100  # Very long content
        tags = parse_markdown_tags(code, "/path/file.md", "file.md", snippet_length=80)
        assert tags[0].signature is not None
        assert "[...]" in tags[0].signature.raw
        assert len(tags[0].signature.raw) < 120  # Should be reasonably bounded

    def test_snippet_shows_head_and_tail(self):
        """Long snippets should show head [...] tail format."""
        code = "# Title\n\nStart of content. " + "middle " * 50 + "End of content."
        tags = parse_markdown_tags(code, "/path/file.md", "file.md", snippet_length=80)
        raw = tags[0].signature.raw
        assert "Start" in raw
        assert "[...]" in raw
        assert "End" in raw or "content" in raw  # Tail should be present

    def test_no_snippet_for_code_only_section(self):
        """Sections with only code blocks should have no snippet."""
        code = """# Title

```python
def foo():
    pass
```

## Next Section
"""
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        title_tag = [t for t in tags if t.name == "Title"][0]
        # Code-only section should have no prose snippet
        assert title_tag.signature is None or title_tag.signature.raw is None

    def test_snippet_strips_markdown_formatting(self):
        """Snippets should strip bold, italic, links."""
        code = "# Title\n\n**Bold** and *italic* with [link](url) here."
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        raw = tags[0].signature.raw
        # Should contain text but not markdown syntax
        assert "Bold" in raw
        assert "**" not in raw
        assert "[link]" not in raw


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_header_at_end_of_file(self):
        code = "# Only Header"  # No trailing newline
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) == 1

    def test_consecutive_headers(self):
        code = "# H1\n## H2\n### H3"  # No content between
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) == 3

    def test_header_with_special_characters(self):
        code = "# Title: With `code` and *emphasis*"
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert "Title" in tags[0].name

    def test_unclosed_code_block(self):
        """Unclosed code blocks shouldn't break parsing."""
        code = "# Header\n```\nunclosed code block\n# Not a header"
        # Should not crash - behavior may vary
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        assert len(tags) >= 1  # At least the first header

    def test_nested_code_fences(self):
        """Code blocks with nested fences."""
        code = '''# Header

````markdown
```python
code
```
````
'''
        tags = parse_markdown_tags(code, "/path/file.md", "file.md")
        # Should only find the real header
        assert len(tags) == 1
        assert tags[0].name == "Header"
