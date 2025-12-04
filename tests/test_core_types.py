"""Tests for core type definitions and their methods."""

from testslide import TestCase  # type: ignore[import-untyped]
from grepmap.core.types import (
    DetailLevel,
    SignatureInfo,
    FieldInfo,
    Tag,
    RankedTag,
    FileReport
)


class TestDetailLevel(TestCase):
    """Tests for DetailLevel enum."""

    def test_detail_levels_have_correct_values(self):
        """Detail levels should have ascending integer values."""
        self.assertEqual(DetailLevel.LOW, 1)
        self.assertEqual(DetailLevel.MEDIUM, 2)
        self.assertEqual(DetailLevel.HIGH, 3)

    def test_detail_levels_are_ordered(self):
        """Detail levels should support comparison."""
        self.assertLess(DetailLevel.LOW, DetailLevel.MEDIUM)
        self.assertLess(DetailLevel.MEDIUM, DetailLevel.HIGH)


class TestSignatureInfo(TestCase):
    """Tests for SignatureInfo rendering at different detail levels."""

    def test_render_at_low_detail(self):
        """LOW detail should return ellipsis."""
        sig = SignatureInfo(
            parameters=(("self", None), ("x", "int")),
            return_type="bool",
            decorators=()
        )
        self.assertEqual(sig.render(DetailLevel.LOW), "...")

    def test_render_at_medium_detail_without_types(self):
        """MEDIUM detail should show param names only."""
        sig = SignatureInfo(
            parameters=(("self", None), ("x", "int"), ("y", "str")),
            return_type="bool",
            decorators=()
        )
        result = sig.render(DetailLevel.MEDIUM)
        self.assertEqual(result, "(self, x, y)")

    def test_render_at_high_detail_with_types(self):
        """HIGH detail should show full type annotations."""
        sig = SignatureInfo(
            parameters=(("self", None), ("x", "int"), ("y", "str")),
            return_type="bool",
            decorators=()
        )
        result = sig.render(DetailLevel.HIGH)
        self.assertEqual(result, "(self, x: int, y: str) -> bool")

    def test_render_with_deduplication(self):
        """HIGH detail should deduplicate repeated param:type patterns across signatures."""
        # First signature with 'x: int'
        sig1 = SignatureInfo(
            parameters=(("x", "int"),),
            return_type=None,
            decorators=()
        )
        seen = set()
        result1 = sig1.render(DetailLevel.HIGH, seen)
        self.assertEqual(result1, "(x: int)")
        self.assertIn("x:int", seen)

        # Second signature with same 'x: int' - should be deduplicated
        sig2 = SignatureInfo(
            parameters=(("x", "int"),),
            return_type=None,
            decorators=()
        )
        result2 = sig2.render(DetailLevel.HIGH, seen)
        self.assertEqual(result2, "(x)")  # Type elided since x:int already seen


class TestFieldInfo(TestCase):
    """Tests for FieldInfo rendering."""

    def test_render_at_low_detail(self):
        """LOW detail should show name only."""
        field = FieldInfo(name="counter", type_annotation="int")
        self.assertEqual(field.render(DetailLevel.LOW), "counter")

    def test_render_at_medium_detail_simplifies_types(self):
        """MEDIUM detail should simplify complex types."""
        field = FieldInfo(name="callback", type_annotation="Callable[[int], str]")
        result = field.render(DetailLevel.MEDIUM)
        self.assertEqual(result, "callback: Callable")

    def test_render_at_high_detail_shows_full_type(self):
        """HIGH detail should show complete type annotation."""
        field = FieldInfo(name="callback", type_annotation="Callable[[int], str]")
        result = field.render(DetailLevel.HIGH)
        self.assertEqual(result, "callback: Callable[[int], str]")


class TestTag(TestCase):
    """Tests for Tag dataclass."""

    def test_tag_creation_with_minimal_fields(self):
        """Tag should be creatable with required fields only."""
        tag = Tag(
            rel_fname="test.py",
            fname="/abs/test.py",
            line=10,
            name="foo",
            kind="def",
            node_type="function",
            parent_name=None,
            parent_line=None
        )
        self.assertEqual(tag.name, "foo")
        self.assertEqual(tag.kind, "def")
        self.assertIsNone(tag.signature)
        self.assertIsNone(tag.fields)

    def test_tag_is_immutable(self):
        """Tag should be frozen and immutable."""
        tag = Tag(
            rel_fname="test.py",
            fname="/abs/test.py",
            line=10,
            name="foo",
            kind="def",
            node_type="function",
            parent_name=None,
            parent_line=None
        )
        with self.assertRaises(Exception):  # FrozenInstanceError
            tag.name = "bar"  # type: ignore[misc]


class TestRankedTag(TestCase):
    """Tests for RankedTag wrapper."""

    def test_ranked_tag_creation(self):
        """RankedTag should wrap Tag with rank score."""
        tag = Tag(
            rel_fname="test.py",
            fname="/abs/test.py",
            line=10,
            name="foo",
            kind="def",
            node_type="function",
            parent_name=None,
            parent_line=None
        )
        ranked = RankedTag(rank=0.5, tag=tag)

        self.assertEqual(ranked.rank, 0.5)
        self.assertEqual(ranked.tag.name, "foo")

    def test_ranked_tags_are_sortable_by_rank(self):
        """RankedTags should be sortable by their rank."""
        tag1 = Tag("a.py", "/a.py", 1, "foo", "def", "function", None, None)
        tag2 = Tag("b.py", "/b.py", 1, "bar", "def", "function", None, None)

        rt1 = RankedTag(0.3, tag1)
        rt2 = RankedTag(0.7, tag2)

        sorted_tags = sorted([rt1, rt2], key=lambda x: x.rank, reverse=True)
        self.assertEqual(sorted_tags[0].tag.name, "bar")
        self.assertEqual(sorted_tags[1].tag.name, "foo")


class TestFileReport(TestCase):
    """Tests for FileReport aggregate statistics."""

    def test_file_report_creation(self):
        """FileReport should store processing statistics."""
        report = FileReport(
            excluded={"file1.py": "[EXCLUDED] Not found"},
            definition_matches=150,
            reference_matches=300,
            total_files_considered=10
        )

        self.assertEqual(len(report.excluded), 1)
        self.assertEqual(report.definition_matches, 150)
        self.assertEqual(report.reference_matches, 300)
        self.assertEqual(report.total_files_considered, 10)
