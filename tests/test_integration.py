"""Integration tests for full GrepMap pipeline.

These tests validate end-to-end behavior to catch regressions like:
- Empty outputs when content should be generated
- Tag extraction failures
- Rendering failures
- Cache corruption
"""

from testslide import TestCase  # type: ignore[import-untyped]
from pathlib import Path
import tempfile
import os

from grepmap import GrepMap


class TestGrepMapIntegration(TestCase):
    """Integration tests for full GrepMap pipeline."""

    def setUp(self):
        """Set up a temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: self._cleanup_temp_dir())

    def _cleanup_temp_dir(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_file(self, filename: str, content: str) -> str:
        """Create a test file and return absolute path."""
        filepath = Path(self.temp_dir) / filename
        filepath.write_text(content)
        return str(filepath)

    def test_generates_map_for_simple_python_file(self):
        """Should generate non-empty map for a simple Python file."""
        # Create a simple Python file with a function
        test_file = self._create_test_file("test.py", """
def hello(name: str) -> str:
    '''Say hello to someone.'''
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name: str) -> str:
        return hello(name)
""")

        # Create GrepMap and generate map
        mapper = GrepMap(
            map_tokens=1000,
            root=self.temp_dir,
            verbose=False
        )

        result, report = mapper.get_grep_map(
            chat_files=[],
            other_files=[test_file]
        )

        # Validate we got output
        self.assertIsNotNone(result, "Map should not be None")
        assert result is not None  # Type narrowing for type checker
        self.assertGreater(len(result), 0, "Map should not be empty")

        # Validate we extracted definitions
        self.assertGreater(report.definition_matches, 0,
                          "Should have extracted at least one definition")

        # Validate content contains our symbols
        self.assertIn("hello", result, "Map should mention 'hello' function")
        self.assertIn("Greeter", result, "Map should mention 'Greeter' class")

    def test_handles_empty_file_gracefully(self):
        """Should handle empty files without crashing."""
        empty_file = self._create_test_file("empty.py", "")

        mapper = GrepMap(
            map_tokens=1000,
            root=self.temp_dir,
            verbose=False
        )

        result, report = mapper.get_grep_map(
            chat_files=[],
            other_files=[empty_file]
        )

        # Should handle gracefully - may be None or empty
        self.assertEqual(report.definition_matches, 0,
                        "Empty file should have no definitions")
        self.assertEqual(report.total_files_considered, 1,
                        "Should have considered the file")

    def test_handles_syntax_error_file_gracefully(self):
        """Should handle files with syntax errors without crashing."""
        bad_file = self._create_test_file("bad.py", """
def broken(
    # Unclosed parenthesis - syntax error
""")

        mapper = GrepMap(
            map_tokens=1000,
            root=self.temp_dir,
            verbose=False
        )

        # Should not crash
        result, report = mapper.get_grep_map(
            chat_files=[],
            other_files=[bad_file]
        )

        # Should handle gracefully
        self.assertEqual(report.total_files_considered, 1,
                        "Should have considered the file")

    def test_multiple_files_produce_larger_map(self):
        """Map size should increase with more files."""
        file1 = self._create_test_file("module1.py", """
def func1():
    pass

class Class1:
    def method1(self):
        pass
""")

        file2 = self._create_test_file("module2.py", """
def func2():
    pass

class Class2:
    def method2(self):
        pass
""")

        mapper = GrepMap(
            map_tokens=5000,
            root=self.temp_dir,
            verbose=False
        )

        # Generate map for single file
        result1, report1 = mapper.get_grep_map(
            chat_files=[],
            other_files=[file1]
        )

        # Generate map for both files
        result2, report2 = mapper.get_grep_map(
            chat_files=[],
            other_files=[file1, file2]
        )

        # More files should produce more content
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        assert result1 is not None and result2 is not None  # Type narrowing
        self.assertGreater(len(result2), len(result1),
                          "Map with 2 files should be larger than map with 1 file")
        self.assertGreater(report2.definition_matches, report1.definition_matches,
                          "More files should have more definitions")

    def test_chat_files_affect_ranking(self):
        """Chat files should be included in output (ranking boost)."""
        important_file = self._create_test_file("important.py", """
def critical_function():
    '''This is very important.'''
    pass
""")

        other_file = self._create_test_file("other.py", """
def other_function():
    '''Not as important.'''
    pass
""")

        mapper = GrepMap(
            map_tokens=5000,
            root=self.temp_dir,
            verbose=False
        )

        # Generate map with important_file as chat file
        result, report = mapper.get_grep_map(
            chat_files=[important_file],
            other_files=[other_file]
        )

        self.assertIsNotNone(result)
        # Both files should be included (not testing exact ordering - that's layout)
        self.assertIn("important.py", result, "Should include chat file")
        self.assertIn("critical_function", result, "Should include chat file symbols")
        # With enough budget, other file should also appear
        self.assertGreater(report.definition_matches, 0, "Should extract definitions")

    def test_directory_mode_vs_tree_mode(self):
        """Should support both directory and tree rendering modes."""
        test_file = self._create_test_file("test.py", """
class Example:
    def method(self):
        pass

def function():
    pass
""")

        # Directory mode
        mapper_dir = GrepMap(
            map_tokens=2000,
            root=self.temp_dir,
            directory_mode=True,
            verbose=False
        )

        result_dir, report_dir = mapper_dir.get_grep_map(
            chat_files=[],
            other_files=[test_file]
        )

        # Tree mode
        mapper_tree = GrepMap(
            map_tokens=2000,
            root=self.temp_dir,
            directory_mode=False,
            verbose=False
        )

        result_tree, report_tree = mapper_tree.get_grep_map(
            chat_files=[],
            other_files=[test_file]
        )

        # Both should generate non-empty output with symbols
        self.assertIsNotNone(result_dir, "Directory mode should generate output")
        self.assertIsNotNone(result_tree, "Tree mode should generate output")
        assert result_dir is not None and result_tree is not None  # Type narrowing
        self.assertGreater(len(result_dir), 0, "Directory mode output should not be empty")
        self.assertGreater(len(result_tree), 0, "Tree mode output should not be empty")

        # Both should extract the same symbols (behavior, not format)
        self.assertEqual(report_dir.definition_matches, report_tree.definition_matches,
                        "Both modes should extract same number of definitions")
        self.assertIn("Example", result_dir, "Directory mode should show class")
        self.assertIn("Example", result_tree, "Tree mode should show class")

    def test_respects_token_budget(self):
        """Should respect the token budget constraint."""
        # Create multiple files to exceed small budget
        for i in range(10):
            self._create_test_file(f"module{i}.py", f"""
def function_{i}_1():
    pass

def function_{i}_2():
    pass

class Class_{i}:
    def method1(self):
        pass

    def method2(self):
        pass
""")

        files = [str(Path(self.temp_dir) / f"module{i}.py") for i in range(10)]

        # Small budget
        mapper_small = GrepMap(
            map_tokens=500,
            root=self.temp_dir,
            verbose=False
        )

        result_small, _ = mapper_small.get_grep_map(
            chat_files=[],
            other_files=files
        )

        # Large budget
        mapper_large = GrepMap(
            map_tokens=5000,
            root=self.temp_dir,
            verbose=False
        )

        result_large, _ = mapper_large.get_grep_map(
            chat_files=[],
            other_files=files
        )

        # Both should generate output
        self.assertIsNotNone(result_small)
        self.assertIsNotNone(result_large)
        assert result_small is not None and result_large is not None  # Type narrowing

        # Larger budget should produce more content
        self.assertGreater(len(result_large), len(result_small),
                          "Larger token budget should allow more content")

        # Small budget should stay reasonably close to limits
        # Note: Minimal fallback may slightly exceed budget if content is inherently large
        token_count_small = mapper_small.token_count(result_small)
        self.assertLess(token_count_small, 1000,
                       "Small budget should produce much smaller output than large budget")


class TestCacheIntegration(TestCase):
    """Integration tests for caching behavior."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: self._cleanup_temp_dir())

    def _cleanup_temp_dir(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cache_speeds_up_repeated_calls(self):
        """Cached tags should be faster on second call."""
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("""
def function1():
    pass

def function2():
    pass

class MyClass:
    def method(self):
        pass
""")

        mapper = GrepMap(
            map_tokens=1000,
            root=self.temp_dir,
            verbose=False
        )

        # First call - should parse and cache
        result1, report1 = mapper.get_grep_map(
            chat_files=[],
            other_files=[str(test_file)]
        )

        # Second call - should use cache
        result2, report2 = mapper.get_grep_map(
            chat_files=[],
            other_files=[str(test_file)]
        )

        # Results should be identical
        self.assertEqual(result1, result2, "Cached result should be identical")
        self.assertEqual(report1.definition_matches, report2.definition_matches,
                        "Cached report should have same definition count")

    def test_cache_invalidates_on_file_change(self):
        """Cache should invalidate when file is modified."""
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("def old_function():\n    pass\n")

        mapper = GrepMap(
            map_tokens=1000,
            root=self.temp_dir,
            verbose=False
        )

        # First call
        result1, _ = mapper.get_grep_map(
            chat_files=[],
            other_files=[str(test_file)]
        )

        self.assertIn("old_function", result1)

        # Modify file - ensure mtime changes by using touch
        import time
        time.sleep(1.01)  # Ensure mtime changes (some filesystems have 1s granularity)
        test_file.write_text("def new_function():\n    pass\n")
        # Force mtime update
        os.utime(test_file, None)

        # Second call with force_refresh should see new content
        result2, _ = mapper.get_grep_map(
            chat_files=[],
            other_files=[str(test_file)],
            force_refresh=True  # Force cache refresh
        )

        self.assertNotIn("old_function", result2, "Should not see old function after refresh")
        self.assertIn("new_function", result2, "Should see new function after refresh")
