"""GrepMap - AI-powered codebase navigator using PageRank and tree-sitter.

This package provides modular, composable components for generating
intelligent repository maps that highlight important code structures.

Main components:
- core: Type definitions and configuration
- cache: Persistent tag caching with diskcache
- extraction: Tree-sitter-based code parsing
- ranking: PageRank-based importance calculation  
- rendering: Multiple output formats (tree view, directory overview)
- facade: Main orchestrator class (GrepMap)

Example:
    from grepmap import GrepMap
    
    mapper = GrepMap(map_tokens=2048, verbose=True)
    map_output, report = mapper.get_grep_map(
        chat_files=[],
        other_files=['src/main.py', 'src/utils.py']
    )
"""

from grepmap.facade import GrepMap
from grepmap.core.types import (
    Tag,
    RankedTag,
    DetailLevel,
    SignatureInfo,
    FieldInfo,
    FileReport
)

__version__ = "0.1.0"

__all__ = [
    'GrepMap',
    'Tag',
    'RankedTag',
    'DetailLevel',
    'SignatureInfo',
    'FieldInfo',
    'FileReport',
    'main',
]


def main():
    """CLI entry point - delegates to standalone grepmap.py module."""
    import sys
    import os
    # Import from the parent directory's grepmap.py (not this package)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("grepmap_cli", os.path.join(parent_dir, "grepmap.py"))
    grepmap_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grepmap_cli)
    grepmap_cli.main()
