"""
Backward compatibility wrapper for GrepMap.

This module maintains the original import path (grepmap_class.GrepMap) for
backward compatibility with existing code. The actual implementation has been
refactored into a modular architecture under the grepmap/ package.

New code should import from grepmap directly:
    from grepmap import GrepMap

This wrapper will be maintained for compatibility but may be deprecated in
future versions.
"""

# Re-export the refactored GrepMap and FileReport from the new modular structure
from grepmap.facade import GrepMap
from grepmap.core.types import FileReport

# Re-export constants for backward compatibility
from grepmap.cache.manager import CACHE_VERSION, TAGS_CACHE_DIR, SQLITE_ERRORS

__all__ = ['GrepMap', 'FileReport', 'CACHE_VERSION', 'TAGS_CACHE_DIR', 'SQLITE_ERRORS']


# Legacy GrepMap class is now just an alias to the refactored version
# All functionality has been preserved, just reorganized into focused modules:
# - grepmap/cache: Persistent tag caching
# - grepmap/extraction: Tree-sitter parsing
# - grepmap/ranking: PageRank and boost calculation
# - grepmap/rendering: Output formatting
# - grepmap/facade: Main orchestrator (GrepMap class)
