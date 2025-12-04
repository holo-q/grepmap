"""Cache management for GrepMap.

Handles persistent tag caching using diskcache, with error recovery
and version management for cache invalidation.
"""

import shutil
import sqlite3
from pathlib import Path
from typing import Optional, List, Callable, Any
from grepmap.core.types import Tag


# Cache version - bump when Tag structure changes
CACHE_VERSION = 5
TAGS_CACHE_DIR = f".grepmap.tags.cache.v{CACHE_VERSION}"
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)


class CacheManager:
    """Manages persistent tag caching with error recovery.

    Uses diskcache for file-level tag persistence, keyed by filepath with
    modification time tracking for invalidation. Falls back to in-memory
    dict on cache errors.

    Args:
        root: Repository root directory
        output_handler: Optional callback for warnings/errors
    """

    def __init__(
        self,
        root: Path,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        self.root = root
        self.output_handler = output_handler or print
        self.tags_cache: Any = {}  # Will be diskcache.Cache or dict
        self.load_tags_cache()

    def load_tags_cache(self) -> None:
        """Load the persistent tags cache from disk.

        Creates a diskcache.Cache instance at .grepmap.tags.cache.v{VERSION}.
        Falls back to in-memory dict on errors.
        """
        cache_dir = self.root / TAGS_CACHE_DIR
        try:
            import diskcache
            self.tags_cache = diskcache.Cache(str(cache_dir))
        except Exception as e:
            self.output_handler(f"Failed to load tags cache: {e}")
            self.tags_cache = {}

    def save_tags_cache(self) -> None:
        """Save the tags cache (no-op as diskcache handles persistence)."""
        pass

    def tags_cache_error(self) -> None:
        """Handle tags cache errors by recreating the cache.

        Removes corrupted cache directory and re-initializes. Falls back
        to in-memory dict if recreation fails.
        """
        try:
            cache_dir = self.root / TAGS_CACHE_DIR
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            self.load_tags_cache()
        except Exception:
            self.output_handler(
                "Failed to recreate tags cache, using in-memory cache"
            )
            self.tags_cache = {}

    def get_cached_tags(
        self,
        fname: str,
        file_mtime: float
    ) -> Optional[List[Tag]]:
        """Retrieve cached tags if file hasn't changed.

        Args:
            fname: Absolute file path
            file_mtime: Current file modification time

        Returns:
            Cached tag list if valid, None if cache miss or stale
        """
        try:
            # Handle both diskcache Cache and in-memory dict
            if isinstance(self.tags_cache, dict):
                cached_entry = self.tags_cache.get(fname)
            else:
                cached_entry = self.tags_cache.get(fname)

            if cached_entry and cached_entry.get("mtime") == file_mtime:
                return cached_entry["data"]
        except SQLITE_ERRORS:
            self.tags_cache_error()

        return None

    def set_cached_tags(
        self,
        fname: str,
        file_mtime: float,
        tags: List[Tag]
    ) -> None:
        """Store tags in cache with modification time.

        Args:
            fname: Absolute file path
            file_mtime: File modification time for invalidation
            tags: Parsed tag list to cache
        """
        try:
            self.tags_cache[fname] = {"mtime": file_mtime, "data": tags}
        except SQLITE_ERRORS:
            self.tags_cache_error()

    def clear_cache(self) -> None:
        """Clear all cached tags and remove cache directory."""
        try:
            cache_dir = self.root / TAGS_CACHE_DIR
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            self.tags_cache = {}
        except Exception as e:
            self.output_handler(f"Failed to clear cache: {e}")
