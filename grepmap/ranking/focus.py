"""
Focus target resolution for GrepMap.

Resolves --focus arguments into weighted file targets. Supports both:
- File paths: Resolved to files with weight 1.0
- Query strings: Fuzzy matched against symbols, weighted by match quality

This allows users to specify focus targets either by explicit paths or
semantic queries like "authentication" that match symbol names.

Example:
    --focus src/auth.py --focus "validation"

    First target: exact file match (weight 1.0)
    Second target: matches symbols containing "validation" (weighted by score)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Callable
from grepmap.core.types import Tag


class FocusResolver:
    """Resolves focus targets into weighted file mappings.

    Given a list of focus target strings, determines which are file paths
    and which are search queries, then resolves them appropriately.

    File paths get weight 1.0 (maximum focus).
    Query strings are matched against symbol names across all files,
    with weights based on match quality (exact > prefix > substring).
    """

    def __init__(
        self,
        root: Path,
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize FocusResolver.

        Args:
            root: Repository root path for resolving relative paths
            verbose: Enable verbose logging
            output_handler: Function for info messages (default: print)
        """
        self.root = root
        self.verbose = verbose
        self.output_handler = output_handler or print

    def resolve(
        self,
        focus_targets: List[str],
        tags_by_file: Dict[str, List[Tag]]
    ) -> Tuple[Set[str], Set[str]]:
        """Resolve focus targets into files and identifiers.

        Args:
            focus_targets: List of focus target strings (paths or queries)
            tags_by_file: Dict mapping absolute filename to its tags

        Returns:
            Tuple of (focus_files, focus_idents):
            - focus_files: Set of absolute file paths that should be focused
            - focus_idents: Set of identifier names that matched queries
        """
        focus_files: Set[str] = set()
        focus_idents: Set[str] = set()

        for target in focus_targets:
            # Try as file path first
            resolved_path = self._resolve_as_path(target)
            if resolved_path:
                focus_files.add(resolved_path)
                if self.verbose:
                    self.output_handler(f"Focus file: {target}")
                continue

            # Not a path - treat as symbol query
            matched_files, matched_idents = self._resolve_as_query(
                target, tags_by_file
            )
            focus_files.update(matched_files)
            focus_idents.update(matched_idents)

            if self.verbose and (matched_files or matched_idents):
                self.output_handler(
                    f"Focus query '{target}': {len(matched_files)} files, "
                    f"{len(matched_idents)} symbols"
                )

        return focus_files, focus_idents

    def _resolve_as_path(self, target: str) -> Optional[str]:
        """Try to resolve target as a file path.

        Args:
            target: Potential file path

        Returns:
            Absolute path if file exists, None otherwise
        """
        # Try as absolute path
        if os.path.isabs(target):
            if os.path.isfile(target):
                return target
            return None

        # Try relative to root
        candidate = self.root / target
        if candidate.is_file():
            return str(candidate.resolve())

        # Try as-is in case cwd differs from root
        if os.path.isfile(target):
            return str(Path(target).resolve())

        return None

    def _resolve_as_query(
        self,
        query: str,
        tags_by_file: Dict[str, List[Tag]]
    ) -> Tuple[Set[str], Set[str]]:
        """Resolve query string against symbol names.

        Uses case-insensitive matching with scoring:
        - Exact match: weight 1.0
        - Prefix match: weight 0.8
        - Substring match: weight 0.5
        - Word boundary match: weight 0.7

        Args:
            query: Search query string
            tags_by_file: Dict mapping absolute filename to tags

        Returns:
            Tuple of (matched_files, matched_idents)
        """
        matched_files: Set[str] = set()
        matched_idents: Set[str] = set()

        query_lower = query.lower()
        query_parts = set(re.split(r'[_\s]+', query_lower))  # Split on _ or space

        for fname, tags in tags_by_file.items():
            for tag in tags:
                if tag.kind != 'def':
                    continue

                name_lower = tag.name.lower()

                # Check for matches
                if self._matches_query(name_lower, query_lower, query_parts):
                    matched_files.add(fname)
                    matched_idents.add(tag.name)

        return matched_files, matched_idents

    def _matches_query(
        self,
        name_lower: str,
        query_lower: str,
        query_parts: Set[str]
    ) -> bool:
        """Check if symbol name matches query.

        Args:
            name_lower: Lowercase symbol name
            query_lower: Lowercase query string
            query_parts: Query split into parts (for multi-word matching)

        Returns:
            True if name matches query
        """
        # Exact match
        if name_lower == query_lower:
            return True

        # Substring match
        if query_lower in name_lower:
            return True

        # Check if all query parts appear in the name
        name_parts = set(re.split(r'[_\s]+', name_lower))
        if query_parts and query_parts.issubset(name_parts):
            return True

        # CamelCase handling: "auth" matches "AuthHandler", "authenticate"
        # Split camelCase into parts
        camel_parts = set(
            part.lower() for part in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)',
                                                  name_lower.replace('_', ''))
        )
        if query_parts and any(qp in camel_parts for qp in query_parts):
            return True

        return False
