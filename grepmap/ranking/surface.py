"""
API surface detection for prioritizing public interfaces over internals.

This module classifies symbols as API surface (called from outside their module)
vs internal implementation details (only called within same file). This enables
prioritizing what to show in the map - users care more about public APIs than
private helpers.

Classification strategy:
- API: >50% of callers are from different files (high "external reference ratio")
- INTERNAL: >80% of callers are from same file (mostly self-contained)
- UNKNOWN: No callers detected or ambiguous pattern

This integrates with the symbol ranking system to boost API surface visibility
while deprioritizing implementation details that don't affect external usage.
"""

import networkx as nx
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

from grepmap.core.types import Tag


# Symbol identifier: (relative_filename, symbol_name)
SymbolId = Tuple[str, str]


class SurfaceType(Enum):
    """Classification of symbol visibility/usage pattern.

    Attributes:
        API: Symbol is called primarily from other modules (public interface)
        INTERNAL: Symbol is called primarily from its own module (implementation detail)
        UNKNOWN: No caller data available or ambiguous pattern
    """
    API = "api"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass
class SurfaceInfo:
    """Surface classification result for a symbol.

    Captures both the classification and the underlying metrics that led to it,
    enabling downstream consumers to make informed decisions about display priority.

    Attributes:
        symbol_id: (rel_fname, symbol_name) identifying the symbol
        surface_type: Classification as API/INTERNAL/UNKNOWN
        external_refs: Count of references from other files
        internal_refs: Count of references from same file
    """
    symbol_id: SymbolId
    surface_type: SurfaceType
    external_refs: int
    internal_refs: int

    @property
    def total_refs(self) -> int:
        """Total reference count across all sources."""
        return self.external_refs + self.internal_refs

    @property
    def external_ratio(self) -> float:
        """Ratio of external to total references (0.0-1.0).

        Returns 0.0 if no references detected.
        """
        if self.total_refs == 0:
            return 0.0
        return self.external_refs / self.total_refs


class SurfaceDetector:
    """Detects API surface vs internal implementation symbols.

    Analyzes the symbol reference graph to identify which symbols form the public
    API (called from many different files) versus which are private implementation
    details (only called locally within their defining module).

    This enables "tree shaking" at the semantic level - when showing a map,
    prioritize public interfaces that external code depends on.
    """

    # Classification thresholds
    API_THRESHOLD = 0.5      # >50% external refs = API
    INTERNAL_THRESHOLD = 0.8  # >80% internal refs = implementation detail

    def __init__(self, verbose: bool = False):
        """Initialize SurfaceDetector.

        Args:
            verbose: Enable diagnostic logging
        """
        self.verbose = verbose

    def classify_symbols(
        self,
        symbol_graph: nx.MultiDiGraph,
        tags_by_file: Dict[str, List[Tag]]
    ) -> Dict[SymbolId, SurfaceInfo]:
        """Classify each symbol as API surface or internal implementation.

        Examines incoming edges in the symbol graph to determine where each
        symbol is called from. Symbols called primarily from other files are
        API surface; symbols called only from their own file are internal.

        Args:
            symbol_graph: Symbol reference graph from SymbolRanker
            tags_by_file: Dict mapping absolute fname to tags (for context)

        Returns:
            Dict mapping (rel_fname, symbol_name) to SurfaceInfo classification
        """
        classifications: Dict[SymbolId, SurfaceInfo] = {}

        for node in symbol_graph.nodes():
            if not isinstance(node, tuple) or len(node) != 2:
                continue  # Skip malformed nodes

            rel_fname, symbol_name = node

            # Analyze incoming edges (who calls this symbol?)
            external_refs = 0
            internal_refs = 0

            for predecessor in symbol_graph.predecessors(node):
                if not isinstance(predecessor, tuple) or len(predecessor) != 2:
                    continue

                caller_file, caller_symbol = predecessor

                if caller_file == rel_fname:
                    # Called from same file
                    internal_refs += 1
                else:
                    # Called from different file
                    external_refs += 1

            # Classify based on reference pattern
            total_refs = external_refs + internal_refs

            if total_refs == 0:
                surface_type = SurfaceType.UNKNOWN
            else:
                external_ratio = external_refs / total_refs

                if external_ratio > self.API_THRESHOLD:
                    surface_type = SurfaceType.API
                elif (1.0 - external_ratio) > self.INTERNAL_THRESHOLD:
                    # internal_ratio > 0.8
                    surface_type = SurfaceType.INTERNAL
                else:
                    # Ambiguous - some external, some internal
                    surface_type = SurfaceType.UNKNOWN

            classifications[node] = SurfaceInfo(
                symbol_id=node,
                surface_type=surface_type,
                external_refs=external_refs,
                internal_refs=internal_refs
            )

        if self.verbose:
            self._log_stats(classifications)

        return classifications

    def get_api_surface(
        self,
        classifications: Dict[SymbolId, SurfaceInfo]
    ) -> List[SymbolId]:
        """Extract list of symbols classified as API surface.

        Args:
            classifications: Result from classify_symbols()

        Returns:
            List of (rel_fname, symbol_name) tuples for API symbols
        """
        return [
            symbol_id
            for symbol_id, info in classifications.items()
            if info.surface_type == SurfaceType.API
        ]

    def get_internal_symbols(
        self,
        classifications: Dict[SymbolId, SurfaceInfo]
    ) -> List[SymbolId]:
        """Extract list of symbols classified as internal implementation.

        Args:
            classifications: Result from classify_symbols()

        Returns:
            List of (rel_fname, symbol_name) tuples for internal symbols
        """
        return [
            symbol_id
            for symbol_id, info in classifications.items()
            if info.surface_type == SurfaceType.INTERNAL
        ]

    def _log_stats(self, classifications: Dict[SymbolId, SurfaceInfo]):
        """Log classification statistics for debugging."""
        by_type = defaultdict(int)
        for info in classifications.values():
            by_type[info.surface_type] += 1

        total = len(classifications)
        print(f"Surface classification: {total} symbols total")
        print(f"  API: {by_type[SurfaceType.API]} ({100*by_type[SurfaceType.API]/total:.1f}%)")
        print(f"  INTERNAL: {by_type[SurfaceType.INTERNAL]} ({100*by_type[SurfaceType.INTERNAL]/total:.1f}%)")
        print(f"  UNKNOWN: {by_type[SurfaceType.UNKNOWN]} ({100*by_type[SurfaceType.UNKNOWN]/total:.1f}%)")

        # Show top API symbols by external reference count
        api_symbols = [
            info for info in classifications.values()
            if info.surface_type == SurfaceType.API
        ]
        if api_symbols:
            sorted_api = sorted(api_symbols, key=lambda x: x.external_refs, reverse=True)
            print("\nTop API surface symbols (by external refs):")
            for i, info in enumerate(sorted_api[:10]):
                rel_fname, symbol = info.symbol_id
                print(f"  {i+1}. {rel_fname}:{symbol} - {info.external_refs} external, {info.internal_refs} internal")
