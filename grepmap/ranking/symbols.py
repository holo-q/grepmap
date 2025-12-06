"""
Symbol-level PageRank for fine-grained importance ranking.

This module provides symbol-level (function/class) PageRank instead of file-level,
enabling "tree shaking" - if utils.py has 50 functions but only 1 is called,
only that 1 function ranks high.

The graph structure:
- Nodes: (rel_fname, symbol_name) tuples identifying each definition
- Edges: Reference relationships (symbol A calls/uses symbol B)

This gives per-symbol importance scores which can be aggregated to file-level
for ordering, while preserving fine-grained selection within files.
"""

import networkx as nx
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Callable

from grepmap.core.types import Tag
from grepmap.core.config import (
    PAGERANK_ALPHA,
    DEPTH_WEIGHT_ROOT, DEPTH_WEIGHT_MODERATE, DEPTH_WEIGHT_DEEP, DEPTH_WEIGHT_VENDOR,
    DEPTH_THRESHOLD_SHALLOW, DEPTH_THRESHOLD_MODERATE,
    PAGERANK_CHAT_MULTIPLIER,
    VENDOR_PATTERNS
)


# Symbol identifier: (relative_filename, symbol_name)
SymbolId = Tuple[str, str]

# Special symbol name for top-level (module) scope
MODULE_SCOPE = "<module>"


class SymbolRanker:
    """Symbol-level PageRank calculator.

    Builds a graph where nodes are individual symbols (functions, classes, methods)
    and edges represent usage relationships. This enables fine-grained ranking
    that can identify the ONE important function in a 50-function file.

    The graph captures:
    - Function calls: function A calls function B
    - Attribute access: code in scope A references symbol B
    - Class usage: class A inherits from or uses class B

    Symbols are identified by (file, name) tuples to handle same-name symbols
    in different files correctly.
    """

    def __init__(
        self,
        get_rel_fname: Callable[[str], str],
        verbose: bool = False,
        output_handlers: Optional[Dict[str, Callable]] = None
    ):
        """Initialize SymbolRanker.

        Args:
            get_rel_fname: Function to convert absolute path to relative path
            verbose: Enable verbose logging
            output_handlers: Optional dict of output handler functions
        """
        self.get_rel_fname = get_rel_fname
        self.verbose = verbose
        self.output_handlers = output_handlers or {'info': print, 'warning': print}

    def compute_ranks(
        self,
        all_fnames: List[str],
        tags_by_file: Dict[str, List[Tag]],
        chat_fnames: List[str]
    ) -> Tuple[Dict[SymbolId, float], Dict[str, float]]:
        """Compute PageRank scores at symbol level.

        Builds a symbol graph and runs PageRank to get per-symbol importance.
        Also aggregates to file level for backwards compatibility.

        Args:
            all_fnames: List of all absolute file paths
            tags_by_file: Dict mapping absolute fname to its list of tags
            chat_fnames: List of chat file absolute paths (for boost)

        Returns:
            Tuple of:
            - symbol_ranks: Dict mapping (rel_fname, symbol) to PageRank score
            - file_ranks: Dict mapping rel_fname to aggregated score (max of symbols)
        """
        # Build definition index: symbol_name -> list of (rel_fname, symbol_name)
        # This handles same-name symbols in different files
        definitions: Dict[str, List[SymbolId]] = defaultdict(list)

        # Also track all symbols (definitions) for node creation
        all_symbols: Set[SymbolId] = set()

        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            tags = tags_by_file.get(fname, [])

            for tag in tags:
                if tag.kind == "def":
                    symbol_id = (rel_fname, tag.name)
                    definitions[tag.name].append(symbol_id)
                    all_symbols.add(symbol_id)

        if not all_symbols:
            return {}, {}

        # Build symbol graph
        G = nx.MultiDiGraph()

        # Add all symbols as nodes
        for symbol_id in all_symbols:
            G.add_node(symbol_id)

        # Add edges based on references
        # A reference inside scope S to symbol X creates edge: S -> X
        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            tags = tags_by_file.get(fname, [])

            for tag in tags:
                if tag.kind != "ref":
                    continue

                # Determine source symbol (the scope containing this reference)
                if tag.parent_name:
                    # Reference inside a function/class
                    source = (rel_fname, tag.parent_name)
                else:
                    # Top-level reference (module scope)
                    source = (rel_fname, MODULE_SCOPE)

                # Only add edges from known source symbols
                # (source might be a method inside a class we don't have as node)
                if source not in all_symbols:
                    # Try to find if source is a valid symbol we know about
                    # This handles cases like method references where parent is class
                    continue

                # Find all definitions of the referenced symbol
                target_symbols = definitions.get(tag.name, [])
                for target in target_symbols:
                    if source != target:
                        G.add_edge(source, target, name=tag.name)

        if self.verbose:
            self.output_handlers['info'](
                f"Symbol graph: {G.number_of_nodes()} symbols, {G.number_of_edges()} edges"
            )

        # Build personalization based on file depth and chat status
        personalization = self._build_personalization(G, chat_fnames)

        # Run PageRank
        try:
            symbol_ranks = nx.pagerank(
                G,
                personalization=personalization,
                alpha=PAGERANK_ALPHA
            )
        except Exception as e:
            if self.verbose:
                self.output_handlers['warning'](
                    f"Symbol PageRank failed: {e}, using uniform ranking"
                )
            symbol_ranks = {node: 1.0 / len(G.nodes()) for node in G.nodes()}

        # Aggregate to file level (max symbol rank per file)
        file_ranks: Dict[str, float] = defaultdict(float)
        for (rel_fname, symbol), rank in symbol_ranks.items():
            file_ranks[rel_fname] = max(file_ranks[rel_fname], rank)

        if self.verbose:
            self._log_stats(symbol_ranks, file_ranks, G)

        # Collect diagnostic data
        self._last_graph = G
        self._last_symbol_ranks = symbol_ranks

        return dict(symbol_ranks), dict(file_ranks)

    def get_diagnostic_data(self) -> dict:
        """Get diagnostic data from last compute_ranks call."""
        if not hasattr(self, '_last_graph') or self._last_graph is None:
            return {}

        G = self._last_graph

        # Hub symbols (highest in-degree)
        in_degrees = [(node, G.in_degree(node)) for node in G.nodes()]
        sorted_hubs = sorted(in_degrees, key=lambda x: x[1], reverse=True)
        hub_symbols = [(f"{node[1]}", deg) for node, deg in sorted_hubs[:10] if deg > 0]

        # Orphan count (no incoming edges)
        orphan_count = sum(1 for node in G.nodes() if G.in_degree(node) == 0)

        return {
            'num_symbols': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'hub_symbols': hub_symbols,
            'orphan_count': orphan_count
        }

    def _build_personalization(
        self,
        G: nx.MultiDiGraph,
        chat_fnames: List[str]
    ) -> Dict[SymbolId, float]:
        """Build personalization weights for symbol-level PageRank.

        Applies depth-based and chat-file weighting at symbol level.

        Args:
            G: The symbol graph
            chat_fnames: List of chat file absolute paths

        Returns:
            Dict mapping symbol_id to personalization weight
        """
        personalization = {}
        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)

        for symbol_id in G.nodes():
            rel_fname, symbol_name = symbol_id
            depth = rel_fname.count('/')

            # Check for vendor patterns
            is_vendor = any(pattern in rel_fname for pattern in VENDOR_PATTERNS)

            if is_vendor:
                weight = DEPTH_WEIGHT_VENDOR
            elif depth <= DEPTH_THRESHOLD_SHALLOW:
                weight = DEPTH_WEIGHT_ROOT
            elif depth <= DEPTH_THRESHOLD_MODERATE:
                weight = DEPTH_WEIGHT_MODERATE
            else:
                weight = DEPTH_WEIGHT_DEEP

            # Boost symbols in chat files
            if rel_fname in chat_rel_fnames:
                weight *= PAGERANK_CHAT_MULTIPLIER

            personalization[symbol_id] = weight

        return personalization

    def _log_stats(
        self,
        symbol_ranks: Dict[SymbolId, float],
        file_ranks: Dict[str, float],
        G: nx.MultiDiGraph
    ):
        """Log symbol ranking statistics for debugging."""
        if not symbol_ranks:
            return

        rank_values = list(symbol_ranks.values())
        self.output_handlers['info'](
            f"Symbol ranks - min: {min(rank_values):.6f}, "
            f"max: {max(rank_values):.6f}, "
            f"avg: {sum(rank_values)/len(rank_values):.6f}"
        )

        # Show top symbols
        sorted_symbols = sorted(symbol_ranks.items(), key=lambda x: x[1], reverse=True)
        self.output_handlers['info']("Top 10 symbols by PageRank:")
        for i, ((rel_fname, symbol), rank) in enumerate(sorted_symbols[:10]):
            in_edges = G.in_degree(((rel_fname, symbol)))
            self.output_handlers['info'](
                f"  {i+1}. {rel_fname}:{symbol} rank={rank:.6f} refs={in_edges}"
            )


def get_symbol_ranks_for_file(
    symbol_ranks: Dict[SymbolId, float],
    rel_fname: str
) -> Dict[str, float]:
    """Extract symbol ranks for a specific file.

    Useful for determining which symbols within a file should be shown.

    Args:
        symbol_ranks: Full symbol rank dict from SymbolRanker
        rel_fname: Relative filename to filter for

    Returns:
        Dict mapping symbol_name to rank for symbols in this file
    """
    return {
        symbol: rank
        for (fname, symbol), rank in symbol_ranks.items()
        if fname == rel_fname
    }
