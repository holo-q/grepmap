"""
PageRank algorithm for file importance ranking.

This module implements the PageRank calculation for ranking files by their
interconnectedness within the repository. It builds a directed graph where:
- Nodes represent files
- Edges represent references (file A references a symbol defined in file B)
- Edge weights are based on reference counts

The PageRank incorporates depth-aware personalization:
- Root/shallow files get higher base weight
- Vendor/third-party code is heavily penalized
- Chat files receive additional boost

This depth-aware approach ensures important root files rank high while still
allowing deeply nested files to rank well if they're heavily interconnected.
"""

import networkx as nx
from collections import defaultdict
from typing import List, Dict, Optional, Callable

from grepmap.core.types import Tag
from grepmap.core.config import (
    PAGERANK_ALPHA,
    DEPTH_WEIGHT_ROOT, DEPTH_WEIGHT_MODERATE, DEPTH_WEIGHT_DEEP, DEPTH_WEIGHT_VENDOR,
    DEPTH_THRESHOLD_SHALLOW, DEPTH_THRESHOLD_MODERATE,
    PAGERANK_CHAT_MULTIPLIER,
    VENDOR_PATTERNS
)


class PageRanker:
    """PageRank-based file importance calculator.

    Builds a graph of file references and computes importance scores using
    the PageRank algorithm with depth-aware personalization.

    The graph structure captures how files reference each other:
    - Definitions create "def" tags in files
    - References create edges from referencing file to defining file
    - Multiple references to the same symbol strengthen the edge

    Personalization is depth-aware to bias toward root files while allowing
    graph structure to override for truly important deep files.
    """

    def __init__(
        self,
        get_rel_fname: Callable[[str], str],
        verbose: bool = False,
        output_handlers: Optional[Dict[str, Callable]] = None
    ):
        """Initialize PageRanker.

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
    ) -> Dict[str, float]:
        """Compute PageRank scores for all files.

        The algorithm:
        1. Build graph with files as nodes
        2. Add edges based on references (ref file -> def file)
        3. Compute depth-aware personalization weights
        4. Run PageRank with personalization
        5. Return rank scores as dict[rel_fname -> score]

        Args:
            all_fnames: List of all absolute file paths to rank
            tags_by_file: Dict mapping absolute fname to its list of tags
            chat_fnames: List of chat file absolute paths (for boost)

        Returns:
            Dict mapping relative filename to PageRank score (0.0-1.0)
        """
        # Collect definitions and references across all files
        # defines[symbol_name] = set of files that define it
        # references[symbol_name] = set of files that reference it
        defines = defaultdict(set)
        references = defaultdict(set)

        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            tags = tags_by_file.get(fname, [])

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)

        # Build graph: files are nodes, references create edges
        G = nx.MultiDiGraph()

        # Add all files as nodes
        for fname in all_fnames:
            rel_fname = self.get_rel_fname(fname)
            G.add_node(rel_fname)

        # Add edges: ref_file -> def_file for each symbol reference
        # MultiDiGraph allows multiple edges for the same file pair
        for name, ref_fnames in references.items():
            def_fnames = defines.get(name, set())
            for ref_fname in ref_fnames:
                for def_fname in def_fnames:
                    if ref_fname != def_fname:
                        G.add_edge(ref_fname, def_fname, name=name)

        if not G.nodes():
            return {}

        if self.verbose:
            self.output_handlers['info'](
                f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )

        # Build depth-based personalization for PageRank
        # Root files get higher bias, but truly important deep files can still rank high
        # if they're heavily interconnected with root files (graph structure wins)
        depth_personalization = self._build_depth_personalization(
            G, chat_fnames
        )

        # Run PageRank with depth-aware personalization
        try:
            ranks = nx.pagerank(
                G,
                personalization=depth_personalization,
                alpha=PAGERANK_ALPHA
            )

            if self.verbose and ranks:
                self._log_pagerank_stats(ranks, G)

            return ranks

        except Exception as e:
            # Fallback to uniform ranking if PageRank fails
            if self.verbose:
                self.output_handlers['warning'](
                    f"PageRank failed: {e}, using uniform ranking"
                )
            return {node: 1.0 for node in G.nodes()}

    def _build_depth_personalization(
        self,
        G: nx.MultiDiGraph,
        chat_fnames: List[str]
    ) -> Dict[str, float]:
        """Build depth-aware personalization weights for PageRank.

        Personalization biases the random walk toward certain nodes using
        configurable weights from grepmap.core.config:
        - Root/shallow files: DEPTH_WEIGHT_ROOT
        - Moderate depth: DEPTH_WEIGHT_MODERATE
        - Deep files: DEPTH_WEIGHT_DEEP
        - Vendor/third-party: DEPTH_WEIGHT_VENDOR
        - Chat files: multiply by PAGERANK_CHAT_MULTIPLIER

        Args:
            G: The networkx graph
            chat_fnames: List of chat file absolute paths

        Returns:
            Dict mapping node name to personalization weight
        """
        depth_personalization = {}
        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)

        for node in G.nodes():
            depth = node.count('/')

            # Check for vendor/third-party patterns
            is_vendor = any(pattern in node for pattern in VENDOR_PATTERNS)

            if is_vendor:
                depth_personalization[node] = DEPTH_WEIGHT_VENDOR
            elif depth <= DEPTH_THRESHOLD_SHALLOW:
                depth_personalization[node] = DEPTH_WEIGHT_ROOT
            elif depth <= DEPTH_THRESHOLD_MODERATE:
                depth_personalization[node] = DEPTH_WEIGHT_MODERATE
            else:
                depth_personalization[node] = DEPTH_WEIGHT_DEEP

            # Apply chat file boost by multiplication
            if node in chat_rel_fnames:
                depth_personalization[node] *= PAGERANK_CHAT_MULTIPLIER

        return depth_personalization

    def _log_pagerank_stats(self, ranks: Dict[str, float], G: nx.MultiDiGraph):
        """Log PageRank statistics for debugging.

        Args:
            ranks: Dict of computed PageRank scores
            G: The networkx graph
        """
        rank_values = list(ranks.values())
        self.output_handlers['info'](
            f"PageRank scores (depth-aware) - min: {min(rank_values):.6f}, "
            f"max: {max(rank_values):.6f}, "
            f"avg: {sum(rank_values)/len(rank_values):.6f}"
        )

        # Show top files and their referrers for debugging
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        self.output_handlers['info']("Top 5 files by PageRank:")
        for i, (node, rank) in enumerate(sorted_ranks[:5]):
            # Count incoming edges (references TO this file)
            in_edges = list(G.in_edges(node))
            referrers = set(edge[0] for edge in in_edges)
            self.output_handlers['info'](
                f"  {i+1}. {node}: rank={rank:.6f}, "
                f"referenced_by={len(referrers)} files, "
                f"in_edges={len(in_edges)}"
            )
