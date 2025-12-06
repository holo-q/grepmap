"""
Bridge detection using betweenness centrality.

This module identifies "load-bearing" files in the codebase - files that connect
different parts of the architecture and whose removal would fragment the dependency
graph. These are distinct from hubs (high degree) in that they measure structural
importance rather than sheer interconnectedness.

Betweenness centrality measures how often a node appears on shortest paths between
other nodes. High betweenness indicates a file is a critical connector or bridge.

Design rationale:
- Hubs (high degree) = files with many connections
- Bridges (high betweenness) = files that connect different clusters
- Bridges are often the most architecturally important files - the "load-bearing walls"
  that define system boundaries and module interfaces

For large graphs we use k-sampling approximation to avoid O(n³) exact computation.
"""

import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class BridgeInfo:
    """Information about a bridge file.

    A bridge file has high betweenness centrality - it sits on many shortest
    paths between other files, connecting different parts of the codebase.
    """
    rel_fname: str
    betweenness: float
    connects: Tuple[str, str]  # Two clusters it bridges (approximated as top neighbors)


class BridgeDetector:
    """Detects load-bearing files using betweenness centrality.

    Betweenness centrality identifies files that are critical connectors in the
    dependency graph. These files often represent architectural boundaries,
    abstraction layers, or key interfaces between subsystems.

    The algorithm:
    1. Compute betweenness centrality (with k-sampling for large graphs)
    2. Rank files by betweenness score
    3. For each bridge, identify the two highest-degree neighbors as proxy
       for "what clusters does this bridge connect"

    Implementation note: We use approximate betweenness for graphs with >100 nodes
    (k=100 sampling) to keep computation tractable. This trades accuracy for speed
    but still identifies the major bridges reliably.
    """

    def __init__(self, verbose: bool = False):
        """Initialize BridgeDetector.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

        # Cached diagnostic data from last detection
        self._graph_size = 0
        self._betweenness_scores: Dict[str, float] = {}
        self._computation_method = ""  # "exact" or "approximate"

    def detect_bridges(
        self,
        file_graph: nx.MultiDiGraph,
        top_n: int = 10
    ) -> List[BridgeInfo]:
        """Find top N bridge files by betweenness centrality.

        Uses approximate betweenness for large graphs (k-sampling) to avoid
        expensive O(n³) exact computation. For graphs with ≤100 nodes, uses
        exact betweenness.

        Args:
            file_graph: NetworkX graph where nodes are files, edges are references
            top_n: Number of top bridges to return

        Returns:
            List of BridgeInfo, sorted by betweenness (highest first)
        """
        if not file_graph.nodes():
            return []

        self._graph_size = len(file_graph)

        # Compute betweenness centrality
        # Use approximate for large graphs (k-sampling) to avoid O(n³) cost
        k_sample = min(100, len(file_graph))
        use_exact = len(file_graph) <= 100

        if use_exact:
            # Exact computation for small graphs
            betweenness = nx.betweenness_centrality(file_graph)
            self._computation_method = "exact"
        else:
            # Approximate using k random nodes as sources
            betweenness = nx.betweenness_centrality(file_graph, k=k_sample)
            self._computation_method = f"approximate (k={k_sample})"

        self._betweenness_scores = betweenness

        # Sort by betweenness score
        sorted_nodes = sorted(
            betweenness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build BridgeInfo for top N
        bridges = []
        for rel_fname, score in sorted_nodes[:top_n]:
            # Find what this bridge connects by looking at its neighbors
            # Proxy: the two highest-degree neighbors represent the clusters
            connects = self._find_connected_clusters(file_graph, rel_fname)

            bridges.append(BridgeInfo(
                rel_fname=rel_fname,
                betweenness=score,
                connects=connects
            ))

        if self.verbose:
            self._log_bridge_stats(bridges)

        return bridges

    def get_diagnostic_data(self) -> dict:
        """Return bridge stats for diagnostics.

        Useful for debugging and understanding the bridge detection results.

        Returns:
            Dict with keys:
                - graph_size: Number of nodes in the graph
                - computation_method: "exact" or "approximate (k=N)"
                - max_betweenness: Highest betweenness score
                - avg_betweenness: Average betweenness across all nodes
                - zero_betweenness_count: Number of nodes with zero betweenness
        """
        if not self._betweenness_scores:
            return {
                'graph_size': 0,
                'computation_method': 'none',
                'max_betweenness': 0.0,
                'avg_betweenness': 0.0,
                'zero_betweenness_count': 0
            }

        scores = list(self._betweenness_scores.values())
        return {
            'graph_size': self._graph_size,
            'computation_method': self._computation_method,
            'max_betweenness': max(scores) if scores else 0.0,
            'avg_betweenness': sum(scores) / len(scores) if scores else 0.0,
            'zero_betweenness_count': sum(1 for s in scores if s == 0.0)
        }

    def _find_connected_clusters(
        self,
        G: nx.MultiDiGraph,
        node: str
    ) -> Tuple[str, str]:
        """Find the two clusters this bridge connects.

        Approximation: we use the two highest-degree neighbors as proxies
        for the clusters on either side of the bridge. This is cheap to
        compute and gives a reasonable intuition.

        In a perfect world we'd do community detection and find which two
        communities this bridge connects, but that's expensive. This proxy
        works well enough for reporting purposes.

        Args:
            G: The file graph
            node: The bridge node to analyze

        Returns:
            Tuple of (neighbor1, neighbor2) representing the two sides
            Returns ("", "") if node has fewer than 2 neighbors
        """
        # Get all neighbors (both incoming and outgoing edges)
        predecessors = set(G.predecessors(node))
        successors = set(G.successors(node))
        all_neighbors = predecessors | successors

        if len(all_neighbors) < 2:
            return ("", "")

        # Sort neighbors by degree (total connections)
        # The highest-degree neighbors likely represent different clusters
        neighbors_by_degree = sorted(
            all_neighbors,
            key=lambda n: G.degree(n),
            reverse=True
        )

        return (neighbors_by_degree[0], neighbors_by_degree[1])

    def _log_bridge_stats(self, bridges: List[BridgeInfo]):
        """Log bridge detection statistics for debugging.

        Args:
            bridges: List of detected bridges
        """
        if not bridges:
            print("No bridges detected")
            return

        print(f"\nBridge detection ({self._computation_method}):")
        print(f"  Graph size: {self._graph_size} nodes")
        print(f"  Top {len(bridges)} bridges:")

        for i, bridge in enumerate(bridges[:5], 1):
            cluster1, cluster2 = bridge.connects
            connects_str = f"{cluster1} <-> {cluster2}" if cluster1 else "isolated"
            print(f"    {i}. {bridge.rel_fname}")
            print(f"       betweenness: {bridge.betweenness:.4f}, connects: {connects_str}")
