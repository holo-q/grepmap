"""
Graph data exporter for visualization.

Converts networkx graphs and ranking data into JSON format suitable for
D3.js force-directed visualization. Captures the full topology:
nodes (symbols), edges (references), ranks, and metadata.

The exported data enables:
- Node sizing by rank (visual importance hierarchy)
- Node coloring by file/cluster/bridge status
- Edge visualization showing dependency flow
- Metadata overlays (gini, cliff, confidence)
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path

import networkx as nx


@dataclass
class NodeData:
    """A node in the visualization graph."""
    id: str  # Unique ID: "file:symbol"
    label: str  # Display name: symbol or truncated
    file: str  # Relative file path
    symbol: str  # Symbol name
    rank: float  # PageRank score
    in_degree: int  # Reference count (how many point to this)
    out_degree: int  # Usage count (how many this points to)
    is_bridge: bool  # Is this file a bridge (high betweenness)?
    is_api: bool  # Is this an API surface symbol?
    is_orphan: bool  # No incoming edges?
    cluster: str  # Directory-based cluster ID
    badges: List[str]  # Git badges: ["recent", "high-churn", "emergent"]


@dataclass
class EdgeData:
    """An edge in the visualization graph."""
    source: str  # Source node ID
    target: str  # Target node ID
    weight: float  # Edge weight (currently 1.0, could be multi-edge count)


@dataclass
class GraphMetadata:
    """Summary statistics for the graph."""
    num_nodes: int
    num_edges: int
    density: float
    gini: float  # Rank concentration
    cliff_percentile: Optional[int]  # Where rank drops sharply
    confidence: str  # "low", "medium", "high"
    orphan_count: int
    hub_symbols: List[Tuple[str, int]]  # Top hubs by in-degree
    focus_query: Optional[str]  # Current focus if any
    intent: str  # Classified intent


@dataclass
class ExportedGraph:
    """Full graph export for visualization."""
    nodes: List[NodeData]
    edges: List[EdgeData]
    metadata: GraphMetadata

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GraphExporter:
    """Exports grepmap graph data for visualization.

    Takes a networkx graph (from SymbolRanker) and ranking metadata,
    produces a structured export suitable for D3.js rendering.

    Usage:
        exporter = GraphExporter()
        exported = exporter.export(
            graph=symbol_ranker._last_graph,
            symbol_ranks=symbol_ranker._last_symbol_ranks,
            bridge_files=set(["core/facade.py"]),
            api_symbols=set([("core/facade.py", "get_grep_map")]),
            confidence_result=confidence,
            focus_query="ranking"
        )
        json_data = exported.to_json()
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def export(
        self,
        graph: nx.MultiDiGraph,
        symbol_ranks: Dict[Tuple[str, str], float],
        bridge_files: Optional[Set[str]] = None,
        api_symbols: Optional[Set[Tuple[str, str]]] = None,
        git_badges: Optional[Dict[str, List[str]]] = None,
        confidence_result: Optional[Any] = None,
        focus_query: Optional[str] = None,
        intent: str = "explore"
    ) -> ExportedGraph:
        """Export the graph to visualization format.

        Args:
            graph: NetworkX graph from SymbolRanker
            symbol_ranks: Dict mapping (file, symbol) to rank
            bridge_files: Set of file paths identified as bridges
            api_symbols: Set of (file, symbol) tuples identified as API
            git_badges: Dict mapping file path to list of badges
            confidence_result: ConfidenceResult from confidence engine
            focus_query: Current focus query string
            intent: Classified intent string

        Returns:
            ExportedGraph ready for serialization
        """
        if graph is None:
            return self._empty_graph()

        bridge_files = bridge_files or set()
        api_symbols = api_symbols or set()
        git_badges = git_badges or {}

        nodes = []
        node_id_map = {}  # (file, symbol) -> id string

        # Build nodes
        for node in graph.nodes():
            if not isinstance(node, tuple) or len(node) != 2:
                continue

            rel_fname, symbol = node
            node_id = f"{rel_fname}:{symbol}"
            node_id_map[node] = node_id

            rank = symbol_ranks.get(node, 0.0)
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)

            # Determine cluster from directory
            cluster = str(Path(rel_fname).parent) if '/' in rel_fname else "root"

            # Get badges for this file
            file_badges = git_badges.get(rel_fname, [])

            nodes.append(NodeData(
                id=node_id,
                label=self._truncate(symbol, 20),
                file=rel_fname,
                symbol=symbol,
                rank=rank,
                in_degree=in_deg,
                out_degree=out_deg,
                is_bridge=rel_fname in bridge_files,
                is_api=node in api_symbols,
                is_orphan=in_deg == 0,
                cluster=cluster,
                badges=file_badges
            ))

        # Build edges
        edges = []
        edge_counts: Dict[Tuple[str, str], int] = {}

        for u, v in graph.edges():
            if u not in node_id_map or v not in node_id_map:
                continue
            source_id = node_id_map[u]
            target_id = node_id_map[v]
            edge_key = (source_id, target_id)

            # Count multi-edges
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        for (source_id, target_id), count in edge_counts.items():
            edges.append(EdgeData(
                source=source_id,
                target=target_id,
                weight=float(count)
            ))

        # Build metadata
        metadata = self._build_metadata(
            graph, nodes, symbol_ranks, confidence_result, focus_query, intent
        )

        return ExportedGraph(nodes=nodes, edges=edges, metadata=metadata)

    def _build_metadata(
        self,
        graph: nx.MultiDiGraph,
        nodes: List[NodeData],
        symbol_ranks: Dict[Tuple[str, str], float],
        confidence_result: Optional[Any],
        focus_query: Optional[str],
        intent: str
    ) -> GraphMetadata:
        """Build summary metadata from graph."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Density
        max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
        density = num_edges / max_edges if max_edges > 0 else 0.0

        # Gini coefficient
        ranks = list(symbol_ranks.values())
        gini = self._compute_gini(ranks) if ranks else 0.0

        # Cliff detection
        cliff = self._find_cliff(ranks) if ranks else None

        # Confidence
        if confidence_result:
            confidence = getattr(confidence_result, 'level', 'unknown')
        else:
            confidence = 'unknown'

        # Orphan count
        orphan_count = sum(1 for n in nodes if n.is_orphan)

        # Hub symbols
        hub_data = sorted(
            [(n.symbol, n.in_degree) for n in nodes],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return GraphMetadata(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            gini=gini,
            cliff_percentile=cliff,
            confidence=confidence,
            orphan_count=orphan_count,
            hub_symbols=hub_data,
            focus_query=focus_query,
            intent=intent
        )

    def _compute_gini(self, values: List[float]) -> float:
        """Gini coefficient: 0=equal, 1=concentrated."""
        if not values or len(values) < 2:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        return (2 * cumsum) / (n * total) - (n + 1) / n

    def _find_cliff(self, ranks: List[float], threshold: float = 0.5) -> Optional[int]:
        """Find percentile where rank drops significantly."""
        if len(ranks) < 10:
            return None
        sorted_ranks = sorted(ranks, reverse=True)
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i - 1] > 0:
                ratio = sorted_ranks[i] / sorted_ranks[i - 1]
                if ratio < threshold:
                    return int(100 * i / len(sorted_ranks))
        return None

    def _truncate(self, s: str, max_len: int) -> str:
        """Truncate string with ellipsis."""
        if len(s) <= max_len:
            return s
        return s[:max_len - 1] + "…"

    def _empty_graph(self) -> ExportedGraph:
        """Return empty graph for error cases."""
        return ExportedGraph(
            nodes=[],
            edges=[],
            metadata=GraphMetadata(
                num_nodes=0,
                num_edges=0,
                density=0.0,
                gini=0.0,
                cliff_percentile=None,
                confidence='unknown',
                orphan_count=0,
                hub_symbols=[],
                focus_query=None,
                intent='explore'
            )
        )
