"""
Graph visualization module for grepmap.

Provides web-based interactive graph exploration for evaluating
ranking topology, cartographic patterns, and rhizome structures.

Usage:
    python -m grepmap.visualization [root] [-p PORT] [--no-open] [-v]

    Or via main CLI:
    grepmap --viz [root]

Features:
- Force-directed D3.js graph of symbol relationships
- Real-time focus query cycling to observe rank topology shifts
- Pluggable heuristic tuning: enable/disable boosts, filters, weights
- Preset configurations: default, minimal, debug, refactor, explore
- Visual encoding: bridges (diamond), API (ring), orphans (gray)
- Live stats: gini, cliff, confidence, hub symbols
"""

from .exporter import GraphExporter, ExportedGraph
from .server import VisualizerServer
from .tuner import (
    HeuristicRegistry, TunerState, RankingPreset,
    Heuristic, HeuristicParam, HeuristicType,
    apply_tuner_state_to_config
)
from .cli import main as run_visualizer

__all__ = [
    'GraphExporter', 'ExportedGraph',
    'VisualizerServer',
    'HeuristicRegistry', 'TunerState', 'RankingPreset',
    'Heuristic', 'HeuristicParam', 'HeuristicType',
    'apply_tuner_state_to_config',
    'run_visualizer'
]
