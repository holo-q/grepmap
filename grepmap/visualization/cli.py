"""
CLI entry point for the graph visualizer.

Launches the visualization server and provides a graph provider
that runs grepmap with the current tuner state.

Usage:
    python -m grepmap.visualization [options]
    # or via the main CLI:
    grepmap --viz [options]
"""

from pathlib import Path
from typing import Optional, List, Set

from .exporter import GraphExporter, ExportedGraph
from .server import VisualizerServer
from .tuner import TunerState, apply_tuner_state_to_config


def create_graph_provider(
    root: Path,
    other_files: List[str],
    verbose: bool = False
):
    """Create a graph provider function for the visualization server.

    The provider runs grepmap with the given tuner state and returns
    an ExportedGraph suitable for D3.js rendering.

    Args:
        root: Repository root directory
        other_files: List of files to include in analysis
        verbose: Enable verbose logging

    Returns:
        A callable that takes focus_query and tuner_state, returns ExportedGraph
    """
    # Delay import to avoid circular dependency
    from grepmap.facade import GrepMap

    exporter = GraphExporter(verbose=verbose)

    def provider(
        focus_query: Optional[str] = None,
        tuner_state: Optional[TunerState] = None
    ) -> ExportedGraph:
        """Generate graph data with current tuner state."""
        tuner_state = tuner_state or TunerState()

        # Convert tuner state to config overrides
        config_overrides = apply_tuner_state_to_config(tuner_state)

        # Build GrepMap with overridden config
        gm = GrepMap(
            root=str(root),
            verbose=verbose,
            directory_mode=True,
            symbol_rank=config_overrides.get("symbol_rank", True),
            git_weight=config_overrides.get("git_weight", False),
            diagnose=False,
            color=False  # No color codes in JSON
        )

        # Build focus targets from query
        focus_targets = []
        if focus_query:
            focus_targets = [focus_query]

        # Run ranking pipeline
        ranked_tags, file_report, focus_rel_fnames = gm.get_ranked_tags(
            focus_targets=focus_targets,
            other_fnames=other_files,
            mentioned_fnames=set(),
            mentioned_idents=set()
        )

        # Extract graph and ranking data
        graph = None
        symbol_ranks = {}
        if hasattr(gm.symbol_ranker, '_last_graph'):
            graph = gm.symbol_ranker._last_graph
        if hasattr(gm.symbol_ranker, '_last_symbol_ranks'):
            symbol_ranks = gm.symbol_ranker._last_symbol_ranks

        # Get bridge files
        bridge_files: Set[str] = set()
        if graph:
            file_graph = gm._build_file_graph_from_symbol_graph(graph)
            bridges = gm.bridge_detector.detect_bridges(file_graph, top_n=10)
            bridge_files = {b.rel_fname for b in bridges if b.betweenness > 0}

        # Get API surface
        api_symbols: Set[tuple] = set()
        if graph:
            classifications = gm.surface_detector.classify_symbols(graph, {})
            api_symbols = {
                info.symbol_id for info in classifications.values()
                if info.surface_type.value == 'api'
            }

        # Get git badges
        rel_fnames = list({rt.tag.rel_fname for rt in ranked_tags})
        git_badges = gm.git_weight_calculator.compute_badges(rel_fnames)

        # Get confidence
        ranks = [rt.rank for rt in ranked_tags]
        graph_data = gm.symbol_ranker.get_diagnostic_data() if graph else {}
        confidence = gm.confidence_engine.analyze(ranks, graph_data)

        # Get intent
        intent = gm.intent_classifier.classify(focus_targets)

        # Export to visualization format
        if graph is None:
            return exporter._empty_graph()

        return exporter.export(
            graph=graph,
            symbol_ranks=symbol_ranks,
            bridge_files=bridge_files,
            api_symbols=api_symbols,
            git_badges=git_badges,
            confidence_result=confidence,
            focus_query=focus_query,
            intent=intent.value
        )

    return provider


def discover_files(root: Path) -> List[str]:
    """Discover source files in the repository."""
    # Delay import
    from grepmap.discovery import find_source_files

    return find_source_files(str(root))


def main(
    root: Optional[str] = None,
    port: int = 8765,
    no_open: bool = False,
    verbose: bool = False
):
    """Launch the graph visualization server.

    Args:
        root: Repository root (default: cwd)
        port: HTTP port to listen on
        no_open: Don't open browser automatically
        verbose: Enable verbose logging
    """
    root_path = Path(root or ".").resolve()

    if verbose:
        print(f"📂 Repository: {root_path}")

    # Discover files
    if verbose:
        print("🔍 Discovering files...")
    files = discover_files(root_path)
    if verbose:
        print(f"   Found {len(files)} files")

    # Create graph provider
    provider = create_graph_provider(root_path, files, verbose=verbose)

    # Start server
    server = VisualizerServer(
        graph_provider=provider,
        port=port,
        auto_open=not no_open
    )
    server.start()

    print("Press Ctrl+C to stop the server")
    try:
        # Keep main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="grepmap graph visualizer")
    parser.add_argument("root", nargs="?", default=".", help="Repository root")
    parser.add_argument("-p", "--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    main(
        root=args.root,
        port=args.port,
        no_open=args.no_open,
        verbose=args.verbose
    )
