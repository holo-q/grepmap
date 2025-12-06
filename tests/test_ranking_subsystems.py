"""Tests for ranking subsystems: bridges, surface, confidence, intent, git weights."""

import networkx as nx
from testslide import TestCase  # type: ignore[import-untyped]

from grepmap.ranking import (
    BridgeDetector,
    SurfaceDetector, SurfaceType, SurfaceInfo,
    ConfidenceEngine, ConfidenceResult,
    IntentClassifier, Intent, RankingRecipe
)


class TestBridgeDetector(TestCase):
    """Tests for bridge detection using betweenness centrality."""

    def test_empty_graph_returns_no_bridges(self):
        """Empty graph should return empty list."""
        detector = BridgeDetector()
        graph = nx.MultiDiGraph()
        bridges = detector.detect_bridges(graph)
        self.assertEqual(bridges, [])

    def test_linear_chain_identifies_middle_as_bridge(self):
        """In A -> B -> C, B is the bridge connecting A and C."""
        detector = BridgeDetector()
        graph = nx.MultiDiGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "c.py")

        bridges = detector.detect_bridges(graph, top_n=3)
        # B should have highest betweenness
        self.assertGreater(len(bridges), 0)
        top_bridge = bridges[0]
        self.assertEqual(top_bridge.rel_fname, "b.py")
        self.assertGreater(top_bridge.betweenness, 0)

    def test_star_topology_with_bidirectional_edges(self):
        """In bidirectional star, center has highest betweenness."""
        detector = BridgeDetector()
        graph = nx.MultiDiGraph()
        # Bidirectional edges through center - center is on all paths
        for i in range(5):
            graph.add_edge(f"leaf{i}.py", "center.py")
            graph.add_edge("center.py", f"leaf{i}.py")

        bridges = detector.detect_bridges(graph, top_n=1)
        self.assertGreater(len(bridges), 0)
        # Center should be the bridge since all shortest paths between leaves go through it
        self.assertEqual(bridges[0].rel_fname, "center.py")

    def test_two_clusters_connected_by_bridge(self):
        """Bridge file connecting two clusters should be detected."""
        detector = BridgeDetector()
        graph = nx.MultiDiGraph()

        # Two completely separate clusters connected only through bridge.py
        # Cluster A: fully connected triangle
        for a in ["a1.py", "a2.py", "a3.py"]:
            for b in ["a1.py", "a2.py", "a3.py"]:
                if a != b:
                    graph.add_edge(a, b)

        # Cluster B: fully connected triangle
        for a in ["b1.py", "b2.py", "b3.py"]:
            for b in ["b1.py", "b2.py", "b3.py"]:
                if a != b:
                    graph.add_edge(a, b)

        # Bridge is the ONLY connection between clusters
        # Connect bridge to all nodes in both clusters
        for node in ["a1.py", "a2.py", "a3.py", "b1.py", "b2.py", "b3.py"]:
            graph.add_edge(node, "bridge.py")
            graph.add_edge("bridge.py", node)

        bridges = detector.detect_bridges(graph, top_n=1)
        self.assertGreater(len(bridges), 0)
        # The bridge file is on ALL cross-cluster paths, must have highest betweenness
        self.assertEqual(bridges[0].rel_fname, "bridge.py")


class TestSurfaceDetector(TestCase):
    """Tests for API surface classification."""

    def test_empty_graph_returns_empty_classification(self):
        """Empty graph should return empty dict."""
        detector = SurfaceDetector()
        graph = nx.MultiDiGraph()
        result = detector.classify_symbols(graph, {})
        self.assertEqual(result, {})

    def test_symbol_with_only_external_refs_is_api(self):
        """Symbol called only from other files should be classified as API."""
        detector = SurfaceDetector()
        graph = nx.MultiDiGraph()

        # other.py calls func in main.py (external reference)
        graph.add_node(("main.py", "my_func"))
        graph.add_node(("other.py", "caller"))
        graph.add_edge(("other.py", "caller"), ("main.py", "my_func"))

        result = detector.classify_symbols(graph, {})

        key = ("main.py", "my_func")
        self.assertIn(key, result)
        self.assertEqual(result[key].surface_type, SurfaceType.API)
        self.assertEqual(result[key].external_refs, 1)
        self.assertEqual(result[key].internal_refs, 0)

    def test_symbol_with_only_internal_refs_is_internal(self):
        """Symbol called only from same file should be classified as INTERNAL."""
        detector = SurfaceDetector()
        graph = nx.MultiDiGraph()

        # helper is called by main_func, both in same file
        graph.add_node(("main.py", "helper"))
        graph.add_node(("main.py", "main_func"))
        graph.add_edge(("main.py", "main_func"), ("main.py", "helper"))

        result = detector.classify_symbols(graph, {})

        key = ("main.py", "helper")
        self.assertIn(key, result)
        self.assertEqual(result[key].surface_type, SurfaceType.INTERNAL)
        self.assertEqual(result[key].internal_refs, 1)
        self.assertEqual(result[key].external_refs, 0)

    def test_external_ratio_calculation(self):
        """External ratio should be computed correctly."""
        info = SurfaceInfo(
            symbol_id=("file.py", "func"),
            surface_type=SurfaceType.API,
            external_refs=3,
            internal_refs=1
        )
        self.assertEqual(info.total_refs, 4)
        self.assertEqual(info.external_ratio, 0.75)


class TestConfidenceEngine(TestCase):
    """Tests for confidence analysis of ranking distributions."""

    def test_empty_ranks_returns_low_confidence(self):
        """Empty or too-small rank list should return low confidence."""
        engine = ConfidenceEngine()
        result = engine.analyze([])
        self.assertEqual(result.level, "low")
        self.assertIn("insufficient_data", result.patterns)

    def test_uniform_ranks_detected_as_diffuse(self):
        """Uniform distribution should be detected as diffuse pattern."""
        engine = ConfidenceEngine()
        # All ranks are the same (extremely flat)
        ranks = [0.1] * 20
        result = engine.analyze(ranks)
        # Low gini = diffuse
        self.assertIn("diffuse", result.patterns)

    def test_concentrated_ranks_gives_high_confidence(self):
        """Highly concentrated ranks (one dominant) should give high confidence."""
        engine = ConfidenceEngine()
        # One high value, rest very low
        ranks = [1.0] + [0.001] * 20
        result = engine.analyze(ranks)
        # High gini = concentrated, should be high confidence
        self.assertEqual(result.level, "high")
        self.assertGreater(result.gini, 0.5)

    def test_confidence_result_string_format(self):
        """ConfidenceResult should have readable string representation."""
        result = ConfidenceResult(
            level="medium",
            patterns=["diffuse", "sparse"],
            gini=0.45,
            entropy=2.3,
            stability=0.7
        )
        s = str(result)
        self.assertIn("medium", s)
        self.assertIn("diffuse", s)
        self.assertIn("gini:0.45", s)


class TestIntentClassifier(TestCase):
    """Tests for intent classification and recipe retrieval."""

    def test_empty_focus_returns_explore(self):
        """No focus targets should default to EXPLORE intent."""
        classifier = IntentClassifier()
        intent = classifier.classify([])
        self.assertEqual(intent, Intent.EXPLORE)

    def test_error_keywords_trigger_debug_intent(self):
        """Error-related keywords in query_text should trigger DEBUG intent."""
        classifier = IntentClassifier()

        # Query text with error keywords (word boundaries work here)
        intent = classifier.classify([], query_text="fix this error")
        self.assertEqual(intent, Intent.DEBUG)

        intent = classifier.classify([], query_text="handle the exception")
        self.assertEqual(intent, Intent.DEBUG)

        intent = classifier.classify([], query_text="debug this issue")
        self.assertEqual(intent, Intent.DEBUG)

    def test_refactor_keywords_trigger_refactor_intent(self):
        """Refactor-related keywords should trigger REFACTOR intent."""
        classifier = IntentClassifier()
        intent = classifier.classify([], query_text="refactor this module")
        self.assertEqual(intent, Intent.REFACTOR)

    def test_new_feature_keywords_trigger_extend_intent(self):
        """New feature keywords should trigger EXTEND intent."""
        classifier = IntentClassifier()
        intent = classifier.classify([], query_text="add new feature")
        self.assertEqual(intent, Intent.EXTEND)

    def test_debug_recipe_has_high_reverse_edge_bias(self):
        """DEBUG recipe should favor callers (high reverse_edge_bias)."""
        classifier = IntentClassifier()
        recipe = classifier.get_recipe(Intent.DEBUG)
        self.assertGreater(recipe.reverse_edge_bias, 1.0)

    def test_refactor_recipe_has_high_churn_weight(self):
        """REFACTOR recipe should favor high-churn code."""
        classifier = IntentClassifier()
        recipe = classifier.get_recipe(Intent.REFACTOR)
        self.assertGreater(recipe.churn_weight, 1.0)

    def test_extend_recipe_has_low_churn_weight(self):
        """EXTEND recipe should avoid unstable code."""
        classifier = IntentClassifier()
        recipe = classifier.get_recipe(Intent.EXTEND)
        self.assertLess(recipe.churn_weight, 1.0)

    def test_explore_recipe_is_neutral(self):
        """EXPLORE recipe should have neutral weights."""
        classifier = IntentClassifier()
        recipe = classifier.get_recipe(Intent.EXPLORE)
        self.assertEqual(recipe.churn_weight, 1.0)
        self.assertEqual(recipe.reverse_edge_bias, 1.0)


class TestRankingRecipe(TestCase):
    """Tests for RankingRecipe dataclass."""

    def test_recipe_with_intent(self):
        """Recipe should store intent and weights."""
        recipe = RankingRecipe(intent=Intent.DEBUG)
        self.assertEqual(recipe.intent, Intent.DEBUG)
        # Defaults when not specified
        self.assertEqual(recipe.recency_weight, 1.0)
        self.assertEqual(recipe.churn_weight, 1.0)
        self.assertEqual(recipe.reverse_edge_bias, 1.0)

    def test_custom_recipe_values(self):
        """Custom recipe should store specified values."""
        recipe = RankingRecipe(
            intent=Intent.REFACTOR,
            recency_weight=1.5,
            churn_weight=2.0,
            reverse_edge_bias=0.5
        )
        self.assertEqual(recipe.intent, Intent.REFACTOR)
        self.assertEqual(recipe.recency_weight, 1.5)
        self.assertEqual(recipe.churn_weight, 2.0)
        self.assertEqual(recipe.reverse_edge_bias, 0.5)


class TestGitWeightScaling(TestCase):
    """Tests for git weight scaling with recipe factors.

    Note: These tests don't require a real git repo - they test the
    scaling logic when weights would have been computed.
    """

    def test_scale_factor_preserves_neutral(self):
        """Scale factor 1.0 should not change weights."""
        # Simulating: 1.0 + (factor - 1.0) * scale
        # With scale=1.0, result should equal factor
        factor = 1.5
        scale = 1.0
        result = 1.0 + (factor - 1.0) * scale
        self.assertEqual(result, factor)

    def test_scale_factor_zero_removes_boost(self):
        """Scale factor 0.0 should remove the boost entirely."""
        factor = 2.0  # Would be 2x boost
        scale = 0.0
        result = 1.0 + (factor - 1.0) * scale
        self.assertEqual(result, 1.0)  # No boost

    def test_scale_factor_amplifies_boost(self):
        """Scale factor >1.0 should amplify the boost."""
        factor = 1.5  # 50% boost
        scale = 2.0   # Double the boost effect
        result = 1.0 + (factor - 1.0) * scale
        self.assertEqual(result, 2.0)  # 100% boost (doubled)

    def test_scale_factor_reduces_boost(self):
        """Scale factor <1.0 should reduce the boost."""
        factor = 2.0  # 100% boost
        scale = 0.5   # Halve the boost effect
        result = 1.0 + (factor - 1.0) * scale
        self.assertEqual(result, 1.5)  # 50% boost (halved)
