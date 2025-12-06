"""
Confidence analysis for ranking system - detects uncertainty and low-confidence patterns.

Analyzes rank distributions to identify situations where rankings may be unreliable:
- Flat distributions (low gini) indicate no clear winners - "diffuse" pattern
- Multi-modal distributions suggest competing clusters - "fragmented" pattern
- High orphan ratios signal weak graph connectivity - "sparse" pattern
- Perturbation testing measures rank stability under hyperparameter changes

Used to provide feedback to users when ranking results are ambiguous and may
benefit from additional context (more chat files, git data, etc).
"""

from typing import List, Optional
from dataclasses import dataclass
import math

from grepmap.diagnostics import compute_gini, percentile


@dataclass
class ConfidenceResult:
    """Result of confidence analysis on a ranking distribution."""

    level: str  # "low", "medium", "high"
    patterns: List[str]  # e.g., ["diffuse", "fragmented", "sparse"]
    gini: float  # Gini coefficient (0=equal, 1=concentrated)
    entropy: float  # Shannon entropy of normalized ranks
    stability: float  # 0-1, stability under perturbation (1=stable)

    def __str__(self) -> str:
        patterns_str = ",".join(self.patterns) if self.patterns else "none"
        return f"{self.level} conf (gini:{self.gini:.2f} ent:{self.entropy:.2f} stab:{self.stability:.2f}) [{patterns_str}]"


class ConfidenceEngine:
    """
    Analyzes ranking distributions to detect uncertainty patterns.

    Confidence thresholds are calibrated to typical PageRank distributions
    in codebases - low gini (<0.3) is unusual and suggests diffuse signal,
    high gini (>0.7) with gaps suggests clear hierarchy exists.
    """

    # Thresholds calibrated from empirical observation of real codebases
    GINI_DIFFUSE_THRESHOLD = 0.35  # Below this: ranks too flat
    GINI_CONFIDENT_THRESHOLD = 0.60  # Above this: clear hierarchy
    ENTROPY_HIGH_THRESHOLD = 3.5  # High entropy = flat distribution
    ORPHAN_RATIO_SPARSE = 0.15  # >15% orphans = weak connectivity
    BIMODAL_GAP_RATIO = 0.4  # Gap indicating two clusters

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def analyze(
        self,
        ranks: List[float],
        graph_stats: Optional[dict] = None
    ) -> ConfidenceResult:
        """
        Analyze rank distribution for confidence and uncertainty patterns.

        Args:
            ranks: List of final rank values
            graph_stats: Optional dict with keys:
                - orphan_count: int
                - num_symbols: int
                - num_edges: int

        Returns:
            ConfidenceResult with level and detected patterns
        """
        if not ranks or len(ranks) < 3:
            # Too few items to assess confidence
            return ConfidenceResult(
                level="low",
                patterns=["insufficient_data"],
                gini=0.0,
                entropy=0.0,
                stability=0.0
            )

        patterns = []

        # Core metrics
        gini = compute_gini(ranks)
        entropy = self._compute_entropy(ranks)

        # Pattern detection: diffuse (flat distribution)
        if gini < self.GINI_DIFFUSE_THRESHOLD:
            patterns.append("diffuse")

        # Pattern detection: fragmented (bimodal/clustered)
        if self._detect_bimodal(ranks):
            patterns.append("fragmented")

        # Pattern detection: sparse (weak graph connectivity)
        if graph_stats:
            orphan_ratio = graph_stats.get("orphan_count", 0) / max(graph_stats.get("num_symbols", 1), 1)
            if orphan_ratio > self.ORPHAN_RATIO_SPARSE:
                patterns.append("sparse")

        # Determine overall confidence level
        level = self._assess_confidence_level(gini, entropy, patterns)

        # Stability measurement (simplified - full version would perturb HPs)
        # For now, use gini as proxy: high gini = more stable hierarchy
        stability = min(gini / self.GINI_CONFIDENT_THRESHOLD, 1.0)

        return ConfidenceResult(
            level=level,
            patterns=patterns,
            gini=gini,
            entropy=entropy,
            stability=stability
        )

    def _compute_entropy(self, ranks: List[float]) -> float:
        """
        Shannon entropy of normalized rank distribution.

        High entropy (approaching log2(N)) indicates flat/diffuse distribution.
        Low entropy indicates concentrated distribution with clear winners.
        """
        if not ranks:
            return 0.0

        total = sum(ranks)
        if total == 0:
            return 0.0

        # Normalize to probability distribution
        probs = [r / total for r in ranks]

        # Shannon entropy: -sum(p * log2(p))
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _detect_bimodal(self, ranks: List[float]) -> bool:
        """
        Detect if distribution has two distinct clusters (bimodal).

        Strategy: Check if there's a large gap between top and middle percentiles
        compared to the gap between middle and bottom. If top tier is significantly
        separated, it suggests competing clusters rather than smooth gradient.
        """
        if len(ranks) < 10:
            return False

        p90 = percentile(ranks, 90)
        p50 = percentile(ranks, 50)
        p10 = percentile(ranks, 10)

        if p50 == 0:
            return False

        # Check for significant gap in the middle
        top_gap = p90 - p50
        bottom_gap = p50 - p10

        # If top gap is much larger than bottom gap, suggests clustering
        if bottom_gap > 0 and top_gap / bottom_gap > (1 / self.BIMODAL_GAP_RATIO):
            return True

        return False

    def _assess_confidence_level(
        self,
        gini: float,
        entropy: float,
        patterns: List[str]
    ) -> str:
        """
        Determine overall confidence level from metrics and patterns.

        Logic:
        - High confidence: Strong gini, low entropy, no problematic patterns
        - Medium confidence: Moderate metrics OR one minor pattern
        - Low confidence: Weak metrics OR multiple patterns
        """
        if gini >= self.GINI_CONFIDENT_THRESHOLD and entropy < self.ENTROPY_HIGH_THRESHOLD:
            # Strong signal, clear hierarchy
            if not patterns or patterns == ["fragmented"]:  # fragmented OK if gini high
                return "high"

        if len(patterns) >= 2 or "diffuse" in patterns:
            # Multiple issues or fundamental flatness problem
            return "low"

        # Default: medium confidence
        return "medium"
