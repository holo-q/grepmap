"""Intent classification for ranking recipes.

Classifies user intent from focus targets and query text to provide appropriate
ranking strategies. Uses pattern matching on keywords and context to determine
whether the user is debugging, exploring, extending, or refactoring code.

Design philosophy:
- Simple regex-based classification (no ML overhead)
- Each intent maps to a RankingRecipe with weight adjustments
- Defaults to EXPLORE when intent is ambiguous
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class Intent(Enum):
    """User intent categories for code navigation tasks."""

    DEBUG = "debug"
    EXPLORE = "explore"
    EXTEND = "extend"
    REFACTOR = "refactor"


@dataclass
class RankingRecipe:
    """Weight configuration for a specific intent.

    Each recipe adjusts ranking weights to surface the most relevant symbols:
    - recency_weight: How much to favor recently changed symbols
    - churn_weight: How much to favor frequently changed symbols
    - reverse_edge_bias: Boost for callers (higher = favor call sites over definitions)
    """

    intent: Intent
    recency_weight: float = 1.0
    churn_weight: float = 1.0
    reverse_edge_bias: float = 1.0
    description: str = ""


# Recipe definitions for each intent
RECIPES = {
    Intent.DEBUG: RankingRecipe(
        intent=Intent.DEBUG,
        recency_weight=1.5,  # Recent changes often correlate with bugs
        churn_weight=0.8,    # Moderate churn interest
        reverse_edge_bias=2.0,  # Strongly favor callers to trace error propagation
        description="Debugging - boost callers and recent changes to trace errors"
    ),
    Intent.EXPLORE: RankingRecipe(
        intent=Intent.EXPLORE,
        recency_weight=1.0,  # Neutral - all code equally relevant
        churn_weight=1.0,
        reverse_edge_bias=1.0,  # Neutral - show natural graph structure
        description="Exploration - neutral weights, favor bridges and hubs"
    ),
    Intent.EXTEND: RankingRecipe(
        intent=Intent.EXTEND,
        recency_weight=1.2,  # Slight bias toward active development areas
        churn_weight=0.5,    # Avoid unstable areas for new features
        reverse_edge_bias=0.7,  # Favor definitions (API surfaces) over call sites
        description="Extension - boost API surfaces and stable patterns"
    ),
    Intent.REFACTOR: RankingRecipe(
        intent=Intent.REFACTOR,
        recency_weight=0.8,  # Less interested in very recent changes
        churn_weight=2.0,    # Heavily favor high-churn code (needs cleanup)
        reverse_edge_bias=1.0,  # Neutral - need full picture
        description="Refactoring - boost high-churn, low-stability symbols"
    ),
}


class IntentClassifier:
    """Classifies user intent from focus targets and query text.

    Uses keyword patterns to determine whether the user is:
    - DEBUG: fixing bugs, investigating errors
    - EXPLORE: learning codebase, broad queries
    - EXTEND: adding features, implementing new functionality
    - REFACTOR: cleaning up, reorganizing code
    """

    # Pattern groups for intent detection
    DEBUG_PATTERNS = [
        r'\b(error|bug|fix|crash|fail|exception|stacktrace|trace|debug)\b',
        r'\b(why|broken|not working|issue)\b',
    ]

    EXTEND_PATTERNS = [
        r'\b(add|new|implement|feature|create|build)\b',
        r'\b(extend|enhance|support)\b',
    ]

    REFACTOR_PATTERNS = [
        r'\b(refactor|clean|reorganize|simplify|restructure)\b',
        r'\b(improve|optimize|consolidate)\b',
    ]

    EXPLORE_PATTERNS = [
        r'\b(what is|how does|explain|understand|explore)\b',
        r'\b(show|find|list|search)\b',
    ]

    def __init__(self, verbose: bool = False):
        """Initialize the intent classifier.

        Args:
            verbose: If True, log classification decisions for debugging
        """
        self.verbose = verbose

    def classify(self, focus_targets: List[str], query_text: str = "") -> Intent:
        """Classify intent from focus targets and optional query text.

        Args:
            focus_targets: List of symbols, files, or patterns user is focusing on
            query_text: Optional natural language query or search terms

        Returns:
            Detected Intent (defaults to EXPLORE if ambiguous)
        """
        # Combine all text for pattern matching
        combined_text = " ".join(focus_targets + [query_text]).lower()

        # No focus targets + no query = EXPLORE
        if not combined_text.strip():
            return Intent.EXPLORE

        # Score each intent based on pattern matches
        scores = {
            Intent.DEBUG: self._match_score(combined_text, self.DEBUG_PATTERNS),
            Intent.EXTEND: self._match_score(combined_text, self.EXTEND_PATTERNS),
            Intent.REFACTOR: self._match_score(combined_text, self.REFACTOR_PATTERNS),
            Intent.EXPLORE: self._match_score(combined_text, self.EXPLORE_PATTERNS),
        }

        # Return highest scoring intent, default to EXPLORE on tie
        detected = max(scores.items(), key=lambda x: x[1])

        if self.verbose:
            print(f"Intent classification: {detected[0].value} (scores: {scores})")

        # If all scores are zero, default to EXPLORE
        return detected[0] if detected[1] > 0 else Intent.EXPLORE

    def get_recipe(self, intent: Intent) -> RankingRecipe:
        """Get the ranking recipe for a given intent.

        Args:
            intent: The intent to get recipe for

        Returns:
            RankingRecipe with weight adjustments for this intent
        """
        return RECIPES[intent]

    def _match_score(self, text: str, patterns: List[str]) -> int:
        """Count pattern matches in text.

        Args:
            text: Text to search (should be lowercased)
            patterns: List of regex patterns to match

        Returns:
            Number of pattern matches found
        """
        return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
