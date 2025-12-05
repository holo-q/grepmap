"""
Multi-configuration optimization for token budget management.

This module implements binary search optimization across different detail levels
to find the best rendering configuration within a token budget. The optimizer:

1. For each detail level (LOW/MEDIUM/HIGH):
   - Binary search to find maximum number of tags that fit
   - Score the configuration: num_tags * 10 + detail_level

2. Select configuration with highest score:
   - Prioritizes coverage (number of tags) over detail
   - 10 extra tags are worth 1 detail level increase
   - Ensures we maximize file visibility before adding signature detail

This approach provides adaptive rendering that maximizes information density
within the available token budget.
"""

from typing import List, Optional, Tuple, Callable
from grepmap.core.types import RankedTag, DetailLevel
from grepmap.core.config import SCORE_COVERAGE_WEIGHT, SCORE_DETAIL_WEIGHT, OVERFLOW_COUNT


class Optimizer:
    """Token-budget optimizer for rendering configurations.

    Uses binary search per detail level to find the optimal number of tags
    that fit within the token budget, then selects the configuration that
    maximizes the score function: coverage * 10 + detail.

    This scoring function ensures we prioritize showing more files over
    adding signature detail, which is consistent with the goal of providing
    broad repository awareness.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        verbose: bool = False,
        output_handlers: Optional[dict] = None
    ):
        """Initialize Optimizer.

        Args:
            token_counter: Function to count tokens in a string
            verbose: Enable verbose logging
            output_handlers: Optional dict of output handler functions
        """
        self.token_counter = token_counter
        self.verbose = verbose
        self.output_handlers = output_handlers or {'info': print}

    def optimize(
        self,
        ranked_tags: List[RankedTag],
        max_tokens: int,
        renderer: Callable[[List[RankedTag], DetailLevel], str],
        detail_levels: Optional[List[DetailLevel]] = None
    ) -> Tuple[List[RankedTag], DetailLevel, str, int]:
        """Find optimal rendering configuration within token budget.

        The algorithm:
        1. For each detail level, binary search for max tags that fit
        2. Score each valid configuration: num_tags * 10 + detail_level.value
        3. Select configuration with highest score
        4. Return selected tags, detail level, rendered output, and token count

        Args:
            ranked_tags: All ranked tags to consider (sorted by rank)
            max_tokens: Maximum token budget
            renderer: Function to render tags at given detail level
            detail_levels: List of detail levels to try (default: all three)

        Returns:
            Tuple of (selected_tags, detail_level, output, token_count)

        Raises:
            ValueError: If no valid configuration fits within budget
        """
        if detail_levels is None:
            detail_levels = [DetailLevel.LOW, DetailLevel.MEDIUM, DetailLevel.HIGH]

        n = len(ranked_tags)
        if n == 0:
            raise ValueError("No ranked tags to optimize")

        # Binary search for each detail level to find max tags that fit
        best_configs: List[Tuple[int, DetailLevel, str, int]] = []

        for detail in detail_levels:
            result = self._binary_search_for_detail(
                ranked_tags, max_tokens, renderer, detail
            )
            if result:
                best_configs.append(result)

        if not best_configs:
            # No configuration fits - return minimal fallback
            if self.verbose:
                self.output_handlers['info']("No config fits budget, using minimal fallback")
            raise ValueError("No configuration fits within token budget")

        # Pick config with highest score using configurable weights
        # Score = num_tags * SCORE_COVERAGE_WEIGHT + detail_level * SCORE_DETAIL_WEIGHT
        # Default values strongly favor coverage over detail
        best = max(best_configs, key=lambda x: x[0] * SCORE_COVERAGE_WEIGHT + x[1].value * SCORE_DETAIL_WEIGHT)
        num_tags, detail, output, tokens = best

        if self.verbose:
            self.output_handlers['info'](
                f"Selected: {num_tags} tags, {detail.name} detail, {tokens} tokens "
                f"(from {len(best_configs)} candidates)"
            )

        selected_tags = ranked_tags[:num_tags]
        return selected_tags, detail, output, tokens

    def _binary_search_for_detail(
        self,
        ranked_tags: List[RankedTag],
        max_tokens: int,
        renderer: Callable[[List[RankedTag], DetailLevel], str],
        detail: DetailLevel
    ) -> Optional[Tuple[int, DetailLevel, str, int]]:
        """Binary search to find max tags that fit for a specific detail level.

        Args:
            ranked_tags: All ranked tags (sorted)
            max_tokens: Token budget
            renderer: Rendering function
            detail: Detail level to search for

        Returns:
            Tuple of (num_tags, detail, output, tokens) if found, None otherwise
        """
        n = len(ranked_tags)
        left, right = 1, n
        best_for_detail = None

        while left <= right:
            mid = (left + right) // 2
            selected = ranked_tags[:mid]

            # Try rendering with this configuration
            output = renderer(selected, detail)
            if not output:
                # Rendering failed, try fewer tags
                right = mid - 1
                continue

            tokens = self.token_counter(output)

            if tokens <= max_tokens:
                # This config fits, save it and try more tags
                best_for_detail = (mid, detail, output, tokens)
                left = mid + 1
            else:
                # Too many tokens, try fewer tags
                right = mid - 1

        return best_for_detail

    def optimize_with_overflow(
        self,
        ranked_tags: List[RankedTag],
        max_tokens: int,
        renderer_with_overflow: Callable[[List[RankedTag], DetailLevel, List[RankedTag]], str],
        detail_levels: Optional[List[DetailLevel]] = None,
        overflow_count: int = OVERFLOW_COUNT
    ) -> Tuple[List[RankedTag], DetailLevel, str, int]:
        """Find optimal config with overflow tags for low-res "also in scope" section.

        Similar to optimize(), but re-renders the final selection with overflow tags
        to include a low-resolution summary of additional files beyond the detailed view.

        Args:
            ranked_tags: All ranked tags to consider
            max_tokens: Maximum token budget
            renderer_with_overflow: Function that takes (tags, detail, overflow_tags)
            detail_levels: List of detail levels to try
            overflow_count: Number of overflow tags to pass to renderer

        Returns:
            Tuple of (selected_tags, detail_level, output, token_count)
        """
        # First find the optimal config without overflow
        selected_tags, detail, _, _ = self.optimize(
            ranked_tags,
            max_tokens,
            lambda tags, d: renderer_with_overflow(tags, d, []),
            detail_levels
        )

        num_selected = len(selected_tags)
        n = len(ranked_tags)

        # Re-render with overflow tags if there are remaining tags
        if num_selected < n:
            overflow_end = min(num_selected + overflow_count, n)
            overflow = ranked_tags[num_selected:overflow_end]
            output = renderer_with_overflow(selected_tags, detail, overflow)
            tokens = self.token_counter(output)

            if self.verbose:
                self.output_handlers['info'](
                    f"Re-rendered with {len(overflow)} overflow tags for extended scope"
                )

            return selected_tags, detail, output, tokens
        else:
            # No overflow needed
            output = renderer_with_overflow(selected_tags, detail, [])
            tokens = self.token_counter(output)
            return selected_tags, detail, output, tokens
