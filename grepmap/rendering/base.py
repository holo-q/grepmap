"""
Base renderer protocol for GrepMap rendering strategies.

Defines the interface that all renderers must implement, enabling polymorphic
rendering of ranked tags at different detail levels. This supports the multi-
configuration optimization strategy where we maximize coverage * detail within
token budgets.
"""

from typing import Protocol, List, Set, Optional
from grepmap.core.types import RankedTag, DetailLevel


class Renderer(Protocol):
    """Protocol for rendering ranked tags into formatted output.

    Renderers convert ranked tag lists into human-readable text representations
    at varying levels of detail. Each implementation provides a distinct view:
    - TreeRenderer: Shows code snippets with syntax highlighting
    - DirectoryRenderer: Shows hierarchical symbol overview

    All renderers must support token estimation for budget optimization.
    """

    def render(
        self,
        tags: List[RankedTag],
        chat_files: Set[str],
        detail: DetailLevel,
        overflow_tags: Optional[List[RankedTag]] = None
    ) -> str:
        """Render ranked tags into formatted output.

        Args:
            tags: Primary ranked tags to render in detail
            chat_files: Set of relative filenames currently in chat context
            detail: Level of detail for rendering (LOW/MEDIUM/HIGH)
            overflow_tags: Additional tags for low-resolution summaries (optional)

        Returns:
            Formatted string suitable for displaying to the user

        Note:
            Implementations should handle empty tag lists gracefully.
            The detail level controls signature/field rendering granularity.
        """
        ...

    def estimate_tokens(self, output: str) -> int:
        """Estimate token count for rendered output.

        Args:
            output: The rendered string from render()

        Returns:
            Estimated number of tokens (for LLM context budget)

        Note:
            This should use the same token counting method as the parent
            GrepMap instance for consistency in optimization.
        """
        ...
