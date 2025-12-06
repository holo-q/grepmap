"""
Cluster-based rendering for low-confidence rankings.

When the ranking confidence is low (fragmented, sparse, or diffuse patterns),
a flat ranked list may be misleading. This module groups symbols into clusters
based on their containing files and directories, providing a structural view
that doesn't pretend to know exact importance order.

Design rationale:
- Low confidence = uncertain ranking → show structure, not false precision
- Clusters by directory help users see architectural organization
- Within clusters, show top symbols but don't claim fine-grained ordering

The cluster renderer is triggered when ConfidenceEngine returns level="low".
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set

from grepmap.core.types import RankedTag, DetailLevel


class ClusterRenderer:
    """Render ranked tags as directory-based clusters.

    Used when ranking confidence is low to avoid misleading precision.
    Groups symbols by directory and renders each cluster with its top symbols.

    The algorithm:
    1. Group tags by top-level directory (depth 1 or 2)
    2. Sort clusters by total rank (sum of member ranks)
    3. Within each cluster, show top N symbols
    4. Render with clear cluster headers
    """

    # How many symbols to show per cluster
    SYMBOLS_PER_CLUSTER = 5

    # Minimum symbols in a cluster to show it
    MIN_CLUSTER_SIZE = 2

    def __init__(self, verbose: bool = False):
        """Initialize ClusterRenderer.

        Args:
            verbose: Enable diagnostic output
        """
        self.verbose = verbose

    def render(
        self,
        ranked_tags: List[RankedTag],
        chat_rel_fnames: Set[str],
        detail_level: DetailLevel = DetailLevel.MEDIUM,
        max_clusters: int = 10
    ) -> str:
        """Render tags as directory clusters.

        Args:
            ranked_tags: Ranked tags to render
            chat_rel_fnames: Files in chat context (highlighted)
            detail_level: Level of detail for signatures
            max_clusters: Maximum number of clusters to show

        Returns:
            Formatted cluster view as string
        """
        if not ranked_tags:
            return ""

        # Group by directory
        clusters = self._group_by_directory(ranked_tags)

        # Sort clusters by total rank (highest first)
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: sum(rt.rank for rt in x[1]),
            reverse=True
        )

        # Filter small clusters
        sorted_clusters = [
            (dir_path, tags) for dir_path, tags in sorted_clusters
            if len(tags) >= self.MIN_CLUSTER_SIZE
        ][:max_clusters]

        if not sorted_clusters:
            # Fallback: no meaningful clusters, just show top tags
            return self._render_flat(ranked_tags[:20], detail_level)

        lines = []
        lines.append("# Repository Structure (confidence: low, showing clusters)")
        lines.append("")

        for dir_path, tags in sorted_clusters:
            cluster_rank = sum(rt.rank for rt in tags)
            lines.append(f"## {dir_path}/ ({len(tags)} symbols, rank={cluster_rank:.2f})")

            # Show top symbols in this cluster
            top_tags = sorted(tags, key=lambda x: x.rank, reverse=True)[:self.SYMBOLS_PER_CLUSTER]

            for rt in top_tags:
                tag = rt.tag
                symbol = tag.name if hasattr(tag, 'name') else str(tag)

                # Format based on detail level
                if detail_level == DetailLevel.HIGH and hasattr(tag, 'signature'):
                    sig = getattr(tag, 'signature', '')
                    lines.append(f"  - {symbol}: {sig}")
                else:
                    lines.append(f"  - {symbol}")

            if len(tags) > self.SYMBOLS_PER_CLUSTER:
                remaining = len(tags) - self.SYMBOLS_PER_CLUSTER
                lines.append(f"  ... +{remaining} more")

            lines.append("")

        return "\n".join(lines)

    def _group_by_directory(
        self,
        ranked_tags: List[RankedTag]
    ) -> Dict[str, List[RankedTag]]:
        """Group ranked tags by their top-level directory.

        Uses the first 2 path components as the cluster key.
        Example: "grepmap/ranking/boosts.py" → "grepmap/ranking"
        """
        clusters: Dict[str, List[RankedTag]] = defaultdict(list)

        for rt in ranked_tags:
            tag = rt.tag
            rel_fname = tag.fname if hasattr(tag, 'fname') else ""

            if not rel_fname:
                clusters["(root)"].append(rt)
                continue

            # Handle absolute paths by extracting relative portion
            path = Path(rel_fname)
            if path.is_absolute():
                # Find common prefix to extract relative path
                # Look for common directory patterns
                parts = path.parts
                for i, part in enumerate(parts):
                    if part in ('grepmap', 'src', 'lib', 'tests', 'app'):
                        parts = parts[i:]
                        break
                else:
                    # Fallback: use last 3 components
                    parts = parts[-3:] if len(parts) > 3 else parts
            else:
                parts = path.parts

            # Exclude filename, get directory parts
            dir_parts = parts[:-1] if parts else []

            if len(dir_parts) >= 2:
                dir_key = "/".join(dir_parts[:2])
            elif len(dir_parts) == 1:
                dir_key = dir_parts[0]
            else:
                dir_key = "(root)"

            clusters[dir_key].append(rt)

        return dict(clusters)

    def _render_flat(
        self,
        ranked_tags: List[RankedTag],
        detail_level: DetailLevel
    ) -> str:
        """Fallback flat rendering when clusters aren't useful."""
        lines = ["# Top Symbols (no clear clusters detected)", ""]

        for rt in ranked_tags:
            tag = rt.tag
            symbol = tag.name if hasattr(tag, 'name') else str(tag)
            fname = tag.fname if hasattr(tag, 'fname') else ""
            lines.append(f"- {fname}:{symbol}")

        return "\n".join(lines)


def should_use_clusters(confidence_level: str, patterns: List[str]) -> bool:
    """Determine if cluster rendering should be used.

    Uses cluster rendering when:
    - Confidence is "low"
    - Patterns indicate uncertainty (fragmented, sparse, diffuse)

    Args:
        confidence_level: Level from ConfidenceEngine ("high", "medium", "low")
        patterns: List of detected patterns from ConfidenceEngine

    Returns:
        True if cluster rendering is recommended
    """
    if confidence_level == "low":
        return True

    uncertain_patterns = {"fragmented", "sparse", "diffuse"}
    if any(p in uncertain_patterns for p in patterns):
        return True

    return False
