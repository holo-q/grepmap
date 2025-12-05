"""
Ranking boost calculations for tag importance.

This module applies multiplicative boosts to PageRank scores based on:
- Chat files: Files currently in conversation get 20x boost
- Mentioned files: Files mentioned in prompts get 5x boost
- Mentioned identifiers: Specific symbols mentioned get 10x boost

These boosts ensure that conversational context directly influences the
ranking, making the map more relevant to the current task.

The boosts are multiplicative and can combine (e.g., a mentioned identifier
in a chat file gets 20x * 10x = 200x boost).
"""

from typing import List, Dict, Set, Optional, Callable
from grepmap.core.types import Tag, RankedTag
from grepmap.core.config import (
    BOOST_MENTIONED_IDENT, BOOST_MENTIONED_FILE, BOOST_CHAT_FILE,
    EXCLUDE_UNRANKED_THRESHOLD
)


class BoostCalculator:
    """Applies contextual boosts to PageRank scores.

    Boosts are multiplicative factors applied to base PageRank scores to
    emphasize files and symbols relevant to the current conversation.

    The boost hierarchy (from strongest to weakest):
    1. Chat files (20x): Files actively being edited or discussed
    2. Mentioned identifiers (10x): Specific symbols mentioned in prompts
    3. Mentioned files (5x): Files referenced in conversation

    These combine multiplicatively for compound effects.
    """

    def __init__(
        self,
        get_rel_fname: Callable[[str], str],
        exclude_unranked: bool = False
    ):
        """Initialize BoostCalculator.

        Args:
            get_rel_fname: Function to convert absolute path to relative path
            exclude_unranked: If True, exclude tags with near-zero PageRank
        """
        self.get_rel_fname = get_rel_fname
        self.exclude_unranked = exclude_unranked

    def apply_boosts(
        self,
        included_files: List[str],
        tags_by_file: Dict[str, List[Tag]],
        ranks: Dict[str, float],
        chat_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> List[RankedTag]:
        """Apply boosts to PageRank scores and create RankedTag list.

        The algorithm:
        1. For each file and its definition tags
        2. Get base PageRank score for the file
        3. Apply multiplicative boosts based on context
        4. Create RankedTag with boosted score
        5. Sort by rank descending

        Args:
            included_files: List of absolute file paths that were processed
            tags_by_file: Dict mapping absolute fname to its list of tags
            ranks: Dict mapping relative fname to PageRank score
            chat_fnames: List of chat file absolute paths
            mentioned_fnames: Set of mentioned file relative paths
            mentioned_idents: Set of mentioned identifier names

        Returns:
            List of RankedTag objects sorted by rank descending
        """
        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()

        # Convert chat files to relative paths for comparison
        chat_rel_fnames = set(self.get_rel_fname(f) for f in chat_fnames)

        ranked_tags = []

        for fname in included_files:
            rel_fname = self.get_rel_fname(fname)
            file_rank = ranks.get(rel_fname, 0.0)

            # Exclude files with low PageRank if exclude_unranked is True
            # Use a small threshold to exclude near-zero ranks (likely disconnected nodes)
            if self.exclude_unranked and file_rank <= EXCLUDE_UNRANKED_THRESHOLD:
                continue

            tags = tags_by_file.get(fname, [])

            # Only boost definition tags (not references)
            for tag in tags:
                if tag.kind == "def":
                    # Calculate multiplicative boost based on context
                    boost = self._calculate_boost(
                        tag,
                        rel_fname,
                        chat_rel_fnames,
                        mentioned_fnames,
                        mentioned_idents
                    )

                    final_rank = file_rank * boost
                    ranked_tags.append(RankedTag(final_rank, tag))

        # Sort by rank descending (highest importance first)
        ranked_tags.sort(key=lambda x: x.rank, reverse=True)

        return ranked_tags

    def _calculate_boost(
        self,
        tag: Tag,
        rel_fname: str,
        chat_rel_fnames: Set[str],
        mentioned_fnames: Set[str],
        mentioned_idents: Set[str]
    ) -> float:
        """Calculate multiplicative boost for a single tag.

        Boosts are applied in order and multiply together using
        configurable values from grepmap.core.config:
        1. Base boost: 1.0 (no boost)
        2. If identifier is mentioned: multiply by BOOST_MENTIONED_IDENT
        3. If file is mentioned: multiply by BOOST_MENTIONED_FILE
        4. If file is in chat: multiply by BOOST_CHAT_FILE

        Example: A mentioned identifier in a chat file gets
                 1.0 * BOOST_MENTIONED_IDENT * BOOST_CHAT_FILE

        Args:
            tag: The tag to boost
            rel_fname: Relative filename of the tag
            chat_rel_fnames: Set of chat file relative paths
            mentioned_fnames: Set of mentioned file relative paths
            mentioned_idents: Set of mentioned identifier names

        Returns:
            Multiplicative boost factor
        """
        boost = 1.0

        # Boost for mentioned identifiers (strongest individual signal)
        if tag.name in mentioned_idents:
            boost *= BOOST_MENTIONED_IDENT

        # Boost for mentioned files
        if rel_fname in mentioned_fnames:
            boost *= BOOST_MENTIONED_FILE

        # Boost for chat files (strongest file-level signal)
        if rel_fname in chat_rel_fnames:
            boost *= BOOST_CHAT_FILE

        return boost
