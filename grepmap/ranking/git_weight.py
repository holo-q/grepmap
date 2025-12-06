"""
Git-based weighting for temporal importance ranking.

This module provides git-aware weighting factors that complement PageRank:
- Recency: Recently modified files are more contextually relevant
- Churn: Frequently changed files (hotspots) indicate active development areas
- Authorship: Files modified by current user may be more relevant

These weights are multiplicative boosts applied to PageRank scores,
biasing the ranking toward code that's actively being worked on.

Design note: This is conditional/opt-in to maintain fast operation when
git data isn't needed and to gracefully handle non-git repositories.
"""

import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Optional, Callable, List

from grepmap.core.config import (
    GIT_RECENCY_DECAY_DAYS,
    GIT_RECENCY_MAX_BOOST,
    GIT_CHURN_THRESHOLD,
    GIT_CHURN_MAX_BOOST,
    GIT_AUTHOR_BOOST,
    GIT_BADGE_RECENT_DAYS,
    GIT_BADGE_CHURN_COMMITS,
    PHASE_CRYSTAL_MIN_AGE_DAYS,
    PHASE_CRYSTAL_MIN_QUIET_DAYS,
    PHASE_ROTTING_MIN_AGE_DAYS,
    PHASE_ROTTING_MAX_QUIET_DAYS,
    PHASE_ROTTING_CHURN_MULTIPLIER,
    PHASE_EMERGENT_MAX_AGE_DAYS
)


class GitWeightCalculator:
    """Calculate git-based importance weights for files.

    Extracts git metadata (commit history, authorship) and converts
    it into multiplicative weights for ranking. Designed for efficiency
    with large repositories by using git log with path limiting.

    The weighting factors:
    - Recency: Exponential decay based on days since last modification
    - Churn: Logarithmic boost based on number of commits touching file
    - Authorship: Fixed boost if file was modified by current git user

    All weights are multiplicative and >= 1.0 (no penalty, only boosts).
    """

    def __init__(
        self,
        root: Path,
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize GitWeightCalculator.

        Args:
            root: Repository root path (must contain .git)
            verbose: Enable verbose logging
            output_handler: Function for info messages (default: print)
        """
        self.root = root
        self.verbose = verbose
        self.output_handler = output_handler or print

        # Cache for git data
        self._current_user: Optional[str] = None
        self._file_stats: Dict[str, dict] = {}

    def compute_weights(
        self,
        rel_fnames: List[str],
        use_recency: bool = True,
        use_churn: bool = True,
        use_authorship: bool = False,
        recency_scale: float = 1.0,
        churn_scale: float = 1.0
    ) -> Dict[str, float]:
        """Compute git-based weights for files.

        Fetches git history for the specified files and computes
        multiplicative weight factors based on enabled criteria.

        The scale parameters allow intent-driven adjustment of boost strength:
        - recency_scale: Multiplier for recency factor (1.0 = default)
        - churn_scale: Multiplier for churn factor (1.0 = default)

        Args:
            rel_fnames: List of relative file paths to weight
            use_recency: Apply recency boost (exponential decay)
            use_churn: Apply churn boost (commit frequency)
            use_authorship: Apply authorship boost (current user)
            recency_scale: Scale factor for recency boost (from RankingRecipe)
            churn_scale: Scale factor for churn boost (from RankingRecipe)

        Returns:
            Dict mapping rel_fname to weight multiplier (>= 1.0)
        """
        if not self._is_git_repo():
            if self.verbose:
                self.output_handler("Not a git repository, skipping git weights")
            return {f: 1.0 for f in rel_fnames}

        # Batch fetch git stats for all files
        self._fetch_git_stats(rel_fnames)

        weights = {}
        now = datetime.now(timezone.utc)

        for rel_fname in rel_fnames:
            weight = 1.0
            stats = self._file_stats.get(rel_fname)

            if not stats:
                weights[rel_fname] = 1.0
                continue

            # Recency boost: exponential decay based on age
            # Scale factor adjusts how strongly recency affects ranking
            if use_recency and stats.get('last_modified'):
                age_days = (now - stats['last_modified']).days
                recency_factor = self._compute_recency_factor(age_days)
                # Apply scale: 1.0 + (factor - 1.0) * scale
                # This preserves neutral=1.0 while scaling the boost magnitude
                scaled_recency = 1.0 + (recency_factor - 1.0) * recency_scale
                weight *= scaled_recency

            # Churn boost: based on number of commits
            # Higher churn_scale favors frequently changed code (e.g., REFACTOR intent)
            if use_churn and stats.get('commit_count'):
                churn_factor = self._compute_churn_factor(stats['commit_count'])
                scaled_churn = 1.0 + (churn_factor - 1.0) * churn_scale
                weight *= scaled_churn

            # Authorship boost: if modified by current user
            if use_authorship and stats.get('authors'):
                if self._current_user and self._current_user in stats['authors']:
                    weight *= GIT_AUTHOR_BOOST

            weights[rel_fname] = weight

        if self.verbose:
            self._log_weight_stats(weights)

        return weights

    def _is_git_repo(self) -> bool:
        """Check if root is a git repository."""
        return (self.root / '.git').exists()

    def _fetch_git_stats(self, rel_fnames: List[str]):
        """Fetch git stats for multiple files efficiently.

        Uses a single git log command with --name-only to get history
        for all tracked files, then filters to requested files.
        """
        if not rel_fnames:
            return

        # Get current git user
        try:
            result = subprocess.run(
                ['git', 'config', 'user.email'],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._current_user = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fetch git log for all files in one call
        # Format: commit_date|author_email|filename
        try:
            result = subprocess.run(
                [
                    'git', 'log',
                    '--format=%aI|%ae',  # ISO date, author email
                    '--name-only',
                    '--all',
                    '-n', '500',  # Limit commits for performance
                    '--'
                ] + [str(self.root / f) for f in rel_fnames[:100]],  # Limit files
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=30
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            if self.verbose:
                self.output_handler("Git log timed out or git not found")
            return

        if result.returncode != 0:
            return

        # Parse git log output
        # Format: alternating header lines (date|email) and filename lines
        stats: Dict[str, dict] = defaultdict(lambda: {
            'last_modified': None,
            'commit_count': 0,
            'authors': set()
        })

        current_date = None
        current_author = None

        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue

            if '|' in line:
                # Header line: date|author
                parts = line.split('|', 1)
                if len(parts) == 2:
                    try:
                        current_date = datetime.fromisoformat(parts[0].replace('Z', '+00:00'))
                        current_author = parts[1]
                    except ValueError:
                        current_date = None
                        current_author = None
            else:
                # Filename line
                # Convert absolute path back to relative
                try:
                    abs_path = Path(line)
                    if abs_path.is_absolute():
                        rel_path = str(abs_path.relative_to(self.root))
                    else:
                        rel_path = line
                except (ValueError, OSError):
                    rel_path = line

                if rel_path in rel_fnames or any(rel_path.endswith(f) for f in rel_fnames):
                    # Find matching rel_fname
                    for rf in rel_fnames:
                        if rel_path == rf or rel_path.endswith(rf):
                            file_stats = stats[rf]
                            file_stats['commit_count'] += 1

                            if current_date:
                                if (file_stats['last_modified'] is None or
                                    current_date > file_stats['last_modified']):
                                    file_stats['last_modified'] = current_date

                            if current_author:
                                file_stats['authors'].add(current_author)
                            break

        self._file_stats = dict(stats)

    def _compute_recency_factor(self, age_days: int) -> float:
        """Compute recency boost factor based on file age.

        Uses exponential decay: newer files get higher boost.
        Files modified today get max boost, decaying to 1.0 over time.

        Args:
            age_days: Days since last modification

        Returns:
            Boost factor >= 1.0
        """
        if age_days <= 0:
            return GIT_RECENCY_MAX_BOOST

        # Exponential decay: boost = 1 + (max_boost - 1) * e^(-age/decay_constant)
        import math
        decay = math.exp(-age_days / GIT_RECENCY_DECAY_DAYS)
        return 1.0 + (GIT_RECENCY_MAX_BOOST - 1.0) * decay

    def _compute_churn_factor(self, commit_count: int) -> float:
        """Compute churn boost factor based on commit frequency.

        Uses logarithmic scaling: more commits = higher boost, but diminishing returns.

        Args:
            commit_count: Number of commits touching this file

        Returns:
            Boost factor >= 1.0
        """
        if commit_count <= GIT_CHURN_THRESHOLD:
            return 1.0

        import math
        # Logarithmic boost: grows slowly after threshold
        excess = commit_count - GIT_CHURN_THRESHOLD
        boost = 1.0 + math.log1p(excess) * (GIT_CHURN_MAX_BOOST - 1.0) / 5
        return min(boost, GIT_CHURN_MAX_BOOST)

    def _log_weight_stats(self, weights: Dict[str, float]):
        """Log git weight statistics for debugging."""
        if not weights:
            return

        values = list(weights.values())
        boosted = [v for v in values if v > 1.0]

        self.output_handler(
            f"Git weights: {len(boosted)}/{len(values)} files boosted, "
            f"max boost: {max(values):.2f}x"
        )

        # Show top boosted files
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_boosted = [(f, w) for f, w in sorted_weights[:5] if w > 1.0]
        if top_boosted:
            self.output_handler("Top git-boosted files:")
            for fname, weight in top_boosted:
                self.output_handler(f"  {fname}: {weight:.2f}x")

    def compute_badges(self, rel_fnames: List[str]) -> Dict[str, List[str]]:
        """Compute display badges for files based on git stats.

        Badges are for visual annotation only, not ranking. They surface
        temporal signals to help orient the reader:
        - [recent]: Modified within GIT_BADGE_RECENT_DAYS
        - [high-churn]: 10+ commits touching this file

        Args:
            rel_fnames: List of relative file paths

        Returns:
            Dict mapping rel_fname to list of badge strings
        """
        if not self._is_git_repo():
            return {}

        # Ensure git stats are fetched
        if not self._file_stats:
            self._fetch_git_stats(rel_fnames)

        now = datetime.now(timezone.utc)
        badges: Dict[str, List[str]] = {}

        for rel_fname in rel_fnames:
            stats = self._file_stats.get(rel_fname)
            if not stats:
                continue

            file_badges: List[str] = []

            # Recent badge
            if stats.get('last_modified'):
                age_days = (now - stats['last_modified']).days
                if age_days <= GIT_BADGE_RECENT_DAYS:
                    file_badges.append("recent")

            # High-churn badge
            if stats.get('commit_count', 0) >= GIT_BADGE_CHURN_COMMITS:
                file_badges.append("high-churn")

            if file_badges:
                badges[rel_fname] = file_badges

        return badges

    def classify_phases(self, rel_fnames: List[str]) -> Dict[str, str]:
        """Classify files into lifecycle phases based on git history.

        Phases provide a "phase portrait" of code stability:
        - crystal: Old, stable code (age > 180d, quiet > 30d, low churn)
        - rotting: Old but churning code (age > 90d, recent changes, high churn)
        - emergent: New code (age < 30d)
        - evolving: Everything else (normal development)

        All classification is purely heuristic from git stats - no learning.

        Args:
            rel_fnames: List of relative file paths

        Returns:
            Dict mapping rel_fname to phase string
        """
        if not self._is_git_repo():
            return {}

        # Ensure git stats are fetched
        if not self._file_stats:
            self._fetch_git_stats(rel_fnames)

        now = datetime.now(timezone.utc)
        phases: Dict[str, str] = {}

        # Compute median churn for rotting detection
        churn_values = [
            s.get('commit_count', 0)
            for s in self._file_stats.values()
            if s.get('commit_count', 0) > 0
        ]
        median_churn = sorted(churn_values)[len(churn_values) // 2] if churn_values else 5

        for rel_fname in rel_fnames:
            stats = self._file_stats.get(rel_fname)
            if not stats:
                continue

            # Compute age metrics
            first_seen = stats.get('first_seen') or stats.get('last_modified')
            last_modified = stats.get('last_modified')
            commit_count = stats.get('commit_count', 0)

            if not first_seen or not last_modified:
                continue

            age_days = (now - first_seen).days
            quiet_days = (now - last_modified).days

            # Classification hierarchy (most specific first)
            if age_days >= PHASE_CRYSTAL_MIN_AGE_DAYS and quiet_days >= PHASE_CRYSTAL_MIN_QUIET_DAYS:
                # Old and stable
                phases[rel_fname] = "crystal"
            elif (age_days >= PHASE_ROTTING_MIN_AGE_DAYS and
                  quiet_days <= PHASE_ROTTING_MAX_QUIET_DAYS and
                  commit_count > median_churn * PHASE_ROTTING_CHURN_MULTIPLIER):
                # Old but churning - maintenance burden
                phases[rel_fname] = "rotting"
            elif age_days <= PHASE_EMERGENT_MAX_AGE_DAYS:
                # New code
                phases[rel_fname] = "emergent"
            else:
                # Normal development
                phases[rel_fname] = "evolving"

        return phases
