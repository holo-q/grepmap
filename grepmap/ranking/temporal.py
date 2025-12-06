"""
Temporal coupling detection for co-changing file analysis.

This module identifies files that frequently change together in git history,
revealing implicit dependencies and architectural coupling that may not be
visible in static import graphs.

The temporal coupling score uses Jaccard similarity:
    coupling(A, B) = |commits(A) ∩ commits(B)| / |commits(A) ∪ commits(B)|

Files that change together often (score > 0.3) are likely coupled in ways
that matter for understanding changes - bug fixes, feature additions, or
refactorings that span both files.

Design note: This complements static graph analysis by surfacing "change-mates"
that evolve together, helping developers understand which files they should
examine together when making modifications.
"""

import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Callable


class TemporalCoupling:
    """Detect files that frequently change together in git history.

    Analyzes git commit history to build a co-occurrence matrix of files
    that appear in the same commits, then computes Jaccard similarity to
    identify significant coupling relationships.

    The coupling score ranges from 0.0 (never change together) to 1.0
    (always change together). We threshold at 0.3 to filter noise and
    focus on meaningful relationships.

    Limits to last 500 commits for performance in large repositories.
    """

    def __init__(
        self,
        root: Path,
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize TemporalCoupling analyzer.

        Args:
            root: Repository root path (must contain .git)
            verbose: Enable verbose logging
            output_handler: Function for info messages (default: print)
        """
        self.root = root
        self.verbose = verbose
        self.output_handler = output_handler or print

        # Cache for diagnostic data
        self._total_commits_analyzed = 0
        self._file_to_commits: Dict[str, Set[int]] = defaultdict(set)
        self._commit_sizes: List[int] = []

    def compute_coupling(self, rel_fnames: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Compute temporal coupling scores for files.

        For each file, returns a list of other files it frequently changes
        with, sorted by coupling strength (highest first).

        Args:
            rel_fnames: List of relative file paths to analyze

        Returns:
            Dict mapping each file to list of (coupled_file, score) tuples,
            filtered to coupling >= 0.3 and sorted by score descending
        """
        if not self._is_git_repo():
            if self.verbose:
                self.output_handler("Not a git repository, skipping temporal coupling")
            return {f: [] for f in rel_fnames}

        # Build co-occurrence matrix from git history
        self._analyze_git_history(rel_fnames)

        # Compute Jaccard similarity for all file pairs
        coupling_scores = self._compute_jaccard_similarities(rel_fnames)

        if self.verbose:
            self._log_coupling_stats(coupling_scores)

        return coupling_scores

    def get_diagnostic_data(self) -> dict:
        """Return diagnostic data for introspection.

        Useful for debugging and understanding the coupling analysis.

        Returns:
            Dict with keys:
                - total_commits: Number of commits analyzed
                - files_tracked: Number of files found in history
                - avg_commit_size: Average files per commit
                - file_commit_counts: Dict of file -> number of commits
        """
        return {
            'total_commits': self._total_commits_analyzed,
            'files_tracked': len(self._file_to_commits),
            'avg_commit_size': (
                sum(self._commit_sizes) / len(self._commit_sizes)
                if self._commit_sizes else 0
            ),
            'file_commit_counts': {
                f: len(commits) for f, commits in self._file_to_commits.items()
            }
        }

    def _is_git_repo(self) -> bool:
        """Check if root is a git repository."""
        return (self.root / '.git').exists()

    def _analyze_git_history(self, rel_fnames: List[str]):
        """Parse git log to build file co-occurrence matrix.

        Uses git log --name-only to get all commits and the files they touch.
        Each commit becomes a set of files, and we track which commits touch
        which files to compute coupling later.
        """
        # Fetch git log with file names
        # Format: commit hash followed by changed files
        try:
            result = subprocess.run(
                [
                    'git', 'log',
                    '--format=%H',  # Commit hash only
                    '--name-only',
                    '--all',
                    '-n', '500',  # Last 500 commits for performance
                    '--',
                ],
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

        # Parse output: alternating commit hash and file list
        lines = result.stdout.strip().split('\n')

        commit_idx = 0
        current_commit_files: Set[str] = set()
        rel_fnames_set = set(rel_fnames)

        for line in lines:
            line = line.strip()
            if not line:
                # Empty line between commits - save current commit
                if current_commit_files:
                    # Record which files appear in this commit
                    for fname in current_commit_files:
                        self._file_to_commits[fname].add(commit_idx)

                    self._commit_sizes.append(len(current_commit_files))
                    commit_idx += 1
                    current_commit_files = set()
            elif len(line) == 40 and all(c in '0123456789abcdef' for c in line):
                # Commit hash line - just skip it
                continue
            else:
                # File path line
                # Only track files we care about
                if line in rel_fnames_set or any(line.endswith(f) for f in rel_fnames):
                    # Normalize to match rel_fnames
                    for rf in rel_fnames:
                        if line == rf or line.endswith(rf):
                            current_commit_files.add(rf)
                            break

        # Don't forget last commit
        if current_commit_files:
            for fname in current_commit_files:
                self._file_to_commits[fname].add(commit_idx)
            self._commit_sizes.append(len(current_commit_files))

        self._total_commits_analyzed = commit_idx

    def _compute_jaccard_similarities(
        self,
        rel_fnames: List[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Compute Jaccard similarity for all file pairs.

        Jaccard similarity = |A ∩ B| / |A ∪ B|
        Where A and B are the sets of commits touching each file.

        Returns coupling scores >= 0.3 threshold, sorted descending.
        """
        coupling_threshold = 0.3
        results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Only compute for files that appear in history
        tracked_files = [f for f in rel_fnames if f in self._file_to_commits]

        # Pairwise comparison - O(n²) but only on tracked files
        for i, file_a in enumerate(tracked_files):
            commits_a = self._file_to_commits[file_a]

            for file_b in tracked_files[i + 1:]:
                commits_b = self._file_to_commits[file_b]

                # Jaccard similarity
                intersection = len(commits_a & commits_b)
                union = len(commits_a | commits_b)

                if union == 0:
                    continue

                score = intersection / union

                if score >= coupling_threshold:
                    results[file_a].append((file_b, score))
                    results[file_b].append((file_a, score))

        # Sort each file's coupling list by score descending
        for fname in results:
            results[fname].sort(key=lambda x: x[1], reverse=True)

        # Ensure all input files have entries (even if empty)
        for fname in rel_fnames:
            if fname not in results:
                results[fname] = []

        return dict(results)

    def _log_coupling_stats(self, coupling_scores: Dict[str, List[Tuple[str, float]]]):
        """Log temporal coupling statistics for debugging."""
        total_files = len(coupling_scores)
        files_with_coupling = sum(1 for pairs in coupling_scores.values() if pairs)
        total_pairs = sum(len(pairs) for pairs in coupling_scores.values()) // 2  # Each pair counted twice

        self.output_handler(
            f"Temporal coupling: {files_with_coupling}/{total_files} files coupled, "
            f"{total_pairs} pairs above 0.3 threshold"
        )

        # Show top coupled file pairs
        all_pairs = []
        for file_a, pairs in coupling_scores.items():
            for file_b, score in pairs:
                if (file_b, file_a, score) not in all_pairs:  # Avoid duplicates
                    all_pairs.append((file_a, file_b, score))

        all_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = all_pairs[:5]

        if top_pairs:
            self.output_handler("Top temporally coupled pairs:")
            for file_a, file_b, score in top_pairs:
                self.output_handler(f"  {file_a} <-> {file_b}: {score:.2f}")
