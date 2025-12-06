"""
Symbol story cards - per-symbol git history extraction.

This module tracks the evolution of individual symbols (functions, classes)
through git history. Each symbol gets a "story card" showing:
- When it was created
- How many times it was modified
- Recent commits touching it
- Who has worked on it

This enables understanding code at the symbol level rather than file level,
answering questions like "who wrote this function?" and "when was it last changed?"

Design rationale:
- Uses git log -L for line-based history (tracks function evolution)
- Caches results since git operations are expensive
- Limits to recent history (last 50 commits) for performance
- Groups authorship for quick understanding of code ownership

Integration point: Called from diagnostics to show symbol-level insights.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime


@dataclass
class SymbolCommit:
    """A single commit that touched a symbol."""
    hash: str
    author: str
    date: str
    message: str


@dataclass
class SymbolStory:
    """Story card for a symbol's evolution.

    Captures the git history of a specific symbol (function/class),
    providing insights into its creation, modification, and ownership.

    Attributes:
        symbol_name: Name of the symbol
        rel_fname: Relative file path containing the symbol
        line_start: Starting line of the symbol
        line_end: Ending line of the symbol
        created_date: When the symbol was first committed
        created_by: Author of the first commit
        total_commits: Number of commits touching this symbol
        recent_commits: List of recent commits (most recent first)
        authors: Dict mapping author names to commit counts
    """
    symbol_name: str
    rel_fname: str
    line_start: int
    line_end: int
    created_date: Optional[str] = None
    created_by: Optional[str] = None
    total_commits: int = 0
    recent_commits: List[SymbolCommit] = field(default_factory=list)
    authors: Dict[str, int] = field(default_factory=dict)

    @property
    def primary_author(self) -> Optional[str]:
        """The author with the most commits to this symbol."""
        if not self.authors:
            return None
        return max(self.authors.items(), key=lambda x: x[1])[0]

    @property
    def age_days(self) -> Optional[int]:
        """Days since the symbol was created."""
        if not self.created_date:
            return None
        try:
            created = datetime.fromisoformat(self.created_date.replace('Z', '+00:00'))
            now = datetime.now(created.tzinfo)
            return (now - created).days
        except (ValueError, TypeError):
            return None


class SymbolStoryExtractor:
    """Extract git history for individual symbols.

    Uses git log -L to trace line-based history of symbols.
    This is more expensive than file-level git operations but provides
    precise symbol-level evolution tracking.

    Performance considerations:
    - Git log -L can be slow for large files/long histories
    - We limit to 50 commits per symbol for practicality
    - Results should be cached at the caller level
    """

    # Maximum commits to retrieve per symbol
    MAX_COMMITS = 50

    def __init__(
        self,
        root: Path,
        verbose: bool = False,
        output_handler: Optional[Callable[[str], None]] = None
    ):
        """Initialize SymbolStoryExtractor.

        Args:
            root: Repository root path
            verbose: Enable verbose logging
            output_handler: Function for info messages
        """
        self.root = root
        self.verbose = verbose
        self.output_handler = output_handler or print

    def get_story(
        self,
        symbol_name: str,
        rel_fname: str,
        line_start: int,
        line_end: int
    ) -> Optional[SymbolStory]:
        """Get the story card for a symbol.

        Uses git log -L to extract line-based history for the symbol's
        line range. Note: git -L tracks the function even as it moves
        through refactoring.

        Args:
            symbol_name: Name of the symbol
            rel_fname: Relative file path
            line_start: Starting line number (1-indexed)
            line_end: Ending line number (1-indexed)

        Returns:
            SymbolStory with git history, or None if extraction fails
        """
        if not self._is_git_repo():
            return None

        story = SymbolStory(
            symbol_name=symbol_name,
            rel_fname=rel_fname,
            line_start=line_start,
            line_end=line_end
        )

        # Use git log -L for line-based history
        # Format: -L<start>,<end>:<file>
        line_spec = f"{line_start},{line_end}:{rel_fname}"

        try:
            result = subprocess.run(
                [
                    'git', 'log',
                    '-L', line_spec,
                    '--no-patch',  # Skip diff output
                    '--format=%H%n%an%n%aI%n%s',  # hash, author, date, subject
                    f'-n{self.MAX_COMMITS}'
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            if self.verbose:
                self.output_handler(f"Git log failed for {symbol_name}")
            return None

        if result.returncode != 0:
            # git log -L might fail if file/lines don't exist
            if self.verbose:
                self.output_handler(f"Git log returned error for {symbol_name}")
            return None

        # Parse output: each commit is 4 lines (hash, author, date, subject)
        commits = self._parse_log_output(result.stdout)

        if not commits:
            return story

        story.total_commits = len(commits)
        story.recent_commits = commits[:10]  # Keep only 10 most recent

        # Track authorship
        for commit in commits:
            story.authors[commit.author] = story.authors.get(commit.author, 0) + 1

        # First commit (oldest) = creation
        oldest = commits[-1]
        story.created_date = oldest.date
        story.created_by = oldest.author

        return story

    def get_stories_for_file(
        self,
        rel_fname: str,
        symbols: List[Tuple[str, int, int]]
    ) -> Dict[str, SymbolStory]:
        """Get story cards for multiple symbols in a file.

        Batch operation for efficiency when processing a whole file.

        Args:
            rel_fname: Relative file path
            symbols: List of (symbol_name, line_start, line_end) tuples

        Returns:
            Dict mapping symbol_name to SymbolStory
        """
        stories = {}
        for symbol_name, line_start, line_end in symbols:
            story = self.get_story(symbol_name, rel_fname, line_start, line_end)
            if story:
                stories[symbol_name] = story
        return stories

    def _is_git_repo(self) -> bool:
        """Check if root is a git repository."""
        return (self.root / '.git').exists()

    def _parse_log_output(self, output: str) -> List[SymbolCommit]:
        """Parse git log output into SymbolCommit objects.

        Expected format (4 lines per commit):
        - Line 1: commit hash
        - Line 2: author name
        - Line 3: ISO date
        - Line 4: commit subject
        """
        lines = output.strip().split('\n')
        commits = []

        i = 0
        while i + 3 < len(lines):
            try:
                commit = SymbolCommit(
                    hash=lines[i].strip(),
                    author=lines[i + 1].strip(),
                    date=lines[i + 2].strip(),
                    message=lines[i + 3].strip()
                )
                # Validate hash format
                if len(commit.hash) == 40 and all(c in '0123456789abcdef' for c in commit.hash):
                    commits.append(commit)
                i += 4
            except (IndexError, ValueError):
                i += 1
                continue

        return commits

    def format_story_card(self, story: SymbolStory) -> str:
        """Format a symbol story as a compact string for display.

        Output format:
        symbol_name: created 2024-01-15 by Author, 12 commits, 3 authors
        """
        parts = [f"{story.symbol_name}:"]

        if story.created_date:
            # Extract just the date part
            date_part = story.created_date[:10] if story.created_date else "?"
            parts.append(f"created {date_part}")

            if story.created_by:
                parts.append(f"by {story.created_by}")

        if story.total_commits:
            parts.append(f"{story.total_commits} commits")

        if len(story.authors) > 1:
            parts.append(f"{len(story.authors)} authors")
        elif story.primary_author and story.created_by != story.primary_author:
            parts.append(f"maintained by {story.primary_author}")

        return " ".join(parts)
