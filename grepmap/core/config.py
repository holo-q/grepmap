"""
Centralized configuration for GrepMap hyperparameters.

All tunable parameters are consolidated here for easy adjustment and experimentation.
This enables systematic tuning of ranking behavior, token budgets, and output density
without hunting through multiple modules.

The parameters are grouped by subsystem:
- PageRank: Graph algorithm tuning
- Boosts: Contextual importance multipliers
- Rendering: Output formatting and token management
- Extraction: Parsing limits

Import and use directly:
    from grepmap.core.config import PAGERANK_ALPHA, CHAT_FILE_BOOST
"""

# =============================================================================
# PageRank Algorithm Parameters
# =============================================================================

# Damping factor for PageRank random walk.
# 0.85 means 85% probability to follow an edge, 15% to jump to a random node
# based on personalization weights. Standard value from original PageRank paper.
PAGERANK_ALPHA = 0.85

# Depth-based personalization weights for PageRank.
# These bias the random walk toward shallower files while still allowing
# graph structure to override for genuinely important deep files.
DEPTH_WEIGHT_ROOT = 1.0        # Root/shallow files (depth <= 2)
DEPTH_WEIGHT_MODERATE = 0.5    # Moderate depth (3-4 levels)
DEPTH_WEIGHT_DEEP = 0.1        # Deep files (5+ levels)
DEPTH_WEIGHT_VENDOR = 0.01     # Vendor/third-party code (strong penalty)

# Depth thresholds (path separator count)
DEPTH_THRESHOLD_SHALLOW = 2    # At or below this = root/shallow
DEPTH_THRESHOLD_MODERATE = 4   # At or below this = moderate (else deep)

# Chat file multiplier in PageRank personalization.
# Applied multiplicatively to personalization weight for files in chat.
# Higher values make chat files more likely to be visited during random walk.
PAGERANK_CHAT_MULTIPLIER = 100.0

# Patterns indicating vendor/third-party code to penalize in ranking.
# These directories are demoted but not excluded - graph structure can
# still surface genuinely important vendor files.
VENDOR_PATTERNS = [
    'node_modules',
    'vendor',
    'third_party',
    'torchhub',
    '__pycache__',
    'site-packages'
]


# =============================================================================
# Boost Multipliers for Contextual Ranking
# =============================================================================

# Boosts are multiplicative and combine.
# Example: A mentioned identifier in a chat file gets:
#   BOOST_MENTIONED_IDENT * BOOST_CHAT_FILE = 10.0 * 20.0 = 200.0

# Boost for identifiers explicitly mentioned in conversation.
# Highest individual signal - user specifically named this symbol.
BOOST_MENTIONED_IDENT = 10.0

# Boost for files mentioned in conversation.
# Mid-level signal - user referenced this file by name.
BOOST_MENTIONED_FILE = 5.0

# Boost for files in active chat context.
# Strongest file-level signal - user is actively working on this file.
BOOST_CHAT_FILE = 20.0

# Threshold for excluding near-zero PageRank tags when exclude_unranked=True.
# Tags below this are likely disconnected graph nodes with no real importance.
EXCLUDE_UNRANKED_THRESHOLD = 0.0001

# Boost for files that frequently change together with chat files.
# Applied to "change-mates" detected by temporal coupling analysis.
# Weaker than direct chat boost but surfaces related files automatically.
BOOST_TEMPORAL_COUPLING = 3.0


# =============================================================================
# Token Budget and Rendering Parameters
# =============================================================================

# Default token budget for map output when not specified.
# 24576 tokens allows HIGH detail for typical repos (~200 tags).
DEFAULT_MAP_TOKENS = 24576

# Multiplier for map tokens when no chat files present.
# Expands the budget since there's no focused context to prioritize.
MAP_MUL_NO_FILES = 8

# Padding subtracted from max_context_window when calculating available tokens.
# Reserves space for other content in the context.
CONTEXT_WINDOW_PADDING = 1024

# Number of overflow tags to include in "also in scope" section.
# These are rendered at low resolution to extend coverage beyond detailed view.
OVERFLOW_COUNT = 2000

# Scoring weights for configuration optimization.
# Score = num_tags * COVERAGE_WEIGHT + detail_level.value * DETAIL_WEIGHT
# Current values strongly favor coverage over detail:
# 10 extra tags is worth 1 detail level increase.
SCORE_COVERAGE_WEIGHT = 10
SCORE_DETAIL_WEIGHT = 1


# =============================================================================
# Extraction Limits
# =============================================================================

# Maximum number of class fields to extract per class definition.
# Limits output size for large dataclasses while capturing key structure.
MAX_CLASS_FIELDS = 10


# =============================================================================
# Token Counting Optimization
# =============================================================================

# Character threshold below which full token counting is used.
# Above this, sampling is used for performance.
TOKEN_COUNT_FULL_THRESHOLD = 200

# Number of lines to sample for token estimation in long texts.
# Higher = more accurate but slower.
TOKEN_COUNT_SAMPLE_LINES = 100


# =============================================================================
# Stats Mode Configuration
# =============================================================================

# Maximum files to show in stats mode before truncating.
# Stats mode is designed to pack many more files than symbol mode.
STATS_MAX_FILES = 500


# =============================================================================
# Git Weighting Configuration
# =============================================================================

# Recency boost: exponential decay based on days since modification.
# Files modified today get GIT_RECENCY_MAX_BOOST, decaying to 1.0 over time.
GIT_RECENCY_DECAY_DAYS = 30.0    # Half-life in days for recency decay
GIT_RECENCY_MAX_BOOST = 2.0     # Max boost for files modified today

# Churn boost: based on number of commits touching a file.
# Files with many commits are "hotspots" indicating active development.
GIT_CHURN_THRESHOLD = 5         # Commits before churn boost kicks in
GIT_CHURN_MAX_BOOST = 1.5       # Max boost for high-churn files

# Authorship boost: applied if file was modified by current git user.
# Assumes user's own code is more contextually relevant.
GIT_AUTHOR_BOOST = 1.5          # Boost for files by current user
