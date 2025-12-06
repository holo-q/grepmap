"""
Ranking module for RepoMapper.

This module provides the ranking infrastructure for computing file and tag
importance scores. It consists of six main components:

1. PageRanker: Computes base PageRank scores using file-level graph
2. SymbolRanker: Computes symbol-level PageRank for fine-grained "tree shaking"
3. BoostCalculator: Applies contextual boosts (chat files, mentioned symbols)
4. GitWeightCalculator: Applies temporal boosts (recency, churn, authorship)
5. Optimizer: Finds optimal rendering configuration within token budget
6. FocusResolver: Resolves --focus targets (paths or queries) to weighted files

Symbol-level ranking enables showing only the used functions from a large file,
rather than ranking the entire file uniformly. Git weighting adds temporal
awareness to favor recently modified code.
"""

from grepmap.ranking.pagerank import PageRanker
from grepmap.ranking.symbols import SymbolRanker, get_symbol_ranks_for_file
from grepmap.ranking.boosts import BoostCalculator
from grepmap.ranking.git_weight import GitWeightCalculator
from grepmap.ranking.optimizer import Optimizer
from grepmap.ranking.focus import FocusResolver

__all__ = [
    'PageRanker', 'SymbolRanker', 'get_symbol_ranks_for_file',
    'BoostCalculator', 'GitWeightCalculator', 'Optimizer', 'FocusResolver'
]
