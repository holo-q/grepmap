"""
Ultra-dense diagnostic output for ranking system introspection.

NOT for human consumption. Designed for maximum information density
so an LLM can quickly parse and understand the full state of the
ranking machine. Every character earns its place.

Output format (single line per section, pipe-separated):
  G:nodes/edges/density% hub:sym:in,... orph:N
  R:min/p50/p90/max gini:X cliff@pNN
  B:stage→stage*mult chain:Nx_max
  HP:α/decay/chat/vend/churn/rec
  TOP:sym:rank:refs↓,...
  GIT:file:boost,...
  TOK:budget/used/fill% detail:LVL tags:N/total
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from grepmap.core.types import RankedTag, DetailLevel


@dataclass
class DiagnosticData:
    """Raw diagnostic data collected during ranking."""
    # Graph stats
    num_symbols: int = 0
    num_edges: int = 0
    hub_symbols: List[Tuple[str, int]] = None  # (name, in_degree)
    orphan_count: int = 0

    # Rank distribution
    ranks: List[float] = None

    # Boost breakdown
    base_ranks: Dict[str, float] = None  # Before boosts
    symbol_multiplier: float = 1.0
    git_max_boost: float = 1.0
    git_boosted_files: List[Tuple[str, float]] = None
    focus_boost: float = 1.0

    # HP values (snapshot)
    hp_alpha: float = 0.85
    hp_decay: float = 30.0
    hp_chat_boost: float = 20.0
    hp_vendor_weight: float = 0.01
    hp_churn_thresh: int = 5
    hp_recency_max: float = 2.0

    # Token budget
    token_budget: int = 0
    tokens_used: int = 0
    detail_level: str = "?"
    tags_selected: int = 0
    tags_total: int = 0

    def __post_init__(self):
        if self.hub_symbols is None:
            self.hub_symbols = []
        if self.ranks is None:
            self.ranks = []
        if self.base_ranks is None:
            self.base_ranks = {}
        if self.git_boosted_files is None:
            self.git_boosted_files = []


def compute_gini(values: List[float]) -> float:
    """Gini coefficient for rank concentration. 0=equal, 1=concentrated."""
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumsum) / (n * sum(sorted_vals)) - (n + 1) / n


def find_cliff(ranks: List[float], threshold: float = 0.5) -> Optional[int]:
    """Find percentile where rank drops by >threshold from previous."""
    if len(ranks) < 10:
        return None
    sorted_ranks = sorted(ranks, reverse=True)
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i-1] > 0 and sorted_ranks[i] / sorted_ranks[i-1] < threshold:
            return int(100 * i / len(sorted_ranks))
    return None


def percentile(values: List[float], p: int) -> float:
    """Get percentile value."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def format_diagnostic(data: DiagnosticData) -> str:
    """Format diagnostic data as ultra-dense single output."""
    lines = []

    # G: Graph structure
    density = 0.0
    if data.num_symbols > 1:
        max_edges = data.num_symbols * (data.num_symbols - 1)
        density = 100 * data.num_edges / max_edges if max_edges > 0 else 0

    hubs = ",".join(f"{name}:{deg}" for name, deg in data.hub_symbols[:5])
    lines.append(f"G:{data.num_symbols}n/{data.num_edges}e/{density:.1f}%d hub:{hubs} orph:{data.orphan_count}")

    # R: Rank distribution
    if data.ranks:
        r = data.ranks
        gini = compute_gini(r)
        cliff = find_cliff(r)
        cliff_str = f" cliff@p{cliff}" if cliff else ""
        lines.append(f"R:{min(r):.4f}/{percentile(r,50):.4f}/{percentile(r,90):.4f}/{max(r):.4f} gini:{gini:.2f}{cliff_str}")

    # B: Boost chain
    chain_max = data.symbol_multiplier * data.git_max_boost * data.focus_boost
    lines.append(f"B:pr→sym*{data.symbol_multiplier:.1f}→git*{data.git_max_boost:.1f}→foc*{data.focus_boost:.0f} chain:{chain_max:.0f}x")

    # HP: Hyperparameters
    lines.append(f"HP:α{data.hp_alpha}/dec{data.hp_decay:.0f}/chat{data.hp_chat_boost:.0f}/vend{data.hp_vendor_weight}/churn{data.hp_churn_thresh}/rec{data.hp_recency_max}")

    # GIT: Top git-boosted files
    if data.git_boosted_files:
        git_top = ",".join(f"{f.split('/')[-1]}:{b:.2f}" for f, b in data.git_boosted_files[:5])
        lines.append(f"GIT:{git_top}")

    # TOK: Token budget
    fill = 100 * data.tokens_used / data.token_budget if data.token_budget > 0 else 0
    lines.append(f"TOK:{data.token_budget}/{data.tokens_used}/{fill:.0f}% {data.detail_level} tags:{data.tags_selected}/{data.tags_total}")

    return " | ".join(lines)


def format_top_symbols(ranked_tags: List[RankedTag], symbol_refs: Dict[str, int], n: int = 10) -> str:
    """Format top N symbols with ranks and ref counts."""
    seen = set()
    parts = []
    for rt in ranked_tags:
        if rt.tag.name in seen:
            continue
        seen.add(rt.tag.name)
        refs = symbol_refs.get((rt.tag.rel_fname, rt.tag.name), 0)
        # Truncate long names
        name = rt.tag.name[:12]
        parts.append(f"{name}:{rt.rank:.3f}:{refs}↓")
        if len(parts) >= n:
            break
    return "TOP:" + ",".join(parts)


def collect_diagnostic_data(
    num_symbols: int,
    num_edges: int,
    hub_symbols: List[Tuple[str, int]],
    orphan_count: int,
    ranked_tags: List[RankedTag],
    git_weights: Optional[Dict[str, float]],
    token_budget: int,
    tokens_used: int,
    detail_level: DetailLevel,
    tags_selected: int
) -> DiagnosticData:
    """Collect all diagnostic data into structured object."""
    from grepmap.core.config import (
        PAGERANK_ALPHA, GIT_RECENCY_DECAY_DAYS, BOOST_CHAT_FILE,
        DEPTH_WEIGHT_VENDOR, GIT_CHURN_THRESHOLD, GIT_RECENCY_MAX_BOOST
    )

    data = DiagnosticData()
    data.num_symbols = num_symbols
    data.num_edges = num_edges
    data.hub_symbols = hub_symbols
    data.orphan_count = orphan_count

    data.ranks = [rt.rank for rt in ranked_tags]
    data.tags_total = len(ranked_tags)
    data.tags_selected = tags_selected

    # Git stats
    if git_weights:
        sorted_git = sorted(git_weights.items(), key=lambda x: x[1], reverse=True)
        data.git_boosted_files = [(f, w) for f, w in sorted_git if w > 1.0][:10]
        data.git_max_boost = max(git_weights.values()) if git_weights else 1.0

    # HP snapshot
    data.hp_alpha = PAGERANK_ALPHA
    data.hp_decay = GIT_RECENCY_DECAY_DAYS
    data.hp_chat_boost = BOOST_CHAT_FILE
    data.hp_vendor_weight = DEPTH_WEIGHT_VENDOR
    data.hp_churn_thresh = GIT_CHURN_THRESHOLD
    data.hp_recency_max = GIT_RECENCY_MAX_BOOST

    # Token stats
    data.token_budget = token_budget
    data.tokens_used = tokens_used
    data.detail_level = detail_level.name if detail_level else "?"

    return data
