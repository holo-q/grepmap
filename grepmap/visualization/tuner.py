"""
Heuristic tuning interface for the visualizer.

This module provides a pluggable system for testing ranking heuristics,
filters, and weighting mechanisms. The UI can hot-swap these components
and observe how the ranking topology shifts.

Design principle: Every ranking decision should be:
1. Named (identifiable in the UI)
2. Parameterized (tunable via sliders/toggles)
3. Observable (see effect on gini, cliff, rank distribution)

Components:
- RankingPreset: Named bundle of heuristic settings
- TunerState: Current active configuration
- HeuristicRegistry: Available heuristics for injection
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class HeuristicType(Enum):
    """Categories of heuristics for UI grouping."""
    BOOST = "boost"  # Multiplicative rank adjustments
    FILTER = "filter"  # Node/edge filtering
    WEIGHT = "weight"  # Edge weight modifiers
    PERSONALIZATION = "personalization"  # PageRank personalization


@dataclass
class HeuristicParam:
    """A tunable parameter for a heuristic."""
    name: str
    label: str  # Human-readable label
    param_type: str  # "float", "int", "bool", "choice"
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None
    description: str = ""


@dataclass
class Heuristic:
    """A named, tunable heuristic that affects ranking.

    Each heuristic has:
    - An identifier (for serialization/API)
    - Display metadata (label, description, category)
    - Parameters (what can be tuned)
    - An apply function (how it modifies ranking)
    """
    id: str
    label: str
    heuristic_type: HeuristicType
    params: List[HeuristicParam] = field(default_factory=list)
    description: str = ""
    enabled_by_default: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.heuristic_type.value,
            "params": [asdict(p) for p in self.params],
            "description": self.description,
            "enabled_by_default": self.enabled_by_default
        }


@dataclass
class TunerState:
    """Current tuner configuration state.

    Tracks which heuristics are enabled and their parameter values.
    This is what gets serialized to/from the API.
    """
    # Enabled heuristics with their param overrides
    enabled: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Current focus query
    focus: Optional[str] = None

    # Preset name if using a named preset
    preset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "focus": self.focus,
            "preset": self.preset
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TunerState":
        return cls(
            enabled=data.get("enabled", {}),
            focus=data.get("focus"),
            preset=data.get("preset")
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TunerState":
        return cls.from_dict(json.loads(json_str))


@dataclass
class RankingPreset:
    """Named bundle of heuristic settings.

    Presets let users quickly switch between configurations:
    - "default": Standard ranking behavior
    - "debug": Favor recency, boost callers
    - "refactor": Favor churn, highlight bridges
    - "minimal": Disable all boosts, see raw PageRank
    """
    id: str
    label: str
    description: str
    state: TunerState


# Registry of available heuristics
# These map to actual config values and boost mechanisms in grepmap
HEURISTICS: List[Heuristic] = [
    # Boost heuristics
    Heuristic(
        id="depth_weight",
        label="Depth Weight",
        heuristic_type=HeuristicType.PERSONALIZATION,
        description="Weight files by directory depth (root files > deep files)",
        params=[
            HeuristicParam("root", "Root weight", "float", 1.0, 0.0, 5.0, 0.1),
            HeuristicParam("moderate", "Moderate depth", "float", 0.5, 0.0, 2.0, 0.1),
            HeuristicParam("deep", "Deep weight", "float", 0.25, 0.0, 1.0, 0.05),
            HeuristicParam("vendor", "Vendor weight", "float", 0.01, 0.0, 0.5, 0.01),
        ]
    ),
    Heuristic(
        id="chat_boost",
        label="Focus/Chat Boost",
        heuristic_type=HeuristicType.BOOST,
        description="Boost files mentioned in focus query or chat",
        params=[
            HeuristicParam("multiplier", "Boost multiplier", "float", 20.0, 1.0, 100.0, 1.0),
        ]
    ),
    Heuristic(
        id="git_recency",
        label="Git Recency",
        heuristic_type=HeuristicType.BOOST,
        description="Boost recently modified files",
        params=[
            HeuristicParam("enabled", "Enable", "bool", False),
            HeuristicParam("max_boost", "Max boost", "float", 2.0, 1.0, 10.0, 0.1),
            HeuristicParam("decay_days", "Decay days", "int", 30, 1, 365, 1),
        ]
    ),
    Heuristic(
        id="git_churn",
        label="Git Churn",
        heuristic_type=HeuristicType.BOOST,
        description="Boost files with high commit frequency",
        params=[
            HeuristicParam("enabled", "Enable", "bool", False),
            HeuristicParam("max_boost", "Max boost", "float", 1.5, 1.0, 5.0, 0.1),
            HeuristicParam("threshold", "Min commits", "int", 5, 1, 50, 1),
        ]
    ),
    Heuristic(
        id="temporal_coupling",
        label="Temporal Coupling",
        heuristic_type=HeuristicType.BOOST,
        description="Boost files that change together with focus files",
        params=[
            HeuristicParam("enabled", "Enable", "bool", True),
            HeuristicParam("boost", "Coupling boost", "float", 1.5, 1.0, 5.0, 0.1),
        ]
    ),
    Heuristic(
        id="caller_boost",
        label="Caller Boost (DEBUG)",
        heuristic_type=HeuristicType.BOOST,
        description="Boost files that call the focus symbols (reverse edge tracing)",
        params=[
            HeuristicParam("enabled", "Enable", "bool", True),
            HeuristicParam("bias", "Reverse edge bias", "float", 2.0, 1.0, 10.0, 0.1),
        ]
    ),
    Heuristic(
        id="mentioned_idents",
        label="Mentioned Identifiers",
        heuristic_type=HeuristicType.BOOST,
        description="Boost symbols matching mentioned identifiers",
        params=[
            HeuristicParam("boost", "Identifier boost", "float", 5.0, 1.0, 20.0, 0.5),
        ]
    ),

    # Filter heuristics
    Heuristic(
        id="exclude_unranked",
        label="Exclude Low-Rank",
        heuristic_type=HeuristicType.FILTER,
        description="Filter out symbols below rank threshold",
        params=[
            HeuristicParam("enabled", "Enable", "bool", False),
            HeuristicParam("threshold", "Rank threshold", "float", 0.001, 0.0, 0.1, 0.001),
        ]
    ),
    Heuristic(
        id="vendor_filter",
        label="Vendor Filter",
        heuristic_type=HeuristicType.FILTER,
        description="De-prioritize vendor/third-party code",
        params=[
            HeuristicParam("enabled", "Enable", "bool", True),
            HeuristicParam("patterns", "Patterns (comma-sep)", "text",
                          "node_modules,vendor,third_party,.venv,dist,build"),
        ]
    ),
    Heuristic(
        id="orphan_filter",
        label="Orphan Filter",
        heuristic_type=HeuristicType.FILTER,
        description="Hide symbols with no incoming references",
        params=[
            HeuristicParam("enabled", "Enable", "bool", False),
        ]
    ),

    # Weight heuristics
    Heuristic(
        id="pagerank_alpha",
        label="PageRank Alpha",
        heuristic_type=HeuristicType.WEIGHT,
        description="Damping factor for PageRank (higher = more diffusion)",
        params=[
            HeuristicParam("alpha", "Alpha", "float", 0.85, 0.1, 0.99, 0.01),
        ]
    ),
    Heuristic(
        id="symbol_vs_file",
        label="Symbol vs File Ranking",
        heuristic_type=HeuristicType.WEIGHT,
        description="Use symbol-level (fine-grained) vs file-level ranking",
        params=[
            HeuristicParam("symbol_rank", "Symbol-level", "bool", True),
        ]
    ),
]


# Presets for quick switching
PRESETS: List[RankingPreset] = [
    RankingPreset(
        id="default",
        label="Default",
        description="Standard ranking: symbol-level, depth-weighted, focus boost",
        state=TunerState(
            enabled={
                "depth_weight": {"root": 1.0, "moderate": 0.5, "deep": 0.25, "vendor": 0.01},
                "chat_boost": {"multiplier": 20.0},
                "temporal_coupling": {"enabled": True, "boost": 1.5},
                "vendor_filter": {"enabled": True},
                "pagerank_alpha": {"alpha": 0.85},
                "symbol_vs_file": {"symbol_rank": True},
            }
        )
    ),
    RankingPreset(
        id="minimal",
        label="Minimal (Raw PageRank)",
        description="Disable all boosts, see pure graph structure",
        state=TunerState(
            enabled={
                "pagerank_alpha": {"alpha": 0.85},
                "symbol_vs_file": {"symbol_rank": True},
            }
        )
    ),
    RankingPreset(
        id="debug",
        label="Debug Mode",
        description="Favor recency, boost callers of focus symbols",
        state=TunerState(
            enabled={
                "depth_weight": {"root": 1.0, "moderate": 0.5, "deep": 0.25, "vendor": 0.01},
                "chat_boost": {"multiplier": 30.0},
                "git_recency": {"enabled": True, "max_boost": 3.0, "decay_days": 14},
                "caller_boost": {"enabled": True, "bias": 3.0},
                "pagerank_alpha": {"alpha": 0.85},
                "symbol_vs_file": {"symbol_rank": True},
            }
        )
    ),
    RankingPreset(
        id="refactor",
        label="Refactor Mode",
        description="Surface high-churn files and bridges",
        state=TunerState(
            enabled={
                "depth_weight": {"root": 1.0, "moderate": 0.5, "deep": 0.25, "vendor": 0.01},
                "chat_boost": {"multiplier": 10.0},
                "git_churn": {"enabled": True, "max_boost": 2.5, "threshold": 3},
                "temporal_coupling": {"enabled": True, "boost": 2.0},
                "pagerank_alpha": {"alpha": 0.85},
                "symbol_vs_file": {"symbol_rank": True},
            }
        )
    ),
    RankingPreset(
        id="explore",
        label="Explore Mode",
        description="Neutral weights, good for orientation",
        state=TunerState(
            enabled={
                "depth_weight": {"root": 0.8, "moderate": 0.6, "deep": 0.4, "vendor": 0.1},
                "chat_boost": {"multiplier": 5.0},
                "vendor_filter": {"enabled": True},
                "pagerank_alpha": {"alpha": 0.85},
                "symbol_vs_file": {"symbol_rank": True},
            }
        )
    ),
]


class HeuristicRegistry:
    """Registry of available heuristics for the tuner.

    Provides:
    - List of all available heuristics with metadata
    - Preset configurations for quick switching
    - Validation of tuner state
    """

    def __init__(self):
        self.heuristics = {h.id: h for h in HEURISTICS}
        self.presets = {p.id: p for p in PRESETS}

    def get_all_heuristics(self) -> List[Dict[str, Any]]:
        """Get all heuristics as serializable dicts."""
        return [h.to_dict() for h in self.heuristics.values()]

    def get_all_presets(self) -> List[Dict[str, Any]]:
        """Get all presets as serializable dicts."""
        return [
            {
                "id": p.id,
                "label": p.label,
                "description": p.description,
                "state": p.state.to_dict()
            }
            for p in self.presets.values()
        ]

    def get_preset(self, preset_id: str) -> Optional[TunerState]:
        """Get a preset's state by ID."""
        preset = self.presets.get(preset_id)
        return preset.state if preset else None

    def validate_state(self, state: TunerState) -> List[str]:
        """Validate a tuner state, return list of warnings."""
        warnings = []
        for heuristic_id in state.enabled:
            if heuristic_id not in self.heuristics:
                warnings.append(f"Unknown heuristic: {heuristic_id}")
        return warnings

    def get_api_schema(self) -> Dict[str, Any]:
        """Get full schema for API consumption."""
        return {
            "heuristics": self.get_all_heuristics(),
            "presets": self.get_all_presets(),
            "default_preset": "default"
        }


def apply_tuner_state_to_config(state: TunerState) -> Dict[str, Any]:
    """Convert TunerState to grepmap configuration overrides.

    This bridges the tuner UI to actual grepmap behavior.
    Returns a dict that can be passed to GrepMap constructor or
    used to override core.config values.

    The mapping:
    - depth_weight → DEPTH_WEIGHT_* constants
    - chat_boost → BOOST_CHAT_FILE
    - git_recency → GIT_RECENCY_* and git_weight flag
    - git_churn → GIT_CHURN_* and git_weight flag
    - pagerank_alpha → PAGERANK_ALPHA
    - etc.
    """
    config = {}

    # Depth weights
    if "depth_weight" in state.enabled:
        params = state.enabled["depth_weight"]
        config["DEPTH_WEIGHT_ROOT"] = params.get("root", 1.0)
        config["DEPTH_WEIGHT_MODERATE"] = params.get("moderate", 0.5)
        config["DEPTH_WEIGHT_DEEP"] = params.get("deep", 0.25)
        config["DEPTH_WEIGHT_VENDOR"] = params.get("vendor", 0.01)

    # Chat/focus boost
    if "chat_boost" in state.enabled:
        params = state.enabled["chat_boost"]
        config["BOOST_CHAT_FILE"] = params.get("multiplier", 20.0)

    # Git recency
    if "git_recency" in state.enabled:
        params = state.enabled["git_recency"]
        if params.get("enabled", False):
            config["git_weight"] = True
            config["GIT_RECENCY_MAX_BOOST"] = params.get("max_boost", 2.0)
            config["GIT_RECENCY_DECAY_DAYS"] = params.get("decay_days", 30)

    # Git churn
    if "git_churn" in state.enabled:
        params = state.enabled["git_churn"]
        if params.get("enabled", False):
            config["git_weight"] = True
            config["GIT_CHURN_MAX_BOOST"] = params.get("max_boost", 1.5)
            config["GIT_CHURN_THRESHOLD"] = params.get("threshold", 5)

    # PageRank alpha
    if "pagerank_alpha" in state.enabled:
        params = state.enabled["pagerank_alpha"]
        config["PAGERANK_ALPHA"] = params.get("alpha", 0.85)

    # Symbol vs file ranking
    if "symbol_vs_file" in state.enabled:
        params = state.enabled["symbol_vs_file"]
        config["symbol_rank"] = params.get("symbol_rank", True)

    # Caller boost (reverse edge bias)
    if "caller_boost" in state.enabled:
        params = state.enabled["caller_boost"]
        if params.get("enabled", False):
            config["reverse_edge_bias"] = params.get("bias", 2.0)

    # Exclude unranked
    if "exclude_unranked" in state.enabled:
        params = state.enabled["exclude_unranked"]
        config["exclude_unranked"] = params.get("enabled", False)
        config["EXCLUDE_UNRANKED_THRESHOLD"] = params.get("threshold", 0.001)

    return config
