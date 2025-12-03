"""
Utility functions for RepoMap.
"""

import sys
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install with: pip install tiktoken")
    sys.exit(1)


# =============================================================================
# Detail Level for Multi-Configuration Optimization
# =============================================================================

class DetailLevel(IntEnum):
    """Rendering detail level for output optimization.

    Higher values = more detail = more tokens consumed.
    The optimizer tries to maximize (coverage * detail) within token budget.
    """
    LOW = 1      # Symbol names only: "connect, disconnect, refresh"
    MEDIUM = 2   # Names + simplified types: "connect(self, hobo, remote)"
    HIGH = 3     # Full signatures: "connect(self, hobo: HoboWindow) -> bool"


# =============================================================================
# Signature and Field Information for Variable-Detail Rendering
# =============================================================================

@dataclass(frozen=True)
class SignatureInfo:
    """Parsed function/method signature for detail-level rendering.

    Extracted during tag parsing, cached alongside Tag.
    Enables rendering at different detail levels without re-parsing.
    """
    parameters: Tuple[Tuple[str, Optional[str]], ...]  # ((name, type_annotation), ...)
    return_type: Optional[str]
    decorators: Tuple[str, ...]  # ("staticmethod", "property", ...)

    def render(self, detail: DetailLevel, seen_patterns: Optional[set] = None) -> str:
        """Render signature at specified detail level with optional deduplication.

        Args:
            detail: The level of detail to render
            seen_patterns: Optional set of "name:type" patterns already shown.
                           If provided, types are elided for seen patterns.
        """
        if detail == DetailLevel.LOW:
            return "..."

        params = []
        for name, typ in self.parameters:
            if detail == DetailLevel.MEDIUM or not typ:
                params.append(name)
            else:  # HIGH detail
                if seen_patterns is not None and typ:
                    pattern = f"{name}:{typ}"
                    if pattern in seen_patterns:
                        params.append(name)  # Elide type - already shown
                    else:
                        seen_patterns.add(pattern)
                        params.append(f"{name}: {typ}")
                else:
                    params.append(f"{name}: {typ}" if typ else name)

        ret = ""
        if detail == DetailLevel.HIGH and self.return_type:
            ret = f" -> {self.return_type}"

        return f"({', '.join(params)}){ret}"


@dataclass(frozen=True)
class FieldInfo:
    """Class field/attribute for dataclass-style display.

    Captured from annotated assignments in class bodies.
    """
    name: str
    type_annotation: Optional[str]
    default_value: Optional[str] = None  # Truncated preview

    def render(self, detail: DetailLevel) -> str:
        """Render field at specified detail level."""
        if detail == DetailLevel.LOW:
            return self.name
        elif detail == DetailLevel.MEDIUM:
            if self.type_annotation:
                # Simplify complex types: "Callable[[int], str]" -> "Callable"
                simple_type = self.type_annotation.split('[')[0]
                return f"{self.name}: {simple_type}"
            return self.name
        else:  # HIGH
            if self.type_annotation:
                return f"{self.name}: {self.type_annotation}"
            return self.name


# =============================================================================
# Render Configuration for Multi-Config Optimization
# =============================================================================

@dataclass
class RenderConfig:
    """Configuration for a single rendering attempt.

    Used by the optimizer to try different combinations of
    coverage (num_tags) and detail (detail_level).
    """
    num_tags: int
    detail_level: DetailLevel

    @property
    def score(self) -> float:
        """Score prioritizing coverage over detail.

        Formula: tags * 10 + detail_weight
        This means 10 extra tags beats 1 detail level increase.
        """
        return self.num_tags * 10 + self.detail_level.value


# =============================================================================
# Tag Structure (Extended with Optional Signature/Field Info)
# =============================================================================

@dataclass(frozen=True)
class Tag:
    """Tag for storing parsed code definitions and references.

    Immutable dataclass representing a code symbol (definition or reference)
    with metadata for ranking, filtering, and multi-detail rendering.

    Attributes:
        rel_fname: Relative filename for display
        fname: Absolute filename for I/O
        line: Line number of definition
        name: Symbol name
        kind: "def" (definition) or "ref" (reference)
        node_type: Tree-sitter node type (e.g., "function", "class")
        parent_name: Enclosing class/function name (None if top-level)
        parent_line: Line number of parent scope (None if top-level)
        signature: Parsed signature info for functions/methods (optional)
        fields: Parsed field info for classes (optional)
    """
    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str
    node_type: str
    parent_name: Optional[str]
    parent_line: Optional[int]
    signature: Optional[SignatureInfo] = None
    fields: Optional[Tuple[FieldInfo, ...]] = None


@dataclass(frozen=True)
class RankedTag:
    """A Tag with its PageRank score for importance-based sorting.

    Used throughout the system to represent ranked tags in lists.
    Replaces the previous Tuple[float, Tag] pattern for type clarity.

    Attributes:
        rank: PageRank score (0.0-1.0, higher = more important)
        tag: The code symbol tag
    """
    rank: float
    tag: Tag


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def read_text(filename: str, encoding: str = "utf-8", silent: bool = False) -> Optional[str]:
    """Read text from file with error handling."""
    try:
        return Path(filename).read_text(encoding=encoding, errors='ignore')
    except FileNotFoundError:
        if not silent:
            print(f"Error: {filename} not found.")
        return None
    except IsADirectoryError:
        if not silent:
            print(f"Error: {filename} is a directory.")
        return None
    except OSError as e:
        if not silent:
            print(f"Error reading {filename}: {e}")
        return None
    except UnicodeError as e:
        if not silent:
            print(f"Error decoding {filename}: {e}")
        return None
    except Exception as e:
        if not silent:
            print(f"An unexpected error occurred while reading {filename}: {e}")
        return None
