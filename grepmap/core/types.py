"""
Type definitions for RepoMapper.

This module consolidates all dataclass and type enum definitions used throughout
the codebase. It serves as the single source of truth for type structures, enabling
clean imports and reducing circular dependencies.

Key types:
- DetailLevel: Enum controlling output detail (LOW/MEDIUM/HIGH)
- SignatureInfo: Parsed function signatures for multi-detail rendering
- FieldInfo: Class field/attribute information
- RenderConfig: Configuration for rendering attempts
- Tag: Code symbol definition with metadata
- RankedTag: Tag paired with PageRank importance score
- FileReport: Aggregate statistics from tag processing
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# =============================================================================
# Detail Level for Multi-Configuration Optimization
# =============================================================================

class DetailLevel(IntEnum):
    """Rendering detail level for output optimization.

    Higher values = more detail = more tokens consumed.
    The optimizer tries to maximize (coverage * detail) within token budget.

    Attributes:
        LOW: Symbol names only: "connect, disconnect, refresh"
        MEDIUM: Names + simplified types: "connect(self, hobo, remote)"
        HIGH: Full signatures: "connect(self, hobo: HoboWindow) -> bool"
    """
    LOW = 1      # Symbol names only
    MEDIUM = 2   # Names + simplified types
    HIGH = 3     # Full signatures with annotations


# =============================================================================
# Signature and Field Information for Variable-Detail Rendering
# =============================================================================

@dataclass(frozen=True)
class SignatureInfo:
    """Parsed function/method signature for detail-level rendering.

    Extracted during tag parsing, cached alongside Tag.
    Enables rendering at different detail levels without re-parsing.

    Attributes:
        parameters: Tuple of (name, type_annotation) pairs for function parameters.
                   Type annotation may be None if not provided in source.
        return_type: Return type annotation as string, or None if not specified.
        decorators: Tuple of decorator names (e.g., "staticmethod", "property").
        raw: Optional pre-formatted string (used for markdown section previews).
             When set, render() returns this directly instead of building from params.
    """
    parameters: Tuple[Tuple[str, Optional[str]], ...]  # ((name, type_annotation), ...)
    return_type: Optional[str]
    decorators: Tuple[str, ...]  # ("staticmethod", "property", ...)
    raw: Optional[str] = None  # Pre-formatted content (e.g., markdown section preview)

    def render(self, detail: DetailLevel, seen_patterns: Optional[set] = None) -> str:
        """Render signature at specified detail level with optional deduplication.

        At LOW detail, returns "...". At MEDIUM and HIGH, builds parameter list
        with increasing type information. Supports seen_patterns for deduplication
        of type information across multiple signatures in HIGH detail.

        Args:
            detail: The level of detail to render (LOW/MEDIUM/HIGH)
            seen_patterns: Optional set of "name:type" patterns already shown.
                           If provided, types are elided for patterns already seen.
                           This reduces redundancy when displaying multiple similar signatures.

        Returns:
            Formatted string like "(param1, param2) -> ReturnType" appropriate for detail level.
        """
        # If raw content is set (e.g., markdown preview), return it directly
        if self.raw is not None:
            return f'— "{self.raw}"'

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
    Enables rendering of class structure with varying detail levels.

    Attributes:
        name: Field name as it appears in the class
        type_annotation: Type annotation string (e.g., "str", "List[int]")
        default_value: Optional truncated preview of default value
    """
    name: str
    type_annotation: Optional[str]
    default_value: Optional[str] = None  # Truncated preview

    def render(self, detail: DetailLevel) -> str:
        """Render field at specified detail level.

        At LOW detail, shows only name. At MEDIUM, simplifies complex types
        by taking the base part (e.g., "Callable[[int], str]" -> "Callable").
        At HIGH detail, shows full type annotation.

        Args:
            detail: The level of detail to render (LOW/MEDIUM/HIGH)

        Returns:
            Formatted field string appropriate for detail level.
        """
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

    The system uses binary search to find the maximum number of tags
    that fit within the token budget for each detail level, then selects
    the configuration with the highest score.

    Attributes:
        num_tags: Number of tags to include in this rendering
        detail_level: Level of detail for signatures/fields (LOW/MEDIUM/HIGH)
    """
    num_tags: int
    detail_level: DetailLevel

    @property
    def score(self) -> float:
        """Score prioritizing coverage over detail.

        Formula: tags * 10 + detail_weight

        This means 10 extra tags are worth 1 detail level increase,
        ensuring we maximize coverage first before adding signature detail.

        Returns:
            Float score for comparing configurations.
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

    The tag system enables efficient caching of parsed symbols and supports
    hierarchical rendering (e.g., methods nested under classes) via parent_name
    and parent_line tracking.

    Attributes:
        rel_fname: Relative filename for display (from repo root)
        fname: Absolute filename for I/O operations
        line: Line number of definition/reference
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
    Replaces the previous Tuple[float, Tag] pattern for type clarity
    and safer access patterns.

    PageRank scoring integrates multiple signals:
    - Graph structure: References between files create edges
    - Depth bias: Root/shallow files get higher base weight
    - Vendor detection: Penalizes third-party code
    - Chat boost: Multiplies score for files in conversation

    Attributes:
        rank: PageRank score (0.0-1.0, higher = more important)
        tag: The code symbol tag
    """
    rank: float
    tag: Tag


# =============================================================================
# File Processing Report
# =============================================================================

@dataclass
class FileReport:
    """Aggregate statistics from tag extraction processing.

    Generated by get_ranked_tags() to provide visibility into which files
    were included/excluded and tag extraction results.

    Attributes:
        excluded: Dict mapping excluded filenames to exclusion reasons.
                 Format: {filename: "reason (status)"}
        definition_matches: Total number of definition tags extracted
        reference_matches: Total number of reference tags extracted
        total_files_considered: Total number of input files processed
    """
    excluded: Dict[str, str]        # File -> exclusion reason with status
    definition_matches: int         # Total definition tags
    reference_matches: int          # Total reference tags
    total_files_considered: int     # Total files provided as input
