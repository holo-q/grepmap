"""Facade orchestrator for GrepMap.

This module provides the main GrepMap class that composes all subsystems:
- CacheManager: Persistent tag caching
- TagParser: Tree-sitter-based code extraction
- PageRanker: Graph-based file importance
- BoostCalculator: Contextual boost application  
- Optimizer: Token budget optimization
- TreeRenderer & DirectoryRenderer: Output formatting

The facade provides a clean, high-level API while delegating to specialized
modules for each concern.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Callable

from grepmap.core.types import Tag, RankedTag, DetailLevel, FileReport
from grepmap.core.config import (
    DEFAULT_MAP_TOKENS, MAP_MUL_NO_FILES, CONTEXT_WINDOW_PADDING,
    OVERFLOW_COUNT, TOKEN_COUNT_FULL_THRESHOLD, TOKEN_COUNT_SAMPLE_LINES
)
from grepmap.cache import CacheManager
from grepmap.extraction import get_tags_raw, extract_signature_info, extract_class_fields
from grepmap.ranking import PageRanker, SymbolRanker, BoostCalculator, GitWeightCalculator, Optimizer, FocusResolver
from grepmap.rendering import TreeRenderer, DirectoryRenderer, StatsRenderer
from utils import count_tokens, read_text


class GrepMap:
    """Main facade for generating grep maps.
    
    Orchestrates all subsystems to produce ranked, token-optimized repository maps.
    Provides backward-compatible interface with the original monolithic implementation.
    
    Args:
        map_tokens: Maximum tokens for the map output
        root: Repository root directory (default: cwd)
        token_counter_func: Function to count tokens in text
        file_reader_func: Function to read file contents
        output_handler_funcs: Dict of output handlers (info, warning, error)
        repo_content_prefix: Optional prefix for repository content
        verbose: Enable verbose logging
        max_context_window: Maximum context window size
        map_mul_no_files: Multiplier when no chat files present
        refresh: Cache refresh mode ("auto", "always", "never")
        exclude_unranked: Exclude files with very low PageRank
        color: Enable colored output
        directory_mode: Use directory overview (vs tree view)
    """
    
    def __init__(
        self,
        map_tokens: int = DEFAULT_MAP_TOKENS,
        root: Optional[str] = None,
        token_counter_func: Callable[[str], int] = count_tokens,
        file_reader_func: Callable[[str], Optional[str]] = read_text,
        output_handler_funcs: Optional[Dict[str, Callable]] = None,
        repo_content_prefix: Optional[str] = None,
        verbose: bool = False,
        max_context_window: Optional[int] = None,
        map_mul_no_files: int = MAP_MUL_NO_FILES,
        refresh: str = "auto",
        exclude_unranked: bool = False,
        color: bool = True,
        directory_mode: bool = True,
        stats_mode: bool = False,
        adaptive_mode: bool = False,
        symbol_rank: bool = True,
        git_weight: bool = False,
        diagnose: bool = False
    ):
        """Initialize GrepMap facade and all subsystems."""
        # Core settings
        self.map_tokens = map_tokens
        self.max_map_tokens = map_tokens
        self.root = Path(root or os.getcwd()).resolve()
        self.token_count_func_internal = token_counter_func
        self.read_text_func_internal = file_reader_func
        self.repo_content_prefix = repo_content_prefix
        self.verbose = verbose
        self.max_context_window = max_context_window
        self.map_mul_no_files = map_mul_no_files
        self.refresh = refresh
        self.exclude_unranked = exclude_unranked
        self.color = color
        self.directory_mode = directory_mode
        self.stats_mode = stats_mode
        self.adaptive_mode = adaptive_mode
        self.symbol_rank = symbol_rank
        self.git_weight = git_weight
        self.diagnose = diagnose

        # Set up output handlers
        if output_handler_funcs is None:
            output_handler_funcs = {
                'info': print,
                'warning': print,
                'error': print
            }
        self.output_handlers = output_handler_funcs
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Legacy in-memory caches (for tree context and map results)
        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
    
    def _init_subsystems(self):
        """Initialize all subsystems with proper dependency injection."""
        # Cache manager for persistent tag storage
        self.cache_manager = CacheManager(
            root=self.root,
            output_handler=self.output_handlers['warning']
        )

        # PageRank calculators (file-level and symbol-level)
        self.pageranker = PageRanker(
            get_rel_fname=self.get_rel_fname,
            verbose=self.verbose,
            output_handlers=self.output_handlers
        )

        # Symbol-level ranker for fine-grained "tree shaking"
        # When enabled, ranks individual symbols instead of whole files,
        # allowing us to surface the ONE important function from a 50-function file
        self.symbol_ranker = SymbolRanker(
            get_rel_fname=self.get_rel_fname,
            verbose=self.verbose,
            output_handlers=self.output_handlers
        )

        # Boost calculator for contextual importance
        self.boost_calculator = BoostCalculator(
            get_rel_fname=self.get_rel_fname,
            exclude_unranked=self.exclude_unranked
        )

        # Token budget optimizer
        self.optimizer = Optimizer(
            token_counter=self.token_count,
            verbose=self.verbose,
            output_handlers=self.output_handlers
        )

        # Renderers for different output modes
        self.tree_renderer = TreeRenderer(
            root=self.root,
            file_reader=self.read_text_func_internal,
            token_counter=self.token_count,
            color=self.color
        )

        self.directory_renderer = DirectoryRenderer(
            root=self.root,
            token_counter=self.token_count,
            verbose=self.verbose,
            output_handler=self.output_handlers['info']
        )

        self.stats_renderer = StatsRenderer(
            root=self.root,
            file_reader=self.read_text_func_internal,
            token_counter=self.token_count,
            verbose=self.verbose,
            output_handler=self.output_handlers['info']
        )

        self.focus_resolver = FocusResolver(
            root=self.root,
            verbose=self.verbose,
            output_handler=self.output_handlers['info']
        )

        # Git weight calculator for temporal relevance (recency, churn)
        self.git_weight_calculator = GitWeightCalculator(
            root=self.root,
            verbose=self.verbose,
            output_handler=self.output_handlers['info']
        )
    
    # =========================================================================
    # Public API - Main entry points
    # =========================================================================
    
    def get_grep_map(
        self,
        focus_targets: Optional[List[str]] = None,
        other_files: Optional[List[str]] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the grep map with file report.

        Main entry point for generating repository maps. Orchestrates all subsystems
        to produce ranked, token-optimized output.

        Args:
            focus_targets: Focus targets - file paths OR search queries. Files get
                          highest priority boost. Queries match symbol names across
                          the codebase and boost matching files/identifiers.
            other_files: Other repository files to consider
            mentioned_fnames: Filenames mentioned in conversation (boost)
            mentioned_idents: Identifiers mentioned in conversation (boost)
            force_refresh: Force cache refresh

        Returns:
            Tuple of (formatted map string, file report)
        """
        if focus_targets is None:
            focus_targets = []
        if other_files is None:
            other_files = []

        # Create empty report for error cases
        empty_report = FileReport({}, 0, 0, 0)

        if self.max_map_tokens <= 0 or not other_files:
            return None, empty_report

        # Adjust max_map_tokens if no focus targets
        max_map_tokens = self.max_map_tokens
        if not focus_targets and self.max_context_window:
            available = self.max_context_window - CONTEXT_WINDOW_PADDING
            max_map_tokens = min(
                max_map_tokens * self.map_mul_no_files,
                available
            )

        try:
            # Delegate to cached map generation
            map_string, file_report = self.get_ranked_tags_map(
                focus_targets, other_files, max_map_tokens,
                mentioned_fnames, mentioned_idents, force_refresh
            )
        except RecursionError:
            self.output_handlers['error']("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return None, FileReport({}, 0, 0, 0)
        
        if map_string is None:
            return None, file_report
        
        if self.verbose:
            tokens = self.token_count(map_string)
            self.output_handlers['info'](f"Repo-map: {tokens / 1024:.1f} k-tokens")
        
        # Format final output
        other = "other " if focus_targets else ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""
        
        repo_content += map_string
        
        return repo_content, file_report
    
    def get_ranked_tags_map(
        self,
        focus_targets: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Get the ranked tags map with caching."""
        cache_key = (
            tuple(sorted(focus_targets)),
            tuple(sorted(other_fnames)),
            max_map_tokens,
            tuple(sorted(mentioned_fnames or [])),
            tuple(sorted(mentioned_idents or [])),
        )

        if not force_refresh and cache_key in self.map_cache:
            return self.map_cache[cache_key]

        result = self.get_ranked_tags_map_uncached(
            focus_targets, other_fnames, max_map_tokens,
            mentioned_fnames, mentioned_idents
        )

        self.map_cache[cache_key] = result
        return result

    def get_ranked_tags_map_uncached(
        self,
        focus_targets: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the ranked tags map without caching.

        Orchestrates the full pipeline:
        1. Extract and rank tags (PageRank + boosts)
        2. Optimize rendering configuration (coverage vs detail tradeoff)
        3. Render with selected configuration
        4. Add overflow section if in directory mode

        In stats_mode, bypasses optimization and renders compact file statistics.
        """
        # Step 1: Get ranked tags using PageRank + boosts
        ranked_tags, file_report, focus_rel_fnames = self.get_ranked_tags(
            focus_targets, other_fnames, mentioned_fnames, mentioned_idents
        )

        if not ranked_tags:
            return None, file_report

        chat_rel_fnames = focus_rel_fnames  # Alias for compatibility
        n = len(ranked_tags)

        # Stats mode: compact diagnostics view (LOC, def counts)
        # Bypasses optimizer since stats are very compact
        # tree_view=True when --tree flag is passed (directory_mode=False)
        if self.stats_mode:
            output = self.stats_renderer.render(
                ranked_tags, chat_rel_fnames,
                tree_view=not self.directory_mode
            )
            if self.verbose:
                tokens = self.token_count(output)
                self.output_handlers['info'](
                    f"Stats mode: {n} tags from {file_report.total_files_considered} files, {tokens} tokens"
                )
            return output, file_report
        
        # For non-directory mode, only use LOW detail (tree view handles its own formatting)
        if self.directory_mode:
            detail_levels = [DetailLevel.LOW, DetailLevel.MEDIUM, DetailLevel.HIGH]
        else:
            detail_levels = [DetailLevel.LOW]
        
        # Step 2: Create renderer function for optimizer
        def render_at_config(tags: List[RankedTag], detail: DetailLevel) -> str:
            """Render callback for optimizer."""
            if self.directory_mode:
                return self.directory_renderer.render(
                    tags, chat_rel_fnames, detail, adaptive=self.adaptive_mode
                )
            else:
                return self.tree_renderer.render(tags, chat_rel_fnames, detail)
        
        # Step 3: Optimize to find best configuration
        try:
            selected_tags, detail, output, tokens = self.optimizer.optimize(
                ranked_tags=ranked_tags,
                max_tokens=max_map_tokens,
                renderer=render_at_config,
                detail_levels=detail_levels
            )
        except ValueError:
            # Fallback: minimal output
            if self.verbose:
                self.output_handlers['info']("Using minimal fallback output")
            minimal_tags = ranked_tags[:min(10, n)]
            if self.directory_mode:
                return self.directory_renderer.render(
                    minimal_tags, chat_rel_fnames, DetailLevel.LOW,
                    adaptive=self.adaptive_mode
                ), file_report
            else:
                return self.tree_renderer.render(minimal_tags, chat_rel_fnames, DetailLevel.LOW), file_report
        
        if self.verbose:
            self.output_handlers['info'](
                f"Selected: {len(selected_tags)} tags, {detail.name} detail, {tokens} tokens"
            )

        # Diagnostic output: ultra-dense machine-readable stats
        if self.diagnose:
            self._output_diagnostics(
                ranked_tags, selected_tags, detail, tokens, max_map_tokens
            )

        # Step 4: Re-render with overflow tags for "also in scope" section
        if self.directory_mode and len(selected_tags) < n:
            num_selected = len(selected_tags)
            overflow_count = min(OVERFLOW_COUNT, n - num_selected)
            overflow = ranked_tags[num_selected:num_selected + overflow_count]
            output = self.directory_renderer.render(
                selected_tags, chat_rel_fnames, detail,
                overflow_tags=overflow, adaptive=self.adaptive_mode
            )
        
        return output, file_report
    
    def get_ranked_tags(
        self,
        focus_targets: List[str],
        other_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[List[RankedTag], FileReport, Set[str]]:
        """Get ranked tags using PageRank algorithm with boosts.

        Orchestrates tag extraction, focus resolution, PageRank computation,
        and boost application.

        Args:
            focus_targets: Focus targets (file paths or search queries)
            other_fnames: Other files to include
            mentioned_fnames: Mentioned file relative paths
            mentioned_idents: Mentioned identifier names

        Returns:
            Tuple of (ranked tags list, file report, focus_rel_fnames)
        """
        # Return empty list and empty report if no files
        if not focus_targets and not other_fnames:
            return [], FileReport({}, 0, 0, 0), set()

        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()

        # Normalize other_fnames paths to absolute
        def normalize_path(path):
            return str(Path(path).resolve())

        other_fnames = [normalize_path(f) for f in other_fnames]

        # Initialize file tracking
        included: List[str] = []
        excluded: Dict[str, str] = {}

        all_fnames = list(set(other_fnames))

        if self.verbose:
            self.output_handlers['info'](f"Processing {len(all_fnames)} files for tag extraction...")

        # Step 1: Extract tags for all files
        tags_by_file: Dict[str, List[Tag]] = {}
        total_definitions = 0
        total_references = 0

        for idx, fname in enumerate(all_fnames):
            rel_fname = self.get_rel_fname(fname)

            # Show progress
            if self.verbose and (idx % 100 == 0 or idx == len(all_fnames) - 1):
                self.output_handlers['info'](f"  [{idx + 1}/{len(all_fnames)}] {rel_fname}")

            if not os.path.exists(fname):
                reason = "File not found"
                excluded[fname] = f"[EXCLUDED] {reason}"
                self.output_handlers['warning'](f"Repo-map can't include {fname}: {reason}")
                continue

            included.append(fname)
            tags = self.get_tags(fname, rel_fname)
            tags_by_file[fname] = tags

            # Count definitions and references
            for tag in tags:
                if tag.kind == "def":
                    total_definitions += 1
                elif tag.kind == "ref":
                    total_references += 1

        # Step 2: Resolve focus targets (file paths or symbol queries)
        focus_files, focus_idents = self.focus_resolver.resolve(
            focus_targets, tags_by_file
        )

        # Combine focus_idents with mentioned_idents for boosting
        combined_mentioned_idents = mentioned_idents | focus_idents

        # Step 3: Compute PageRank scores
        # Use symbol-level ranking when enabled for fine-grained "tree shaking"
        symbol_ranks = None
        if self.symbol_rank:
            symbol_ranks, ranks = self.symbol_ranker.compute_ranks(
                all_fnames=included,
                tags_by_file=tags_by_file,
                chat_fnames=list(focus_files)
            )
        else:
            # Fall back to file-level ranking
            ranks = self.pageranker.compute_ranks(
                all_fnames=included,
                tags_by_file=tags_by_file,
                chat_fnames=list(focus_files)
            )

        # Step 4: Compute git weights if enabled (recency/churn/authorship)
        git_weights = None
        if self.git_weight:
            rel_fnames = [self.get_rel_fname(f) for f in included]
            git_weights = self.git_weight_calculator.compute_weights(
                rel_fnames,
                use_recency=True,
                use_churn=True,
                use_authorship=False  # Optional, can be enabled later
            )
        self._last_git_weights = git_weights  # Store for diagnostics

        # Step 5: Apply boosts and create ranked tags
        # Pass symbol_ranks for per-symbol ranking and git_weights for temporal boost
        ranked_tags = self.boost_calculator.apply_boosts(
            included_files=included,
            tags_by_file=tags_by_file,
            ranks=ranks,
            chat_fnames=list(focus_files),
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=combined_mentioned_idents,
            symbol_ranks=symbol_ranks,
            git_weights=git_weights
        )

        # Sort by rank (descending)
        ranked_tags.sort(key=lambda x: x.rank, reverse=True)

        # Create file report
        file_report = FileReport(
            excluded=excluded,
            definition_matches=total_definitions,
            reference_matches=total_references,
            total_files_considered=len(all_fnames)
        )

        # Compute focus_rel_fnames for rendering
        focus_rel_fnames = set(self.get_rel_fname(f) for f in focus_files)

        return ranked_tags, file_report, focus_rel_fnames
    
    # =========================================================================
    # Tag Extraction with Caching
    # =========================================================================
    
    def get_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Get tags for a file, using cache when possible.

        Delegates to CacheManager for cache lookup, get_tags_raw for extraction.
        """
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        # Try cache first
        cached = self.cache_manager.get_cached_tags(fname, file_mtime)
        if cached is not None:
            return cached

        # Cache miss - extract tags using get_tags_raw
        tags = get_tags_raw(
            fname=fname,
            rel_fname=rel_fname,
            read_text_func=self.read_text_func_internal,
            error_handler=self.output_handlers.get('error')
        )

        # Store in cache
        self.cache_manager.set_cached_tags(fname, file_mtime, tags)

        return tags
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_rel_fname(self, fname: str) -> str:
        """Get relative filename from absolute path."""
        try:
            return str(Path(fname).relative_to(self.root))
        except ValueError:
            return fname
    
    def get_mtime(self, fname: str) -> Optional[float]:
        """Get file modification time."""
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.output_handlers['warning'](f"File not found: {fname}")
            return None
    
    def token_count(self, text: str) -> int:
        """Count tokens in text with sampling optimization for long texts."""
        if not text:
            return 0

        len_text = len(text)
        if len_text < TOKEN_COUNT_FULL_THRESHOLD:
            return self.token_count_func_internal(text)

        # Sample for longer texts
        lines = text.splitlines(keepends=True)
        num_lines = len(lines)

        step = max(1, num_lines // TOKEN_COUNT_SAMPLE_LINES)
        sampled_lines = lines[::step]
        sample_text = "".join(sampled_lines)
        
        if not sample_text:
            return self.token_count_func_internal(text)
        
        sample_tokens = self.token_count_func_internal(sample_text)
        
        if len(sample_text) == 0:
            return self.token_count_func_internal(text)
        
        est_tokens = (sample_tokens / len(sample_text)) * len_text
        return int(est_tokens)

    def render_tree(self, abs_fname: str, rel_fname: str, lines_of_interest: List[int]) -> str:
        """Render specific lines from a file with syntax highlighting.

        Args:
            abs_fname: Absolute file path
            rel_fname: Relative file path
            lines_of_interest: List of line numbers to display

        Returns:
            Formatted code snippet with syntax highlighting
        """
        return self.tree_renderer._render_tree(abs_fname, rel_fname, lines_of_interest, None)

    def _output_diagnostics(
        self,
        ranked_tags: List[RankedTag],
        selected_tags: List[RankedTag],
        detail: DetailLevel,
        tokens_used: int,
        token_budget: int
    ):
        """Output ultra-dense diagnostic data for machine parsing.

        Format: pipe-separated sections, each maximally compressed.
        Designed for LLM consumption, not human readability.
        """
        from grepmap.diagnostics import collect_diagnostic_data, format_diagnostic

        # Get graph data from symbol ranker
        graph_data = {}
        if self.symbol_rank and hasattr(self.symbol_ranker, 'get_diagnostic_data'):
            graph_data = self.symbol_ranker.get_diagnostic_data()

        # Get git weights if available
        git_weights = getattr(self, '_last_git_weights', None)

        data = collect_diagnostic_data(
            num_symbols=graph_data.get('num_symbols', 0),
            num_edges=graph_data.get('num_edges', 0),
            hub_symbols=graph_data.get('hub_symbols', []),
            orphan_count=graph_data.get('orphan_count', 0),
            ranked_tags=ranked_tags,
            git_weights=git_weights,
            token_budget=token_budget,
            tokens_used=tokens_used,
            detail_level=detail,
            tags_selected=len(selected_tags)
        )

        diag_line = format_diagnostic(data)

        # Also add top symbols
        symbol_refs = {}
        if hasattr(self.symbol_ranker, '_last_graph'):
            G = self.symbol_ranker._last_graph
            if G:
                for node in G.nodes():
                    symbol_refs[node] = G.in_degree(node)

        from grepmap.diagnostics import format_top_symbols
        top_line = format_top_symbols(ranked_tags, symbol_refs, n=15)

        self.output_handlers['info'](f"DIAG: {diag_line}")
        self.output_handlers['info'](f"DIAG: {top_line}")

    # =========================================================================
    # Legacy Compatibility Methods
    # =========================================================================
    
    def load_tags_cache(self):
        """Legacy method - delegates to cache manager."""
        pass  # Already initialized in __init__
    
    def save_tags_cache(self):
        """Legacy method - delegates to cache manager."""
        self.cache_manager.save_tags_cache()
    
    def tags_cache_error(self):
        """Legacy method - delegates to cache manager."""
        self.cache_manager.tags_cache_error()
    
    def get_language(self, fname: str) -> str:
        """Get the language name for a file."""
        from grep_ast import filename_to_lang
        return filename_to_lang(fname) or "python"
    
    # Expose extraction methods for backward compatibility
    extract_signature_info = staticmethod(extract_signature_info)
    extract_class_fields = staticmethod(extract_class_fields)
