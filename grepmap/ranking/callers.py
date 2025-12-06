"""
Caller resolution for reverse-edge boosting in GrepMap.

This module finds callers of focus symbols by tracing edges backwards in the
symbol graph. Used primarily by DEBUG intent to answer "what calls this broken
function?" and surface the call sites that lead to the problematic code.

The reverse-edge bias in IntentRecipe controls how strongly we boost these
callers. A higher bias (e.g., 2.0 for DEBUG) means "show me the call sites",
while a lower bias (e.g., 0.7 for REFERENCE) means "show me the definitions".

Design rationale:
- SymbolRanker builds a directed graph: caller → callee
- For debugging, we want to trace backwards: given a broken function B,
  find all functions A where A→B exists in the graph
- This surfaces the calling context that triggers the bug

Example:
    Focus symbols: {("auth.py", "validate_token")}
    Graph edges: ("api.py", "login") → ("auth.py", "validate_token")
                 ("api.py", "refresh") → ("auth.py", "validate_token")
    Result: {"api.py"} (file containing callers)
"""

import networkx as nx
from typing import Set, Tuple, Optional, Callable


# Symbol identifier: (relative_filename, symbol_name)
# Matches SymbolId from symbols.py for consistency
SymbolId = Tuple[str, str]


class CallerResolver:
    """Finds callers of focus symbols for reverse-edge boosting.

    Used by DEBUG intent to surface "what calls this broken function?"
    Walks the symbol graph backwards to find predecessor symbols (callers).

    The algorithm:
    1. For each focus symbol (rel_fname, symbol_name)
    2. Find all predecessor nodes in the graph (symbols that have edges TO the focus)
    3. Extract the files containing those caller symbols
    4. Return as a set of rel_fnames for downstream boosting

    This enables intent-driven ranking: when debugging auth.py:validate_token,
    automatically boost api.py:login and api.py:refresh that call it.
    """

    def __init__(self, verbose: bool = False):
        """Initialize CallerResolver.

        Args:
            verbose: Enable verbose logging for debugging
        """
        self.verbose = verbose

    def find_callers(
        self,
        focus_symbols: Set[SymbolId],
        symbol_graph: nx.MultiDiGraph,
        output_handler: Optional[Callable[[str], None]] = None
    ) -> Set[str]:
        """Find all files containing symbols that call any focus symbol.

        Walks edges backwards in the symbol graph: if A→B and B is in focus_symbols,
        then A's file is added to the result set.

        This is the core of reverse-edge boosting: we identify the calling context
        that leads to the focused code, which is essential for debugging.

        Args:
            focus_symbols: Set of (rel_fname, symbol_name) tuples to find callers for
            symbol_graph: MultiDiGraph where edges represent calls/references
            output_handler: Optional function for logging (default: print)

        Returns:
            Set of rel_fnames (relative file paths) containing caller symbols.
            These files should receive the reverse_edge_bias boost.

        Example:
            focus_symbols = {("auth.py", "validate_token")}
            graph edges: ("api.py", "login") → ("auth.py", "validate_token")
            returns: {"api.py"}
        """
        if output_handler is None:
            output_handler = print

        caller_files: Set[str] = set()
        caller_count = 0

        # For each focus symbol, find its predecessors (symbols that call it)
        for focus_symbol in focus_symbols:
            if focus_symbol not in symbol_graph:
                # Focus symbol not in graph (maybe external or misspelled)
                if self.verbose:
                    output_handler(
                        f"Focus symbol {focus_symbol[1]} in {focus_symbol[0]} "
                        f"not found in symbol graph"
                    )
                continue

            # Get all predecessors (callers) of this focus symbol
            # predecessors() returns nodes that have edges TO the focus node
            predecessors = symbol_graph.predecessors(focus_symbol)

            for caller_symbol in predecessors:
                # caller_symbol is (rel_fname, symbol_name)
                caller_fname, caller_name = caller_symbol
                caller_files.add(caller_fname)
                caller_count += 1

                if self.verbose:
                    output_handler(
                        f"  Caller: {caller_fname}:{caller_name} → "
                        f"{focus_symbol[0]}:{focus_symbol[1]}"
                    )

        if self.verbose and caller_files:
            output_handler(
                f"Found {caller_count} caller symbols across {len(caller_files)} files"
            )

        return caller_files
