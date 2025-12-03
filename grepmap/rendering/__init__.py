"""
Rendering subsystem for GrepMap.

This module provides multiple rendering strategies for displaying ranked tags:

- base.Renderer: Protocol defining the renderer interface
- TreeRenderer: Code snippet view with syntax highlighting
- DirectoryRenderer: Hierarchical symbol overview
- syntax helpers: Color and icon utilities

Each renderer takes ranked tags and produces formatted output optimized for
different use cases (code inspection vs. architectural overview).
"""

from grepmap.rendering.base import Renderer
from grepmap.rendering.tree import TreeRenderer
from grepmap.rendering.directory import DirectoryRenderer
from grepmap.rendering.syntax import (
    get_token_color,
    get_symbol_icon,
    get_symbol_color
)

__all__ = [
    'Renderer',
    'TreeRenderer',
    'DirectoryRenderer',
    'get_token_color',
    'get_symbol_icon',
    'get_symbol_color',
]
