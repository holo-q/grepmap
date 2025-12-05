"""
File discovery utilities for grepmap.

This module consolidates file discovery logic used by both the CLI and MCP server.
It handles finding source files in directories, respecting .gitignore, and reading
pyproject.toml for source directory hints.

Key features:
- Respects .gitignore via `git ls-files` when available
- Reads pyproject.toml for source directory configuration
- Excludes markdown files by default from directory scans
- Falls back to manual directory walking when git isn't available
"""

import os
import subprocess
from pathlib import Path
from typing import List, Set


# File extensions excluded from directory scans by default.
# These files can still be processed when explicitly specified via chat_files or other_files.
EXCLUDED_EXTENSIONS: Set[str] = {'.md', '.markdown', '.mdown', '.mkd'}

# Directories always excluded from manual walks
EXCLUDED_DIRS: Set[str] = {'node_modules', '__pycache__', 'venv', 'env', '.git'}


def is_excluded_by_extension(filename: str) -> bool:
    """Check if a file should be excluded from directory scans by default.

    Markdown files are excluded from automatic discovery but can still be
    processed when explicitly specified.
    """
    return any(filename.lower().endswith(ext) for ext in EXCLUDED_EXTENSIONS)


def get_source_dirs_from_pyproject(directory: str) -> List[str]:
    """Extract source directories from pyproject.toml if it exists.

    Checks multiple common configurations:
    - [tool.ty.src] for type checker configuration
    - [tool.setuptools] packages and py-modules
    - Common 'src' directory convention

    Returns:
        List of source directory/file paths relative to directory
    """
    pyproject_path = Path(directory) / "pyproject.toml"
    if not pyproject_path.exists():
        return []

    import tomllib  # Python 3.11+ (required by project)

    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)

        src_dirs = []

        # Check [tool.ty.src] for type checker source configuration
        ty_config = data.get('tool', {}).get('ty', {}).get('src', {})
        if 'include' in ty_config:
            includes = ty_config['include']
            if isinstance(includes, list):
                src_dirs.extend(includes)

        # Check [tool.setuptools] for packages or py-modules
        setuptools = data.get('tool', {}).get('setuptools', {})

        # Handle packages (list of package names or find directive)
        if 'packages' in setuptools:
            packages = setuptools['packages']
            if isinstance(packages, list):
                src_dirs.extend(packages)

        # Handle py-modules (individual module files)
        if 'py-modules' in setuptools:
            modules = setuptools['py-modules']
            if isinstance(modules, list):
                for module in modules:
                    module_path = Path(directory) / f"{module}.py"
                    if module_path.exists():
                        src_dirs.append(str(module_path))

        # Check for package-dir (custom source directory mapping)
        if 'package-dir' in setuptools:
            pkg_dir = setuptools['package-dir']
            if isinstance(pkg_dir, dict) and '' in pkg_dir:
                src_dirs.append(pkg_dir[''])

        # Common convention: check if 'src' directory exists
        src_path = Path(directory) / "src"
        if src_path.exists() and src_path.is_dir() and "src" not in src_dirs:
            src_dirs.append("src")

        return src_dirs
    except Exception:
        return []


def find_source_files(directory: str) -> List[str]:
    """Find source files in a directory, respecting .gitignore if in a git repo.

    Discovery strategy:
    1. If pyproject.toml specifies source dirs, scope to those
    2. Use `git ls-files` to respect .gitignore when available
    3. Fall back to manual directory walk (excludes common noise dirs)

    Markdown files are excluded by default but can be processed when
    explicitly specified via chat_files or other_files parameters.

    Args:
        directory: Path to directory (or single file) to scan

    Returns:
        List of absolute file paths
    """
    if not os.path.isdir(directory):
        return [directory] if os.path.isfile(directory) else []

    # Check if pyproject.toml defines source directories
    src_dirs = get_source_dirs_from_pyproject(directory)

    # Try using git to respect .gitignore
    try:
        if src_dirs:
            return _find_files_in_source_dirs(directory, src_dirs)
        else:
            return _find_files_with_git(directory)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Git not available or timed out, fall back to manual walk
        pass

    # Fallback: manual walk (less reliable, doesn't respect .gitignore)
    return _find_files_manual(directory, src_dirs)


def _find_files_in_source_dirs(directory: str, src_dirs: List[str]) -> List[str]:
    """Find files in specific source directories using git ls-files."""
    src_files = []
    for src_dir in src_dirs:
        src_path = Path(directory) / src_dir

        # Handle individual .py files
        if src_dir.endswith('.py') and src_path.is_file():
            result = subprocess.run(
                ['git', 'ls-files', '--error-unmatch', src_dir],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                src_files.append(str(src_path))
        # Handle directories
        elif src_path.is_dir():
            result = subprocess.run(
                ['git', 'ls-files', src_dir],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and not is_excluded_by_extension(line):
                        full_path = Path(directory) / line
                        if full_path.is_file():
                            src_files.append(str(full_path))
    return src_files


def _find_files_with_git(directory: str) -> List[str]:
    """Find all git-tracked files in directory."""
    result = subprocess.run(
        ['git', 'ls-files'],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode != 0:
        raise FileNotFoundError("git ls-files failed")

    src_files = []
    for line in result.stdout.strip().split('\n'):
        if line and not is_excluded_by_extension(line):
            full_path = os.path.join(directory, line)
            if os.path.isfile(full_path):
                src_files.append(full_path)
    return src_files


def _find_files_manual(directory: str, src_dirs: List[str]) -> List[str]:
    """Manually walk directory tree to find source files.

    Used as fallback when git is not available.
    """
    src_files = []

    if src_dirs:
        # Walk only specified source directories
        for src_dir in src_dirs:
            src_path = Path(directory) / src_dir
            if src_path.is_file() and src_dir.endswith('.py'):
                src_files.append(str(src_path))
            elif src_path.is_dir():
                src_files.extend(_walk_directory(src_path))
    else:
        # Walk entire directory
        src_files.extend(_walk_directory(Path(directory)))

    return src_files


def _walk_directory(root_path: Path) -> List[str]:
    """Walk a directory tree, excluding noise directories and files."""
    src_files = []
    for root, dirs, files in os.walk(root_path):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_DIRS]
        for file in files:
            if not file.startswith('.') and not is_excluded_by_extension(file):
                src_files.append(os.path.join(root, file))
    return src_files
