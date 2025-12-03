# grepmap

**Dox your codebase for Claude.**

You know how Claude sometimes fumbles around your repo like a tourist with a bad map? grepmap fixes that. It generates a structural exposé of your codebase—ranking files by *actual importance* using PageRank, not just alphabetical order or vibes.

Feed it to your AI assistant and watch it suddenly *get* your project.

## What It Does

grepmap builds a dependency graph of your code using tree-sitter, then runs PageRank to find out which files are the main characters and which are just extras. The result is a condensed map that fits in an LLM's context window, showing:

- **Who talks to who** — files ranked by how much other code depends on them
- **The family tree** — classes, functions, methods organized by directory topology
- **The VIPs** — symbols that actually matter, not just whatever's alphabetically first

```
src/
  session.py: Session, FramedirStorage, ResdirMixin, +7 more
  hud.py: HUD, DrawSignal, NameContext
  deploy/
    DiscoSSH.py: DiscoSSH, LogBuffer, NoTermSSHClientConnection, +1 more
    manage_layout.py: ManageLayout
  gui/
    old/
      DPG.py: ContainerProxy, DPG, HorizontalColumnContext
```

The deeply nested vendor code that nobody actually uses? Demoted. Your core `session.py` that everything imports? Promoted. PageRank doesn't lie.

## Installation

```bash
pip install grepmap
```

Or with uv:
```bash
uv tool install grepmap
```

## Usage

```bash
# Dox the current directory
grepmap .

# Dox with more context
grepmap . --map-tokens 4096

# Focus on specific files (higher priority in ranking)
grepmap --chat-files main.py session.py --other-files src/

# Tree view for deep-diving a single file
grepmap path/to/file.py --tree

# Clear the cache if things get weird
grepmap . --clear-cache
```

### CLI Options

| Flag | What it does |
|------|--------------|
| `--map-tokens N` | Token budget for the map (default: 8192) |
| `--chat-files` | Files you're actively working on (get ranking boost) |
| `--other-files` | Additional context files |
| `--mentioned-files` | Files mentioned in conversation (mid-level boost) |
| `--mentioned-idents` | Identifiers to boost (function names, classes, etc.) |
| `--tree` | Detailed tree view instead of directory overview |
| `--exclude-unranked` | Hide files with PageRank of 0 |
| `--no-color` | Disable syntax highlighting |
| `--verbose` | Show ranking details and debug info |
| `--force-refresh` | Ignore cache, reparse everything |
| `--clear-cache` | Nuke the cache directory |

## How It Actually Works

1. **Parse everything** — tree-sitter extracts definitions and references from your code
2. **Build the graph** — files become nodes, symbol references become edges
3. **Run PageRank** — with depth-aware personalization that penalizes `node_modules` and `vendor/` while letting truly interconnected deep files rise
4. **Binary search for fit** — finds the maximum content that fits your token budget
5. **Render prettily** — syntax highlighting, directory trees, class summaries

The depth penalty is applied *in graph space*, not as post-processing. This means if some deeply nested file is genuinely crucial (referenced by 166 files with 313 edges), PageRank will surface it anyway. The algorithm respects reality over appearances.

## MCP Server

grepmap ships with an MCP server for integration with AI tools:

```bash
grepmap-mcp
```

Configure in your MCP client (e.g., Claude Desktop, Cline):

```json
{
  "mcpServers": {
    "grepmap": {
      "command": "grepmap-mcp"
    }
  }
}
```

### MCP Tools

- **`grep_map`** — Generate a ranked map of a project
- **`search_identifiers`** — Search for symbols across the codebase

## Why PageRank?

Because popularity matters. A file that's imported by 50 other files is probably more important than one imported by 2. PageRank captures this transitive importance—if A imports B and B imports C, then C gets credit for being foundational.

We also apply depth-aware personalization:
- Root-level files: full weight
- Shallow files (depth ≤ 2): full weight
- Moderate depth (3-4): 0.5x weight
- Deep files (5+): 0.1x weight
- Vendor patterns (`node_modules`, `vendor/`, `torchhub/`): 0.01x weight

But here's the key: this is a *prior*, not a hard rule. If your `vendor/something/critical.py` is genuinely the backbone of your app, the graph structure will override the depth penalty.

## Caching

grepmap caches parsed tags in `.grepmap.tags.cache.v{N}/` to avoid re-parsing unchanged files. The cache auto-invalidates when files change (based on mtime).

## Supported Languages

Anything tree-sitter supports:

Python, JavaScript, TypeScript, Rust, Go, C, C++, Java, Ruby, PHP, C#, Kotlin, Swift, Scala, Elixir, Erlang, Haskell, OCaml, Lua, R, Julia, and [many more](https://tree-sitter.github.io/tree-sitter/#available-parsers).

## Origins

grepmap is a spiritual successor to [aider's RepoMap](https://github.com/paul-gauthier/aider). We took the core concept—PageRank over code graphs—and rebuilt it with:

- Cleaner output optimized for AI consumption
- Depth-aware ranking to handle real-world monorepos
- Directory topology views for quick orientation
- MCP server for tool integration
- Multi-detail rendering (LOW/MEDIUM/HIGH) for token optimization

## License

MIT

---

*"It's like showing Claude the social network of your code."*
