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
# Run directly without installing
uvx grepmap .

# Or install globally
uv tool install grepmap

# Or with pip
pip install grepmap
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

## Workflow: GPS + Microscope

grepmap is your codebase GPS. grep/rg is your microscope. Use them together.

**The pattern that works:**

```
Orient (grepmap) → Hypothesize → Verify (rg) → Deep dive (--tree)
```

### 1. Start with the map

```bash
grepmap ~/myproject --map-tokens 2048
```

This shows you the "main characters"—files ranked by actual importance. Don't dive into random files; let PageRank tell you where the action is.

### 2. Follow the rankings

High-ranked files are the ones everything depends on. If you're looking for "where sessions are managed," the map already told you `session.py` is a VIP. Start there.

### 3. Zoom in with --chat-files

Found the neighborhood? Focus on it:

```bash
grepmap ~/myproject --chat-files src/session.py src/renderer.py
```

This boosts those files and shows you their connections.

### 4. Deep dive with --tree

Need to understand one file completely?

```bash
grepmap src/gui/dear_discore.py --tree
```

This shows every class, method, and function in the file with their signatures.

### 5. Verify with grep

Once you know WHERE to look, use rg for the last mile:

```bash
rg "fps.*=" src/session.py -n
```

**The insight:** grepmap shows you WHAT and WHERE. grep shows you HOW. Together they're like having a codebase GPS and a microscope.

## MCP Server

grepmap ships with an MCP server for integration with AI tools.

### Claude Code

```bash
claude mcp add grepmap -- uvx grepmap-mcp
```

### Claude Desktop / Cline / Other MCP Clients

Add to your MCP config (e.g., `~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "grepmap": {
      "command": "uvx",
      "args": ["grepmap-mcp"]
    }
  }
}
```

### CLI via CLAUDE.md (Alternative to MCP)

If you prefer CLI invocation over MCP, add this to your project's `CLAUDE.md`:

```markdown
=== Codebase Navigation: Topology-Aware Sampling ===

`grepmap` provides density-optimized structural maps of codebases using PageRank over dependency graphs.

=== What It Samples ===

**Not** alphabetical listings or random file walks. **Yes** graph-theoretic importance:
- Parses all code with tree-sitter (functions, classes, imports, references)
- Builds dependency graph: files as nodes, symbol references as edges
- Runs PageRank with depth-aware personalization (root files = 1.0x, deep vendor code = 0.01x)
- Binary-searches token budget to maximize information density

**Result**: You see files ranked by *transitive importance* (what depends on what), not file size or proximity.

=== Topology Preservation ===

The output maintains **directory hierarchy** and **class structure**:
- Directory nesting shows architectural layers
- Classes display fields and methods grouped by visibility (properties, methods)
- Signatures normalized to one line (multi-line defs collapsed with full type info)
- Trailing colons stripped (saves 1 token/symbol)

**Scale handling**:
- Adjusts detail level (LOW/MEDIUM/HIGH) based on token budget
- Shows "main characters" first, demotes vendor/generated code
- Overflow section lists additional files at low resolution for extended orientation

=== Usage Patterns ===

**Orient first** (prevent random file diving):
uvx grepmap . --map-tokens 4096  # Denser for smaller budgets

**Focus on active work** (boost specific files in ranking):
uvx grepmap . --chat-files src/session.py src/renderer.py

**Deep dive one file** (complete symbol listing with signatures):
uvx grepmap src/gui/component.py --tree

**After modifying graph** (cache invalidation):
uvx grepmap . --force-refresh

=== Information Density Characteristics ===

- **Breadth**: Covers entire dependency graph, not just git root files
- **Sampling**: PageRank surfaces central nodes; depth penalty demotes leaf files
- **Compression**: Multi-line signatures → one line, type deduplication, colon stripping
- **Topology**: Hierarchical (directories → files → classes → methods), not flat
- **Scale**: Binary search finds maximum coverage within token budget
- **Causality**: High-ranked files are *dependencies* of many others (causal anchors)

Workflow: GPS → Microscope

1. **Orient** (`grepmap .`) — learn graph topology, identify VIP files
2. **Hypothesize** — "session management probably in high-ranked session.py"
3. **Verify** (`rg "keyword" file.py`) — grep exact content in discovered files
4. **Deep dive** (`grepmap file.py --tree`) — see complete symbol listing

grepmap shows **WHAT exists** and **WHERE it matters** (ranked). grep shows **HOW it works** (content).

Don't search blindly. Let PageRank guide you to causal anchors first.
```

### MCP Tools

#### `grep_map`

Generate a ranked map of a project. The main tool.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `project_root` | string | **Required.** Absolute path to the project root |
| `chat_files` | string[] | Files currently in focus (highest ranking boost) |
| `other_files` | string[] | Additional files to consider |
| `token_limit` | int | Token budget for the map (default: 8192) |
| `mentioned_files` | string[] | Files mentioned in conversation (mid boost) |
| `mentioned_idents` | string[] | Identifiers to boost (function names, etc.) |
| `exclude_unranked` | bool | Hide files with PageRank of 0 |
| `force_refresh` | bool | Bypass cache |
| `verbose` | bool | Show ranking details |

**Returns:** `{ map: string, report: { excluded, definition_matches, reference_matches, total_files_considered } }`

#### `search_identifiers`

Search for symbols across the codebase. Useful for finding where things are defined or referenced.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `project_root` | string | **Required.** Absolute path to the project root |
| `query` | string | **Required.** Identifier name to search for |
| `max_results` | int | Maximum results to return (default: 50) |
| `context_lines` | int | Lines of context around matches (default: 2) |
| `include_definitions` | bool | Include where symbols are defined (default: true) |
| `include_references` | bool | Include where symbols are used (default: true) |

**Returns:** `{ results: [{ file, line, name, kind, context }] }`

## How It Actually Works

1. **Parse everything** — tree-sitter extracts definitions and references from your code
2. **Build the graph** — files become nodes, symbol references become edges
3. **Run PageRank** — with depth-aware personalization that penalizes `node_modules` and `vendor/` while letting truly interconnected deep files rise
4. **Binary search for fit** — finds the maximum content that fits your token budget
5. **Render prettily** — syntax highlighting, directory trees, class summaries

The depth penalty is applied *in graph space*, not as post-processing. This means if some deeply nested file is genuinely crucial (referenced by 166 files with 313 edges), PageRank will surface it anyway. The algorithm respects reality over appearances.

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
