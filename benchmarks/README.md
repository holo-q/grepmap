# Agent Navigation Benchmarks

**Purpose:** Discover missing affordances in grepmap by observing how AI agents navigate codebases.

## Philosophy

grepmap's interface should emerge from agent behavior, not top-down design. These benchmarks let us:

1. **Watch agents struggle** - Where do they repeat searches? Where do they read many files?
2. **Spot workarounds** - What patterns do they improvise to compensate for missing tools?
3. **Crystallize affordances** - Turn recurring workarounds into first-class features

The agent is the hamster. The codebase is the maze. We watch from above.

## Quick Start

```bash
# Watch a single navigation challenge (streaming output)
./observe.py ~/projects/myapp "Where is the main entry point?"

# Compare tool configurations
./observe.py ~/projects/myapp "Find all usages of Session" --tools grepmap
./observe.py ~/projects/myapp "Find all usages of Session" --tools grep-only

# Run full benchmark suite
./run_challenges.py ~/projects/myapp --output results.json
```

## Scripts

### `observe.py` - Interactive Observation

Runs a single question and streams output. Use this to:
- Watch agent reasoning in real-time
- Spot moments of confusion
- See what tools they reach for

```bash
./observe.py <repo> "<question>" [--tools grepmap|grep-only|both]
```

### `run_challenges.py` - Batch Benchmarking

Runs multiple challenges and collects metrics:
- Token usage (effort proxy)
- Time taken
- Tool usage patterns
- Success rate

```bash
./run_challenges.py <repo> [--challenges pattern*] [--output results.json]
```

### `challenges.yaml` - Challenge Definitions

Categories:
- **location** - "Where is X defined?"
- **relationship** - "How does X connect to Y?"
- **comprehension** - "What does X do?"
- **modification** - "How would I change X?"

## What to Watch For

### Signals of Missing Affordances

| Observation | Possible Missing Feature |
|-------------|-------------------------|
| Repeated `grepmap` calls with different tokens | Auto-scaling budget |
| `grepmap` then `rg` for same symbol | Combined search mode |
| Many `cat` calls on related files | Dependency chain view |
| Manual import tracing | `--follow-imports` flag |
| Searching for "who calls X" | `--callers-of` flag |
| Reading files to understand types | Type-aware search |

### Recording Insights

When you spot a pattern, document it:

```yaml
# In challenges.yaml or a separate observations.yaml
observations:
  - date: 2024-01-15
    challenge: trace_import_chain
    agent_behavior: "Called grepmap 4 times, manually traced imports with cat"
    missing_affordance: "--follow-imports flag that walks the dependency graph"
    priority: high
```

## Metrics

Token usage is a proxy for cognitive effort:
- **Lower is better** (for same success rate)
- Compare across tool configs to see which affordances help
- Watch for diminishing returns (more tokens ≠ more insight)

## Contributing New Challenges

Add to `challenges.yaml`:

```yaml
- id: your_challenge_id
  category: location|relationship|comprehension|modification
  difficulty: easy|medium|hard
  question: "The question to ask, with {params} if needed"
  params:
    target: "auto-detect or specific value"
  validation: "How to verify correct answer"
```

## Integration with Development

Run benchmarks before/after interface changes:

```bash
# Before adding new flag
./run_challenges.py ~/test-repo -o before.json

# After adding new flag
./run_challenges.py ~/test-repo -o after.json

# Compare token usage and success rates
```

This creates evolutionary pressure on the interface.
