╔══════════════════════════════════════════════════════════════════════════════╗
║  CRITICAL: USE GREPMAP FIRST - THIS IS NOT OPTIONAL                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Before using Glob, Grep, Read, or Task(Explore) on this codebase:

    uv run python grepmap.py . --diag

This is the ONLY way to:
1. Experience the UX we're building (friction = bugs to fix)
2. Validate the map orients YOU in our own territory
3. Catch regressions in output quality before users do

If grepmap's output doesn't answer your question → THAT'S A FEATURE GAP.
File it mentally, then proceed with other tools.

The map should make Glob/Grep unnecessary 80% of the time.
If you're reaching for search tools first, ask: "Why didn't grepmap surface this?"

=== Introspection via --diag ===

`grepmap . --diag` outputs ultra-dense machine-readable diagnostics:
```
G:265n/332e/0.5%d hub:render:32 orph:172
R:0.003/0.007/0.009/0.018 gini:0.08
B:pr→sym*1.0→git*2.3→foc*20 chain:46x
TOP:parse_markdo:0.018:23↓,Tag:0.008:6↓...
```

Key metrics to watch:
- **gini**: 0.08=flat (no signal), 0.8+=concentrated (focus working)
- **orph**: orphan symbols with 0 refs - noise floor indicator
- **cliff@pN**: rank cliff at percentile N - how sharp is the cutoff
- **chain**: max boost multiplication - sanity check on boost stacking

Use --diag to feel the machine responding to inputs. Vary --focus, --git-weight,
--map-tokens and observe how gini/cliff shift. The art is developing intuition
for how the ranking beast steers.

=== Situational Thinking ===

Don't mechanically sweep hyperparameters. Instead, hallucinate situations:

**Common scenarios:**
- "Just cloned, wtf is this" → needs orientation, maybe --stats default?
- "Debugging facade.py" → should auto-focus from git status, not require flag
- "What calls get_ranked_tags?" → need graph navigation, not just ranking

**Flag hell smell:** If using >3 flags, something's wrong. Good defaults should
cover 90% of situations. Auto-detection > manual specification.

**The question:** Does the beast steer itself cleanly, or require human twiddling?
Observe whether focus shifts gini from 0.08→0.9 without explicit flags.

=== Work Standard ===

* Use `ty check` and `uvx ruff check` after big changes to verify
* Keep your eyes peeled and proactively whisper to the user when you notice a refactor that can simplify the code or architectural or abstraction extraction, suggesting possible encapsulations that would be elegant. Sometimes we hack on a project and we don't expect it to grow and all these features to come in, and we end up with a 1500 line monolith!
