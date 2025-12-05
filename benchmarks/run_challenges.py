#!/usr/bin/env python3
"""
Agent Navigation Challenge Runner

Runs AI agents through codebase navigation challenges to:
1. Measure navigation efficiency (tokens, time)
2. Observe tool usage patterns
3. Identify missing affordances in grepmap

Usage:
    ./run_challenges.py <target_repo> [options]

Examples:
    ./run_challenges.py ~/projects/myapp --tools grepmap
    ./run_challenges.py ~/projects/myapp --tools grep-only --challenges locate_*
    ./run_challenges.py ~/projects/myapp --tools both --output results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ChallengeResult:
    """Result of a single challenge run."""
    challenge_id: str
    tool_config: str
    success: bool
    answer: str
    tokens_used: int  # Estimated from output length
    time_seconds: float
    stdout: str
    stderr: str
    strategy_notes: str  # What patterns did the agent use?


@dataclass
class BenchmarkRun:
    """Complete benchmark run metadata."""
    timestamp: str
    target_repo: str
    tool_config: str
    challenges_run: int
    challenges_passed: int
    total_tokens: int
    total_time_seconds: float
    results: List[ChallengeResult]


def load_challenges(challenges_file: Path) -> Dict[str, Any]:
    """Load challenge definitions from YAML."""
    with open(challenges_file) as f:
        return yaml.safe_load(f)


def build_prompt(challenge: Dict, tool_config: Dict, target_repo: str) -> str:
    """Build the prompt for Codex."""
    setup = tool_config.get('setup', '')
    question = challenge['question']

    # Auto-detect params if needed (simplified - real impl would analyze repo)
    params = challenge.get('params', {})
    for key, value in params.items():
        if value == "auto-detect" or value.startswith("auto-detect"):
            # Placeholder - in real impl, would analyze repo
            params[key] = f"[{key}]"
        question = question.replace(f"{{{key}}}", str(params.get(key, f'[{key}]')))

    prompt = f"""You are navigating a codebase to answer a question.

TARGET REPOSITORY: {target_repo}

{setup}

IMPORTANT RULES:
- Use ONLY the tools mentioned above
- Show your work: explain what you're searching for and why
- Be efficient: don't read entire files if you don't need to
- When you find the answer, state it clearly

CHALLENGE ({challenge['category']}, {challenge['difficulty']}):
{question}

Begin your investigation:
"""
    return prompt


def run_codex(prompt: str, timeout: int = 300) -> tuple[str, str, float]:
    """Run Codex with the given prompt, return (stdout, stderr, elapsed_time)."""
    start = time.time()

    try:
        # Use heredoc pattern from CLAUDE.md
        result = subprocess.run(
            ['codex', 'exec', '-'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start
        return result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return "", f"TIMEOUT after {timeout}s", elapsed
    except FileNotFoundError:
        return "", "ERROR: codex command not found", 0.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


def analyze_strategy(stdout: str) -> str:
    """Analyze what tools/patterns the agent used."""
    patterns = []

    if 'grepmap' in stdout.lower():
        patterns.append('used_grepmap')
    if 'rg ' in stdout or 'ripgrep' in stdout.lower():
        patterns.append('used_ripgrep')
    if 'cat ' in stdout:
        patterns.append('used_cat')
    if 'find ' in stdout:
        patterns.append('used_find')

    # Detect repeated searches (potential missing affordance)
    grepmap_count = stdout.lower().count('grepmap')
    rg_count = stdout.count('rg ') + stdout.count('ripgrep')

    if grepmap_count > 3:
        patterns.append(f'repeated_grepmap({grepmap_count})')
    if rg_count > 5:
        patterns.append(f'repeated_grep({rg_count})')

    # Detect file reading patterns
    cat_count = stdout.count('cat ')
    if cat_count > 3:
        patterns.append(f'many_file_reads({cat_count})')

    return ', '.join(patterns) if patterns else 'no_tools_detected'


def run_challenge(
    challenge: Dict,
    tool_config: Dict,
    target_repo: str,
    timeout: int
) -> ChallengeResult:
    """Run a single challenge and return results."""
    prompt = build_prompt(challenge, tool_config, target_repo)
    stdout, stderr, elapsed = run_codex(prompt, timeout)

    # Rough success detection (would need manual validation in real use)
    success = len(stdout) > 100 and 'error' not in stderr.lower()

    return ChallengeResult(
        challenge_id=challenge['id'],
        tool_config=list(tool_config.keys())[0] if isinstance(tool_config, dict) else str(tool_config),
        success=success,
        answer=stdout[-500:] if len(stdout) > 500 else stdout,  # Last 500 chars
        tokens_used=estimate_tokens(stdout + stderr),
        time_seconds=elapsed,
        stdout=stdout,
        stderr=stderr,
        strategy_notes=analyze_strategy(stdout)
    )


def print_result(result: ChallengeResult):
    """Print a single result to console."""
    status = "✓" if result.success else "✗"
    print(f"  {status} {result.challenge_id}")
    print(f"    Time: {result.time_seconds:.1f}s | Tokens: ~{result.tokens_used}")
    print(f"    Strategy: {result.strategy_notes}")
    if not result.success and result.stderr:
        print(f"    Error: {result.stderr[:100]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run agent navigation challenges on a codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('target_repo', help='Path to the repository to navigate')
    parser.add_argument('--tools', choices=['grepmap', 'grep-only', 'both'],
                       default='both', help='Tool configuration to use')
    parser.add_argument('--challenges', nargs='+', help='Specific challenge IDs to run (supports glob)')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per challenge in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Print prompts without running')
    parser.add_argument('--challenges-file', default=None,
                       help='Path to challenges YAML (default: bundled challenges.yaml)')

    args = parser.parse_args()

    # Resolve paths
    target_repo = Path(args.target_repo).resolve()
    if not target_repo.exists():
        print(f"Error: Repository not found: {target_repo}", file=sys.stderr)
        sys.exit(1)

    # Load challenges
    if args.challenges_file:
        challenges_file = Path(args.challenges_file)
    else:
        challenges_file = Path(__file__).parent / 'challenges.yaml'

    if not challenges_file.exists():
        print(f"Error: Challenges file not found: {challenges_file}", file=sys.stderr)
        sys.exit(1)

    data = load_challenges(challenges_file)
    challenges = data['challenges']
    config = data['config']
    tool_configs = config['tool_configs']

    # Filter challenges if specified
    if args.challenges:
        import fnmatch
        filtered = []
        for c in challenges:
            for pattern in args.challenges:
                if fnmatch.fnmatch(c['id'], pattern):
                    filtered.append(c)
                    break
        challenges = filtered

    if not challenges:
        print("No challenges matched the filter", file=sys.stderr)
        sys.exit(1)

    # Get tool config
    tool_config_name = args.tools.replace('-', '_')
    tool_config = tool_configs.get(tool_config_name, tool_configs['both'])

    print(f"=== Agent Navigation Benchmark ===")
    print(f"Target: {target_repo}")
    print(f"Tools: {args.tools}")
    print(f"Challenges: {len(challenges)}")
    print(f"Timeout: {args.timeout}s per challenge")
    print()

    if args.dry_run:
        print("=== DRY RUN - Prompts Only ===\n")
        for challenge in challenges:
            prompt = build_prompt(challenge, tool_config, str(target_repo))
            print(f"--- {challenge['id']} ---")
            print(prompt)
            print()
        return

    # Run challenges
    results: List[ChallengeResult] = []

    for i, challenge in enumerate(challenges, 1):
        print(f"[{i}/{len(challenges)}] Running: {challenge['id']}")
        result = run_challenge(challenge, tool_config, str(target_repo), args.timeout)
        results.append(result)
        print_result(result)

    # Summary
    passed = sum(1 for r in results if r.success)
    total_tokens = sum(r.tokens_used for r in results)
    total_time = sum(r.time_seconds for r in results)

    print("=== Summary ===")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Total tokens: ~{total_tokens}")
    print(f"Total time: {total_time:.1f}s")

    # Strategy analysis
    print("\n=== Strategy Patterns ===")
    all_strategies = [r.strategy_notes for r in results]
    from collections import Counter
    for pattern, count in Counter(' '.join(all_strategies).split(', ')).most_common():
        if pattern:
            print(f"  {pattern}: {count}")

    # Save results
    if args.output:
        run = BenchmarkRun(
            timestamp=datetime.now().isoformat(),
            target_repo=str(target_repo),
            tool_config=args.tools,
            challenges_run=len(results),
            challenges_passed=passed,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            results=[asdict(r) for r in results]
        )

        with open(args.output, 'w') as f:
            json.dump(asdict(run), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
