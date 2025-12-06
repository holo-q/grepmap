#!/usr/bin/env python3
"""
Single Challenge Observer

Runs one challenge and streams output in real-time for observation.
Use this to watch agent behavior and spot missing affordances.

Usage:
    ./observe.py <target_repo> "<question>" [--tools grepmap|grep-only|both]

Examples:
    ./observe.py ~/projects/discore "Where is the main window class defined?"
    ./observe.py ~/projects/discore "How does session data flow to the renderer?" --tools grepmap
    ./observe.py . "What's the most important file in this repo?" --tools grep-only
"""

import argparse
import subprocess
import sys
from pathlib import Path


TOOL_SETUPS = {
    'grepmap': """Available tools:
- grepmap <path> [--focus file1 file2] [--map-tokens N] [--diag]
  Shows ranked map of code structures with annotations:
  [bridge] = load-bearing connector files
  [api] = public interface symbols
  [recent] [high-churn] = git activity badges
  [crystal/rotting/emergent/evolving] = lifecycle phase
  Use --focus to prioritize specific files.
- cat <file> - Read a file
- head -n N <file> - Read first N lines""",

    'grep-only': """Available tools:
- rg <pattern> [path] - Ripgrep search (regex supported)
- rg -l <pattern> - List files containing pattern
- find <path> -name "pattern" - Find files by name
- cat <file> - Read a file
- head -n N <file> - Read first N lines
- tail -n N <file> - Read last N lines
- ls <path> - List directory""",

    'both': """Available tools:
- grepmap <path> [--focus file1 file2] [--map-tokens N] [--diag]
  Shows ranked map with annotations: [bridge] [api] [recent] [crystal/rotting/emergent]
  Start here for orientation. Use --focus to prioritize specific files.
- rg <pattern> [path] - Ripgrep for specific strings/patterns
- cat/head/tail - Read files
- find - Find files by name
- ls - List directories

Strategy hint: Use grepmap first to orient, then rg/cat for details."""
}


def build_prompt(target_repo: str, question: str, tools: str) -> str:
    setup = TOOL_SETUPS.get(tools, TOOL_SETUPS['both'])

    return f"""You are an expert developer navigating an unfamiliar codebase.

TARGET REPOSITORY: {target_repo}

{setup}

RULES:
- Think out loud: explain what you're looking for and why
- Be methodical: start broad, then narrow down
- Show your commands and their output
- When you find the answer, state it clearly with file:line references

YOUR TASK:
{question}

Begin your investigation. Use the tools above to explore the codebase.
"""


def main():
    parser = argparse.ArgumentParser(
        description='Observe an agent navigating a codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('target_repo', help='Path to the repository')
    parser.add_argument('question', help='Navigation question to answer')
    parser.add_argument('--tools', choices=['grepmap', 'grep-only', 'both'],
                       default='both', help='Tool configuration')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds (default: 600)')
    parser.add_argument('--print-prompt', action='store_true',
                       help='Print the prompt and exit')

    args = parser.parse_args()

    target_repo = Path(args.target_repo).resolve()
    if not target_repo.exists():
        print(f"Error: {target_repo} not found", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(str(target_repo), args.question, args.tools)

    if args.print_prompt:
        print(prompt)
        return

    print("=" * 60)
    print(f"Target: {target_repo}")
    print(f"Tools: {args.tools}")
    print(f"Question: {args.question}")
    print("=" * 60)
    print("\n>>> Launching Claude (streaming output)...\n")
    print("-" * 60)

    try:
        # Stream output in real-time using claude -p (print mode)
        process = subprocess.Popen(
            ['claude', '-p', '--output-format', 'text'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=str(target_repo)  # Run in target repo for tool access
        )

        # Send prompt
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(prompt)
        process.stdin.close()

        # Stream output in real-time (flush=True ensures immediate visibility)
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

        process.wait(timeout=args.timeout)

        print("-" * 60)
        print("\n>>> Session complete")

        # Quick analysis
        output = ''.join(output_lines)
        print("\n=== Quick Analysis ===")
        print(f"Output length: {len(output)} chars (~{len(output)//4} tokens)")

        # Count actual tool invocations by looking for shell commands
        # Pattern: command appears after "exec" or at start of line with typical args
        import re

        grepmap_invocations = len(re.findall(r"grepmap\s+[\.\/\w]", output))
        rg_invocations = len(re.findall(r"(?:^|\s)rg\s+-?[nlwi]", output, re.MULTILINE))
        cat_invocations = len(re.findall(r"(?:^|\s)(?:cat|head|tail)\s+", output, re.MULTILINE))
        find_invocations = len(re.findall(r"(?:^|\s)find\s+", output, re.MULTILINE))

        tools_used = []
        if grepmap_invocations:
            tools_used.append(f'grepmap({grepmap_invocations})')
        if rg_invocations:
            tools_used.append(f'ripgrep({rg_invocations})')
        if cat_invocations:
            tools_used.append(f'cat/head/tail({cat_invocations})')
        if find_invocations:
            tools_used.append(f'find({find_invocations})')

        print(f"Tool invocations: {', '.join(tools_used) if tools_used else 'none detected'}")

        # Analysis hints
        if grepmap_invocations > 3:
            print(f"⚠ Many grepmap calls ({grepmap_invocations}) - might need better single-shot coverage")
        if rg_invocations > 5:
            print(f"⚠ Many grep searches ({rg_invocations}) - might need structured search affordance")
        if cat_invocations > 3:
            print(f"⚠ Many file reads ({cat_invocations}) - might need better preview/context")
        if grepmap_invocations == 0 and '--tools grepmap' in ' '.join(sys.argv):
            print("⚠ grepmap not used despite being available - check if it's working in target env")

    except subprocess.TimeoutExpired:
        print(f"\n\n>>> TIMEOUT after {args.timeout}s")
        process.kill()
    except FileNotFoundError:
        print("Error: 'claude' command not found. Install with: npm install -g @anthropic-ai/claude-code", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n>>> Interrupted by user")
        process.kill()


if __name__ == '__main__':
    main()
