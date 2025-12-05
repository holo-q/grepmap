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
- grepmap <path> [--chat-files file1 file2] [--map-tokens N] [--tree]
  Shows ranked map of important code structures. Use --chat-files to focus on specific files.
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
- grepmap <path> [--chat-files file1 file2] [--map-tokens N] [--tree]
  Shows ranked map of important code structures. Start here for orientation.
- rg <pattern> [path] - Ripgrep for specific strings/patterns
- cat/head/tail - Read files
- find - Find files by name
- ls - List directories

Strategy hint: Use grepmap first to find important files, then rg/cat for details."""
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
    print("\n>>> Launching Codex (streaming output)...\n")
    print("-" * 60)

    try:
        # Stream output in real-time
        process = subprocess.Popen(
            ['codex', 'exec', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Send prompt
        process.stdin.write(prompt)
        process.stdin.close()

        # Stream output
        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait(timeout=args.timeout)

        print("-" * 60)
        print("\n>>> Session complete")

        # Quick analysis
        output = ''.join(output_lines)
        print("\n=== Quick Analysis ===")
        print(f"Output length: {len(output)} chars (~{len(output)//4} tokens)")

        tools_used = []
        if 'grepmap' in output.lower():
            tools_used.append('grepmap')
        if 'rg ' in output:
            tools_used.append('ripgrep')
        if 'cat ' in output:
            tools_used.append('cat')
        if 'find ' in output:
            tools_used.append('find')

        print(f"Tools used: {', '.join(tools_used) if tools_used else 'none detected'}")

        # Count tool invocations
        grepmap_count = output.lower().count('grepmap')
        rg_count = output.count('rg ')
        cat_count = output.count('cat ')

        if grepmap_count > 2:
            print(f"⚠ Multiple grepmap calls ({grepmap_count}) - might need better single-shot coverage")
        if rg_count > 5:
            print(f"⚠ Many grep searches ({rg_count}) - might need structured search affordance")
        if cat_count > 3:
            print(f"⚠ Many file reads ({cat_count}) - might need better preview/context")

    except subprocess.TimeoutExpired:
        print(f"\n\n>>> TIMEOUT after {args.timeout}s")
        process.kill()
    except FileNotFoundError:
        print("Error: 'codex' command not found. Is it installed?", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n>>> Interrupted by user")
        process.kill()


if __name__ == '__main__':
    main()
