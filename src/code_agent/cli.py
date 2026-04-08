"""CLI interface for the coding agent — REPL and one-shot modes."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .agent import Agent
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-agent",
        description="A coding agent powered by Claude API",
    )
    parser.add_argument("prompt", nargs="?", help="One-shot prompt (omit for interactive REPL)")
    parser.add_argument("--model", default=None, help="Model to use (e.g. claude-sonnet-4-6)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic loop turns")
    parser.add_argument("--working-dir", default=None, help="Working directory")

    args = parser.parse_args()

    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    if args.max_turns:
        overrides["max_turns"] = args.max_turns
    if args.working_dir:
        overrides["working_dir"] = os.path.abspath(args.working_dir)
    else:
        overrides["working_dir"] = os.getcwd()

    config = load_config(overrides)
    agent = Agent(config)

    if args.prompt:
        result = asyncio.run(agent.run(args.prompt))
        print(result)
    else:
        asyncio.run(_interactive_loop(agent))


async def _interactive_loop(agent: Agent) -> None:
    print("Code Agent (type 'exit' or Ctrl+C to quit)\n", file=sys.stderr)

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!", file=sys.stderr)
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit"):
            print("Goodbye!", file=sys.stderr)
            break

        try:
            result = await agent.run(stripped)
            print(result)
        except KeyboardInterrupt:
            print("\n[Interrupted]", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

        # Reset context between interactions in REPL mode
        agent.reset()
