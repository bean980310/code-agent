"""CLI interface for the coding agent — REPL and one-shot modes."""

from __future__ import annotations

import argparse
import asyncio
import os

from .agent import Agent
from .config import load_config
from .ui import TerminalUI, set_terminal_ui


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-agent",
        description="A coding agent with pluggable model providers",
    )
    parser.add_argument("prompt", nargs="?", help="One-shot prompt (omit for interactive REPL)")
    parser.add_argument(
        "--provider",
        default=None,
        help="Model provider to use (anthropic, openai, google)",
    )
    parser.add_argument("--model", default=None, help="Model to use (provider-specific)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic loop turns")
    parser.add_argument("--working-dir", default=None, help="Working directory")

    args = parser.parse_args()

    overrides: dict = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.max_turns:
        overrides["max_turns"] = args.max_turns
    if args.working_dir:
        overrides["working_dir"] = os.path.abspath(args.working_dir)
    else:
        overrides["working_dir"] = os.getcwd()

    config = load_config(overrides)
    ui = TerminalUI()
    set_terminal_ui(ui)
    agent = Agent(config)

    ui.show_banner(
        provider=config.provider,
        model=config.resolved_model(),
        working_dir=config.working_dir,
        interactive=args.prompt is None,
    )

    if args.prompt:
        asyncio.run(agent.run(args.prompt))
    else:
        asyncio.run(_interactive_loop(agent))


async def _interactive_loop(agent: Agent) -> None:
    ui = agent.ui

    while True:
        try:
            user_input = ui.prompt()
        except (EOFError, KeyboardInterrupt):
            ui.note("Goodbye.")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit"):
            ui.note("Goodbye.")
            break

        try:
            await agent.run(stripped)
        except KeyboardInterrupt:
            ui.error("Interrupted.")
        except Exception as e:
            ui.error(str(e))

        # Reset context between interactions in REPL mode
        agent.reset()
