"""CLI interface for the coding agent — REPL and one-shot modes."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex

from .agent import Agent
from .config import load_config
from .types import normalize_provider
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
        help="Model provider to use (anthropic, openai, google, huggingface, ollama, lmstudio)",
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
    if args.max_turns is not None:
        if args.max_turns <= 0:
            parser.error("--max-turns must be a positive integer")
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
        if stripped.startswith("/"):
            command_result = _handle_repl_command(agent, stripped)
            if command_result == "exit":
                ui.note("Goodbye.")
                break
            continue

        try:
            await agent.run(stripped)
        except KeyboardInterrupt:
            ui.error("Interrupted.")
        except Exception as e:
            ui.error(str(e))


def _handle_repl_command(agent: Agent, raw_command: str) -> bool | str:
    """Handle REPL slash commands. Returns True or 'exit' when handled."""
    ui = agent.ui

    try:
        tokens = shlex.split(raw_command)
    except ValueError as exc:
        ui.error(f"Invalid command syntax: {exc}")
        return True

    if not tokens:
        return True

    command = tokens[0].lower()

    if command in {"/help", "/?"}:
        ui.note("Commands: /provider <anthropic|openai|google|huggingface|ollama|lmstudio>, /model <name>, /summary-model <name>, /subagent-model <name>, /max-turns <n>, /working-dir <path>, /reset, /config, /exit")
        return True

    if command in {"/exit", "/quit"}:
        return "exit"

    if command == "/config":
        _show_runtime_config(agent)
        return True

    if command == "/reset":
        agent.reset()
        ui.note("Conversation context cleared.")
        return True

    if command == "/provider":
        if len(tokens) != 2:
            ui.error("Usage: /provider <anthropic|openai|google|huggingface|ollama|lmstudio>")
            return True

        try:
            provider = normalize_provider(tokens[1])
        except ValueError as exc:
            ui.error(str(exc))
            return True

        config = agent.update_runtime_config(provider=provider, reset_model_overrides=True)
        agent.reset()
        ui.note(f"Provider set to {config.provider}. Model reset to default: {config.resolved_model()}. Context cleared.")
        return True

    model_commands = {
        "/model": ("model", "resolved_model"),
        "/summary-model": ("summary_model", "resolved_summary_model"),
        "/subagent-model": ("subagent_model", "resolved_subagent_model"),
    }
    if command in model_commands:
        if len(tokens) < 2:
            ui.error(f"Usage: {command} <provider-specific-model-name>")
            return True

        field, resolver = model_commands[command]
        model = " ".join(tokens[1:]).strip()
        config = agent.update_runtime_config(**{field: model})
        resolved = getattr(config, resolver)()
        ui.note(f"{field} set to {resolved} for provider {config.provider}.")
        return True

    if command == "/max-turns":
        if len(tokens) != 2:
            ui.error("Usage: /max-turns <positive-integer>")
            return True

        try:
            max_turns = int(tokens[1])
        except ValueError:
            ui.error("`max_turns` must be an integer.")
            return True

        if max_turns <= 0:
            ui.error("`max_turns` must be greater than 0.")
            return True

        config = agent.update_runtime_config(max_turns=max_turns)
        ui.note(f"Max turns set to {config.max_turns}.")
        return True

    if command == "/working-dir":
        if len(tokens) < 2:
            ui.error("Usage: /working-dir <path>")
            return True

        path = os.path.abspath(os.path.expanduser(" ".join(tokens[1:]).strip()))
        if not os.path.isdir(path):
            ui.error(f"Not a directory: {path}")
            return True

        config = agent.update_runtime_config(working_dir=path)
        ui.note(f"Working directory set to {config.working_dir}.")
        return True

    ui.error(f"Unknown command: {command}. Try /help.")
    return True


def _show_runtime_config(agent: Agent) -> None:
    config = agent.config
    lines = [
        f"provider={config.provider}",
        f"model={config.resolved_model()}",
        f"summary model={config.resolved_summary_model()}",
        f"subagent model={config.resolved_subagent_model()}",
        f"max turns={config.max_turns}",
        f"max tokens={config.max_tokens}",
        f"summarize threshold={config.summarize_threshold}",
        f"working dir={config.working_dir}",
    ]
    agent.ui.note("\n".join(lines))
