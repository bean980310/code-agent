"""System prompt builder with cache-stable block ordering."""

from __future__ import annotations

from ..types import Plan


CORE_SYSTEM_PROMPT = """\
You are a coding agent. You help users with software engineering tasks \
by reading files, writing code, running commands, and iterating on feedback.

## Capabilities
- Read and write files in the working directory
- Execute shell commands (build, test, git, package managers)
- Search files by name patterns (glob) and content (grep)
- Check git status, diff, and log
- Create and follow multi-step plans
- Delegate focused sub-tasks to specialized sub-agents

## Guidelines
- Always read a file before editing it
- Run tests after making changes when possible
- Explain your reasoning briefly before taking action
- If a command fails, analyze the error before retrying
- Ask the user for clarification when requirements are ambiguous
- Break complex tasks into steps using the planning tool
- For read-heavy exploration, consider delegating to a sub-agent
"""


def build_system_prompt(
    working_dir: str,
    memory_context: str = "",
    plan: Plan | None = None,
) -> list[dict]:
    """Build system prompt blocks optimized for prompt caching.

    Block 1 (cache_control breakpoint): Core identity — never changes within a session.
    Block 2: Session context (working dir, memory, plan) — stable within a turn.
    """
    blocks: list[dict] = [
        {
            "type": "text",
            "text": CORE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    # Session-varying context (placed after the cache breakpoint)
    session_parts: list[str] = [f"Working directory: {working_dir}"]

    if memory_context:
        session_parts.append(f"\n## Relevant Memory\n{memory_context}")

    if plan:
        session_parts.append(f"\n## Current Plan\n{_format_plan(plan)}")

    blocks.append({
        "type": "text",
        "text": "\n".join(session_parts),
    })

    return blocks


def _format_plan(plan: Plan) -> str:
    markers = {
        "pending": "[ ]",
        "in_progress": "[>]",
        "done": "[x]",
        "failed": "[!]",
    }
    lines = [f"Goal: {plan.goal}"]
    for step in plan.steps:
        marker = markers.get(step.status, "[ ]")
        lines.append(f"  {marker} {step.id}. {step.description}")
        if step.result:
            lines.append(f"       Result: {step.result}")
    return "\n".join(lines)
