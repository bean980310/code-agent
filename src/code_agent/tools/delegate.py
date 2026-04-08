"""Sub-agent delegation tool."""

from __future__ import annotations

from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class DelegateTaskTool(BaseTool):
    name = "delegate_task"
    description = (
        "Delegate a focused sub-task to a specialized sub-agent with an isolated context. "
        "The sub-agent runs its own agentic loop and returns only the final result. "
        "Use for read-heavy exploration, code review, or independent research tasks. "
        "Sub-agents default to read-only tools (read_file, glob, grep, git)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of the sub-task to complete",
            },
            "context": {
                "type": "string",
                "description": "Relevant context the sub-agent needs (file paths, decisions so far, etc.)",
                "default": "",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tool names available to the sub-agent. "
                    "Defaults to ['read_file', 'glob', 'grep', 'git']. "
                    "Add 'write_file' or 'run_command' for write tasks."
                ),
            },
            "model": {
                "type": "string",
                "description": "Model for the sub-agent. Defaults to claude-haiku-4-5 for cost efficiency.",
            },
        },
        "required": ["task"],
    }

    # The spawner is injected by the Agent after initialization
    _spawner: Any = None

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._spawner is None:
            return ToolResult(
                tool_use_id="",
                content="Error: Sub-agent spawner not initialized",
                is_error=True,
            )

        task = kwargs["task"]
        context = kwargs.get("context", "")
        tool_names = kwargs.get("tools")
        model = kwargs.get("model")

        full_task = task
        if context:
            full_task = f"{task}\n\nContext:\n{context}"

        try:
            result = await self._spawner.spawn(
                task=full_task,
                tool_names=tool_names,
                model=model,
            )
            return ToolResult(tool_use_id="", content=result)
        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Sub-agent error: {e}",
                is_error=True,
            )


_delegate_tool = DelegateTaskTool()
register_tool(_delegate_tool)
