"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class ShellTool(BaseTool):
    name = "run_command"
    description = (
        "Execute a shell command and return stdout/stderr. "
        "Use for running tests, build tools, git commands, package managers, etc. "
        "Commands run in the agent's working directory."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Defaults to 120.",
                "default": 120,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Defaults to agent working directory.",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]
        timeout = kwargs.get("timeout", 120)
        cwd = kwargs.get("cwd")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                parts.append(f"stderr:\n{stderr.decode('utf-8', errors='replace')}")
            parts.append(f"exit_code: {proc.returncode}")

            return ToolResult(
                tool_use_id="",
                content="\n".join(parts),
                is_error=proc.returncode != 0,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_use_id="",
                content=f"Command timed out after {timeout}s",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Error executing command: {e}",
                is_error=True,
            )


register_tool(ShellTool())
