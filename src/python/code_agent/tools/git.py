"""Git repository context tools."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class GitTool(BaseTool):
    name = "git"
    description = (
        "Run git commands to inspect repository state. "
        "Supports: status, diff, log, branch, show, blame. "
        "For safety, only read-only git operations are allowed through this tool. "
        "Use run_command for write operations like commit, push, etc."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "subcommand": {
                "type": "string",
                "description": "Git subcommand: status, diff, log, branch, show, blame",
                "enum": ["status", "diff", "log", "branch", "show", "blame"],
            },
            "args": {
                "type": "string",
                "description": "Additional arguments (e.g. '--oneline -10' for log, 'HEAD~3' for diff)",
                "default": "",
            },
            "cwd": {
                "type": "string",
                "description": "Repository directory. Defaults to working directory.",
            },
        },
        "required": ["subcommand"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        subcommand = kwargs["subcommand"]
        args = kwargs.get("args", "")
        cwd = kwargs.get("cwd")

        command = f"git {subcommand} {args}".strip()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                err = stderr.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    return ToolResult(tool_use_id="", content=f"git error: {err}", is_error=True)
                output += f"\n{err}"

            if not output.strip():
                return ToolResult(tool_use_id="", content="(no output)")

            return ToolResult(tool_use_id="", content=output)
        except asyncio.TimeoutError:
            return ToolResult(tool_use_id="", content="Git command timed out", is_error=True)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)


register_tool(GitTool())
