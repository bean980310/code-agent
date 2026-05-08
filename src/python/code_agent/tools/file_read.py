"""File read tool."""

from __future__ import annotations

import os
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class FileReadTool(BaseTool):
    name = "read_file"
    description = (
        "Read the contents of a file at the given path. "
        "Returns file content with line numbers. "
        "Use offset and limit to read specific portions of large files."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative to working directory)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-based). Defaults to 1.",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Defaults to 2000.",
                "default": 2000,
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs["path"]
        offset = max(1, kwargs.get("offset", 1))
        limit = kwargs.get("limit", 2000)

        if not os.path.exists(path):
            return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

        if os.path.isdir(path):
            return ToolResult(tool_use_id="", content=f"Error: Path is a directory: {path}", is_error=True)

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = offset - 1
            selected = lines[start_idx : start_idx + limit]

            if not selected:
                return ToolResult(
                    tool_use_id="",
                    content=f"File has {total_lines} lines. Offset {offset} is beyond end of file.",
                )

            numbered = [
                f"{i + offset:>6}\t{line.rstrip()}"
                for i, line in enumerate(selected)
            ]
            result = "\n".join(numbered)

            if start_idx + limit < total_lines:
                result += f"\n\n... ({total_lines - start_idx - limit} more lines)"

            return ToolResult(tool_use_id="", content=result)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error reading file: {e}", is_error=True)


register_tool(FileReadTool())
