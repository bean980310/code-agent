"""File write and edit tool."""

from __future__ import annotations

import os
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class FileWriteTool(BaseTool):
    name = "write_file"
    description = (
        "Write content to a file. Creates the file if it doesn't exist. "
        "Can either overwrite the entire file or perform a targeted edit "
        "by replacing old_string with new_string."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Full content to write (for creating or overwriting a file)",
            },
            "old_string": {
                "type": "string",
                "description": "String to find and replace (for targeted edits). Must be unique in the file.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement string (used with old_string)",
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs["path"]
        content = kwargs.get("content")
        old_string = kwargs.get("old_string")
        new_string = kwargs.get("new_string")

        # Targeted edit mode
        if old_string is not None:
            if new_string is None:
                return ToolResult(
                    tool_use_id="",
                    content="Error: new_string is required when using old_string",
                    is_error=True,
                )
            return await self._edit_file(path, old_string, new_string)

        # Full write mode
        if content is None:
            return ToolResult(
                tool_use_id="",
                content="Error: Either 'content' or 'old_string'+'new_string' must be provided",
                is_error=True,
            )

        return await self._write_file(path, content)

    async def _write_file(self, path: str, content: str) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(tool_use_id="", content=f"Successfully wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error writing file: {e}", is_error=True)

    async def _edit_file(self, path: str, old_string: str, new_string: str) -> ToolResult:
        if not os.path.isfile(path):
            return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

        try:
            with open(path, "r", encoding="utf-8") as f:
                file_content = f.read()

            count = file_content.count(old_string)
            if count == 0:
                return ToolResult(
                    tool_use_id="",
                    content="Error: old_string not found in file",
                    is_error=True,
                )
            if count > 1:
                return ToolResult(
                    tool_use_id="",
                    content=f"Error: old_string found {count} times (must be unique). Provide more context.",
                    is_error=True,
                )

            new_content = file_content.replace(old_string, new_string, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(tool_use_id="", content=f"Successfully edited {path}")
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error editing file: {e}", is_error=True)


register_tool(FileWriteTool())
