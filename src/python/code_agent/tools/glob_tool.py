"""File pattern matching tool using glob."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class GlobTool(BaseTool):
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports patterns like '**/*.py', 'src/**/*.ts', '*.json'. "
        "Returns matching file paths sorted by modification time (newest first)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g. '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to working directory.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results. Defaults to 100.",
                "default": 100,
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        search_dir = kwargs.get("path", ".")
        limit = kwargs.get("limit", 100)

        try:
            base = Path(search_dir)
            if not base.is_dir():
                return ToolResult(
                    tool_use_id="",
                    content=f"Error: Directory not found: {search_dir}",
                    is_error=True,
                )

            matches = sorted(
                base.glob(pattern),
                key=lambda p: os.path.getmtime(p) if p.exists() else 0,
                reverse=True,
            )

            # Filter to files only
            files = [str(m) for m in matches if m.is_file()]

            if not files:
                return ToolResult(tool_use_id="", content=f"No files matching pattern '{pattern}'")

            total = len(files)
            files = files[:limit]
            result = "\n".join(files)

            if total > limit:
                result += f"\n\n... ({total - limit} more files)"

            return ToolResult(tool_use_id="", content=result)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)


register_tool(GlobTool())
