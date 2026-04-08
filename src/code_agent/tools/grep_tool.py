"""Content search tool using regex."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class GrepTool(BaseTool):
    name = "grep"
    description = (
        "Search file contents for a regex pattern. "
        "Returns matching lines with file paths and line numbers. "
        "Use glob parameter to filter which files to search."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search. Defaults to working directory.",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '**/*.py'). Defaults to '**/*'.",
                "default": "**/*",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case insensitive search. Defaults to false.",
                "default": False,
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines before and after each match. Defaults to 0.",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches. Defaults to 50.",
                "default": 50,
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        search_path = kwargs.get("path", ".")
        file_glob = kwargs.get("glob", "**/*")
        case_insensitive = kwargs.get("case_insensitive", False)
        context_lines = kwargs.get("context_lines", 0)
        limit = kwargs.get("limit", 50)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(tool_use_id="", content=f"Invalid regex: {e}", is_error=True)

        base = Path(search_path)
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = [f for f in base.glob(file_glob) if f.is_file()]
        else:
            return ToolResult(tool_use_id="", content=f"Error: Path not found: {search_path}", is_error=True)

        results: list[str] = []
        match_count = 0

        for file_path in sorted(files):
            if match_count >= limit:
                break
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (OSError, PermissionError):
                continue

            for i, line in enumerate(lines):
                if match_count >= limit:
                    break
                if regex.search(line):
                    match_count += 1
                    # Add context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    for j in range(start, end):
                        prefix = ">" if j == i else " "
                        results.append(f"{file_path}:{j + 1}{prefix} {lines[j].rstrip()}")
                    if context_lines > 0:
                        results.append("---")

        if not results:
            return ToolResult(tool_use_id="", content=f"No matches for pattern '{pattern}'")

        result = "\n".join(results)
        if match_count >= limit:
            result += f"\n\n... (results limited to {limit} matches)"

        return ToolResult(tool_use_id="", content=result)


register_tool(GrepTool())
