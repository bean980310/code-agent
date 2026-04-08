"""Tool to ask the user for input or clarification."""

from __future__ import annotations

from typing import Any

from .base import BaseTool, register_tool
from ..types import ToolResult


class AskUserTool(BaseTool):
    name = "ask_user"
    description = (
        "Ask the user a question to get clarification or additional input. "
        "Use when requirements are ambiguous or you need a decision from the user."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
        },
        "required": ["question"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        question = kwargs["question"]
        try:
            print(f"\n[Agent asks]: {question}")
            answer = input("> ")
            return ToolResult(tool_use_id="", content=f"User answered: {answer}")
        except (EOFError, KeyboardInterrupt):
            return ToolResult(tool_use_id="", content="User did not provide an answer", is_error=True)


register_tool(AskUserTool())
