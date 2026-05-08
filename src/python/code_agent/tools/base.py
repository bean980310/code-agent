"""Base tool class and global tool registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import ToolResult

_TOOL_REGISTRY: dict[str, BaseTool] = {}


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given input parameters."""
        ...

    def to_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def register_tool(tool: BaseTool) -> BaseTool:
    """Register a tool instance in the global registry."""
    _TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> BaseTool | None:
    """Look up a tool by name."""
    return _TOOL_REGISTRY.get(name)


def get_all_tools() -> list[BaseTool]:
    """Return all registered tools."""
    return list(_TOOL_REGISTRY.values())


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON-schema tool definitions for the API, sorted by name for cache stability."""
    return sorted(
        [t.to_definition() for t in _TOOL_REGISTRY.values()],
        key=lambda d: d["name"],
    )
