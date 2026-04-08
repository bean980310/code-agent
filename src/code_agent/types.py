"""Shared type definitions for the code agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class PlanStep:
    id: int
    description: str
    status: str = "pending"  # pending | in_progress | done | failed
    result: str | None = None


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 16000
    max_turns: int = 50
    summarize_threshold: int = 150_000
    memory_dir: str = ".code-agent/memory"
    working_dir: str = "."
