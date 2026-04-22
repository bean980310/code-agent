"""Shared type definitions for the code agent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


SUPPORTED_PROVIDERS = {"anthropic", "openai", "google"}
PROVIDER_ALIASES = {
    "claude": "anthropic",
    "gemini": "google",
    "google-genai": "google",
}
DEFAULT_PROVIDER_MODELS = {
    "anthropic": {
        "main": "claude-sonnet-4-6",
        "summary": "claude-haiku-4-5-20251001",
        "subagent": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "main": "gpt-4.1",
        "summary": "gpt-4.1-mini",
        "subagent": "gpt-4.1-mini",
    },
    "google": {
        "main": "gemini-2.5-pro",
        "summary": "gemini-2.5-flash",
        "subagent": "gemini-2.5-flash",
    },
}


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
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class AgentResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


def normalize_provider(provider: str) -> str:
    normalized = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {supported}")
    return normalized


@dataclass
class AgentConfig:
    provider: str = "anthropic"
    model: str | None = None
    summary_model: str | None = None
    subagent_model: str | None = None
    max_tokens: int = 16000
    max_turns: int = 50
    summarize_threshold: int = 150_000
    memory_dir: str = ".code-agent/memory"
    working_dir: str = "."

    def __post_init__(self) -> None:
        self.provider = normalize_provider(self.provider)

    def resolved_model(self) -> str:
        return self.model or DEFAULT_PROVIDER_MODELS[self.provider]["main"]

    def resolved_summary_model(self) -> str:
        return self.summary_model or DEFAULT_PROVIDER_MODELS[self.provider]["summary"]

    def resolved_subagent_model(self) -> str:
        return self.subagent_model or DEFAULT_PROVIDER_MODELS[self.provider]["subagent"]

    def with_overrides(self, **kwargs: Any) -> AgentConfig:
        return replace(self, **kwargs)
