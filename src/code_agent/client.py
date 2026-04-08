"""Anthropic API client wrapper."""

from __future__ import annotations

import sys
from typing import Any, AsyncIterator

import openai
import openai.types.responses
from openai import AsyncOpenAI

import anthropic
from anthropic import AsyncAnthropic

from google import genai


from .types import AgentConfig


class ClaudeClient:
    """Wrapper around the Anthropic async client with cache-aware defaults."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = AsyncAnthropic()

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """Create a message with tool use support and prompt caching."""
        return await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN,
        )

    async def create_message_streaming(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """Create a message with streaming — prints text tokens in real-time.

        Returns the final assembled Message (same interface as create_message).
        """
        async with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN,
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True, file=sys.stdout)
            print(file=sys.stdout)  # newline after stream
            return await stream.get_final_message()

    async def create_message_simple(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> anthropic.types.Message:
        """Simple message creation without tools (for summarization, sub-agents)."""
        return await self.client.messages.create(
            model=model or self.config.model,
            max_tokens=max_tokens,
            messages=messages,
        )
