"""Core agentic loop — the heart of the coding agent."""

from __future__ import annotations

import sys
from typing import Any

from .client import (
    ProviderAPIError,
    ProviderConnectionError,
    create_client,
    extract_text_from_blocks,
)
from .config import load_config
from .context.manager import ContextManager
from .context.prompt import build_system_prompt
from .memory.store import MemoryStore
from .planning.planner import Planner
from .subagent.spawner import SubAgentSpawner
from .tools.base import get_tool, get_tool_definitions
from .tools.delegate import _delegate_tool
from .types import AgentConfig

# Maximum consecutive API errors before giving up
MAX_API_RETRIES = 3


class Agent:
    """Main agent that runs the observe-think-act-feedback loop."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or load_config()
        self.client = create_client(self.config)
        self.context = ContextManager(self.config)
        self.planner = Planner()
        self.memory = MemoryStore(self.config.memory_dir)
        self.spawner = SubAgentSpawner(self.config)
        self.streaming = True  # Enable streaming output by default
        # Wire spawner into the delegate tool
        _delegate_tool._spawner = self.spawner

    async def run(self, user_input: str) -> str:
        """Run the full agentic loop for a user request.

        Returns the final text response from the agent.
        """
        # Load memory context
        memory_context = self.memory.get_relevant_context(user_input)

        # Get current plan
        current_plan = self.planner.current_plan

        # Build system prompt
        system_prompt = build_system_prompt(
            working_dir=self.config.working_dir,
            memory_context=memory_context,
            plan=current_plan,
        )

        # Add user message to context
        self.context.add_user_message(user_input)

        turn_count = 0
        consecutive_errors = 0

        while turn_count < self.config.max_turns:
            turn_count += 1

            # Check if context needs summarization
            if self.context.estimated_tokens() > self.config.summarize_threshold:
                print("[context] Summarizing conversation history...", file=sys.stderr)
                await self.context.summarize(self.client)

            # Call Claude API with error recovery
            try:
                response = await self._call_api(system_prompt)
                consecutive_errors = 0
            except ProviderAPIError as e:
                consecutive_errors += 1
                status = e.status_code if e.status_code is not None else "unknown"
                print(f"[error] API error ({status}): {e.message}", file=sys.stderr)
                if consecutive_errors >= MAX_API_RETRIES:
                    return f"Failed after {MAX_API_RETRIES} API errors. Last error: {e.message}"
                if e.status_code in {429, 503, 529}:  # Rate limit or overloaded
                    import asyncio
                    await asyncio.sleep(2 ** consecutive_errors)
                continue
            except ProviderConnectionError as e:
                consecutive_errors += 1
                print(f"[error] Connection error: {e}", file=sys.stderr)
                if consecutive_errors >= MAX_API_RETRIES:
                    return f"Connection failed after {MAX_API_RETRIES} retries."
                import asyncio
                await asyncio.sleep(2 ** consecutive_errors)
                continue

            # Log usage for cache debugging
            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0)
            cache_create = getattr(usage, "cache_creation_input_tokens", 0)
            if cache_read or cache_create:
                print(
                    f"[cache] read={cache_read} create={cache_create}",
                    file=sys.stderr,
                )

            # Store assistant response
            self.context.add_assistant_message(response.content)

            # Route based on stop reason
            if response.stop_reason == "end_turn":
                final_text = extract_text_from_blocks(response.content)
                await self.memory.save_session_summary(user_input, final_text)
                return final_text

            if response.stop_reason == "tool_use":
                tool_results = await self._execute_tools(response.content)
                self.context.add_tool_results(tool_results)
                continue

            if response.stop_reason == "max_tokens":
                return extract_text_from_blocks(response.content) + "\n\n[Response truncated due to max tokens]"

            # Unknown stop reason — break to avoid infinite loop
            break

        return extract_text_from_blocks(self.context.get_last_assistant_content())

    async def _call_api(self, system_prompt: list[dict]) -> Any:
        """Call the API with optional streaming."""
        tools = get_tool_definitions()
        messages = self.context.get_messages()

        if self.streaming:
            return await self.client.create_message_streaming(
                messages=messages,
                system=system_prompt,
                tools=tools,
            )
        else:
            return await self.client.create_message(
                messages=messages,
                system=system_prompt,
                tools=tools,
            )

    async def _execute_tools(self, content: Any) -> list[dict]:
        """Execute all tool_use blocks and return tool_result messages."""
        results: list[dict] = []

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            print(f"[tool] {block['name']}({_summarize_input(block['input'])})", file=sys.stderr)

            tool = get_tool(block["name"])
            if tool is None:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error: Unknown tool '{block['name']}'",
                    "is_error": True,
                })
                continue

            try:
                result = await tool.execute(**block["input"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.content,
                    "is_error": result.is_error,
                })
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error executing tool '{block['name']}': {e}",
                    "is_error": True,
                })

        return results

    def reset(self) -> None:
        """Reset conversation state for a new interaction."""
        self.context.clear()

def _summarize_input(input_data: dict) -> str:
    """Create a short summary of tool input for logging."""
    parts: list[str] = []
    for key, value in input_data.items():
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        parts.append(f"{key}={val_str}")
    return ", ".join(parts)
