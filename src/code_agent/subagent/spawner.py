"""Sub-agent spawner — runs child agents with isolated contexts."""

from __future__ import annotations

import sys
from typing import Any

from ..client import create_client, extract_text_from_blocks
from ..tools.base import get_tool, get_tool_definitions
from ..types import AgentConfig


class SubAgentSpawner:
    """Spawns sub-agents with isolated conversation contexts and restricted tool sets."""

    def __init__(self, parent_config: AgentConfig):
        self.parent_config = parent_config

    async def spawn(
        self,
        task: str,
        *,
        system_prompt: str = "",
        tool_names: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        max_turns: int = 20,
    ) -> str:
        """Spawn a sub-agent with an isolated context.

        Args:
            task: The focused task description.
            system_prompt: System prompt for this sub-agent.
            tool_names: Restrict to these tools. None = read-only defaults.
            model: Model override (defaults to haiku for cost efficiency).
            max_tokens: Max tokens per API call.
            max_turns: Max loop iterations.

        Returns:
            The sub-agent's final text response.
        """
        child_config = self.parent_config.with_overrides(
            model=model or self.parent_config.resolved_subagent_model(),
            max_tokens=max_tokens,
        )
        client = create_client(child_config)
        model_name = child_config.resolved_model()

        # Determine tool set
        if tool_names is None:
            tool_names = ["read_file", "glob", "grep", "git"]

        all_defs = get_tool_definitions()
        tools = [d for d in all_defs if d["name"] in tool_names]

        if not system_prompt:
            system_prompt = (
                "You are a focused sub-agent. Complete the given task concisely. "
                "Only use the provided tools when necessary. "
                "Return your findings as structured text."
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        system_blocks = [{"type": "text", "text": system_prompt}]

        print(f"[subagent] Spawning ({model_name}) task: {task[:80]}...", file=sys.stderr)

        for turn in range(max_turns):
            response = await client.create_message(
                messages=messages,
                system=system_blocks,
                tools=tools,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                result = extract_text_from_blocks(response.content)
                print(f"[subagent] Completed in {turn + 1} turns", file=sys.stderr)
                return result

            if response.stop_reason == "tool_use":
                tool_results: list[dict] = []
                for block in response.content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_impl = get_tool(block["name"])
                    if tool_impl and block["name"] in tool_names:
                        try:
                            result = await tool_impl.execute(**block["input"])
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": result.content,
                                "is_error": result.is_error,
                            })
                        except Exception as e:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": f"Error: {e}",
                                "is_error": True,
                            })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"Tool '{block['name']}' not available to this sub-agent",
                            "is_error": True,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            break

        # Fallback: extract whatever text we have
        return extract_text_from_blocks(messages[-1].get("content", ""))
