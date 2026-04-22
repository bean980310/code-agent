"""Context window manager — tracks messages and handles summarization."""

from __future__ import annotations

from typing import Any

from ..client import extract_text_from_blocks
from ..types import AgentConfig


def _serialize_content(content: Any) -> str:
    """Serialize message content to a string for token estimation."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use: {block.get('name', '')}({block.get('input', '')})]")
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result: {block.get('content', '')[:500]}]")
                else:
                    parts.append(str(block))
            elif hasattr(block, "type"):
                # Anthropic SDK objects
                if block.type == "text":
                    parts.append(block.text)
                elif block.type == "tool_use":
                    parts.append(f"[tool_use: {block.name}({block.input})]")
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


class ContextManager:
    """Manages conversation messages and tracks context window usage."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.messages: list[dict[str, Any]] = []
        self._token_estimate: int = 0

    def add_user_message(self, content: str | list[dict]) -> None:
        self.messages.append({"role": "user", "content": content})
        self._update_token_estimate()

    def add_assistant_message(self, content: Any) -> None:
        """Store the full response content (preserves tool_use blocks)."""
        self.messages.append({"role": "assistant", "content": content})
        self._update_token_estimate()

    def add_tool_results(self, results: list[dict]) -> None:
        """Add tool results as a user message."""
        self.messages.append({"role": "user", "content": results})
        self._update_token_estimate()

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    def get_last_assistant_content(self) -> Any:
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def estimated_tokens(self) -> int:
        return self._token_estimate

    def _update_token_estimate(self) -> None:
        """Rough heuristic: ~4 characters per token."""
        total_chars = sum(len(_serialize_content(m["content"])) for m in self.messages)
        self._token_estimate = total_chars // 4

    async def summarize(self, client: Any) -> None:
        """Compress older messages to stay within context window.

        Strategy: keep last N turn pairs intact, summarize the rest.
        """
        keep_recent = 20  # ~10 turns (user + assistant pairs)

        if len(self.messages) <= keep_recent:
            return

        old_messages = self.messages[:-keep_recent]
        recent_messages = self.messages[-keep_recent:]

        # Serialize old messages for summarization
        serialized = []
        for msg in old_messages:
            role = msg["role"]
            text = _serialize_content(msg["content"])
            serialized.append(f"[{role}]: {text}")
        old_text = "\n\n".join(serialized)

        # Use Haiku for cheap summarization
        summary_response = await client.create_message_simple(
            messages=[{
                "role": "user",
                "content": (
                    "Summarize the following conversation history concisely. "
                    "Preserve: key decisions made, files read/modified, errors encountered, "
                    "and the current state of the task. Format as structured notes.\n\n"
                    f"{old_text}"
                ),
            }],
            model=self.config.resolved_summary_model(),
            max_tokens=4096,
        )

        summary_text = extract_text_from_blocks(summary_response.content)

        # Replace old messages with summary
        self.messages = [
            {
                "role": "user",
                "content": f"[Previous conversation summary]\n{summary_text}",
            },
            {
                "role": "assistant",
                "content": "Understood. I have the context from our previous conversation and will continue from here.",
            },
            *recent_messages,
        ]
        self._update_token_estimate()

    def clear(self) -> None:
        self.messages.clear()
        self._token_estimate = 0
