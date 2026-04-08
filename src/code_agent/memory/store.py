"""File-based persistent memory store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class MemoryStore:
    """Stores session summaries and explicit memories on disk."""

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "index.json"

    def get_relevant_context(self, query: str) -> str:
        """Load recent memory entries as context for the system prompt.

        Currently returns the last 20 session summaries.
        Can be extended with keyword/embedding search for larger stores.
        """
        entries = self._load_index()
        if not entries:
            return ""

        recent = entries[-20:]
        lines = ["Previous session notes:"]
        for entry in recent:
            lines.append(f"- [{entry['timestamp']}] {entry['summary'][:200]}")
        return "\n".join(lines)

    async def save_session_summary(self, user_input: str, result: str) -> None:
        """Save a summary of this interaction."""
        entries = self._load_index()
        entries.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": user_input[:200],
            "summary": result[:500],
        })
        # Keep last 100 entries
        if len(entries) > 100:
            entries = entries[-100:]
        self._save_index(entries)

    def save_explicit(self, key: str, content: str) -> None:
        """Save a named memory entry (e.g. project conventions, architecture notes)."""
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        file_path = self.memory_dir / f"{safe_key}.md"
        file_path.write_text(content, encoding="utf-8")

    def load_explicit(self, key: str) -> str | None:
        """Load a named memory entry."""
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        file_path = self.memory_dir / f"{safe_key}.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    def list_explicit(self) -> list[str]:
        """List all named memory keys."""
        return [
            p.stem for p in self.memory_dir.glob("*.md")
        ]

    def _load_index(self) -> list[dict]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self, entries: list[dict]) -> None:
        self.index_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
