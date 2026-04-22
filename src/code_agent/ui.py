"""Terminal UI helpers for a more polished CLI experience."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    from rich import box
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - only happens before dependencies are installed
    Console = Markdown = Panel = Text = Theme = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment]
    RICH_AVAILABLE = False


class TerminalUI:
    """Shared terminal renderer used across the CLI, agent, and tools."""

    def __init__(self) -> None:
        self._stream_open = False

        if RICH_AVAILABLE:
            theme = Theme(
                {
                    "brand": "bold cyan",
                    "accent": "bold bright_white",
                    "muted": "dim white",
                    "info": "cyan",
                    "success": "green",
                    "warning": "yellow",
                    "danger": "bold red",
                    "response": "bright_cyan",
                }
            )
            self.console = Console(theme=theme)
            self.error_console = Console(stderr=True, theme=theme)
            self._badge_styles = {
                "info": "black on cyan",
                "success": "black on green",
                "warning": "black on yellow",
                "danger": "bold white on red",
                "accent": "black on bright_white",
            }
        else:
            self.console = None
            self.error_console = None
            self._badge_styles = {}

    def show_banner(self, *, provider: str, model: str, working_dir: str, interactive: bool) -> None:
        """Render the top-level CLI banner."""
        if not RICH_AVAILABLE:
            mode = "interactive" if interactive else "one-shot"
            print(f"Code Agent [{mode}]")
            print(f"provider={provider} model={model}")
            print(f"cwd={working_dir}")
            print()
            return

        mode = "interactive session" if interactive else "one-shot run"

        banner = Text(justify="center")
        banner.append("CODE", style="brand")
        banner.append(" AGENT", style="accent")
        banner.append("\n")
        banner.append(mode.upper(), style="muted")
        banner.append("\n")
        banner.append(f"{provider} / {model}", style="info")
        banner.append("\n")
        banner.append(self._shorten_path(working_dir), style="muted")

        self.console.print(
            Panel(
                banner,
                border_style="brand",
                box=box.HEAVY,
                padding=(1, 3),
            )
        )
        if interactive:
            self.note("Type 'exit' or press Ctrl+C to quit.")

    def prompt(self) -> str:
        """Prompt the user for the next REPL input."""
        self.finish_stream()
        if not RICH_AVAILABLE:
            return input("> ")
        return self.console.input("[brand]agent[/brand] [muted]>[/muted] ")

    def note(self, message: str) -> None:
        self._event("info", "note", message)

    def tool(self, message: str) -> None:
        self._event("warning", "tool", message, stderr=True)

    def cache(self, message: str) -> None:
        self._event("success", "cache", message, stderr=True)

    def subagent(self, message: str) -> None:
        self._event("accent", "subagent", message, stderr=True)

    def error(self, message: str) -> None:
        self._event("danger", "error", message, stderr=True)

    def render_response(self, text: str, *, streamed: bool = False) -> None:
        """Render the final assistant response."""
        if not text:
            if streamed:
                self.finish_stream()
            return

        if streamed:
            self.finish_stream()
            return

        if not RICH_AVAILABLE:
            print(text)
            return

        self.console.print()
        self.console.print(
            Panel(
                Markdown(text),
                title=" response ",
                title_align="left",
                border_style="response",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.console.print()

    def stream_text(self, text: str) -> None:
        """Render a streamed response chunk."""
        if not text:
            return

        if not RICH_AVAILABLE:
            print(text, end="", flush=True)
            self._stream_open = True
            return

        if not self._stream_open:
            self.console.print()
            self.console.rule("[response]assistant[/response]", style="response")
            self._stream_open = True

        self.console.print(text, end="", markup=False, highlight=False)

    def finish_stream(self) -> None:
        """Close the current streamed response block."""
        if not self._stream_open:
            return

        if RICH_AVAILABLE:
            self.console.print()
            self.console.print()
        else:
            print()
        self._stream_open = False

    def ask_user(self, question: str) -> str:
        """Render an inline clarification prompt."""
        self.finish_stream()
        if not RICH_AVAILABLE:
            print(f"\n[Agent asks] {question}")
            return input("> ")

        self.console.print()
        self.console.print(
            Panel(
                Text(question, style="accent"),
                title=" clarification ",
                title_align="left",
                border_style="warning",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        return self.console.input("[warning]answer[/warning] [muted]>[/muted] ")

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Show a temporary spinner while the model is working."""
        if not RICH_AVAILABLE:
            yield
            return

        with self.console.status(f"[info]{message}[/info]", spinner="dots"):
            yield

    def _event(self, style: str, label: str, message: str, *, stderr: bool = False) -> None:
        self.finish_stream()
        console = self.error_console if stderr else self.console

        if not RICH_AVAILABLE or console is None:
            print(f"[{label.upper()}] {message}", file=sys.stderr if stderr else sys.stdout)
            return

        badge_style = self._badge_styles.get(style, "black on white")
        badge = Text(f" {label.upper()} ", style=badge_style)
        line = Text.assemble(badge, " ", (message, style))
        console.print(line)

    def _shorten_path(self, working_dir: str) -> str:
        path = Path(working_dir).expanduser()
        parts = path.parts
        if len(parts) <= 4:
            return str(path)
        return ".../" + "/".join(parts[-3:])


_terminal_ui: TerminalUI | None = None


def get_terminal_ui() -> TerminalUI:
    """Return the process-wide terminal UI singleton."""
    global _terminal_ui
    if _terminal_ui is None:
        _terminal_ui = TerminalUI()
    return _terminal_ui


def set_terminal_ui(ui: TerminalUI) -> None:
    """Override the process-wide terminal UI singleton."""
    global _terminal_ui
    _terminal_ui = ui
