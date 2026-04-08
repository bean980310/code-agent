"""Configuration loading for the code agent."""

from __future__ import annotations

import os

from .types import AgentConfig


def load_config(overrides: dict | None = None) -> AgentConfig:
    """Load config from environment variables and optional overrides."""
    config = AgentConfig()

    if model := os.getenv("CODE_AGENT_MODEL"):
        config.model = model
    if max_turns := os.getenv("CODE_AGENT_MAX_TURNS"):
        config.max_turns = int(max_turns)
    if working_dir := os.getenv("CODE_AGENT_WORKING_DIR"):
        config.working_dir = working_dir

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

    return config
