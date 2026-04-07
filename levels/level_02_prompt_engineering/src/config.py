"""Configuration loader for Level 2."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class OllamaConfig(BaseModel):
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024


class AgentConfig(BaseModel):
    system_prompt: str
    prompt_mode: str = "basic"
    max_history_turns: int = 20


class Config(BaseModel):
    ollama: OllamaConfig
    agent: AgentConfig


def get_config(path: Path | None = None) -> Config:
    """Load config from YAML file.

    Args:
        path: Path to config.yaml. Defaults to config.yaml next to src/.

    Returns:
        Parsed Config object.
    """
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
