"""
Why YAML? -> https://zenn.dev/acntechjp/articles/a39e710de5d744

Loads config.yaml and converts it into a Python object.

Approach:
https://python.land/data-processing/python-yaml
YAML -> dict (yaml.safe_load) -> Pydantic model (with type validation)
    yaml.safe_load converts a YAML-formatted string into a Python dict,
    allowing the YAML contents to be used as Python data structures.
    A Pydantic model is a Python class that provides type checking and
    validation, ensuring the YAML contents match the expected format.
    Why Pydantic:
        Typos like cfg["ollama"]["base_ur"] are not caught until runtime.
        cfg.ollama.base_url lets the IDE autocomplete and catch typos early.
"""
# pathlib is a standard library for working with file and directory paths
# as objects rather than plain strings. Path is its central class.
from pathlib import Path
# PyYAML is a library for reading and writing YAML-formatted files.
import yaml
# BaseModel is the base class from Pydantic used to define typed data models.
# It validates incoming data and converts it into a convenient Python object.
# https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/
from pydantic import BaseModel, Field

class OllamaConfig(BaseModel):
    base_url: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)  # must be between 0.0 and 2.0
    max_tokens: int = Field(gt=0)  # must be a positive integer

class Config(BaseModel):
    # ollama holds the connection settings defined in OllamaConfig (URL, model name, etc.)
    ollama: OllamaConfig

def get_config() -> Config:
    # Build the path to config.yaml relative to this file.
    # Path(__file__) is the path to this file (config.py).
    # .parent.parent navigates two levels up to the project root where config.yaml lives.
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    # Open with UTF-8 encoding to support non-ASCII characters (e.g. Japanese system prompts).
    with open(config_path, "r", encoding="utf-8") as f:
        # yaml.safe_load converts the YAML content into a Python dict.
        config_dict = yaml.safe_load(f)
    # Expand the dict as keyword arguments to construct the Config instance.
    return Config(**config_dict)