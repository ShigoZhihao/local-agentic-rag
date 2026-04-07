"""Ollama model discovery helpers.

Queries the Ollama REST API to list downloaded models
and retrieve context window sizes for dynamic UI controls.
"""

import logging

import requests

logger = logging.getLogger(__name__)


def list_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Fetch downloaded model names from Ollama API.

    Args:
        base_url: Ollama server URL (without /v1 suffix).

    Returns:
        Sorted list of model name strings.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return sorted(m["name"] for m in models)
    except Exception as e:
        logger.warning("Failed to list Ollama models: %s", e)
        return []


def get_context_window(
    model_name: str,
    base_url: str = "http://127.0.0.1:11434",
) -> int:
    """Get a model's context window size via Ollama API.

    Args:
        model_name: Name of the Ollama model.
        base_url: Ollama server URL (without /v1 suffix).

    Returns:
        Context window size in tokens, or 131072 as fallback.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        # Try model_info first (structured data)
        model_info = data.get("model_info", {})
        for key, value in model_info.items():
            if "context_length" in key:
                return int(value)

        # Fallback: parse from parameters string
        params = data.get("parameters", "")
        for line in params.split("\n"):
            if "num_ctx" in line:
                return int(line.split()[-1])

        return 131072
    except Exception as e:
        logger.warning(
            "Failed to get context window for %s: %s", model_name, e,
        )
        return 131072
