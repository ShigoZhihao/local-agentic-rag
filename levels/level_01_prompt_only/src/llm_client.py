"""Thin OpenAI-compatible client for Ollama.

Core loop at this level:
    user input → LLM → output

No tools, no retrieval, no memory beyond what the caller passes in.
"""

import logging

from openai import OpenAI

from config import OllamaConfig

logger = logging.getLogger(__name__)


def create_client(cfg: OllamaConfig) -> OpenAI:
    """Create an OpenAI client pointed at a local Ollama instance.

    Args:
        cfg: Ollama connection settings.

    Returns:
        Configured OpenAI client.
    """
    return OpenAI(base_url=cfg.base_url, api_key="ollama")


def chat(
    client: OpenAI,
    cfg: OllamaConfig,
    messages: list[dict[str, str]],
) -> str:
    """Send a list of messages to the model and return the reply text.

    Args:
        client: OpenAI client instance.
        cfg: Ollama model settings.
        messages: Full conversation history including system prompt,
                  e.g. [{"role": "system", "content": "..."}, ...].

    Returns:
        The assistant's reply as a plain string.
    """
    try:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise
