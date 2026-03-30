"""Thin OpenAI-compatible streaming client for Ollama.

Core loop at this level:
    user input → LLM → output (streamed token by token)
"""

import logging

from openai import OpenAI

from config import OllamaConfig

logger = logging.getLogger(__name__)


def create_client(cfg: OllamaConfig) -> OpenAI:
    """Create an OpenAI client pointed at a local Ollama instance."""
    return OpenAI(base_url=cfg.base_url, api_key="ollama")


class StreamingChat:
    """Streams text chunks from Ollama one token at a time.

    Usage:
        stream = StreamingChat(client, cfg, messages)
        reply = st.write_stream(stream)          # Streamlit streams to UI
        tokens_in  = stream.prompt_tokens        # available after iteration
        tokens_out = stream.completion_tokens
    """

    def __init__(
        self,
        client: OpenAI,
        cfg: OllamaConfig,
        messages: list[dict[str, str]],
    ) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        try:
            self._response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            logger.error("Failed to start streaming: %s", e)
            raise

    def __iter__(self):
        """Yield text delta strings. Populate token counts from the final chunk."""
        try:
            for chunk in self._response:
                content = (
                    chunk.choices[0].delta.content
                    if chunk.choices
                    else None
                )
                if content:
                    yield content
                if chunk.usage:
                    self.prompt_tokens = chunk.usage.prompt_tokens
                    self.completion_tokens = chunk.usage.completion_tokens
        except Exception as e:
            logger.error("Streaming interrupted: %s", e)
            raise
