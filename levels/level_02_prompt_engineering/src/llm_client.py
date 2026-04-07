"""Thin OpenAI-compatible streaming client for Ollama.

Core loop at this level:
    user input → LLM → output (streamed token by token)

Thinking support:
    gemma4 models emit <think>...</think> before the answer.
    StreamingChat yields (state, token) tuples so the UI can
    route thinking tokens to a separate panel.
"""

import logging
from typing import Iterator

from openai import OpenAI

from src.config import OllamaConfig

logger = logging.getLogger(__name__)


def create_client(cfg: OllamaConfig) -> OpenAI:
    """Create an OpenAI client pointed at a local Ollama instance."""
    return OpenAI(base_url=cfg.base_url, api_key="ollama")


class StreamingChat:
    """Streams (state, token) tuples from Ollama.

    *state* is ``"thinking"`` while inside ``<think>...</think>``
    and ``"answering"`` otherwise.

    Usage::

        stream = StreamingChat(client, cfg, messages)
        for state, token in stream:
            if state == "thinking":
                thinking_panel.write(token)
            else:
                answer_panel.write(token)
        tokens_in  = stream.prompt_tokens
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

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Yield ``(state, token)`` tuples.

        Parses ``<think>``/``</think>`` tags to separate thinking from answer.
        """
        buf = ""          # small lookahead buffer for tag detection
        state = "init"    # init → thinking → answering
        try:
            for chunk in self._response:
                content = (
                    chunk.choices[0].delta.content
                    if chunk.choices
                    else None
                )
                if chunk.usage:
                    self.prompt_tokens = chunk.usage.prompt_tokens
                    self.completion_tokens = chunk.usage.completion_tokens
                if not content:
                    continue

                buf += content

                while buf:
                    if state == "init":
                        # Look for <think> at the very start
                        if buf.startswith("<think>"):
                            buf = buf[7:]
                            state = "thinking"
                        elif len(buf) < 7 and "<think>".startswith(buf):
                            # Might be partial tag — wait for more tokens
                            break
                        else:
                            # No <think> tag: model answered directly
                            state = "answering"

                    elif state == "thinking":
                        end_idx = buf.find("</think>")
                        if end_idx != -1:
                            # Flush everything before </think>
                            if end_idx > 0:
                                yield ("thinking", buf[:end_idx])
                            buf = buf[end_idx + 8:]
                            state = "answering"
                        elif "</think>".startswith(buf[-7:]) and len(buf) <= 8:
                            # Partial closing tag at end — wait
                            break
                        else:
                            # Check if partial </think> at tail
                            safe = self._flush_safe(buf, "</think>")
                            if safe:
                                yield ("thinking", safe)
                                buf = buf[len(safe):]
                            else:
                                break

                    else:  # answering
                        yield ("answering", buf)
                        buf = ""

            # Flush remaining buffer
            if buf:
                yield (state if state != "init" else "answering", buf)

        except Exception as e:
            logger.error("Streaming interrupted: %s", e)
            raise

    @staticmethod
    def _flush_safe(buf: str, tag: str) -> str:
        """Return the prefix of *buf* that cannot be part of a partial *tag*.

        If the tail of *buf* is a prefix of *tag*, we hold it back so we
        can detect the full tag on the next iteration.
        """
        for i in range(min(len(tag), len(buf)), 0, -1):
            if buf.endswith(tag[:i]):
                return buf[: -i]
        return buf
