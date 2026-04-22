"""Ollama client.

Approach:
  Ollama exposes an OpenAI-compatible API, so it can be called with the openai library.
  Streaming is also supported.
"""
from openai import OpenAI
from .config import OllamaConfig
import re

def create_client(cfg: OllamaConfig) -> OpenAI:
    """Create an OpenAI client pointed at the Ollama server.

    Args:
        cfg: Ollama connection config.

    Returns:
        An OpenAI client configured to use the Ollama base URL.
    """
    return OpenAI(base_url=cfg.base_url, api_key="ollama")

def response(
        client: OpenAI,
        cfg: OllamaConfig,
        message: list[dict[str, str]],
) -> str:
    """Send a message to Ollama and receive the full response as a string.

    Args:
        client: OpenAI client pointed at the Ollama server.
        cfg: Ollama config (model name, temperature, etc.).
        message: Messages to send. e.g. [{"role": "user", "content": "Hello!"}]

    Returns:
        The full response string.
    """
    response_text = client.chat.completions.create(
        model = cfg.model,
        messages = message,
        temperature = cfg.temperature,
        max_tokens = cfg.max_tokens,
    )
    text = (response_text.choices[0].message.content or "") if response_text.choices else "" # For the case when API responses content=None
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()  # Remove <think> sections from the final response.

def stream_response(
        client: OpenAI,
        cfg: OllamaConfig,
        message: list[dict[str, str]],
) -> str:
    """Send a message to Ollama and receive the response token by token via streaming.

    Args:
        client: OpenAI client pointed at the Ollama server.
        cfg: Ollama config (model name, temperature, etc.).
        message: Messages to send. e.g. [{"role": "user", "content": "Hello!"}]

    Returns:
        The full response string (thinking section excluded).
    """
    # Use the Chat Completions API with streaming enabled.
    response = client.chat.completions.create(
        model = cfg.model,
        messages = message,
        temperature = cfg.temperature,
        max_tokens = cfg.max_tokens,
        stream = True,
    )

    # Receive the stream and display tokens as they arrive.
    buf = ""
    state = "init"
    answer_tokens: list[str] = []

    for chunk in response:
        token = chunk.choices[0].delta.content if chunk.choices else None
        if not token:
            continue
        buf += token

        # --- <think> tag state machine ---
        # "init": determine whether the first tokens contain <think>
        if state == "init":
            if "<think>" in buf:
                # Strip the <think> tag and switch to thinking mode.
                buf = buf.replace("<think>", "")
                state = "thinking"
                # Print a dim "[thinking]" label at the start of the thinking section.
                # \033[2m = ANSI dim, \033[0m = reset.
                # end="" and flush=True print immediately without a newline.
                # Reference: https://qiita.com/arakaki_tokyo/items/e54d95911ec7a22a7846
                print("\033[2m[thinking]\033[0m ", end="", flush=True)

            elif len(buf) > 7:
                # No <think> tag after 7 characters — switch directly to answering mode.
                state = "answering"
                print(buf, end="", flush=True)
                answer_tokens.append(buf)
                buf = ""

        # "thinking": display tokens in dim color until </think> arrives
        elif state == "thinking":
            if "</think>" in buf:
                # Split on </think>: display everything before it, then switch to answering.
                before, _, after = buf.partition("</think>")
                if before:
                    print(f"\033[2m{before}\033[0m ", end="", flush=True)
                print("\n", end="", flush=True)  # blank line to separate thinking from answer

                buf = ""
                state = "answering"
                # If there is content after </think>, display and record it immediately.
                if after:
                    print(after, end="", flush=True)
                    answer_tokens.append(after)

            else:
                print(f"\033[2m{buf}\033[0m ", end="", flush=True)
                buf = ""

        # "answering": print tokens as-is and record them
        else:
            print(buf, end="", flush=True)
            # Append to answer_tokens so the full response can be returned at the end.
            answer_tokens.append(buf)
            buf = ""

    # Flush any remaining content in the buffer after the stream ends.
    if buf:
        print(buf, end="", flush=True)
        if state != "thinking":
            answer_tokens.append(buf)

    print()  # final newline

    return "".join(answer_tokens)