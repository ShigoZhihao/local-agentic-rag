"""CLI entry point for Level 2 — Prompt Engineering.

Interactive chat REPL with selectable prompt modes (basic, cot, few_shot, structured).
Streams responses from a local Ollama model with thinking block display.

Usage:
    python main.py
    python main.py --mode cot --model gemma4:e2b
    python main.py --mode structured --temperature 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    from src.prompts import PROMPT_MODES

    parser = argparse.ArgumentParser(description="Level 2 — Prompt Engineering CLI chat")
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(PROMPT_MODES.keys()),
        default=None,
        help="Prompt mode (default: from config.yaml)",
    )
    parser.add_argument("--model", type=str, default="", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens")
    return parser.parse_args()


def _hr(char: str = "─", width: int = 72) -> str:
    """Return a horizontal rule string."""
    return char * width


def main() -> None:
    """Run the interactive CLI chat loop with prompt mode selection."""
    args = _parse_args()

    from src.config import OllamaConfig, get_config
    from src.llm_client import StreamingChat, create_client
    from src.metrics import format_metrics, measure_delta, take_snapshot
    from src.ollama_models import list_models
    from src.prompts import PROMPT_MODES

    cfg = get_config()
    ollama_base = cfg.ollama.base_url.replace("/v1", "")
    available = list_models(ollama_base) or [cfg.ollama.model]

    model = args.model if args.model else cfg.ollama.model
    if model not in available:
        model = available[0]

    temperature = args.temperature if args.temperature is not None else cfg.ollama.temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.ollama.max_tokens

    prompt_mode = args.mode if args.mode else getattr(cfg.agent, "prompt_mode", "basic")
    if prompt_mode not in PROMPT_MODES:
        prompt_mode = "basic"

    system_prompt = PROMPT_MODES[prompt_mode]

    print(_hr("═"))
    print("  Level 2 — Prompt Engineering  |  Ollama CLI Chat")
    print(_hr("═"))
    print(f"  Model       : {model}")
    print(f"  Temperature : {temperature}")
    print(f"  Max tokens  : {max_tokens}")
    print(f"  Prompt mode : {prompt_mode}  (choices: {', '.join(PROMPT_MODES)})")
    print(_hr())
    print("  Type your message and press Enter. Ctrl+C or 'exit' to quit.")
    print("  During a session: type '/mode <name>' to switch prompt mode.")
    print(_hr("═"))
    print()

    runtime_cfg = OllamaConfig(
        base_url=cfg.ollama.base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client = create_client(runtime_cfg)
    history: list[dict[str, str]] = []
    max_turns: int = getattr(cfg.agent, "max_history_turns", 10)

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # In-session mode switch command
        if user_input.startswith("/mode "):
            new_mode = user_input.split(maxsplit=1)[1].strip()
            if new_mode in PROMPT_MODES:
                prompt_mode = new_mode
                system_prompt = PROMPT_MODES[prompt_mode]
                print(f"  [Switched to prompt mode: {prompt_mode}]\n")
            else:
                print(f"  [Unknown mode '{new_mode}'. Available: {', '.join(PROMPT_MODES)}]\n")
            continue

        trimmed = history[-(max_turns * 2):]
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(trimmed)
        full_messages.append({"role": "user", "content": user_input})

        print()
        thinking_buf = ""
        answer_buf = ""
        t_start = time.perf_counter()
        baseline = take_snapshot()

        try:
            stream = StreamingChat(client, runtime_cfg, full_messages)
            in_thinking = False

            for state, token in stream:
                if state == "thinking":
                    if not in_thinking:
                        print("[Thinking]", flush=True)
                        in_thinking = True
                    print(token, end="", flush=True)
                    thinking_buf += token
                else:
                    if in_thinking:
                        print("\n")
                        in_thinking = False
                        print("Assistant> ", end="", flush=True)
                    elif not answer_buf:
                        print("Assistant> ", end="", flush=True)
                    print(token, end="", flush=True)
                    answer_buf += token

            elapsed = time.perf_counter() - t_start
            metrics = measure_delta(
                baseline=baseline,
                prompt_tokens=stream.prompt_tokens,
                completion_tokens=stream.completion_tokens,
                elapsed_sec=elapsed,
            )
            print(
                f"\n\n  [{format_metrics(metrics)} | model: {model} | mode: {prompt_mode}]",
            )

        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            answer_buf = f"Error: {exc}"

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer_buf})

        print()


if __name__ == "__main__":
    main()
