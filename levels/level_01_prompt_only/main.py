"""
Approach:
  1. Load config.yaml
  2. Create the Ollama client
  3. Run the conversation loop:
       user input → send to LLM → display response → user input → ...
"""

from src.config import get_config
from src.llm_client import create_client, stream_response

# Call main() only when this file is executed directly, not when imported.
# Note: Python's multiprocessing re-imports the entry-point file in each worker.
# Without this guard, every worker would spawn another main() call,
# creating an infinite loop that freezes the machine.
# This guard is required on any file that serves as the program entry point.

def main() -> None:
    """Entry point for the conversation loop."""
    cfg = get_config()
    client = create_client(cfg.ollama)

    print("=== Level 1: Prompt Only ChatBot ===")
    print(f"Model: {cfg.ollama.model}")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            break

        if not user_input or user_input.lower() == "quit":
            print("Bye!")
            break   


        messages = [
            {"role": "user", "content": user_input}
        ]

        # Send to LLM and display the streamed response. No memory — each turn is stateless.
        print(f"{cfg.ollama.model}: ", end="", flush=True)
        stream_response(client, cfg.ollama, messages)


if __name__ == "__main__":
    main()