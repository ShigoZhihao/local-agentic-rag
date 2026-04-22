"""
Approach:
  1. Load config.yaml
  2. Create the Ollama client
  3. Run the conversation loop with rewriter option:
       - User input → Rewrite query option → Send to LLM → Display response → User input → ...
"""

from src.config import get_config
from src.llm_client import create_client, stream_response, response
from src.prompts import MAIN_SYSTEM, REWRITER_SYSTEM, CLARIFIER_SYSTEM

# We keep the last N turns of conversation history to provide context for the LLM.
max_history_turns = 10

def main() -> None:
    """まずこれは返り値がない関数。とりあえずconfig.pyのget_config関数からconfig.yamlの情報を読み取ってcfgに格納する。
    そしてllm_client.pyのcreate_client関数からcfgのurl情報を読み取り、Ollamaサーバーへの接続クライアントを作る。
    その後まずmessagesにMAIN_SYSTEMプロンプトを追加する。
    その後while文に入り、とりあえずYou:を表示させ、ユーザーの入力を待つ。入力された文字から余白を消して、変数をuser_inputとして保存。
    もしKeyboardInterruptつまりCtrl +Cがあったら改行してBye！を表示して終了。
    もしただ何も入力しなかったか、quitと入力したらBye！を表示して終了。
    それ以外入力されたものはmessagesリストに追加される。
    そしてモデル名が表示され、messagesがllm_clientのstream_response関数で処理され、推論が始まり、回答が表示される、
    その解答がmessagesに追加される。最後にもしmessagesの長さが21個を超えたらシステムプロンプトと最新の20個だけを残す。"""
    """Entry point for the conversation loop."""
    cfg = get_config()
    client = create_client(cfg.ollama)

    print("=== Level 2: Prompt Engineering ChatBot ===")
    print(f"Model: {cfg.ollama.model}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": MAIN_SYSTEM}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nSee you next time!")
            break

        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("See you next time!")
            break   


        # Send user input to Clarifier LLM to get questions for improving the query
        clarifier_messages = [
             {"role": "system", "content": CLARIFIER_SYSTEM},
             {"role": "user", "content": user_input}
        ]
        print(f"{cfg.ollama.model}: ", end="", flush=True)
        questions = stream_response(client, cfg.ollama, clarifier_messages)

        # Get user answers
        answers = input(f"\nYou: ").strip()
        # Send user input and clarifier answer to Rewriter LLM to get the improved query
        rewriter_messages = [
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": questions},
            {"role": "user", "content": answers}
        ]
        improved_query = response(client, cfg.ollama, rewriter_messages)
        # Send to LLM and display the streamed response. 
        print(f"{cfg.ollama.model}: ", end="", flush=True)
        answer = stream_response(client, cfg.ollama, messages + [{"role": "user", "content": improved_query}])
        messages.append({"role": "assistant", "content": answer})
        
        # To prevent the message history from growing indefinitely, 
        # we trim it to keep only the most recent turns.
        # Because max_history_turns=10, we keep the system prompt + last 10 user-assistant pairs = 21 messages total.
        if len(messages) > 1 + 2*max_history_turns:
            messages = [messages[0]] + messages[-(2*max_history_turns):]

if __name__ == "__main__":
    main()