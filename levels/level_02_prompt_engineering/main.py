"""
Approach:
  1. Load config.yaml
  2. Create the Ollama client
  3. Run the conversation loop with rewriter option:
       - User input → Rewrite query option → Send to LLM → Display response → User input → ...
"""

from pyexpat.errors import messages

from src.config import get_config
from src.llm_client import create_client, stream_response
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
            original_user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nSee you next time!")
            break

        if not original_user_input:
            continue
        
        if original_user_input.lower() == "quit":
            print("See you next time!")
            break   


       

        # Ask user if they want to improve their query, if yes, send the original query to Clarifier LLM
        # and get three questions for further context from the user
        print(f"{cfg.ollama.model}: Do you want to rewrite your query to be more clear, detailed and structured? (yes/no)")
        rewrite_input = input("You: ").strip().lower()

        user_input = original_user_input  # デフォルトは元の入力

        if rewrite_input in ["yes", "y"]:
            while True:
                clarifier_messages = [
                    {"role": "system", "content": CLARIFIER_SYSTEM},
                    {"role": "user", "content": user_input}
                ]
                print(f"{cfg.ollama.model}: I need more context to improve your query. Please answer the following questions. Just wait seconds...")
                # Questions from Clarifier LLM, to get more context for rewriting the query.
                questions = stream_response(client, cfg.ollama, clarifier_messages)

                # Get additional context from user by answering three questions
                answers = input("You: ").strip()
                rewriter_messages = [
                    {"role": "system", "content": REWRITER_SYSTEM},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": questions},
                    {"role": "user", "content": answers}
                ]

                # Show the improved query
                print(f"{cfg.ollama.model}: Improved query:\n")
                improved_query = stream_response(client, cfg.ollama, rewriter_messages)

                # Ask user if they want to use the improved query or they wan to edit it by themselves, they want to improve again
                print(f"{cfg.ollama.model}: Do you want to use the improved query? edit it by yourself? or try improving again? (use/edit/improve)")
                choice = input("You: ").strip().lower()

                if choice == "use":
                    print(f"{cfg.ollama.model}: Ok then, let me use the improved query.")
                    # Send to LLM and display the streamed response. 
                    print(f"{cfg.ollama.model}: ", end="", flush=True)
                    answer = stream_response(client, cfg.ollama, messages + [{"role": "user", "content": improved_query}])
                    messages.append({"role": "user", "content": original_user_input})
                    messages.append({"role": "assistant", "content": answer})
                    break
                elif choice == "edit":
                    print(f"{cfg.ollama.model}: Ok then, please edit the improved query and send it to me.")
                    print(f"{cfg.ollama.model}: {improved_query}")
                    edited = input("You: ").strip()
                    if edited:
                        improved_query = edited  # 入力があれば上書き
                                           
                    print(f"{cfg.ollama.model}: ", end="", flush=True)
                    answer = stream_response(client, cfg.ollama, messages + [{"role": "user", "content": improved_query}])
                    messages.append({"role": "user", "content": original_user_input})
                    messages.append({"role": "assistant", "content": answer})
                    break
                elif choice == "improve":
                    print(f"{cfg.ollama.model}: Ok then, let me try improving the query again.")
                    user_input = improved_query  # 上書きしてループを回す

            
        else:
            print(f"{cfg.ollama.model}: Ok then, let me use the original query.")
            messages.append({"role": "user", "content": user_input})
            # Send to LLM and display the streamed response. 
            print(f"{cfg.ollama.model}: ", end="", flush=True)
            answer = stream_response(client, cfg.ollama, messages)
            messages.append({"role": "assistant", "content": answer})
            
        # To prevent the message history from growing indefinitely, 
        # we trim it to keep only the most recent turns.
        # Because max_history_turns=10, we keep the system prompt + last 10 user-assistant pairs = 21 messages total.
        if len(messages) > 1 + 2*max_history_turns:
            messages = [messages[0]] + messages[-(2*max_history_turns):]

        # デバッグ用: メッセージ数を表示
        print(f"[DEBUG] messages count: {len(messages)}")

if __name__ == "__main__":
    main()