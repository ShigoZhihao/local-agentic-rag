"""CLI entry point for Level 7 — Multi-Agent RAG.

Subcommands:
    chat    Interactive REPL backed by the 4-agent LangGraph pipeline.
    ingest  Ingest one or more document files into Weaviate.

Usage:
    python main.py chat
    python main.py chat --planner gemma4:e4b --executor gemma4:e2b
    python main.py ingest path/to/doc.pdf path/to/notes.md
    python main.py ingest --vision path/to/slides.pptx
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hr(char: str = "─", width: int = 72) -> str:
    """Return a horizontal rule string."""
    return char * width


def _print_citations(citations: list) -> None:
    """Print citation list to stdout."""
    if not citations:
        return
    print(f"\n  Citations ({len(citations)}):")
    for cit in citations:
        page = f" (p.{cit.page_number})" if cit.page_number else ""
        preview = cit.original_text[:200] + ("..." if len(cit.original_text) > 200 else "")
        print(f"    [{cit.citation_id}] {cit.source_file}{page}: {preview}")


def _print_scores(scores: dict) -> None:
    """Print validation scores to stdout."""
    if not scores:
        return
    avg = scores.get("average", 0)
    label = "PASS" if avg >= 80 else "WARN" if avg >= 60 else "FAIL"
    print(
        f"\n  Validation [{label}] {avg:.1f}/100 — "
        f"Completeness:{scores.get('completeness', 0)} "
        f"Accuracy:{scores.get('accuracy', 0)} "
        f"Relevance:{scores.get('relevance', 0)} "
        f"Faithfulness:{scores.get('faithfulness', 0)}",
    )


# ── Chat subcommand ───────────────────────────────────────────────────────────

def cmd_chat(args: argparse.Namespace) -> None:
    """Run the interactive 4-agent RAG chat loop."""
    from src.agents.graph import build_graph, make_initial_state
    from src.config import get_config
    from src.generation.ollama_models import list_models

    cfg = get_config()
    ollama_base = cfg.ollama.base_url.replace("/v1", "")
    available = list_models(ollama_base) or [cfg.ollama.planner_model]

    planner = args.planner if args.planner else cfg.ollama.planner_model
    executor = args.executor if args.executor else cfg.ollama.executor_model
    if planner not in available:
        planner = available[0]
    if executor not in available:
        executor = available[0]

    print(_hr("═"))
    print("  Level 7 — Multi-Agent RAG  |  CLI Chat")
    print(_hr("═"))
    print(f"  Planner  : {planner}")
    print(f"  Executor : {executor}")
    print(_hr())
    print("  Type your question and press Enter.")
    print("  Commands: 'reset' — new conversation, 'exit' — quit.")
    print(_hr("═"))
    print()

    graph = build_graph(debug=False)
    thread_id = str(uuid.uuid4())
    waiting_for_user = False
    pending_state: dict = {}

    while True:
        try:
            prompt = "> " if not waiting_for_user else "Facilitator reply> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if user_input.lower() == "reset":
            thread_id = str(uuid.uuid4())
            waiting_for_user = False
            pending_state = {}
            print("  [Conversation reset]\n")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        print()

        try:
            if waiting_for_user:
                from src.agents.facilitator import enrich_with_user_response

                new_state = enrich_with_user_response(pending_state, user_input)
                graph.update_state(config, new_state)
                waiting_for_user = False
                pending_state = {}
                stream_input = None
            else:
                stream_input = make_initial_state(user_input)

            final_answer = ""
            final_citations: list = []
            final_scores: dict = {}
            shown_agents: set[int] = set()
            last_agent_label = ""

            for event in graph.stream(stream_input, config, stream_mode="values"):
                state = event if isinstance(event, dict) else {}

                # Show agent progress
                thinking_list = state.get("agent_thinking", [])
                for i, entry in enumerate(thinking_list):
                    if i not in shown_agents and entry.get("thinking"):
                        shown_agents.add(i)
                        agent = entry["agent"]
                        label = f"[{agent.upper()}]"
                        if label != last_agent_label:
                            print(f"\n{label}", flush=True)
                            last_agent_label = label
                        snippet = entry["thinking"][:300].replace("\n", " ")
                        print(f"  {snippet}{'...' if len(entry['thinking']) > 300 else ''}")

                # Human-in-the-Loop pause
                if state.get("needs_user_input"):
                    waiting_for_user = True
                    pending_state = state
                    feedback = state.get("feedback_to_user", "")
                    print(f"\nFacilitator: {feedback}\n")
                    break

                if state.get("answer"):
                    final_answer = state["answer"]

                if state.get("citations"):
                    final_citations = state["citations"]

                if state.get("validation") and state["validation"]:
                    v = state["validation"]
                    final_scores = {
                        "completeness": v.scores.completeness,
                        "accuracy": v.scores.accuracy,
                        "relevance": v.scores.relevance,
                        "faithfulness": v.scores.faithfulness,
                        "average": v.scores.average,
                    }

            if not waiting_for_user:
                print(f"\nAssistant> {final_answer or 'Could not generate an answer.'}")
                _print_citations(final_citations)
                _print_scores(final_scores)
                print(f"\n  [planner: {planner} | executor: {executor}]")

        except Exception as exc:
            logging.getLogger(__name__).error("Graph error: %s", exc, exc_info=True)
            print(f"\nError: {exc}", file=sys.stderr)

        print()


# ── Ingest subcommand ─────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest document files into Weaviate."""
    from src.ingestion.pipeline import IngestionPipeline

    paths = [Path(p) for p in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"File not found: {p}", file=sys.stderr)
        sys.exit(1)

    print(_hr("═"))
    print("  Level 7 — Multi-Agent RAG  |  Document Ingestion")
    print(_hr("═"))
    print(f"  Files      : {len(paths)}")
    print(f"  Vision LLM : {'yes' if args.vision else 'no'}")
    print(_hr("═"))
    print()

    pipeline = IngestionPipeline(use_vision=args.vision, rendered_dir=None)
    try:
        for path in paths:
            print(f"  Processing: {path.name} ... ", end="", flush=True)
            stats = pipeline.ingest_file(path)
            if stats.failed_files > 0:
                err = stats.errors[0][1] if stats.errors else "Unknown error"
                print(f"FAILED — {err}")
            else:
                print(f"OK ({stats.total_chunks} chunks)")
    finally:
        pipeline.close()

    # Show Weaviate collection count
    try:
        from src.retrieval.weaviate_client import COLLECTION_NAME, get_client

        client = get_client()
        collection = client.collections.get(COLLECTION_NAME)
        count = collection.aggregate.over_all(total_count=True).total_count
        client.close()
        print(f"\n  Weaviate total chunks: {count}")
    except Exception as exc:
        print(f"\n  Weaviate stats unavailable: {exc}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(description="Level 7 — Multi-Agent RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # chat
    chat_p = sub.add_parser("chat", help="Interactive RAG chat (4-agent pipeline)")
    chat_p.add_argument("--planner", type=str, default="", help="Planner model (Facilitator/Validator)")
    chat_p.add_argument("--executor", type=str, default="", help="Executor model (Synthesizer)")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest documents into Weaviate")
    ingest_p.add_argument("files", nargs="+", metavar="FILE", help="Document files to ingest")
    ingest_p.add_argument(
        "--vision", action="store_true", help="Use vision LLM for PDF/PPTX image pages",
    )

    return parser


def main() -> None:
    """Dispatch to the selected subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "ingest":
        cmd_ingest(args)


if __name__ == "__main__":
    main()
