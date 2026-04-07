"""Application state for Level 7 — Multi-Agent RAG.

Manages the 4-agent LangGraph pipeline (Chat) and document
ingestion (Ingest). State is server-side per-user via WebSocket.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import reflex as rx

from src.config import get_config
from src.generation.ollama_models import list_models

logger = logging.getLogger(__name__)


class ChatState(rx.State):
    """Reactive state for the multi-agent RAG chat page.

    Handles dual model selection, LangGraph pipeline execution,
    agent thinking display, citations, and validation scores.
    """

    # ── Settings ───────────────────────────────────────────────────────
    available_models: list[str] = []
    planner_model: str = ""
    executor_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 128000
    use_colbert: bool = False

    # ── Chat ───────────────────────────────────────────────────────────
    messages: list[dict[str, str]] = []
    current_answer: str = ""
    agent_steps: list[dict[str, str]] = []
    is_generating: bool = False

    # ── Human-in-the-Loop ──────────────────────────────────────────────
    waiting_for_user: bool = False
    feedback_prompt: str = ""

    # ── Private ────────────────────────────────────────────────────────
    _graph: object = None
    _thread_id: str = ""
    _pending_state: dict = {}

    @rx.event
    def on_load(self) -> None:
        """Load configuration and build the agent graph on page load."""
        cfg = get_config()
        ollama_base = cfg.ollama.base_url.replace("/v1", "")
        models = list_models(ollama_base) or [cfg.ollama.planner_model]

        self.available_models = models
        self.planner_model = (
            cfg.ollama.planner_model
            if cfg.ollama.planner_model in models
            else models[0]
        )
        self.executor_model = (
            cfg.ollama.executor_model
            if cfg.ollama.executor_model in models
            else models[0]
        )
        self.temperature = cfg.ollama.temperature

        from src.agents.graph import build_graph

        self._graph = build_graph(debug=False)
        self._thread_id = str(uuid.uuid4())

    @rx.event
    def set_temperature_value(self, value: list[float]) -> None:
        """Update temperature from slider value."""
        self.temperature = round(value[0], 1)

    @rx.event
    def set_max_tokens_value(self, value: list[float]) -> None:
        """Update max tokens from slider value."""
        self.max_tokens = int(value[0])

    @rx.event
    async def handle_submit(self, form_data: dict) -> None:
        """Run the 4-agent RAG pipeline for the user's query.

        Args:
            form_data: Form submission data containing the message text.
        """
        message = form_data.get("message", "").strip()
        if not message or self.is_generating:
            return

        self.messages.append({
            "role": "user",
            "content": message,
            "thinking": "",
            "metrics": "",
        })
        self.is_generating = True
        self.current_answer = ""
        self.agent_steps = []
        yield

        config = {"configurable": {"thread_id": self._thread_id}}

        try:
            if self.waiting_for_user:
                from src.agents.facilitator import enrich_with_user_response

                new_state = enrich_with_user_response(
                    self._pending_state, message,
                )
                self._graph.update_state(config, new_state)
                self.waiting_for_user = False
                self.feedback_prompt = ""
                self._pending_state = {}
                stream_input = None
            else:
                from src.agents.graph import make_initial_state

                stream_input = make_initial_state(message)

            final_answer = ""
            final_citations: list = []
            final_scores: dict = {}
            shown_thinking: set[int] = set()

            for event in self._graph.stream(
                stream_input, config, stream_mode="values",
            ):
                state = event if isinstance(event, dict) else {}

                # Collect agent thinking steps
                thinking_list = state.get("agent_thinking", [])
                for i, entry in enumerate(thinking_list):
                    if i not in shown_thinking and entry.get("thinking"):
                        shown_thinking.add(i)
                        self.agent_steps.append({
                            "agent": entry["agent"],
                            "thinking": entry["thinking"],
                        })
                        yield

                # Human-in-the-Loop check
                if state.get("needs_user_input"):
                    self.waiting_for_user = True
                    self.feedback_prompt = state.get(
                        "feedback_to_user", "",
                    )
                    self._pending_state = state
                    self.current_answer = (
                        f"**Facilitator:**\n\n{self.feedback_prompt}"
                    )
                    self.is_generating = False
                    yield
                    return

                if state.get("answer"):
                    final_answer = state["answer"]
                    self.current_answer = final_answer
                    yield

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

            # Build final content with citations and scores
            parts = [final_answer] if final_answer else [
                "Could not generate an answer.",
            ]

            if final_citations:
                citation_lines = [
                    f"\n\n**Citations ({len(final_citations)}):**",
                ]
                for cit in final_citations:
                    page = (
                        f" (p.{cit.page_number})" if cit.page_number else ""
                    )
                    preview = cit.original_text[:200] + (
                        "..." if len(cit.original_text) > 200 else ""
                    )
                    citation_lines.append(
                        f"- **[{cit.citation_id}] "
                        f"{cit.source_file}{page}**: {preview}",
                    )
                parts.append("\n".join(citation_lines))

            # Build metrics string
            metrics_parts = []
            if final_scores:
                avg = final_scores.get("average", 0)
                metrics_parts.append(
                    f"Validation: {avg:.1f}/100 "
                    f"(C:{final_scores.get('completeness', 0)} "
                    f"A:{final_scores.get('accuracy', 0)} "
                    f"R:{final_scores.get('relevance', 0)} "
                    f"F:{final_scores.get('faithfulness', 0)})",
                )
            metrics_parts.append(
                f"planner: {self.planner_model} "
                f"| executor: {self.executor_model}",
            )

            self.messages.append({
                "role": "assistant",
                "content": "\n".join(parts),
                "thinking": "",
                "metrics": " | ".join(metrics_parts),
            })

        except Exception as exc:
            logger.error("Graph execution error: %s", exc, exc_info=True)
            self.messages.append({
                "role": "assistant",
                "content": f"Error: {exc}",
                "thinking": "",
                "metrics": "",
            })

        self.is_generating = False
        self.current_answer = ""
        self.agent_steps = []
        yield


class IngestState(rx.State):
    """Reactive state for the document ingestion page.

    Handles file upload, ingestion pipeline execution,
    and displays results with Weaviate collection stats.
    """

    results: list[str] = []
    is_processing: bool = False
    collection_count: int = 0

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]) -> None:
        """Process uploaded files through the ingestion pipeline.

        Args:
            files: List of uploaded files from rx.upload.
        """
        if not files:
            return

        self.is_processing = True
        self.results = []
        yield

        from src.ingestion.pipeline import IngestionPipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            rendered_dir = Path(tmp_dir) / "rendered"
            pipeline = IngestionPipeline(
                use_vision=False,
                rendered_dir=rendered_dir,
            )

            try:
                for file in files:
                    upload_data = await file.read()
                    dest_path = Path(tmp_dir) / file.filename
                    dest_path.write_bytes(upload_data)

                    stats = pipeline.ingest_file(dest_path)

                    if stats.failed_files > 0:
                        error_msg = (
                            stats.errors[0][1] if stats.errors else "Unknown"
                        )
                        self.results.append(
                            f"{file.filename}: Error — {error_msg}",
                        )
                    else:
                        self.results.append(
                            f"{file.filename}: "
                            f"{stats.total_chunks} chunks ingested",
                        )
                    yield
            finally:
                pipeline.close()

        # Get collection stats
        try:
            from src.retrieval.weaviate_client import (
                COLLECTION_NAME,
                get_client,
            )

            client = get_client()
            collection = client.collections.get(COLLECTION_NAME)
            count = collection.aggregate.over_all(
                total_count=True,
            ).total_count
            client.close()
            self.collection_count = count
        except Exception as exc:
            logger.warning("Weaviate stats unavailable: %s", exc)

        self.is_processing = False
        yield
