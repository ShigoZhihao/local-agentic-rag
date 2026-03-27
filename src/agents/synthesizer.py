"""
Synthesizer agent node.

Responsibilities:
1. Assess whether it can answer the enriched prompt from its own knowledge.
2. If yes: generate a direct answer.
3. If no: identify information gaps and request Researcher retrieval.
4. When citations are provided: generate an answer with inline [1][2] references.

Hallucination is prevented by:
- Explicit prompt instruction: never generate content not in citations.
- Citation-only answer mode when search results are available.
"""

from __future__ import annotations

import json
import logging

from src.agents.state import RAGState
from src.config import get_config
from src.generation.llm_client import call_executor
from src.generation.prompts import (
    SYNTHESIZER_ASSESS,
    SYNTHESIZER_GENERATE_DIRECT,
    SYNTHESIZER_GENERATE_WITH_CONTEXT,
    SYNTHESIZER_SYSTEM,
)
from src.models import Citation, ConversationTurn

logger = logging.getLogger(__name__)


def _format_citations(citations: list[Citation]) -> str:
    """Format citations as a numbered reference list for the prompt."""
    if not citations:
        return "（引用情報なし）"
    lines: list[str] = []
    for cit in citations:
        page_info = f" (p.{cit.page_number})" if cit.page_number else ""
        lines.append(
            f"[{cit.citation_id}] {cit.source_file}{page_info}\n"
            f"{cit.original_text}"
        )
    return "\n\n".join(lines)


def _parse_llm_json(raw: str, context: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s JSON: %s\nRaw: %s", context, exc, raw[:200])
        return {}


def run(state: RAGState) -> RAGState:
    """Execute the Synthesizer node.

    If citations are already in state (from Researcher), generates a
    citation-grounded answer. Otherwise assesses whether direct answer
    is possible and either generates it or requests retrieval.

    Args:
        state: Current RAGState.

    Returns:
        Updated state with answer set, or needs_research=True with
        information_needs populated.
    """
    enriched_prompt = state.get("enriched_prompt", state.get("user_query", ""))
    citations = state.get("citations", [])
    chat_history = state.get("chat_history", [])
    loop_count = state.get("loop_count", 0)

    if citations:
        # Phase 2: answer using provided citations
        return _generate_with_citations(state, enriched_prompt, citations, chat_history, loop_count)
    else:
        # Phase 1: assess whether direct answer is possible
        return _assess_and_route(state, enriched_prompt, chat_history, loop_count)


def _assess_and_route(
    state: RAGState,
    enriched_prompt: str,
    chat_history: list[ConversationTurn],
    loop_count: int,
) -> RAGState:
    """Assess self-knowledge and decide: direct answer or retrieve."""
    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {"role": "user", "content": SYNTHESIZER_ASSESS.format(enriched_prompt=enriched_prompt)},
    ]

    raw = call_executor(messages, json_mode=True)
    result = _parse_llm_json(raw, "SYNTHESIZER_ASSESS")

    can_answer = result.get("can_answer_directly", False)
    confidence = result.get("confidence", 0)
    logger.info("Synthesizer: can_answer=%s, confidence=%d", can_answer, confidence)

    if can_answer and confidence >= 70:
        # Generate direct answer (no retrieval needed)
        answer_messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {
                "role": "user",
                "content": SYNTHESIZER_GENERATE_DIRECT.format(enriched_prompt=enriched_prompt),
            },
        ]
        answer = call_executor(answer_messages)
        logger.info("Synthesizer: generated direct answer (%d chars)", len(answer))

        new_turn = ConversationTurn(
            role="synthesizer",
            content=answer,
            loop_count=loop_count,
        )
        return {
            **state,
            "answer": answer,
            "needs_research": False,
            "chat_history": chat_history + [new_turn],
        }
    else:
        # Request retrieval from Researcher
        information_needs: list[str] = result.get("information_needs", [enriched_prompt])
        logger.info("Synthesizer: needs research — %d information gaps", len(information_needs))
        return {
            **state,
            "needs_research": True,
            "information_needs": information_needs,
        }


def _generate_with_citations(
    state: RAGState,
    enriched_prompt: str,
    citations: list[Citation],
    chat_history: list[ConversationTurn],
    loop_count: int,
) -> RAGState:
    """Generate an answer grounded in the provided citations."""
    citations_text = _format_citations(citations)

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": SYNTHESIZER_GENERATE_WITH_CONTEXT.format(
                enriched_prompt=enriched_prompt,
                citations_text=citations_text,
            ),
        },
    ]

    answer = call_executor(messages)
    logger.info(
        "Synthesizer: generated citation-grounded answer (%d chars, %d citations)",
        len(answer),
        len(citations),
    )

    new_turn = ConversationTurn(
        role="synthesizer",
        content=answer,
        citations=citations,
        loop_count=loop_count,
    )
    return {
        **state,
        "answer": answer,
        "needs_research": False,
        "chat_history": chat_history + [new_turn],
    }
