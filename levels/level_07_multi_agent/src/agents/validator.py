"""
Validator agent node — LLM-as-Judge.

Evaluates the Synthesizer's answer on 4 axes (each 0-100):
  1. completeness  — all enriched_prompt requirements covered
  2. accuracy      — no contradictions with citation sources
  3. relevance     — aligns with the user's original intent
  4. faithfulness  — no hallucinated content beyond citations

Pass threshold: 4-axis average >= config.agents.validation_threshold (default 80).

On failure: populates missing_info with specific gaps for Facilitator to relay
to the user. After max_loop_count loops: always returns is_valid=True to
prevent infinite looping.
"""

from __future__ import annotations

import json
import logging

from src.agents.state import RAGState
from src.config import get_config
from src.generation.llm_client import call_planner
from src.generation.prompts import VALIDATOR_EVALUATE, VALIDATOR_SYSTEM
from src.models import Citation, ConversationTurn, ValidationResult, ValidationScores

logger = logging.getLogger(__name__)


def _format_citations(citations: list[Citation]) -> str:
    """Format citations for the validator prompt."""
    if not citations:
        return "(No citation information — direct answer)"
    lines: list[str] = []
    for cit in citations:
        page_info = f" (p.{cit.page_number})" if cit.page_number else ""
        lines.append(
            f"[{cit.citation_id}] {cit.source_file}{page_info}\n"
            f"{cit.original_text}"
        )
    return "\n\n".join(lines)


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Validator: failed to parse JSON: %s\nRaw: %s", exc, raw[:200])
        return {}


def run(state: RAGState) -> RAGState:
    """Execute the Validator node.

    Scores the Synthesizer's answer and decides whether to accept it or
    send it back to the Facilitator for improvement.

    At max_loop_count, always passes to avoid infinite loops.

    Args:
        state: Current RAGState with answer and citations populated.

    Returns:
        Updated state with validation result and incremented loop_count.
    """
    cfg = get_config()
    threshold = cfg.agents.validation_threshold
    max_loops = cfg.agents.max_loop_count

    user_query = state.get("user_query", "")
    enriched_prompt = state.get("enriched_prompt", user_query)
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    chat_history = state.get("chat_history", [])
    loop_count = state.get("loop_count", 0)

    logger.info("Validator: evaluating answer (loop=%d/%d)", loop_count, max_loops)

    # Build scoring prompt with threshold substituted
    system_prompt = VALIDATOR_SYSTEM.format(threshold=threshold)
    citations_text = _format_citations(citations)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": VALIDATOR_EVALUATE.format(
                user_query=user_query,
                enriched_prompt=enriched_prompt,
                answer=answer,
                citations_text=citations_text,
            ),
        },
    ]

    response = call_planner(messages, json_mode=True)
    raw = response.answer
    result = _parse_llm_json(raw)

    # Save thinking to state
    agent_thinking = list(state.get("agent_thinking", []))
    if response.thinking:
        agent_thinking.append({"agent": "Validator", "thinking": response.thinking})

    # Parse scores with safe defaults
    completeness = int(result.get("completeness", 0))
    accuracy = int(result.get("accuracy", 0))
    relevance = int(result.get("relevance", 0))
    faithfulness = int(result.get("faithfulness", 0))
    average = float(result.get("average", (completeness + accuracy + relevance + faithfulness) / 4))
    reason = result.get("reason", "Failed to parse evaluation result")
    missing_info: list[str] = result.get("missing_info", [])

    # Force pass at max loops
    force_pass = loop_count + 1 >= max_loops
    is_valid_by_score = average >= threshold
    is_valid = is_valid_by_score or force_pass

    if force_pass and not is_valid_by_score:
        reason = f"[Max loops reached ({max_loops})] " + reason
        logger.warning(
            "Validator: max loops reached (%d), forcing pass. avg=%.1f",
            max_loops, average,
        )

    scores = ValidationScores(
        completeness=completeness,
        accuracy=accuracy,
        relevance=relevance,
        faithfulness=faithfulness,
        average=average,
    )
    validation = ValidationResult(
        scores=scores,
        is_valid=is_valid,
        reason=reason,
        missing_info=missing_info,
    )

    logger.info(
        "Validator: avg=%.1f, valid=%s (completeness=%d, accuracy=%d, relevance=%d, faithfulness=%d)",
        average, is_valid, completeness, accuracy, relevance, faithfulness,
    )

    # Add validation turn to history
    new_turn = ConversationTurn(
        role="system",
        content=f"Validator score: {average:.1f}/100 — {'PASS' if is_valid else 'FAIL'}",
        validation=validation,
        loop_count=loop_count + 1,
    )

    return {
        **state,
        "validation": validation,
        "loop_count": loop_count + 1,
        "chat_history": chat_history + [new_turn],
        "agent_thinking": agent_thinking,
    }
