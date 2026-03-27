"""
Facilitator agent node.

Responsibilities:
1. Understand the user's intent from their query.
2. Identify missing information needed for a complete answer.
3. Ask targeted clarification questions (max 3) or build an enriched prompt.
4. When receiving Validator feedback: formulate specific follow-up questions.

The Facilitator is the entry point of the agent graph and may pause
execution (needs_user_input=True) to await user clarification.
"""

from __future__ import annotations

import json
import logging

from src.agents.state import RAGState
from src.config import get_config
from src.generation.llm_client import call_planner
from src.generation.prompts import FACILITATOR_ANALYZE, FACILITATOR_ENRICH, FACILITATOR_SYSTEM
from src.models import ConversationTurn, ValidationResult

logger = logging.getLogger(__name__)


def _format_chat_history(history: list[ConversationTurn]) -> str:
    """Format conversation history as a readable string for the prompt."""
    if not history:
        return "（会話履歴なし）"
    lines: list[str] = []
    for turn in history[-6:]:  # Last 6 turns to keep context manageable
        lines.append(f"[{turn.role}]: {turn.content}")
    return "\n".join(lines)


def _format_validator_feedback(validation: ValidationResult | None) -> str:
    """Extract actionable feedback from Validator result."""
    if not validation:
        return "（フィードバックなし）"
    missing = "、".join(validation.missing_info) if validation.missing_info else "なし"
    return (
        f"検証スコア: {validation.scores.average:.1f}/100\n"
        f"理由: {validation.reason}\n"
        f"不足情報: {missing}"
    )


def _parse_llm_json(raw: str, context: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` wrapping
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s JSON: %s\nRaw: %s", context, exc, raw[:200])
        return {}


def run(state: RAGState) -> RAGState:
    """Execute the Facilitator node.

    Analyzes the user query (with optional Validator feedback) and either:
    - Returns needs_user_input=True with clarifying questions, or
    - Returns needs_user_input=False with an enriched_prompt ready for retrieval.

    Args:
        state: Current RAGState from LangGraph.

    Returns:
        Updated RAGState with enriched_prompt or feedback_to_user set.
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    validation = state.get("validation")
    loop_count = state.get("loop_count", 0)

    logger.info("Facilitator: analyzing query (loop=%d)", loop_count)

    # Build the analysis prompt
    prompt = FACILITATOR_ANALYZE.format(
        user_query=user_query,
        chat_history=_format_chat_history(chat_history),
        validator_feedback=_format_validator_feedback(validation),
    )

    messages = [
        {"role": "system", "content": FACILITATOR_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    raw = call_planner(messages, json_mode=True)
    result = _parse_llm_json(raw, "FACILITATOR_ANALYZE")

    if not result:
        # Fallback: use query as-is
        logger.warning("Facilitator: JSON parse failed, using raw query as enriched prompt")
        return {
            **state,
            "enriched_prompt": user_query,
            "needs_user_input": False,
            "feedback_to_user": "",
        }

    needs_clarification = result.get("needs_clarification", False)
    intent = result.get("intent", "")
    logger.info("Facilitator: intent=%r, needs_clarification=%s", intent, needs_clarification)

    if needs_clarification and loop_count == 0:
        # Ask the user for more information
        questions = result.get("questions", [])
        question_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        feedback = f"より良い回答のために、以下を教えてください:\n{question_text}"

        new_turn = ConversationTurn(
            role="facilitator",
            content=feedback,
            loop_count=loop_count,
        )
        return {
            **state,
            "needs_user_input": True,
            "feedback_to_user": feedback,
            "chat_history": chat_history + [new_turn],
        }
    else:
        # Build enriched prompt
        enriched = result.get("enriched_prompt", user_query)
        return {
            **state,
            "enriched_prompt": enriched,
            "needs_user_input": False,
            "feedback_to_user": "",
        }


def enrich_with_user_response(state: RAGState, user_response: str) -> RAGState:
    """Process the user's response to Facilitator clarification questions.

    Called by the UI layer when the user provides additional information
    after needs_user_input=True was set.

    Args:
        state: Current RAGState (should have needs_user_input=True).
        user_response: The user's follow-up message.

    Returns:
        Updated RAGState with enriched_prompt set and needs_user_input=False.
    """
    original_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    loop_count = state.get("loop_count", 0)

    logger.info("Facilitator: enriching with user response")

    prompt = FACILITATOR_ENRICH.format(
        original_query=original_query,
        user_response=user_response,
    )

    messages = [
        {"role": "system", "content": FACILITATOR_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    raw = call_planner(messages, json_mode=True)
    result = _parse_llm_json(raw, "FACILITATOR_ENRICH")

    enriched = result.get("enriched_prompt", f"{original_query}\n{user_response}")

    # Record user response in history
    new_turn = ConversationTurn(
        role="user",
        content=user_response,
        loop_count=loop_count,
    )

    return {
        **state,
        "enriched_prompt": enriched,
        "needs_user_input": False,
        "feedback_to_user": "",
        "chat_history": chat_history + [new_turn],
    }
