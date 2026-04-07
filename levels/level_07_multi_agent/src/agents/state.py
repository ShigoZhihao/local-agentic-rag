"""
LangGraph shared state definition.

RAGState is the TypedDict that all agent nodes read from and write to.
LangGraph manages merging partial updates returned by each node.

All complex objects use the Pydantic models from src/models.py for type safety.
"""

from __future__ import annotations

from typing import TypedDict

from src.models import Citation, ConversationTurn, ValidationResult


class RAGState(TypedDict, total=False):
    """Shared state object passed through the LangGraph agent graph.

    Attributes:
        user_query: The original user input string (never modified).
        enriched_prompt: The Facilitator's rewritten, search-optimized prompt.
        chat_history: All conversation turns in order.
        citations: Search results from the Researcher (Citation objects).
        answer: The Synthesizer's generated answer string.
        validation: The Validator's latest scoring result.
        loop_count: Number of Validator→Facilitator feedback loops so far.
        needs_user_input: True when Facilitator is waiting for user clarification.
        feedback_to_user: Message to display to the user (questions or status).
        needs_research: True when Synthesizer cannot answer directly and
                        delegates to Researcher.
        information_needs: List of specific information gaps Synthesizer
                           identified (passed to Researcher).
    """

    user_query: str
    enriched_prompt: str
    chat_history: list[ConversationTurn]
    citations: list[Citation]
    answer: str
    validation: ValidationResult | None
    loop_count: int
    needs_user_input: bool
    feedback_to_user: str
    needs_research: bool
    information_needs: list[str]
    agent_thinking: list[dict]
