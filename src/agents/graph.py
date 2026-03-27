"""
LangGraph agent graph definition.

Builds the 4-agent StateGraph:
  Facilitator → Synthesizer → [Researcher →] Synthesizer → Validator
                                ↑ (loop if not valid and < max_loops)

Human-in-the-Loop is implemented via interrupt_before=["facilitator"]:
- On first entry, Facilitator runs normally.
- If needs_user_input=True, LangGraph interrupts and the graph thread waits.
- The UI resumes via graph.update_state() + graph.stream().

Usage:
    graph = build_graph()
    config = {"configurable": {"thread_id": "session-123"}}
    for event in graph.stream(initial_state, config):
        handle(event)
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents import facilitator, researcher, synthesizer, validator
from src.agents.state import RAGState
from src.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (edges)
# ---------------------------------------------------------------------------

def _route_after_facilitator(state: RAGState) -> str:
    """Route after Facilitator runs.

    Returns:
        "wait_for_user" if Facilitator needs more info,
        "synthesizer" if enriched_prompt is ready.
    """
    if state.get("needs_user_input", False):
        return "wait_for_user"
    return "synthesizer"


def _route_after_synthesizer(state: RAGState) -> str:
    """Route after Synthesizer runs.

    Returns:
        "researcher" if Synthesizer needs retrieval,
        "validator" if answer is ready.
    """
    if state.get("needs_research", False):
        return "researcher"
    return "validator"


def _route_after_validator(state: RAGState) -> str:
    """Route after Validator scores the answer.

    Returns:
        "end" if validation passed or max loops reached,
        "facilitator" to ask user for more info and retry.
    """
    validation = state.get("validation")
    if validation and validation.is_valid:
        return "end"
    # loop_count was already incremented in validator.run()
    return "facilitator"


# ---------------------------------------------------------------------------
# Passthrough node (Human-in-the-Loop pause point)
# ---------------------------------------------------------------------------

def _wait_for_user(state: RAGState) -> RAGState:
    """No-op node that serves as the interrupt point for user input.

    LangGraph will pause here when interrupt_before=["wait_for_user"] is set.
    The UI updates state with user input then resumes the graph.
    """
    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(*, debug: bool = False) -> "CompiledStateGraph":  # noqa: F821
    """Build and compile the RAG agent graph.

    Args:
        debug: If True, enables LangGraph debug logging for state tracing.

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("facilitator", facilitator.run)
    graph.add_node("synthesizer", synthesizer.run)
    graph.add_node("researcher", researcher.run)
    graph.add_node("validator", validator.run)
    graph.add_node("wait_for_user", _wait_for_user)

    # Entry point
    graph.set_entry_point("facilitator")

    # Facilitator → synthesizer or wait_for_user
    graph.add_conditional_edges(
        "facilitator",
        _route_after_facilitator,
        {"synthesizer": "synthesizer", "wait_for_user": "wait_for_user"},
    )

    # wait_for_user → facilitator (user provides input via update_state)
    graph.add_edge("wait_for_user", "facilitator")

    # Synthesizer → researcher or validator
    graph.add_conditional_edges(
        "synthesizer",
        _route_after_synthesizer,
        {"researcher": "researcher", "validator": "validator"},
    )

    # Researcher → synthesizer (with citations in state)
    graph.add_edge("researcher", "synthesizer")

    # Validator → end or facilitator (loop)
    graph.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"end": END, "facilitator": "facilitator"},
    )

    # Compile with MemorySaver for Human-in-the-Loop + debugging
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_user"],  # Pause when user input needed
        debug=debug,
    )

    logger.info("RAG agent graph compiled (debug=%s)", debug)
    return compiled


# ---------------------------------------------------------------------------
# Convenience: create initial state
# ---------------------------------------------------------------------------

def make_initial_state(user_query: str) -> RAGState:
    """Create a clean initial RAGState for a new query.

    Args:
        user_query: The user's question string.

    Returns:
        RAGState with all fields initialized to sensible defaults.
    """
    return RAGState(
        user_query=user_query,
        enriched_prompt="",
        chat_history=[],
        citations=[],
        answer="",
        validation=None,
        loop_count=0,
        needs_user_input=False,
        feedback_to_user="",
        needs_research=False,
        information_needs=[],
    )
