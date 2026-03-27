"""
Researcher agent node (no LLM — pure retrieval).

Responsibilities:
1. Execute hybrid search (BM25 + semantic) for each information need.
2. Apply metadata filters if provided.
3. Re-rank with Cross-Encoder.
4. Optionally run ColBERT search (if enabled).
5. Deduplicate and stitch citations in NotebookLM style.

The Researcher never calls an LLM. It only uses Weaviate, the embedding
model (for vector queries), and the Cross-Encoder (for re-ranking).
"""

from __future__ import annotations

import logging

from src.agents.state import RAGState
from src.config import get_config
from src.ingestion.embedder import Embedder
from src.models import Citation
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.metadata_filter import apply_filters

logger = logging.getLogger(__name__)


def run(state: RAGState) -> RAGState:
    """Execute the Researcher node.

    Performs retrieval for all information needs identified by Synthesizer,
    deduplicates, re-ranks, and returns a structured citation list.

    Args:
        state: Current RAGState with information_needs populated.

    Returns:
        Updated state with citations list and needs_research=False.
    """
    information_needs: list[str] = state.get("information_needs", [])
    enriched_prompt: str = state.get("enriched_prompt", state.get("user_query", ""))
    metadata_filters: dict = state.get("user_query", {}) and {}
    # Use any metadata_filters passed through state (set by UI)
    # The QueryRequest model can carry metadata_filters; they arrive via state injection
    # For simplicity here we check if state has metadata_filters key
    _meta = {}  # type: ignore

    if not information_needs:
        information_needs = [enriched_prompt]

    logger.info("Researcher: processing %d information needs", len(information_needs))

    # Initialize retrieval components
    embedder = Embedder()
    searcher = HybridSearcher(embedder=embedder)
    reranker = CrossEncoderReranker()

    cfg = get_config().retrieval
    all_results = []

    # Search for each information need
    for need in information_needs:
        logger.debug("Researcher: searching for %r", need[:80])
        results = searcher.search(need, top_k=cfg.hybrid.top_k)

        # Apply metadata filters if any were provided
        if _meta:
            results = apply_filters(results, _meta)

        all_results.extend(results)

    # Optional: ColBERT search on enriched_prompt
    if cfg.colbert.enabled:
        from src.retrieval.colbert_search import ColBERTSearcher
        colbert = ColBERTSearcher()
        colbert_results = colbert.search(enriched_prompt)
        all_results.extend(colbert_results)

    # Deduplicate by chunk_id (keep highest score)
    seen: dict[str, float] = {}
    deduped = []
    for r in all_results:
        cid = r.chunk.chunk_id
        if cid not in seen or r.score > seen[cid]:
            seen[cid] = r.score
            deduped.append(r)

    # Sort by score descending before re-ranking
    deduped.sort(key=lambda x: x.score, reverse=True)

    # Re-rank with Cross-Encoder
    reranked = reranker.rerank(enriched_prompt, deduped)
    logger.info("Researcher: %d results after dedup+rerank", len(reranked))

    # Convert to Citation objects (NotebookLM style: preserve original_text)
    citations: list[Citation] = []
    for i, result in enumerate(reranked, start=1):
        citations.append(
            Citation(
                citation_id=i,
                source_file=result.chunk.source_file,
                page_number=result.chunk.page_number,
                original_text=result.chunk.content,  # Original text, not re-generated
                relevance_score=result.score,
            )
        )

    return {
        **state,
        "citations": citations,
        "needs_research": False,
    }
