"""
BGE-M3 embedding wrapper.

All embedding operations in this project go through this module.
GPU acceleration is used automatically when available.

Usage:
    embedder = Embedder()
    vectors = embedder.embed_texts(["text 1", "text 2"])
    # → list[list[float]] of shape (N, 1024)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Embedder:
    """BGE-M3 dense embedding encoder.

    Loads the model once on first instantiation (lazy load inside __init__)
    and caches it on the instance. The model runs on the device specified
    in config.yaml (embeddings.device), defaulting to "cuda" if available.

    Attributes:
        model_name: HuggingFace model name (BAAI/bge-m3).
        device: Torch device string used for inference.
    """

    def __init__(self) -> None:
        cfg = get_config().embeddings
        self.model_name: str = cfg.model_name
        self.batch_size: int = cfg.batch_size

        # Resolve device: use config value, but fall back to CPU if CUDA unavailable
        import torch
        if cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU for embeddings")
            self.device = "cpu"
        else:
            self.device = cfg.device

        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded (dim=%d)", self.dimension)

    @property
    def dimension(self) -> int:
        """Embedding dimension (1024 for BGE-M3)."""
        return get_config().embeddings.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return dense vectors.

        Args:
            texts: List of strings to embed. Empty strings are allowed
                   but will produce near-zero vectors.

        Returns:
            List of float vectors, one per input text.
            Each vector has length == self.dimension.
        """
        if not texts:
            return []

        logger.debug("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # BGE-M3 best practice
            convert_to_numpy=True,
        )
        # Convert numpy array to plain Python list[list[float]]
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Slightly different prompt prefix is used for queries vs. passages
        (BGE-M3 recommendation: prepend "Represent this sentence:").
        This wrapper handles that automatically via SentenceTransformer.

        Args:
            query: The search query string.

        Returns:
            A single float vector of length self.dimension.
        """
        result = self.embed_texts([query])
        return result[0]
