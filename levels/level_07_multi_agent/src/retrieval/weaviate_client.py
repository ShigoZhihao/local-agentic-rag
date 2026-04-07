"""
Weaviate client wrapper.

Handles:
- Connecting to a local Weaviate instance
- Creating/resetting the Documents collection (with BM25 config)
- Upserting and deleting chunks
- Low-level query helpers used by hybrid_search.py

Usage:
    from src.retrieval.weaviate_client import get_client, ensure_collection
    client = get_client()
    ensure_collection(client)
"""

from __future__ import annotations

import logging
import uuid

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter

from src.config import get_config
from src.models import Chunk

logger = logging.getLogger(__name__)


def get_client() -> weaviate.WeaviateClient:
    """Create and return a connected Weaviate client.

    Uses host/port from config.yaml.

    Returns:
        Connected WeaviateClient instance.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is not running.
    """
    cfg = get_config().weaviate
    client = weaviate.connect_to_local(
        host=cfg.host,
        port=cfg.port,
        grpc_port=cfg.grpc_port,
    )
    logger.debug("Connected to Weaviate at %s:%s", cfg.host, cfg.port)
    return client


def ensure_collection(
    client: weaviate.WeaviateClient,
    *,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    recreate: bool = False,
) -> None:
    """Create the Documents collection if it does not exist.

    Args:
        client: Connected Weaviate client.
        bm25_k1: BM25 term frequency saturation (overrides config if given).
        bm25_b: BM25 document length normalization (overrides config if given).
        recreate: If True, delete and recreate the collection.
                  Used by the BM25 tuning pipeline.
    """
    cfg = get_config()
    collection_name = cfg.weaviate.collection_name
    k1 = bm25_k1 if bm25_k1 is not None else cfg.retrieval.bm25.k1
    b = bm25_b if bm25_b is not None else cfg.retrieval.bm25.b

    if recreate and client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        logger.info("Deleted collection '%s' for recreation", collection_name)

    if client.collections.exists(collection_name):
        logger.debug("Collection '%s' already exists — skipping creation", collection_name)
        return

    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="chunk_strategy", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="image_path", data_type=DataType.TEXT),
            # JSON-serialized extra metadata
            Property(name="metadata_json", data_type=DataType.TEXT),
        ],
        # HNSW vector index with cosine distance
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        # BM25 inverted index parameters
        inverted_index_config=Configure.inverted_index(
            bm25_b=b,
            bm25_k1=k1,
        ),
    )
    logger.info(
        "Created collection '%s' (BM25 k1=%.2f, b=%.2f)", collection_name, k1, b
    )


def upsert_chunks(
    client: weaviate.WeaviateClient,
    chunks: list[Chunk],
) -> None:
    """Insert or update chunks in Weaviate using batch operations.

    Chunks that already exist (same chunk_id) are overwritten.

    Args:
        client: Connected Weaviate client.
        chunks: List of Chunk objects with embeddings set.

    Raises:
        ValueError: If any chunk is missing its embedding.
    """
    cfg = get_config()
    collection = client.collections.get(cfg.weaviate.collection_name)

    import json

    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.chunk_id}' has no embedding")

            properties = {
                "content": chunk.content,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "source_type": chunk.source_type.value,
                "chunk_strategy": chunk.chunk_strategy.value,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "image_path": chunk.image_path or "",
                "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
            }

            # Use a deterministic UUID based on chunk_id so upsert is idempotent
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            batch.add_object(
                properties=properties,
                vector=chunk.embedding,
                uuid=obj_uuid,
            )

    logger.info("Upserted %d chunks into Weaviate", len(chunks))


def delete_by_doc_id(client: weaviate.WeaviateClient, doc_id: str) -> None:
    """Delete all chunks belonging to a document.

    Args:
        client: Connected Weaviate client.
        doc_id: The document ID whose chunks should be removed.
    """
    cfg = get_config()
    collection = client.collections.get(cfg.weaviate.collection_name)
    collection.data.delete_many(
        where=Filter.by_property("doc_id").equal(doc_id)
    )
    logger.info("Deleted chunks for doc_id='%s'", doc_id)


def _row_to_chunk(obj: wvc.query.Object) -> Chunk:
    """Convert a raw Weaviate object into a Chunk model.

    Args:
        obj: Raw Weaviate query result object.

    Returns:
        Reconstructed Chunk instance.
    """
    import json
    from src.models import ChunkStrategy, SourceType

    props = obj.properties
    page = props.get("page_number", -1)

    return Chunk(
        chunk_id=props["chunk_id"],
        doc_id=props["doc_id"],
        content=props["content"],
        chunk_index=int(props.get("chunk_index", 0)),
        chunk_strategy=ChunkStrategy(props["chunk_strategy"]),
        source_file=props["source_file"],
        source_type=SourceType(props["source_type"]),
        page_number=page if page >= 0 else None,
        image_path=props.get("image_path") or None,
        metadata=json.loads(props.get("metadata_json", "{}")),
    )
