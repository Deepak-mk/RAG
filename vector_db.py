"""
vector_db.py - Qdrant vector database client.

Supports both:
  - Local Docker Qdrant (dev): host=localhost, port=6333
  - Qdrant Cloud (prod/Streamlit): url=QDRANT_URL, api_key=QDRANT_API_KEY
"""

import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)

from customtypes import RagSearchResult

COLLECTION_NAME = "documents"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimensions


class QuadrantStorage:
    """Wrapper around Qdrant for managing document vectors.
    
    Automatically connects to Qdrant Cloud if QDRANT_URL is set,
    otherwise falls back to localhost:6333 (Docker dev mode).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
    ):
        if url:
            # Qdrant Cloud mode
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"[vector_db] Connected to Qdrant Cloud: {url}")
        else:
            # Local Docker mode
            self.client = QdrantClient(host=host, port=port)
            print(f"[vector_db] Connected to local Qdrant at {host}:{port}")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[vector_db] Created collection '{COLLECTION_NAME}'.")
        else:
            print(f"[vector_db] Collection '{COLLECTION_NAME}' already exists.")

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> int:
        """Upsert a batch of vectors with associated payloads."""
        assert len(vectors) == len(payloads), "Vectors and payloads must have the same length."
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            )
            for vec, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[RagSearchResult]:
        """Search the collection for the top-k most similar vectors."""
        results: List[ScoredPoint] = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RagSearchResult(
                text=hit.payload.get("text", ""),
                source=hit.payload.get("source", "unknown"),
                score=hit.score,
            )
            for hit in results
        ]
