"""
customtypes.py - Pydantic models for strict type checking across the RAG pipeline.
"""

from pydantic import BaseModel
from typing import List, Optional


class RagChunkAndSrc(BaseModel):
    """Represents a text chunk along with its source metadata."""
    text: str
    source: str         # e.g. filename or document identifier
    chunk_index: int    # position of this chunk in the document


class RagUpsertResult(BaseModel):
    """Result returned after upserting vectors into Qdrant."""
    source: str
    chunks_upserted: int
    status: str         # "success" | "error"


class RagSearchResult(BaseModel):
    """Represents a single search result returned from Qdrant."""
    text: str
    source: str
    score: float        # cosine similarity score
