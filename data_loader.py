"""
data_loader.py - PDF loading, chunking, and embedding logic.

Uses LlamaIndex for PDF parsing and SentenceSplitter for chunking.
Embeddings are generated locally via sentence-transformers (no API key needed).
"""

import os
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

from customtypes import RagChunkAndSrc

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, runs locally
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load model once at module level (cached on first run, ~80 MB)
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def load_and_chunk_pdf(file_path: str, original_filename: str | None = None) -> List[RagChunkAndSrc]:
    """
    Load a PDF and split it into overlapping text chunks.

    Args:
        file_path: Absolute or relative path to the PDF file.
        original_filename: Optional name to use for the 'source' metadata.

    Returns:
        List of RagChunkAndSrc objects containing text and source info.
    """
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    source_name = original_filename or os.path.basename(file_path)
    chunks = [
        RagChunkAndSrc(
            text=node.get_content(),
            source=source_name,
            chunk_index=i,
        )
        for i, node in enumerate(nodes)
    ]
    print(f"[data_loader] Loaded '{source_name}' → {len(chunks)} chunks.")
    return chunks


def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using a local model.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of 384-dimensional embedding vectors.
    """
    vectors = _embedding_model.encode(texts, convert_to_numpy=True).tolist()
    print(f"[data_loader] Generated {len(vectors)} embeddings (dim=384, local model).")
    return vectors
