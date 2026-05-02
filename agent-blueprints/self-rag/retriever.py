"""
FAISS-based dense retriever using sentence-transformers for the Self-RAG pipeline.

Uses all-MiniLM-L6-v2 to embed documents and queries, then performs
approximate nearest-neighbor search via FAISS flat index.
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports - only loaded when retriever is initialized
_faiss = None
_sentence_transformer = None


def _load_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformer model: {model_name}")
        _sentence_transformer = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
    return _sentence_transformer


class FAISSRetriever:
    """
    Dense retriever backed by a FAISS flat index.

    Embeds documents using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
    and performs cosine similarity search via FAISS IndexFlatIP.

    Usage:
        retriever = FAISSRetriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("What is RAG?", k=5)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize the retriever.

        Args:
            model_name: Sentence-transformer model identifier.
            embedding_dim: Embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents: list[dict[str, Any]] = []
        self._model = None
        logger.info(f"FAISSRetriever initialized (model={model_name}, dim={embedding_dim})")

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into normalized float32 vectors."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.astype(np.float32)

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Build the FAISS index from a list of documents.

        Each document should have a 'content' key.
        Additional fields (id, title, category) are stored for retrieval.

        Args:
            documents: List of document dicts with 'content' field.

        Raises:
            ValueError: If documents list is empty or missing 'content'.
        """
        if not documents:
            raise ValueError("Cannot index empty document list.")

        for i, doc in enumerate(documents):
            if "content" not in doc:
                raise ValueError(f"Document at index {i} is missing 'content' field.")

        logger.info(f"Indexing {len(documents)} documents...")
        self.documents = documents

        texts = [doc["content"] for doc in documents]
        embeddings = self._embed(texts)

        faiss = _load_faiss()

        # IndexFlatIP = exact inner product search (cosine sim on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.embedding_dim}")

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query: Natural language search query.
            k: Number of documents to return.

        Returns:
            List of document dicts with an added 'score' field (cosine similarity).

        Raises:
            RuntimeError: If index_documents() has not been called yet.
        """
        if self.index is None or not self.documents:
            raise RuntimeError("Retriever is not initialized. Call index_documents() first.")

        k = min(k, len(self.documents))
        logger.info(f"Retrieving top-{k} docs for query: '{query[:80]}'")

        query_embedding = self._embed([query])
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            doc["rank"] = len(results) + 1
            results.append(doc)

        logger.info(f"Retrieved {len(results)} documents." if results else "No results.")
        return results

    def save_index(self, path: str) -> None:
        """Persist the FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call index_documents() first.")
        faiss = _load_faiss()
        faiss.write_index(self.index, path)
        logger.info(f"FAISS index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load a previously saved FAISS index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        faiss = _load_faiss()
        self.index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path}: {self.index.ntotal} vectors")

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self.documents)
