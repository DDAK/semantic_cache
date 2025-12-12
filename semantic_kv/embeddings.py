"""
Embedding providers for semantic key matching.

Supports:
- SentenceTransformers (local, recommended)
- OpenAI embeddings (API-based)
- Simple hash-based fallback (for testing)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
import hashlib


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as float32 numpy array
        """
        pass

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            2D numpy array of shape (len(texts), dimension)
        """
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


class HashEmbedding(EmbeddingProvider):
    """
    Deterministic hash-based pseudo-embeddings.

    Useful for testing without loading ML models.
    NOT suitable for real semantic search - keys must match exactly
    for high similarity.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic pseudo-embedding from hash."""
        # Use hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._dimension).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class SentenceTransformerEmbedding(EmbeddingProvider):
    """
    Embeddings using sentence-transformers library.

    Recommended models:
    - 'all-MiniLM-L6-v2': Fast, 384 dims, good quality
    - 'all-mpnet-base-v2': Better quality, 768 dims, slower
    - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using sentence transformer."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding is more efficient with transformers."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)


class OpenAIEmbedding(EmbeddingProvider):
    """
    Embeddings using OpenAI API.

    Models:
    - 'text-embedding-3-small': 1536 dims, cheap
    - 'text-embedding-3-large': 3072 dims, better quality
    - 'text-embedding-ada-002': 1536 dims, legacy
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        self._model = model
        self._client = openai.OpenAI(api_key=api_key)

        # Dimension depends on model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        response = self._client.embeddings.create(
            model=self._model,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding with OpenAI API."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts
        )
        embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
        return np.array(embeddings)


def create_embedding_provider(
    provider: str = "hash",
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider: One of "hash", "sentence-transformer", "openai"
        **kwargs: Provider-specific arguments

    Returns:
        EmbeddingProvider instance
    """
    if provider == "hash":
        return HashEmbedding(dimension=kwargs.get("dimension", 384))

    elif provider == "sentence-transformer":
        model = kwargs.get("model", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbedding(model_name=model)

    elif provider == "openai":
        model = kwargs.get("model", "text-embedding-3-small")
        api_key = kwargs.get("api_key")
        return OpenAIEmbedding(model=model, api_key=api_key)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
