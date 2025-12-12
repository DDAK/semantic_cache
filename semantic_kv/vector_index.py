"""
Vector index implementations for semantic search.

Provides:
- HNSWIndex: Fast approximate nearest neighbor search (requires hnswlib)
- BruteForceIndex: Simple fallback that works everywhere
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BruteForceIndex:
    """
    Simple brute-force vector index.

    Uses numpy for cosine similarity computation.
    Suitable for small datasets (< 10k vectors) or as fallback.
    """

    def __init__(self, dim: int, max_elements: int = 100000):
        self._dim = dim
        self._max_elements = max_elements
        self._vectors: Dict[int, np.ndarray] = {}
        self._next_id = 0

    def add_items(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Add vectors to the index."""
        for vec, idx in zip(vectors, ids):
            self._vectors[idx] = vec / (np.linalg.norm(vec) + 1e-9)  # Normalize

    def knn_query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors.

        Returns:
            Tuple of (labels, distances) arrays
        """
        if not self._vectors:
            return np.array([[]]), np.array([[]])

        query = query.reshape(-1)
        query = query / (np.linalg.norm(query) + 1e-9)

        # Compute cosine similarity with all vectors
        similarities = []
        ids = []
        for idx, vec in self._vectors.items():
            sim = np.dot(query, vec)
            similarities.append(sim)
            ids.append(idx)

        similarities = np.array(similarities)
        ids = np.array(ids)

        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]

        labels = ids[top_indices].reshape(1, -1)
        # Convert similarity to distance (cosine distance = 1 - similarity)
        distances = (1 - similarities[top_indices]).reshape(1, -1)

        return labels, distances

    def get_current_count(self) -> int:
        """Get number of vectors in index."""
        return len(self._vectors)

    def set_ef(self, ef: int) -> None:
        """No-op for compatibility with HNSW."""
        pass


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) index.

    Fast approximate nearest neighbor search using hnswlib.
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50
    ):
        import hnswlib

        self._index = hnswlib.Index(space='cosine', dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        self._index.set_ef(ef_search)

    def add_items(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Add vectors to the index."""
        self._index.add_items(vectors, ids)

    def knn_query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors."""
        return self._index.knn_query(query, k=k)

    def get_current_count(self) -> int:
        """Get number of vectors in index."""
        return self._index.get_current_count()

    def set_ef(self, ef: int) -> None:
        """Set search-time ef parameter."""
        self._index.set_ef(ef)

    def save_index(self, path: str) -> None:
        """Save index to file."""
        self._index.save_index(path)

    def load_index(self, path: str) -> None:
        """Load index from file."""
        self._index.load_index(path)


def create_vector_index(
    dim: int,
    max_elements: int = 100000,
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 50,
    force_brute_force: bool = False
):
    """
    Create a vector index, using HNSW if available, else brute-force.

    Args:
        dim: Vector dimension
        max_elements: Maximum number of vectors
        ef_construction: HNSW build parameter
        M: HNSW connections parameter
        ef_search: HNSW search parameter
        force_brute_force: Force use of brute-force index

    Returns:
        Vector index instance
    """
    if force_brute_force:
        logger.info("Using BruteForceIndex (forced)")
        return BruteForceIndex(dim, max_elements)

    try:
        import hnswlib
        index = HNSWIndex(
            dim=dim,
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            ef_search=ef_search
        )
        logger.info("Using HNSWIndex")
        return index
    except (ImportError, OSError) as e:
        logger.warning(f"hnswlib not available ({e}), falling back to BruteForceIndex")
        return BruteForceIndex(dim, max_elements)
