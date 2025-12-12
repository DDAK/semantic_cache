"""
Semantic KV Cache - Distributed Key-Value Store with Semantic Search

A distributed KV store built on Apache Arrow Flight that supports
semantic similarity search in addition to exact key lookups.

Usage:
    # Start nodes (in separate processes)
    from semantic_kv import SemanticKVNode
    node = SemanticKVNode("grpc://0.0.0.0:8815", node_id="node-a")
    node.serve()

    # Use distributed client
    from semantic_kv import DistributedSemanticKV

    kv = DistributedSemanticKV([
        "grpc://localhost:8815",
        "grpc://localhost:8816",
        "grpc://localhost:8817",
        "grpc://localhost:8818",
    ])

    # Store and retrieve
    kv.put("user:123:profile", b'{"name": "John"}')
    value = kv.get("user:123:profile")

    # Semantic search
    results = kv.search("user profile information", top_k=5)
    for key, value, similarity, node in results:
        print(f"{key}: {value} (score: {similarity:.2f} from {node})")
"""

from .node import SemanticKVNode, run_node
from .client import DistributedSemanticKV, SemanticKVClient
from .hash_ring import ConsistentHashRing
from .embeddings import (
    EmbeddingProvider,
    HashEmbedding,
    SentenceTransformerEmbedding,
    create_embedding_provider,
)
from .persistence import ArrowKVStore, SnapshotManager
from .vector_index import BruteForceIndex, HNSWIndex, create_vector_index

__version__ = "0.1.0"

__all__ = [
    # Node
    "SemanticKVNode",
    "run_node",
    # Client
    "DistributedSemanticKV",
    "SemanticKVClient",
    # Hash Ring
    "ConsistentHashRing",
    # Embeddings
    "EmbeddingProvider",
    "HashEmbedding",
    "SentenceTransformerEmbedding",
    "create_embedding_provider",
    # Persistence
    "ArrowKVStore",
    "SnapshotManager",
    # Vector Index
    "BruteForceIndex",
    "HNSWIndex",
    "create_vector_index",
]
