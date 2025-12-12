"""
Single node server for the distributed semantic KV cache.

Each node stores a shard of the total data and provides:
- Key-value storage with Arrow format
- HNSW index for local semantic search
- Async replication to backup nodes
"""

import pyarrow as pa
import pyarrow.flight as flight
import numpy as np
import threading
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import EmbeddingProvider, HashEmbedding
from .vector_index import create_vector_index

logger = logging.getLogger(__name__)


@dataclass
class KVEntry:
    """In-memory representation of a key-value entry."""
    key: str
    value: bytes
    embedding: np.ndarray
    created_at: int  # milliseconds since epoch
    ttl_ms: int  # 0 = no expiration
    access_count: int = 0
    last_accessed: int = 0


class SemanticKVNode(flight.FlightServerBase):
    """
    Single node in the distributed semantic KV cluster.

    Each node is responsible for a shard of keys (determined by
    consistent hashing on the client side). Provides:

    - put: Store key-value pair
    - put_replica: Store replica (internal, from other nodes)
    - get: Exact key lookup
    - search_local: Semantic search on local HNSW index
    - delete: Remove key
    - scan: List keys by prefix
    - stats: Node statistics
    - health: Health check
    """

    def __init__(
        self,
        location: str = "grpc://0.0.0.0:8815",
        node_id: str = "node-0",
        embedding_provider: EmbeddingProvider = None,
        replica_nodes: List[str] = None,
        data_dir: Optional[str] = None,
        max_entries: int = 100000,
        hnsw_ef_construction: int = 200,
        hnsw_m: int = 16,
        hnsw_ef_search: int = 50,
    ):
        """
        Initialize a semantic KV node.

        Args:
            location: gRPC address to listen on
            node_id: Unique identifier for this node
            embedding_provider: Provider for generating embeddings
            replica_nodes: List of node addresses to replicate to
            data_dir: Directory for persistence (None = in-memory only)
            max_entries: Maximum number of entries in HNSW index
            hnsw_ef_construction: HNSW build-time parameter
            hnsw_m: HNSW connections per node
            hnsw_ef_search: HNSW search-time parameter
        """
        super().__init__(location)

        self.node_id = node_id
        self._location = location
        self._embedding_provider = embedding_provider or HashEmbedding(dimension=384)
        self._replica_nodes = replica_nodes or []
        self._data_dir = Path(data_dir) if data_dir else None
        self._max_entries = max_entries

        # Thread safety
        self._lock = threading.RLock()

        # Storage
        self._store: Dict[str, KVEntry] = {}

        # Vector index for semantic search (HNSW if available, else brute-force)
        self._index = create_vector_index(
            dim=self._embedding_provider.dimension,
            max_elements=max_entries,
            ef_construction=hnsw_ef_construction,
            M=hnsw_m,
            ef_search=hnsw_ef_search
        )

        # Index mappings
        self._key_to_idx: Dict[str, int] = {}
        self._idx_to_key: Dict[int, str] = {}
        self._next_idx = 0

        # Metrics
        self._metrics = {
            'puts': 0,
            'gets': 0,
            'searches': 0,
            'hits': 0,
            'misses': 0,
            'deletes': 0,
            'replications': 0,
        }

        # Start time
        self._start_time = time.time()

        logger.info(f"Node {node_id} initialized at {location}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the provider."""
        return self._embedding_provider.embed(text)

    def _is_expired(self, entry: KVEntry) -> bool:
        """Check if an entry has expired."""
        if entry.ttl_ms <= 0:
            return False
        now_ms = int(time.time() * 1000)
        return (now_ms - entry.created_at) > entry.ttl_ms

    def _delete_entry(self, key: str) -> bool:
        """Delete an entry from storage and index."""
        if key not in self._store:
            return False

        del self._store[key]

        if key in self._key_to_idx:
            idx = self._key_to_idx[key]
            if idx in self._idx_to_key:
                del self._idx_to_key[idx]
            del self._key_to_idx[key]

        return True

    def _replicate_async(self, data: dict) -> None:
        """Asynchronously replicate to backup nodes."""
        if not self._replica_nodes:
            return

        def do_replicate():
            for node_addr in self._replica_nodes:
                try:
                    client = flight.FlightClient(node_addr)
                    action = flight.Action("put_replica", json.dumps(data).encode())
                    list(client.do_action(action))
                    client.close()
                    self._metrics['replications'] += 1
                except Exception as e:
                    logger.warning(f"Replication to {node_addr} failed: {e}")

        thread = threading.Thread(target=do_replicate, daemon=True)
        thread.start()

    # ==================== Flight Methods ====================

    def do_action(self, context, action):
        """Handle all KV operations via Flight actions."""
        action_type = action.type
        body = action.body.to_pybytes() if action.body else b""

        try:
            if action_type == "put":
                return self._action_put(body, is_replica=False)
            elif action_type == "put_replica":
                return self._action_put(body, is_replica=True)
            elif action_type == "get":
                return self._action_get(body)
            elif action_type == "search_local":
                return self._action_search_local(body)
            elif action_type == "delete":
                return self._action_delete(body)
            elif action_type == "scan":
                return self._action_scan(body)
            elif action_type == "stats":
                return self._action_stats()
            elif action_type == "health":
                return [flight.Result(b"ok")]
            elif action_type == "clear":
                return self._action_clear()
            else:
                raise flight.FlightBadRequestError(f"Unknown action: {action_type}")

        except Exception as e:
            logger.error(f"Error in action {action_type}: {e}")
            raise

    def _action_put(self, body: bytes, is_replica: bool = False):
        """Store a key-value pair."""
        data = json.loads(body.decode())
        key = data['key']

        # Handle value encoding
        value = data['value']
        if isinstance(value, str):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = bytes(value)

        ttl_ms = data.get('ttl_ms', 0)

        # Generate embedding
        embedding = self._get_embedding(key)

        with self._lock:
            now_ms = int(time.time() * 1000)

            # Remove existing entry if present
            if key in self._store:
                old_idx = self._key_to_idx.get(key)
                if old_idx is not None:
                    if old_idx in self._idx_to_key:
                        del self._idx_to_key[old_idx]
                    del self._key_to_idx[key]

            # Create entry
            entry = KVEntry(
                key=key,
                value=value,
                embedding=embedding,
                created_at=now_ms,
                ttl_ms=ttl_ms,
                access_count=0,
                last_accessed=now_ms
            )
            self._store[key] = entry

            # Add to HNSW index
            idx = self._next_idx
            self._index.add_items(embedding.reshape(1, -1), [idx])
            self._key_to_idx[key] = idx
            self._idx_to_key[idx] = key
            self._next_idx += 1

            self._metrics['puts'] += 1

        # Replicate to backup nodes (async, fire-and-forget)
        if not is_replica:
            self._replicate_async(data)

        return [flight.Result(b"ok")]

    def _action_get(self, body: bytes):
        """Get value by exact key match."""
        key = body.decode()

        with self._lock:
            self._metrics['gets'] += 1

            if key not in self._store:
                self._metrics['misses'] += 1
                return [flight.Result(b"")]

            entry = self._store[key]

            # Check expiration
            if self._is_expired(entry):
                self._delete_entry(key)
                self._metrics['misses'] += 1
                return [flight.Result(b"")]

            # Update access stats
            entry.access_count += 1
            entry.last_accessed = int(time.time() * 1000)

            self._metrics['hits'] += 1
            return [flight.Result(entry.value)]

    def _action_search_local(self, body: bytes):
        """
        Search local HNSW index for similar keys.

        This is called by the distributed client during scatter-gather search.
        """
        params = json.loads(body.decode())
        query = params['query']
        top_k = params.get('top_k', 10)
        threshold = params.get('threshold', 0.7)

        query_embedding = self._get_embedding(query)

        with self._lock:
            self._metrics['searches'] += 1

            if self._index.get_current_count() == 0:
                return [flight.Result(b"[]")]

            # Search index
            k = min(top_k, self._index.get_current_count())
            labels, distances = self._index.knn_query(
                query_embedding.reshape(1, -1),
                k=k
            )

            results = []
            for label, dist in zip(labels[0], distances[0]):
                # Convert cosine distance to similarity
                similarity = 1.0 - dist

                if similarity >= threshold and label in self._idx_to_key:
                    key = self._idx_to_key[label]

                    if key in self._store:
                        entry = self._store[key]

                        # Skip expired entries
                        if self._is_expired(entry):
                            continue

                        results.append({
                            'key': key,
                            'value': entry.value.decode('utf-8', errors='replace'),
                            'similarity': float(similarity),
                            'node': self.node_id
                        })

            return [flight.Result(json.dumps(results).encode())]

    def _action_delete(self, body: bytes):
        """Delete a key."""
        key = body.decode()

        with self._lock:
            self._metrics['deletes'] += 1

            if self._delete_entry(key):
                return [flight.Result(b"deleted")]
            return [flight.Result(b"not_found")]

    def _action_scan(self, body: bytes):
        """List keys, optionally filtered by prefix."""
        params = json.loads(body.decode())
        prefix = params.get('prefix', '')
        limit = params.get('limit', 100)

        with self._lock:
            keys = []
            for key in self._store.keys():
                if key.startswith(prefix):
                    entry = self._store[key]
                    if not self._is_expired(entry):
                        keys.append(key)
                        if len(keys) >= limit:
                            break

            return [flight.Result(json.dumps(keys).encode())]

    def _action_stats(self):
        """Return node statistics."""
        with self._lock:
            uptime = time.time() - self._start_time
            total_ops = self._metrics['gets'] + self._metrics['puts']
            hit_rate = (
                self._metrics['hits'] / self._metrics['gets']
                if self._metrics['gets'] > 0 else 0.0
            )

            stats = {
                'node_id': self.node_id,
                'location': self._location,
                'total_keys': len(self._store),
                'index_size': self._index.get_current_count(),
                'embedding_dim': self._embedding_provider.dimension,
                'uptime_seconds': uptime,
                'metrics': self._metrics.copy(),
                'hit_rate': hit_rate,
                'total_ops': total_ops,
            }

            return [flight.Result(json.dumps(stats).encode())]

    def _action_clear(self):
        """Clear all data from this node."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._key_to_idx.clear()
            self._idx_to_key.clear()

            # Reinitialize vector index
            self._index = create_vector_index(
                dim=self._embedding_provider.dimension,
                max_elements=self._max_entries
            )
            self._next_idx = 0

            return [flight.Result(f"cleared:{count}".encode())]

    def list_actions(self, context):
        """List available Flight actions."""
        return [
            flight.ActionType("put", "Store key-value pair"),
            flight.ActionType("put_replica", "Store replica (internal)"),
            flight.ActionType("get", "Get value by exact key"),
            flight.ActionType("search_local", "Semantic search on local index"),
            flight.ActionType("delete", "Delete key"),
            flight.ActionType("scan", "List keys by prefix"),
            flight.ActionType("stats", "Get node statistics"),
            flight.ActionType("health", "Health check"),
            flight.ActionType("clear", "Clear all data"),
        ]


def run_node(
    host: str = "0.0.0.0",
    port: int = 8815,
    node_id: str = None,
    embedding_provider: str = "hash",
    embedding_model: str = "all-MiniLM-L6-v2",
    replica_nodes: List[str] = None,
):
    """
    Run a semantic KV node.

    Args:
        host: Host to bind to
        port: Port to listen on
        node_id: Node identifier (defaults to host:port)
        embedding_provider: "hash", "sentence-transformer", or "openai"
        embedding_model: Model name for the embedding provider
        replica_nodes: List of replica node addresses
    """
    from .embeddings import create_embedding_provider

    location = f"grpc://{host}:{port}"
    node_id = node_id or f"{host}:{port}"

    # Create embedding provider
    if embedding_provider == "hash":
        provider = create_embedding_provider("hash")
    elif embedding_provider == "sentence-transformer":
        provider = create_embedding_provider(
            "sentence-transformer",
            model=embedding_model
        )
    elif embedding_provider == "openai":
        provider = create_embedding_provider("openai", model=embedding_model)
    else:
        provider = create_embedding_provider("hash")

    # Create and start node
    node = SemanticKVNode(
        location=location,
        node_id=node_id,
        embedding_provider=provider,
        replica_nodes=replica_nodes,
    )

    logger.info(f"Starting semantic KV node {node_id} on {location}")
    node.serve()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run a semantic KV node")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8815, help="Port to listen on")
    parser.add_argument("--node-id", help="Node identifier")
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "sentence-transformer", "openai"],
        default="hash",
        help="Embedding provider to use"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Model name for embedding provider"
    )
    parser.add_argument(
        "--replica-nodes",
        nargs="*",
        help="Replica node addresses"
    )

    args = parser.parse_args()

    run_node(
        host=args.host,
        port=args.port,
        node_id=args.node_id,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        replica_nodes=args.replica_nodes,
    )
