"""
Distributed client for the semantic KV cache.

Routes operations to appropriate nodes using consistent hashing.
For semantic search, scatters query to ALL nodes and merges results.
"""

import pyarrow.flight as flight
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import time

from .hash_ring import ConsistentHashRing
from .embeddings import EmbeddingProvider, HashEmbedding

logger = logging.getLogger(__name__)


class DistributedSemanticKV:
    """
    Distributed client for semantic KV cache.

    Routes put/get/delete to responsible node via consistent hashing.
    Scatters search queries to ALL nodes and merges results.

    Usage:
        kv = DistributedSemanticKV([
            "grpc://node1:8815",
            "grpc://node2:8816",
            "grpc://node3:8817",
            "grpc://node4:8818",
        ])

        kv.put("user:123", b"John Doe")
        value = kv.get("user:123")
        results = kv.search("user information", top_k=5)
    """

    def __init__(
        self,
        nodes: List[str],
        replication_factor: int = 1,
        search_timeout: float = 5.0,
        connection_timeout: float = 2.0,
        max_workers: int = None,
        retry_count: int = 2,
    ):
        """
        Initialize the distributed client.

        Args:
            nodes: List of node addresses (e.g., ["grpc://localhost:8815"])
            replication_factor: Number of nodes to write to (1 = no replication)
            search_timeout: Timeout for scatter-gather search (seconds)
            connection_timeout: Timeout for individual connections (seconds)
            max_workers: Max threads for parallel operations
            retry_count: Number of retries on failure
        """
        if not nodes:
            raise ValueError("At least one node is required")

        self._nodes = nodes
        self._ring = ConsistentHashRing(nodes)
        self._replication_factor = replication_factor
        self._search_timeout = search_timeout
        self._connection_timeout = connection_timeout
        self._retry_count = retry_count

        # Thread pool for parallel operations
        self._max_workers = max_workers or len(nodes) * 2
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # Connection pool with lazy initialization
        self._clients: Dict[str, flight.FlightClient] = {}
        self._clients_lock = threading.Lock()

        # Metrics
        self._metrics = {
            'puts': 0,
            'gets': 0,
            'searches': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0,
        }

        logger.info(f"Distributed client initialized with {len(nodes)} nodes")

    def _get_client(self, node: str) -> flight.FlightClient:
        """Get or create a Flight client for a node."""
        with self._clients_lock:
            if node not in self._clients:
                try:
                    self._clients[node] = flight.FlightClient(node)
                except Exception as e:
                    logger.error(f"Failed to connect to {node}: {e}")
                    raise
            return self._clients[node]

    def _execute_with_retry(
        self,
        node: str,
        action_type: str,
        body: bytes,
        retries: int = None
    ) -> Optional[bytes]:
        """Execute an action with retry logic."""
        retries = retries if retries is not None else self._retry_count

        for attempt in range(retries + 1):
            try:
                client = self._get_client(node)
                action = flight.Action(action_type, body)
                results = list(client.do_action(action))
                if results:
                    return results[0].body.to_pybytes()
                return b""
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {node}: {e}")
                if attempt < retries:
                    # Remove stale connection
                    with self._clients_lock:
                        if node in self._clients:
                            del self._clients[node]
                    time.sleep(0.1 * (attempt + 1))  # Backoff
                else:
                    self._metrics['errors'] += 1
                    raise

        return None

    def put(
        self,
        key: str,
        value: bytes,
        ttl_ms: int = 0
    ) -> bool:
        """
        Store a key-value pair.

        The key is hashed to determine the responsible node.

        Args:
            key: The key string
            value: The value as bytes
            ttl_ms: Time-to-live in milliseconds (0 = no expiration)

        Returns:
            True if successful
        """
        node = self._ring.get_node(key)

        data = json.dumps({
            'key': key,
            'value': list(value),
            'ttl_ms': ttl_ms
        })

        try:
            result = self._execute_with_retry(node, "put", data.encode())
            self._metrics['puts'] += 1
            return result == b"ok"
        except Exception as e:
            logger.error(f"Put failed for key '{key}': {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """
        Get value by exact key match.

        Args:
            key: The key to look up

        Returns:
            The value as bytes, or None if not found
        """
        node = self._ring.get_node(key)

        try:
            result = self._execute_with_retry(node, "get", key.encode())
            self._metrics['gets'] += 1

            if result:
                self._metrics['hits'] += 1
                return result
            else:
                self._metrics['misses'] += 1
                return None
        except Exception as e:
            logger.error(f"Get failed for key '{key}': {e}")
            return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, bytes, float, str]]:
        """
        Distributed semantic search.

        Scatters the query to ALL nodes in parallel, then merges
        results by similarity score.

        Args:
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (key, value, similarity, node_id) tuples,
            sorted by similarity descending
        """
        params = json.dumps({
            'query': query,
            'top_k': top_k,
            'threshold': threshold
        })

        def search_node(node: str) -> List[dict]:
            """Search a single node."""
            try:
                result = self._execute_with_retry(
                    node,
                    "search_local",
                    params.encode(),
                    retries=1
                )
                if result:
                    return json.loads(result.decode())
                return []
            except Exception as e:
                logger.warning(f"Search failed on {node}: {e}")
                return []

        # Scatter: query all nodes in parallel
        all_nodes = self._ring.get_all_nodes()
        futures = {
            self._executor.submit(search_node, node): node
            for node in all_nodes
        }

        # Gather: collect results with timeout
        all_results = []
        try:
            for future in as_completed(futures, timeout=self._search_timeout):
                try:
                    node_results = future.result()
                    all_results.extend(node_results)
                except Exception as e:
                    node = futures[future]
                    logger.warning(f"Search result failed from {node}: {e}")
        except TimeoutError:
            logger.warning("Search timed out, returning partial results")

        self._metrics['searches'] += 1

        # Merge: sort by similarity, return top-k
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = all_results[:top_k]

        return [
            (r['key'], r['value'].encode(), r['similarity'], r['node'])
            for r in top_results
        ]

    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair.

        Args:
            key: The key to delete

        Returns:
            True if deleted, False if not found
        """
        node = self._ring.get_node(key)

        try:
            result = self._execute_with_retry(node, "delete", key.encode())
            return result == b"deleted"
        except Exception as e:
            logger.error(f"Delete failed for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists
        """
        return self.get(key) is not None

    def scan(
        self,
        prefix: str = "",
        limit: int = 100
    ) -> List[str]:
        """
        Scan keys across all nodes.

        Args:
            prefix: Optional key prefix to filter by
            limit: Maximum number of keys to return

        Returns:
            List of keys matching the prefix
        """
        params = json.dumps({
            'prefix': prefix,
            'limit': limit
        })

        def scan_node(node: str) -> List[str]:
            try:
                result = self._execute_with_retry(
                    node,
                    "scan",
                    params.encode(),
                    retries=1
                )
                if result:
                    return json.loads(result.decode())
                return []
            except Exception:
                return []

        # Gather from all nodes
        all_keys = []
        futures = [
            self._executor.submit(scan_node, node)
            for node in self._ring.get_all_nodes()
        ]

        for future in as_completed(futures, timeout=self._search_timeout):
            try:
                all_keys.extend(future.result())
            except Exception:
                continue

        # Deduplicate and sort
        unique_keys = sorted(set(all_keys))
        return unique_keys[:limit]

    def cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all nodes.

        Returns:
            Dict mapping node address to stats dict
        """
        def get_node_stats(node: str) -> dict:
            try:
                result = self._execute_with_retry(
                    node,
                    "stats",
                    b"",
                    retries=1
                )
                if result:
                    return json.loads(result.decode())
                return {'error': 'no response'}
            except Exception as e:
                return {'error': str(e)}

        stats = {}
        futures = {
            self._executor.submit(get_node_stats, node): node
            for node in self._ring.get_all_nodes()
        }

        for future in as_completed(futures, timeout=self._search_timeout):
            node = futures[future]
            try:
                stats[node] = future.result()
            except Exception as e:
                stats[node] = {'error': str(e)}

        # Add client-side metrics
        stats['_client'] = {
            'nodes': len(self._ring),
            'metrics': self._metrics.copy(),
        }

        return stats

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all nodes.

        Returns:
            Dict mapping node address to health status
        """
        def check_node(node: str) -> bool:
            try:
                result = self._execute_with_retry(
                    node,
                    "health",
                    b"",
                    retries=0
                )
                return result == b"ok"
            except Exception:
                return False

        health = {}
        futures = {
            self._executor.submit(check_node, node): node
            for node in self._ring.get_all_nodes()
        }

        for future in as_completed(futures, timeout=2.0):
            node = futures[future]
            try:
                health[node] = future.result()
            except Exception:
                health[node] = False

        return health

    def clear_all(self) -> Dict[str, int]:
        """
        Clear all data from all nodes.

        Returns:
            Dict mapping node address to number of entries cleared
        """
        def clear_node(node: str) -> int:
            try:
                result = self._execute_with_retry(node, "clear", b"")
                if result and result.startswith(b"cleared:"):
                    return int(result.decode().split(":")[1])
                return 0
            except Exception:
                return -1

        results = {}
        futures = {
            self._executor.submit(clear_node, node): node
            for node in self._ring.get_all_nodes()
        }

        for future in as_completed(futures, timeout=self._search_timeout):
            node = futures[future]
            try:
                results[node] = future.result()
            except Exception:
                results[node] = -1

        return results

    def close(self):
        """Close all connections and shutdown thread pool."""
        with self._clients_lock:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        self._executor.shutdown(wait=False)
        logger.info("Distributed client closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Convenience methods for string values

    def put_string(self, key: str, value: str, ttl_ms: int = 0) -> bool:
        """Store a string value."""
        return self.put(key, value.encode('utf-8'), ttl_ms)

    def get_string(self, key: str) -> Optional[str]:
        """Get a string value."""
        value = self.get(key)
        if value is not None:
            return value.decode('utf-8')
        return None

    def put_json(self, key: str, value: Any, ttl_ms: int = 0) -> bool:
        """Store a JSON-serializable value."""
        return self.put(key, json.dumps(value).encode('utf-8'), ttl_ms)

    def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON value."""
        value = self.get(key)
        if value is not None:
            return json.loads(value.decode('utf-8'))
        return None


# Alias for simpler imports
SemanticKVClient = DistributedSemanticKV
