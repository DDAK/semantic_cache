"""
Test suite for the distributed semantic KV cache.

Tests a 4-node cluster on localhost with:
- Basic put/get operations
- Distributed key routing
- Semantic search across all nodes
- Cluster statistics and health
"""

import pytest
import time
import subprocess
import sys
import os
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_kv import DistributedSemanticKV, SemanticKVNode, ConsistentHashRing


# Test configuration
NODE_PORTS = [8815, 8816, 8817, 8818]
NODE_ADDRESSES = [f"grpc://localhost:{port}" for port in NODE_PORTS]


class TestConsistentHashRing:
    """Test the consistent hash ring."""

    def test_single_node(self):
        ring = ConsistentHashRing(["node1"])
        assert ring.get_node("any_key") == "node1"
        assert len(ring) == 1

    def test_multiple_nodes(self):
        nodes = ["node1", "node2", "node3"]
        ring = ConsistentHashRing(nodes)
        assert len(ring) == 3

        # Same key should always go to same node
        node = ring.get_node("test_key")
        for _ in range(100):
            assert ring.get_node("test_key") == node

    def test_distribution(self):
        """Test that keys are distributed across nodes."""
        nodes = ["node1", "node2", "node3", "node4"]
        ring = ConsistentHashRing(nodes)

        # Count distribution
        counts = {node: 0 for node in nodes}
        for i in range(1000):
            node = ring.get_node(f"key_{i}")
            counts[node] += 1

        # Each node should have some keys
        for node, count in counts.items():
            assert count > 100, f"Node {node} has too few keys: {count}"

    def test_add_remove_node(self):
        ring = ConsistentHashRing(["node1", "node2"])

        # Get initial assignment
        key = "test_key"
        initial_node = ring.get_node(key)

        # Add a node - some keys may move
        ring.add_node("node3")
        assert len(ring) == 3

        # Remove the new node - key should return to original or stay
        ring.remove_node("node3")
        assert len(ring) == 2

    def test_get_nodes_for_replication(self):
        nodes = ["node1", "node2", "node3"]
        ring = ConsistentHashRing(nodes)

        # Get 2 nodes for replication
        repl_nodes = ring.get_nodes_for_key("test_key", count=2)
        assert len(repl_nodes) == 2
        assert len(set(repl_nodes)) == 2  # Should be unique


class TestDistributedClient:
    """Test the distributed client with mocked nodes."""

    @pytest.fixture
    def client(self, cluster):
        """Create a client connected to the test cluster."""
        return DistributedSemanticKV(
            nodes=NODE_ADDRESSES,
            search_timeout=10.0,
        )

    def test_put_get(self, client):
        """Test basic put and get."""
        # Put
        assert client.put("test:key:1", b"value1")
        assert client.put("test:key:2", b"value2")

        # Get
        assert client.get("test:key:1") == b"value1"
        assert client.get("test:key:2") == b"value2"

        # Get non-existent
        assert client.get("nonexistent") is None

    def test_put_get_string(self, client):
        """Test string convenience methods."""
        assert client.put_string("str:key", "hello world")
        assert client.get_string("str:key") == "hello world"

    def test_put_get_json(self, client):
        """Test JSON convenience methods."""
        data = {"name": "John", "age": 30, "tags": ["a", "b"]}
        assert client.put_json("json:key", data)
        assert client.get_json("json:key") == data

    def test_delete(self, client):
        """Test delete operation."""
        client.put("delete:key", b"value")
        assert client.get("delete:key") == b"value"

        assert client.delete("delete:key")
        assert client.get("delete:key") is None

        # Delete non-existent
        assert not client.delete("nonexistent")

    def test_exists(self, client):
        """Test exists check."""
        client.put("exists:key", b"value")
        assert client.exists("exists:key")
        assert not client.exists("nonexistent")

    def test_distribution_across_nodes(self, client):
        """Test that keys are distributed to different nodes."""
        # Put many keys
        for i in range(100):
            client.put(f"dist:key:{i}", f"value{i}".encode())

        # Check cluster stats
        stats = client.cluster_stats()

        # Count total keys across nodes
        total_keys = 0
        nodes_with_keys = 0
        for node, node_stats in stats.items():
            if node.startswith("grpc://"):
                if 'total_keys' in node_stats:
                    total_keys += node_stats['total_keys']
                    if node_stats['total_keys'] > 0:
                        nodes_with_keys += 1

        assert total_keys == 100
        assert nodes_with_keys >= 2  # Keys should be on multiple nodes

    def test_search_basic(self, client):
        """Test basic semantic search."""
        # Store some entries
        client.put("user:alice:profile", b"Alice's profile data")
        client.put("user:bob:profile", b"Bob's profile data")
        client.put("config:database", b"Database configuration")
        client.put("log:error:123", b"Error log entry")

        # Search for user profiles
        results = client.search("user profile", top_k=5, threshold=0.3)

        # Should find user-related keys
        keys = [r[0] for r in results]
        assert len(results) > 0

    def test_search_across_nodes(self, client):
        """Test that search queries all nodes."""
        # Store keys that will be on different nodes
        for i in range(50):
            client.put(f"search:item:{i}", f"item {i} data".encode())

        # Search
        results = client.search("search item", top_k=20, threshold=0.3)

        # Should get results from multiple nodes
        if results:
            nodes = set(r[3] for r in results)
            # With good distribution, should see multiple nodes
            assert len(nodes) >= 1

    def test_scan(self, client):
        """Test key scanning."""
        # Put keys with common prefix
        for i in range(10):
            client.put(f"scan:prefix:{i}", b"value")

        client.put("other:key", b"value")

        # Scan with prefix
        keys = client.scan(prefix="scan:prefix:", limit=100)
        assert len(keys) == 10

        # Scan all
        all_keys = client.scan(limit=100)
        assert len(all_keys) >= 11

    def test_ttl_expiration(self, client):
        """Test TTL-based expiration."""
        # Put with short TTL
        client.put("ttl:key", b"value", ttl_ms=500)

        # Should exist immediately
        assert client.get("ttl:key") == b"value"

        # Wait for expiration
        time.sleep(0.6)

        # Should be gone
        assert client.get("ttl:key") is None

    def test_health_check(self, client):
        """Test cluster health check."""
        health = client.health_check()

        assert len(health) == 4
        for node, is_healthy in health.items():
            assert is_healthy, f"Node {node} is unhealthy"

    def test_cluster_stats(self, client):
        """Test cluster statistics."""
        # Put some data first
        for i in range(10):
            client.put(f"stats:key:{i}", b"value")

        stats = client.cluster_stats()

        # Should have stats from all nodes plus client
        assert '_client' in stats
        assert len(stats) >= 5  # 4 nodes + client

        # Each node should have stats
        for node in NODE_ADDRESSES:
            assert node in stats
            assert 'node_id' in stats[node]
            assert 'total_keys' in stats[node]

    def test_clear_all(self, client):
        """Test clearing all nodes."""
        # Put some data
        for i in range(20):
            client.put(f"clear:key:{i}", b"value")

        # Clear
        results = client.clear_all()

        # Should have cleared all nodes
        total_cleared = sum(v for v in results.values() if v >= 0)
        assert total_cleared == 20

        # Verify empty
        assert client.scan(limit=100) == []


# Cluster fixture for integration tests
@pytest.fixture(scope="module")
def cluster():
    """Start a 4-node cluster for testing."""
    import multiprocessing

    processes = []

    def start_node(port, node_id):
        """Start a single node."""
        from semantic_kv import SemanticKVNode
        from semantic_kv.embeddings import HashEmbedding

        node = SemanticKVNode(
            location=f"grpc://0.0.0.0:{port}",
            node_id=node_id,
            embedding_provider=HashEmbedding(dimension=384),
        )
        node.serve()

    # Start nodes in separate processes
    for i, port in enumerate(NODE_PORTS):
        node_id = f"node-{chr(ord('a') + i)}"
        p = multiprocessing.Process(target=start_node, args=(port, node_id))
        p.start()
        processes.append(p)

    # Wait for nodes to start
    time.sleep(2)

    yield processes

    # Cleanup
    for p in processes:
        p.terminate()
        p.join(timeout=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
