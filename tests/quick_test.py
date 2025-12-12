#!/usr/bin/env python3
"""
Quick integration test that starts nodes in-process using threads.
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_kv import DistributedSemanticKV, SemanticKVNode
from semantic_kv.embeddings import HashEmbedding


def start_node_thread(port, node_id):
    """Start a node in a thread."""
    node = SemanticKVNode(
        location=f"grpc://0.0.0.0:{port}",
        node_id=node_id,
        embedding_provider=HashEmbedding(dimension=384),
    )
    # Run in thread (non-blocking)
    thread = threading.Thread(target=node.serve, daemon=True)
    thread.start()
    return node, thread


def main():
    print("=" * 50)
    print("Quick Integration Test")
    print("=" * 50)

    # Start 4 nodes
    print("\n1. Starting 4 nodes...")
    nodes = []
    ports = [8815, 8816, 8817, 8818]

    for i, port in enumerate(ports):
        node_id = f"node-{chr(ord('a') + i)}"
        node, thread = start_node_thread(port, node_id)
        nodes.append((node, thread))
        print(f"   Started {node_id} on port {port}")

    # Wait for nodes to start
    time.sleep(2)

    # Create client
    print("\n2. Connecting client...")
    kv = DistributedSemanticKV([
        f"grpc://localhost:{port}" for port in ports
    ])

    # Health check
    print("\n3. Health check...")
    health = kv.health_check()
    healthy_count = sum(1 for v in health.values() if v)
    print(f"   {healthy_count}/{len(health)} nodes healthy")

    if healthy_count == 0:
        print("   ERROR: No healthy nodes!")
        return

    # Put/Get test
    print("\n4. Put/Get test...")
    test_data = [
        ("user:alice", b"Alice's data"),
        ("user:bob", b"Bob's data"),
        ("config:db:host", b"localhost"),
        ("config:db:port", b"5432"),
        ("log:error:1", b"Error message"),
    ]

    for key, value in test_data:
        success = kv.put(key, value)
        print(f"   PUT {key}: {'✓' if success else '✗'}")

    print()
    for key, expected in test_data:
        actual = kv.get(key)
        match = actual == expected
        print(f"   GET {key}: {'✓' if match else '✗'}")

    # Semantic search test
    print("\n5. Semantic search test...")
    searches = ["user data", "database config", "error log"]

    for query in searches:
        results = kv.search(query, top_k=3, threshold=0.3)
        print(f"   Search '{query}': {len(results)} results")
        for key, value, score, node in results[:2]:
            print(f"      - {key} (score: {score:.2f}, node: {node})")

    # Distribution test
    print("\n6. Distribution test...")
    for i in range(100):
        kv.put(f"dist:key:{i}", f"value{i}".encode())

    stats = kv.cluster_stats()
    print("   Keys per node:")
    for node, node_stats in stats.items():
        if node.startswith("grpc://"):
            node_id = node_stats.get('node_id', node)
            keys = node_stats.get('total_keys', 0)
            print(f"      {node_id}: {keys} keys")

    # Cleanup
    print("\n7. Cleanup...")
    cleared = kv.clear_all()
    total = sum(v for v in cleared.values() if v >= 0)
    print(f"   Cleared {total} keys")

    kv.close()

    print("\n" + "=" * 50)
    print("✓ All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
