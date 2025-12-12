#!/usr/bin/env python3
"""
Demo script for the distributed semantic KV cache.

This script demonstrates:
1. Connecting to a 4-node cluster
2. Storing key-value pairs
3. Exact key lookups
4. Semantic similarity search
5. Cluster statistics

Prerequisites:
    Start the cluster first:
    python tests/run_cluster.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_kv import DistributedSemanticKV


def main():
    print("\n" + "=" * 60)
    print("Distributed Semantic KV Cache Demo")
    print("=" * 60 + "\n")

    # Connect to cluster
    print("Connecting to 4-node cluster...")
    kv = DistributedSemanticKV([
        "grpc://localhost:8815",
        "grpc://localhost:8816",
        "grpc://localhost:8817",
        "grpc://localhost:8818",
    ])

    # Health check
    print("\n1. Health Check")
    print("-" * 40)
    health = kv.health_check()
    for node, is_healthy in health.items():
        status = "✓ healthy" if is_healthy else "✗ unhealthy"
        print(f"  {node}: {status}")

    # Store some data
    print("\n2. Storing Key-Value Pairs")
    print("-" * 40)

    test_data = [
        ("user:alice:profile", '{"name": "Alice", "role": "admin"}'),
        ("user:bob:profile", '{"name": "Bob", "role": "user"}'),
        ("user:charlie:settings", '{"theme": "dark", "notifications": true}'),
        ("config:database:host", "localhost"),
        ("config:database:port", "5432"),
        ("config:cache:ttl", "3600"),
        ("log:error:2024-01-01", "Connection timeout at startup"),
        ("log:error:2024-01-02", "Database query failed"),
        ("log:info:2024-01-01", "Service started successfully"),
        ("metric:cpu:avg", "45.2"),
        ("metric:memory:used", "2048MB"),
    ]

    for key, value in test_data:
        kv.put_string(key, value)
        print(f"  PUT {key}")

    # Exact lookups
    print("\n3. Exact Key Lookups")
    print("-" * 40)

    lookup_keys = ["user:alice:profile", "config:database:host", "nonexistent:key"]
    for key in lookup_keys:
        value = kv.get_string(key)
        if value:
            print(f"  GET {key} → {value[:50]}...")
        else:
            print(f"  GET {key} → (not found)")

    # Semantic search
    print("\n4. Semantic Search")
    print("-" * 40)

    search_queries = [
        "user profile information",
        "database configuration",
        "error logs",
        "system metrics",
    ]

    for query in search_queries:
        print(f"\n  Search: '{query}'")
        results = kv.search(query, top_k=3, threshold=0.3)

        if results:
            for key, value, similarity, node in results:
                value_str = value.decode('utf-8')[:40]
                print(f"    → {key} (score: {similarity:.2f}, node: {node})")
                print(f"      value: {value_str}...")
        else:
            print("    → (no results)")

    # Key scanning
    print("\n5. Key Scanning")
    print("-" * 40)

    print("  All keys with 'user:' prefix:")
    user_keys = kv.scan(prefix="user:", limit=10)
    for key in user_keys:
        print(f"    - {key}")

    print("\n  All keys with 'config:' prefix:")
    config_keys = kv.scan(prefix="config:", limit=10)
    for key in config_keys:
        print(f"    - {key}")

    # Cluster statistics
    print("\n6. Cluster Statistics")
    print("-" * 40)

    stats = kv.cluster_stats()

    total_keys = 0
    for node, node_stats in stats.items():
        if node.startswith("grpc://"):
            print(f"\n  {node_stats.get('node_id', node)}:")
            print(f"    Keys: {node_stats.get('total_keys', 0)}")
            print(f"    Index size: {node_stats.get('index_size', 0)}")
            print(f"    Puts: {node_stats.get('metrics', {}).get('puts', 0)}")
            print(f"    Gets: {node_stats.get('metrics', {}).get('gets', 0)}")
            print(f"    Searches: {node_stats.get('metrics', {}).get('searches', 0)}")
            total_keys += node_stats.get('total_keys', 0)

    print(f"\n  Total keys across cluster: {total_keys}")

    # TTL demo
    print("\n7. TTL Expiration Demo")
    print("-" * 40)

    print("  Storing key with 2-second TTL...")
    kv.put("ttl:demo:key", b"This will expire!", ttl_ms=2000)

    print("  Immediate read:", kv.get_string("ttl:demo:key"))

    print("  Waiting 2.5 seconds...")
    time.sleep(2.5)

    result = kv.get("ttl:demo:key")
    print("  After TTL:", "expired (None)" if result is None else result)

    # Cleanup
    print("\n8. Cleanup")
    print("-" * 40)

    print("  Clearing all data...")
    clear_results = kv.clear_all()
    total_cleared = sum(v for v in clear_results.values() if v >= 0)
    print(f"  Cleared {total_cleared} entries")

    # Close connection
    kv.close()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
