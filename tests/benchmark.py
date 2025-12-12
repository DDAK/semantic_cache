#!/usr/bin/env python3
"""
Benchmark suite for the Distributed Semantic KV Cache.

Tests:
1. Single-key operations (put, get, delete)
2. Batch operations
3. Semantic search at various scales
4. Cluster scalability (1, 2, 4 nodes)
5. Concurrent operations
"""

import sys
import time
import threading
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_kv import DistributedSemanticKV, SemanticKVNode
from semantic_kv.embeddings import HashEmbedding


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    operations: int
    total_time_ms: float
    ops_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def benchmark_operation(
    name: str,
    operation,
    iterations: int,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Benchmark a single operation.

    Args:
        name: Name of the benchmark
        operation: Callable that performs the operation
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        operation()

    # Benchmark
    latencies = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    end_total = time.perf_counter()
    total_time_ms = (end_total - start_total) * 1000

    return BenchmarkResult(
        name=name,
        operations=iterations,
        total_time_ms=total_time_ms,
        ops_per_second=iterations / (total_time_ms / 1000),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
    )


def start_cluster(num_nodes: int, base_port: int = 8815) -> Tuple[List, List[str]]:
    """Start a cluster of nodes."""
    nodes = []
    addresses = []

    for i in range(num_nodes):
        port = base_port + i
        node_id = f"node-{chr(ord('a') + i)}"
        address = f"grpc://localhost:{port}"
        addresses.append(address)

        node = SemanticKVNode(
            location=f"grpc://0.0.0.0:{port}",
            node_id=node_id,
            embedding_provider=HashEmbedding(dimension=384),
        )

        thread = threading.Thread(target=node.serve, daemon=True)
        thread.start()
        nodes.append((node, thread))

    # Wait for nodes to start
    time.sleep(1.5)

    return nodes, addresses


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version.split()[0],
        },
        "benchmarks": {}
    }

    print("=" * 60)
    print("Distributed Semantic KV Cache - Benchmark Suite")
    print("=" * 60)

    # Start 4-node cluster
    print("\nStarting 4-node cluster...")
    nodes, addresses = start_cluster(4)
    kv = DistributedSemanticKV(addresses)

    # Verify cluster health
    health = kv.health_check()
    healthy = sum(1 for v in health.values() if v)
    print(f"Cluster ready: {healthy}/{len(health)} nodes healthy\n")

    # ========================================
    # 1. Single PUT operations
    # ========================================
    print("1. Benchmarking PUT operations...")

    key_counter = [0]

    def put_op():
        key_counter[0] += 1
        kv.put(f"bench:put:{key_counter[0]}", b"x" * 100)

    result = benchmark_operation("put_100b", put_op, iterations=1000)
    results["benchmarks"]["put_100b_value"] = result.__dict__
    print(f"   PUT (100B value): {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # PUT with larger values
    def put_1kb():
        key_counter[0] += 1
        kv.put(f"bench:put1k:{key_counter[0]}", b"x" * 1024)

    result = benchmark_operation("put_1kb", put_1kb, iterations=500)
    results["benchmarks"]["put_1kb_value"] = result.__dict__
    print(f"   PUT (1KB value):  {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # ========================================
    # 2. Single GET operations
    # ========================================
    print("\n2. Benchmarking GET operations...")

    # Pre-populate keys for GET test
    for i in range(1000):
        kv.put(f"bench:get:{i}", f"value-{i}".encode())

    get_counter = [0]

    def get_op():
        get_counter[0] = (get_counter[0] + 1) % 1000
        kv.get(f"bench:get:{get_counter[0]}")

    result = benchmark_operation("get_hit", get_op, iterations=2000)
    results["benchmarks"]["get_hit"] = result.__dict__
    print(f"   GET (cache hit): {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # GET miss
    miss_counter = [0]

    def get_miss():
        miss_counter[0] += 1
        kv.get(f"bench:miss:{miss_counter[0]}")

    result = benchmark_operation("get_miss", get_miss, iterations=1000)
    results["benchmarks"]["get_miss"] = result.__dict__
    print(f"   GET (cache miss): {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # ========================================
    # 3. DELETE operations
    # ========================================
    print("\n3. Benchmarking DELETE operations...")

    # Pre-populate keys for DELETE test
    for i in range(500):
        kv.put(f"bench:del:{i}", b"delete-me")

    del_counter = [0]

    def delete_op():
        del_counter[0] += 1
        kv.delete(f"bench:del:{del_counter[0] % 500}")

    result = benchmark_operation("delete", delete_op, iterations=500)
    results["benchmarks"]["delete"] = result.__dict__
    print(f"   DELETE: {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # ========================================
    # 4. SEARCH operations (scatter-gather)
    # ========================================
    print("\n4. Benchmarking SEARCH operations...")

    # Clear and populate with searchable data
    kv.clear_all()
    time.sleep(0.5)

    # Populate different data sizes
    for size_name, num_keys in [("100", 100), ("1k", 1000), ("10k", 10000)]:
        print(f"\n   Populating {num_keys} keys...")
        for i in range(num_keys):
            kv.put(f"search:{size_name}:key:{i}", f"value-{i}".encode())

        time.sleep(0.5)

        def search_op():
            kv.search("search key value", top_k=10, threshold=0.3)

        result = benchmark_operation(f"search_{size_name}", search_op, iterations=100)
        results["benchmarks"][f"search_{size_name}_keys"] = result.__dict__
        print(f"   SEARCH ({num_keys} keys): {result.ops_per_second:.0f} ops/s, "
              f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

        # Clear for next size
        if size_name != "10k":
            kv.clear_all()
            time.sleep(0.3)

    # ========================================
    # 5. SCAN operations
    # ========================================
    print("\n5. Benchmarking SCAN operations...")

    def scan_op():
        kv.scan(prefix="search:", limit=100)

    result = benchmark_operation("scan_100", scan_op, iterations=200)
    results["benchmarks"]["scan_100_limit"] = result.__dict__
    print(f"   SCAN (limit 100): {result.ops_per_second:.0f} ops/s, "
          f"avg={result.avg_latency_ms:.2f}ms, p99={result.p99_latency_ms:.2f}ms")

    # ========================================
    # 6. Concurrent operations
    # ========================================
    print("\n6. Benchmarking CONCURRENT operations...")

    kv.clear_all()
    time.sleep(0.3)

    for num_threads in [4, 8, 16]:
        ops_completed = [0]
        lock = threading.Lock()

        def concurrent_mixed():
            for i in range(100):
                op = i % 3
                if op == 0:
                    kv.put(f"conc:{threading.current_thread().name}:{i}", b"value")
                elif op == 1:
                    kv.get(f"conc:{threading.current_thread().name}:{i-1}")
                else:
                    kv.search("concurrent test", top_k=5, threshold=0.3)

                with lock:
                    ops_completed[0] += 1

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_mixed) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        end = time.perf_counter()
        total_ops = ops_completed[0]
        elapsed = end - start
        ops_per_sec = total_ops / elapsed

        results["benchmarks"][f"concurrent_{num_threads}_threads"] = {
            "threads": num_threads,
            "total_ops": total_ops,
            "elapsed_seconds": elapsed,
            "ops_per_second": ops_per_sec,
        }
        print(f"   {num_threads} threads: {ops_per_sec:.0f} ops/s total "
              f"({total_ops} ops in {elapsed:.2f}s)")

    # ========================================
    # 7. Data distribution
    # ========================================
    print("\n7. Checking data distribution...")

    kv.clear_all()
    time.sleep(0.3)

    # Insert keys and check distribution
    num_keys = 10000
    for i in range(num_keys):
        kv.put(f"dist:{i}", f"v{i}".encode())

    stats = kv.cluster_stats()
    distribution = {}
    total_keys = 0

    for node, node_stats in stats.items():
        if node.startswith("grpc://"):
            node_id = node_stats.get('node_id', node)
            keys = node_stats.get('total_keys', 0)
            distribution[node_id] = keys
            total_keys += keys

    results["benchmarks"]["distribution"] = {
        "total_keys": total_keys,
        "per_node": distribution,
        "std_dev": statistics.stdev(distribution.values()) if len(distribution) > 1 else 0,
    }

    print(f"   Total keys: {total_keys}")
    for node_id, keys in sorted(distribution.items()):
        pct = (keys / total_keys * 100) if total_keys > 0 else 0
        print(f"   {node_id}: {keys} keys ({pct:.1f}%)")

    # ========================================
    # 8. Throughput test
    # ========================================
    print("\n8. Throughput test (sustained load)...")

    kv.clear_all()
    time.sleep(0.3)

    duration = 5.0  # seconds
    ops_count = [0]
    stop_flag = [False]

    def throughput_worker():
        i = 0
        while not stop_flag[0]:
            kv.put(f"tp:{threading.current_thread().name}:{i}", b"x" * 100)
            i += 1
            with lock:
                ops_count[0] += 1

    threads = []
    for _ in range(8):
        t = threading.Thread(target=throughput_worker)
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_flag[0] = True

    for t in threads:
        t.join()

    throughput = ops_count[0] / duration
    results["benchmarks"]["throughput_8_threads"] = {
        "duration_seconds": duration,
        "total_ops": ops_count[0],
        "ops_per_second": throughput,
    }
    print(f"   8 threads, {duration}s: {throughput:.0f} ops/s "
          f"({ops_count[0]} total ops)")

    # Cleanup
    kv.close()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return results


def format_results_markdown(results: Dict[str, Any]) -> str:
    """Format benchmark results as Markdown."""
    md = []
    md.append("# Distributed Semantic KV Cache - Benchmark Results\n")
    md.append(f"**Date:** {results['metadata']['timestamp']}\n")
    md.append(f"**Python:** {results['metadata']['python_version']}\n")
    md.append(f"**Cluster:** 4 nodes on localhost\n")
    md.append(f"**Index:** BruteForce (fallback)\n")
    md.append("")

    md.append("## Summary\n")
    md.append("| Operation | Ops/sec | Avg Latency | P99 Latency |")
    md.append("|-----------|---------|-------------|-------------|")

    benchmarks = results["benchmarks"]

    # Single operations
    for key in ["put_100b_value", "put_1kb_value", "get_hit", "get_miss", "delete"]:
        if key in benchmarks:
            b = benchmarks[key]
            name = key.replace("_", " ").title()
            md.append(f"| {name} | {b['ops_per_second']:.0f} | "
                     f"{b['avg_latency_ms']:.2f}ms | {b['p99_latency_ms']:.2f}ms |")

    # Search operations
    for key in ["search_100_keys", "search_1k_keys", "search_10k_keys"]:
        if key in benchmarks:
            b = benchmarks[key]
            name = key.replace("_", " ").title()
            md.append(f"| {name} | {b['ops_per_second']:.0f} | "
                     f"{b['avg_latency_ms']:.2f}ms | {b['p99_latency_ms']:.2f}ms |")

    # Scan
    if "scan_100_limit" in benchmarks:
        b = benchmarks["scan_100_limit"]
        md.append(f"| Scan (100 limit) | {b['ops_per_second']:.0f} | "
                 f"{b['avg_latency_ms']:.2f}ms | {b['p99_latency_ms']:.2f}ms |")

    md.append("")

    md.append("## Detailed Results\n")

    # PUT operations
    md.append("### PUT Operations\n")
    md.append("| Metric | 100B Value | 1KB Value |")
    md.append("|--------|------------|-----------|")

    put_100 = benchmarks.get("put_100b_value", {})
    put_1k = benchmarks.get("put_1kb_value", {})

    md.append(f"| Operations | {put_100.get('operations', '-')} | {put_1k.get('operations', '-')} |")
    md.append(f"| Ops/sec | {put_100.get('ops_per_second', 0):.0f} | {put_1k.get('ops_per_second', 0):.0f} |")
    md.append(f"| Avg Latency | {put_100.get('avg_latency_ms', 0):.2f}ms | {put_1k.get('avg_latency_ms', 0):.2f}ms |")
    md.append(f"| P50 Latency | {put_100.get('p50_latency_ms', 0):.2f}ms | {put_1k.get('p50_latency_ms', 0):.2f}ms |")
    md.append(f"| P95 Latency | {put_100.get('p95_latency_ms', 0):.2f}ms | {put_1k.get('p95_latency_ms', 0):.2f}ms |")
    md.append(f"| P99 Latency | {put_100.get('p99_latency_ms', 0):.2f}ms | {put_1k.get('p99_latency_ms', 0):.2f}ms |")
    md.append(f"| Min Latency | {put_100.get('min_latency_ms', 0):.2f}ms | {put_1k.get('min_latency_ms', 0):.2f}ms |")
    md.append(f"| Max Latency | {put_100.get('max_latency_ms', 0):.2f}ms | {put_1k.get('max_latency_ms', 0):.2f}ms |")
    md.append("")

    # GET operations
    md.append("### GET Operations\n")
    md.append("| Metric | Cache Hit | Cache Miss |")
    md.append("|--------|-----------|------------|")

    get_hit = benchmarks.get("get_hit", {})
    get_miss = benchmarks.get("get_miss", {})

    md.append(f"| Ops/sec | {get_hit.get('ops_per_second', 0):.0f} | {get_miss.get('ops_per_second', 0):.0f} |")
    md.append(f"| Avg Latency | {get_hit.get('avg_latency_ms', 0):.2f}ms | {get_miss.get('avg_latency_ms', 0):.2f}ms |")
    md.append(f"| P99 Latency | {get_hit.get('p99_latency_ms', 0):.2f}ms | {get_miss.get('p99_latency_ms', 0):.2f}ms |")
    md.append("")

    # Search operations
    md.append("### SEARCH Operations (Scatter-Gather)\n")
    md.append("| Dataset Size | Ops/sec | Avg Latency | P99 Latency |")
    md.append("|--------------|---------|-------------|-------------|")

    for key, label in [("search_100_keys", "100 keys"), ("search_1k_keys", "1,000 keys"), ("search_10k_keys", "10,000 keys")]:
        if key in benchmarks:
            b = benchmarks[key]
            md.append(f"| {label} | {b['ops_per_second']:.0f} | "
                     f"{b['avg_latency_ms']:.2f}ms | {b['p99_latency_ms']:.2f}ms |")
    md.append("")

    # Concurrent operations
    md.append("### Concurrent Operations\n")
    md.append("| Threads | Total Ops | Elapsed | Ops/sec |")
    md.append("|---------|-----------|---------|---------|")

    for threads in [4, 8, 16]:
        key = f"concurrent_{threads}_threads"
        if key in benchmarks:
            b = benchmarks[key]
            md.append(f"| {threads} | {b['total_ops']} | {b['elapsed_seconds']:.2f}s | {b['ops_per_second']:.0f} |")
    md.append("")

    # Throughput
    md.append("### Sustained Throughput\n")
    if "throughput_8_threads" in benchmarks:
        tp = benchmarks["throughput_8_threads"]
        md.append(f"- **8 threads, {tp['duration_seconds']}s sustained load**")
        md.append(f"- Total operations: {tp['total_ops']:,}")
        md.append(f"- Throughput: **{tp['ops_per_second']:,.0f} ops/sec**")
    md.append("")

    # Distribution
    md.append("### Data Distribution (10,000 keys)\n")
    if "distribution" in benchmarks:
        dist = benchmarks["distribution"]
        md.append(f"- Total keys: {dist['total_keys']:,}")
        md.append(f"- Standard deviation: {dist['std_dev']:.1f}")
        md.append("")
        md.append("| Node | Keys | Percentage |")
        md.append("|------|------|------------|")
        total = dist['total_keys']
        for node, keys in sorted(dist['per_node'].items()):
            pct = (keys / total * 100) if total > 0 else 0
            md.append(f"| {node} | {keys:,} | {pct:.1f}% |")
    md.append("")

    # Configuration
    md.append("## Test Configuration\n")
    md.append("- **Nodes:** 4 (localhost ports 8815-8818)")
    md.append("- **Embedding:** HashEmbedding (384 dimensions)")
    md.append("- **Index:** BruteForceIndex (hnswlib fallback)")
    md.append("- **Replication:** Disabled")
    md.append("- **Value sizes:** 100B (default), 1KB (large)")
    md.append("")

    md.append("## Notes\n")
    md.append("- All benchmarks run on localhost (no network latency)")
    md.append("- BruteForce index used due to hnswlib compatibility issues")
    md.append("- Search results may differ with real embeddings (sentence-transformers)")
    md.append("- Concurrent tests use mixed workload (PUT/GET/SEARCH)")
    md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    results = run_benchmarks()

    # Save results as JSON
    json_path = Path(__file__).parent.parent / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save results as Markdown
    md_path = Path(__file__).parent.parent / "benchmark.md"
    md_content = format_results_markdown(results)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown results saved to: {md_path}")
