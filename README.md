<p align="center">
  <h1 align="center">Semantic KV</h1>
  <p align="center">
    <strong>A blazing-fast distributed key-value store with semantic search superpowers</strong>
  </p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="#performance"><img src="https://img.shields.io/badge/GET_latency-0.14ms-brightgreen.svg" alt="GET Latency"></a>
    <a href="#performance"><img src="https://img.shields.io/badge/throughput-7.2k_ops/sec-orange.svg" alt="Throughput"></a>
  </p>
</p>

---

**Semantic KV** combines the simplicity of a key-value store with the power of semantic search. Built on Apache Arrow Flight for zero-copy data transfer and HNSW for lightning-fast vector similarity search.

```python
# Store data like a normal KV store
kv.put("user:alice", '{"role": "engineer", "skills": ["python", "rust"]}')
kv.put("user:bob", '{"role": "designer", "skills": ["figma", "css"]}')

# But search by meaning, not just keys
results = kv.search("backend developer")  # Finds alice!
```

## Why Semantic KV?

| Traditional KV Stores | Semantic KV |
|----------------------|-------------|
| Exact key match only | Find by meaning |
| `get("user:123")` | `search("experienced python developer")` |
| Manual indexing | Automatic embeddings |
| Single node | Distributed & scalable |

**Use Cases:**
- **Semantic Caching** - Cache LLM responses, retrieve by similar queries
- **User/Content Matching** - Find similar profiles, documents, products
- **Log Analysis** - Search error logs by description, not keywords
- **Feature Stores** - ML feature retrieval by semantic similarity

## Performance

Benchmarked on a 4-node cluster (localhost):

| Operation | Throughput | Avg Latency | P99 Latency |
|-----------|-----------|-------------|-------------|
| **GET** | 7,195 ops/sec | 0.14 ms | 0.18 ms |
| **PUT** | 3,579 ops/sec | 0.28 ms | 0.43 ms |
| **DELETE** | 7,100 ops/sec | 0.14 ms | 0.20 ms |
| **SEARCH** (1K keys) | 344 ops/sec | 2.9 ms | 3.4 ms |
| **SEARCH** (10K keys) | 46 ops/sec | 21.6 ms | 30.5 ms |

**Concurrent Performance:** 3,726 ops/sec sustained (8 threads)

## Features

- **Distributed Architecture** - Consistent hashing shards data across nodes
- **Semantic Search** - HNSW-powered similarity search across all nodes
- **Sub-millisecond Lookups** - Apache Arrow Flight zero-copy transport
- **TTL Support** - Automatic key expiration
- **Fault Tolerant** - Async replication, health checks, retry logic
- **Multiple Embedding Backends** - SentenceTransformers, OpenAI, or custom
- **Persistence** - Arrow IPC snapshots with configurable intervals

## Installation

```bash
pip install pyarrow numpy hnswlib

# For real semantic search (recommended):
pip install sentence-transformers
```

## Quick Start

### 1. Start a Cluster

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/semantic_cache.git
cd semantic_cache

# Start 4-node cluster
python tests/run_cluster.py
```

### 2. Connect and Use

```python
from semantic_kv import DistributedSemanticKV

# Connect to cluster
kv = DistributedSemanticKV([
    "grpc://localhost:8815",
    "grpc://localhost:8816",
    "grpc://localhost:8817",
    "grpc://localhost:8818",
])

# Store data (routed to responsible node via consistent hashing)
kv.put("doc:readme", b"How to install and configure the system")
kv.put("doc:api", b"REST API endpoints and authentication")
kv.put("doc:deploy", b"Kubernetes deployment guide")

# Exact lookup - O(1)
value = kv.get("doc:readme")

# Semantic search - queries ALL nodes, merges results
results = kv.search("how to set up authentication", top_k=3)
for key, value, similarity, node in results:
    print(f"{similarity:.2f} | {key}: {value[:50]}...")

# Output:
# 0.89 | doc:api: REST API endpoints and authentication...
# 0.72 | doc:readme: How to install and configure the sys...
```

### 3. Convenience Methods

```python
# String values
kv.put_string("config:host", "localhost")
host = kv.get_string("config:host")

# JSON values
kv.put_json("metrics:cpu", {"value": 45.2, "unit": "percent"})
metrics = kv.get_json("metrics:cpu")

# Key operations
exists = kv.exists("config:host")        # True
keys = kv.scan(prefix="config:", limit=100)  # ["config:host", ...]
kv.delete("config:host")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DistributedSemanticKV Client                  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Consistent Hash │  │  Thread Pool    │  │  Retry Logic    │  │
│  │     Ring        │  │  (parallel I/O) │  │  + Metrics      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│    Node A     │      │    Node B     │      │    Node C     │
│    :8815      │      │    :8816      │      │    :8817      │
│               │      │               │      │               │
│ ┌───────────┐ │      │ ┌───────────┐ │      │ ┌───────────┐ │
│ │ KV Store  │ │      │ │ KV Store  │ │      │ │ KV Store  │ │
│ │ (Arrow)   │ │      │ │ (Arrow)   │ │      │ │ (Arrow)   │ │
│ └───────────┘ │      │ └───────────┘ │      │ └───────────┘ │
│ ┌───────────┐ │      │ ┌───────────┐ │      │ ┌───────────┐ │
│ │HNSW Index │ │      │ │HNSW Index │ │      │ │HNSW Index │ │
│ └───────────┘ │      │ └───────────┘ │      │ └───────────┘ │
└───────────────┘      └───────────────┘      └───────────────┘
```

**How it works:**

| Operation | Routing Strategy |
|-----------|-----------------|
| `put(key)` | Hash key → route to single responsible node |
| `get(key)` | Hash key → route to single responsible node |
| `search(query)` | Scatter to ALL nodes → gather & merge results |

## Configuration

### Node Setup

```python
from semantic_kv import SemanticKVNode
from semantic_kv.embeddings import SentenceTransformerEmbedding

node = SemanticKVNode(
    location="grpc://0.0.0.0:8815",
    node_id="node-a",
    embedding_provider=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    max_entries=100_000,           # Max vectors in index
    hnsw_ef_construction=200,      # Index build quality
    hnsw_m=16,                     # Connections per node
    data_dir="./data/node-a",      # Persistence directory
)
node.serve()
```

### Client Setup

```python
kv = DistributedSemanticKV(
    nodes=["grpc://node1:8815", "grpc://node2:8816", "grpc://node3:8817"],
    replication_factor=2,          # Write to N nodes
    search_timeout=5.0,            # Scatter-gather timeout
    connection_timeout=2.0,        # Per-connection timeout
    max_workers=8,                 # Thread pool size
)
```

### Embedding Providers

| Provider | Dimensions | Best For |
|----------|-----------|----------|
| `HashEmbedding` | 384 | Testing, development |
| `SentenceTransformerEmbedding` | 384-768 | Production (local, fast) |
| `OpenAIEmbedding` | 1536+ | Highest quality (API costs) |

```python
from semantic_kv.embeddings import (
    HashEmbedding,                    # Testing
    SentenceTransformerEmbedding,     # Recommended
    OpenAIEmbedding,                  # Premium
)

# Local ML embeddings (recommended)
embedding = SentenceTransformerEmbedding("all-MiniLM-L6-v2")

# OpenAI embeddings
embedding = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
```

## API Reference

### Core Operations

| Method | Description | Returns |
|--------|-------------|---------|
| `put(key, value, ttl_ms=0)` | Store key-value pair | `bool` |
| `get(key)` | Get value by exact key | `bytes \| None` |
| `delete(key)` | Delete key | `bool` |
| `exists(key)` | Check if key exists | `bool` |
| `scan(prefix, limit)` | List keys by prefix | `List[str]` |
| `search(query, top_k, threshold)` | Semantic similarity search | `List[Tuple]` |

### Cluster Operations

| Method | Description |
|--------|-------------|
| `health_check()` | Check all node health |
| `cluster_stats()` | Get cluster-wide statistics |
| `clear_all()` | Clear all data from all nodes |
| `close()` | Close all connections |

## Testing

```bash
# Run the demo
python tests/demo.py

# Run tests
pytest tests/ -v

# Run benchmarks
python tests/benchmark.py
```

## Roadmap

- [ ] Kubernetes operator for easy deployment
- [ ] Web UI for cluster monitoring
- [ ] Redis protocol compatibility layer
- [ ] Automatic rebalancing on node add/remove
- [ ] GPU-accelerated similarity search

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>If you find this project useful, please consider giving it a star!</strong>
  <br>
  <sub>Built with Apache Arrow Flight and HNSW</sub>
</p>
