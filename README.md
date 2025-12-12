# Semantic KV Cache

A distributed key-value store with semantic search capabilities, built on Apache Arrow Flight.

## Features

- **Distributed**: Data sharded across multiple nodes using consistent hashing
- **Semantic Search**: Find values by meaning, not just exact keys
- **High Performance**: Arrow Flight for zero-copy data transfer
- **Simple API**: Standard KV operations (put/get/delete/scan)
- **TTL Support**: Automatic key expiration
- **Fault Tolerant**: Async replication to backup nodes

## Installation

```bash
pip install -r requirements.txt

# For real semantic search (recommended):
pip install sentence-transformers
```

## Quick Start

### 1. Start a Cluster

```bash
# Start 4 nodes on localhost
python tests/run_cluster.py
```

### 2. Use the Client

```python
from semantic_kv import DistributedSemanticKV

# Connect to cluster
kv = DistributedSemanticKV([
    "grpc://localhost:8815",
    "grpc://localhost:8816",
    "grpc://localhost:8817",
    "grpc://localhost:8818",
])

# Store data
kv.put("user:123:profile", b'{"name": "John"}')
kv.put("config:database:host", b"localhost")

# Exact lookup
value = kv.get("user:123:profile")

# Semantic search - find related keys
results = kv.search("user profile information", top_k=5)
for key, value, similarity, node in results:
    print(f"{key}: {value} (score: {similarity:.2f})")

# Scan keys
keys = kv.scan(prefix="user:", limit=100)
```

### 3. Run Demo

```bash
# Start cluster in one terminal
python tests/run_cluster.py

# Run demo in another terminal
python tests/demo.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│               DistributedSemanticKV Client               │
│                                                          │
│   put(key) ──► hash(key) ──► route to responsible node  │
│   get(key) ──► hash(key) ──► route to responsible node  │
│   search() ──► scatter to ALL nodes ──► merge results   │
└──────────────────────────┬──────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Node A     │   │   Node B     │   │   Node C     │
│   :8815      │   │   :8816      │   │   :8817      │
│              │   │              │   │              │
│ KV Store     │   │ KV Store     │   │ KV Store     │
│ HNSW Index   │   │ HNSW Index   │   │ HNSW Index   │
└──────────────┘   └──────────────┘   └──────────────┘
```

## API Reference

### Client Methods

| Method | Description |
|--------|-------------|
| `put(key, value, ttl_ms=0)` | Store key-value pair |
| `get(key)` | Get value by exact key |
| `search(query, top_k, threshold)` | Semantic similarity search |
| `delete(key)` | Delete key |
| `scan(prefix, limit)` | List keys by prefix |
| `exists(key)` | Check if key exists |
| `health_check()` | Check all nodes |
| `cluster_stats()` | Get cluster statistics |
| `clear_all()` | Clear all data |

### Convenience Methods

| Method | Description |
|--------|-------------|
| `put_string(key, value)` | Store string value |
| `get_string(key)` | Get string value |
| `put_json(key, value)` | Store JSON value |
| `get_json(key)` | Get JSON value |

## Configuration

### Node Configuration

```python
from semantic_kv import SemanticKVNode
from semantic_kv.embeddings import SentenceTransformerEmbedding

node = SemanticKVNode(
    location="grpc://0.0.0.0:8815",
    node_id="node-a",
    embedding_provider=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    max_entries=100000,
)
node.serve()
```

### Client Configuration

```python
kv = DistributedSemanticKV(
    nodes=["grpc://node1:8815", "grpc://node2:8816"],
    replication_factor=2,
    search_timeout=5.0,
)
```

## Embedding Providers

| Provider | Dimension | Description |
|----------|-----------|-------------|
| `HashEmbedding` | 384 | Deterministic pseudo-embeddings (testing) |
| `SentenceTransformerEmbedding` | 384-768 | Local ML models (recommended) |
| `OpenAIEmbedding` | 1536+ | OpenAI API |

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_cluster.py -v
```

## Performance

| Operation | Latency (4 nodes) |
|-----------|-------------------|
| put | ~10-15ms |
| get | ~5-10ms |
| search (10k keys/node) | ~50-100ms |
| scan | ~20-50ms |

## License

MIT
