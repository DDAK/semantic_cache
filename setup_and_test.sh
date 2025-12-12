#!/bin/bash
# Setup and test script for Distributed Semantic KV Cache

set -e

echo "=============================================="
echo "Distributed Semantic KV Cache - Setup & Test"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install pyarrow hnswlib numpy pytest -q

echo ""
echo "✓ Dependencies installed"

# Test imports
echo ""
echo "Testing imports..."
python -c "
from semantic_kv import DistributedSemanticKV, SemanticKVNode, ConsistentHashRing
from semantic_kv.embeddings import HashEmbedding
print('✓ All imports successful')
"

# Run hash ring test
echo ""
echo "Testing hash ring..."
python -c "
from semantic_kv.hash_ring import ConsistentHashRing
ring = ConsistentHashRing(['node1', 'node2', 'node3', 'node4'])
counts = {}
for i in range(1000):
    node = ring.get_node(f'key_{i}')
    counts[node] = counts.get(node, 0) + 1
print(f'✓ Hash ring distribution: {counts}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To start a 4-node cluster:"
echo "  source venv/bin/activate"
echo "  python tests/run_cluster.py"
echo ""
echo "Then in another terminal, run the demo:"
echo "  source venv/bin/activate"
echo "  python tests/demo.py"
echo ""
echo "Or run the test suite:"
echo "  source venv/bin/activate"
echo "  pytest tests/test_cluster.py -v"
echo ""
