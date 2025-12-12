#!/usr/bin/env python3
"""
Script to run a 4-node semantic KV cluster for testing.

Usage:
    python run_cluster.py              # Start 4 nodes
    python run_cluster.py --nodes 2    # Start 2 nodes
    python run_cluster.py --embedding sentence-transformer  # Use real embeddings
"""

import argparse
import multiprocessing
import signal
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_node(port: int, node_id: str, embedding_provider: str, embedding_model: str):
    """Start a single node in a subprocess."""
    from semantic_kv import SemanticKVNode
    from semantic_kv.embeddings import create_embedding_provider

    # Create embedding provider
    if embedding_provider == "hash":
        provider = create_embedding_provider("hash", dimension=384)
    elif embedding_provider == "sentence-transformer":
        provider = create_embedding_provider(
            "sentence-transformer",
            model=embedding_model
        )
    else:
        provider = create_embedding_provider("hash", dimension=384)

    location = f"grpc://0.0.0.0:{port}"

    logger.info(f"Starting node {node_id} on {location}")

    node = SemanticKVNode(
        location=location,
        node_id=node_id,
        embedding_provider=provider,
    )

    # Handle shutdown gracefully
    def shutdown(signum, frame):
        logger.info(f"Shutting down node {node_id}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    node.serve()


def main():
    parser = argparse.ArgumentParser(description="Run a semantic KV cluster")
    parser.add_argument(
        "--nodes",
        type=int,
        default=4,
        help="Number of nodes to start (default: 4)"
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8815,
        help="Base port number (default: 8815)"
    )
    parser.add_argument(
        "--embedding",
        choices=["hash", "sentence-transformer"],
        default="hash",
        help="Embedding provider (default: hash)"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    processes = []
    ports = []

    print(f"\n{'='*60}")
    print(f"Starting {args.nodes}-node Semantic KV Cluster")
    print(f"{'='*60}\n")

    try:
        # Start nodes
        for i in range(args.nodes):
            port = args.base_port + i
            node_id = f"node-{chr(ord('a') + i)}"
            ports.append(port)

            p = multiprocessing.Process(
                target=start_node,
                args=(port, node_id, args.embedding, args.model),
                name=node_id
            )
            p.start()
            processes.append(p)

            print(f"  Node {node_id}: grpc://localhost:{port}")

        print(f"\n{'='*60}")
        print("Cluster is running. Press Ctrl+C to stop.")
        print(f"{'='*60}\n")

        # Print connection info
        print("Connect with Python:")
        print("  from semantic_kv import DistributedSemanticKV")
        print(f"  kv = DistributedSemanticKV([")
        for port in ports:
            print(f'      "grpc://localhost:{port}",')
        print("  ])")
        print()

        # Wait for processes
        while True:
            time.sleep(1)

            # Check if any process died
            for p in processes:
                if not p.is_alive():
                    logger.error(f"Process {p.name} died unexpectedly")

    except KeyboardInterrupt:
        print("\n\nShutting down cluster...")

    finally:
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Wait for processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

        print("Cluster stopped.")


if __name__ == "__main__":
    main()
