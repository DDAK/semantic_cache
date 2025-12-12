"""
Consistent Hash Ring for distributed key routing.

Uses virtual nodes to ensure even distribution across physical nodes.
"""

import hashlib
from bisect import bisect_right
from typing import List, Tuple, Set


class ConsistentHashRing:
    """
    Consistent hashing implementation for distributing keys across nodes.

    Uses MD5 hashing with virtual nodes to ensure even distribution.
    When a node is added/removed, only ~1/N keys need to be remapped.
    """

    def __init__(self, nodes: List[str] = None, virtual_nodes: int = 150):
        """
        Initialize the hash ring.

        Args:
            nodes: List of node addresses (e.g., ["grpc://node1:8815"])
            virtual_nodes: Number of virtual nodes per physical node.
                          Higher = more even distribution, more memory.
        """
        self._virtual_nodes = virtual_nodes
        self._ring: List[Tuple[int, str]] = []
        self._nodes: Set[str] = set()
        self._hash_values: List[int] = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """Generate hash value for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        """
        Add a node to the ring.

        Args:
            node: Node address (e.g., "grpc://localhost:8815")
        """
        if node in self._nodes:
            return

        self._nodes.add(node)

        # Add virtual nodes
        for i in range(self._virtual_nodes):
            virtual_key = f"{node}:vnode:{i}"
            hash_val = self._hash(virtual_key)
            self._ring.append((hash_val, node))

        # Re-sort the ring
        self._ring.sort(key=lambda x: x[0])
        self._hash_values = [h for h, _ in self._ring]

    def remove_node(self, node: str) -> None:
        """
        Remove a node from the ring.

        Args:
            node: Node address to remove
        """
        if node not in self._nodes:
            return

        self._nodes.discard(node)
        self._ring = [(h, n) for h, n in self._ring if n != node]
        self._hash_values = [h for h, _ in self._ring]

    def get_node(self, key: str) -> str:
        """
        Get the node responsible for a key.

        Args:
            key: The key to look up

        Returns:
            Node address responsible for this key

        Raises:
            ValueError: If ring is empty
        """
        if not self._ring:
            raise ValueError("Hash ring is empty - no nodes available")

        hash_val = self._hash(key)

        # Find first node with hash >= key hash (binary search)
        idx = bisect_right(self._hash_values, hash_val)

        # Wrap around to first node if past the end
        if idx >= len(self._ring):
            idx = 0

        return self._ring[idx][1]

    def get_nodes_for_key(self, key: str, count: int = 1) -> List[str]:
        """
        Get multiple nodes for a key (for replication).

        Args:
            key: The key to look up
            count: Number of distinct nodes to return

        Returns:
            List of node addresses (up to count unique nodes)
        """
        if not self._ring:
            raise ValueError("Hash ring is empty - no nodes available")

        if count > len(self._nodes):
            count = len(self._nodes)

        hash_val = self._hash(key)
        idx = bisect_right(self._hash_values, hash_val)

        nodes = []
        seen = set()

        # Walk around the ring collecting unique nodes
        for i in range(len(self._ring)):
            ring_idx = (idx + i) % len(self._ring)
            node = self._ring[ring_idx][1]

            if node not in seen:
                nodes.append(node)
                seen.add(node)

                if len(nodes) >= count:
                    break

        return nodes

    def get_all_nodes(self) -> List[str]:
        """Get all unique physical nodes."""
        return list(self._nodes)

    def __len__(self) -> int:
        """Number of physical nodes in the ring."""
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        """Check if a node is in the ring."""
        return node in self._nodes
