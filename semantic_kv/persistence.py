"""
Persistence layer for semantic KV cache using Apache Arrow.

Provides:
- Arrow IPC file storage for KV data
- HNSW index persistence
- Snapshot and recovery
"""

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import json
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


# Arrow schema for persisted KV entries
KV_SCHEMA = pa.schema([
    ('key', pa.string()),
    ('value', pa.large_binary()),
    ('embedding', pa.list_(pa.float32())),
    ('created_at', pa.int64()),
    ('ttl_ms', pa.int64()),
    ('access_count', pa.int64()),
    ('last_accessed', pa.int64()),
])


@dataclass
class PersistedEntry:
    """Entry loaded from persistence."""
    key: str
    value: bytes
    embedding: np.ndarray
    created_at: int
    ttl_ms: int
    access_count: int
    last_accessed: int


class ArrowKVStore:
    """
    Arrow-based persistent storage for KV entries.

    Stores data in Arrow IPC format for efficient serialization
    and compatibility with Arrow Flight.
    """

    def __init__(self, data_dir: str, node_id: str = "node"):
        """
        Initialize the Arrow KV store.

        Args:
            data_dir: Directory for storing data files
            node_id: Node identifier for file naming
        """
        self._data_dir = Path(data_dir)
        self._node_id = node_id
        self._data_file = self._data_dir / f"{node_id}_data.arrow"
        self._index_file = self._data_dir / f"{node_id}_index.hnsw"
        self._meta_file = self._data_dir / f"{node_id}_meta.json"

        # Ensure directory exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        entries: Dict[str, Any],
        index: Any = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save entries and optionally HNSW index to disk.

        Args:
            entries: Dict of key -> KVEntry (or similar dataclass)
            index: HNSW index object (optional)
            metadata: Additional metadata to save

        Returns:
            True if successful
        """
        try:
            # Convert entries to Arrow table
            if entries:
                keys = []
                values = []
                embeddings = []
                created_ats = []
                ttl_mss = []
                access_counts = []
                last_accesseds = []

                for key, entry in entries.items():
                    keys.append(key)
                    values.append(entry.value)
                    embeddings.append(entry.embedding.tolist())
                    created_ats.append(entry.created_at)
                    ttl_mss.append(entry.ttl_ms)
                    access_counts.append(getattr(entry, 'access_count', 0))
                    last_accesseds.append(getattr(entry, 'last_accessed', 0))

                table = pa.Table.from_arrays([
                    pa.array(keys),
                    pa.array(values, type=pa.large_binary()),
                    pa.array(embeddings, type=pa.list_(pa.float32())),
                    pa.array(created_ats, type=pa.int64()),
                    pa.array(ttl_mss, type=pa.int64()),
                    pa.array(access_counts, type=pa.int64()),
                    pa.array(last_accesseds, type=pa.int64()),
                ], schema=KV_SCHEMA)

                # Write to Arrow IPC file
                with pa.OSFile(str(self._data_file), 'wb') as sink:
                    writer = ipc.new_file(sink, table.schema)
                    writer.write_table(table)
                    writer.close()

                logger.info(f"Saved {len(entries)} entries to {self._data_file}")
            else:
                # Write empty file
                table = pa.Table.from_arrays([
                    pa.array([], type=pa.string()),
                    pa.array([], type=pa.large_binary()),
                    pa.array([], type=pa.list_(pa.float32())),
                    pa.array([], type=pa.int64()),
                    pa.array([], type=pa.int64()),
                    pa.array([], type=pa.int64()),
                    pa.array([], type=pa.int64()),
                ], schema=KV_SCHEMA)

                with pa.OSFile(str(self._data_file), 'wb') as sink:
                    writer = ipc.new_file(sink, table.schema)
                    writer.write_table(table)
                    writer.close()

            # Save HNSW index if provided
            if index is not None:
                index.save_index(str(self._index_file))
                logger.info(f"Saved HNSW index to {self._index_file}")

            # Save metadata
            meta = {
                'node_id': self._node_id,
                'entry_count': len(entries),
                'saved_at': int(time.time() * 1000),
                'version': '1.0',
            }
            if metadata:
                meta.update(metadata)

            with open(self._meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def load(self) -> Tuple[List[PersistedEntry], Dict[str, Any]]:
        """
        Load entries from disk.

        Returns:
            Tuple of (list of PersistedEntry, metadata dict)
        """
        entries = []
        metadata = {}

        try:
            # Load metadata
            if self._meta_file.exists():
                with open(self._meta_file, 'r') as f:
                    metadata = json.load(f)

            # Load Arrow data
            if self._data_file.exists():
                with pa.memory_map(str(self._data_file), 'r') as source:
                    reader = ipc.open_file(source)
                    table = reader.read_all()

                for i in range(table.num_rows):
                    entry = PersistedEntry(
                        key=table['key'][i].as_py(),
                        value=table['value'][i].as_py(),
                        embedding=np.array(table['embedding'][i].as_py(), dtype=np.float32),
                        created_at=table['created_at'][i].as_py(),
                        ttl_ms=table['ttl_ms'][i].as_py(),
                        access_count=table['access_count'][i].as_py(),
                        last_accessed=table['last_accessed'][i].as_py(),
                    )
                    entries.append(entry)

                logger.info(f"Loaded {len(entries)} entries from {self._data_file}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")

        return entries, metadata

    def load_index(self, dim: int) -> Optional[Any]:
        """
        Load HNSW index from disk.

        Args:
            dim: Embedding dimension

        Returns:
            HNSW index object, or None if not found
        """
        if not self._index_file.exists():
            return None

        try:
            import hnswlib
            index = hnswlib.Index(space='cosine', dim=dim)
            index.load_index(str(self._index_file))
            logger.info(f"Loaded HNSW index from {self._index_file}")
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return None

    def exists(self) -> bool:
        """Check if persisted data exists."""
        return self._data_file.exists()

    def delete(self) -> bool:
        """Delete all persisted data."""
        try:
            if self._data_file.exists():
                self._data_file.unlink()
            if self._index_file.exists():
                self._index_file.unlink()
            if self._meta_file.exists():
                self._meta_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'data_file': str(self._data_file),
            'data_exists': self._data_file.exists(),
            'data_size_bytes': self._data_file.stat().st_size if self._data_file.exists() else 0,
            'index_file': str(self._index_file),
            'index_exists': self._index_file.exists(),
            'index_size_bytes': self._index_file.stat().st_size if self._index_file.exists() else 0,
        }
        return stats


class SnapshotManager:
    """
    Manages snapshots for backup and recovery.
    """

    def __init__(self, data_dir: str, max_snapshots: int = 5):
        """
        Initialize snapshot manager.

        Args:
            data_dir: Base data directory
            max_snapshots: Maximum number of snapshots to keep
        """
        self._data_dir = Path(data_dir)
        self._snapshots_dir = self._data_dir / "snapshots"
        self._max_snapshots = max_snapshots

        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, name: str = None) -> str:
        """
        Create a snapshot of current data.

        Args:
            name: Optional snapshot name (default: timestamp)

        Returns:
            Snapshot name
        """
        if name is None:
            name = f"snapshot_{int(time.time() * 1000)}"

        snapshot_dir = self._snapshots_dir / name

        try:
            # Copy all data files to snapshot
            shutil.copytree(
                self._data_dir,
                snapshot_dir,
                ignore=shutil.ignore_patterns('snapshots')
            )
            logger.info(f"Created snapshot: {name}")

            # Cleanup old snapshots
            self._cleanup_old_snapshots()

            return name

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    def restore_snapshot(self, name: str) -> bool:
        """
        Restore from a snapshot.

        Args:
            name: Snapshot name to restore

        Returns:
            True if successful
        """
        snapshot_dir = self._snapshots_dir / name

        if not snapshot_dir.exists():
            logger.error(f"Snapshot not found: {name}")
            return False

        try:
            # Remove current data (except snapshots)
            for item in self._data_dir.iterdir():
                if item.name != 'snapshots':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            # Copy snapshot to data dir
            for item in snapshot_dir.iterdir():
                dest = self._data_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            logger.info(f"Restored snapshot: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots = []

        for item in self._snapshots_dir.iterdir():
            if item.is_dir():
                meta_file = item / "meta.json"
                meta = {}
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)

                snapshots.append({
                    'name': item.name,
                    'created_at': item.stat().st_mtime,
                    'size_bytes': sum(f.stat().st_size for f in item.rglob('*') if f.is_file()),
                    'metadata': meta,
                })

        return sorted(snapshots, key=lambda x: x['created_at'], reverse=True)

    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot."""
        snapshot_dir = self._snapshots_dir / name

        if not snapshot_dir.exists():
            return False

        try:
            shutil.rmtree(snapshot_dir)
            logger.info(f"Deleted snapshot: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete snapshot: {e}")
            return False

    def _cleanup_old_snapshots(self):
        """Remove old snapshots if exceeding max."""
        snapshots = self.list_snapshots()

        if len(snapshots) > self._max_snapshots:
            for snapshot in snapshots[self._max_snapshots:]:
                self.delete_snapshot(snapshot['name'])
