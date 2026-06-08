"""
moses/data/storage.py
=====================
Efficient data storage for Moses v5.0 continuous training.

Features:
  • Format: Parquet for tabular metadata + Zarr for dense tensor arrays
  • Versioning: dataset versions with lineage tracking
  • Indexing: fast queries by task, quality, date
  • Compression: Blosc / LZ4 / Zstd to reduce storage footprint
  • Streaming: memory-mapped reads for TB-scale datasets

Design decisions (locked 2026-06-08):
  - Parquet + Zarr chosen over TFRecord/HDF5 for:
    • Columnar queryability (Parquet)
    • Chunked n-dimensional arrays (Zarr)
    • Cloud-native / object-store friendly
    • No hard dependency on TensorFlow
  - TFRecord writer provided as optional adapter for TF ecosystems.
  - Lineage stored as JSON sidecar files.
  - Index is a SQLite DB for fast metadata queries.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from moses.data.ingestion import Episode

logger = logging.getLogger("moses.data.storage")

# ---------------------------------------------------------------------------
# Optional soft-deps
# ---------------------------------------------------------------------------
try:
    import zarr  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    zarr = None  # type: ignore[assignment]

try:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import blosc  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    blosc = None  # type: ignore[assignment]

try:
    import tensorflow as tf  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    tf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataset version / lineage
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DatasetVersion:
    """Immutable dataset version descriptor."""

    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_version: str | None = None
    created_at: float = field(default_factory=time.time)
    description: str = ""
    sources: list[str] = field(default_factory=list)
    num_episodes: int = 0
    num_steps: int = 0
    size_bytes: int = 0
    checksum: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    transforms_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "description": self.description,
            "sources": self.sources,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "tags": self.tags,
            "transforms_applied": self.transforms_applied,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetVersion":
        return cls(
            version_id=d["version_id"],
            parent_version=d.get("parent_version"),
            created_at=d.get("created_at", time.time()),
            description=d.get("description", ""),
            sources=d.get("sources", []),
            num_episodes=d.get("num_episodes", 0),
            num_steps=d.get("num_steps", 0),
            size_bytes=d.get("size_bytes", 0),
            checksum=d.get("checksum", ""),
            tags=d.get("tags", {}),
            transforms_applied=d.get("transforms_applied", []),
        )


# ---------------------------------------------------------------------------
# Episode serialization helpers
# ---------------------------------------------------------------------------

def episode_to_flat_dict(ep: Episode) -> dict[str, Any]:
    """Convert an Episode to a flat dict suitable for Parquet rows."""
    return {
        "episode_id": ep.episode_id,
        "source": ep.source,
        "task_label": ep.task_label,
        "length": len(ep),
        "action_dim": ep.actions.shape[1] if ep.actions.ndim >= 2 else 1,
        "total_reward": float(ep.rewards.sum()) if ep.rewards.size > 0 else 0.0,
        "mean_reward": float(ep.rewards.mean()) if ep.rewards.size > 0 else 0.0,
        "success": bool(ep.dones[-1]) if ep.dones.size > 0 else False,
        "start_time": float(ep.timestamps[0]) if ep.timestamps.size > 0 else 0.0,
        "end_time": float(ep.timestamps[-1]) if ep.timestamps.size > 0 else 0.0,
        "metadata_json": json.dumps(ep.metadata, default=str),
    }


def _zarr_create_array(group: "zarr.Group", name: str, data: np.ndarray) -> None:
    """Create a Zarr array using v3 API."""
    group.create_array(name, data=data, chunks=True)


def episode_to_zarr_group(ep: Episode, group: "zarr.Group", episode_idx: int) -> None:
    """Write episode tensors into a Zarr group."""
    if zarr is None:
        raise RuntimeError("zarr is required for Zarr storage")
    ep_group = group.create_group(f"episode_{episode_idx:08d}")
    _zarr_create_array(ep_group, "actions", ep.actions)
    _zarr_create_array(ep_group, "rewards", ep.rewards)
    _zarr_create_array(ep_group, "dones", ep.dones)
    _zarr_create_array(ep_group, "timestamps", ep.timestamps)
    # Observations: store vector fields as datasets, images as separate arrays
    obs_group = ep_group.create_group("observations")
    if ep.observations:
        # Infer keys from first frame
        first = ep.observations[0]
        for key in first.keys():
            if key in ("rgb", "image", "depth"):
                # Stack image frames
                try:
                    imgs = np.stack([obs[key] for obs in ep.observations if key in obs])
                    _zarr_create_array(obs_group, key, imgs)
                except Exception as exc:
                    logger.warning("Could not stack image key %s for episode %s: %s", key, ep.episode_id, exc)
            elif isinstance(first.get(key), np.ndarray):
                try:
                    arr = np.stack([obs[key] for obs in ep.observations if key in obs])
                    _zarr_create_array(obs_group, key, arr)
                except Exception as exc:
                    logger.warning("Could not stack obs key %s for episode %s: %s", key, ep.episode_id, exc)
            else:
                # Scalar / string metadata per step
                vals = [obs.get(key) for obs in ep.observations]
                obs_group.attrs[key] = json.dumps(vals, default=str)
    ep_group.attrs["episode_id"] = ep.episode_id
    ep_group.attrs["source"] = ep.source
    ep_group.attrs["task_label"] = ep.task_label
    ep_group.attrs["metadata"] = json.dumps(ep.metadata, default=str)


def episode_from_zarr_group(group: "zarr.Group") -> Episode:
    """Reconstruct an Episode from a Zarr group."""
    actions = np.array(group["actions"])
    rewards = np.array(group["rewards"])
    dones = np.array(group["dones"])
    timestamps = np.array(group["timestamps"])
    obs_group = group["observations"]
    n = len(actions)
    observations: list[dict[str, Any]] = []
    # Reconstruct per-step observations
    array_keys = [k for k in obs_group.keys() if hasattr(obs_group[k], 'shape')]
    for i in range(n):
        frame: dict[str, Any] = {}
        for key in array_keys:
            frame[key] = np.array(obs_group[key][i])
        # Restore scalar attrs
        for key in obs_group.attrs:
            if key not in ("episode_id", "source", "task_label", "metadata"):
                try:
                    vals = json.loads(obs_group.attrs[key])
                    frame[key] = vals[i]
                except Exception:
                    pass
        observations.append(frame)
    meta = json.loads(group.attrs.get("metadata", "{}"))
    return Episode(
        episode_id=group.attrs.get("episode_id", "unknown"),
        source=group.attrs.get("source", "unknown"),
        task_label=group.attrs.get("task_label", "unknown"),
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        timestamps=timestamps,
        metadata=meta,
    )


def _default_compressor() -> Any:
    """Return a Zarr compressor (Blosc LZ4 if available, else Zstd)."""
    if zarr is None:
        return None
    try:
        return zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE)
    except Exception:
        try:
            return zarr.Zstd(level=3)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Indexing (SQLite)
# ---------------------------------------------------------------------------

class DatasetIndex:
    """SQLite-backed index for fast metadata queries."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    source TEXT,
                    task_label TEXT,
                    length INTEGER,
                    action_dim INTEGER,
                    total_reward REAL,
                    mean_reward REAL,
                    success INTEGER,
                    start_time REAL,
                    end_time REAL,
                    version_id TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task ON episodes(task_label)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source ON episodes(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_version ON episodes(version_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_time ON episodes(start_time, end_time)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    parent_version TEXT,
                    created_at REAL,
                    description TEXT,
                    num_episodes INTEGER,
                    num_steps INTEGER,
                    size_bytes INTEGER,
                    checksum TEXT,
                    tags_json TEXT,
                    transforms_json TEXT
                )
                """
            )
            conn.commit()

    def insert_episodes(self, rows: list[dict[str, Any]], version_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes
                    (episode_id, source, task_label, length, action_dim,
                     total_reward, mean_reward, success, start_time, end_time,
                     version_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["episode_id"],
                        row.get("source", ""),
                        row.get("task_label", ""),
                        row.get("length", 0),
                        row.get("action_dim", 0),
                        row.get("total_reward", 0.0),
                        row.get("mean_reward", 0.0),
                        int(row.get("success", False)),
                        row.get("start_time", 0.0),
                        row.get("end_time", 0.0),
                        version_id,
                        row.get("metadata_json", "{}"),
                    ),
                )
            conn.commit()

    def insert_version(self, version: DatasetVersion) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO versions
                (version_id, parent_version, created_at, description,
                 num_episodes, num_steps, size_bytes, checksum,
                 tags_json, transforms_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.parent_version,
                    version.created_at,
                    version.description,
                    version.num_episodes,
                    version.num_steps,
                    version.size_bytes,
                    version.checksum,
                    json.dumps(version.tags),
                    json.dumps(version.transforms_applied),
                ),
            )
            conn.commit()

    def query(
        self,
        task_label: str | None = None,
        source: str | None = None,
        version_id: str | None = None,
        success: bool | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query the index and return matching episode metadata rows."""
        conditions: list[str] = []
        params: list[Any] = []
        if task_label is not None:
            conditions.append("task_label = ?")
            params.append(task_label)
        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        if version_id is not None:
            conditions.append("version_id = ?")
            params.append(version_id)
        if success is not None:
            conditions.append("success = ?")
            params.append(int(success))
        if min_length is not None:
            conditions.append("length >= ?")
            params.append(min_length)
        if max_length is not None:
            conditions.append("length <= ?")
            params.append(max_length)
        if min_reward is not None:
            conditions.append("total_reward >= ?")
            params.append(min_reward)
        if max_reward is not None:
            conditions.append("total_reward <= ?")
            params.append(max_reward)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM episodes {where} ORDER BY start_time DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_version(self, version_id: str) -> DatasetVersion | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM versions WHERE version_id = ?", (version_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            return DatasetVersion(
                version_id=d["version_id"],
                parent_version=d.get("parent_version"),
                created_at=d["created_at"],
                description=d["description"],
                num_episodes=d["num_episodes"],
                num_steps=d["num_steps"],
                size_bytes=d["size_bytes"],
                checksum=d["checksum"],
                tags=json.loads(d.get("tags_json", "{}")),
                transforms_applied=json.loads(d.get("transforms_json", "[]")),
            )

    def list_versions(self) -> list[DatasetVersion]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM versions ORDER BY created_at DESC")
            rows = cur.fetchall()
        versions = []
        for d in (dict(r) for r in rows):
            versions.append(
                DatasetVersion(
                    version_id=d["version_id"],
                    parent_version=d.get("parent_version"),
                    created_at=d["created_at"],
                    description=d["description"],
                    num_episodes=d["num_episodes"],
                    num_steps=d["num_steps"],
                    size_bytes=d["size_bytes"],
                    checksum=d["checksum"],
                    tags=json.loads(d.get("tags_json", "{}")),
                    transforms_applied=json.loads(d.get("transforms_json", "[]")),
                )
            )
        return versions


# ---------------------------------------------------------------------------
# Main storage backend
# ---------------------------------------------------------------------------

class DatasetStore:
    """Production storage backend: Parquet metadata + Zarr tensors + SQLite index."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.zarr_dir = self.root / "zarr"
        self.parquet_dir = self.root / "parquet"
        self.lineage_dir = self.root / "lineage"
        self.index = DatasetIndex(self.root / "index.db")
        self.zarr_dir.mkdir(exist_ok=True)
        self.parquet_dir.mkdir(exist_ok=True)
        self.lineage_dir.mkdir(exist_ok=True)
        self._current_version: DatasetVersion | None = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_episodes(
        self,
        episodes: Iterator[Episode],
        version: DatasetVersion | None = None,
        batch_size: int = 256,
    ) -> DatasetVersion:
        """Write episodes to storage and update the index."""
        version = version or DatasetVersion()
        self._current_version = version
        zarr_path = str(self.zarr_dir / version.version_id)
        root_group = zarr.open_group(zarr_path, mode="w")
        rows: list[dict[str, Any]] = []
        episode_idx = 0
        total_steps = 0
        size_bytes = 0

        for ep in episodes:
            episode_to_zarr_group(ep, root_group, episode_idx)
            rows.append(episode_to_flat_dict(ep))
            total_steps += len(ep)
            episode_idx += 1
            if episode_idx % batch_size == 0:
                logger.info("Written %d episodes for version %s", episode_idx, version.version_id)

        # Write metadata as Parquet
        if rows and pd is not None and pa is not None:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            parquet_path = self.parquet_dir / f"{version.version_id}.parquet"
            pq.write_table(
                table,
                parquet_path,
                compression="zstd",
                use_dictionary=["episode_id", "source", "task_label"],
            )
            size_bytes += parquet_path.stat().st_size
        elif rows:
            # Fallback: JSONL
            jsonl_path = self.parquet_dir / f"{version.version_id}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
            size_bytes += jsonl_path.stat().st_size

        # Update version stats
        version.num_episodes = episode_idx
        version.num_steps = total_steps
        version.size_bytes = size_bytes + self._estimate_zarr_size(self.zarr_dir / version.version_id)
        version.checksum = self._checksum_version(version)
        self.index.insert_version(version)
        self.index.insert_episodes(rows, version.version_id)
        self._write_lineage(version)
        logger.info(
            "Version %s committed: %d episodes, %d steps, %d bytes",
            version.version_id,
            version.num_episodes,
            version.num_steps,
            version.size_bytes,
        )
        return version

    def _estimate_zarr_size(self, path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def _checksum_version(self, version: DatasetVersion) -> str:
        data = f"{version.version_id}:{version.parent_version}:{version.num_episodes}:{version.num_steps}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _write_lineage(self, version: DatasetVersion) -> None:
        path = self.lineage_dir / f"{version.version_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(version.to_dict(), fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_episodes(
        self,
        version_id: str | None = None,
        episode_ids: Sequence[str] | None = None,
        task_label: str | None = None,
        source: str | None = None,
    ) -> Iterator[Episode]:
        """Stream episodes back from storage."""
        version_id = version_id or (self._current_version.version_id if self._current_version else None)
        if version_id is None:
            raise ValueError("No version_id specified and no current version")
        zarr_path = str(self.zarr_dir / version_id)
        root_group = zarr.open_group(zarr_path, mode="r")

        # Use index to find matching episode_ids if filters provided
        if episode_ids is None and (task_label or source):
            rows = self.index.query(
                task_label=task_label,
                source=source,
                version_id=version_id,
            )
            episode_ids = [r["episode_id"] for r in rows]

        for key in sorted(root_group.group_keys()):
            group = root_group[key]
            ep_id = group.attrs.get("episode_id", "")
            if episode_ids is not None and ep_id not in episode_ids:
                continue
            yield episode_from_zarr_group(group)

    def read_metadata(self, version_id: str | None = None) -> list[dict[str, Any]]:
        """Read episode metadata as a list of dicts (fast, no tensors)."""
        version_id = version_id or (self._current_version.version_id if self._current_version else None)
        if version_id is None:
            return []
        parquet_path = self.parquet_dir / f"{version_id}.parquet"
        if parquet_path.exists() and pq is not None:
            table = pq.read_table(parquet_path)
            return table.to_pandas().to_dict("records")
        jsonl_path = self.parquet_dir / f"{version_id}.jsonl"
        if jsonl_path.exists():
            rows = []
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    rows.append(json.loads(line))
            return rows
        return []

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def create_version(
        self,
        parent_version: str | None = None,
        description: str = "",
        sources: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> DatasetVersion:
        version = DatasetVersion(
            parent_version=parent_version,
            description=description,
            sources=sources or [],
            tags=tags or {},
        )
        self._current_version = version
        return version

    def get_version(self, version_id: str) -> DatasetVersion | None:
        return self.index.get_version(version_id)

    def list_versions(self) -> list[DatasetVersion]:
        return self.index.list_versions()

    def lineage(self, version_id: str) -> list[DatasetVersion]:
        """Return the ancestry chain from *version_id* back to root."""
        chain: list[DatasetVersion] = []
        current = self.get_version(version_id)
        while current is not None:
            chain.append(current)
            if current.parent_version is None:
                break
            current = self.get_version(current.parent_version)
        return list(reversed(chain))

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def compress_version(self, version_id: str, method: str = "zstd") -> Path:
        """Create a compressed archive of a version for cold storage."""
        import tarfile

        src = self.zarr_dir / version_id
        dst = self.root / "archives"
        dst.mkdir(exist_ok=True)
        archive_path = dst / f"{version_id}.tar.{method}"
        mode = f"w:{method}" if method != "zstd" else "w"
        with tarfile.open(archive_path, mode) as tar:
            tar.add(src, arcname=version_id)
        logger.info("Archived version %s to %s", version_id, archive_path)
        return archive_path


# ---------------------------------------------------------------------------
# Streaming dataloader interface
# ---------------------------------------------------------------------------

class StreamingDataset:
    """PyTorch-compatible streaming dataset that reads from DatasetStore.

    Does NOT inherit from torch.utils.data.Dataset to avoid a hard
    dependency on torch in this module.  The training loop can wrap
    this in a torch.utils.data.IterableDataset if desired.
    """

    def __init__(
        self,
        store: DatasetStore,
        version_id: str | None = None,
        task_label: str | None = None,
        source: str | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.store = store
        self.version_id = version_id
        self.task_label = task_label
        self.source = source
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        # Pre-load metadata (lightweight)
        self._meta = store.index.query(
            task_label=task_label,
            source=source,
            version_id=version_id,
        )
        self._episode_ids = [m["episode_id"] for m in self._meta]
        if self.shuffle:
            self._rng.shuffle(self._episode_ids)
        self._idx = 0

    def __len__(self) -> int:
        return len(self._episode_ids)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield training samples as dicts with torch-ready tensors."""
        for ep_id in self._episode_ids:
            # Read single episode
            eps = list(self.store.read_episodes(
                version_id=self.version_id,
                episode_ids=[ep_id],
            ))
            if not eps:
                continue
            ep = eps[0]
            # Yield each step as a sample
            for t in range(len(ep)):
                sample: dict[str, Any] = {
                    "episode_id": ep.episode_id,
                    "step": t,
                    "observation": self._extract_obs(ep.observations, t),
                    "action": ep.actions[t],
                    "reward": ep.rewards[t] if ep.rewards.size > 0 else 0.0,
                    "done": ep.dones[t] if ep.dones.size > 0 else False,
                    "timestamp": ep.timestamps[t] if ep.timestamps.size > 0 else float(t),
                    "task_label": ep.task_label,
                }
                yield sample
            self._idx += 1

    @staticmethod
    def _extract_obs(observations: list[dict[str, Any]], t: int) -> dict[str, Any]:
        """Extract observation dict at step t, converting paths to arrays if needed."""
        obs = observations[t]
        result: dict[str, Any] = {}
        for key, val in obs.items():
            if key.endswith("_path"):
                continue
            result[key] = val
        return result


# ---------------------------------------------------------------------------
# Optional TFRecord adapter
# ---------------------------------------------------------------------------

class TFRecordWriter:
    """Write episodes to TFRecord format (for TF/RLDS ecosystems)."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if tf is None:
            raise RuntimeError("tensorflow is required for TFRecord export")

    def write(self, episodes: Iterator[Episode], filename: str = "data.tfrecord") -> Path:
        path = self.output_dir / filename
        with tf.io.TFRecordWriter(str(path)) as writer:
            for ep in episodes:
                feature: dict[str, tf.train.Feature] = {}
                feature["episode_id"] = self._bytes_feature(ep.episode_id.encode())
                feature["source"] = self._bytes_feature(ep.source.encode())
                feature["task_label"] = self._bytes_feature(ep.task_label.encode())
                feature["actions"] = self._bytes_feature(ep.actions.tobytes())
                feature["rewards"] = self._bytes_feature(ep.rewards.tobytes())
                feature["dones"] = self._bytes_feature(ep.dones.tobytes())
                feature["timestamps"] = self._bytes_feature(ep.timestamps.tobytes())
                feature["length"] = self._int64_feature(len(ep))
                # Encode observations as JSON bytes
                obs_json = json.dumps(
                    [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items()} for o in ep.observations],
                    default=str,
                ).encode()
                feature["observations"] = self._bytes_feature(obs_json)
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        return path

    @staticmethod
    def _bytes_feature(value: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def init_store(root_dir: Path | str) -> DatasetStore:
    """Initialise a new DatasetStore at *root_dir*."""
    return DatasetStore(root_dir)
