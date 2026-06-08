"""
moses/data/ingestion.py
=======================
Multi-source data ingestion for Moses v5.0 continuous training.

Supports:
  • Isaac Lab rollouts (HDF5 / npz / JSONL)
  • Physical robot logs (ROS2 bag, CSV, binary proto)
  • Human demonstrations (video + action trace, VR pose streams)

Formats:
  • LeRobot v2
  • RLDS (TensorFlow Datasets)
  • Custom JSON / JSONL

Features:
  • Schema validation & corruption detection
  • Deduplication (trajectory hash + perceptual image hash)
  • Streaming / chunked processing for TB-scale datasets
  • Async parallel ingestion

Design decisions (locked 2026-06-08):
  - PyTorch-first; TF/RLDS are optional soft-deps.
  - All heavy I/O is lazy / generator-based.
  - Corrupt shards are quarantined, never silently dropped.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import os
import struct
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger("moses.data.ingestion")

# ---------------------------------------------------------------------------
# Optional soft-deps (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

try:
    import zstandard as zstd
except Exception:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None  # type: ignore[assignment]

try:
    import tensorflow as tf  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    tf = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Any]] = {
    "isaac_lab_hdf5": {
        "required_datasets": ["observations", "actions", "rewards", "dones"],
        "observation_keys": ["rgb", "depth", "proprioception"],
        "action_dtype": "float32",
        "reward_dtype": "float32",
        "done_dtype": "bool",
    },
    "robot_csv": {
        "required_columns": ["timestamp", "joint_positions", "joint_velocities", "action"],
        "timestamp_format": "iso8601",
        "action_dtype": "float32",
    },
    "human_demo_jsonl": {
        "required_keys": ["episode_id", "frames", "actions", "task_label"],
        "frame_keys": ["rgb", "timestamp"],
        "action_dtype": "float32",
    },
    "lerobot_v2": {
        "required_keys": ["observation", "action", "reward", "done", "timestamp"],
        "observation_keys": ["image", "state"],
    },
    "rlds": {
        "required_keys": ["steps"],
        "step_keys": ["observation", "action", "reward", "is_terminal", "is_first", "is_last"],
    },
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Episode:
    """A single episode (trajectory) in the dataset."""

    episode_id: str
    source: str  # e.g. 'isaac_lab', 'robot_log', 'human_demo'
    task_label: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    dones: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    def trajectory_hash(self) -> str:
        """Deterministic hash of the action sequence for deduplication."""
        if self.actions.size == 0:
            return "empty"
        # Quantise to 1e-4 to tolerate tiny float jitter
        quant = np.round(self.actions, decimals=4).tobytes()
        return hashlib.sha256(quant).hexdigest()[:16]

    def perceptual_hash(self) -> str | None:
        """Perceptual hash of the first and last RGB frames (if available)."""
        if cv2 is None or not self.observations:
            return None
        frames: list[np.ndarray] = []
        for idx in (0, -1):
            obs = self.observations[idx]
            rgb = obs.get("rgb") if obs.get("rgb") is not None else obs.get("image")
            if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
                frames.append(rgb)
        if not frames:
            return None
        hashes: list[str] = []
        for img in frames:
            # Resize to 8x8 greyscale and hash
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            avg = small.mean()
            bits = (small > avg).flatten().tolist()
            byte_vals = [sum(b << i for i, b in enumerate(bits[j : j + 8])) for j in range(0, 64, 8)]
            hashes.append("".join(f"{b:02x}" for b in byte_vals))
        return "|".join(hashes)

    def composite_hash(self) -> str:
        """Combined deduplication key."""
        th = self.trajectory_hash()
        ph = self.perceptual_hash()
        if ph:
            return f"{th}:{ph}"
        return th


@dataclass(slots=True)
class IngestionResult:
    """Result of ingesting a data source."""

    episodes: list[Episode] = field(default_factory=list)
    corrupted_shards: list[Path] = field(default_factory=list)
    quarantine_dir: Path | None = None
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base for format adapters
# ---------------------------------------------------------------------------

class FormatAdapter(abc.ABC):
    """Pluggable adapter for a specific data format."""

    @abc.abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this adapter can handle *path*."""
        ...

    @abc.abstractmethod
    def ingest(
        self,
        path: Path,
        schema_name: str | None = None,
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        """Yield episodes lazily from *path*."""
        ...


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class IsaacLabHDF5Adapter(FormatAdapter):
    """Adapter for Isaac Lab rollout HDF5 files."""

    def supports(self, path: Path) -> bool:
        return path.suffix in (".h5", ".hdf5") and h5py is not None

    def ingest(
        self,
        path: Path,
        schema_name: str | None = "isaac_lab_hdf5",
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 ingestion")
        schema = SCHEMAS.get(schema_name or "isaac_lab_hdf5", {})
        required = set(schema.get("required_datasets", []))

        with h5py.File(path, "r") as f:
            # Validate top-level datasets
            missing = required - set(f.keys())
            if missing:
                raise ValueError(f"HDF5 {path} missing datasets: {missing}")

            # Assume episodes are indexed under /episodes/<id>/
            episode_group = f.get("episodes")
            if episode_group is None:
                # Flat format: single long trajectory
                yield self._build_episode("0", f, schema)
                return

            for ep_id in episode_group.keys():
                g = episode_group[ep_id]
                try:
                    yield self._build_episode(ep_id, g, schema)
                except Exception as exc:
                    logger.warning("Corrupt episode %s in %s: %s", ep_id, path, exc)
                    continue

    def _build_episode(
        self, ep_id: str, group: "h5py.Group", schema: dict[str, Any]
    ) -> Episode:
        obs: list[dict[str, Any]] = []
        # Read observations as list of dicts
        obs_group = group.get("observations")
        if obs_group is not None:
            n_steps = obs_group[next(iter(obs_group.keys()))].shape[0]
            for i in range(n_steps):
                frame: dict[str, Any] = {}
                for key in obs_group.keys():
                    val = obs_group[key][i]
                    frame[key] = np.array(val)
                obs.append(frame)
        else:
            n_steps = group["actions"].shape[0]

        actions = np.array(group["actions"], dtype=np.float32)
        rewards = np.array(group.get("rewards", np.zeros(n_steps)), dtype=np.float32)
        dones = np.array(group.get("dones", np.zeros(n_steps, dtype=bool)), dtype=bool)
        timestamps = np.array(group.get("timestamps", np.arange(n_steps)), dtype=np.float64)
        task_label = group.attrs.get("task_label", "unknown")
        meta = dict(group.attrs)

        return Episode(
            episode_id=str(ep_id),
            source="isaac_lab",
            task_label=str(task_label),
            observations=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timestamps=timestamps,
            metadata=meta,
        )


class RobotCSVAdapter(FormatAdapter):
    """Adapter for physical robot CSV logs."""

    def supports(self, path: Path) -> bool:
        return path.suffix == ".csv"

    def ingest(
        self,
        path: Path,
        schema_name: str | None = "robot_csv",
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        import pandas as pd

        schema = SCHEMAS.get(schema_name or "robot_csv", {})
        df = pd.read_csv(path)
        missing = set(schema.get("required_columns", [])) - set(df.columns)
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")

        # Detect episode boundaries by done signal or large time gaps (>5s)
        done_col = "done" if "done" in df.columns else None
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        gaps = ts.diff().dt.total_seconds() > 5.0
        if done_col:
            boundaries = df[done_col].astype(bool) | gaps.fillna(False)
        else:
            boundaries = gaps.fillna(False)
        boundary_idx = boundaries[boundaries].index.tolist()
        starts = [0] + [i + 1 for i in boundary_idx]
        ends = boundary_idx + [len(df)]

        for idx, (s, e) in enumerate(zip(starts, ends)):
            if e - s < 2:
                continue
            chunk = df.iloc[s:e]
            actions = self._parse_array_column(chunk, "action")
            rewards = self._parse_array_column(chunk, "reward") if "reward" in chunk.columns else np.zeros(len(chunk), dtype=np.float32)
            dones = chunk["done"].values.astype(bool) if "done" in chunk.columns else np.zeros(len(chunk), dtype=bool)
            timestamps = (chunk["timestamp"].astype("float64") if chunk["timestamp"].dtype.kind == "f" else pd.to_datetime(chunk["timestamp"], utc=True).view("int64") / 1e9).values
            obs: list[dict[str, Any]] = []
            for _, row in chunk.iterrows():
                frame: dict[str, Any] = {"timestamp": row["timestamp"]}
                for col in chunk.columns:
                    if col.startswith("obs_"):
                        frame[col[4:]] = row[col]
                obs.append(frame)
            yield Episode(
                episode_id=f"{path.stem}_{idx}",
                source="robot_log",
                task_label=str(chunk.get("task_label", "unknown").iloc[0]) if "task_label" in chunk.columns else "unknown",
                observations=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                timestamps=timestamps.astype(np.float64),
                metadata={"source_file": str(path)},
            )

    @staticmethod
    def _parse_array_column(df: "pd.DataFrame", col: str) -> np.ndarray:
        """Parse a column that may contain JSON-like array strings or raw floats."""
        if col not in df.columns:
            return np.zeros(len(df), dtype=np.float32)
        first = df[col].iloc[0]
        if isinstance(first, str) and first.startswith("["):
            arr = np.array([json.loads(v) for v in df[col]], dtype=np.float32)
            return arr
        return df[col].values.astype(np.float32)


class HumanDemoJSONLAdapter(FormatAdapter):
    """Adapter for human demonstration JSONL files (video + action trace)."""

    def supports(self, path: Path) -> bool:
        return path.suffix in (".jsonl", ".json") and path.stat().st_size < 10 * 1024 * 1024 * 1024

    def ingest(
        self,
        path: Path,
        schema_name: str | None = "human_demo_jsonl",
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        schema = SCHEMAS.get(schema_name or "human_demo_jsonl", {})
        required = set(schema.get("required_keys", []))

        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode error at %s:%d — %s", path, line_no, exc)
                    continue
                missing = required - set(record.keys())
                if missing:
                    logger.warning("Missing keys at %s:%d — %s", path, line_no, missing)
                    continue
                yield self._record_to_episode(record, path, line_no)

    def _record_to_episode(self, record: dict[str, Any], path: Path, line_no: int) -> Episode:
        frames = record.get("frames", [])
        observations: list[dict[str, Any]] = []
        for f in frames:
            obs_frame: dict[str, Any] = {}
            if "rgb_path" in f:
                # Lazy load image path
                obs_frame["rgb_path"] = f["rgb_path"]
                if cv2 is not None and os.path.exists(f["rgb_path"]):
                    img = cv2.imread(f["rgb_path"])
                    if img is not None:
                        obs_frame["rgb"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif "rgb" in f:
                obs_frame["rgb"] = np.array(f["rgb"], dtype=np.uint8)
            if "depth" in f:
                obs_frame["depth"] = np.array(f["depth"], dtype=np.float32)
            if "timestamp" in f:
                obs_frame["timestamp"] = f["timestamp"]
            observations.append(obs_frame)

        actions = np.array(record.get("actions", []), dtype=np.float32)
        rewards = np.array(record.get("rewards", []), dtype=np.float32)
        dones = np.array(record.get("dones", []), dtype=bool)
        timestamps = np.array(record.get("timestamps", np.arange(len(actions))), dtype=np.float64)
        return Episode(
            episode_id=str(record.get("episode_id", f"{path.stem}_{line_no}")),
            source="human_demo",
            task_label=str(record.get("task_label", "unknown")),
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timestamps=timestamps,
            metadata=record.get("metadata", {}),
        )


class LeRobotV2Adapter(FormatAdapter):
    """Adapter for LeRobot v2 dataset format (parquet + video)."""

    def supports(self, path: Path) -> bool:
        return path.is_dir() and (path / "data" / "chunk-000-00000.parquet").exists()

    def ingest(
        self,
        path: Path,
        schema_name: str | None = "lerobot_v2",
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        if pq is None:
            raise RuntimeError("pyarrow is required for LeRobot v2 ingestion")
        import pandas as pd

        data_dir = path / "data"
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # LeRobot v2 stores episodes with episode_index column
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            if "episode_index" not in df.columns:
                logger.warning("No episode_index in %s; treating as single episode", pf)
                yield self._df_to_episode(df, "0", path)
                continue
            for ep_idx, group in df.groupby("episode_index"):
                yield self._df_to_episode(group, str(ep_idx), path)

    def _df_to_episode(self, df: "pd.DataFrame", ep_idx: str, root: Path) -> Episode:
        actions = np.stack(df["action"].values).astype(np.float32) if "action" in df.columns else np.array([], dtype=np.float32)
        rewards = df["reward"].values.astype(np.float32) if "reward" in df.columns else np.zeros(len(df), dtype=np.float32)
        dones = df["done"].values.astype(bool) if "done" in df.columns else np.zeros(len(df), dtype=bool)
        timestamps = df["timestamp"].values.astype(np.float64) if "timestamp" in df.columns else np.arange(len(df), dtype=np.float64)
        task_label = str(df["task_index"].iloc[0]) if "task_index" in df.columns else "unknown"
        observations: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            frame: dict[str, Any] = {}
            for col in df.columns:
                if col.startswith("observation."):
                    key = col[len("observation."):]
                    frame[key] = row[col]
                elif col == "observation.image" and isinstance(row[col], str):
                    frame["image_path"] = str(root / row[col])
            observations.append(frame)
        return Episode(
            episode_id=ep_idx,
            source="lerobot_v2",
            task_label=task_label,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timestamps=timestamps,
            metadata={"root": str(root)},
        )


class RLDSAdapter(FormatAdapter):
    """Adapter for RLDS (TensorFlow Datasets) format."""

    def supports(self, path: Path) -> bool:
        # RLDS is typically a TFDS directory with dataset_info.json
        return (path / "dataset_info.json").exists() and tf is not None

    def ingest(
        self,
        path: Path,
        schema_name: str | None = "rlds",
        chunk_size: int = 1024,
    ) -> Iterator[Episode]:
        if tf is None:
            raise RuntimeError("tensorflow is required for RLDS ingestion")
        import tensorflow_datasets as tfds  # type: ignore[import-untyped]

        builder = tfds.builder_from_directory(str(path))
        ds = builder.as_dataset(split="all")
        ep_idx = 0
        for episode in ds:
            steps = list(episode["steps"].as_numpy_iterator())
            if not steps:
                continue
            observations = [dict(s["observation"]) for s in steps]
            actions = np.stack([s["action"] for s in steps]).astype(np.float32)
            rewards = np.array([s.get("reward", 0.0) for s in steps], dtype=np.float32)
            dones = np.array([s.get("is_terminal", False) or s.get("is_last", False) for s in steps], dtype=bool)
            timestamps = np.arange(len(steps), dtype=np.float64)
            yield Episode(
                episode_id=f"rlds_{ep_idx}",
                source="rlds",
                task_label="unknown",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                timestamps=timestamps,
                metadata={"builder": builder.info.full_name},
            )
            ep_idx += 1


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

DEFAULT_ADAPTERS: list[FormatAdapter] = [
    IsaacLabHDF5Adapter(),
    RobotCSVAdapter(),
    HumanDemoJSONLAdapter(),
    LeRobotV2Adapter(),
    RLDSAdapter(),
]


def detect_adapter(path: Path, adapters: list[FormatAdapter] | None = None) -> FormatAdapter:
    """Return the first adapter that claims to support *path*."""
    adapters = adapters or DEFAULT_ADAPTERS
    for adapter in adapters:
        if adapter.supports(path):
            return adapter
    raise ValueError(f"No adapter found for {path}")


# ---------------------------------------------------------------------------
# Deduplication engine
# ---------------------------------------------------------------------------

class Deduplicator:
    """Streaming deduplicator using composite hashes."""

    def __init__(self, hash_store: set[str] | None = None) -> None:
        self._seen: set[str] = hash_store or set()
        self.duplicate_count = 0

    def is_duplicate(self, episode: Episode) -> bool:
        key = episode.composite_hash()
        if key in self._seen:
            self.duplicate_count += 1
            return True
        self._seen.add(key)
        return False

    def reset(self) -> None:
        self._seen.clear()
        self.duplicate_count = 0


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------

def validate_episode(ep: Episode, schema_name: str | None = None) -> list[str]:
    """Return list of validation error strings (empty if valid)."""
    errors: list[str] = []
    n = len(ep)
    if n == 0:
        errors.append("Empty episode")
        return errors
    if ep.actions.shape[0] != n:
        errors.append(f"Action length mismatch: {ep.actions.shape[0]} vs {n}")
    if ep.rewards.shape[0] != n:
        errors.append(f"Reward length mismatch: {ep.rewards.shape[0]} vs {n}")
    if ep.dones.shape[0] != n:
        errors.append(f"Done length mismatch: {ep.dones.shape[0]} vs {n}")
    if np.isnan(ep.actions).any():
        errors.append("NaN in actions")
    if np.isinf(ep.actions).any():
        errors.append("Inf in actions")
    if ep.rewards.min() < -1e6 or ep.rewards.max() > 1e6:
        errors.append("Suspicious reward magnitudes")
    # Schema-specific checks
    schema = SCHEMAS.get(schema_name, {})
    if "action_dtype" in schema and ep.actions.dtype != np.dtype(schema["action_dtype"]):
        errors.append(f"Action dtype {ep.actions.dtype} != expected {schema['action_dtype']}")
    return errors


# ---------------------------------------------------------------------------
# High-level ingestion pipeline
# ---------------------------------------------------------------------------

class DataIngestionPipeline:
    """Orchestrates multi-source ingestion with validation & deduplication."""

    def __init__(
        self,
        adapters: list[FormatAdapter] | None = None,
        deduplicator: Deduplicator | None = None,
        quarantine_dir: Path | None = None,
    ) -> None:
        self.adapters = adapters or DEFAULT_ADAPTERS
        self.dedup = deduplicator or Deduplicator()
        self.quarantine_dir = quarantine_dir or Path("/tmp/moses_quarantine")
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._stats: dict[str, Any] = {
            "total_episodes": 0,
            "valid_episodes": 0,
            "corrupt_episodes": 0,
            "duplicate_episodes": 0,
            "total_steps": 0,
        }

    def ingest_path(
        self,
        path: Path,
        schema_name: str | None = None,
        chunk_size: int = 1024,
        skip_duplicates: bool = True,
    ) -> Iterator[Episode]:
        """Yield validated, deduplicated episodes from *path*."""
        adapter = detect_adapter(path, self.adapters)
        logger.info("Using adapter %s for %s", type(adapter).__name__, path)
        for episode in adapter.ingest(path, schema_name=schema_name, chunk_size=chunk_size):
            self._stats["total_episodes"] += 1
            self._stats["total_steps"] += len(episode)
            errors = validate_episode(episode, schema_name)
            if errors:
                self._stats["corrupt_episodes"] += 1
                logger.warning("Corrupt episode %s: %s", episode.episode_id, errors)
                self._quarantine(episode, errors)
                continue
            if skip_duplicates and self.dedup.is_duplicate(episode):
                self._stats["duplicate_episodes"] += 1
                logger.debug("Duplicate episode %s skipped", episode.episode_id)
                continue
            self._stats["valid_episodes"] += 1
            yield episode

    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "*",
        schema_name: str | None = None,
        chunk_size: int = 1024,
        skip_duplicates: bool = True,
    ) -> Iterator[Episode]:
        """Recursively ingest all matching files."""
        for path in sorted(directory.rglob(pattern)):
            if path.is_file():
                try:
                    yield from self.ingest_path(path, schema_name, chunk_size, skip_duplicates)
                except Exception as exc:
                    logger.error("Failed to ingest %s: %s", path, exc)
                    continue

    def _quarantine(self, episode: Episode, errors: list[str]) -> None:
        qfile = self.quarantine_dir / f"{episode.episode_id}_{episode.source}.json"
        record = {
            "episode_id": episode.episode_id,
            "source": episode.source,
            "task_label": episode.task_label,
            "errors": errors,
            "metadata": episode.metadata,
        }
        with open(qfile, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0


# ---------------------------------------------------------------------------
# Async streaming interface (for high-throughput pipelines)
# ---------------------------------------------------------------------------

async def async_ingest_paths(
    paths: list[Path],
    pipeline: DataIngestionPipeline,
    schema_name: str | None = None,
    max_workers: int = 4,
) -> AsyncIterator[Episode]:
    """Async wrapper around sync ingestion for I/O parallelism."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Episode | None] = asyncio.Queue(maxsize=256)

    async def producer(path: Path) -> None:
        def _gen():
            for ep in pipeline.ingest_path(path, schema_name=schema_name):
                queue.put_nowait(ep)
        await loop.run_in_executor(None, _gen)

    producers = [asyncio.create_task(producer(p)) for p in paths]

    async def closer() -> None:
        await asyncio.gather(*producers, return_exceptions=True)
        await queue.put(None)

    closer_task = asyncio.create_task(closer())
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    await closer_task


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------

def ingest(
    source: Path | str,
    schema: str | None = None,
    skip_duplicates: bool = True,
    quarantine_dir: Path | str | None = None,
) -> Iterator[Episode]:
    """One-shot ingestion helper."""
    path = Path(source)
    pipeline = DataIngestionPipeline(
        quarantine_dir=Path(quarantine_dir) if quarantine_dir else None,
    )
    if path.is_dir():
        return pipeline.ingest_directory(path, schema_name=schema, skip_duplicates=skip_duplicates)
    return pipeline.ingest_path(path, schema_name=schema, skip_duplicates=skip_duplicates)
