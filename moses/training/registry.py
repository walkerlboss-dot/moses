"""
Moses Model Registry
====================
Version control, artifact storage, and lineage tracking for trained policies.

Integrates with MLflow (optional) for UI and W&B (optional) for experiment tracking.
Can also run in a standalone mode using a local SQLite/PostgreSQL backend.

Example
-------
>>> from moses.training.registry import ModelRegistry
>>> reg = ModelRegistry(backend_url="sqlite:///moses_registry.db")
>>> version = reg.register(
...     run_name="ppo-v5.1",
...     checkpoint_path="/checkpoints/ppo-v5.1.ckpt",
...     metrics={"mean_reward": 142.0},
...     hyperparameters={"lr": 3e-4},
...     tags=["production"],
... )
>>> reg.promote(version.version_id, "production")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import yaml

logger = logging.getLogger("moses.training.registry")


# ---------------------------------------------------------------------------
# Protocols / Pluggable interfaces
# ---------------------------------------------------------------------------

class AlertChannel(Protocol):
    """Pluggable alert channel for registry events."""

    def send(self, subject: str, body: str) -> None:
        ...


class _LogAlertChannel:
    """Default alert channel that logs at WARNING level."""

    def send(self, subject: str, body: str) -> None:
        logger.warning("[ALERT] %s | %s", subject, body)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelVersion:
    """Immutable descriptor for a registered model version."""

    version_id: str
    run_name: str
    created_at: float
    git_commit: Optional[str]
    dataset_version: Optional[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parent_version_id: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # role -> path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        return cls(**d)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _SqliteBackend:
    """Lightweight SQLite backend for standalone registry usage."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    git_commit TEXT,
                    dataset_version TEXT,
                    hyperparameters TEXT,
                    metrics TEXT,
                    tags TEXT,
                    parent_version_id TEXT,
                    artifacts TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS version_tags (
                    version_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (version_id, tag)
                )
                """
            )
            conn.commit()

    def insert(self, version: ModelVersion) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_versions
                (version_id, run_name, created_at, git_commit, dataset_version,
                 hyperparameters, metrics, tags, parent_version_id, artifacts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.run_name,
                    version.created_at,
                    version.git_commit,
                    version.dataset_version,
                    json.dumps(version.hyperparameters),
                    json.dumps(version.metrics),
                    json.dumps(version.tags),
                    version.parent_version_id,
                    json.dumps(version.artifacts),
                ),
            )
            for tag in version.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO version_tags (version_id, tag) VALUES (?, ?)",
                    (version.version_id, tag),
                )
            conn.commit()

    def get(self, version_id: str) -> Optional[ModelVersion]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def list_versions(
        self,
        run_name: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelVersion]:
        query = "SELECT * FROM model_versions WHERE 1=1"
        params: List[Any] = []
        if run_name:
            query += " AND run_name = ?"
            params.append(run_name)
        if tag:
            query += (
                " AND version_id IN (SELECT version_id FROM version_tags WHERE tag = ?)"
            )
            params.append(tag)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_version(r) for r in rows]

    def add_tag(self, version_id: str, tag: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO version_tags (version_id, tag) VALUES (?, ?)",
                (version_id, tag),
            )
            # Also update JSON tags column for convenience
            conn.execute(
                """
                UPDATE model_versions
                SET tags = (
                    SELECT json_group_array(tag)
                    FROM version_tags
                    WHERE version_id = ?
                )
                WHERE version_id = ?
                """,
                (version_id, version_id),
            )
            conn.commit()

    def remove_tag(self, version_id: str, tag: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM version_tags WHERE version_id = ? AND tag = ?",
                (version_id, tag),
            )
            conn.execute(
                """
                UPDATE model_versions
                SET tags = (
                    SELECT json_group_array(tag)
                    FROM version_tags
                    WHERE version_id = ?
                )
                WHERE version_id = ?
                """,
                (version_id, version_id),
            )
            conn.commit()

    def _row_to_version(self, row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            version_id=row[0],
            run_name=row[1],
            created_at=row[2],
            git_commit=row[3],
            dataset_version=row[4],
            hyperparameters=json.loads(row[5]),
            metrics=json.loads(row[6]),
            tags=json.loads(row[7]),
            parent_version_id=row[8],
            artifacts=json.loads(row[9]),
        )


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Central registry for Moses model versions.

    Parameters
    ----------
    backend_url :
        Connection string. ``sqlite:///path.db`` for local SQLite,
        or ``mlflow://tracking-uri`` to delegate to MLflow.
    artifact_store :
        Directory or URI prefix where checkpoints / ONNX / TensorRT files are copied.
    alert_channel :
        Optional pluggable alert channel for critical events.
    """

    def __init__(
        self,
        backend_url: str = "sqlite:///moses_registry.db",
        artifact_store: Union[str, Path] = "./moses_artifacts",
        alert_channel: Optional[AlertChannel] = None,
    ) -> None:
        self.backend_url = backend_url
        self.artifact_store = Path(artifact_store)
        self.artifact_store.mkdir(parents=True, exist_ok=True)
        self._alert = alert_channel or _LogAlertChannel()

        if backend_url.startswith("mlflow://"):
            self._backend: Union[_SqliteBackend, "_MlflowBackend"] = _MlflowBackend(
                tracking_uri=backend_url.replace("mlflow://", "")
            )
        else:
            db_path = backend_url.replace("sqlite:///", "")
            self._backend = _SqliteBackend(db_path)

    # -- Public API ----------------------------------------------------------

    def register(
        self,
        run_name: str,
        checkpoint_path: Union[str, Path],
        metrics: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_artifacts: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        The checkpoint is copied into *artifact_store* under a versioned folder.
        Additional artifacts (ONNX, TensorRT, etc.) can be supplied via
        *extra_artifacts* (role -> path).
        """
        version_id = self._generate_version_id(run_name)
        version_dir = self.artifact_store / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        artifacts: Dict[str, str] = {}

        # Copy primary checkpoint
        src_ckpt = Path(checkpoint_path)
        if not src_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {src_ckpt}")
        dst_ckpt = version_dir / src_ckpt.name
        shutil.copy2(src_ckpt, dst_ckpt)
        artifacts["checkpoint"] = str(dst_ckpt)

        # Copy extra artifacts
        if extra_artifacts:
            for role, path in extra_artifacts.items():
                src = Path(path)
                if not src.exists():
                    logger.warning("Artifact '%s' not found at %s", role, src)
                    continue
                dst = version_dir / f"{role}_{src.name}"
                shutil.copy2(src, dst)
                artifacts[role] = str(dst)

        version = ModelVersion(
            version_id=version_id,
            run_name=run_name,
            created_at=time.time(),
            git_commit=git_commit or self._get_git_commit(),
            dataset_version=dataset_version,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            tags=tags or [],
            parent_version_id=parent_version_id,
            artifacts=artifacts,
        )

        try:
            self._backend.insert(version)
        except Exception as exc:
            self._alert.send(
                subject=f"Registry insert failed for {version_id}",
                body=str(exc),
            )
            raise

        logger.info("Registered model version %s (%s)", version_id, run_name)
        return version

    def get(self, version_id: str) -> Optional[ModelVersion]:
        """Retrieve a version by ID."""
        return self._backend.get(version_id)

    def list_versions(
        self,
        run_name: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelVersion]:
        """List versions, optionally filtered."""
        return self._backend.list_versions(run_name, tag, limit)

    def promote(self, version_id: str, tag: str) -> None:
        """Tag a version (e.g. 'production', 'staging')."""
        version = self.get(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")
        self._backend.add_tag(version_id, tag)
        logger.info("Promoted %s -> tag '%s'", version_id, tag)

    def demote(self, version_id: str, tag: str) -> None:
        """Remove a tag from a version."""
        self._backend.remove_tag(version_id, tag)
        logger.info("Demoted %s -> removed tag '%s'", version_id, tag)

    def lineage(self, version_id: str) -> List[ModelVersion]:
        """Return ancestor chain from *version_id* back to root."""
        chain: List[ModelVersion] = []
        current_id: Optional[str] = version_id
        while current_id:
            version = self.get(current_id)
            if version is None:
                break
            chain.append(version)
            current_id = version.parent_version_id
        return list(reversed(chain))

    def best_by_metric(self, run_name: str, metric_key: str, top_k: int = 1) -> List[ModelVersion]:
        """Return top-k versions for *run_name* ordered by *metric_key* descending."""
        versions = self.list_versions(run_name=run_name, limit=1000)
        scored = [(v, v.metrics.get(metric_key, float("-inf"))) for v in versions]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in scored[:top_k]]

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _generate_version_id(run_name: str) -> str:
        unique = f"{run_name}-{uuid.uuid4().hex[:8]}-{time.time():.0f}"
        return hashlib.sha256(unique.encode()).hexdigest()[:16]

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Optional MLflow backend (import guarded so MLflow is not a hard dep)
# ---------------------------------------------------------------------------

class _MlflowBackend:
    """Delegate storage to MLflow Tracking."""

    def __init__(self, tracking_uri: str) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("MLflow backend requires 'mlflow' package") from exc

        mlflow.set_tracking_uri(tracking_uri)
        self._mlflow = mlflow
        self._client = mlflow.tracking.MlflowClient()

    def insert(self, version: ModelVersion) -> None:
        with self._mlflow.start_run(run_name=version.run_name):
            self._mlflow.log_params(version.hyperparameters)
            self._mlflow.log_metrics(version.metrics)
            self._mlflow.set_tag("version_id", version.version_id)
            self._mlflow.set_tag("parent_version_id", version.parent_version_id or "")
            self._mlflow.set_tag("dataset_version", version.dataset_version or "")
            self._mlflow.set_tag("git_commit", version.git_commit or "")
            for role, path in version.artifacts.items():
                self._mlflow.log_artifact(path, artifact_path=role)
            for tag in version.tags:
                self._mlflow.set_tag(tag, "true")

    def get(self, version_id: str) -> Optional[ModelVersion]:
        # MLflow search is run-centric; we tag version_id for lookup
        runs = self._client.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.version_id = '{version_id}'",
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return self._run_to_version(run)

    def list_versions(
        self,
        run_name: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelVersion]:
        filter_parts = []
        if run_name:
            filter_parts.append(f"tags.`run_name` = '{run_name}'")
        if tag:
            filter_parts.append(f"tags.`{tag}` = 'true'")
        filter_string = " AND ".join(filter_parts) if filter_parts else ""
        runs = self._client.search_runs(
            experiment_ids=["0"],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )
        return [self._run_to_version(r) for r in runs]

    def add_tag(self, version_id: str, tag: str) -> None:
        runs = self._client.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.version_id = '{version_id}'",
            max_results=1,
        )
        if runs:
            self._client.set_tag(runs[0].info.run_id, tag, "true")

    def remove_tag(self, version_id: str, tag: str) -> None:
        runs = self._client.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.version_id = '{version_id}'",
            max_results=1,
        )
        if runs:
            # MLflow does not support tag deletion via client; we overwrite
            self._client.set_tag(runs[0].info.run_id, tag, "")

    def _run_to_version(self, run: Any) -> ModelVersion:
        info = run.info
        data = run.data
        return ModelVersion(
            version_id=data.tags.get("version_id", info.run_id),
            run_name=info.run_name or "unknown",
            created_at=info.start_time / 1000.0,
            git_commit=data.tags.get("git_commit") or None,
            dataset_version=data.tags.get("dataset_version") or None,
            hyperparameters=dict(data.params),
            metrics=dict(data.metrics),
            tags=[k for k, v in data.tags.items() if v == "true"],
            parent_version_id=data.tags.get("parent_version_id") or None,
            artifacts={},  # Could be fetched via list_artifacts if needed
        )
