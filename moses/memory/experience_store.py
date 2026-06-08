"""
moses/memory/experience_store.py
Experience Replay Database for Moses v4.0

Persistent storage for hyperparameters, architectures, environment configs,
and their outcomes. Supports similarity-based retrieval via vector embeddings.

Example queries (see bottom of file for runnable examples):
    >>> from moses.memory.experience_store import ExperienceStore
    >>> store = ExperienceStore("/tmp/moses_experience.db")
    >>> # Record an experiment
    >>> store.record(
    ...     experiment_id="exp_001",
    ...     hyperparams={"lr": 3e-4, "batch_size": 256, "entropy_coef": 0.01},
    ...     architecture={"type": "transformer", "layers": 4, "heads": 8},
    ...     env_config={"robot": "humanoid_28dof", "terrain": "flat"},
    ...     metrics={"reward_mean": 8420.0, "success_rate": 0.91, "training_hours": 12.5},
    ...     tags=["ppo", "humanoid"]
    ... )
    >>> # Query similar experiments
    >>> results = store.query_similar(
    ...     hyperparams={"lr": 3e-4, "batch_size": 256},
    ...     top_k=5
    ... )
    >>> # Retrieve best configs for a robot
    >>> best = store.get_best_for_env("humanoid_28dof", metric="reward_mean", top_k=3)
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Single experiment outcome."""
    experiment_id: str
    hyperparams: Dict[str, Any]
    architecture: Dict[str, Any]
    env_config: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    timestamp: str
    embedding: Optional[np.ndarray] = None


class ExperienceStore:
    """
    SQLite-backed experience store with optional FAISS/Annoy vector index
    for similarity search.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        hyperparams   TEXT NOT NULL,   -- JSON
        architecture  TEXT NOT NULL,   -- JSON
        env_config    TEXT NOT NULL,   -- JSON
        metrics       TEXT NOT NULL,   -- JSON
        tags          TEXT NOT NULL,   -- JSON list
        timestamp     TEXT NOT NULL,
        hp_hash       TEXT NOT NULL,   -- for fast exact-match lookups
        arch_hash     TEXT NOT NULL,
        env_hash      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hp_hash   ON experiments(hp_hash);
    CREATE INDEX IF NOT EXISTS idx_arch_hash ON experiments(arch_hash);
    CREATE INDEX IF NOT EXISTS idx_env_hash  ON experiments(env_hash);
    CREATE INDEX IF NOT EXISTS idx_tags      ON experiments(tags);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp);

    CREATE TABLE IF NOT EXISTS embeddings (
        experiment_id TEXT PRIMARY KEY,
        embedding     BLOB NOT NULL,   -- np.ndarray bytes
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            ON DELETE CASCADE
    );
    """

    def __init__(self, db_path: Union[str, Path], vector_dim: int = 128):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    # ------------------------------------------------------------------ #
    # Connection helpers
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_tables(self) -> None:
        conn = self._connect()
        conn.executescript(self.SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def record(self, experiment_id: str,
               hyperparams: Dict[str, Any],
               architecture: Dict[str, Any],
               env_config: Dict[str, Any],
               metrics: Dict[str, float],
               tags: Optional[List[str]] = None,
               timestamp: Optional[str] = None) -> None:
        """Persist a new experiment result."""
        tags = tags or []
        timestamp = timestamp or datetime.utcnow().isoformat()
        conn = self._connect()

        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
            (experiment_id, hyperparams, architecture, env_config,
             metrics, tags, timestamp, hp_hash, arch_hash, env_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                json.dumps(hyperparams, sort_keys=True),
                json.dumps(architecture, sort_keys=True),
                json.dumps(env_config, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                json.dumps(tags, sort_keys=True),
                timestamp,
                self._hash_dict(hyperparams),
                self._hash_dict(architecture),
                self._hash_dict(env_config),
            ),
        )

        # Derive and store embedding
        emb = self._embed(hyperparams, architecture, env_config)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (experiment_id, embedding) VALUES (?, ?)",
            (experiment_id, emb.tobytes()),
        )
        conn.commit()
        logger.info("Recorded experiment %s", experiment_id)

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve a single experiment by ID."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def query(self,
              hyperparams: Optional[Dict[str, Any]] = None,
              architecture: Optional[Dict[str, Any]] = None,
              env_config: Optional[Dict[str, Any]] = None,
              tags: Optional[List[str]] = None,
              limit: int = 100) -> List[ExperimentRecord]:
        """Structured query with exact-match filtering."""
        conn = self._connect()
        conditions: List[str] = []
        params: List[Any] = []

        if hyperparams is not None:
            conditions.append("hp_hash = ?")
            params.append(self._hash_dict(hyperparams))
        if architecture is not None:
            conditions.append("arch_hash = ?")
            params.append(self._hash_dict(architecture))
        if env_config is not None:
            conditions.append("env_hash = ?")
            params.append(self._hash_dict(env_config))
        if tags:
            # Fallback: LIKE match on JSON string (works without JSON1 extension)
            for t in tags:
                conditions.append("tags LIKE '%' || ? || '%'")
                params.append(t)

        where = " AND ".join(conditions) if conditions else "1=1"

        rows = conn.execute(
            f"SELECT * FROM experiments WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_similar(self,
                      hyperparams: Optional[Dict[str, Any]] = None,
                      architecture: Optional[Dict[str, Any]] = None,
                      env_config: Optional[Dict[str, Any]] = None,
                      top_k: int = 5) -> List[Tuple[ExperimentRecord, float]]:
        """
        Similarity search via cosine distance over stored embeddings.
        Returns list of (record, distance) sorted ascending.
        """
        target = self._embed(hyperparams or {}, architecture or {}, env_config or {})
        conn = self._connect()
        rows = conn.execute(
            "SELECT e.*, em.embedding FROM experiments e "
            "JOIN embeddings em ON e.experiment_id = em.experiment_id"
        ).fetchall()

        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.shape[0] != self.vector_dim:
                continue
            dist = float(self._cosine_distance(target, emb))
            scored.append((dist, row))

        scored.sort(key=lambda x: x[0])
        return [(self._row_to_record(r), d) for d, r in scored[:top_k]]

    def get_best_for_env(self,
                         robot: str,
                         metric: str = "reward_mean",
                         top_k: int = 5) -> List[ExperimentRecord]:
        """
        Return experiments for a given robot sorted by a metric descending.
        Example:
            >>> store.get_best_for_env("humanoid_28dof", metric="success_rate", top_k=3)
        """
        conn = self._connect()
        # env_config is JSON; we use LIKE for simple key-value match.
        rows = conn.execute(
            "SELECT * FROM experiments WHERE env_config LIKE ? ORDER BY timestamp DESC",
            (f'%"robot": "{robot}"%',),
        ).fetchall()

        def metric_score(r: sqlite3.Row) -> float:
            m = json.loads(r["metrics"])
            return m.get(metric, float("-inf"))

        rows_sorted = sorted(rows, key=metric_score, reverse=True)
        return [self._row_to_record(r) for r in rows_sorted[:top_k]]

    def delete(self, experiment_id: str) -> bool:
        conn = self._connect()
        cur = conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        conn.commit()
        return cur.rowcount > 0

    def stats(self) -> Dict[str, Any]:
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        with_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"total_experiments": total, "experiments_with_embeddings": with_emb}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hash_dict(d: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def _embed(self, hp: Dict[str, Any], arch: Dict[str, Any], env: Dict[str, Any]) -> np.ndarray:
        """
        Deterministic embedding from JSON-serialized config.
        In production, swap for a learned encoder (e.g. small MLP or sentence-transformer).
        """
        text = json.dumps({"hp": hp, "arch": arch, "env": env}, sort_keys=True)
        # Simple hash-based projection for determinism without external deps
        rng = np.random.RandomState(int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31))
        vec = rng.randn(self.vector_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b))

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            hyperparams=json.loads(row["hyperparams"]),
            architecture=json.loads(row["architecture"]),
            env_config=json.loads(row["env_config"]),
            metrics=json.loads(row["metrics"]),
            tags=json.loads(row["tags"]),
            timestamp=row["timestamp"],
        )


# ---------------------------------------------------------------------- #
# Example / self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as td:
        store = ExperienceStore(Path(td) / "exp.db", vector_dim=64)

        # Seed data
        for i in range(10):
            store.record(
                experiment_id=f"exp_{i:03d}",
                hyperparams={"lr": 3e-4 * (1 + i * 0.1), "batch_size": 256},
                architecture={"type": "mlp", "layers": 3 + i % 3},
                env_config={"robot": "humanoid_28dof" if i % 2 == 0 else "quadruped_12dof", "terrain": "flat"},
                metrics={"reward_mean": 5000 + i * 400, "success_rate": 0.7 + i * 0.02},
                tags=["ppo"],
            )

        print("Stats:", store.stats())

        # Exact query
        exact = store.query(hyperparams={"lr": 3e-4, "batch_size": 256})
        print("Exact matches:", len(exact))

        # Similarity query
        sim = store.query_similar(hyperparams={"lr": 3.3e-4, "batch_size": 256}, top_k=3)
        print("Similar experiments:", [(r.experiment_id, f"{d:.4f}") for r, d in sim])

        # Best for robot
        best = store.get_best_for_env("humanoid_28dof", metric="reward_mean", top_k=3)
        print("Best humanoid configs:", [r.experiment_id for r in best])
