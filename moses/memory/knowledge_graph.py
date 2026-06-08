"""
moses/memory/knowledge_graph.py
Structured Knowledge Graph for Moses v4.0

Nodes: concepts (algorithms, hyperparameters, robot designs, tasks)
Edges: relationships (improves, degrades, compatible_with, requires, is_a)

Auto-updates from experiment results via the experience store.
Supports natural-language-style querying via graph traversal.

Example queries:
    >>> from moses.memory.knowledge_graph import KnowledgeGraph
    >>> kg = KnowledgeGraph("/tmp/moses_kg.db")
    >>> kg.add_experiment_result(
    ...     experiment_id="exp_001",
    ...     hyperparams={"lr": 3e-4, "optimizer": "adam"},
    ...     architecture={"type": "transformer", "layers": 4},
    ...     env_config={"robot": "humanoid_28dof"},
    ...     metrics={"reward_mean": 9000, "success_rate": 0.95},
    ...     baseline_metrics={"reward_mean": 7000, "success_rate": 0.80},
    ... )
    >>> # Query: "What learning rate works best for 28-DOF humanoids?"
    >>> answer = kg.query_best_hyperparam(
    ...     robot="humanoid_28dof",
    ...     hyperparam="lr",
    ...     metric="reward_mean"
    ... )
    >>> print(answer)
    >>> # Query compatibility
    >>> kg.are_compatible("transformer", "humanoid_28dof")
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class Node:
    node_id: str
    node_type: str          # e.g. "hyperparam", "algorithm", "robot", "metric", "task"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str           # improves, degrades, compatible_with, requires, is_a
    weight: float = 1.0     # confidence / strength
    evidence_count: int = 1
    last_updated: str = ""


class KnowledgeGraph:
    """
    SQLite-backed property graph with auto-extraction from experiment records.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS nodes (
        node_id   TEXT PRIMARY KEY,
        node_type TEXT NOT NULL,
        name      TEXT NOT NULL,
        properties TEXT NOT NULL DEFAULT '{}'  -- JSON
    );
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
    CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

    CREATE TABLE IF NOT EXISTS edges (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        relation TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        evidence_count INTEGER NOT NULL DEFAULT 1,
        last_updated TEXT NOT NULL,
        PRIMARY KEY (source, target, relation)
    );
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
    CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);

    CREATE TABLE IF NOT EXISTS node_embeddings (
        node_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        FOREIGN KEY (node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
    );
    """

    def __init__(self, db_path: Union[str, Path], vector_dim: int = 64):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

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
        self._connect().executescript(self.SCHEMA)
        self._connect().commit()

    # ------------------------------------------------------------------ #
    # Node / Edge CRUD
    # ------------------------------------------------------------------ #
    def add_node(self, node_id: str, node_type: str, name: str,
                 properties: Optional[Dict[str, Any]] = None,
                 embedding: Optional[np.ndarray] = None) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO nodes (node_id, node_type, name, properties)
            VALUES (?, ?, ?, ?)
            """,
            (node_id, node_type, name, json.dumps(properties or {}, sort_keys=True)),
        )
        if embedding is not None:
            emb = embedding.astype(np.float32)
            conn.execute(
                "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
                (node_id, emb.tobytes()),
            )
        conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        row = self._connect().execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return Node(
            node_id=row["node_id"],
            node_type=row["node_type"],
            name=row["name"],
            properties=json.loads(row["properties"]),
        )

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0, evidence_delta: int = 1) -> None:
        conn = self._connect()
        now = datetime.utcnow().isoformat()
        # Upsert: increment evidence_count and update weight as weighted average
        existing = conn.execute(
            "SELECT weight, evidence_count FROM edges WHERE source=? AND target=? AND relation=?",
            (source, target, relation),
        ).fetchone()
        if existing:
            old_w = existing["weight"]
            old_n = existing["evidence_count"]
            new_n = old_n + evidence_delta
            new_w = (old_w * old_n + weight * evidence_delta) / new_n
            conn.execute(
                """
                UPDATE edges SET weight=?, evidence_count=?, last_updated=?
                WHERE source=? AND target=? AND relation=?
                """,
                (new_w, new_n, now, source, target, relation),
            )
        else:
            conn.execute(
                """
                INSERT INTO edges (source, target, relation, weight, evidence_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, target, relation, weight, evidence_delta, now),
            )
        conn.commit()

    def get_edges(self, source: Optional[str] = None,
                  target: Optional[str] = None,
                  relation: Optional[str] = None) -> List[Edge]:
        conn = self._connect()
        conditions = []
        params = []
        if source:
            conditions.append("source = ?")
            params.append(source)
        if target:
            conditions.append("target = ?")
            params.append(target)
        if relation:
            conditions.append("relation = ?")
            params.append(relation)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(f"SELECT * FROM edges WHERE {where}", params).fetchall()
        return [
            Edge(
                source=r["source"],
                target=r["target"],
                relation=r["relation"],
                weight=r["weight"],
                evidence_count=r["evidence_count"],
                last_updated=r["last_updated"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Auto-population from experiments
    # ------------------------------------------------------------------ #
    def add_experiment_result(
        self,
        experiment_id: str,
        hyperparams: Dict[str, Any],
        architecture: Dict[str, Any],
        env_config: Dict[str, Any],
        metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Ingest an experiment result, creating nodes and deriving edges.
        If baseline_metrics provided, creates improves/degrades edges.
        """
        tags = tags or []
        now = datetime.utcnow().isoformat()

        # --- Nodes ---
        # Architecture node
        arch_name = architecture.get("type", "unknown")
        arch_id = f"arch:{arch_name}"
        self.add_node(arch_id, "algorithm", arch_name, architecture)

        # Robot / env node
        robot = env_config.get("robot", "unknown")
        robot_id = f"robot:{robot}"
        self.add_node(robot_id, "robot", robot, env_config)

        # Hyperparameter nodes
        for k, v in hyperparams.items():
            hp_id = f"hp:{k}={v}"
            self.add_node(hp_id, "hyperparam", f"{k}={v}", {"key": k, "value": v})
            # Link hyperparam to architecture
            self.add_edge(hp_id, arch_id, "configures")

        # Metric nodes
        for m_name, m_val in metrics.items():
            metric_id = f"metric:{m_name}"
            self.add_node(metric_id, "metric", m_name, {})
            # Link architecture -> metric (value as edge weight proxy)
            # Normalize weight roughly to 0..1 for reward-like metrics
            weight = float(np.tanh(m_val / 1e4)) if isinstance(m_val, (int, float)) else 0.5
            self.add_edge(arch_id, metric_id, "achieves", weight=weight)

        # Robot compatibility
        self.add_edge(arch_id, robot_id, "compatible_with", weight=0.5)

        # --- Baseline comparison edges ---
        if baseline_metrics:
            for m_name, m_val in metrics.items():
                base_val = baseline_metrics.get(m_name)
                if base_val is None or base_val == 0:
                    continue
                delta = (m_val - base_val) / (abs(base_val) + 1e-6)
                metric_id = f"metric:{m_name}"
                if delta > 0.05:
                    self.add_edge(arch_id, metric_id, "improves", weight=min(delta, 1.0))
                elif delta < -0.05:
                    self.add_edge(arch_id, metric_id, "degrades", weight=min(abs(delta), 1.0))

        # Tags -> is_a
        for t in tags:
            tag_id = f"tag:{t}"
            self.add_node(tag_id, "task", t, {})
            self.add_edge(arch_id, tag_id, "is_a", weight=1.0)

        logger.info("Ingested experiment %s into knowledge graph", experiment_id)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def query_best_hyperparam(
        self,
        robot: str,
        hyperparam: str,
        metric: str = "reward_mean",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Answer: "What value of <hyperparam> works best for <robot> on <metric>?"
        Returns list of {value, score, evidence} sorted by score descending.
        """
        robot_id = f"robot:{robot}"
        # Find architectures linked to this robot
        arch_edges = self.get_edges(target=robot_id, relation="compatible_with")
        arch_ids = [e.source for e in arch_edges]

        # For each architecture, find hyperparam configs and metric achievements
        results: List[Dict[str, Any]] = []
        for arch_id in arch_ids:
            # configs edges: hp -> arch
            hp_edges = self.get_edges(target=arch_id, relation="configures")
            for hp_edge in hp_edges:
                hp_node = self.get_node(hp_edge.source)
                if hp_node is None or hp_node.properties.get("key") != hyperparam:
                    continue
                hp_value = hp_node.properties.get("value")
                # Metric achievement
                metric_edges = self.get_edges(source=arch_id, relation="achieves")
                for me in metric_edges:
                    if me.target == f"metric:{metric}":
                        results.append({
                            "value": hp_value,
                            "score": me.weight,
                            "evidence": me.evidence_count,
                            "architecture": self.get_node(arch_id).name if self.get_node(arch_id) else arch_id,
                        })

        # Deduplicate by value, keeping highest score
        best: Dict[Any, Dict[str, Any]] = {}
        for r in results:
            v = r["value"]
            if v not in best or r["score"] > best[v]["score"]:
                best[v] = r
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def query_compatible_algorithms(self, robot: str, min_weight: float = 0.0) -> List[Dict[str, Any]]:
        """Return algorithms marked compatible with a robot."""
        robot_id = f"robot:{robot}"
        edges = self.get_edges(target=robot_id, relation="compatible_with")
        return [
            {
                "algorithm": self.get_node(e.source).name if self.get_node(e.source) else e.source,
                "weight": e.weight,
                "evidence": e.evidence_count,
            }
            for e in edges
            if e.weight >= min_weight
        ]

    def query_what_improves(self, metric: str, min_weight: float = 0.1) -> List[Dict[str, Any]]:
        """Return architectures / configs that improve a given metric."""
        metric_id = f"metric:{metric}"
        edges = self.get_edges(target=metric_id, relation="improves")
        return [
            {
                "source": self.get_node(e.source).name if self.get_node(e.source) else e.source,
                "improvement": e.weight,
                "evidence": e.evidence_count,
            }
            for e in edges
            if e.weight >= min_weight
        ]

    def are_compatible(self, algorithm: str, robot: str) -> Tuple[bool, float]:
        """Check compatibility and return confidence."""
        arch_id = f"arch:{algorithm}"
        robot_id = f"robot:{robot}"
        edges = self.get_edges(source=arch_id, target=robot_id, relation="compatible_with")
        if not edges:
            return False, 0.0
        return True, edges[0].weight

    def traverse(self, start_node: str, relation: str, depth: int = 2) -> Dict[str, Any]:
        """
        BFS traversal from start_node following a specific relation.
        Returns adjacency-style dict.
        """
        visited: Set[str] = set()
        frontier = {start_node}
        levels: List[List[str]] = []
        for _ in range(depth):
            next_frontier: Set[str] = set()
            level_nodes = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                level_nodes.append(node)
                edges = self.get_edges(source=node, relation=relation)
                for e in edges:
                    next_frontier.add(e.target)
            levels.append(level_nodes)
            frontier = next_frontier
        return {"start": start_node, "relation": relation, "depth": depth, "levels": levels}

    # ------------------------------------------------------------------ #
    # Similarity search on nodes
    # ------------------------------------------------------------------ #
    def find_similar_nodes(self, node_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Cosine similarity over node embeddings."""
        conn = self._connect()
        row = conn.execute("SELECT embedding FROM node_embeddings WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return []
        target = np.frombuffer(row["embedding"], dtype=np.float32)
        if target.shape[0] != self.vector_dim:
            return []

        all_rows = conn.execute("SELECT node_id, embedding FROM node_embeddings").fetchall()
        scored = []
        for r in all_rows:
            if r["node_id"] == node_id:
                continue
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            if emb.shape[0] != self.vector_dim:
                continue
            sim = float(np.dot(target, emb))
            scored.append((r["node_id"], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def stats(self) -> Dict[str, int]:
        conn = self._connect()
        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"nodes": n_nodes, "edges": n_edges}


# ---------------------------------------------------------------------- #
# Example / self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as td:
        kg = KnowledgeGraph(Path(td) / "kg.db", vector_dim=32)

        # Simulate ingestion
        kg.add_experiment_result(
            experiment_id="exp_001",
            hyperparams={"lr": 3e-4, "batch_size": 256},
            architecture={"type": "transformer", "layers": 4, "heads": 8},
            env_config={"robot": "humanoid_28dof", "terrain": "flat"},
            metrics={"reward_mean": 9200, "success_rate": 0.95},
            baseline_metrics={"reward_mean": 7000, "success_rate": 0.80},
            tags=["ppo"],
        )
        kg.add_experiment_result(
            experiment_id="exp_002",
            hyperparams={"lr": 1e-3, "batch_size": 128},
            architecture={"type": "mlp", "layers": 3},
            env_config={"robot": "humanoid_28dof", "terrain": "flat"},
            metrics={"reward_mean": 6500, "success_rate": 0.78},
            baseline_metrics={"reward_mean": 7000, "success_rate": 0.80},
            tags=["ppo"],
        )

        print("KG stats:", kg.stats())

        # Query best LR for humanoid
        best_lr = kg.query_best_hyperparam("humanoid_28dof", "lr", "reward_mean")
        print("Best LR for humanoid:", best_lr)

        # Compatible algorithms
        compat = kg.query_compatible_algorithms("humanoid_28dof")
        print("Compatible algorithms:", compat)

        # What improves reward_mean?
        improvers = kg.query_what_improves("reward_mean")
        print("Improvers:", improvers)

        # Compatibility check
        ok, conf = kg.are_compatible("transformer", "humanoid_28dof")
        print(f"transformer compatible with humanoid_28dof? {ok} (confidence={conf:.2f})")

        # Traversal
        tr = kg.traverse("arch:transformer", "compatible_with", depth=2)
        print("Traversal:", tr)
