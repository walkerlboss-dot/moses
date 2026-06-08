"""
moses/memory/transfer_learning.py
Cross-Task Knowledge Transfer for Moses v4.0

Extracts general principles from specific tasks and applies them to new tasks.
Measures transferability of knowledge across domains (walking -> running -> manipulation).

Example queries:
    >>> from moses.memory.transfer_learning import TransferEngine
    >>> engine = TransferEngine(experience_store, knowledge_graph)
    >>> # Extract principles from walking experiments
    >>> principles = engine.extract_principles(source_task="walking", min_evidence=3)
    >>> # Apply to running
    >>> recs = engine.apply_principles(principles, target_task="running")
    >>> # Measure transferability score
    >>> score = engine.transferability("walking", "running")
    >>> print(f"Transferability: {score:.2f}")
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class Principle:
    """A transferable principle extracted from experiments."""
    principle_id: str
    description: str          # human-readable summary
    source_task: str
    hyperparam_pattern: Dict[str, Any]
    architecture_pattern: Dict[str, Any]
    outcome_metric: str
    outcome_direction: str    # "increase" | "decrease"
    effect_size: float
    confidence: float
    evidence_count: int
    scope: List[str] = field(default_factory=list)  # tasks where observed


@dataclass
class TransferRecommendation:
    """A concrete recommendation for a target task."""
    target_task: str
    recommended_hyperparams: Dict[str, Any]
    recommended_architecture: Dict[str, Any]
    expected_metric: str
    expected_improvement: float
    confidence: float
    rationale: str


class TransferEngine:
    """
    Learns what transfers across tasks and quantifies transferability.
    Backed by SQLite for persistence.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS principles (
        principle_id       TEXT PRIMARY KEY,
        description        TEXT NOT NULL,
        source_task        TEXT NOT NULL,
        hyperparam_pattern TEXT NOT NULL,  -- JSON
        architecture_pattern TEXT NOT NULL, -- JSON
        outcome_metric     TEXT NOT NULL,
        outcome_direction  TEXT NOT NULL,
        effect_size        REAL NOT NULL,
        confidence         REAL NOT NULL,
        evidence_count     INTEGER NOT NULL,
        scope              TEXT NOT NULL,  -- JSON list
        created_at         TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_principles_task ON principles(source_task);

    CREATE TABLE IF NOT EXISTS transfer_scores (
        source_task TEXT NOT NULL,
        target_task TEXT NOT NULL,
        score       REAL NOT NULL,
        evidence    INTEGER NOT NULL,
        last_updated TEXT NOT NULL,
        PRIMARY KEY (source_task, target_task)
    );
    """

    def __init__(self,
                 experience_store: Any,
                 knowledge_graph: Any,
                 db_path: Optional[Union[str, Path]] = None):
        self.store = experience_store
        self.kg = knowledge_graph
        self.db_path = db_path
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()
        else:
            self._conn = None

    # ------------------------------------------------------------------ #
    # Principle extraction
    # ------------------------------------------------------------------ #
    def extract_principles(
        self,
        source_task: str,
        min_evidence: int = 3,
        metric: str = "reward_mean",
    ) -> List[Principle]:
        """
        Mine the experience store for recurring patterns in <source_task>
        that reliably improve <metric>.
        """
        # Pull all experiments tagged with source_task
        records = self.store.query(tags=[source_task], limit=10000)
        if len(records) < min_evidence:
            logger.warning("Not enough records for task %s (%d < %d)", source_task, len(records), min_evidence)
            return []

        # Group by hyperparam + architecture signature
        groups: Dict[str, List[Any]] = {}
        for r in records:
            sig = json.dumps({"hp": r.hyperparams, "arch": r.architecture}, sort_keys=True)
            groups.setdefault(sig, []).append(r)

        principles: List[Principle] = []
        for sig, recs in groups.items():
            if len(recs) < min_evidence:
                continue
            vals = [r.metrics.get(metric, np.nan) for r in recs]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) < min_evidence:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            # Confidence = inverse CV, clamped
            confidence = float(1.0 / (1.0 + std_val / (abs(mean_val) + 1e-6)))
            # Compare to task baseline (median of all task records)
            all_vals = [r.metrics.get(metric, np.nan) for r in records]
            all_vals = [v for v in all_vals if not np.isnan(v)]
            baseline = float(np.median(all_vals))
            effect = (mean_val - baseline) / (abs(baseline) + 1e-6)
            if abs(effect) < 0.02:
                continue  # not a meaningful principle

            direction = "increase" if effect > 0 else "decrease"
            parsed = json.loads(sig)
            principle = Principle(
                principle_id=f"principle:{source_task}:{hashlib.sha256(sig.encode()).hexdigest()[:16]}",
                description=f"Config improves {metric} by {effect*100:.1f}% on {source_task}",
                source_task=source_task,
                hyperparam_pattern=parsed["hp"],
                architecture_pattern=parsed["arch"],
                outcome_metric=metric,
                outcome_direction=direction,
                effect_size=effect,
                confidence=confidence,
                evidence_count=len(vals),
                scope=[source_task],
            )
            principles.append(principle)
            self._persist_principle(principle)

        # Sort by confidence * effect_size
        principles.sort(key=lambda p: p.confidence * abs(p.effect_size), reverse=True)
        return principles

    def _persist_principle(self, p: Principle) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO principles
            (principle_id, description, source_task, hyperparam_pattern, architecture_pattern,
             outcome_metric, outcome_direction, effect_size, confidence, evidence_count, scope, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.principle_id, p.description, p.source_task,
                json.dumps(p.hyperparam_pattern, sort_keys=True),
                json.dumps(p.architecture_pattern, sort_keys=True),
                p.outcome_metric, p.outcome_direction, p.effect_size,
                p.confidence, p.evidence_count, json.dumps(p.scope, sort_keys=True),
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Principle application
    # ------------------------------------------------------------------ #
    def apply_principles(
        self,
        principles: List[Principle],
        target_task: str,
        target_env_hint: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[TransferRecommendation]:
        """
        Adapt extracted principles to a new target task.
        Uses similarity between source and target env configs to weight recommendations.
        """
        # Pull target task baseline from KG or store
        target_records = self.store.query(tags=[target_task])
        baseline_metric = "reward_mean"
        baseline_val = 0.0
        if target_records:
            vals = [r.metrics.get(baseline_metric, np.nan) for r in target_records]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                baseline_val = float(np.median(vals))

        recommendations: List[TransferRecommendation] = []
        for p in principles:
            # Simple adaptation: keep hyperparams, optionally adjust LR by task similarity heuristic
            adapted_hp = dict(p.hyperparam_pattern)
            adapted_arch = dict(p.architecture_pattern)

            # Heuristic: if target is more complex (e.g. more DOF), scale LR down
            if target_env_hint:
                source_dof = self._extract_dof(p.source_task)
                target_dof = self._extract_dof(target_task)
                if target_dof > source_dof and "lr" in adapted_hp:
                    adapted_hp["lr"] = float(adapted_hp["lr"]) * (source_dof / target_dof)

            expected_improvement = p.effect_size * p.confidence
            rec = TransferRecommendation(
                target_task=target_task,
                recommended_hyperparams=adapted_hp,
                recommended_architecture=adapted_arch,
                expected_metric=p.outcome_metric,
                expected_improvement=expected_improvement,
                confidence=p.confidence,
                rationale=f"Transferred from {p.source_task} (effect={p.effect_size:.2f}, conf={p.confidence:.2f})",
            )
            recommendations.append(rec)

        recommendations.sort(key=lambda r: r.confidence * abs(r.expected_improvement), reverse=True)
        return recommendations[:top_k]

    @staticmethod
    def _extract_dof(task_name: str) -> int:
        """Naive DOF extraction from task name for heuristic scaling."""
        import re
        m = re.search(r'(\d+)dof', task_name.lower())
        return int(m.group(1)) if m else 12  # default moderate DOF

    # ------------------------------------------------------------------ #
    # Transferability measurement
    # ------------------------------------------------------------------ #
    def transferability(self, source_task: str, target_task: str,
                        metric: str = "reward_mean") -> float:
        """
        Compute a transferability score in [0, 1].
        High score = principles from source_task reliably help target_task.
        """
        # Check cache
        if self._conn:
            row = self._conn.execute(
                "SELECT score FROM transfer_scores WHERE source_task=? AND target_task=?",
                (source_task, target_task),
            ).fetchone()
            if row:
                return float(row[0])

        # 1. Extract principles from source
        principles = self.extract_principles(source_task, min_evidence=2, metric=metric)
        if not principles:
            self._cache_score(source_task, target_task, 0.0, 0)
            return 0.0

        # 2. See if target_task records exist that match these principles
        target_records = self.store.query(tags=[target_task])
        if not target_records:
            # No target data yet; estimate from env similarity
            score = self._estimate_transferability_from_similarity(source_task, target_task)
            self._cache_score(source_task, target_task, score, 0)
            return score

        # 3. Measure how often source principles align with top target performers
        matched = 0
        total = 0
        for p in principles:
            for r in target_records:
                total += 1
                # Check hyperparam overlap
                hp_overlap = sum(
                    1 for k, v in p.hyperparam_pattern.items()
                    if r.hyperparams.get(k) == v
                ) / max(len(p.hyperparam_pattern), 1)
                # Check architecture overlap
                arch_overlap = sum(
                    1 for k, v in p.architecture_pattern.items()
                    if r.architecture.get(k) == v
                ) / max(len(p.architecture_pattern), 1)
                overlap = (hp_overlap + arch_overlap) / 2.0
                if overlap > 0.5:
                    matched += 1

        score = matched / total if total > 0 else 0.0
        # Adjust by principle confidence
        avg_conf = float(np.mean([p.confidence for p in principles])) if principles else 0.0
        score = score * avg_conf
        self._cache_score(source_task, target_task, score, total)
        return score

    def _estimate_transferability_from_similarity(self, source_task: str, target_task: str) -> float:
        """Fallback: estimate transferability from task name / env similarity."""
        # Simple heuristic: shared substrings and DOF proximity
        s = source_task.lower()
        t = target_task.lower()
        shared = sum(1 for a, b in zip(s, t) if a == b) / max(len(s), len(t))
        dof_s = self._extract_dof(s)
        dof_t = self._extract_dof(t)
        dof_sim = 1.0 - abs(dof_s - dof_t) / max(dof_s + dof_t, 1)
        return float(np.clip((shared + dof_sim) / 2.0, 0.0, 1.0))

    def _cache_score(self, source: str, target: str, score: float, evidence: int) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO transfer_scores
            (source_task, target_task, score, evidence, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, target, score, evidence, datetime.utcnow().isoformat()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Cross-task recommendation API
    # ------------------------------------------------------------------ #
    def recommend_for_new_task(
        self,
        target_task: str,
        target_env_hint: Optional[Dict[str, Any]] = None,
        source_tasks: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[TransferRecommendation]:
        """
        Full pipeline: find best source tasks, extract principles, adapt,
        and return ranked recommendations.
        """
        if source_tasks is None:
            # Discover all tasks from store tags
            # Since store.query doesn't expose DISTINCT tags, we approximate
            # by pulling recent records and collecting tags.
            all_recs = self.store.query(limit=1000)
            source_tasks = list({t for r in all_recs for t in r.tags if t != target_task})

        # Rank source tasks by transferability
        scored_sources = [
            (src, self.transferability(src, target_task))
            for src in source_tasks
        ]
        scored_sources.sort(key=lambda x: x[1], reverse=True)

        all_recs: List[TransferRecommendation] = []
        for src, score in scored_sources[:5]:  # top-5 sources
            if score < 0.1:
                continue
            principles = self.extract_principles(src, min_evidence=2)
            adapted = self.apply_principles(principles, target_task, target_env_hint, top_k=top_k)
            for rec in adapted:
                rec.confidence *= score  # down-weight by transferability
                rec.rationale += f" | transferability={score:.2f}"
            all_recs.extend(adapted)

        all_recs.sort(key=lambda r: r.confidence * abs(r.expected_improvement), reverse=True)
        return all_recs[:top_k]

    def list_principles(self, source_task: Optional[str] = None) -> List[Principle]:
        """List persisted principles."""
        if self._conn is None:
            return []
        sql = "SELECT * FROM principles"
        params = ()
        if source_task:
            sql += " WHERE source_task = ?"
            params = (source_task,)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Principle(
                principle_id=r["principle_id"],
                description=r["description"],
                source_task=r["source_task"],
                hyperparam_pattern=json.loads(r["hyperparam_pattern"]),
                architecture_pattern=json.loads(r["architecture_pattern"]),
                outcome_metric=r["outcome_metric"],
                outcome_direction=r["outcome_direction"],
                effect_size=r["effect_size"],
                confidence=r["confidence"],
                evidence_count=r["evidence_count"],
                scope=json.loads(r["scope"]),
            )
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        out = {}
        if self._conn:
            out["principles"] = self._conn.execute("SELECT COUNT(*) FROM principles").fetchone()[0]
            out["transfer_scores"] = self._conn.execute("SELECT COUNT(*) FROM transfer_scores").fetchone()[0]
        return out


# ---------------------------------------------------------------------- #
# Example / self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import hashlib
    from pathlib import Path
    from moses.memory.experience_store import ExperienceStore
    from moses.memory.knowledge_graph import KnowledgeGraph

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as td:
        store = ExperienceStore(Path(td) / "exp.db", vector_dim=32)
        kg = KnowledgeGraph(Path(td) / "kg.db", vector_dim=32)
        engine = TransferEngine(store, kg, db_path=Path(td) / "transfer.db")

        # Seed walking experiments
        for i in range(5):
            store.record(
                f"walk_{i}",
                hyperparams={"lr": 3e-4, "batch_size": 256},
                architecture={"type": "transformer", "layers": 4},
                env_config={"robot": "walker_12dof"},
                metrics={"reward_mean": 8000 + np.random.randn() * 300},
                tags=["walking"],
            )
        # Seed running experiments (sparse)
        for i in range(2):
            store.record(
                f"run_{i}",
                hyperparams={"lr": 3e-4, "batch_size": 256},
                architecture={"type": "transformer", "layers": 4},
                env_config={"robot": "runner_12dof"},
                metrics={"reward_mean": 7500 + np.random.randn() * 200},
                tags=["running"],
            )

        # Extract principles
        principles = engine.extract_principles("walking", min_evidence=3)
        print("Extracted principles:", len(principles))
        for p in principles[:3]:
            print(" -", p.principle_id, p.description, f"conf={p.confidence:.2f}")

        # Apply to running
        if principles:
            recs = engine.apply_principles(principles, "running")
            print("Recommendations for running:")
            for r in recs:
                print(" -", r.rationale, f"expected={r.expected_improvement:.2f} conf={r.confidence:.2f}")

        # Transferability
        score = engine.transferability("walking", "running")
        print(f"Transferability walking->running: {score:.2f}")

        # Full recommendation pipeline
        full = engine.recommend_for_new_task("running", source_tasks=["walking"])
        print("Full recommendations:", len(full))
