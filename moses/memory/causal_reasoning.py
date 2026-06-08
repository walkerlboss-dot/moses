"""
moses/memory/causal_reasoning.py
Causal Inference Engine for Moses v4.0

Given a code change and a metric change, estimate whether the change caused
improvement or if it was coincidence.

Supported methods:
    - Difference-in-differences (DiD)  : compare treated vs control trajectories
    - Synthetic control                : weighted combination of donor experiments
    - Counterfactual regression        : predict "what would have happened"

Example queries:
    >>> from moses.memory.causal_reasoning import CausalEngine
    >>> engine = CausalEngine(experience_store)
    >>> result = engine.estimate_effect(
    ...     treatment={"lr": 3e-4},
    ...     outcome_metric="reward_mean",
    ...     control_query={"lr": 1e-3},
    ...     method="did"
    ... )
    >>> print(result.causal_effect, result.confidence)
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalEstimate:
    """Result of a causal inference query."""
    treatment: Dict[str, Any]
    outcome_metric: str
    causal_effect: float          # estimated delta attributable to treatment
    confidence: float             # 0..1 heuristic confidence
    method: str
    p_value: Optional[float] = None
    std_error: Optional[float] = None
    control_mean: Optional[float] = None
    treated_mean: Optional[float] = None
    notes: List[str] = field(default_factory=list)


class CausalEngine:
    """
    Lightweight causal inference over experiment records.
    No heavy ML dependencies; uses numpy + sqlite.
    """

    def __init__(self, experience_store: Any, db_path: Optional[Union[str, Path]] = None):
        """
        Args:
            experience_store: instance with `.query()` and `.query_similar()`
            db_path: optional separate SQLite for cached models / priors
        """
        self.store = experience_store
        self.db_path = db_path
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._ensure_cache_table()
        else:
            self._conn = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def estimate_effect(
        self,
        treatment: Dict[str, Any],
        outcome_metric: str,
        control_query: Optional[Dict[str, Any]] = None,
        treated_query: Optional[Dict[str, Any]] = None,
        method: str = "did",
        min_samples: int = 3,
    ) -> CausalEstimate:
        """
        Estimate causal effect of a treatment configuration on a metric.

        Args:
            treatment: the specific change being evaluated (e.g. {"lr": 3e-4})
            outcome_metric: metric name, e.g. "reward_mean"
            control_query: dict of filters for control group (default: everything else)
            treated_query: dict of filters for treated group (default: matches treatment)
            method: "did" | "synthetic_control" | "regression"
            min_samples: minimum records per arm

        Returns:
            CausalEstimate with effect size and confidence.
        """
        if method == "did":
            return self._did(treatment, outcome_metric, control_query, treated_query, min_samples)
        elif method == "synthetic_control":
            return self._synthetic_control(treatment, outcome_metric, control_query, treated_query, min_samples)
        elif method == "regression":
            return self._regression(treatment, outcome_metric, control_query, treated_query, min_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

    def did_from_records(
        self,
        treated_records: List[Any],
        control_records: List[Any],
        outcome_metric: str,
        pre_period_key: str = "timestamp",
    ) -> CausalEstimate:
        """
        Raw DiD given two lists of ExperimentRecord-like objects.
        Assumes records have `.metrics` dict and `.hyperparams` dict.
        """
        treated_outcomes = np.array([r.metrics[outcome_metric] for r in treated_records if outcome_metric in r.metrics])
        control_outcomes = np.array([r.metrics[outcome_metric] for r in control_records if outcome_metric in r.metrics])

        if len(treated_outcomes) < 2 or len(control_outcomes) < 2:
            return CausalEstimate(
                treatment={},
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="did",
                notes=["Insufficient samples for DiD"],
            )

        # Simple DiD: treated_mean - control_mean (assuming parallel trends heuristically)
        effect = float(np.mean(treated_outcomes) - np.mean(control_outcomes))
        pooled_std = float(np.sqrt(np.var(treated_outcomes, ddof=1) + np.var(control_outcomes, ddof=1)))
        se = pooled_std / np.sqrt(len(treated_outcomes) + len(control_outcomes)) if pooled_std > 0 else 1e-6
        # Heuristic t-statistic
        t_stat = effect / se if se > 0 else 0.0
        # Approximate p-value from t-stat (very rough, assumes normal)
        p_value = float(np.exp(-0.5 * t_stat ** 2))  # crude surrogate
        confidence = float(1.0 - min(p_value, 1.0))

        return CausalEstimate(
            treatment={},
            outcome_metric=outcome_metric,
            causal_effect=effect,
            confidence=confidence,
            method="did",
            p_value=p_value,
            std_error=se,
            control_mean=float(np.mean(control_outcomes)),
            treated_mean=float(np.mean(treated_outcomes)),
            notes=[f"DiD over {len(treated_outcomes)} treated, {len(control_outcomes)} control"],
        )

    def counterfactual_prediction(
        self,
        target_config: Dict[str, Any],
        outcome_metric: str,
        donor_pool_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Predict what the metric *would have been* for target_config
        using a k-NN weighted average of similar experiments.

        Returns (predicted_mean, predicted_std).
        """
        donors = self.store.query_similar(
            hyperparams=target_config.get("hyperparams"),
            architecture=target_config.get("architecture"),
            env_config=target_config.get("env_config"),
            top_k=10,
        )
        if not donors:
            return float("nan"), float("nan")

        values = []
        weights = []
        for rec, dist in donors:
            if outcome_metric not in rec.metrics:
                continue
            # inverse distance weighting
            w = 1.0 / (1e-6 + dist)
            values.append(rec.metrics[outcome_metric])
            weights.append(w)

        if not values:
            return float("nan"), float("nan")

        weights = np.array(weights)
        weights /= weights.sum()
        pred_mean = float(np.dot(weights, values))
        pred_var = float(np.dot(weights, (np.array(values) - pred_mean) ** 2))
        return pred_mean, float(np.sqrt(pred_var))

    # ------------------------------------------------------------------ #
    # Internal estimators
    # ------------------------------------------------------------------ #
    def _did(self, treatment, outcome_metric, control_query, treated_query, min_samples):
        treated = self.store.query(**(treated_query or {"hyperparams": treatment}))
        control = self.store.query(**(control_query or {}))
        # Exclude treated from control
        treated_ids = {r.experiment_id for r in treated}
        control = [r for r in control if r.experiment_id not in treated_ids]

        if len(treated) < min_samples or len(control) < min_samples:
            return CausalEstimate(
                treatment=treatment,
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="did",
                notes=[f"Insufficient samples: treated={len(treated)}, control={len(control)}"],
            )

        est = self.did_from_records(treated, control, outcome_metric)
        est.treatment = treatment
        return est

    def _synthetic_control(self, treatment, outcome_metric, control_query, treated_query, min_samples):
        """
        Build a synthetic control as a weighted average of donor experiments
        that best match the pre-treatment (here: hyperparameter) space.
        """
        treated = self.store.query(**(treated_query or {"hyperparams": treatment}))
        donors = self.store.query(**(control_query or {}))
        donor_ids = {r.experiment_id for r in donors}
        donor_ids -= {r.experiment_id for r in treated}
        donors = [r for r in donors if r.experiment_id in donor_ids]

        if len(treated) < min_samples or len(donors) < min_samples:
            return CausalEstimate(
                treatment=treatment,
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="synthetic_control",
                notes=[f"Insufficient donors: treated={len(treated)}, donors={len(donors)}"],
            )

        # Simple synthetic control: solve for weights that minimize
        # distance between treated mean feature vector and donor weighted sum.
        # Feature vector = flattened hyperparams + architecture + env_config (numeric only)
        def _numeric_vec(record):
            flat = {**record.hyperparams, **record.architecture, **record.env_config}
            nums = []
            for v in flat.values():
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return np.array(nums, dtype=np.float32)

        treated_vecs = np.array([_numeric_vec(r) for r in treated])
        donor_vecs = np.array([_numeric_vec(r) for r in donors])
        target = treated_vecs.mean(axis=0)

        # NNLS-like weights via least squares (non-negative via clipping)
        A = donor_vecs.T
        b = target
        try:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            w = np.ones(len(donors)) / len(donors)
        w = np.clip(w, 0, None)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(len(donors)) / len(donors)

        treated_outcomes = np.array([r.metrics.get(outcome_metric, np.nan) for r in treated])
        donor_outcomes = np.array([r.metrics.get(outcome_metric, np.nan) for r in donors])
        mask = ~np.isnan(treated_outcomes) & ~np.isnan(donor_outcomes)
        if mask.sum() < 2:
            return CausalEstimate(
                treatment=treatment,
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="synthetic_control",
                notes=["Missing outcome data"],
            )

        synthetic_mean = float(np.dot(w, donor_outcomes))
        effect = float(np.mean(treated_outcomes[mask])) - synthetic_mean
        # Heuristic confidence based on fit residual
        residual = float(np.linalg.norm(A @ w - b))
        confidence = float(max(0.0, 1.0 - residual / (np.linalg.norm(b) + 1e-6)))

        return CausalEstimate(
            treatment=treatment,
            outcome_metric=outcome_metric,
            causal_effect=effect,
            confidence=confidence,
            method="synthetic_control",
            control_mean=synthetic_mean,
            treated_mean=float(np.mean(treated_outcomes[mask])),
            notes=[f"Synthetic control from {len(donors)} donors, residual={residual:.4f}"],
        )

    def _regression(self, treatment, outcome_metric, control_query, treated_query, min_samples):
        """
        Counterfactual regression: linear model over numeric features -> outcome.
        Treatment effect = observed - predicted.
        """
        pool = self.store.query(**(control_query or {}))
        if treated_query:
            pool += self.store.query(**treated_query)
        # dedupe
        seen = set()
        pool = [r for r in pool if not (r.experiment_id in seen or seen.add(r.experiment_id))]

        def _features(record):
            flat = {**record.hyperparams, **record.architecture, **record.env_config}
            nums = []
            for v in flat.values():
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return np.array(nums, dtype=np.float32)

        X = []
        y = []
        for r in pool:
            if outcome_metric not in r.metrics:
                continue
            f = _features(r)
            if len(f) == 0:
                continue
            X.append(f)
            y.append(r.metrics[outcome_metric])

        if len(X) < min_samples:
            return CausalEstimate(
                treatment=treatment,
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="regression",
                notes=[f"Insufficient data for regression: {len(X)} samples"],
            )

        # Pad to common length
        max_len = max(len(v) for v in X)
        Xp = np.zeros((len(X), max_len), dtype=np.float32)
        for i, v in enumerate(X):
            Xp[i, :len(v)] = v
        y = np.array(y, dtype=np.float32)

        # Ridge regression closed form
        lam = 0.1
        I = np.eye(max_len)
        try:
            beta = np.linalg.solve(Xp.T @ Xp + lam * I, Xp.T @ y)
        except np.linalg.LinAlgError:
            beta = np.zeros(max_len)

        # Predict for treatment
        treated_recs = self.store.query(**(treated_query or {"hyperparams": treatment}))
        treated_outcomes = [r.metrics[outcome_metric] for r in treated_recs if outcome_metric in r.metrics]
        if not treated_outcomes:
            return CausalEstimate(
                treatment=treatment,
                outcome_metric=outcome_metric,
                causal_effect=0.0,
                confidence=0.0,
                method="regression",
                notes=["No treated outcomes available"],
            )

        treated_f = np.zeros(max_len, dtype=np.float32)
        tf = _features(treated_recs[0])
        treated_f[:len(tf)] = tf
        pred = float(treated_f @ beta)
        obs = float(np.mean(treated_outcomes))
        effect = obs - pred

        # Confidence from R^2 on training data
        y_pred = Xp @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-6)
        confidence = float(np.clip(r2, 0.0, 1.0))

        return CausalEstimate(
            treatment=treatment,
            outcome_metric=outcome_metric,
            causal_effect=effect,
            confidence=confidence,
            method="regression",
            control_mean=pred,
            treated_mean=obs,
            notes=[f"Ridge regression R^2={r2:.3f}, n={len(X)}"],
        )

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _ensure_cache_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_cache (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _cache_key(self, **kwargs) -> str:
        return "causal_" + hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:24]

    def _get_cached(self, key: str) -> Optional[CausalEstimate]:
        if not self._conn:
            return None
        row = self._conn.execute("SELECT result_json FROM causal_cache WHERE cache_key = ?", (key,)).fetchone()
        if row:
            return CausalEstimate(**json.loads(row[0]))
        return None

    def _set_cached(self, key: str, estimate: CausalEstimate) -> None:
        if not self._conn:
            return
        from datetime import datetime
        self._conn.execute(
            "INSERT OR REPLACE INTO causal_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps({k: v for k, v in estimate.__dict__.items() if v is not None}), datetime.utcnow().isoformat()),
        )
        self._conn.commit()


# ---------------------------------------------------------------------- #
# Example / self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from moses.memory.experience_store import ExperienceStore

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as td:
        store = ExperienceStore(Path(td) / "exp.db", vector_dim=32)
        # Seed: low LR baseline
        for i in range(5):
            store.record(
                f"ctrl_{i}",
                hyperparams={"lr": 1e-3, "batch_size": 256},
                architecture={"layers": 3},
                env_config={"robot": "humanoid_28dof"},
                metrics={"reward_mean": 4000 + np.random.randn() * 200},
                tags=["baseline"],
            )
        # Seed: high LR treatment
        for i in range(5):
            store.record(
                f"treat_{i}",
                hyperparams={"lr": 3e-4, "batch_size": 256},
                architecture={"layers": 3},
                env_config={"robot": "humanoid_28dof"},
                metrics={"reward_mean": 5500 + np.random.randn() * 200},
                tags=["treatment"],
            )

        engine = CausalEngine(store)

        for method in ("did", "synthetic_control", "regression"):
            est = engine.estimate_effect(
                treatment={"lr": 3e-4},
                outcome_metric="reward_mean",
                control_query={"hyperparams": {"lr": 1e-3, "batch_size": 256}},
                treated_query={"hyperparams": {"lr": 3e-4, "batch_size": 256}},
                method=method,
            )
            print(f"[{method}] effect={est.causal_effect:.1f} confidence={est.confidence:.2f} notes={est.notes}")

        # Counterfactual prediction
        pred_mean, pred_std = engine.counterfactual_prediction(
            target_config={"hyperparams": {"lr": 5e-4, "batch_size": 256}},
            outcome_metric="reward_mean",
        )
        print(f"Counterfactual prediction: {pred_mean:.1f} +/- {pred_std:.1f}")
