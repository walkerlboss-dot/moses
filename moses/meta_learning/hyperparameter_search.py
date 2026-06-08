"""
Hyperparameter Search for Moses RL Training

Integrates Optuna for efficient hyperparameter optimization with early pruning,
database persistence, and resumption support.

Example
-------
>>> from moses.meta_learning import HyperparameterSearch
>>> search = HyperparameterSearch(
...     study_name="moses_ppo_walk",
...     storage_url="sqlite:///moses_hpo.db",
...     direction="maximize",
... )
>>> best_cfg = search.run(n_trials=128, timeout=3600)
>>> print(best_cfg["learning_rate"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optuna is an optional dependency at import time so the module can be inspected
# without it installed; it is required at runtime for search().
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    _HAS_OPTUNA = True
except Exception:  # pragma: no cover
    optuna = None  # type: ignore
    TPESampler = None  # type: ignore
    MedianPruner = None  # type: ignore
    _HAS_OPTUNA = False


@dataclass
class SearchSpace:
    """Declarative search-space bounds for Moses PPO-style training."""

    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True

    batch_size_choices: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    minibatch_size_choices: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    n_layers_min: int = 2
    n_layers_max: int = 5

    hidden_size_min: int = 128
    hidden_size_max: int = 1024
    hidden_size_step: int = 128

    entropy_coef_min: float = 0.0
    entropy_coef_max: float = 0.1

    gae_lambda_min: float = 0.9
    gae_lambda_max: float = 1.0

    gamma_min: float = 0.95
    gamma_max: float = 0.9999

    clip_epsilon_min: float = 0.05
    clip_epsilon_max: float = 0.3

    n_epochs_min: int = 3
    n_epochs_max: int = 15

    vf_coef_min: float = 0.25
    vf_coef_max: float = 1.0

    max_grad_norm_min: float = 0.3
    max_grad_norm_max: float = 1.0

    def sample(self, trial: Any) -> Dict[str, Any]:
        """Sample a configuration from the search space using an Optuna trial."""
        if optuna is None:
            raise RuntimeError("optuna is required for sampling")

        cfg: Dict[str, Any] = {}

        # Learning rate
        cfg["learning_rate"] = trial.suggest_float(
            "learning_rate",
            self.lr_min,
            self.lr_max,
            log=self.lr_log,
        )

        # Batch sizes
        cfg["batch_size"] = trial.suggest_categorical("batch_size", self.batch_size_choices)
        valid_minibatch = [m for m in self.minibatch_size_choices if m <= cfg["batch_size"]]
        if not valid_minibatch:
            valid_minibatch = [cfg["batch_size"]]
        cfg["minibatch_size"] = trial.suggest_categorical("minibatch_size", valid_minibatch)

        # Network architecture
        cfg["n_layers"] = trial.suggest_int("n_layers", self.n_layers_min, self.n_layers_max)
        cfg["hidden_size"] = trial.suggest_int(
            "hidden_size",
            self.hidden_size_min,
            self.hidden_size_max,
            step=self.hidden_size_step,
        )

        # PPO-specific
        cfg["entropy_coef"] = trial.suggest_float(
            "entropy_coef", self.entropy_coef_min, self.entropy_coef_max
        )
        cfg["gae_lambda"] = trial.suggest_float(
            "gae_lambda", self.gae_lambda_min, self.gae_lambda_max
        )
        cfg["gamma"] = trial.suggest_float("gamma", self.gamma_min, self.gamma_max)
        cfg["clip_epsilon"] = trial.suggest_float(
            "clip_epsilon", self.clip_epsilon_min, self.clip_epsilon_max
        )
        cfg["n_epochs"] = trial.suggest_int("n_epochs", self.n_epochs_min, self.n_epochs_max)
        cfg["vf_coef"] = trial.suggest_float("vf_coef", self.vf_coef_min, self.vf_coef_max)
        cfg["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm", self.max_grad_norm_min, self.max_grad_norm_max
        )

        return cfg


class HyperparameterSearch:
    """
    Optuna-based hyperparameter search with early pruning and resumption.

    Parameters
    ----------
    study_name : str
        Unique identifier for the study (used for DB storage and resumption).
    storage_url : str, optional
        RDB storage URL. Defaults to a local SQLite file under ``./optuna_studies/``.
    direction : str
        ``"maximize"`` or ``"minimize"``.
    search_space : SearchSpace, optional
        Bounds and choices. A default instance is used if omitted.
    n_startup_trials : int
        Number of random trials before TPE takes over.
    n_warmup_steps : int
        Steps before MedianPruner activates.
    interval_steps : int
        Pruning check interval.
    """

    def __init__(
        self,
        study_name: str,
        storage_url: Optional[str] = None,
        direction: str = "maximize",
        search_space: Optional[SearchSpace] = None,
        n_startup_trials: int = 10,
        n_warmup_steps: int = 5,
        interval_steps: int = 1,
    ) -> None:
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for HyperparameterSearch. "
                "Install it with: pip install optuna"
            )

        self.study_name = study_name
        self.direction = direction
        self.search_space = search_space or SearchSpace()
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps

        if storage_url is None:
            os.makedirs("./optuna_studies", exist_ok=True)
            storage_url = f"sqlite:///optuna_studies/{study_name}.db"
        self.storage_url = storage_url

        self._study: Optional[Any] = None
        self._objective_fn: Optional[Callable[[Dict[str, Any]], float]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_objective(
        self, fn: Callable[[Dict[str, Any]], float]
    ) -> "HyperparameterSearch":
        """
        Set the black-box objective.

        The callable receives a sampled config dict and must return a scalar
        score (higher is better when ``direction="maximize"``).
        """
        self._objective_fn = fn
        return self

    def run(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the search.

        Parameters
        ----------
        n_trials : int
            Maximum number of trials.
        timeout : int, optional
            Stop after ``timeout`` seconds.
        n_jobs : int
            Number of parallel workers (use >1 with caution; objective must be thread-safe).
        show_progress : bool
            Forwarded to Optuna's ``show_progress_bar``.

        Returns
        -------
        dict
            Best hyperparameter configuration found.
        """
        if self._objective_fn is None:
            raise RuntimeError(
                "No objective function registered. Call set_objective() first."
            )

        self._ensure_study()
        assert self._study is not None

        logger.info(
            "Starting HPO study=%s trials=%s timeout=%s jobs=%s",
            self.study_name,
            n_trials,
            timeout,
            n_jobs,
        )

        self._study.optimize(
            self._wrap_objective(),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )

        best = self._study.best_params
        best_value = self._study.best_value
        logger.info(
            "HPO complete. best_value=%.4f config=%s",
            best_value,
            best,
        )
        return dict(best)

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Return the best config seen so far (works after resumption)."""
        self._ensure_study()
        assert self._study is not None
        if self._study.best_trial is None:
            return None
        return dict(self._study.best_params)

    def report_intermediate(
        self,
        trial_id: int,
        step: int,
        value: float,
    ) -> None:
        """
        Report an intermediate value for a running trial from an external process.

        This is useful when the objective runs out-of-process (e.g., on a cluster)
        and you want to enable pruning.
        """
        self._ensure_study()
        assert self._study is not None
        trial = self._study._storage.get_trial(trial_id)
        if trial is None:
            logger.warning("Trial %d not found in study storage", trial_id)
            return
        # Optuna's storage API is the stable public surface for this
        self._study._storage.set_trial_intermediate_value(trial_id, step, value)
        should_prune = self._study.pruner.prune(self._study, trial)
        if should_prune:
            self._study._storage.set_trial_state_values(trial_id, optuna.trial.TrialState.PRUNED)
            logger.info("Trial %d pruned at step %d", trial_id, step)

    def save_best_config(self, path: str) -> None:
        """Persist the best configuration to a JSON file."""
        best = self.get_best_config()
        if best is None:
            raise RuntimeError("No best config available yet.")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2)
        logger.info("Best config saved to %s", path)

    def load_best_config(self, path: str) -> Dict[str, Any]:
        """Load a previously saved best configuration."""
        with open(path, "r", encoding="utf-8") as fh:
            cfg: Dict[str, Any] = json.load(fh)
        logger.info("Loaded config from %s", path)
        return cfg

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_study(self) -> None:
        if self._study is not None:
            return
        sampler = TPESampler(n_startup_trials=self.n_startup_trials, multivariate=True)
        pruner = MedianPruner(
            n_startup_trials=0,
            n_warmup_steps=self.n_warmup_steps,
            interval_steps=self.interval_steps,
        )
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            sampler=sampler,
            pruner=pruner,
            direction=self.direction,
            load_if_exists=True,
        )
        logger.info(
            "Study '%s' loaded/created with %d completed trials",
            self.study_name,
            len(self._study.trials),
        )

    def _wrap_objective(self) -> Callable[[Any], float]:
        assert self._objective_fn is not None

        def _objective(trial: Any) -> float:
            cfg = self.search_space.sample(trial)
            logger.debug("Trial %d config: %s", trial.number, cfg)
            try:
                score = self._objective_fn(cfg)
            except Exception as exc:
                logger.exception("Trial %d failed: %s", trial.number, exc)
                raise optuna.TrialPruned()
            logger.debug("Trial %d score: %.4f", trial.number, score)
            return float(score)

        return _objective
