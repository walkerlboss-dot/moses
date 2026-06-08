"""
Experiment runner for Moses v5.0.

Orchestrates experiments in parallel, tracks results, performs statistical
comparison, and auto-promotes winners to production.
"""

from __future__ import annotations

import os
import json
import time
import copy
import logging
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .search_space import SearchSpace, ComposableSearchSpace
from .budget import BudgetManager, ComputeBudget, ExperimentBudget, EarlyStoppingCallback


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moses.experiments")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    exp_id: str
    algorithm: str  # ppo, sac, gr00t
    search_space: SearchSpace
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)

    # Execution config
    n_workers: int = 1
    device_ids: List[int] = field(default_factory=lambda: [0])
    seed: int = 42

    # Meta-learner integration
    use_optuna: bool = True
    optuna_sampler: str = "tpe"  # tpe, cmaes, random
    optuna_pruner: str = "median"  # median, hyperband

    # Evaluation
    n_eval_episodes: int = 10
    eval_frequency: int = 10_000

    # Auto-promotion
    auto_promote: bool = True
    promote_threshold: float = 0.05  # 5% improvement required
    promote_min_trials: int = 20

    # Storage
    output_dir: str = "./experiments"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 50_000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["search_space"] = self.search_space.__class__.__name__
        return d


@dataclass
class ExperimentResult:
    """Result from a single experiment trial."""

    trial_id: int
    exp_id: str
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    std_error: float = 0.0
    n_eval_episodes: int = 0
    wall_time_seconds: float = 0.0
    gpu_hours_used: float = 0.0
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentComparison:
    """Statistical comparison between experiments."""

    baseline_exp_id: str
    candidate_exp_id: str
    baseline_mean: float
    candidate_mean: float
    mean_diff: float
    relative_improvement: float
    p_value_ttest: float
    p_value_bootstrap: float
    effect_size_cohens_d: float
    confidence_interval_95: Tuple[float, float]
    is_significant: bool
    sample_size_baseline: int
    sample_size_candidate: int
    recommendation: str  # "promote", "reject", "inconclusive"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Main experiment orchestrator for Moses v5.0."""

    def __init__(
        self,
        budget_manager: Optional[BudgetManager] = None,
        train_fn: Optional[Callable[[Dict[str, Any], int], Tuple[float, Dict[str, float]]]] = None,
        eval_fn: Optional[Callable[[Any, int], Tuple[float, Dict[str, float]]]] = None,
    ):
        self.budget_manager = budget_manager or BudgetManager()
        self.train_fn = train_fn
        self.eval_fn = eval_fn

        # Results storage
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.best_results: Dict[str, Optional[ExperimentResult]] = {}
        self.studies: Dict[str, Any] = {}  # Optuna studies

        # Threading
        self._lock = threading.Lock()
        self._running: Dict[str, bool] = {}

        # Production pipeline
        self.production_candidates: List[ExperimentResult] = []

    # ------------------------------------------------------------------
    # Core Experiment Execution
    # ------------------------------------------------------------------

    def run_experiment(self, config: ExperimentConfig) -> List[ExperimentResult]:
        """Run a complete experiment with the given config."""
        exp_id = config.exp_id
        logger.info(f"Starting experiment: {exp_id}")

        # Register with budget manager
        self.budget_manager.register_experiment(exp_id, config.budget)

        # Check if we can start
        est_hours = self._estimate_experiment_hours(config)
        can_start, reason = self.budget_manager.can_start_experiment(exp_id, est_hours)
        if not can_start:
            logger.error(f"Cannot start experiment {exp_id}: {reason}")
            return []

        with self._lock:
            self.results[exp_id] = []
            self._running[exp_id] = True

        # Create output directory
        output_dir = Path(config.output_dir) / exp_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)

        try:
            if config.use_optuna and HAS_OPTUNA:
                results = self._run_with_optuna(config, output_dir)
            else:
                results = self._run_grid_search(config, output_dir)
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
            traceback.print_exc()
            results = []
        finally:
            with self._lock:
                self._running[exp_id] = False

        # Save results
        self._save_results(exp_id, output_dir)

        # Check auto-promotion
        if config.auto_promote and results:
            self._check_auto_promote(config, results, output_dir)

        logger.info(f"Experiment {exp_id} completed with {len(results)} trials")
        return results

    def run_experiments_parallel(
        self,
        configs: List[ExperimentConfig],
        max_parallel: Optional[int] = None,
    ) -> Dict[str, List[ExperimentResult]]:
        """Run multiple experiments in parallel."""
        max_parallel = max_parallel or min(len(configs), mp.cpu_count())
        results = {}

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(self.run_experiment, config): config.exp_id
                for config in configs
            }

            for future in as_completed(futures):
                exp_id = futures[future]
                try:
                    results[exp_id] = future.result()
                except Exception as e:
                    logger.error(f"Experiment {exp_id} failed in parallel run: {e}")
                    results[exp_id] = []

        return results

    # ------------------------------------------------------------------
    # Optuna Integration
    # ------------------------------------------------------------------

    def _run_with_optuna(self, config: ExperimentConfig, output_dir: Path) -> List[ExperimentResult]:
        """Run experiment using Optuna for hyperparameter optimization."""
        exp_id = config.exp_id

        # Select sampler
        if config.optuna_sampler == "tpe":
            sampler = TPESampler(seed=config.seed)
        elif config.optuna_sampler == "cmaes":
            sampler = CmaEsSampler(seed=config.seed)
        else:
            sampler = RandomSampler(seed=config.seed)

        # Select pruner
        if config.optuna_pruner == "hyperband":
            pruner = HyperbandPruner()
        else:
            pruner = MedianPruner()

        # Create study
        study = optuna.create_study(
            study_name=exp_id,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        self.studies[exp_id] = study

        # Budget callback
        budget_callback = EarlyStoppingCallback(self.budget_manager, exp_id)

        # Objective wrapper
        def objective(trial: optuna.Trial) -> float:
            return self._run_trial(trial, config, output_dir)

        # Run optimization
        study.optimize(
            objective,
            n_trials=config.budget.max_trials,
            callbacks=[budget_callback],
            n_jobs=config.n_workers,
            show_progress_bar=True,
        )

        return self.results.get(exp_id, [])

    def _run_trial(self, trial: Any, config: ExperimentConfig, output_dir: Path) -> float:
        """Run a single trial."""
        exp_id = config.exp_id
        trial_id = trial.number

        start_time = time.time()

        try:
            # Sample parameters
            params = config.search_space.sample(trial)

            # Check budget
            can_continue, reason = self.budget_manager.report_trial(exp_id, float("-inf"))
            if can_continue:
                logger.warning(f"Trial {trial_id} stopped: {reason}")
                raise optuna.TrialPruned()

            # Train
            if self.train_fn is None:
                raise RuntimeError("No train_fn provided")

            score, metrics = self.train_fn(params, config.seed + trial_id)

            # Evaluate
            if self.eval_fn is not None:
                eval_score, eval_metrics = self.eval_fn(None, config.n_eval_episodes)
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                score = eval_score

            # Calculate std error
            std_error = metrics.get("score_std", 0.0) / max(metrics.get("n_eval", 1), 1) ** 0.5

            # GPU hours estimate
            elapsed = time.time() - start_time
            gpu_hours = elapsed * len(config.device_ids) / 3600.0

            # Create result
            result = ExperimentResult(
                trial_id=trial_id,
                exp_id=exp_id,
                params=params,
                score=score,
                metrics=metrics,
                std_error=std_error,
                n_eval_episodes=config.n_eval_episodes,
                wall_time_seconds=elapsed,
                gpu_hours_used=gpu_hours,
            )

            # Save checkpoint
            if config.save_checkpoints:
                result.checkpoint_path = str(output_dir / f"trial_{trial_id}.pt")

            # Store result
            with self._lock:
                self.results[exp_id].append(result)
                if (self.best_results.get(exp_id) is None or
                    score > self.best_results[exp_id].score):
                    self.best_results[exp_id] = result

            # Report to budget manager
            self.budget_manager.report_trial(exp_id, score, gpu_hours)

            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            error_result = ExperimentResult(
                trial_id=trial_id,
                exp_id=exp_id,
                params={},
                score=float("-inf"),
                error=str(e),
                wall_time_seconds=time.time() - start_time,
            )
            with self._lock:
                self.results[exp_id].append(error_result)
            raise optuna.TrialPruned()

    # ------------------------------------------------------------------
    # Grid Search Fallback
    # ------------------------------------------------------------------

    def _run_grid_search(self, config: ExperimentConfig, output_dir: Path) -> List[ExperimentResult]:
        """Fallback grid search when Optuna is unavailable."""
        from .search_space import grid_search_space

        exp_id = config.exp_id
        params_list = grid_search_space(config.search_space, config.budget.max_trials)

        for trial_id, params in enumerate(params_list):
            start_time = time.time()

            # Check budget
            exhausted, _ = config.budget.is_exhausted()
            if exhausted:
                break

            try:
                score, metrics = self.train_fn(params, config.seed + trial_id)

                elapsed = time.time() - start_time
                gpu_hours = elapsed * len(config.device_ids) / 3600.0

                result = ExperimentResult(
                    trial_id=trial_id,
                    exp_id=exp_id,
                    params=params,
                    score=score,
                    metrics=metrics,
                    wall_time_seconds=elapsed,
                    gpu_hours_used=gpu_hours,
                )

                with self._lock:
                    self.results[exp_id].append(result)
                    if (self.best_results.get(exp_id) is None or
                        score > self.best_results[exp_id].score):
                        self.best_results[exp_id] = result

                self.budget_manager.report_trial(exp_id, score, gpu_hours)
                config.budget.check_early_stop(score)

            except Exception as e:
                logger.error(f"Grid search trial {trial_id} failed: {e}")

        return self.results.get(exp_id, [])

    # ------------------------------------------------------------------
    # Statistical Comparison
    # ------------------------------------------------------------------

    def compare_experiments(
        self,
        baseline_exp_id: str,
        candidate_exp_id: str,
        metric: str = "score",
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
    ) -> ExperimentComparison:
        """
        Statistically compare two experiments.

        Uses t-test and bootstrap confidence intervals.
        Returns recommendation: "promote", "reject", or "inconclusive".
        """
        baseline_scores = self._get_scores(baseline_exp_id, metric)
        candidate_scores = self._get_scores(candidate_exp_id, metric)

        if len(baseline_scores) == 0 or len(candidate_scores) == 0:
            raise ValueError("Insufficient data for comparison")

        baseline_mean = np.mean(baseline_scores)
        candidate_mean = np.mean(candidate_scores)
        mean_diff = candidate_mean - baseline_mean
        relative_improvement = mean_diff / max(abs(baseline_mean), 1e-6)

        # T-test
        if HAS_SCIPY and len(baseline_scores) > 1 and len(candidate_scores) > 1:
            t_stat, p_value_ttest = stats.ttest_ind(
                candidate_scores, baseline_scores, equal_var=False
            )
        else:
            p_value_ttest = 1.0

        # Bootstrap confidence interval
        ci_low, ci_high = self._bootstrap_ci(
            baseline_scores, candidate_scores, n_bootstrap=n_bootstrap
        )

        # Bootstrap p-value
        p_value_bootstrap = self._bootstrap_pvalue(
            baseline_scores, candidate_scores, n_bootstrap=n_bootstrap
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.var(baseline_scores, ddof=1) + np.var(candidate_scores, ddof=1)) / 2
        )
        cohens_d = mean_diff / max(pooled_std, 1e-6)

        # Significance check
        is_significant = (p_value_ttest < alpha) and (p_value_bootstrap < alpha)

        # Recommendation
        if is_significant and relative_improvement > 0:
            recommendation = "promote"
        elif is_significant and relative_improvement < 0:
            recommendation = "reject"
        else:
            recommendation = "inconclusive"

        return ExperimentComparison(
            baseline_exp_id=baseline_exp_id,
            candidate_exp_id=candidate_exp_id,
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            mean_diff=mean_diff,
            relative_improvement=relative_improvement,
            p_value_ttest=p_value_ttest,
            p_value_bootstrap=p_value_bootstrap,
            effect_size_cohens_d=cohens_d,
            confidence_interval_95=(ci_low, ci_high),
            is_significant=is_significant,
            sample_size_baseline=len(baseline_scores),
            sample_size_candidate=len(candidate_scores),
            recommendation=recommendation,
        )

    def _get_scores(self, exp_id: str, metric: str = "score") -> np.ndarray:
        """Extract scores from experiment results."""
        results = self.results.get(exp_id, [])
        scores = []
        for r in results:
            if r.error is None:
                if metric == "score":
                    scores.append(r.score)
                else:
                    scores.append(r.metrics.get(metric, r.score))
        return np.array(scores)

    def _bootstrap_ci(
        self,
        baseline: np.ndarray,
        candidate: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for mean difference."""
        rng = np.random.RandomState(42)
        diffs = []

        for _ in range(n_bootstrap):
            b_sample = rng.choice(baseline, size=len(baseline), replace=True)
            c_sample = rng.choice(candidate, size=len(candidate), replace=True)
            diffs.append(np.mean(c_sample) - np.mean(b_sample))

        diffs = np.array(diffs)
        alpha = 1 - confidence
        ci_low = np.percentile(diffs, alpha / 2 * 100)
        ci_high = np.percentile(diffs, (1 - alpha / 2) * 100)
        return ci_low, ci_high

    def _bootstrap_pvalue(
        self,
        baseline: np.ndarray,
        candidate: np.ndarray,
        n_bootstrap: int = 10000,
    ) -> float:
        """Bootstrap p-value for difference in means."""
        rng = np.random.RandomState(42)
        observed_diff = np.mean(candidate) - np.mean(baseline)
        pooled = np.concatenate([baseline, candidate])

        count = 0
        for _ in range(n_bootstrap):
            shuffled = rng.permutation(pooled)
            b_sample = shuffled[:len(baseline)]
            c_sample = shuffled[len(baseline):]
            diff = np.mean(c_sample) - np.mean(b_sample)
            if abs(diff) >= abs(observed_diff):
                count += 1

        return count / n_bootstrap

    # ------------------------------------------------------------------
    # Auto-Promotion
    # ------------------------------------------------------------------

    def _check_auto_promote(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult],
        output_dir: Path,
    ) -> None:
        """Check if experiment should be promoted to production."""
        if len(results) < config.promote_min_trials:
            return

        valid_results = [r for r in results if r.error is None]
        if len(valid_results) < config.promote_min_trials:
            return

        best = max(valid_results, key=lambda r: r.score)

        # Check against production candidates
        if not self.production_candidates:
            self.production_candidates.append(best)
            self._save_production_candidate(best, output_dir)
            logger.info(f"Experiment {config.exp_id}: First production candidate (score={best.score:.4f})")
            return

        # Compare to current best production candidate
        current_best = max(self.production_candidates, key=lambda r: r.score)

        if best.score > current_best.score * (1 + config.promote_threshold):
            self.production_candidates.append(best)
            self._save_production_candidate(best, output_dir)
            logger.info(
                f"Experiment {config.exp_id}: PROMOTED to production! "
                f"Score: {best.score:.4f} vs current {current_best.score:.4f} "
                f"(+{(best.score/current_best.score - 1)*100:.1f}%)"
            )

            # Save promotion record
            promotion = {
                "timestamp": time.time(),
                "exp_id": config.exp_id,
                "trial_id": best.trial_id,
                "new_score": best.score,
                "previous_score": current_best.score,
                "improvement": best.score - current_best.score,
                "relative_improvement": best.score / current_best.score - 1,
                "params": best.params,
            }
            with open(output_dir / "promotion.json", "w") as f:
                json.dump(promotion, f, indent=2, default=str)

    def _save_production_candidate(self, result: ExperimentResult, output_dir: Path) -> None:
        """Save a production candidate result."""
        prod_dir = Path(output_dir) / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)

        with open(prod_dir / f"candidate_{result.trial_id}.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def get_production_candidates(self) -> List[ExperimentResult]:
        """Get all production candidates sorted by score."""
        return sorted(self.production_candidates, key=lambda r: r.score, reverse=True)

    def promote_to_production(self, result: ExperimentResult) -> None:
        """Manually promote a result to production."""
        self.production_candidates.append(result)
        logger.info(f"Manually promoted trial {result.trial_id} from {result.exp_id}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _estimate_experiment_hours(self, config: ExperimentConfig) -> float:
        """Estimate GPU hours for an experiment."""
        from .budget import estimate_gpu_hours
        return estimate_gpu_hours(
            n_trials=config.budget.max_trials,
            iterations_per_trial=config.budget.max_iterations // max(config.budget.max_trials, 1),
            seconds_per_iteration=0.1,  # Default estimate
            n_gpus=len(config.device_ids),
        )

    def _save_results(self, exp_id: str, output_dir: Path) -> None:
        """Save all results for an experiment."""
        results = self.results.get(exp_id, [])
        data = [r.to_dict() for r in results]

        with open(output_dir / "results.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Save summary
        valid = [r for r in results if r.error is None]
        if valid:
            summary = {
                "exp_id": exp_id,
                "n_trials": len(results),
                "n_successful": len(valid),
                "best_score": max(r.score for r in valid),
                "mean_score": np.mean([r.score for r in valid]),
                "std_score": np.std([r.score for r in valid]),
                "total_gpu_hours": sum(r.gpu_hours_used for r in valid),
                "total_wall_time": sum(r.wall_time_seconds for r in valid),
            }
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    def get_best_result(self, exp_id: str) -> Optional[ExperimentResult]:
        """Get the best result for an experiment."""
        return self.best_results.get(exp_id)

    def get_all_results(self, exp_id: str) -> List[ExperimentResult]:
        """Get all results for an experiment."""
        return self.results.get(exp_id, [])

    def is_running(self, exp_id: str) -> bool:
        """Check if an experiment is currently running."""
        return self._running.get(exp_id, False)

    def get_status(self) -> Dict[str, Any]:
        """Get overall runner status."""
        return {
            "running_experiments": [k for k, v in self._running.items() if v],
            "completed_experiments": [k for k, v in self._running.items() if not v],
            "total_experiments": len(self.results),
            "production_candidates": len(self.production_candidates),
            "budget_status": self.budget_manager.get_global_status(),
        }

    def export_results(self, path: str) -> None:
        """Export all results to a JSON file."""
        data = {}
        for exp_id, results in self.results.items():
            data[exp_id] = [r.to_dict() for r in results]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_results(self, path: str) -> None:
        """Load results from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        for exp_id, results_data in data.items():
            self.results[exp_id] = [
                ExperimentResult(**{k: v for k, v in r.items() if k in ExperimentResult.__dataclass_fields__})
                for r in results_data
            ]
            valid = [r for r in self.results[exp_id] if r.error is None]
            if valid:
                self.best_results[exp_id] = max(valid, key=lambda r: r.score)
