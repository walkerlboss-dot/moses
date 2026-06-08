"""
Budget management for Moses v5.0 experiments.

Tracks compute budgets, experiment budgets, early stopping, and cost estimation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import threading


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ComputeBudget:
    """Compute resource budget (GPU-hours)."""

    gpu_hours_per_day: float = 24.0
    gpu_hours_per_week: float = 168.0
    max_parallel_gpus: int = 4
    gpu_type: str = "a100"  # For cost estimation

    # Tracking
    used_gpu_hours: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allocate(self, hours: float) -> bool:
        """Try to allocate GPU hours. Returns True if successful."""
        with self._lock:
            daily_remaining = self.gpu_hours_per_day - self.get_daily_usage()
            if hours <= daily_remaining:
                self.used_gpu_hours += hours
                return True
            return False

    def get_daily_usage(self) -> float:
        """Get GPU hours used in last 24h (simplified - tracks total)."""
        # In production, track with timestamps
        return self.used_gpu_hours % self.gpu_hours_per_day

    def get_weekly_usage(self) -> float:
        """Get GPU hours used in last 7 days."""
        return self.used_gpu_hours % self.gpu_hours_per_week

    def remaining_daily(self) -> float:
        return self.gpu_hours_per_day - self.get_daily_usage()

    def remaining_weekly(self) -> float:
        return self.gpu_hours_per_week - self.get_weekly_usage()

    def estimate_cost(self, hours: float) -> float:
        """Estimate cloud cost for GPU hours (USD)."""
        rates = {
            "a100": 2.5,
            "h100": 4.0,
            "v100": 1.5,
            "t4": 0.5,
            "a10": 1.0,
        }
        return hours * rates.get(self.gpu_type, 1.0)


@dataclass
class ExperimentBudget:
    """Budget for a single experiment or study."""

    max_trials: int = 100
    max_iterations: int = 1_000_000
    max_wall_time_seconds: float = 86400.0  # 24 hours
    min_trials: int = 10  # Minimum before early stopping

    # Early stopping thresholds
    early_stop_patience: int = 20  # Trials without improvement
    early_stop_min_delta: float = 0.01  # Minimum improvement

    # Tracking
    trial_count: int = field(default=0, repr=False)
    iteration_count: int = field(default=0, repr=False)
    start_time: float = field(default_factory=time.time, repr=False)
    best_score: float = field(default=float("-inf"), repr=False)
    trials_without_improvement: int = field(default=0, repr=False)
    scores: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)

    def check_trial_budget(self) -> bool:
        """Check if we can run another trial."""
        return self.trial_count < self.max_trials

    def check_iteration_budget(self, n_iterations: int = 1) -> bool:
        """Check if we can run more iterations."""
        return (self.iteration_count + n_iterations) <= self.max_iterations

    def check_time_budget(self) -> bool:
        """Check if we're within wall time budget."""
        elapsed = time.time() - self.start_time
        return elapsed < self.max_wall_time_seconds

    def check_early_stop(self, score: float) -> Tuple[bool, str]:
        """
        Check if experiment should stop early.
        Returns (should_stop, reason).
        """
        self.trial_count += 1
        self.scores.append(score)

        if self.trial_count < self.min_trials:
            return False, "below_min_trials"

        if score > self.best_score + self.early_stop_min_delta:
            self.best_score = score
            self.trials_without_improvement = 0
            return False, "improved"

        self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.early_stop_patience:
            return True, f"no_improvement_for_{self.early_stop_patience}_trials"

        return False, "no_improvement"

    def is_exhausted(self) -> Tuple[bool, str]:
        """Check if any budget is exhausted."""
        if not self.check_trial_budget():
            return True, "max_trials_reached"
        if not self.check_time_budget():
            return True, "time_budget_exhausted"
        if not self.check_iteration_budget():
            return True, "iteration_budget_exhausted"
        return False, "ok"

    def get_progress(self) -> Dict[str, Any]:
        """Get current budget consumption status."""
        elapsed = time.time() - self.start_time
        return {
            "trial_count": self.trial_count,
            "max_trials": self.max_trials,
            "trial_progress": self.trial_count / max(self.max_trials, 1),
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "iteration_progress": self.iteration_count / max(self.max_iterations, 1),
            "elapsed_seconds": elapsed,
            "max_wall_time": self.max_wall_time_seconds,
            "time_progress": elapsed / max(self.max_wall_time_seconds, 1),
            "best_score": self.best_score,
            "trials_without_improvement": self.trials_without_improvement,
        }


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------

class BudgetManager:
    """Central budget manager for all experiments."""

    def __init__(
        self,
        compute_budget: Optional[ComputeBudget] = None,
        global_experiment_budget: Optional[ExperimentBudget] = None,
    ):
        self.compute = compute_budget or ComputeBudget()
        self.global_budget = global_experiment_budget or ExperimentBudget()

        # Per-experiment budgets
        self.experiment_budgets: Dict[str, ExperimentBudget] = {}
        self._lock = threading.Lock()

        # Cost tracking
        self.actual_costs: Dict[str, float] = {}
        self.estimated_costs: Dict[str, float] = {}
        self.cost_history: deque = deque(maxlen=1000)

    def register_experiment(self, exp_id: str, budget: Optional[ExperimentBudget] = None) -> ExperimentBudget:
        """Register a new experiment with its own budget."""
        with self._lock:
            if exp_id not in self.experiment_budgets:
                self.experiment_budgets[exp_id] = budget or ExperimentBudget()
                self.actual_costs[exp_id] = 0.0
                self.estimated_costs[exp_id] = 0.0
            return self.experiment_budgets[exp_id]

    def can_start_experiment(self, exp_id: str, estimated_gpu_hours: float) -> Tuple[bool, str]:
        """Check if an experiment can be started."""
        with self._lock:
            # Check global compute
            if not self.compute.allocate(estimated_gpu_hours):
                return False, "insufficient_daily_compute"

            # Check global experiment budget
            exhausted, reason = self.global_budget.is_exhausted()
            if exhausted:
                return False, f"global_budget_exhausted:{reason}"

            # Check per-experiment budget
            if exp_id in self.experiment_budgets:
                exp_budget = self.experiment_budgets[exp_id]
                exhausted, reason = exp_budget.is_exhausted()
                if exhausted:
                    return False, f"experiment_budget_exhausted:{reason}"

            self.estimated_costs[exp_id] = self.estimated_costs.get(exp_id, 0.0) + estimated_gpu_hours
            return True, "ok"

    def report_trial(self, exp_id: str, score: float, gpu_hours_used: float = 0.0) -> Tuple[bool, str]:
        """Report trial completion and check budgets."""
        with self._lock:
            self.actual_costs[exp_id] = self.actual_costs.get(exp_id, 0.0) + gpu_hours_used
            self.cost_history.append({
                "exp_id": exp_id,
                "score": score,
                "gpu_hours": gpu_hours_used,
                "timestamp": time.time(),
            })

            if exp_id in self.experiment_budgets:
                exp_budget = self.experiment_budgets[exp_id]
                should_stop, reason = exp_budget.check_early_stop(score)
                if should_stop:
                    return True, f"early_stop:{reason}"

                exhausted, reason = exp_budget.is_exhausted()
                if exhausted:
                    return True, f"budget_exhausted:{reason}"

            return False, "continue"

    def report_iterations(self, exp_id: str, n_iterations: int) -> bool:
        """Report iterations used. Returns False if budget exceeded."""
        with self._lock:
            if exp_id in self.experiment_budgets:
                budget = self.experiment_budgets[exp_id]
                budget.iteration_count += n_iterations
                return budget.check_iteration_budget()
            return True

    def get_experiment_status(self, exp_id: str) -> Dict[str, Any]:
        """Get full status for an experiment."""
        with self._lock:
            status = {
                "exp_id": exp_id,
                "compute": {
                    "used_gpu_hours": self.actual_costs.get(exp_id, 0.0),
                    "estimated_gpu_hours": self.estimated_costs.get(exp_id, 0.0),
                    "estimated_cost_usd": self.compute.estimate_cost(
                        self.actual_costs.get(exp_id, 0.0)
                    ),
                },
            }

            if exp_id in self.experiment_budgets:
                status["budget"] = self.experiment_budgets[exp_id].get_progress()

            return status

    def get_global_status(self) -> Dict[str, Any]:
        """Get global budget status across all experiments."""
        with self._lock:
            total_actual = sum(self.actual_costs.values())
            total_estimated = sum(self.estimated_costs.values())

            return {
                "compute": {
                    "total_used_gpu_hours": total_actual,
                    "total_estimated_gpu_hours": total_estimated,
                    "daily_remaining": self.compute.remaining_daily(),
                    "weekly_remaining": self.compute.remaining_weekly(),
                    "total_estimated_cost_usd": self.compute.estimate_cost(total_actual),
                },
                "global_experiment": self.global_budget.get_progress(),
                "active_experiments": len(self.experiment_budgets),
                "experiments": {
                    exp_id: self.get_experiment_status(exp_id)
                    for exp_id in self.experiment_budgets
                },
            }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        with self._lock:
            total_actual = sum(self.actual_costs.values())
            total_estimated = sum(self.estimated_costs.values())

            return {
                "total_actual_gpu_hours": total_actual,
                "total_estimated_gpu_hours": total_estimated,
                "variance": total_actual - total_estimated,
                "variance_ratio": (total_actual - total_estimated) / max(total_estimated, 1e-6),
                "total_cost_usd": self.compute.estimate_cost(total_actual),
                "by_experiment": {
                    exp_id: {
                        "actual": self.actual_costs.get(exp_id, 0.0),
                        "estimated": self.estimated_costs.get(exp_id, 0.0),
                        "variance": self.actual_costs.get(exp_id, 0.0) - self.estimated_costs.get(exp_id, 0.0),
                    }
                    for exp_id in self.actual_costs
                },
            }

    def reset_experiment(self, exp_id: str) -> None:
        """Reset budget tracking for an experiment."""
        with self._lock:
            if exp_id in self.experiment_budgets:
                del self.experiment_budgets[exp_id]
            self.actual_costs.pop(exp_id, None)
            self.estimated_costs.pop(exp_id, None)


# ---------------------------------------------------------------------------
# Early Stopping Callbacks
# ---------------------------------------------------------------------------

class EarlyStoppingCallback:
    """Callback for framework-specific early stopping (Optuna, etc.)."""

    def __init__(self, budget_manager: BudgetManager, exp_id: str):
        self.budget_manager = budget_manager
        self.exp_id = exp_id

    def __call__(self, study: Any, trial: Any) -> None:
        """Optuna study pruner callback."""
        if len(study.trials) == 0:
            return

        # Get best score so far
        best_trial = study.best_trial
        score = best_trial.value if best_trial else float("-inf")

        should_stop, reason = self.budget_manager.report_trial(self.exp_id, score)
        if should_stop:
            study.stop()


class MedianPrunerWithBudget:
    """Optuna pruner that also respects budget constraints."""

    def __init__(
        self,
        budget_manager: BudgetManager,
        exp_id: str,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
    ):
        self.budget_manager = budget_manager
        self.exp_id = exp_id
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps

    def __call__(self, study: Any, trial: Any) -> bool:
        """Return True if trial should be pruned."""
        # Check budget first
        should_stop, _ = self.budget_manager.report_trial(
            self.exp_id,
            trial.value if hasattr(trial, "value") and trial.value is not None else float("-inf"),
        )
        if should_stop:
            return True

        # Standard median pruner logic
        if len(study.trials) < self.n_startup_trials:
            return False

        # Get completed trials at this step
        step = trial.last_step or 0
        if step < self.n_warmup_steps:
            return False

        if step % self.interval_steps != 0:
            return False

        # Compare to median
        scores = []
        for t in study.trials:
            if t.state.is_complete() and t.last_step is not None and t.last_step >= step:
                if hasattr(t, "intermediate_values") and step in t.intermediate_values:
                    scores.append(t.intermediate_values[step])

        if len(scores) == 0:
            return False

        median = sorted(scores)[len(scores) // 2]
        current = trial.intermediate_values.get(step, float("-inf"))

        return current < median


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def estimate_gpu_hours(
    n_trials: int,
    iterations_per_trial: int,
    seconds_per_iteration: float,
    n_gpus: int = 1,
) -> float:
    """Estimate GPU hours for an experiment."""
    total_seconds = n_trials * iterations_per_trial * seconds_per_iteration
    total_hours = total_seconds / 3600.0
    return total_hours / max(n_gpus, 1)


def estimate_experiment_duration(
    n_trials: int,
    iterations_per_trial: int,
    batch_size: int,
    env_steps_per_sec: float,
    n_envs: int = 1,
) -> float:
    """Estimate wall-clock duration in seconds."""
    total_env_steps = n_trials * iterations_per_trial * batch_size
    return total_env_steps / max(env_steps_per_sec * n_envs, 1e-6)
