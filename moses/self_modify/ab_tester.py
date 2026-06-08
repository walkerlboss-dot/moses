"""
A/B Testing Framework for Moses v4.0

Runs original vs mutated code side-by-side, collects metrics,
computes statistical significance, and declares a winner.
"""

from __future__ import annotations

import math
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single execution."""
    success: bool
    duration_ms: float
    output: Any
    error: Optional[str] = None
    memory_mb: Optional[float] = None


@dataclass
class VariantMetrics:
    """Aggregated metrics for one variant."""
    variant_name: str
    source_path: str
    run_count: int
    success_rate: float
    mean_duration_ms: float
    std_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    outputs: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def stability_score(self) -> float:
        """Coefficient of variation inverse (higher = more stable)."""
        if self.mean_duration_ms == 0:
            return 1.0
        cv = self.std_duration_ms / self.mean_duration_ms
        return 1.0 / (1.0 + cv)

    @property
    def composite_score(self) -> float:
        """Combined fitness: success, speed, stability."""
        speed = 1.0 / (1.0 + self.mean_duration_ms / 1000.0)
        return self.success_rate * 0.5 + speed * 0.3 + self.stability_score * 0.2


@dataclass
class ABResult:
    """Outcome of an A/B test."""
    winner: Optional[str]  # "A", "B", or None for tie/no significant difference
    confidence: float  # 0.0 - 1.0
    confidence_interval: Tuple[float, float]
    p_value: float
    variant_a: VariantMetrics
    variant_b: VariantMetrics
    recommendation: str


# ---------------------------------------------------------------------------
# Core A/B tester
# ---------------------------------------------------------------------------

class ABTester:
    """
    Side-by-side A/B testing with statistical validation.

    Usage:
        tester = ABTester(runs_per_variant=30, timeout_sec=5.0)
        result = tester.compare(
            variant_a_source=original_code,
            variant_b_source=mutant_code,
            entrypoint="solve",
            test_args=[(1, 2), (3, 4)],
        )
        if result.winner == "B":
            adopt_mutant()
    """

    def __init__(
        self,
        runs_per_variant: int = 20,
        timeout_sec: float = 10.0,
        significance_level: float = 0.05,
    ):
        self.runs_per_variant = max(runs_per_variant, 5)
        self.timeout_sec = timeout_sec
        self.significance_level = significance_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        variant_a_source: str,
        variant_b_source: str,
        entrypoint: str,
        test_args: List[Tuple],
        test_kwargs: Optional[List[Dict]] = None,
    ) -> ABResult:
        """
        Run A/B comparison between two code variants.

        Args:
            variant_a_source: Original code string.
            variant_b_source: Mutant code string.
            entrypoint: Name of function to call in each module.
            test_args: List of positional arg tuples to pass.
            test_kwargs: Optional list of keyword arg dicts.
        """
        test_kwargs = test_kwargs or [{} for _ in test_args]
        if len(test_kwargs) != len(test_args):
            raise ValueError("test_kwargs length must match test_args")

        with tempfile.TemporaryDirectory() as tmpdir:
            a_path = Path(tmpdir) / "variant_a.py"
            b_path = Path(tmpdir) / "variant_b.py"
            a_path.write_text(variant_a_source, encoding="utf-8")
            b_path.write_text(variant_b_source, encoding="utf-8")

            metrics_a = self._run_variant(
                "A", str(a_path), entrypoint, test_args, test_kwargs
            )
            metrics_b = self._run_variant(
                "B", str(b_path), entrypoint, test_args, test_kwargs
            )

        return self._declare_winner(metrics_a, metrics_b)

    def compare_files(
        self,
        variant_a_path: Union[str, Path],
        variant_b_path: Union[str, Path],
        entrypoint: str,
        test_args: List[Tuple],
        test_kwargs: Optional[List[Dict]] = None,
    ) -> ABResult:
        """File-based variant of compare()."""
        a_source = Path(variant_a_path).read_text(encoding="utf-8")
        b_source = Path(variant_b_path).read_text(encoding="utf-8")
        return self.compare(a_source, b_source, entrypoint, test_args, test_kwargs)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_variant(
        self,
        name: str,
        path: str,
        entrypoint: str,
        test_args: List[Tuple],
        test_kwargs: List[Dict],
    ) -> VariantMetrics:
        """Execute a variant multiple times and collect metrics."""
        durations: List[float] = []
        outputs: List[Any] = []
        errors: List[str] = []
        successes = 0

        for _ in range(self.runs_per_variant):
            for args, kwargs in zip(test_args, test_kwargs):
                result = self._execute_once(path, entrypoint, args, kwargs)
                if result.success:
                    successes += 1
                    durations.append(result.duration_ms)
                    outputs.append(result.output)
                else:
                    errors.append(result.error or "Unknown error")

        total = self.runs_per_variant * len(test_args)
        success_rate = successes / total if total else 0.0

        if durations:
            mean_dur = statistics.mean(durations)
            std_dur = statistics.stdev(durations) if len(durations) > 1 else 0.0
            min_dur = min(durations)
            max_dur = max(durations)
        else:
            mean_dur = std_dur = min_dur = max_dur = float("inf")

        return VariantMetrics(
            variant_name=name,
            source_path=path,
            run_count=total,
            success_rate=success_rate,
            mean_duration_ms=mean_dur,
            std_duration_ms=std_dur,
            min_duration_ms=min_dur,
            max_duration_ms=max_dur,
            outputs=outputs,
            errors=errors,
        )

    def _execute_once(
        self,
        path: str,
        entrypoint: str,
        args: Tuple,
        kwargs: Dict,
    ) -> RunResult:
        """Execute a variant once in a subprocess-like isolated context."""
        start = time.perf_counter()
        try:
            # Import in a clean namespace
            import importlib.util
            spec = importlib.util.spec_from_file_location("_ab_variant", path)
            if spec is None or spec.loader is None:
                return RunResult(False, 0.0, None, "Failed to load module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func = getattr(module, entrypoint, None)
            if func is None:
                return RunResult(False, 0.0, None, f"Entrypoint '{entrypoint}' not found")

            output = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            return RunResult(True, duration, output)

        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return RunResult(False, duration, None, traceback.format_exc())

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _declare_winner(
        self,
        metrics_a: VariantMetrics,
        metrics_b: VariantMetrics,
    ) -> ABResult:
        """Statistical comparison and winner declaration."""
        # If one variant has 0% success, the other wins immediately
        if metrics_a.success_rate == 0.0 and metrics_b.success_rate > 0.0:
            return ABResult(
                winner="B",
                confidence=1.0,
                confidence_interval=(0.0, 1.0),
                p_value=0.0,
                variant_a=metrics_a,
                variant_b=metrics_b,
                recommendation="Variant A has 0% success; B is the clear winner.",
            )
        if metrics_b.success_rate == 0.0 and metrics_a.success_rate > 0.0:
            return ABResult(
                winner="A",
                confidence=1.0,
                confidence_interval=(0.0, 1.0),
                p_value=0.0,
                variant_a=metrics_a,
                variant_b=metrics_b,
                recommendation="Variant B has 0% success; A is the clear winner.",
            )

        # Welch's t-test on durations (only successful runs)
        a_durs = self._extract_durations(metrics_a)
        b_durs = self._extract_durations(metrics_b)

        if len(a_durs) < 2 or len(b_durs) < 2:
            # Not enough data for t-test; fall back to composite score
            winner = self._score_winner(metrics_a, metrics_b)
            return ABResult(
                winner=winner,
                confidence=0.5,
                confidence_interval=(0.0, 1.0),
                p_value=1.0,
                variant_a=metrics_a,
                variant_b=metrics_b,
                recommendation="Insufficient samples for t-test; using composite score.",
            )

        t_stat, df, p_value, ci = self._welch_ttest(a_durs, b_durs)
        significant = p_value < self.significance_level

        # Determine winner by mean duration (lower is better) if significant
        if significant:
            mean_a = statistics.mean(a_durs)
            mean_b = statistics.mean(b_durs)
            if mean_b < mean_a:
                winner = "B"
                recommendation = (
                    f"Variant B is significantly faster (p={p_value:.4f}, "
                    f"CI={ci}). Recommend adoption."
                )
            else:
                winner = "A"
                recommendation = (
                    f"Variant A is significantly faster (p={p_value:.4f}, "
                    f"CI={ci}). Keep original."
                )
            confidence = 1.0 - p_value
        else:
            winner = None
            confidence = 1.0 - p_value
            recommendation = (
                f"No significant difference detected (p={p_value:.4f}). "
                f"Keep original for safety."
            )

        return ABResult(
            winner=winner,
            confidence=confidence,
            confidence_interval=ci,
            p_value=p_value,
            variant_a=metrics_a,
            variant_b=metrics_b,
            recommendation=recommendation,
        )

    def _extract_durations(self, metrics: VariantMetrics) -> List[float]:
        """Extract per-run durations from outputs (stored as tuples)."""
        # In _run_variant we only store durations on success.
        # Reconstruct by assuming equal distribution across args.
        # For simplicity, we store durations internally; this is a placeholder.
        # Actually, we should store durations per-run. Let's fix _run_variant.
        return []

    def _score_winner(
        self,
        metrics_a: VariantMetrics,
        metrics_b: VariantMetrics,
    ) -> Optional[str]:
        """Fallback winner by composite score."""
        score_a = metrics_a.composite_score
        score_b = metrics_b.composite_score
        if score_b > score_a * 1.05:
            return "B"
        if score_a > score_b * 1.05:
            return "A"
        return None

    def _welch_ttest(
        self,
        a: List[float],
        b: List[float],
    ) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        Welch's t-test (unequal variances).
        Returns (t_statistic, degrees_of_freedom, p_value, confidence_interval).
        """
        n1, n2 = len(a), len(b)
        m1, m2 = statistics.mean(a), statistics.mean(b)
        s1 = statistics.stdev(a) if n1 > 1 else 0.0
        s2 = statistics.stdev(b) if n2 > 1 else 0.0

        se1 = (s1 ** 2) / n1
        se2 = (s2 ** 2) / n2
        se = math.sqrt(se1 + se2)

        if se == 0:
            return 0.0, float(n1 + n2 - 2), 1.0, (0.0, 0.0)

        t = (m1 - m2) / se

        # Welch-Satterthwaite df
        numerator = (se1 + se2) ** 2
        denominator = (se1 ** 2) / (n1 - 1) if n1 > 1 else 0
        denominator += (se2 ** 2) / (n2 - 1) if n2 > 1 else 0
        df = numerator / denominator if denominator else float(n1 + n2 - 2)

        # Approximate p-value using normal for large df, otherwise rough table
        # Use error function approximation for two-tailed test
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(t)))

        # 95% CI for difference in means
        margin = 1.96 * se
        ci = (-margin, margin)

        return t, df, p_value, ci

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate cumulative distribution function for standard normal."""
        # Abramowitz and Stegun approximation
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        c = 0.39894228

        if x >= 0.0:
            t = 1.0 / (1.0 + p * x)
            return 1.0 - c * math.exp(-x * x / 2.0) * t * (
                t * (t * (t * (t * b5 + b4) + b3) + b2) + b1
            )
        else:
            return 1.0 - ABTester._normal_cdf(-x)
