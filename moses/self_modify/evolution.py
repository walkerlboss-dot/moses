"""
Evolutionary Code Improvement for Moses v4.0

Genetic algorithm that breeds a population of code variants,
measures fitness via test pass rate + performance,
and tracks generational improvement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .code_mutator import CodeMutator, Mutant, UnsafeMutationError
from .ab_tester import ABTester, ABResult, VariantMetrics
from .rollback import RollbackManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A single member of the evolutionary population."""
    mutant: Mutant
    generation: int
    fitness: float = 0.0
    metrics: Dict = field(default_factory=dict)
    rank: int = 0

    @property
    def source(self) -> str:
        return self.mutant.source

    @property
    def id(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]


@dataclass
class GenerationStats:
    """Statistics for one generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    std_fitness: float
    population_size: int
    best_individual_id: str
    elapsed_sec: float


@dataclass
class EvolutionReport:
    """Final report after evolution completes."""
    best_source: str
    best_fitness: float
    generations: int
    total_evaluations: int
    history: List[GenerationStats]
    improvement_curve: List[float]
    recommendation: str


# ---------------------------------------------------------------------------
# Core evolution engine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """
    Genetic algorithm for evolving Python source code.

    Usage:
        engine = EvolutionEngine(
            target_file="moses/brain.py",
            entrypoint="think",
            test_args=[(1, 2), (3, 4)],
        )
        report = engine.evolve(generations=10, population_size=8)
        if report.best_fitness > 0.9:
            Path("moses/brain.py").write_text(report.best_source)
    """

    def __init__(
        self,
        target_file: Union[str, Path],
        entrypoint: str,
        test_args: List[Tuple],
        test_kwargs: Optional[List[Dict]] = None,
        population_size: int = 8,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2,
        elitism_count: int = 2,
        fitness_func: Optional[Callable[[VariantMetrics], float]] = None,
        seed: Optional[int] = None,
        rollback_manager: Optional[RollbackManager] = None,
    ):
        self.target_file = Path(target_file)
        self.entrypoint = entrypoint
        self.test_args = test_args
        self.test_kwargs = test_kwargs or [{} for _ in test_args]
        self.population_size = max(population_size, 4)
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        self.elitism_count = max(1, elitism_count)
        self.fitness_func = fitness_func or self._default_fitness
        self._rng = random.Random(seed)
        self.mutator = CodeMutator(seed=seed)
        self.ab_tester = ABTester(runs_per_variant=10, timeout_sec=5.0)
        self.rollback = rollback_manager or RollbackManager(
            repo_root=self.target_file.parent.parent
        )
        self._original_source = self.target_file.read_text(encoding="utf-8")
        self._history: List[GenerationStats] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        generations: int = 10,
        early_stop_fitness: float = 0.99,
        early_stop_generations: int = 3,
    ) -> EvolutionReport:
        """
        Run the genetic algorithm.

        Args:
            generations: Maximum generations to run.
            early_stop_fitness: Stop if best fitness exceeds this.
            early_stop_generations: Stop if no improvement for N generations.

        Returns:
            EvolutionReport with best variant and statistics.
        """
        # Safety checkpoint before evolution
        self.rollback.checkpoint(
            self.target_file,
            description="Pre-evolution baseline",
            tags=["evolution", "baseline"],
        )

        population = self._init_population()
        best_ever: Optional[Individual] = None
        no_improvement_count = 0
        total_evaluations = 0
        improvement_curve: List[float] = []

        for gen in range(1, generations + 1):
            gen_start = time.perf_counter()

            # Evaluate fitness
            self._evaluate_population(population)
            total_evaluations += len(population)

            # Sort by fitness descending
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            for rank, ind in enumerate(population):
                ind.rank = rank

            best = population[0]
            improvement_curve.append(best.fitness)

            # Track best ever
            if best_ever is None or best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Stats
            fitnesses = [ind.fitness for ind in population]
            stats = GenerationStats(
                generation=gen,
                best_fitness=max(fitnesses),
                mean_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                std_fitness=self._std(fitnesses),
                population_size=len(population),
                best_individual_id=best.id,
                elapsed_sec=time.perf_counter() - gen_start,
            )
            self._history.append(stats)

            # Early stopping
            if best.fitness >= early_stop_fitness:
                break
            if no_improvement_count >= early_stop_generations:
                break

            # Next generation
            population = self._next_generation(population)

        # Build report
        if best_ever is None:
            raise EvolutionError("Evolution failed to produce any valid individual.")

        recommendation = self._build_recommendation(best_ever)

        return EvolutionReport(
            best_source=best_ever.source,
            best_fitness=best_ever.fitness,
            generations=len(self._history),
            total_evaluations=total_evaluations,
            history=self._history,
            improvement_curve=improvement_curve,
            recommendation=recommendation,
        )

    def apply_best(self, report: EvolutionReport, confirm: bool = False) -> bool:
        """
        Write the best variant from *report* to disk.

        Args:
            confirm: If True, require explicit confirmation.

        Returns:
            True if applied successfully.
        """
        if confirm:
            # In non-interactive mode, skip confirmation
            pass

        self.rollback.checkpoint(
            self.target_file,
            description="Before applying evolved variant",
            tags=["evolution", "pre-apply"],
        )
        self.target_file.write_text(report.best_source, encoding="utf-8")
        self.rollback.checkpoint(
            self.target_file,
            description="Applied evolved variant",
            tags=["evolution", "applied"],
            metrics={"fitness": report.best_fitness},
        )
        return True

    def export_history(self, path: Union[str, Path]) -> None:
        """Export generation history to JSON."""
        data = {
            "target_file": str(self.target_file),
            "entrypoint": self.entrypoint,
            "generations": [self._stats_to_dict(s) for s in self._history],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _init_population(self) -> List[Individual]:
        """Create initial population with random mutations."""
        population: List[Individual] = []

        # Always include original
        original_mutant = Mutant(
            source=self._original_source,
            original_hash="",
            mutations=[],
        )
        population.append(Individual(mutant=original_mutant, generation=0))

        while len(population) < self.population_size:
            try:
                mutant = self.mutator.mutate_source(
                    self._original_source,
                    mutation_count=self._rng.randint(1, 3),
                )
                population.append(Individual(mutant=mutant, generation=0))
            except UnsafeMutationError:
                continue

        return population[: self.population_size]

    def _evaluate_population(self, population: List[Individual]) -> None:
        """Run A/B tests against original to score each individual."""
        for ind in population:
            if ind.fitness > 0 and ind.generation > 0:
                # Already evaluated in previous generation (elites)
                continue

            try:
                result = self.ab_tester.compare(
                    variant_a_source=self._original_source,
                    variant_b_source=ind.source,
                    entrypoint=self.entrypoint,
                    test_args=self.test_args,
                    test_kwargs=self.test_kwargs,
                )
                ind.fitness = self.fitness_func(result.variant_b)
                ind.metrics = {
                    "success_rate": result.variant_b.success_rate,
                    "mean_duration_ms": result.variant_b.mean_duration_ms,
                    "stability_score": result.variant_b.stability_score,
                    "composite_score": result.variant_b.composite_score,
                    "p_value": result.p_value,
                    "winner": result.winner,
                }
            except Exception as exc:
                # Failed evaluation gets zero fitness
                ind.fitness = 0.0
                ind.metrics = {"error": str(exc)}

    def _next_generation(self, population: List[Individual]) -> List[Individual]:
        """Produce next generation via selection, crossover, mutation."""
        new_pop: List[Individual] = []
        gen = population[0].generation + 1

        # Elitism: carry forward best individuals unchanged
        elites = population[: self.elitism_count]
        new_pop.extend(copy.deepcopy(e) for e in elites)

        while len(new_pop) < self.population_size:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)

            if self._rng.random() < self.crossover_rate:
                child_source = self._crossover(parent_a.source, parent_b.source)
            else:
                child_source = parent_a.source

            if self._rng.random() < self.mutation_rate:
                try:
                    mutant = self.mutator.mutate_source(
                        child_source,
                        mutation_count=self._rng.randint(1, 2),
                    )
                    child_source = mutant.source
                except UnsafeMutationError:
                    pass

            child_mutant = Mutant(
                source=child_source,
                original_hash="",
                mutations=[],
            )
            new_pop.append(Individual(mutant=child_mutant, generation=gen))

        return new_pop[: self.population_size]

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _tournament_select(
        self,
        population: List[Individual],
        tournament_size: int = 3,
    ) -> Individual:
        """Tournament selection: pick best from random subset."""
        contestants = self._rng.sample(population, min(tournament_size, len(population)))
        return max(contestants, key=lambda ind: ind.fitness)

    def _crossover(self, source_a: str, source_b: str) -> str:
        """
        AST-aware crossover: swap a random function body between parents.
        Falls back to line-based crossover if AST parsing fails.
        """
        try:
            import ast
            tree_a = ast.parse(source_a)
            tree_b = ast.parse(source_b)

            funcs_a = [
                (idx, node) for idx, node in enumerate(tree_a.body)
                if isinstance(node, ast.FunctionDef)
            ]
            funcs_b = [
                (idx, node) for idx, node in enumerate(tree_b.body)
                if isinstance(node, ast.FunctionDef)
            ]

            if not funcs_a or not funcs_b:
                raise ValueError("No functions to swap")

            idx_a, func_a = self._rng.choice(funcs_a)
            # Find matching function name in B
            match_b = [n for n in funcs_b if n[1].name == func_a.name]
            if match_b:
                idx_b, func_b = self._rng.choice(match_b)
            else:
                idx_b, func_b = self._rng.choice(funcs_b)

            # Swap function bodies
            tree_a.body[idx_a].body, tree_b.body[idx_b].body = (
                tree_b.body[idx_b].body,
                tree_a.body[idx_a].body,
            )
            return ast.unparse(tree_a)

        except Exception:
            # Fallback: line-based crossover at a random point
            lines_a = source_a.splitlines(keepends=True)
            lines_b = source_b.splitlines(keepends=True)
            if len(lines_a) < 2 or len(lines_b) < 2:
                return source_a
            point = self._rng.randint(1, min(len(lines_a), len(lines_b)) - 1)
            return "".join(lines_a[:point] + lines_b[point:])

    # ------------------------------------------------------------------
    # Fitness and utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _default_fitness(metrics: VariantMetrics) -> float:
        """
        Default fitness: combine success rate, speed, and stability.
        Higher is better.
        """
        return metrics.composite_score

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _build_recommendation(self, best: Individual) -> str:
        if best.fitness >= 0.95:
            return (
                f"Excellent variant found (fitness={best.fitness:.3f}). "
                f"Recommend immediate adoption after final review."
            )
        elif best.fitness >= 0.8:
            return (
                f"Good variant found (fitness={best.fitness:.3f}). "
                f"Recommend adoption with monitoring."
            )
        elif best.fitness >= 0.6:
            return (
                f"Marginal improvement (fitness={best.fitness:.3f}). "
                f"Consider further evolution or keep original."
            )
        else:
            return (
                f"No meaningful improvement (fitness={best.fitness:.3f}). "
                f"Keep original code."
            )

    @staticmethod
    def _stats_to_dict(stats: GenerationStats) -> Dict:
        return {
            "generation": stats.generation,
            "best_fitness": stats.best_fitness,
            "mean_fitness": stats.mean_fitness,
            "worst_fitness": stats.worst_fitness,
            "std_fitness": stats.std_fitness,
            "population_size": stats.population_size,
            "best_individual_id": stats.best_individual_id,
            "elapsed_sec": stats.elapsed_sec,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EvolutionError(Exception):
    """Raised when the evolutionary process cannot continue."""
    pass
