"""
Moses Continuous Training Pipeline
==================================
Orchestrates the end-to-end lifecycle of a training run:

    trigger → data validation → preprocessing → training → evaluation → registry

Triggers
--------
- **Event-based**: new data arrival, performance degradation alert.
- **Schedule-based**: cron expressions managed by :class:`TrainingScheduler`.
- **Manual**: ad-hoc API or CLI invocation.

Configuration
-------------
All stages are config-driven via YAML. Example::

    pipeline:
      name: "ppo_continuous"
      stages:
        - name: validate
          module: moses.training.stages.validate
          retries: 2
        - name: preprocess
          module: moses.training.stages.preprocess
          retries: 1
        - name: train
          module: moses.training.stages.train
          retries: 2
          fallback: evaluate  # on failure, skip to eval with last good checkpoint
        - name: evaluate
          module: moses.training.stages.evaluate
          retries: 1
        - name: register
          module: moses.training.stages.register
          retries: 0
      alerting:
        webhook: "https://hooks.slack.com/services/..."
      tracking:
        wandb:
          project: "moses-rl"
          entity: "alex-lab"
        mlflow:
          tracking_uri: "http://localhost:5000"

Dependencies
------------
- ``prefect`` (optional) for workflow orchestration and observability.
- ``pydantic`` for config validation.
- Existing Moses infra: scheduler, registry, data sources.

Example
-------
>>> from moses.training.pipeline import ContinuousTrainingPipeline
>>> pipe = ContinuousTrainingPipeline.from_yaml("pipeline.yaml")
>>> pipe.run(trigger="new_data", dataset_path="/data/batch_42")
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import yaml

from moses.training.registry import ModelRegistry
from moses.training.scheduler import JobSpec, PreemptionPolicy, TrainingScheduler

logger = logging.getLogger("moses.training.pipeline")

# Optional Prefect integration
try:
    from prefect import flow, task  # type: ignore[import-untyped]
    from prefect.states import State  # type: ignore[import-untyped]

    _PREFECT_AVAILABLE = True
except Exception:
    _PREFECT_AVAILABLE = False
    flow = lambda **kw: lambda f: f  # type: ignore[assignment]
    task = lambda **kw: lambda f: f  # type: ignore[assignment]

# Optional W&B integration
try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False
    wandb = None  # type: ignore[misc,assignment]

# Optional MLflow integration
try:
    import mlflow  # type: ignore[import-untyped]

    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StageResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    FALLBACK = "fallback"


@dataclass
class StageConfig:
    name: str
    module: str
    retries: int = 0
    fallback: Optional[str] = None
    timeout_seconds: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    name: str
    stages: List[StageConfig]
    alerting: Dict[str, Any] = field(default_factory=dict)
    tracking: Dict[str, Any] = field(default_factory=dict)
    artifact_store: str = "./moses_artifacts"
    registry_backend: str = "sqlite:///moses_registry.db"


@dataclass
class PipelineContext:
    """Mutable bag passed through every stage."""

    pipeline_name: str
    trigger: str
    run_id: str
    config: PipelineConfig
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Union[str, Path]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

class AlertManager:
    """Simple alert dispatcher (webhook, email, log)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def send(self, subject: str, body: str, level: str = "warning") -> None:
        logger.log(
            getattr(logging, level.upper(), logging.WARNING),
            "[ALERT] %s | %s",
            subject,
            body,
        )
        webhook = self._config.get("webhook")
        if webhook:
            self._post_webhook(webhook, subject, body)

    def _post_webhook(self, url: str, subject: str, body: str) -> None:
        try:
            import urllib.request

            payload = json.dumps({"text": f"*{subject}*\n{body}"}).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.debug("Webhook response: %s", resp.status)
        except Exception as exc:
            logger.error("Webhook alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Stage loader
# ---------------------------------------------------------------------------

class Stage:
    """Abstract base for a pipeline stage."""

    def run(self, ctx: PipelineContext) -> StageResult:
        raise NotImplementedError


def _load_stage_class(module_path: str) -> Type[Stage]:
    """Dynamically import ``module_path`` and return its ``Stage`` subclass."""
    mod_name, cls_name = module_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not issubclass(cls, Stage):
        raise TypeError(f"{module_path} does not subclass Stage")
    return cls


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ContinuousTrainingPipeline:
    """
    End-to-end continuous training pipeline for Moses.

    Parameters
    ----------
    config :
        Parsed pipeline configuration.
    registry :
        Model registry instance (optional — created from config if omitted).
    scheduler :
        Training scheduler instance (optional).
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: Optional[ModelRegistry] = None,
        scheduler: Optional[TrainingScheduler] = None,
    ) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(
            backend_url=config.registry_backend,
            artifact_store=config.artifact_store,
            alert_channel=None,  # uses AlertManager instead
        )
        self.scheduler = scheduler
        self._alert = AlertManager(config.alerting)
        self._stages: Dict[str, Type[Stage]] = {}
        self._init_tracking()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ContinuousTrainingPipeline":
        """Load pipeline from a YAML configuration file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        cfg = raw.get("pipeline", raw)
        stages = [
            StageConfig(
                name=s["name"],
                module=s["module"],
                retries=s.get("retries", 0),
                fallback=s.get("fallback"),
                timeout_seconds=s.get("timeout_seconds"),
                params=s.get("params", {}),
            )
            for s in cfg["stages"]
        ]
        config = PipelineConfig(
            name=cfg["name"],
            stages=stages,
            alerting=raw.get("alerting", {}),
            tracking=raw.get("tracking", {}),
            artifact_store=cfg.get("artifact_store", "./moses_artifacts"),
            registry_backend=cfg.get("registry_backend", "sqlite:///moses_registry.db"),
        )
        return cls(config)

    # -- Tracking setup ------------------------------------------------------

    def _init_tracking(self) -> None:
        self._wandb_run: Optional[Any] = None
        if _WANDB_AVAILABLE and "wandb" in self.config.tracking:
            wandb_cfg = self.config.tracking["wandb"]
            wandb.init(
                project=wandb_cfg.get("project", "moses"),
                entity=wandb_cfg.get("entity"),
                config={"pipeline": self.config.name},
            )
            self._wandb_run = wandb.run

        if _MLFLOW_AVAILABLE and "mlflow" in self.config.tracking:
            mlflow_cfg = self.config.tracking["mlflow"]
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
            mlflow.set_experiment(self.config.name)

    # -- Public API ----------------------------------------------------------

    def run(
        self,
        trigger: str,
        dataset_path: Optional[Union[str, Path]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineContext:
        """
        Execute the full pipeline.

        Parameters
        ----------
        trigger :
            Reason for the run (``new_data``, ``schedule``, ``degradation``).
        dataset_path :
            Path to the new dataset (if applicable).
        extra_context :
            Additional key/value pairs injected into the pipeline context.
        """
        run_id = f"{self.config.name}-{int(time.time())}"
        ctx = PipelineContext(
            pipeline_name=self.config.name,
            trigger=trigger,
            run_id=run_id,
            config=self.config,
            data={"dataset_path": str(dataset_path)} if dataset_path else {},
        )
        if extra_context:
            ctx.data.update(extra_context)

        logger.info("Pipeline %s started (run_id=%s, trigger=%s)", self.config.name, run_id, trigger)

        if _PREFECT_AVAILABLE:
            return self._run_prefect(ctx)
        return self._run_native(ctx)

    def schedule_recurring(self, cron: str, priority: int = 5) -> None:
        """Submit this pipeline as a recurring scheduled job."""
        if self.scheduler is None:
            raise RuntimeError("No scheduler attached to pipeline")
        job = JobSpec(
            name=f"{self.config.name}_scheduled",
            priority=priority,
            cron=cron,
            preemption_policy=PreemptionPolicy.YIELD,
        )
        self.scheduler.submit(job)
        logger.info("Scheduled pipeline %s with cron '%s'", self.config.name, cron)

    # -- Execution engines ---------------------------------------------------

    def _run_native(self, ctx: PipelineContext) -> PipelineContext:
        """Execute stages sequentially with retry / fallback logic."""
        stage_index = 0
        while stage_index < len(self.config.stages):
            stage_cfg = self.config.stages[stage_index]
            result = self._execute_stage(stage_cfg, ctx)

            if result == StageResult.SUCCESS:
                stage_index += 1
                continue

            if result == StageResult.FALLBACK and stage_cfg.fallback:
                # Jump to fallback stage
                fallback_names = [s.name for s in self.config.stages]
                try:
                    stage_index = fallback_names.index(stage_cfg.fallback)
                    logger.info("Falling back to stage '%s'", stage_cfg.fallback)
                    continue
                except ValueError:
                    ctx.errors.append(f"Fallback stage '{stage_cfg.fallback}' not found")

            # Failure or unhandled fallback
            self._alert.send(
                subject=f"Pipeline {ctx.run_id} failed at stage '{stage_cfg.name}'",
                body="\n".join(ctx.errors),
                level="error",
            )
            raise PipelineError(f"Stage '{stage_cfg.name}' failed: {ctx.errors[-1]}")

        logger.info("Pipeline %s completed successfully", ctx.run_id)
        return ctx

    @flow(name="moses-continuous-training")  # type: ignore[misc]
    def _run_prefect(self, ctx: PipelineContext) -> PipelineContext:
        """Prefect-native flow execution with task-level observability."""
        for stage_cfg in self.config.stages:
            result = self._execute_stage_prefect(stage_cfg, ctx)
            if result != StageResult.SUCCESS:
                self._alert.send(
                    subject=f"Pipeline {ctx.run_id} failed at stage '{stage_cfg.name}'",
                    body="\n".join(ctx.errors),
                    level="error",
                )
                raise PipelineError(f"Stage '{stage_cfg.name}' failed")
        logger.info("Pipeline %s completed successfully (Prefect)", ctx.run_id)
        return ctx

    # -- Stage execution -----------------------------------------------------

    def _execute_stage(self, cfg: StageConfig, ctx: PipelineContext) -> StageResult:
        """Run a single stage with retries and timeout."""
        logger.info("Executing stage '%s' (retries=%d)", cfg.name, cfg.retries)
        attempt = 0
        while attempt <= cfg.retries:
            try:
                stage_cls = self._stages.get(cfg.name) or _load_stage_class(cfg.module)
                self._stages[cfg.name] = stage_cls
                stage = stage_cls()

                if cfg.timeout_seconds:
                    # Simple timeout via signal (Unix) or threading
                    result = self._run_with_timeout(stage, ctx, cfg.timeout_seconds)
                else:
                    result = stage.run(ctx)

                if result == StageResult.SUCCESS:
                    self._log_stage_metrics(cfg.name, ctx)
                    return StageResult.SUCCESS

                if result == StageResult.FALLBACK:
                    return StageResult.FALLBACK

                # Treat anything else as failure for retry purposes
                raise StageFailure(f"Stage returned {result.value}")

            except Exception as exc:
                attempt += 1
                msg = f"Stage '{cfg.name}' attempt {attempt} failed: {exc}"
                logger.exception(msg)
                ctx.errors.append(msg)
                if attempt > cfg.retries:
                    break
                time.sleep(2 ** attempt)  # exponential back-off

        return StageResult.FAILURE

    @task(name="execute-stage", retries=0)  # type: ignore[misc]
    def _execute_stage_prefect(self, cfg: StageConfig, ctx: PipelineContext) -> StageResult:
        """Prefect-wrapped stage execution (retry handled by Prefect if configured)."""
        return self._execute_stage(cfg, ctx)

    def _run_with_timeout(
        self, stage: Stage, ctx: PipelineContext, timeout: float
    ) -> StageResult:
        """Run a stage with a timeout (best-effort via threading)."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stage.run, ctx)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                ctx.errors.append(f"Stage timed out after {timeout}s")
                return StageResult.FAILURE

    def _log_stage_metrics(self, stage_name: str, ctx: PipelineContext) -> None:
        """Push metrics to W&B / MLflow."""
        metrics = {f"{stage_name}/{k}": v for k, v in ctx.metrics.items()}
        if self._wandb_run:
            self._wandb_run.log(metrics)
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            for k, v in metrics.items():
                # MLflow requires scalar metrics
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Raised when the pipeline cannot recover from a stage failure."""


class StageFailure(Exception):
    """Raised internally when a stage returns a non-success result."""


# ---------------------------------------------------------------------------
# Convenience CLI / entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Moses Continuous Training Pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML")
    parser.add_argument("--trigger", default="manual", help="Trigger type")
    parser.add_argument("--dataset", default=None, help="Dataset path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = ContinuousTrainingPipeline.from_yaml(args.config)
    ctx = pipeline.run(trigger=args.trigger, dataset_path=args.dataset)
    print(json.dumps(ctx.metrics, indent=2))


if __name__ == "__main__":
    main()
