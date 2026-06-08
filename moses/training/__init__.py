"""
Moses Training Package
======================
Continuous training infrastructure for the Moses agent system.

Modules
-------
pipeline :
    End-to-end training pipeline with configurable stages, retry logic,
    fallback handling, and integrations with W&B / MLflow / Prefect.
scheduler :
    Priority-queue-based job scheduler with GPU-aware resource allocation,
    preemption policies, and optional Celery distributed execution.
registry :
    Model version control, artifact storage, lineage tracking, and tagging
    (production / staging / experiment). Supports SQLite and MLflow backends.

Quick Start
-----------
>>> from moses.training import ContinuousTrainingPipeline, TrainingScheduler, ModelRegistry
>>> pipe = ContinuousTrainingPipeline.from_yaml("pipeline.yaml")
>>> ctx = pipe.run(trigger="new_data", dataset_path="/data/batch_7")

>>> sched = TrainingScheduler.from_yaml("scheduler.yaml")
>>> sched.submit(JobSpec(name="nightly-ppo", priority=10, gpu_count=2))
>>> sched.run_forever()

>>> reg = ModelRegistry(backend_url="sqlite:///moses_registry.db")
>>> version = reg.register(run_name="ppo-v5", checkpoint_path="ckpt.pt")
>>> reg.promote(version.version_id, "production")
"""

from moses.training.pipeline import (
    ContinuousTrainingPipeline,
    PipelineConfig,
    PipelineContext,
    Stage,
    StageConfig,
    StageResult,
)
from moses.training.registry import (
    AlertChannel,
    ModelRegistry,
    ModelVersion,
)
from moses.training.scheduler import (
    JobRecord,
    JobSpec,
    JobStatus,
    PreemptionPolicy,
    ResourcePool,
    TrainingScheduler,
)

__all__ = [
    # Pipeline
    "ContinuousTrainingPipeline",
    "PipelineConfig",
    "PipelineContext",
    "Stage",
    "StageConfig",
    "StageResult",
    # Registry
    "AlertChannel",
    "ModelRegistry",
    "ModelVersion",
    # Scheduler
    "JobRecord",
    "JobSpec",
    "JobStatus",
    "PreemptionPolicy",
    "ResourcePool",
    "TrainingScheduler",
]
