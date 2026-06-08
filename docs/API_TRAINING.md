# Training API — Moses

> **Continuous training pipeline for humanoid robots.**

---

## TrainingPipeline

```python
from moses.training.pipeline import TrainingPipeline
```

Continuous training pipeline with triggers, stages, and error handling.

### Constructor

```python
TrainingPipeline(
    config: PipelineConfig,  # YAML-defined pipeline
    registry: ModelRegistry | None = None,
    scheduler: TrainingScheduler | None = None,
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `run()` | `trigger="new_data"` | `dict` | Execute full pipeline |
| `validate_data()` | `dataset` | `bool` | Validate data schema |
| `preprocess()` | `dataset` | `Dataset` | Preprocess data |
| `train()` | `dataset`, `config` | `Policy` | Train policy |
| `evaluate()` | `policy`, `env` | `dict` | Evaluate policy |
| `register()` | `policy`, `metrics` | `str` | Register model (returns SHA) |

### Triggers

| Trigger | Description |
|---------|-------------|
| `new_data` | New data arrived in ingestion pipeline |
| `schedule` | Cron-based scheduled training |
| `degradation` | Production performance dropped below threshold |
| `manual` | Human-triggered training |

### Pipeline Stages

```yaml
stages:
  - name: validate
    retries: 3
    timeout: 300
    fallback: skip
  - name: preprocess
    retries: 2
    timeout: 600
    fallback: abort
  - name: train
    retries: 1
    timeout: 86400
    fallback: notify
  - name: evaluate
    retries: 2
    timeout: 3600
    fallback: rollback
  - name: register
    retries: 3
    timeout: 60
    fallback: abort
```

### Example

```python
from moses.training import TrainingPipeline

pipeline = TrainingPipeline.from_yaml("configs/pipeline.yaml")
result = pipeline.run(trigger="new_data")
# Returns: {"model_sha": "abc123", "metrics": {...}, "status": "success"}
```

---

## TrainingScheduler

```python
from moses.training.scheduler import TrainingScheduler
```

Priority queue with GPU allocation and preemption.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `submit_job()` | `job_config`, `priority` | `str` | Submit job, return ID |
| `get_queue()` | — | `list` | Current queue |
| `preempt_job()` | `job_id`, `strategy` | `bool` | Preempt low-priority job |
| `allocate_gpu()` | `job_id`, `gpus` | `bool` | Allocate GPUs |

### Preemption Strategies

| Strategy | Behavior |
|----------|----------|
| `YIELD` | Pause job, resume later |
| `KILL` | Terminate job, lose progress |
| `NONE` | No preemption, wait for completion |

---

## ModelRegistry

```python
from moses.training.registry import ModelRegistry
```

Version-controlled model registry with lineage.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `register()` | `policy`, `metrics`, `config` | `str` | Register model, return SHA |
| `get_model()` | `sha` | `Policy` | Load model by SHA |
| `list_models()` | `tags=None` | `list` | List registered models |
| `get_lineage()` | `sha` | `dict` | Parent → child relationships |
| `tag()` | `sha`, `tag` | — | Tag model ("production", "staging") |

---

*See `configs/train_ppo.yaml` for training configuration.*
