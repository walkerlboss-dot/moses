# Data API — Moses

> **Multi-source data ingestion, preprocessing, and efficient storage.**

---

## DataIngestion

```python
from moses.data.ingestion import DataIngestion
```

Ingests data from multiple sources.

### Supported Sources

| Source | Format | Adapter |
|--------|--------|---------|
| Isaac Lab | HDF5 | `IsaacLabAdapter` |
| Physical Robot | CSV | `RobotCSVAdapter` |
| Human Demo | JSONL | `HumanDemoAdapter` |
| LeRobot v2 | Parquet | `LeRobotAdapter` |
| RLDS | TFRecord | `RLDSAdapter` |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `ingest()` | `source`, `path` | `Dataset` | Ingest data |
| `validate()` | `dataset` | `bool` | Schema validation |
| `deduplicate()` | `dataset` | `Dataset` | Remove duplicates |

### Deduplication

```python
ingestion = DataIngestion()

# Composite hash: trajectory + perceptual
dataset = ingestion.deduplicate(dataset, method="composite_hash")
# Removes episodes with same trajectory and visual hash
```

---

## PreprocessingPipeline

```python
from moses.data.preprocessing import PreprocessingPipeline
```

Composable preprocessing pipeline.

### Stages

| Stage | Description | Config |
|-------|-------------|--------|
| **Normalize** | Scale observations/actions | `method`: min-max, z-score |
| **Augment** | Data augmentation | `methods`: noise, time warp, mirror |
| **Filter** | Remove bad episodes | `criteria`: length, outliers, reward |
| **Balance** | Balance task distribution | `method`: oversample, undersample |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `add_stage()` | `stage`, `config` | — | Add pipeline stage |
| `process()` | `dataset` | `Dataset` | Run full pipeline |
| `get_stats()` | — | `dict` | Pipeline statistics |

### Example

```python
pipeline = PreprocessingPipeline()

pipeline.add_stage("normalize", {"method": "z-score"})
pipeline.add_stage("augment", {"methods": ["noise", "mirror"]})
pipeline.add_stage("filter", {"min_length": 100, "max_reward": 1000})
pipeline.add_stage("balance", {"method": "oversample", "target_ratio": 0.5})

processed = pipeline.process(raw_dataset)
```

---

## DatasetStore

```python
from moses.data.storage import DatasetStore
```

Efficient storage with versioning and lineage.

### Storage Format

| Component | Format | Purpose |
|-----------|--------|---------|
| Metadata | Parquet | Queryable metadata |
| Tensors | Zarr v3 | Chunked array storage |
| Index | SQLite | Fast lookups |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `write()` | `episodes` | `DatasetVersion` | Write dataset |
| `read()` | `version` | `Dataset` | Read dataset |
| `query()` | `filters` | `list` | Query episodes |
| `get_lineage()` | `version` | `list` | Version lineage |
| `stream()` | `version` | `iterator` | Streaming read |

### Versioning

```python
store = DatasetStore()

# Write dataset
v1 = store.write(episodes, metadata={"task": "walk", "date": "2026-06-08"})

# Query
results = store.query({"task": "walk", "success_rate": ">0.8"})

# Lineage
lineage = store.get_lineage(v1)
# Returns: [v1, v0] (parent chain)

# Stream for training
for batch in store.stream(v1, batch_size=256):
    train_step(batch)
```

---

*Data is the new oil. Clean data is the new gold.*
