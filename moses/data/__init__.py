"""
moses/data
==========
Data pipeline for Moses v5.0 continuous training.

Submodules:
  • ingestion      — Multi-source data ingestion (Isaac Lab, robot logs, human demos)
  • preprocessing  — Normalisation, augmentation, filtering, balancing
  • storage        — Efficient storage with versioning, indexing, and compression

Quick start:
    from moses.data import ingest, DatasetStore, PreprocessingPipeline

    # Ingest raw data
    for episode in ingest("/data/isaac_lab_rollouts", schema="isaac_lab_hdf5"):
        ...

    # Preprocess
    pipe = PreprocessingPipeline()
    pipe.add(ObservationNormalizer().fit(train_eps))
    pipe.add(ActionClipper())
    for ep in pipe.apply(raw_eps):
        ...

    # Store
    store = DatasetStore("/data/moses_dataset")
    version = store.create_version(description="v1.0 sim + real")
    store.write_episodes(episodes, version=version)
"""

from __future__ import annotations

# Ingestion
from moses.data.ingestion import (
    Episode,
    IngestionResult,
    DataIngestionPipeline,
    Deduplicator,
    FormatAdapter,
    IsaacLabHDF5Adapter,
    RobotCSVAdapter,
    HumanDemoJSONLAdapter,
    LeRobotV2Adapter,
    RLDSAdapter,
    detect_adapter,
    ingest,
    validate_episode,
    async_ingest_paths,
)

# Preprocessing
from moses.data.preprocessing import (
    Transform,
    IdentityTransform,
    ObservationNormalizer,
    ActionNormalizer,
    ActionClipper,
    RewardNormalizer,
    ActionNoise,
    TimeWarp,
    MirrorTransform,
    ImageColorJitter,
    LengthFilter,
    OutlierFilter,
    RewardFilter,
    SuccessRateFilter,
    TaskBalancer,
    QualityWeightedSampler,
    PreprocessingPipeline,
    standard_pipeline,
)

# Storage
from moses.data.storage import (
    DatasetVersion,
    DatasetStore,
    DatasetIndex,
    StreamingDataset,
    TFRecordWriter,
    init_store,
    episode_to_flat_dict,
    episode_to_zarr_group,
    episode_from_zarr_group,
)

__all__ = [
    # ingestion
    "Episode",
    "IngestionResult",
    "DataIngestionPipeline",
    "Deduplicator",
    "FormatAdapter",
    "IsaacLabHDF5Adapter",
    "RobotCSVAdapter",
    "HumanDemoJSONLAdapter",
    "LeRobotV2Adapter",
    "RLDSAdapter",
    "detect_adapter",
    "ingest",
    "validate_episode",
    "async_ingest_paths",
    # preprocessing
    "Transform",
    "IdentityTransform",
    "ObservationNormalizer",
    "ActionNormalizer",
    "ActionClipper",
    "RewardNormalizer",
    "ActionNoise",
    "TimeWarp",
    "MirrorTransform",
    "ImageColorJitter",
    "LengthFilter",
    "OutlierFilter",
    "RewardFilter",
    "SuccessRateFilter",
    "TaskBalancer",
    "QualityWeightedSampler",
    "PreprocessingPipeline",
    "standard_pipeline",
    # storage
    "DatasetVersion",
    "DatasetStore",
    "DatasetIndex",
    "StreamingDataset",
    "TFRecordWriter",
    "init_store",
    "episode_to_flat_dict",
    "episode_to_zarr_group",
    "episode_from_zarr_group",
]

__version__ = "5.0.0"
