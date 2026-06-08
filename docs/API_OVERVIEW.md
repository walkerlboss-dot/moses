# API Overview — Moses v6.0

> **Complete API reference for all Moses modules.**
> **Last Updated:** 2026-06-08

---

## Module Index

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `moses.envs` | Isaac Lab environments | `MosesHumanoidEnvCfg`, `MosesHumanoidEnv` |
| `moses.cad` | Parametric CAD design | `ParametricHumanoid`, `MeshGenerator`, `AssemblyManager` |
| `moses.manufacturing` | Manufacturing output | `CNCGenerator`, `PrintGenerator`, `SheetMetal`, `CarbonFiber` |
| `moses.design` | Design optimization | `StructuralAnalyzer`, `BiomechanicsDB`, `WeightOptimizer`, `CostModel` |
| `moses.gr00t` | NVIDIA GR00T integration | `Gr00TAdapter`, `GR00TFineTuner`, `EmbodimentConfig` |
| `moses.training` | Continuous training | `TrainingPipeline`, `TrainingScheduler`, `ModelRegistry` |
| `moses.experiments` | Auto experimentation | `ExperimentRunner`, `PPOSearchSpace`, `BudgetManager` |
| `moses.data` | Data pipeline | `DataIngestion`, `PreprocessingPipeline`, `DatasetStore` |
| `moses.deploy` | Safe deployment | `SafeDeploy`, `ShadowMode`, `ValidationPipeline` |
| `moses.monitoring` | Production monitoring | `ProductionMonitor`, `DriftDetector`, `DashboardApp` |
| `moses.meta_learning` | Meta-learning | `HyperparameterSearch`, `NeuralArchitectureSearch`, `CurriculumScheduler` |
| `moses.self_modify` | Self-modification | `CodeMutator`, `ABTester`, `RollbackManager`, `EvolutionEngine` |
| `moses.memory` | Knowledge accumulation | `ExperienceStore`, `CausalReasoner`, `KnowledgeGraph`, `TransferLearner` |
| `moses.safety` | Safety guardrails | `BoundsChecker`, `ApprovalGates`, `DriftDetector`, `IntegrityChecker` |
| `moses.recursion` | Deep recursion | `MetaMetaLearner`, `SelfHealing`, `WorldModel`, `Predictor` |
| `moses.emergence` | Emergent behavior | `SwarmIntelligence`, `BehaviorArbitrator`, `CuriosityEngine`, `ConsciousnessSim` |
| `moses.sim` | Advanced simulation | `MultiPhysics`, `Manipulation`, `Deformable`, `AdvancedSensors` |
| `moses.realworld` | Sim-to-real bridge | `SystemID`, `DomainAdaptation`, `Calibration`, `RealWorldDeploy` |
| `moses.industrial` | Industrial integration | `PLCInterface`, `EtherCATMaster`, `ROS2Industrial`, `SafetySystem` |
| `moses.perception` | Advanced perception | `Vision3D`, `TactileSensing`, `ForceEstimator`, `SensorFusion` |

---

## Quick Reference

### Environment Creation
```python
from moses.envs import MosesHumanoidEnvCfg

cfg = MosesHumanoidEnvCfg(
    num_envs=4096,
    env_spacing=4.0,
    robot_cfg=RobotCfg(usd_path="assets/moses_humanoid.usd"),
    rewards=RewardCfg(velocity_tracking_weight=1.0),
)
```

### CAD Generation
```python
from moses.cad import ParametricHumanoid

humanoid = ParametricHumanoid(
    height=1.75,
    mass=75.0,
    dof=28,
)
humanoid.export("output/", formats=["urdf", "usd", "step", "stl"])
```

### GR00T Inference
```python
from moses.gr00t import Gr00TAdapter

adapter = Gr00TAdapter(
    model_path="nvidia/GR00T-N1.7-3B",
    embodiment_tag="UNITREE_G1_SONIC",
    device="cuda:0",
)
action = adapter.get_action(obs, task_text="pick up the cube")
```

### Continuous Training
```python
from moses.training import TrainingPipeline

pipeline = TrainingPipeline.from_yaml("configs/pipeline.yaml")
pipeline.run(trigger="new_data")
```

### Safe Deployment
```python
from moses.deploy import SafeDeploy

deploy = SafeDeploy(config)
deploy.staged_rollout(
    new_policy="checkpoints/v2.pt",
    stages=[0.01, 0.10, 0.50, 1.00],
    metrics=["success_rate", "energy_efficiency"],
)
```

---

*For detailed API docs per module, see API_*.md files in this directory.*
