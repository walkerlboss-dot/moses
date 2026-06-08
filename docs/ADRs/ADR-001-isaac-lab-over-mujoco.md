# ADR-001: Isaac Lab as Primary Simulation Platform

## Status
Accepted — 2026-06-08

## Context

Moses needs a physics simulation platform for training humanoid robot policies. Two main candidates:

1. **MuJoCo** — Mature, CPU-based, excellent for research
2. **Isaac Lab** — NVIDIA's GPU-accelerated platform, built on Omniverse

## Decision

**Use Isaac Lab as the primary simulation platform**, with MuJoCo as a fallback for lightweight testing.

## Consequences

### Positive
- **Massive parallelization**: 4,096 environments on a single GPU vs. ~100 with MuJoCo
- **GPU acceleration**: Physics and rendering on GPU, freeing CPU for training
- **USD pipeline**: Native support for Universal Scene Description, industry standard
- **NVIDIA ecosystem**: Direct integration with TensorRT, Omniverse, DGX
- **Ray tracing**: Photorealistic sensor simulation for sim-to-real transfer
- **Domain randomization**: Built-in support for mass, friction, lighting randomization

### Negative
- **Hardware requirement**: Requires NVIDIA GPU; won't run on CPU-only systems
- **Container size**: Isaac Sim container is ~20GB
- **API stability**: Isaac Lab is actively developed; APIs may change
- **Licensing**: Free for research/individual; enterprise license for commercial use
- **macOS unsupported**: Cannot develop locally on macOS (Apple Silicon)

### Mitigations
- MuJoCo fallback for development and lightweight testing
- Pin Isaac Lab to stable version (v1.2.0)
- Container-based deployment for reproducibility
- CI/CD pipeline validates against both platforms

## Alternatives Considered

| Platform | Parallel Envs | GPU | USD | Decision |
|----------|--------------|-----|-----|----------|
| MuJoCo | ~100 (CPU) | No | No | Fallback |
| Isaac Lab | 4,096+ (GPU) | Yes | Yes | **Primary** |
| Gazebo | ~50 (CPU) | No | No | Rejected |
| RaiSim | ~500 (CPU/GPU) | Partial | No | Rejected |
| Drake | ~10 (CPU) | No | No | Rejected |

## References
- NVIDIA Isaac Sim Documentation: https://docs.isaacsim.omniverse.nvidia.com/
- Isaac Lab GitHub: https://github.com/isaac-sim/IsaacLab
- MuJoCo Documentation: https://mujoco.readthedocs.io/
