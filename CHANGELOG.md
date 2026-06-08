# Changelog

All notable changes to Moses will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] — 2026-06-08

### Added
- Complete Isaac Lab integration with working environment configuration
- Domain randomization utilities for sim-to-real transfer
- Checkpoint manager with resume capability
- GitHub Actions CI/CD pipeline (lint → test → build → sim-test)
- Makefile with common development tasks
- Docker Compose multi-service stack (train + monitor + tensorboard)
- Pre-commit hooks for code quality
- pyproject.toml with full dependency specification
- Comprehensive test suite (unit + integration)
- Kubernetes manifests for DGX Spark deployment
- Architecture Decision Records (ADRs)
- API documentation
- Benchmark results and performance targets
- Usage examples and troubleshooting guide
- GitHub issue/PR templates
- Code of Conduct and Contributing guide
- LICENSE (MIT)

### Changed
- AGENT.md upgraded to v3.0 with full DGX Spark specification
- README redesigned with badges, Mermaid diagram, feature matrix
- Build loop enhanced with retry logic and error recovery
- Security hardening with sandbox and audit logging

### Fixed
- TOOLS-REALITY.md now maps fictional tool names to actual OpenClaw tools
- Knowledge base linked properly between Titan and Moses

---

## [2.1.0] — 2026-06-08

### Added
- 5 runnable Python scripts: train_humanoid.py, eval_policy.py, export_tensorrt.py, run_tests.py, moses_loop.py
- Live monitoring dashboard (Rich TUI)
- Smart alert system with deduplication
- SQLite metrics collector
- System health checker
- Security sandbox with AST scanning
- Immutable audit logs with hash-chain tamper detection
- Secret manager with dual-loading and rotation
- Compliance checker for secrets and hardcoded IPs
- Titan-Moses collaboration protocol
- Human-in-the-loop checkpoint system (9 Tier-3 gates)
- Orchestration graph and activation sequence

---

## [2.0.0] — 2026-06-08

### Added
- DGX Spark optimization: Dockerfile, Slurm scripts, PPO config
- NVIDIA stack: CUDA 12.x, cuDNN, NCCL, TensorRT
- Isaac Lab primary simulation platform
- Multi-GPU distributed training with PyTorch DDP
- Knowledge base expanded to 7 domains (2,739 lines)
- Manufacturing methods, materials selection, electronics stack
- Software architecture, testing strategy

---

## [1.0.0] — 2026-06-08

### Added
- Initial agent configuration: AGENT.md, SOUL.md, HEARTBEAT.md
- Bootstrap ritual and activation guide
- Tooling reality documentation
- Component database and code patterns
- Basic build loop specification

---

## [Unreleased]

### Planned
- Physical prototype integration (Titan Mark I)
- Real-world policy deployment
- Sim-to-real validation pipeline
- Open-source community launch

---

*Moses is built by Boss Industries. "Build first, perfect later."*
