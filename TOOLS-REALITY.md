# TOOLS-REALITY.md — Moses

> **Audited ground truth of what actually runs on the box vs. what is spec vs. aspiration.**
> **Last Updated:** 2026-06-08  
> **Rule:** Consult this file before claiming any tool works.

---

## Environment Reality

| Parameter | Value | Source |
|-----------|-------|--------|
| OS | macOS 26.3 (Darwin 25.3.0, ARM64 T8132) | `uname -a` |
| Python | 3.14.3 | `python3 --version` |
| Python path | `/opt/homebrew/bin/python3` | `which python3` |
| Architecture | ARM64 (Apple Silicon) | `uname -m` |
| Package manager | Homebrew (`brew`) | `/opt/homebrew/bin/brew` |

**Implications:**
- Native ARM64 Python packages preferred (faster, no Rosetta)
- Some robotics tools may need x86_64 emulation or source build
- GPU = Apple Silicon Neural Engine / Metal (not CUDA)
- NVIDIA Isaac Sim **will not run** (requires CUDA)
- MuJoCo runs natively on ARM64 (supported)

---

## Tier S: Actually Wired (Verified Running)

| Moses Claim | Actual Tool | OpenClaw Tool | Status | How to Invoke |
|-------------|-------------|---------------|--------|---------------|
| `code_generate` | Python code generation via LLM | Native (model response) | ✅ Works | Write code in response, save via `file_write` |
| `file_write` | Write files to disk | `file_write` | ✅ Works | `file_write` tool |
| `file_fetch` | Read files from disk | `file_fetch` | ✅ Works | `file_fetch` tool |
| `exec` | Run shell commands | `exec` | ✅ Works | `exec` tool |
| `memory_search` | Search agent memory | `memory_search` | ✅ Works | `memory_search` tool |
| `memory_get` | Read memory excerpts | `memory_get` | ✅ Works | `memory_get` tool |
| `web_fetch` | Fetch web pages | `web_fetch` | ✅ Works | `web_fetch` tool |
| `web_search` | Search the web | `web_search` | ✅ Works | `web_search` tool |
| `message` | Send messages to Alex | `message` | ✅ Works | `message` tool |
| `edit` | Edit existing files | `edit` | ✅ Works | `edit` tool |
| `read` | Read file contents | `read` | ✅ Works | `read` tool |
| `image` | Analyze images | `image` | ✅ Works | `image` tool |
| `pdf` | Analyze PDFs | `pdf` | ✅ Works | `pdf` tool |

**Key insight:** There is no magic `mujoco_deploy` tool. To run MuJoCo, Moses uses `exec` to run Python scripts that import `mujoco`. To generate CAD, Moses uses `exec` to run FreeCAD Python scripts. The tools are **composition**, not primitives.

---

## Tier A: Installable Now (One Command Away)

| Tool | Install Command | Verification | Blocker |
|------|-----------------|--------------|---------|
| MuJoCo | `pip install mujoco` | `python -c "import mujoco; print(mujoco.__version__)"` | None |
| PyTorch | `pip install torch torchvision` | `python -c "import torch; print(torch.__version__)"` | None (ARM64 wheels available) |
| NumPy/SciPy | `pip install numpy scipy` | Already installed (see package list) | None |
| Matplotlib | `pip install matplotlib` | Already installed | None |
| pytest | `pip install pytest pytest-cov hypothesis` | `pytest --version` | None |
| GitPython | `pip install GitPython` | `python -c "import git"` | None |
| OpenCV | `pip install opencv-python` | `python -c "import cv2"` | None |
| Stable Baselines3 | `pip install stable-baselines3` | `python -c "import stable_baselines3"` | None |
| Gymnasium | `pip install gymnasium` | `python -c "import gymnasium"` | None |

---

## Tier B: Needs Setup (Multi-Step)

| Tool | Setup Steps | Blocker | ETA |
|------|-------------|---------|-----|
| FreeCAD | `brew install --cask freecad` | GUI app; Python API needs path setup | 30 min |
| ROS2 | Install from source (macOS not officially supported) | Complex dependency chain | 2-4 hours |
| Docker | `brew install --cask docker` | Needs Docker Desktop | 15 min |
| Ollama (local LLM) | `brew install ollama` | For local code generation | 10 min |
| Jupyter | `pip install jupyterlab` | For interactive analysis | 5 min |

---

## Tier C: Not Available on This Box (Aspiration Only)

| Tool | Why Not Available | Alternative |
|------|-------------------|-------------|
| NVIDIA Isaac Sim | Requires CUDA; Apple Silicon has no CUDA | MuJoCo + Metal backend |
| NVIDIA GPU training | No NVIDIA GPU | Apple Neural Engine (limited), or cloud GPU |
| EtherCAT | Industrial protocol, needs hardware | CAN over USB for early prototypes |
| Real-time Linux (PREEMPT_RT) | macOS is not real-time | Best-effort real-time with high priority |

---

## Tool Chains (How Moses Actually Builds)

### Chain 1: Generate + Save Code
```
Moses thinks → writes Python code in response → file_write to moses-builds/code/
→ exec to run pytest → read test output → edit to fix → file_write updated version
→ exec git commit
```

### Chain 2: Run Simulation
```
Moses writes sim script → file_write to moses-builds/sim/
→ exec "python3 sim_script.py" → read stdout/stderr
→ analyze results → edit script → re-run
```

### Chain 3: Research + Extract
```
web_search for paper → web_fetch PDF → pdf analysis
→ extract methods → file_write to knowledge/
→ memory_search for prior related work
```

### Chain 4: Design Review
```
Moses generates design doc → file_write
→ memory_search for similar prior designs
→ web_search for SOTA comparison
→ edit to incorporate findings → file_write final
```

---

## Tool Gotchas

1. **exec timeout:** Long-running sims may hit exec timeout. Use `background=true` for >60s runs.
2. **file_write overwrites:** Default refuses overwrite. Pass `overwrite=true` when updating.
3. **read truncation:** Large files truncated at 2000 lines. Use offset/limit for big outputs.
4. **web_fetch limits:** Max chars returned. Use for focused extraction, not bulk download.
5. **memory isolation:** Moses cannot read Titan's memory directly. Shared knowledge must be in files.
6. **No persistent state between turns:** Each conversation turn is fresh context. Use files for state.

---

## Verified Constants

| Check | Command | Expected Result | Last Verified |
|-------|---------|-----------------|---------------|
| Python available | `python3 --version` | 3.14.3 | 2026-06-08 |
| pip works | `pip --version` | pip 25.x | 2026-06-08 |
| Homebrew works | `brew --version` | Homebrew 4.x | 2026-06-08 |
| Git works | `git --version` | git 2.x | 2026-06-08 |
| NumPy installed | `python3 -c "import numpy"` | No error | 2026-06-08 |
| SciPy installed | `python3 -c "import scipy"` | No error | 2026-06-08 |

---

*TOOLS-REALITY.md v1.0 — Moses — Boss Industries — June 2026*
