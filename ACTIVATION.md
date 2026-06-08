# ACTIVATION.md — Moses v1.0

> **How to wake Moses up and start building**

---

## Pre-Flight Checklist

Before enabling Moses, confirm:

- [ ] Alex has reviewed and approved `AGENT.md` v1
- [ ] Alex has reviewed and approved `SOUL.md` v1
- [ ] Compute resources confirmed (CPU minimum; GPU strongly preferred)
- [ ] GitHub repo `moses-builds` created (or local git init approved)
- [ ] Storage space confirmed (artifacts will grow ~50-100 MB/week)
- [ ] Alex understands: **standing session = continuous cost**

---

## Step 1: Initialize Build Environment

```bash
# Create build repo
mkdir -p ~/.openclaw/workspace/moses-builds
cd ~/.openclaw/workspace/moses-builds
git init
git config user.name "Moses Builder"
git config user.email "moses@boss.industries"

# Create directory structure
mkdir -p {cad,code,sim,tests,docs,bom,pcb}
```

---

## Step 2: Install Toolchain

```bash
# Core physics simulation
pip install mujoco
pip install mujoco_menagerie

# ML/RL (for learned controllers)
pip install torch torchvision
pip install stable-baselines3
pip install gymnasium

# ROS2 (Ubuntu 22.04)
# sudo apt install ros-humble-desktop
# pip install rclpy

# CAD (macOS)
brew install --cask freecad

# Testing
pip install pytest pytest-cov hypothesis

# Utilities
pip install numpy scipy matplotlib pandas
```

---

## Step 3: Configure Autonomous Build Loop

```bash
# Hourly build cycle
# Daily report: 9:00 AM ET
# Exception alerts: immediate
```

Moses will configure these automatically when enabled.

---

## Step 4: Enable Standing Session

**Alex must explicitly authorize this. It is Tier 3.**

Alex says: **"Moses, wake up"** or **"Start building"**

Moses will:
1. Verify config checksums
2. Initialize `moses-builds` repo
3. Install toolchain (if not present)
4. Configure hourly build cron + daily report
5. Run first build cycle
6. Report: "Moses is building. First artifacts in 24 hours."

---

## Step 5: First Daily Report

Target: 24 hours after activation.
Format: See `AGENT.md` → Communication Protocol.

---

## Emergency Stop

At any time, Alex can say:
- **"Moses, sleep"** — Pause build loop, preserve state, reduce cost
- **"Moses, stop"** — Halt all activity, dump status
- **"Moses, reset"** — Clear build state, restart from seed

---

*ACTIVATION.md v1.0 — Moses — Boss Industries — June 2026*
