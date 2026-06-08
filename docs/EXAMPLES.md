# Examples — Getting Started with Moses

> **Practical examples for training, evaluating, and deploying humanoid policies.**

---

## Example 1: Train a Walking Policy

### Goal
Train a PPO policy that makes a humanoid robot walk forward at 1.0 m/s.

### Command

```bash
# Single GPU (development)
python scripts/train_humanoid.py \
  --config configs/train_ppo.yaml \
  --num_envs 4096 \
  --headless \
  --wandb_project moses-humanoid \
  --experiment_name walk_forward_v1
```

### Expected Output

```
[INFO] Isaac Sim headless mode
[INFO] Created 4096 parallel environments
[INFO] Policy: ActorCritic [512, 256, 128]
[INFO] Algorithm: PPO (lr=3e-4, clip=0.2)
[INFO] Starting training...

Iteration 0:    reward=12.45    episode_len=45.2    fps=348,160
Iteration 50:   reward=28.91    episode_len=120.5   fps=351,200
Iteration 100:  reward=38.72    episode_len=280.1   fps=349,800
Iteration 500:  reward=45.23    episode_len=450.8   fps=352,100
Iteration 1000: reward=46.81    episode_len=512.3   fps=350,500

[INFO] Training complete. Best reward: 46.81
[INFO] Checkpoint saved: checkpoints/walk_forward_v1/best.pt
[INFO] W&B run: https://wandb.ai/moses-humanoid/runs/abc123
```

### Visualize

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Or view in Weights & Biases
wandb login
# Open https://wandb.ai/moses-humanoid/runs/abc123
```

---

## Example 2: Evaluate and Export to TensorRT

### Goal
Evaluate a trained policy and export it for fast inference.

### Step 1: Evaluate

```bash
python scripts/eval_policy.py \
  --checkpoint checkpoints/walk_forward_v1/best.pt \
  --num_episodes 100 \
  --render \
  --save_video
```

### Expected Output

```
[INFO] Loaded policy from checkpoints/walk_forward_v1/best.pt
[INFO] Running 100 evaluation episodes...

Episode 1/100:  reward=47.2  len=520  survived=True
Episode 2/100:  reward=45.8  len=498  survived=True
...
Episode 100/100: reward=46.1  len=512  survived=True

--- Evaluation Summary ---
Mean reward:      46.32 ± 1.84
Mean episode len: 508.4 ± 25.2
Success rate:     97.0%
Energy efficiency: 42.1 J/m

[INFO] Video saved: eval_videos/walk_forward_v1.mp4
[INFO] Metrics saved: eval_results/walk_forward_v1.json
```

### Step 2: Export to ONNX

```bash
python scripts/eval_policy.py \
  --checkpoint checkpoints/walk_forward_v1/best.pt \
  --export_onnx \
  --onnx_path models/walk_forward_v1.onnx
```

### Step 3: Build TensorRT Engine

```bash
python scripts/export_tensorrt.py \
  --onnx models/walk_forward_v1.onnx \
  --output models/walk_forward_v1.trt \
  --fp16 \
  --benchmark
```

### Expected Output

```
[INFO] Loading ONNX model: models/walk_forward_v1.onnx
[INFO] Building TensorRT engine (FP16)...
[INFO] Engine built. Size: 18.4 MB
[INFO] Benchmarking inference speed...

Backend:          TensorRT FP16
Batch size:       1
Latency:          0.58 ± 0.03 ms
Throughput:       1,724 inf/s
GPU memory:       18.4 MB

[INFO] Engine saved: models/walk_forward_v1.trt
```

---

## Example 3: Run Autonomous Build Loop

### Goal
Let Moses run the full build cycle autonomously.

### Command

```bash
python scripts/moses_loop.py \
  --config configs/autonomous_build.yaml \
  --mode full \
  --notify_telegram
```

### Expected Behavior

```
[INFO] Moses Autonomous Build Loop v3.0
[INFO] Mode: full (DESIGN → CODE → SIM → TRAIN → TEST → REPORT)

[PHASE: DESIGN] Generating humanoid URDF...
  → Generated: cad/moses_humanoid_v3.urdf
  → Mass: 58.2 kg, DOF: 28

[PHASE: CODE] Generating control software...
  → Generated: moses/controllers/ppo_walk_v3.py
  → Tests: 12/12 passed

[PHASE: SIM] Running Isaac Lab simulation...
  → Environments: 4096
  → Sim FPS: 348,160

[PHASE: TRAIN] Training PPO policy...
  → Iteration 0/5000: reward=12.45
  → Iteration 500/5000: reward=42.18
  → Iteration 1000/5000: reward=45.91
  → ...
  → Best reward: 46.82

[PHASE: TEST] Running test suite...
  → Unit tests: 45/45 passed
  → Integration tests: 8/8 passed
  → Sim tests: 3/3 passed

[PHASE: REPORT] Generating daily report...
  → Report sent to Alex via Telegram

[INFO] Build cycle complete. Artifacts committed: a3f7d2e
[INFO] Next cycle in: 24 hours
```

---

## Example 4: Multi-GPU Distributed Training

### Goal
Train on 8x A100 GPUs for maximum speed.

### Command

```bash
# Using torchrun
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  scripts/train_humanoid.py \
  --config configs/train_ppo.yaml \
  --num_envs 4096 \
  --distributed \
  --headless

# Or using Slurm on DGX Spark
sbatch slurm-train.sh
```

### Expected Output

```
[INFO] Distributed training: 8 GPUs
[INFO] Local rank: 0, World size: 8
[INFO] NCCL backend initialized
[INFO] NVLink topology detected

Iteration 0:    reward=11.23    fps=2,784,000 (8x 348K)
Iteration 50:   reward=29.45    fps=2,810,000
Iteration 100:  reward=39.12    fps=2,798,000

[INFO] Training complete in 0.6 hours
[INFO] 8.3x speedup vs single GPU
```

---

## Example 5: Resume from Checkpoint

### Goal
Resume training after interruption.

```bash
python scripts/train_humanoid.py \
  --config configs/train_ppo.yaml \
  --resume \
  --checkpoint checkpoints/walk_forward_v1/iter_500.pt \
  --num_envs 4096 \
  --headless
```

---

## Example 6: Domain Randomization for Sim-to-Real

### Goal
Train a robust policy that transfers to physical robot.

```bash
python scripts/train_humanoid.py \
  --config configs/train_ppo.yaml \
  --num_envs 4096 \
  --domain_randomization \
  --randomize_mass 0.9 1.1 \
  --randomize_friction 0.5 1.2 \
  --randomize_gravity 9.0 10.0 \
  --headless
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

---

*More examples coming. Contribute your own!*
