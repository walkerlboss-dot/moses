#!/bin/bash
#SBATCH --job-name=moses-humanoid-train
#SBATCH --partition=dgx-spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moses@boss.industries

# Moses Humanoid Training Job — DGX Spark
# Distributed RL with Isaac Lab on 8x A100/H100

set -euo pipefail

echo "=== Moses Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Load modules
module load cuda/12.3
module load cudnn/8.9
module load nccl/2.18

# Activate environment
source /workspace/moses-builds/venv/bin/activate

# Set NCCL environment for optimal DGX performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=RING

# PyTorch distributed settings
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Training config
ENV_NAME="MosesHumanoid-v1"
NUM_ENVS=4096
SEED=42
TOTAL_STEPS=100000000
CHECKPOINT_INTERVAL=1000000
WANDB_PROJECT="moses-humanoid"

# Isaac Sim headless
export DISPLAY=""

# Launch distributed training
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --env $ENV_NAME \
    --num_envs $NUM_ENVS \
    --seed $SEED \
    --total_steps $TOTAL_STEPS \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --headless \
    --enable_cameras \
    --device cuda

echo "End: $(date)"
