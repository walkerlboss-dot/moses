# Troubleshooting — Moses

> **Common issues and their solutions.**

---

## Installation Issues

### Issue: `ImportError: No module named 'isaaclab'`

**Cause:** Isaac Lab not installed or not in PYTHONPATH.

**Solution:**
```bash
# Install Isaac Lab (inside Isaac Sim container)
cd /isaac-sim/IsaacLab
./isaaclab.sh --install

# Or via pip (if Isaac Sim is already installed)
pip install isaaclab
```

### Issue: `CUDA out of memory`

**Cause:** GPU memory exhausted by large batch or too many environments.

**Solutions:**
```bash
# Reduce number of environments
python scripts/train_humanoid.py --num_envs 2048  # instead of 4096

# Enable gradient checkpointing
python scripts/train_humanoid.py --checkpoint_gradients

# Use mixed precision
python scripts/train_humanoid.py --mixed_precision

# Clear GPU cache
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
```

### Issue: Docker container fails to start

**Cause:** NVIDIA Container Toolkit not installed or Docker daemon not running.

**Solution:**
```bash
# Verify NVIDIA Container Toolkit
nvidia-ctk --version

# Restart Docker
sudo systemctl restart docker

# Run with GPU support
docker run --gpus all -it moses-dgx:latest
```

---

## Training Issues

### Issue: NaN loss during training

**Cause:** Learning rate too high, reward scaling issue, or numerical instability.

**Solutions:**
```bash
# Reduce learning rate
python scripts/train_humanoid.py --learning_rate 1e-4

# Enable reward clipping
python scripts/train_humanoid.py --clip_rewards

# Check for inf/nan in observations
python scripts/train_humanoid.py --debug_nan
```

### Issue: Policy not improving (stuck at low reward)

**Cause:** Poor reward shaping, insufficient exploration, or hyperparameter mismatch.

**Solutions:**
- Review reward function in `moses/envs/humanoid_env.py`
- Increase entropy coefficient: `--entropy_coef 0.02`
- Check observation normalization is enabled
- Verify domain randomization is active

### Issue: Training very slow

**Cause:** CPU bottleneck, slow disk I/O, or suboptimal GPU utilization.

**Solutions:**
```bash
# Check GPU utilization
nvidia-smi dmon

# Use NVMe for checkpoints
export MOSES_SCRATCH=/nvme/checkpoints

# Enable CUDA graphs (experimental)
python scripts/train_humanoid.py --cuda_graphs

# Profile to find bottleneck
nsys profile -o profile.qdrep python scripts/train_humanoid.py
```

---

## Simulation Issues

### Issue: Isaac Sim headless mode fails

**Cause:** Missing display drivers, EGL not available, or Vulkan issues.

**Solutions:**
```bash
# Use EGL (software rendering)
export DISPLAY=""
python scripts/train_humanoid.py --headless --enable_cameras

# Or disable cameras entirely
python scripts/train_humanoid.py --headless --disable_cameras

# Check Vulkan
vulkaninfo | head -20
```

### Issue: Robot falls immediately in simulation

**Cause:** Incorrect initial pose, insufficient joint damping, or gravity issue.

**Solutions:**
- Check initial joint positions in URDF/USD
- Increase joint damping in actuator config
- Verify gravity is set correctly (9.81 m/s²)
- Reduce initial height if penetration occurs

---

## Multi-GPU Issues

### Issue: NCCL timeout or communication error

**Cause:** Network misconfiguration, firewall blocking, or incorrect NCCL settings.

**Solutions:**
```bash
# Debug NCCL
export NCCL_DEBUG=INFO

# Disable InfiniBand if not available
export NCCL_IB_DISABLE=1

# Set correct network interface
export NCCL_SOCKET_IFNAME=eth0

# Increase timeout
export NCCL_TIMEOUT=600
```

### Issue: Uneven GPU utilization

**Cause:** Imbalanced workload or data loading bottleneck.

**Solutions:**
- Ensure `num_envs` is divisible by `num_gpus`
- Check CPU isn't bottlenecked (use `htop`)
- Enable `pin_memory` in data loader

---

## Inference Issues

### Issue: TensorRT engine build fails

**Cause:** ONNX model has unsupported ops, wrong GPU architecture, or dynamic shapes.

**Solutions:**
```bash
# Check ONNX model
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# Build with explicit batch size
python scripts/export_tensorrt.py --onnx model.onnx --batch_size 1

# Use FP32 instead of FP16
python scripts/export_tensorrt.py --onnx model.onnx --fp32

# Update TensorRT
pip install --upgrade tensorrt
```

### Issue: ROS2 node crashes on startup

**Cause:** Missing ROS2 environment, wrong DDS implementation, or topic mismatch.

**Solutions:**
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Check ROS2 installation
ros2 doctor

# Verify topic names
ros2 topic list

# Check QoS compatibility
ros2 topic info /joint_commands --verbose
```

---

## Monitoring Issues

### Issue: Dashboard not updating

**Cause:** Metrics collector not running, SQLite lock, or wrong paths.

**Solutions:**
```bash
# Check metrics collector is running
python monitoring/metrics_collector.py --status

# Reset metrics database
rm ~/.moses/metrics.db
python monitoring/metrics_collector.py --init

# Check file permissions
ls -la ~/.moses/
```

---

## General Tips

1. **Always check logs first:** `tail -f logs/moses.log`
2. **Verify environment:** `python monitoring/health_check.py`
3. **Start small:** Test with `--num_envs 128` before scaling
4. **Use checkpoints:** Always enable `--checkpoint_interval 50`
5. **Monitor resources:** `nvidia-smi`, `htop`, `df -h`

---

*Still stuck? Open an issue with logs and config:*
*https://github.com/walkerlboss-dot/moses/issues*
