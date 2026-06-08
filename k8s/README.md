# Moses v3.0 — Kubernetes Deployment Guide (DGX Spark)

This directory contains Kubernetes manifests to deploy the Moses v3.0 humanoid training pipeline on a DGX Spark node (8× NVIDIA A100).

---

## Prerequisites

1. **Kubernetes Cluster** with:
   - NVIDIA GPU Operator installed (or equivalent device plugin + driver)
   - `nvidia.com/gpu` resource available
   - DGX Spark node labeled with `accelerator=nvidia-dgx-spark`

2. **Storage Classes** configured:
   - `nvme-local` — for fast ephemeral scratch (NVMe)
   - `nfs-client` — for shared datasets and checkpoints (NFS)

3. **Container Image** built and pushed:
   ```bash
   docker build -t moses-dgx:latest .
   docker push <registry>/moses-dgx:latest
   ```
   Update `image:` fields in `job-train.yaml` and `job-eval.yaml` if using a registry.

4. **Secrets** (create before applying jobs):
   ```bash
   kubectl create secret generic wandb-secret \
     --from-literal=api-key=$WANDB_API_KEY \
     -n moses
   ```

5. **kubectl** configured and authenticated to your cluster.

---

## Quick Start

Apply all manifests in order:

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. RBAC
kubectl apply -f k8s/rbac.yaml

# 3. ConfigMap (training configuration)
kubectl apply -f k8s/configmap.yaml

# 4. Persistent Volume Claims
kubectl apply -f k8s/pvc.yaml

# 5. Monitoring service
kubectl apply -f k8s/service-monitor.yaml

# 6. Training job
kubectl apply -f k8s/job-train.yaml

# 7. Evaluation job (run after training produces checkpoints)
kubectl apply -f k8s/job-eval.yaml
```

---

## Monitoring

### Check Job Status
```bash
kubectl get jobs -n moses
kubectl get pods -n moses -l app.kubernetes.io/component=training
```

### Stream Training Logs
```bash
kubectl logs -n moses -l job-name=moses-train-humanoid -f
```

### Stream Evaluation Logs
```bash
kubectl logs -n moses -l job-name=moses-eval-policy -f
```

### Prometheus Metrics
The `moses-metrics` Service exposes port `9090`. If you have the Prometheus Operator installed, the `ServiceMonitor` will auto-discover the endpoint. Otherwise, manually scrape:
```bash
kubectl port-forward -n moses svc/moses-metrics 9090:9090
# Open http://localhost:9090/metrics
```

### Resource Usage
```bash
kubectl top pod -n moses
kubectl describe node <dgx-node-name>
```

---

## Scaling Jobs

### Re-run Training with Different Hyperparameters
Edit `k8s/configmap.yaml` and re-apply. Then delete and recreate the job:
```bash
kubectl delete job moses-train-humanoid -n moses
kubectl apply -f k8s/job-train.yaml
```

### Parallel Evaluation
Increase `parallelism` in `job-eval.yaml` (ensure enough GPUs):
```yaml
spec:
  parallelism: 4   # Runs 4 eval pods concurrently
```

### Multi-Node Training
For clusters with multiple DGX nodes, change the torchrun parameters:
```yaml
command:
  - torchrun
  - --nnodes=2
  - --nproc_per_node=8
  - --rdzv_id=moses-train
  - --rdzv_backend=c10d
  - --rdzv_endpoint=moses-train-0.moses.svc.cluster.local:29500
```
And add a `StatefulSet` or `MPIJob` (via MPI Operator) for coordinated multi-node launch.

---

## Storage Notes

| PVC | Purpose | Access Mode | Lifecycle |
|-----|---------|-------------|-----------|
| `moses-nvme-scratch` | Fast temp checkpoints, replay buffers | RWO | Ephemeral (reclaim policy depends on SC) |
| `moses-nfs-shared` | Datasets, persistent checkpoints, eval results | RWX | Persistent |

Ensure the NFS server exports the share with `no_root_squash` or matching UID `1000` for the `runAsUser` in the jobs.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pod stuck `Pending` | Check GPU availability: `kubectl describe node <node>` |
| `ImagePullBackOff` | Verify image name/tag and registry credentials |
| `CrashLoopBackOff` | Check logs: `kubectl logs <pod> -n moses --previous` |
| NCCL errors | Ensure `NCCL_P2P_LEVEL=NVL` and NVSwitch is functional |
| Permission denied on NFS | Match `runAsUser: 1000` to NFS share ownership |

---

## File Reference

| File | Description |
|------|-------------|
| `namespace.yaml` | `moses` namespace |
| `configmap.yaml` | PPO training hyperparameters |
| `pvc.yaml` | NVMe scratch + NFS shared claims |
| `job-train.yaml` | 8× A100 distributed training job |
| `job-eval.yaml` | 1× A100 evaluation job |
| `service-monitor.yaml` | Prometheus metrics service + ServiceMonitor |
| `rbac.yaml` | ServiceAccount, Role, RoleBinding |
| `README.md` | This guide |
