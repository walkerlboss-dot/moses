# Perception API — Moses

> **3D vision, tactile sensing, force estimation, and multi-modal fusion.**

---

## Vision3D

```python
from moses.perception.vision3d import Vision3D
```

3D vision with depth estimation and point cloud processing.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `estimate_depth()` | `stereo_images` | `DepthMap` | Stereo depth |
| `detect_objects_3d()` | `point_cloud` | `list` | 3D object detection |
| `segment_scene()` | `point_cloud` | `list` | Scene segmentation |
| `track_objects()` | `frames` | `list` | Multi-object tracking |
| `visual_slam()` | `frames` | `PoseGraph` | Visual SLAM |

### Stereo Depth

```python
vision = Vision3D()

# Stereo depth estimation
depth = vision.estimate_depth(
    left_image=left_cam,
    right_image=right_cam,
    baseline=0.12,  # 12 cm baseline
    focal_length=600,  # pixels
)
# Returns: DepthMap (H, W) in meters
```

### Point Cloud Processing

```python
# Segment scene
segments = vision.segment_scene(point_cloud)
# Returns: [
#   {"label": "table", "points": [...], "bbox": [...]},
#   {"label": "cup", "points": [...], "bbox": [...]},
# ]

# Detect objects
objects = vision.detect_objects_3d(point_cloud)
# Returns: [
#   {"class": "mug", "confidence": 0.95, "pose": [...]},
# ]
```

---

## TactileSensing

```python
from moses.perception.tactile import TactileSensing
```

Tactile sensor processing.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `process_pressure()` | `tactile_image` | `dict` | Pressure distribution |
| `detect_slip()` | `tactile_sequence` | `bool` | Slip detection |
| `classify_texture()` | `tactile_image` | `str` | Texture classification |
| `estimate_contact()` | `tactile_image` | `dict` | Contact geometry |

### Slip Detection

```python
tactile = TactileSensing()

# Detect slip
is_slipping = tactile.detect_slip(
    tactile_sequence=recent_frames,
    threshold=0.5,
)

if is_slipping:
    robot.increase_grasp_force()
```

### Texture Classification

| Texture | Features | Confidence |
|---------|----------|------------|
| Smooth | Low variance | > 0.9 |
| Rough | High variance | > 0.85 |
| Grooved | Directional patterns | > 0.8 |
| Bumpy | Periodic peaks | > 0.75 |

---

## ForceEstimator

```python
from moses.perception.force_estimation import ForceEstimator
```

Estimates external forces from proprioception.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `estimate_external_force()` | `joint_torques`, `joint_positions` | `Wrench` | External force |
| `estimate_contact_force()` | `joint_torques` | `dict` | Contact force distribution |
| `detect_collision()` | `residual` | `bool` | Collision detection |

### Force Estimation

```python
estimator = ForceEstimator(robot_model)

# Estimate external force
external_force = estimator.estimate_external_force(
    joint_torques=robot.get_joint_torques(),
    joint_positions=robot.get_joint_positions(),
    joint_velocities=robot.get_joint_velocities(),
)
# Returns: Wrench (force: [Fx, Fy, Fz], torque: [Tx, Ty, Tz])

# Detect collision
if estimator.detect_collision(external_force, threshold=10.0):
    robot.trigger_safety_stop()
```

---

## SensorFusion

```python
from moses.perception.fusion import SensorFusion
```

Multi-modal sensor fusion.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `fuse()` | `measurements` | `state` | Fused state estimate |
| `kalman_update()` | `prediction`, `measurement` | `state` | Kalman filter update |
| `attention_select()` | `inputs`, `task` | `weights` | Attention-based selection |

### Fusion Architecture

```python
fusion = SensorFusion(
    sensors=["vision", "tactile", "force", "imu"],
    fusion_type="kalman_attention",
)

# Fuse measurements
state = fusion.fuse({
    "vision": vision_obs,
    "tactile": tactile_obs,
    "force": force_obs,
    "imu": imu_obs,
})

# Attention-based selection
weights = fusion.attention_select(
    inputs={"vision": vision_obs, "tactile": tactile_obs},
    task="grasp",
)
# Returns: {"vision": 0.3, "tactile": 0.7} (tactile more important for grasping)
```

---

*Perception is reality — for robots.*
