# code-patterns.md — Moses Knowledge Base

> **Domain:** Reusable Code Patterns for Humanoid Robotics  
> **Status:** Seed document — will grow with builds  
> **Last Updated:** 2026-06-08  
> **Confidence:** Medium — patterns from literature, not yet battle-tested

---

## ROS2 Control Patterns

### Minimal Publisher (Joint Commands)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_commands', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Hz
        self.joint_names = ['hip_l', 'knee_l', 'ankle_l', 'hip_r', 'knee_r', 'ankle_r']
        
    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)  # Replace with controller output
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointCommandPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Minimal Subscriber (Joint States)

```python
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.listener_callback, 10)
        self.current_positions = {}
        
    def listener_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_positions[name] = pos
```

---

## MuJoCo Patterns

### Load Model + Run Simulation

```python
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

# Reset
mujoco.mj_resetData(model, data)

# Simulation loop
with mujoco.Renderer(model) as renderer:
    for step in range(1000):
        # Controller: compute torques
        data.ctrl[:] = compute_torques(data)  # Your controller
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Log data
        if step % 100 == 0:
            print(f"Step {step}: CoM = {data.subtree_com[1]}")
```

### Domain Randomization (for sim-to-real)

```python
def randomize_model(model):
    """Randomize physics parameters for robustness."""
    # Mass perturbation ±10%
    model.body_mass[:] *= np.random.uniform(0.9, 1.1, size=model.nbody)
    
    # Friction perturbation ±20%
    model.geom_friction[:] *= np.random.uniform(0.8, 1.2, size=(model.ngeom, 3))
    
    # Joint damping perturbation ±15%
    model.dof_damping[:] *= np.random.uniform(0.85, 1.15, size=model.nv)
    
    # COM perturbation ±2cm
    model.body_ipos[:] += np.random.uniform(-0.02, 0.02, size=(model.nbody, 3))
```

---

## Control Patterns

### Zero Moment Point (ZMP) Preview Control

```python
import numpy as np
from scipy.linalg import solve_discrete_are

class ZMPPreviewController:
    def __init__(self, dt=0.01, zc=0.8, preview_steps=320):
        self.dt = dt
        self.zc = zc
        self.preview_steps = preview_steps
        
        # LIPM dynamics
        A = np.array([[1, dt, dt**2/2],
                      [0, 1, dt],
                      [0, 0, 1]])
        B = np.array([[dt**3/6], [dt**2/2], [dt]])
        C = np.array([[1, 0, -zc/9.81]])
        
        # LQR
        Q = np.diag([1000, 1, 1])
        R = np.array([[1e-6]])
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
    def compute(self, x, zmp_ref):
        """x: [com, com_vel, com_acc], zmp_ref: preview window"""
        u = -self.K @ x
        # Add preview term (simplified)
        return u
```

### Reinforcement Learning Policy Wrapper

```python
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # Normalized actions
        )
        
    def forward(self, obs):
        return self.net(obs)
    
    def act(self, obs, deterministic=True):
        with torch.no_grad():
            action = self.forward(torch.FloatTensor(obs))
            return action.numpy()
```

---

## Testing Patterns

### Property-Based Test (Hypothesis)

```python
from hypothesis import given, strategies as st
import numpy as np

def test_forward_kinematics_invertible():
    """FK(q) should be invertible by IK within tolerance."""
    @given(st.lists(st.floats(min_value=-np.pi, max_value=np.pi), 
                     min_size=7, max_size=7))
    def property_test(joint_angles):
        target = forward_kinematics(joint_angles)
        recovered_angles = inverse_kinematics(target)
        recovered_target = forward_kinematics(recovered_angles)
        assert np.allclose(target, recovered_target, atol=1e-3)
    return property_test
```

### Simulation Regression Test

```python
def test_walking_policy_stable():
    """Policy should not fall within 10 seconds."""
    policy = load_policy('walking_v1.pt')
    sim = HumanoidSim()
    
    for _ in range(1000):  # 10 seconds at 100 Hz
        obs = sim.get_observation()
        action = policy.act(obs)
        sim.step(action)
        
        # Check: CoM height > 0.5m (not fallen)
        assert sim.com_height() > 0.5, f"Fell at t={sim.time}"
        
        # Check: ZMP in support polygon
        assert sim.zmp_in_support_polygon(), f"ZMP violation at t={sim.time}"
```

---

## CAD Patterns

### Parametric Joint Module (FreeCAD Python)

```python
import FreeCAD as App
import Part

def create_revolute_joint(name, axis='z', length=0.1, diameter=0.05):
    """Create a parametric revolute joint."""
    doc = App.newDocument(name)
    
    # Shaft
    shaft = doc.addObject("Part::Cylinder", "Shaft")
    shaft.Radius = diameter / 2
    shaft.Height = length
    
    # Housing
    housing = doc.addObject("Part::Cylinder", "Housing")
    housing.Radius = diameter / 2 + 0.005  # 5mm wall
    housing.Height = length
    
    # Bearing seats
    # ... (parametric features)
    
    doc.recompute()
    return doc
```

---

## Verified Patterns

| Pattern | Source | Confidence | Tests |
|---------|--------|------------|-------|
| ROS2 pub/sub | ROS2 docs | High | Manual |
| MuJoCo load/step | MuJoCo docs | High | Manual |
| ZMP preview | Kajita et al. | High | Literature |
| RL policy MLP | Stable Baselines3 | High | Community |
| Domain randomization | OpenAI, SOTA | High | Literature |
| FreeCAD scripting | FreeCAD wiki | Medium | Manual |

---

## Open Patterns to Build

1. Whole-body MPC (model predictive control)
2. Sim-to-real adaptation (system ID, domain adaptation)
3. Tactile sensor integration
4. Vision-based state estimation
5. Multi-contact planning
6. Falling recovery controller
7. Grasp planning (analytical + learned)

---

*Last verified: 2026-06-08. Next review: with each build cycle.*
