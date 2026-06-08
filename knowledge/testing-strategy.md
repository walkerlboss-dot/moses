# testing-strategy.md — Moses Knowledge Base

> **Domain:** Testing & Validation for Humanoid Robotics
> **Status:** Seed document — will grow with test infrastructure and results
> **Last Updated:** 2026-06-08
> **Confidence:** Medium — strategies from software engineering and robotics literature; not yet implemented

---

## 1. Test Pyramid for Robotics

### 1.1 The Robotics Test Pyramid

```
                    ▲
                   / \
                  / E \      End-to-End (System) Tests
                 /  2  \     • Full robot walking on terrain
                /   E   \    • Manipulation tasks
               /─────────\   • Hours-long autonomy runs
              /     I     \  Integration Tests
             /      N      \ • Multi-node ROS2 graph
            /       T       \• Sim-to-real validation
           /─────────────────\• Hardware-in-the-loop
          /        U          \ Unit Tests
         /         N           \• Individual controller modules
        /          I            \• Math utilities (IK, dynamics)
       /           T             \• Sensor drivers
      /───────────────────────────\• Message serialization
     /            S               \ Static Analysis / Lint
    /             M                \• Code style, type checking
   /              O                \• Cyclomatic complexity
  /               K                \• Dependency scanning
 /──────────────────────────────────\• Documentation coverage
/                 E                  \
                  S
```

**Distribution (target):**

| Level | Target % of Tests | Execution Frequency | Source |
|-------|-------------------|---------------------|--------|
| Static / Lint | 5–10% | Every commit | Software engineering practice |
| Unit tests | 50–60% | Every commit / PR | Software engineering practice |
| Integration tests | 20–30% | Every PR / nightly | Robotics practice |
| End-to-end tests | 5–10% | Nightly / weekly | Robotics practice |

**Source:** Adapted from standard test pyramid (Cohn, "Succeeding with Agile") and robotics-specific extensions.

---

## 2. Unit Testing

### 2.1 Scope

| Component | Test Target | Framework | Source |
|-----------|-------------|-----------|--------|
| Forward kinematics | FK(q) accuracy vs. analytical solution | Google Test / pytest | Practice |
| Inverse kinematics | Convergence rate, solution quality | Google Test / pytest | Practice |
| Jacobian computation | Numerical vs. analytical Jacobian | Google Test / pytest | Practice |
| Dynamics (Mass matrix, Coriolis) | Energy conservation, symmetry | Google Test / pytest | Practice |
| Quaternions / rotations | Identity, composition, normalization | Google Test / pytest | Practice |
| Trajectory generation | Continuity, boundary conditions | Google Test / pytest | Practice |
| PID controller | Step response, steady-state error | Google Test / pytest | Practice |
| Matrix operations | Correctness, numerical stability | Google Test / Eigen tests | Practice |
| Message serialization | Round-trip integrity | ROS2 testing tools | Practice |

### 2.2 Property-Based Testing (Hypothesis / RapidCheck)

```python
# Example: FK/IK round-trip property test
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.floats(min_value=-np.pi, max_value=np.pi),
                min_size=7, max_size=7))
def test_fk_ik_roundtrip(joint_angles):
    """FK(q) followed by IK should recover q within tolerance."""
    target_pose = forward_kinematics(joint_angles)
    recovered_angles = inverse_kinematics(target_pose)
    recovered_pose = forward_kinematics(recovered_angles)
    
    assert np.allclose(target_pose, recovered_pose, atol=1e-4)
    # Note: joint angles may differ (redundancy) but pose must match
```

**Source:** Hypothesis docs, property-based testing literature.

### 2.3 Verified Constants for Unit Tests

| Test | Expected Result | Tolerance | Source |
|------|-----------------|-----------|--------|
| FK/IK round-trip (7-DOF arm) | Position error | **< 1e-4 m** | Literature |
| Jacobian transpose vs. numerical | Relative error | **< 1e-3** | Numerical analysis |
| Mass matrix symmetry | M - M.T = 0 | **< 1e-6** | Physics (must be symmetric) |
| Mass matrix positive definite | All eigenvalues > 0 | **> 1e-6** | Physics |
| Quaternion normalization | \|q\| = 1 | **< 1e-6** | Math |
| Rotation matrix orthogonality | R·R.T = I | **< 1e-6** | Math |
| Trajectory C² continuity | Jerk bounded | **< 1e3 m/s³** | Practice |

---

## 3. Integration Testing

### 3.1 ROS2 Node Graph Testing

| Test | Description | Tool | Source |
|------|-------------|------|--------|
| Node launch | All nodes start without errors | `launch_testing` | ROS2 docs |
| Topic connectivity | All expected topics have publishers/subscribers | `ros2 topic list` + custom | ROS2 docs |
| Message flow | Data flows end-to-end within latency budget | Custom + `ros2 bag` | Practice |
| Parameter loading | All parameters load correctly from YAML | `launch_testing` | ROS2 docs |
| Service calls | All services respond within timeout | `ros2 service` + custom | ROS2 docs |
| Action servers | Goals accepted, executed, completed | `launch_testing` | ROS2 docs |

### 3.2 Controller Integration Tests

| Test | Setup | Pass Criteria | Source |
|------|-------|---------------|--------|
| Joint position tracking | Command sinusoidal trajectory | Tracking error < 0.01 rad RMS | Practice |
| Gravity compensation | Command zero torque, verify static pose | Position drift < 0.05 rad in 10s | Practice |
| Impedance control | Apply external force, measure displacement | Stiffness matches commanded value ±10% | Literature |
| Emergency stop | Trigger e-stop during motion | All joints stop within 100 ms | Safety standards |
| Mode transitions | Switch position→torque→impedance | No jumps, smooth transition | Practice |

---

## 4. Simulation Testing

### 4.1 Simulation Fidelity Levels

| Level | Description | Fidelity | Use Case | Source |
|-------|-------------|----------|----------|--------|
| L0: Kinematic | Geometry only, no physics | Low | Collision checking, workspace analysis | Practice |
| L1: Rigid body dynamics | Mass, inertia, contacts, friction | Medium | Controller development, gait tuning | Practice |
| L2: Actuator dynamics | Motor model, gear backlash, friction | Medium-High | Sim-to-real gap reduction | Literature |
| L3: Sensor simulation | Noise models, latency, dropouts | High | State estimator validation | Literature |
| L4: Domain randomization | Parameter distributions | High | Policy robustness (RL) | Literature |
| L5: Soft contacts / deformable | FEM, contact patches | Very High | Foot-ground interaction research | Research |

### 4.2 MuJoCo-Specific Testing

| Test | Configuration | Metric | Source |
|------|---------------|--------|--------|
| Balance robustness | Perturb CoM by ±5 cm | Recovery time < 2 s | Literature |
| Push recovery | Apply 50 N impulse for 100 ms | Does not fall | Literature |
| Slope walking | 5°, 10°, 15° slopes | Stable gait, ZMP in polygon | Literature |
| Step over obstacle | 5 cm, 10 cm obstacles | Clearance, stability | Practice |
| Walking on uneven terrain | Random height ±2 cm | Stable gait for 60 s | Practice |
| Carrying payload | +5 kg, +10 kg | Gait stable, joint limits respected | Practice |
| Trip recovery | Stumble on unseen obstacle | Recovery or controlled fall | Literature |

### 4.3 Domain Randomization Parameters

| Parameter | Nominal | Randomization Range | Source |
|-----------|---------|---------------------|--------|
| Link mass | m₀ | ±10% | Literature (OpenAI, SOTA) |
| Link CoM offset | [x₀, y₀, z₀] | ±2 cm per axis | Literature |
| Joint damping | d₀ | ±20% | Literature |
| Joint friction | f₀ | ±30% | Literature |
| Ground friction | μ = 0.8 | 0.5 – 1.2 | Literature |
| Ground restitution | e = 0.0 | 0.0 – 0.2 | Literature |
| Motor torque constant | Kt₀ | ±10% | Literature |
| Motor resistance | R₀ | ±10% | Literature |
| Sensor noise (position) | 0 | ±0.001 rad | Literature |
| Sensor noise (velocity) | 0 | ±0.01 rad/s | Literature |
| Sensor delay | 0 | 0 – 5 ms | Literature |
| Actuator delay | 0 | 0 – 5 ms | Literature |

**Source:** OpenAI robotics sim-to-real papers, MIT Cheetah sim-to-real work, general RL robotics literature.

### 4.4 Simulation Regression Test Suite

```python
def test_walking_policy_regression():
    """Policy must pass all scenarios without falling."""
    policy = load_policy('walking_v2.pt')
    scenarios = [
        ('flat_ground', {'duration': 60.0}),
        ('5_degree_slope', {'duration': 30.0}),
        ('10_degree_slope', {'duration': 30.0}),
        ('push_recovery', {'force': 50.0, 'duration': 0.1}),
        ('uneven_terrain', {'roughness': 0.02, 'duration': 60.0}),
        ('payload_10kg', {'payload_mass': 10.0, 'duration': 30.0}),
    ]
    
    for name, params in scenarios:
        sim = HumanoidSim(randomize=True)
        result = sim.run_policy(policy, **params)
        
        assert result.min_com_height > 0.5, f"{name}: Fell (CoM < 0.5m)"
        assert result.max_zmp_error < 0.05, f"{name}: ZMP violation"
        assert result.max_joint_torque < 0.9 * TAU_MAX, f"{name}: Torque limit"
        assert result.energy_efficiency > 0.5, f"{name}: Efficiency too low"
```

---

## 5. Hardware-in-the-Loop (HIL) Testing

### 5.1 HIL Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HARDWARE-IN-THE-LOOP                        │
│                                                                  │
│   ┌──────────────┐         ┌──────────────┐                     │
│   │   Simulation │◄───────►│   Real-Time  │                     │
│   │   (MuJoCo /  │  EtherCAT│   Target     │                     │
│   │    Gazebo)   │   or CAN │   (STM32 /   │                     │
│   │              │          │    Teensy)   │                     │
│   │  • Physics   │          │              │                     │
│   │  • Sensors   │          │  • Motor     │                     │
│   │  (simulated) │          │    driver    │                     │
│   │  • Environment│         │  • Real      │                     │
│   │              │          │    controller│                     │
│   └──────────────┘          └──────┬───────┘                     │
│                                    │                             │
│                           ┌────────┴────────┐                    │
│                           │  Real Actuator  │                    │
│                           │  (on test stand)│                    │
│                           │                 │                    │
│                           │  • Motor + gear │                    │
│                           │  • Encoder      │                    │
│                           │  • Load cell    │                    │
│                           └─────────────────┘                    │
│                                                                  │
│   Loop closure: Sim sends joint torques → Real controller →      │
│   Real actuator moves → Real encoder → Real controller →         │
│   Sim reads actual position → Sim updates physics                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 HIL Test Scenarios

| Test | Setup | Pass Criteria | Source |
|------|-------|---------------|--------|
| Single joint tracking | One actuator on test stand, sim provides load | Tracking error < 0.5° RMS | Practice |
| Multi-joint coordination | 3–4 actuators, simulated leg dynamics | Coordinated motion, no oscillation | Practice |
| Impact response | Sim applies sudden load (trip simulation) | Peak torque < limit, recovery < 500 ms | Practice |
| Thermal test | Run at max continuous torque for 30 min | Driver temp < 80 °C, no thermal shutdown | Datasheet |
| Communication stress | Max bus load, dropped packets | Controller degrades gracefully | Practice |
| Power sag test | Simulate battery voltage drop to 80% | Controller remains stable | Practice |

---

## 6. Hardware Testing

### 6.1 Single Joint Characterization

| Test | Procedure | Measured | Source |
|------|-----------|----------|--------|
| Friction identification | Command slow velocity sweep | Coulomb + viscous friction | System ID literature |
| Backlash measurement | Command small sinusoidal position | Hysteresis width | Mechanical engineering |
| Torque constant (Kt) | Measure torque vs. current | Kt = τ / I | Motor theory |
| Electrical time constant | Step voltage, measure current rise | τe = L / R | Motor theory |
| Mechanical time constant | Step torque, measure speed rise | τm = J / b | Motor theory |
| Gear efficiency | Input/output torque ratio | η = τout / (τin × N) | Mechanical engineering |
| Max continuous torque | Run at increasing torque until thermal limit | Thermal limit torque | Datasheet validation |
| Peak torque | Short burst (< 1 s) at max current | Verify datasheet peak | Datasheet validation |
| Encoder resolution | Command fine position steps | Min detectable step | Encoder datasheet |
| Encoder accuracy | Compare to external reference | Absolute error | Calibration practice |

### 6.2 Full Robot Testing (Progressive)

| Stage | Test | Prerequisites | Pass Criteria | Source |
|-------|------|-------------|---------------|--------|
| S1: Power-on | All electronics boot, no smoke | Wiring check | No faults, all LEDs correct | Practice |
| S2: Communication | All nodes reachable on CAN/EtherCAT | S1 | All nodes respond to ping | Practice |
| S3: Sensor check | All sensors return plausible data | S2 | IMU gravity ~9.81 m/s², encoders zero at home | Practice |
| S4: Single joint | Each joint moves individually | S3 | Smooth motion, no vibration, limits respected | Practice |
| S5: Gravity comp | Robot holds pose with zero command | S4 | Position drift < 0.1 rad in 30 s | Practice |
| S6: Joint tracking | Sinusoidal trajectories on all joints | S5 | Tracking error < 0.02 rad RMS | Practice |
| S7: Static balance | Robot stands, CoM over support | S6 | Stable for 60 s, no oscillation | Literature |
| S8: Weight shift | Shift CoM left/right, forward/back | S7 | Smooth transition, ZMP in polygon | Literature |
| S9: Squat | CoM up/down 10 cm | S8 | Stable, knees don't buckle | Practice |
| S10: Single step | Lift one foot, place down | S9 | Stable, no fall | Literature |
| S11: Walking in place | Alternating steps, zero forward velocity | S10 | 10 steps stable | Literature |
| S12: Forward walking | Walk 2 m forward | S11 | Completes without falling | Literature |
| S13: Turning | Walk and turn 90° | S12 | Completes without falling | Literature |
| S14: Push recovery | External push while walking | S13 | Recovers or controlled stop | Literature |
| S15: Uneven terrain | Walk over 2 cm obstacles | S14 | Completes without falling | Literature |

---

## 7. Benchmark Suites

### 7.1 Control Performance Benchmarks

| Benchmark | Metric | Target | Source |
|-----------|--------|--------|--------|
| Position tracking (sinusoidal) | RMS error | **< 0.01 rad** | Practice |
| Position tracking (step) | Settling time (2%) | **< 100 ms** | Control theory |
| Velocity tracking | RMS error | **< 0.05 rad/s** | Practice |
| Torque tracking | RMS error | **< 5% of max** | Practice |
| Impedance rendering | Stiffness accuracy | **±10% of commanded** | Literature |
| Bandwidth (-3 dB) | Frequency | **> 50 Hz** for position | Actuator theory |

### 7.2 Locomotion Benchmarks

| Benchmark | Metric | Target | Source |
|-----------|--------|--------|--------|
| Walking speed | m/s | **> 0.5 m/s** (first), **> 1.0 m/s** (mature) | Literature |
| Walking efficiency | Cost of Transport (CoT) | **< 1.0** (dimensionless) | Literature |
| Maximum slope | Degrees | **> 10°** | Literature |
| Step over obstacle | Height | **> 5 cm** | Practice |
| Push recovery | Max impulse | **> 20 N·s** | Literature |
| Single-foot balance | Duration | **> 10 s** | Literature |
| Turning rate | °/s | **> 30 °/s** | Practice |

**Verified Constants:**
- Human Cost of Transport (walking): **~0.2** (dimensionless) (source: biomechanics literature)
- Honda ASIMO CoT: **~1.0–1.5** (source: Honda technical papers)
- Boston Dynamics Atlas CoT: estimated **~0.5–1.0** (source: indirect estimates from specs)
- MIT Cheetah 3 CoT: **~0.5** (source: MIT technical papers)
- Dimensionless CoT = Energy / (mass × g × distance) (source: biomechanics convention)

### 7.3 Computational Benchmarks

| Benchmark | Metric | Target | Source |
|-----------|--------|--------|--------|
| Control loop jitter | Max deviation from period | **< 5%** of period | Real-time practice |
| ROS2 message latency | End-to-end delay | **< 5 ms** for control topics | ROS2 practice |
| EKF update rate | Hz | **> 200 Hz** | State estimation practice |
| MPC solve time | ms | **< 10 ms** for horizon 10 | Optimization practice |
| Neural policy inference | ms | **< 5 ms** on target hardware | ML deployment practice |
| Sim-to-real transfer | Performance drop | **< 20%** vs. simulation | RL robotics literature |

---

## 8. Regression Testing Infrastructure

### 8.1 CI/CD Pipeline

```yaml
# Conceptual CI pipeline for robotics
stages:
  - lint
  - unit_test
  - sim_test
  - hil_test
  - hardware_test

lint:
  script:
    - cpplint --recursive src/
    - clang-tidy src/**/*.cpp
    - mypy src/**/*.py

unit_test:
  script:
    - colcon test --packages-select *
    - pytest tests/unit/
  artifacts:
    reports:
      junit: build/*/test_results/**/*.xml

sim_test:
  script:
    - pytest tests/simulation/ --suite=regression
    - pytest tests/simulation/ --suite=domain_randomization -n 20
  timeout: 2 hours

hil_test:
  script:
    - ./scripts/hil_test.sh --actuators=leg_left
  when: manual  # Requires hardware
  timeout: 1 hour

hardware_test:
  script:
    - ./scripts/hardware_test.sh --stage=S7
  when: manual
  timeout: 30 minutes
```

### 8.2 Test Data Management

| Data Type | Storage | Retention | Source |
|-----------|---------|-----------|--------|
| Unit test results | CI artifact | 30 days | Practice |
| Simulation logs (bags) | NAS / cloud | 90 days | Practice |
| HIL test logs | NAS / cloud | 1 year | Practice |
| Hardware test logs | NAS / cloud | Permanent | Practice |
| Video recordings | NAS / cloud | 1 year | Practice |
| Failure analysis | Issue tracker + wiki | Permanent | Practice |

---

## 9. Safety Testing

### 9.1 Safety Function Tests

| Function | Test | Frequency | Source |
|----------|------|-----------|--------|
| Emergency stop (e-stop) | Press e-stop at max speed | Every session | ISO 10218 |
| Software e-stop | Trigger software fault | Every session | Practice |
| Torque limit enforcement | Command torque > limit | Every session | Practice |
| Joint limit enforcement | Command position > limit | Every session | Practice |
| Velocity limit enforcement | Command velocity > limit | Every session | Practice |
| Communication timeout | Disconnect CAN/EtherCAT | Weekly | Practice |
| Power loss behavior | Disconnect battery mid-motion | Monthly | Practice |
| Thermal shutdown | Overheat simulation | Monthly | Datasheet |
| Collision detection | Unexpected contact detection | Weekly | Practice |

### 9.2 Failure Mode Testing

| Failure Mode | Test | Expected Behavior | Source |
|--------------|------|-------------------|--------|
| Single encoder failure | Disconnect one encoder | Detect fault, hold position or safe pose | Practice |
| Single motor driver failure | Power off one driver | Compensate or enter safe mode | Practice |
| IMU failure | Disconnect IMU | Fall back to kinematic-only state est | Practice |
| Camera failure | Cover camera | Continue with degraded perception | Practice |
| CAN bus fault | Short CAN_H to ground | Detect fault, isolate segment | Practice |
| Battery undervoltage | Simulate low battery | Reduce power, announce shutdown | Practice |
| Software crash | Kill controller node | Watchdog triggers safe state | Practice |

---

## Verified Data Summary

| Data Point | Source | Confidence |
|------------|--------|------------|
| Test pyramid distribution | Software engineering literature (Cohn, etc.) | High |
| Unit test tolerance values | Numerical analysis, robotics practice | Medium |
| Domain randomization parameters | OpenAI, MIT Cheetah sim-to-real papers | High |
| Simulation fidelity levels | Robotics literature | Medium |
| HIL architecture | Aerospace/automotive practice, robotics literature | Medium |
| Progressive hardware testing stages | Humanoid robotics literature, practice | Medium |
| CoT benchmarks (human, ASIMO, Cheetah) | Biomechanics, technical papers | High |
| Control performance targets | Control theory, actuator datasheets | Medium |
| Safety test frequency | ISO 10218, practice | Medium |
| CI/CD patterns | Software engineering practice | High |

---

## Open Questions / Learning Targets

1. What is our baseline sim-to-real gap for each subsystem (actuator, sensor, contact)?
2. How many domain randomization seeds are needed for reliable policy transfer?
3. What is the minimum HIL test set that catches 90% of integration bugs?
4. Can we automate the progressive hardware test stages (S1–S15) with scripted procedures?
5. What is the actual CoT of our robot, and how does it compare to Cheetah/ASIMO?
6. How do we quantify and track sim-to-real gap over development iterations?
7. What is the MTBF (mean time between failures) for each actuator under humanoid loads?
8. Can we build a "digital twin" that stays synchronized with the physical robot?
9. What automated visual inspection can detect mechanical wear before failure?
10. How do we regression-test learned policies without hours of simulation per commit?

---

*Last verified: 2026-06-08. Next review: after first test campaign results.*
