# software-architecture.md — Moses Knowledge Base

> **Domain:** Software Architecture for Humanoid Robotics
> **Status:** Seed document — will grow with implementation and profiling
> **Last Updated:** 2026-06-08
> **Confidence:** Medium — patterns from ROS2 docs, robotics literature, and real-time systems theory; not yet validated on hardware

---

## 1. ROS2 Node Graph

### 1.1 Typical Humanoid ROS2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 DOMAIN (e.g., domain_id=0)                    │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  /hardware   │    │  /state_est  │    │  /controller │                   │
│  │   interface  │◄──►│   imation    │◄──►│   (WBC/MPC)  │                   │
│  │              │    │              │    │              │                   │
│  │ Pubs:        │    │ Pubs:        │    │ Pubs:        │                   │
│  │  /joint_states│    │  /odom       │    │  /joint_cmds │                   │
│  │  /imu/data   │    │  /tf         │    │  /balance    │                   │
│  │  /ft_sensor  │    │  /body_pose  │    │  /gait_phase │                   │
│  │              │    │              │    │              │                   │
│  │ Subs:        │    │ Subs:        │    │ Subs:        │                   │
│  │  /joint_cmds │    │  /joint_states│    │  /odom       │                   │
│  │              │    │  /imu/data   │    │  /body_pose    │                   │
│  │              │    │  /ft_sensor  │    │  /cmd_vel    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         ▲                   ▲                   ▲                            │
│         │                   │                   │                            │
│  ┌──────┴──────┐     ┌─────┴──────┐     ┌─────┴──────┐                      │
│  │  /sensors   │     │  /vision   │     │  /planner  │                      │
│  │   (drivers) │     │  (perception)│    │  (behavior)│                      │
│  │             │     │            │     │            │                      │
│  │ Pubs:       │     │ Pubs:      │     │ Pubs:      │                      │
│  │  /imu/data  │     │  /camera/  │     │  /cmd_vel  │                      │
│  │  /ft_sensor │     │   color/   │     │  /footsteps│                      │
│  │  /joint_states│   │   depth    │     │  /manip/   │                      │
│  │             │     │  /detected_│     │   target   │                      │
│  │             │     │   objects  │     │            │                      │
│  └─────────────┘     └────────────┘     └────────────┘                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  /diagnostics  │  /logging  │  /parameter_server  │  /rosbridge     │    │
│  │  (health monitoring, data recording, config, remote UI)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 ROS2 QoS (Quality of Service) Policies

| Use Case | Reliability | Durability | History | Depth | Source |
|----------|-------------|------------|---------|-------|--------|
| Joint commands (real-time) | Best Effort | Volatile | Keep Last | 1 | ROS2 docs |
| Joint states (real-time) | Best Effort | Volatile | Keep Last | 1 | ROS2 docs |
| IMU data (real-time) | Best Effort | Volatile | Keep Last | 1 | ROS2 docs |
| Camera images | Best Effort | Volatile | Keep Last | 1 | ROS2 docs |
| Parameter updates | Reliable | Volatile | Keep Last | 1 | ROS2 docs |
| Diagnostics / logs | Reliable | Transient Local | Keep Last | 10 | ROS2 docs |
| Map / static TF | Reliable | Transient Local | Keep Last | 1 | ROS2 docs |

**Verified Constants:**
- ROS2 default middleware (RMW): **Fast DDS** (formerly Fast RTPS) on most platforms (source: ROS2 docs)
- DDS discovery time: **~1–5 seconds** for node-to-node matching (source: DDS spec, ROS2 docs)
- ROS2 Humble LTS: supported until **May 2027** (source: ROS2 release schedule)
- ROS2 Jazzy LTS: supported until **May 2029** (source: ROS2 release schedule)

**Feasibility Boundaries:**
- ROS2 is **not hard real-time by default**; requires real-time kernel (PREEMPT_RT) and careful node design
- DDS overhead: serialization, deserialization, memory allocation in data path → latency
- For <1 kHz control loops, consider bypassing ROS2 for the fast path (shared memory, EtherCAT, direct SPI)
- ROS2 is excellent for: sensor fusion, visualization (RViz), logging, parameter management, high-level planning

---

## 2. Real-Time Constraints

### 2.1 Control Frequency Budgets

| Control Layer | Typical Frequency | Latency Budget | Jitter Tolerance | Source |
|---------------|-------------------|----------------|------------------|--------|
| Motor current/torque (FOC) | **8–20 kHz** | **< 50 µs** | **< 5 µs** | Motor control theory |
| Joint position/velocity (PD) | **1–5 kHz** | **< 200 µs** | **< 20 µs** | Robotics practice |
| Whole-body inverse dynamics | **500 Hz – 1 kHz** | **< 1 ms** | **< 100 µs** | Literature (MIT Cheetah, etc.) |
| Model predictive control (MPC) | **50–200 Hz** | **< 5 ms** | **< 1 ms** | Literature |
| State estimation (EKF) | **100–500 Hz** | **< 2 ms** | **< 500 µs** | Robotics practice |
| Gait planning / footstep | **10–50 Hz** | **< 20 ms** | **< 5 ms** | Literature |
| High-level behavior | **1–10 Hz** | **< 100 ms** | **< 20 ms** | Robotics practice |
| Vision / perception | **10–30 Hz** | **< 50 ms** | **< 10 ms** | CV/robotics practice |

**Verified Constants:**
- Humanoid CoM stabilization bandwidth: **~5–10 Hz** (source: Kajita et al., "Introduction to Humanoid Robotics")
- Joint mechanical bandwidth (typical QDD actuator): **~20–70 Hz** (source: actuator datasheets, system ID)
- Nyquist criterion: control loop must sample at **> 2×** the system bandwidth (source: control theory)
- Practical rule: sample at **> 10×** bandwidth for good performance (source: digital control practice)

### 2.2 Latency Breakdown (Example: Joint Position Control)

| Stage | Typical Latency | Notes |
|-------|-----------------|-------|
| Encoder sampling | **10–50 µs** | SPI/I2C read or ABI interrupt |
| Sensor data transmission (CAN) | **0.1–1 ms** | Depends on bus load, priority |
| State estimation | **0.2–1 ms** | EKF/UKF update |
| Controller computation | **0.1–2 ms** | PD, inverse dynamics, or MPC |
| Command transmission (CAN) | **0.1–1 ms** | |
| Motor driver processing | **50–200 µs** | ODrive/SimpleFOC loop |
| Power stage response | **10–50 µs** | Gate drive + MOSFET switching |
| Motor mechanical response | **1–10 ms** | Electrical + mechanical time constants |
| **Total (end-to-end)** | **~2–15 ms** | Budget: < 5 ms for stable walking |

**Feasibility Boundaries:**
- End-to-end latency > **20 ms** → noticeable instability in walking (source: humanoid robotics literature)
- Jitter > **20% of control period** → degraded performance, potential oscillation (source: digital control theory)
- For 1 kHz whole-body control: total latency must be **< 1 ms** for the fast path
- ROS2 DDS adds **0.1–1 ms** per hop; bypass DDS for fastest loops

---

## 3. Multi-Threading and Execution

### 3.1 Threading Model

| Thread / Task | Priority | Core Affinity | Period | Source |
|---------------|----------|---------------|--------|--------|
| Motor control ISR | **Highest (99, SCHED_FIFO)** | Dedicated core | **Event-driven (encoder interrupt)** | Real-time practice |
| FOC loop | **Very High (90–95)** | Dedicated core | **100–125 µs** (8–10 kHz) | Motor control practice |
| Joint servo loop | **High (80–89)** | Dedicated core | **200–1000 µs** (1–5 kHz) | Robotics practice |
| Whole-body controller | **High (70–79)** | Shared core | **1–2 ms** (500–1000 Hz) | Robotics practice |
| State estimator | **Medium-High (60–69)** | Shared core | **2–10 ms** (100–500 Hz) | Robotics practice |
| Sensor drivers | **Medium (50–59)** | Shared core | **Event-driven / polled** | ROS2 practice |
| ROS2 executor | **Medium (40–49)** | Shared core | **10 ms** (100 Hz) | ROS2 docs |
| Logging / diagnostics | **Low (10–19)** | Any core | **100 ms** | System practice |
| UI / visualization | **Lowest (1–9)** | Any core | **As needed** | System practice |

**Verified Constants:**
- Linux PREEMPT_RT patch: reduces kernel latency to **< 100 µs** typical, **< 30 µs** tuned (source: OSADL tests, PREEMPT_RT docs)
- Standard Linux kernel latency: **~1–10 ms** (source: cyclictest benchmarks)
- `SCHED_FIFO` priority range: **1–99** (source: Linux man pages)
- Cache coherence penalty (cross-core access): **~50–100 ns** (source: CPU architecture docs)
- Context switch time (Linux): **~1–10 µs** (source: OS benchmarks)

**Feasibility Boundaries:**
- Real-time threads must **never block** (no malloc, no file I/O, no page faults)
- Lock-free data structures (ring buffers, atomic variables) for inter-thread communication
- Memory must be pre-allocated and locked (`mlockall`) to prevent page faults
- CPU isolation (`isolcpus`) for real-time cores prevents kernel tasks from preempting control threads

### 3.2 Lock-Free Patterns

```cpp
// Single-producer, single-consumer ring buffer (lock-free)
template<typename T, size_t N>
class SPSC_RingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    std::array<T, N> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    
public:
    bool push(const T& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t next = (h + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire)) return false; // full
        buffer_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false; // empty
        item = buffer_[t];
        tail_.store((t + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};
```

**Source:** Classic lock-free algorithm, Herb Sutter / Dmitry Vyukov / Linux kernel ring buffer.

---

## 4. Sensor Fusion Architecture

### 4.1 State Estimation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   IMU       │    │   Encoders  │    │  Force/Torque│    │   Vision    │
│  (200–1000  │    │  (1–5 kHz)  │    │  (100–500   │    │  (10–30 Hz) │
│    Hz)      │    │             │    │    Hz)      │    │             │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SENSOR PREPROCESSING                             │
│  • IMU: bias estimation, temperature compensation, low-pass filter   │
│  • Encoders: velocity filtering (finite diff + LPF), wrap handling   │
│  • F/T: calibration matrix, bias removal, coordinate transform       │
│  • Vision: feature extraction, depth alignment, timestamp sync       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STATE ESTIMATOR (EKF / UKF)                      │
│                                                                      │
│  State vector x: [position, velocity, orientation, angular_vel,      │
│                   imu_bias_accel, imu_bias_gyro, foot_positions]     │
│                                                                      │
│  Prediction: IMU integration (high rate, ~1 kHz)                     │
│  Update:                                                                   │
│    • Zero-velocity update (when foot stationary)                     │
│    • Kinematic update (encoder-based CoM position)                   │
│    • F/T-based contact detection                                     │
│    • Visual odometry (when available, lower weight)                  │
│                                                                      │
│  Output: /odom, /body_pose, /tf (odom→base_link)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 EKF State Vector (Humanoid Example)

| State Element | Dimension | Update Source | Frequency | Source |
|---------------|-----------|---------------|-----------|--------|
| Base position (x, y, z) | 3 | Kinematic chain, VO | 100–500 Hz | Literature |
| Base velocity (vx, vy, vz) | 3 | IMU integration, ZVU | 1 kHz | Literature |
| Base orientation (quaternion) | 4 | IMU (gyro + accel fusion) | 1 kHz | Literature |
| Angular velocity | 3 | IMU gyro | 1 kHz | Literature |
| IMU accel bias | 3 | Zero-velocity updates | 10–100 Hz | Literature |
| IMU gyro bias | 3 | Static periods | 10–100 Hz | Literature |
| Foot positions (x, y, z per foot) | 6 | Kinematic chain, F/T | 100–500 Hz | Literature |
| Joint positions | N_joints | Encoders | 1–5 kHz | Literature |
| Joint velocities | N_joints | Encoders (filtered) | 1–5 kHz | Literature |

**Verified Constants:**
- IMU bias drift (MEMS, consumer): **~0.01–0.1 °/s** gyro, **~0.01–0.1 m/s²** accel (source: IMU datasheets)
- IMU bias drift (tactical-grade): **~0.001–0.01 °/s** gyro (source: VectorNav, MicroStrain datasheets)
- Double integration of accel bias → position drift **~t²** (source: inertial navigation theory)
- Zero-velocity update (ZVU) reduces velocity drift to **~0.01 m/s** (source: legged robotics literature)

### 4.3 Sensor Synchronization

| Method | Accuracy | Complexity | Best For | Source |
|--------|----------|------------|----------|--------|
| Hardware trigger (GPIO) | **< 1 µs** | High | Multi-camera, IMU+camera | Hardware design |
| PTP (IEEE 1588) | **< 1 µs** (hardware) | Medium | Networked sensors, EtherCAT | IEEE 1588 |
| NTP | **1–10 ms** | Low | Non-critical timestamping | NTP spec |
| ROS2 message timestamps | **~1 ms** | Low | ROS2 ecosystem | ROS2 docs |
| Software polling | **1–10 ms** | Low | Simple setups | Practice |

**Verified Constants:**
- PTP hardware timestamping: **±25 ns** accuracy with hardware support (source: IEEE 1588-2008)
- ROS2 `builtin_interfaces/Time`: nanosecond resolution, microsecond practical accuracy (source: ROS2 docs)
- Camera rolling shutter readout time: **~10–30 ms** for full frame (source: camera datasheets)

**Feasibility Boundaries:**
- Misaligned timestamps between IMU and camera → incorrect state estimates in visual-inertial odometry
- Rolling shutter cameras require motion compensation if robot is moving during exposure
- For VIO: IMU and camera should be hardware-triggered and rigidly mounted together

---

## 5. Control Frequency Budgets

### 5.1 Computational Load Estimates

| Task | FLOPs / Cycle | Typical CPU Time | Target Platform | Source |
|------|---------------|------------------|-----------------|--------|
| FOC (per motor) | ~500 FLOPs | **< 50 µs** @ 168 MHz | STM32F4 | Estimated |
| PD control (12 joints) | ~100 FLOPs | **< 10 µs** @ 168 MHz | STM32F4 | Estimated |
| Inverse kinematics (leg) | ~5,000 FLOPs | **< 100 µs** @ 168 MHz | STM32F4 | Literature |
| Inverse dynamics (whole body) | ~50,000 FLOPs | **< 1 ms** @ 2.4 GHz | x86 / ARM A72 | Literature |
| QP solver (small, 20 vars) | ~1M FLOPs | **< 5 ms** @ 2.4 GHz | x86 / ARM A72 | Literature |
| MPC (horizon 10, 12 states) | ~10–100M FLOPs | **< 20 ms** @ 2.4 GHz | x86 / ARM A72 | Literature |
| EKF update (20 states) | ~100K FLOPs | **< 2 ms** @ 2.4 GHz | x86 / ARM A72 | Literature |
| Neural network policy (MLP, 256×256) | ~1M FLOPs | **< 5 ms** @ 2.4 GHz | x86 / ARM A72 | Estimated |
| Neural network (GPU, Jetson) | ~1M FLOPs | **< 1 ms** @ Jetson GPU | Jetson AGX Orin | Estimated |
| Point cloud processing | ~100M–1B FLOPs | **< 50 ms** | Jetson AGX Orin | Practice |

**Verified Constants:**
- STM32F4 @ 168 MHz: **~210 DMIPS**, **~300 MFLOPS** (single-precision, theoretical) (source: ST datasheet)
- Jetson AGX Orin: **170 INT8 TOPS**, **~5.3 TFLOPS FP16** (source: NVIDIA datasheet)
- Raspberry Pi 5 @ 2.4 GHz: **~3,000 DMIPS** per core (source: benchmarks)

### 5.2 CPU Budget Allocation (Example: Raspberry Pi 5, 4 cores)

| Core | Assignment | Tasks | Expected Load |
|------|------------|-------|---------------|
| Core 0 | Linux + ROS2 | DDS, node graph, non-RT tasks | 30–60% |
| Core 1 | Real-time control | Whole-body controller, IK/ID, QP | 40–70% |
| Core 2 | State estimation | EKF/UKF, sensor fusion | 20–40% |
| Core 3 | Perception + planning | Vision, gait planner, behavior | 30–60% |

**Feasibility Boundaries:**
- Real-time tasks should be **isolated** to dedicated cores with `isolcpus` + `taskset` + `SCHED_FIFO`
- Leave **> 30% headroom** on real-time cores to handle worst-case execution time (WCET)
- Cache thrashing: avoid bouncing data between cores; use per-core data structures

---

## 6. Software Patterns

### 6.1 Control Loop Template (C++)

```cpp
#include <pthread.h>
#include <time.h>

void* control_thread(void* arg) {
    // Real-time setup
    struct sched_param param = {.sched_priority = 80};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    const int64_t period_ns = 1'000'000; // 1 ms = 1 kHz
    struct timespec next_time;
    clock_gettime(CLOCK_MONOTONIC, &next_time);
    
    // Pre-allocate all memory (no malloc in loop)
    State state;
    Command cmd;
    
    while (running) {
        // 1. Read sensors (lock-free from sensor thread)
        state = sensor_buffer.read();
        
        // 2. Run controller
        cmd = controller.update(state);
        
        // 3. Write commands (lock-free to motor thread)
        command_buffer.write(cmd);
        
        // 4. Sleep until next period
        next_time.tv_nsec += period_ns;
        if (next_time.tv_nsec >= 1'000'000'000) {
            next_time.tv_sec++;
            next_time.tv_nsec -= 1'000'000'000;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_time, nullptr);
    }
    return nullptr;
}
```

**Source:** Classic real-time control loop pattern, Linux real-time programming guides.

### 6.2 ROS2 Real-Time Executor (rclcpp)

```cpp
// Single-threaded executor with custom priority
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControllerNode>();
    
    // Set real-time priority for this thread
    struct sched_param param = {.sched_priority = 70};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
```

**Feasibility Boundaries:**
- `SingleThreadedExecutor`: deterministic, single core, good for real-time
- `MultiThreadedExecutor`: better throughput, worse determinism
- Callback groups: `Reentrant` vs `MutuallyExclusive` for concurrency control
- For hard real-time: avoid `spin()` in fast loop; use custom executor or bypass ROS2 for fast path

---

## 7. Logging and Diagnostics

### 7.1 Logging Strategy

| Log Level | Frequency | Content | Storage | Source |
|-----------|-----------|---------|---------|--------|
| DEBUG | Per cycle (1 kHz) | Raw sensor values, controller internals | Ring buffer in RAM (circular) | Practice |
| INFO | Per step / event | State changes, mode transitions | SD card / SSD | Practice |
| WARN | As needed | Near-limit conditions, soft faults | SD card / SSD + telemetry | Practice |
| ERROR | As needed | Faults, safety violations | SD card / SSD + telemetry + alert | Practice |
| FATAL | Immediate | Critical failure, emergency stop | All channels + persistent storage | Practice |

**Verified Constants:**
- ROS2 bag file format: **MCAP** (modern), **SQLite3** (legacy) (source: ROS2 docs)
- MCAP write throughput: **> 1 GB/s** possible (source: Foxglove benchmarks)
- SD card write latency: **~1–100 ms** (highly variable) (source: storage benchmarks)
- eMMC write latency: **~1–10 ms** (more consistent) (source: storage benchmarks)

**Feasibility Boundaries:**
- Never log to disk from real-time thread; use lock-free ring buffer + separate logger thread
- Ring buffer size: **~10–60 seconds** of high-rate data at 1 kHz (source: memory budget)
- Triggered logging: capture N seconds before/after an event (source: flight recorder pattern)

---

## Verified Data Summary

| Data Point | Source | Confidence |
|------------|--------|------------|
| ROS2 architecture patterns | ROS2 docs, design articles | High |
| ROS2 QoS recommendations | ROS2 docs, community best practices | High |
| DDS / Fast DDS behavior | eProsima docs, ROS2 docs | High |
| Control frequency budgets | Robotics literature (Kajita, Raibert, etc.) | High |
| Latency requirements for stable walking | Humanoid robotics literature | High |
| Linux PREEMPT_RT performance | OSADL, PREEMPT_RT docs, benchmarks | High |
| Linux scheduling / priorities | Linux man pages, kernel docs | High |
| Lock-free algorithms | Academic literature, Linux kernel | High |
| EKF/UKF for humanoids | Literature (Bloesch, Rotella, etc.) | High |
| IMU bias drift specs | IMU datasheets, inertial nav theory | High |
| Sensor synchronization methods | IEEE 1588, hardware datasheets | High |
| CPU performance (STM32, Pi, Jetson) | Manufacturer datasheets, benchmarks | High |
| Computational load estimates | Literature, rough calculations | Medium |
| Real-world latency measurements | Not yet measured on our hardware | Low |

---

## Open Questions / Learning Targets

1. What is the actual end-to-end latency and jitter of our ROS2 node graph on target hardware?
2. Can we achieve < 1 ms whole-body control loop on Raspberry Pi 5 with PREEMPT_RT?
3. What is the worst-case execution time (WCET) of our inverse dynamics solver?
4. How does DDS serialization overhead affect joint command latency at 1 kHz?
5. Can we implement a lock-free shared-memory bridge between ROS2 and bare-metal motor controller?
6. What is the optimal EKF state vector size for our sensor set (minimize without losing observability)?
7. How do we handle timestamp misalignment between IMU (1 kHz) and camera (30 Hz) in VIO?
8. What is the memory bandwidth bottleneck when logging 12 joints × 1 kHz to SD card?
9. Can we run MPC at 200 Hz on Jetson AGX Orin for dynamic walking?
10. How do we implement graceful degradation (fallback controllers) when a sensor fails?

---

*Last verified: 2026-06-08. Next review: after first software integration and profiling.*
