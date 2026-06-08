# Industrial Integration API — Moses

> **PLC, EtherCAT, ROS2 Industrial, and safety systems.**

---

## PLCInterface

```python
from moses.industrial.plc import PLCInterface
```

Interfaces with Programmable Logic Controllers.

### Supported Protocols

| Protocol | Standard | Use Case |
|----------|----------|----------|
| **Modbus TCP** | IEC 61158 | Simple I/O |
| **OPC-UA** | IEC 62541 | Complex data |
| **EtherNet/IP** | ODVA | Allen-Bradley |
| **Profinet** | IEC 61158 | Siemens |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `connect()` | `ip`, `protocol` | `bool` | Connect to PLC |
| `read_digital()` | `address` | `bool` | Read digital input |
| `write_digital()` | `address`, `value` | — | Write digital output |
| `read_analog()` | `address` | `float` | Read analog input |
| `write_analog()` | `address`, `value` | — | Write analog output |
| `read_safety()` | `address` | `bool` | Read safety I/O |

### Safety I/O

| Signal | Type | Description |
|--------|------|-------------|
| E-stop | Input | Emergency stop button |
| Light curtain | Input | Safety perimeter |
| Door lock | Input | Enclosure door |
| Safety gate | Input | Access gate |
| Reset | Output | Reset safety circuit |

---

## EtherCATMaster

```python
from moses.industrial.ethercat import EtherCATMaster
```

EtherCAT master for real-time servo control.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `scan_slaves()` | — | `list` | Discover slaves |
| `configure_pdo()` | `slave_id`, `pdo_mapping` | — | Configure PDO |
| `start_cyclic()` | `frequency` | — | Start cyclic operation |
| `write_outputs()` | `slave_id`, `data` | — | Write servo commands |
| `read_inputs()` | `slave_id` | `dict` | Read servo state |
| `sync_distributed_clocks()` | — | — | Sync DC |

### CiA 402 Drive Profile

| Object | Index | Description |
|--------|-------|-------------|
| Control word | 0x6040 | Enable, fault reset |
| Status word | 0x6041 | Ready, error |
| Target position | 0x607A | Position setpoint |
| Target velocity | 0x60FF | Velocity setpoint |
| Target torque | 0x6071 | Torque setpoint |
| Actual position | 0x6064 | Current position |
| Actual velocity | 0x606C | Current velocity |
| Actual torque | 0x6077 | Current torque |

### Cycle Time

| Application | Cycle Time | Jitter |
|-------------|-----------|--------|
| Position control | 1 ms | < 100 µs |
| Velocity control | 500 µs | < 50 µs |
| Torque control | 250 µs | < 25 µs |

---

## ROS2Industrial

```python
from moses.industrial.ros2_industrial import ROS2Industrial
```

ROS2 Industrial integration.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `load_moveit_config()` | `robot_name` | — | Load MoveIt config |
| `plan_trajectory()` | `start`, `goal` | `RobotTrajectory` | Plan motion |
| `execute_trajectory()` | `trajectory` | `bool` | Execute motion |
| `load_calibration()` | `calibration_file` | — | Load calibration |

### Supported Robots

| Robot | Driver | Status |
|-------|--------|--------|
| Universal Robots UR10 | `ur_robot_driver` | ✅ Supported |
| FANUC LR Mate | `fanuc_driver` | ✅ Supported |
| ABB IRB 1200 | `abb_driver` | 🟡 In progress |
| KUKA iiwa | `kuka_driver` | 🟡 In progress |

---

## SafetySystem

```python
from moses.industrial.safety import SafetySystem
```

Industrial safety compliance.

### Risk Assessment (ISO 12100)

| Step | Description |
|------|-------------|
| 1. Hazard identification | Identify all hazards |
| 2. Risk estimation | Severity × Probability |
| 3. Risk evaluation | Acceptable? |
| 4. Risk reduction | Design guards |

### Performance Level (ISO 13849)

| PL | PFH (1/h) | Category | Use Case |
|----|-----------|----------|----------|
| a | 5e-5 | B | Low risk |
| b | 3e-5 | 1 | Low risk |
| c | 1e-5 | 2 | Medium risk |
| d | 1e-6 | 3 | High risk |
| e | 1e-7 | 4 | Very high risk |

### Collaborative Robot Safety (ISO/TS 15066)

| Parameter | Limit | Description |
|-----------|-------|-------------|
| Force | 150 N | Max contact force |
| Pressure | 150 N/cm² | Max pressure |
| Speed | 1.5 m/s | Max tool speed |
| Power | 80 W | Max mechanical power |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `assess_risk()` | `hazards` | `dict` | Risk assessment |
| `calculate_pl()` | `safety_functions` | `str` | Calculate PL |
| `verify_compliance()` | — | `dict` | Compliance report |

---

*Safety first. Production second.*
