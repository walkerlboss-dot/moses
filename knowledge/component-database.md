# component-database.md — Moses Knowledge Base

> **Domain:** Humanoid Robot Components — Motors, Sensors, Electronics  
> **Status:** Seed document — will grow with vendor research  
> **Last Updated:** 2026-06-08  
> **Confidence:** Medium — specs from datasheets, prices approximate

---

## Actuators

### Electric Motors (Quasi-Direct Drive)

| Motor | Peak Torque | Cont. Torque | Max Speed | Mass | Voltage | Price (est.) | Source |
|-------|-------------|--------------|-----------|------|---------|--------------|--------|
| T-Motor AK80-9 | 48 N·m | 12 N·m | 300 rpm | 0.49 kg | 48V | ~$400 | T-Motor |
| T-Motor AK60-6 | 18 N·m | 6 N·m | 400 rpm | 0.32 kg | 48V | ~$300 | T-Motor |
| T-Motor AK10-9 | 18 N·m | 4.5 N·m | 600 rpm | 0.19 kg | 24V | ~$200 | T-Motor |
| Unitree Go2 motor | 23.7 N·m | ? | ? | ~0.4 kg | ? | ~$350 | Unitree |
| Maxon EC-i 52 | 1.7 N·m | 0.56 N·m | 10,000 rpm | 0.29 kg | 24V | ~$500 | Maxon |
| MyActuator RMD-X8 | 8 N·m | 2.5 N·m | 300 rpm | 0.28 kg | 24V | ~$250 | MyActuator |
| MyActuator RMD-X10 | 13 N·m | 4 N·m | 250 rpm | 0.35 kg | 24V | ~$300 | MyActuator |

**Recommendation for humanoid:**
- **Hip/Knee:** T-Motor AK80-9 or equivalent (high torque, ~40-50 N·m peak)
- **Ankle:** T-Motor AK60-6 or RMD-X10 (moderate torque, fast response)
- **Shoulder/Elbow:** T-Motor AK10-9 or RMD-X8 (lighter, faster)
- **Wrist/Neck:** Maxon EC-i or smaller RMD (precision, light)

### Series Elastic Actuators (SEA)

| Actuator | Peak Torque | Bandwidth | Mass | Price (est.) | Source |
|----------|-------------|-----------|------|--------------|--------|
| MIT Mini-Cheetah | 3.5 N·m | 20 Hz | 0.48 kg | ~$600 (DIY) | MIT Open Source |
| ANYbotics SEA | ? | ? | ? | N/A | Commercial only |
| Custom SEA | Design-dependent | 10-50 Hz | 0.3-1.0 kg | $200-500 (parts) | Self-build |

**Note:** SEA provides force control and shock tolerance but adds compliance complexity. QDD is simpler for first build.

---

## Sensors

### IMU (Inertial Measurement Unit)

| Sensor | DOF | Accuracy | Interface | Mass | Price (est.) | Source |
|--------|-----|----------|-----------|------|--------------|--------|
| VectorNav VN-100 | 9 | ±0.5° heading | UART/SPI | 3.5g | ~$500 | VectorNav |
| TDK ICM-20948 | 9 | Consumer grade | I2C/SPI | 1g | ~$5 | TDK/InvenSense |
| Bosch BMI088 | 6 | High stability | SPI | 1g | ~$10 | Bosch |
| MicroStrain 3DM-GX5 | 9 | ±0.25° heading | UART/USB | 18g | ~$2000 | MicroStrain |

**Recommendation:** Start with BMI088 (robust, cheap) or ICM-20948 (cheap, common). Upgrade to VectorNav for production.

### Force/Torque Sensors

| Sensor | DOF | Max Force | Interface | Price (est.) | Source |
|--------|-----|-----------|-----------|--------------|--------|
| ATI Nano25 | 6 | 250N | Analog/Digital | ~$5000 | ATI |
| Robotiq FT-300 | 6 | 300N | Modbus | ~$4000 | Robotiq |
| Custom strain gauge | 1-6 | Design-dependent | Wheatstone bridge | $50-200 | Self-build |

**Recommendation:** Start with custom 1-DOF foot force sensors (strain gauges). Upgrade to 6-DOF for manipulation.

### Joint Encoders

| Encoder | Resolution | Interface | Price (est.) | Source |
|---------|------------|-----------|--------------|--------|
| AMT102-V | 2048 PPR | Quadrature | ~$30 | CUI Devices |
| AS5048A | 14-bit | SPI/PWM | ~$15 | AMS |
| RLS AksIM-2 | 19-bit | SPI | ~$200 | RLS |

**Recommendation:** AS5048A for most joints (cheap, magnetic, no contact). AksIM-2 for high-precision joints.

### Cameras / Vision

| Camera | Resolution | FPS | Depth | Interface | Price (est.) | Source |
|--------|------------|-----|-------|-----------|--------------|--------|
| Intel RealSense D455 | 1280×720 | 90 | Stereo IR | USB3 | ~$400 | Intel |
| Intel RealSense D405 | 1280×720 | 90 | Close-range | USB3 | ~$300 | Intel |
| OAK-D | 4K | 60 | Stereo + AI | USB3 | ~$300 | Luxonis |
| FLIR Blackfly S | 5MP | 75 | No | GigE | ~$800 | FLIR |

**Recommendation:** OAK-D or RealSense D455 for onboard perception. OAK-D has onboard neural inference.

---

## Compute

| Board | CPU | GPU | RAM | Power | Price (est.) | Source |
|-------|-----|-----|-----|-------|--------------|--------|
| NVIDIA Jetson AGX Orin | 12-core ARM | 2048-core Ampere | 32GB | 15-60W | ~$1500 | NVIDIA |
| NVIDIA Jetson Orin NX | 8-core ARM | 1024-core Ampere | 16GB | 10-25W | ~$600 | NVIDIA |
| Intel NUC 13 Pro | i7-1360P | Iris Xe | 64GB | 65W | ~$800 | Intel |
| Raspberry Pi 5 | 4-core ARM | VideoCore VII | 8GB | 8W | ~$80 | Raspberry Pi |

**Recommendation:** Jetson AGX Orin for primary compute (AI + control). NUC for development/debugging. Pi 5 for low-level I/O.

---

## Power

| Component | Spec | Price (est.) | Source |
|-----------|------|--------------|--------|
| LiPo Battery 6S 10000mAh | 22.2V, 222Wh | ~$150 | Tattu/HRB |
| LiPo Battery 12S 10000mAh | 44.4V, 444Wh | ~$300 | Tattu |
| DC-DC Buck 48V→12V | 10A | ~$30 | Mean Well |
| DC-DC Buck 48V→5V | 5A | ~$15 | Pololu |
| Battery Management System | 6S-12S | ~$100 | Various |

**Power budget estimate (full humanoid):**
- 12× high-torque motors @ 100W peak = 1200W
- Compute (Jetson) = 30W
- Sensors = 10W
- **Total peak:** ~1250W
- **Runtime target:** 1 hour → ~1.3 kWh battery
- **Battery mass:** ~5-8 kg (LiPo)

---

## Fasteners & Bearings

| Component | Spec | Price (est.) | Source |
|-----------|------|--------------|--------|
| Deep groove ball bearing 6204 | 20×47×14mm | ~$5 | SKF/NSK |
| Angular contact bearing 7204 | 20×47×14mm | ~$15 | SKF/NSK |
| Crossed roller bearing | Compact, high rigidity | ~$50-200 | THK/IKO |
| SHCS M4×20 | Stainless | ~$0.10 | McMaster-Carr |
| SHCS M6×30 | Stainless | ~$0.15 | McMaster-Carr |
| Helicoil M6×1.0 | Thread repair/insert | ~$0.50 | McMaster-Carr |

---

## Cost Estimate (Titan Mark I — Reference)

| Subsystem | Components | Est. Cost |
|-----------|------------|-----------|
| Actuators (12×) | T-Motor AK series mix | ~$4,000 |
| Motor drivers | Integrated or separate | ~$1,500 |
| Sensors (IMU, encoders, F/T) | Mid-range | ~$1,000 |
| Cameras (2×) | RealSense/OAK-D | ~$700 |
| Compute | Jetson AGX Orin | ~$1,500 |
| Power | Battery + BMS + converters | ~$600 |
| Frame/materials | Aluminum, 3D printed parts | ~$1,000 |
| Fasteners, bearings, misc | | ~$500 |
| **Total** | | **~$10,800** |

**Note:** This is a research prototype estimate. Commercial humanoids target $20K-100K. DIY with careful sourcing can reduce to $5K-8K.

---

## Verified Data

| Data Point | Source | Confidence |
|------------|--------|------------|
| T-Motor AK80-9 specs | T-Motor datasheet | High |
| Jetson AGX Orin specs | NVIDIA datasheet | High |
| RealSense D455 specs | Intel datasheet | High |
| Power budget estimate | Calculated from specs | Medium |
| Cost estimates | Approximate, market-dependent | Low-Medium |

---

## Open Questions / Research Targets

1. What are actual lead times for T-Motor actuators?
2. Can we get educational pricing on Jetson or RealSense?
3. What is the failure rate of QDD actuators under humanoid loads?
4. Are there open-source motor driver designs compatible with T-Motor?
5. What is the minimum viable sensor set for stable walking?

---

*Last verified: 2026-06-08. Update frequency: weekly (as market changes).*
