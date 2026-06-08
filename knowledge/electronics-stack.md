# electronics-stack.md — Moses Knowledge Base

> **Domain:** Electronics & Embedded Systems for Humanoid Robotics
> **Status:** Seed document — will grow with hardware bring-up and integration
> **Last Updated:** 2026-06-08
> **Confidence:** Medium — specs from datasheets and manuals; integration wisdom from community and literature

---

## 1. Motor Drivers

### 1.1 ODrive

**Overview:** Open-source brushless DC motor controller with FOC (field-oriented control), designed for robotics.

| Parameter | ODrive v3.6 | ODrive Pro | Source |
|-----------|-------------|------------|--------|
| Voltage range | 12–56 V | 12–56 V | ODrive docs |
| Continuous current | 50 A (per axis) | 100 A (per axis) | ODrive docs |
| Peak current | 100 A | 150 A | ODrive docs |
| Max power (per axis) | ~2.8 kW @ 56V | ~5.6 kW @ 56V | Calculated |
| Control modes | Position, velocity, torque (current) | Same + improved | ODrive docs |
| Encoder interfaces | ABI, SPI, RS-485 (varies by version) | Same | ODrive docs |
| Communication | USB, UART, CAN | USB, UART, CAN, EtherCAT (Pro) | ODrive docs |
| Loop frequency | 8 kHz (FOC) | 8 kHz (FOC) | ODrive docs |
| Price | ~$150–200 | ~$300–400 | ODrive store |

**Verified Constants:**
- FOC loop frequency: **8 kHz** (source: ODrive firmware, ODrive docs)
- Current control bandwidth: **~1–2 kHz** achievable (source: ODrive community, depends on motor inductance)
- Encoder resolution with AS5047P (14-bit): **16,384 counts/rev** → **0.022°** resolution (source: AMS datasheet)

**Feasibility Boundaries:**
- Requires careful tuning (PI gains, current limits, encoder calibration)
- Regenerative braking can overvoltage the DC bus if battery cannot absorb; needs brake resistor or overvoltage protection
- Thermal management critical at high currents; PCB and MOSFET temperatures must be monitored
- ODrive v3.6 has known limitations (single-ended encoder inputs, no hardware STO); Pro improves many of these

**Open Questions:**
1. What is the actual current derating curve vs. ambient temperature for ODrive Pro?
2. Can ODrive handle the peak current demands of a humanoid hip joint during a stumble recovery?
3. What is the EMI/RFI footprint of ODrive and how does it affect nearby IMU and comms?

---

### 1.2 SimpleFOC

**Overview:** Open-source FOC library for Arduino/ESP32/STM32, lower-level than ODrive.

| Parameter | Value | Source |
|-----------|-------|--------|
| Supported MCUs | Arduino (Uno, Mega), ESP32, STM32, Teensy | SimpleFOC docs |
| PWM frequency | 20–50 kHz typical | SimpleFOC docs |
| FOC loop frequency | 1–10 kHz (MCU-dependent) | SimpleFOC docs |
| Current sensing | Inline, low-side (driver-dependent) | SimpleFOC docs |
| Sensor interfaces | Encoder, Hall, sensorless | SimpleFOC docs |
| Price | Free (open source) | SimpleFOC GitHub |

**Feasibility Boundaries:**
- More DIY than ODrive; requires custom power stage (MOSFET driver, gate drive)
- Best for smaller motors (<500W) or custom driver designs
- Excellent for learning FOC internals and rapid prototyping
- ESP32 variant offers WiFi/Bluetooth for wireless tuning

---

### 1.3 Commercial Servo Drives (Elmo, Ingenia, Copley)

| Parameter | Elmo Gold Solo Twitter | Ingenia Summit | Copley Xenus | Source |
|-----------|------------------------|----------------|--------------|--------|
| Voltage | 12–100 V | 12–100 V | 12–180 V | Datasheets |
| Continuous current | 10–40 A | 10–60 A | 5–30 A | Datasheets |
| Peak current | 20–80 A | 20–120 A | 10–60 A | Datasheets |
| Communication | CANopen, EtherCAT, RS-232 | CANopen, EtherCAT | CANopen, EtherCAT | Datasheets |
| Safety | STO, SS1 | STO | STO | Datasheets |
| Price | ~$500–1,500 | ~$300–800 | ~$400–1,000 | Distributors |

**Feasibility Boundaries:**
- Industrial-grade reliability, safety certifications (STO = safe torque off)
- Higher cost than ODrive; justified for production or safety-critical joints
- Smaller form factors available (e.g., Elmo Twitter is credit-card sized)

---

### 1.4 Integrated Smart Actuators (T-Motor, Unitree, MyActuator)

| Parameter | T-Motor AK80-9 | Unitree Go2 motor | MyActuator RMD-X10 | Source |
|-----------|---------------|-------------------|--------------------|--------|
| Peak torque | 48 N·m | 23.7 N·m | 13 N·m | Datasheets |
| Continuous torque | 12 N·m | ? | 4 N·m | Datasheets |
| Max speed | 300 rpm | ? | 250 rpm | Datasheets |
| Voltage | 48 V | ? | 24 V | Datasheets |
| Communication | CAN | CAN | CAN/RS-485 | Datasheets |
| Integrated encoder | Yes (multi-turn absolute) | Yes | Yes | Datasheets |
| Price | ~$400 | ~$350 | ~$300 | Distributors |

**Feasibility Boundaries:**
- All-in-one: motor + driver + encoder + gearbox in one package
- Reduced wiring, easier integration
- Less flexibility in control algorithms (closed firmware)
- Gear backlash and efficiency vary by model; must characterize for each joint

---

## 2. Microcontrollers

### 2.1 STM32 (ARM Cortex-M)

| Family | Core | Clock | FPU | Best For | Price | Source |
|--------|------|-------|-----|----------|-------|--------|
| STM32F4 (F407, F446) | Cortex-M4 | 168 MHz | Yes | General robotics, motor control | ~$5–15 | ST datasheet |
| STM32F7 (F767, H743) | Cortex-M7 | 216–480 MHz | Yes (DP) | High-performance control, DSP | ~$10–25 | ST datasheet |
| STM32G4 (G474) | Cortex-M4 | 170 MHz | Yes | Motor control (CORDIC, FMAC) | ~$5–10 | ST datasheet |
| STM32H7 (H743, H750) | Cortex-M7 | 480 MHz | Yes (DP) | AI at the edge, complex control | ~$10–20 | ST datasheet |

**Verified Constants:**
- STM32F407: **168 MHz**, **1 MB Flash**, **192 KB RAM**, **DSP + FPU** (source: ST RM0090 reference manual)
- STM32H743: **480 MHz**, **2 MB Flash**, **1 MB RAM**, **DMIPS: 1027** (source: ST RM0433 reference manual)
- ADC sampling rate (STM32F4): **up to 2.4 MSPS** (triple interleaved) (source: ST datasheet)
- Timer resolution: **up to 168 MHz** (source: ST reference manual)

**Feasibility Boundaries:**
- Excellent ecosystem (HAL, LL, FreeRTOS, mbed, Arduino core)
- Hardware timers essential for motor PWM, encoder input capture, comms timing
- F4 is workhorse; G4 is optimized for motor control; H7 is overkill for simple joints but good for central controller
- Debug: ST-Link v2/v3, SWD interface

---

### 2.2 ESP32

| Parameter | ESP32 | ESP32-S3 | ESP32-C3 | Source |
|-----------|-------|----------|----------|--------|
| Core | Xtensa LX6 (dual) | Xtensa LX7 (dual) | RISC-V (single) | Espressif datasheet |
| Clock | 240 MHz | 240 MHz | 160 MHz | Espressif datasheet |
| RAM | 520 KB | 512 KB | 400 KB | Espressif datasheet |
| WiFi | 802.11 b/g/n | 802.11 b/g/n | 802.11 b/g/n | Espressif datasheet |
| Bluetooth | BLE + Classic | BLE 5.0 | BLE 5.0 | Espressif datasheet |
| FPU | Yes | Yes (with vector instructions) | No | Espressif datasheet |
| Price | ~$3–6 | ~$4–8 | ~$2–4 | Distributors |

**Verified Constants:**
- GPIO toggle speed: **~40 MHz** max (source: Espressif docs)
- ADC resolution: **12-bit**, **~2,000 samples/sec** practical (source: Espressif docs, limited by SAR ADC design)
- DAC: **2× 8-bit** (source: Espressif docs)

**Feasibility Boundaries:**
- Built-in wireless → excellent for telemetry, debugging, OTA updates
- ADC is relatively slow and noisy; not ideal for high-speed current sensing
- Great for sensor nodes, distributed I/O, wireless bridges
- Less deterministic than STM32 for hard real-time motor control

---

### 2.3 Teensy

| Parameter | Teensy 4.0 | Teensy 4.1 | Source |
|-----------|------------|------------|--------|
| Core | ARM Cortex-M7 | ARM Cortex-M7 | PJRC specs |
| Clock | 600 MHz | 600 MHz | PJRC specs |
| RAM | 1 MB | 1 MB | PJRC specs |
| Flash | 2 MB | 8 MB | PJRC specs |
| FPU | Yes (DP) | Yes (DP) | PJRC specs |
| Ethernet | No | 10/100 Mbps | PJRC specs |
| USB | USB 2.0 (480 Mbps) | USB 2.0 (480 Mbps) | PJRC specs |
| Price | ~$25 | ~$30 | PJRC store |

**Verified Constants:**
- Teensy 4.1: **600 MHz**, **1 MB RAM**, **8 MB Flash** (source: PJRC website)
- GPIO toggle: **~150 MHz** (source: PJRC benchmarks)
- ADC: **12-bit**, **~1 MSPS** (source: NXP i.MX RT1060 datasheet)

**Feasibility Boundaries:**
- Extremely fast for an MCU; approaches low-end Linux SBC performance
- Arduino-compatible ecosystem with excellent libraries
- No built-in wireless (4.1 has Ethernet; wireless requires add-on)
- Good for central real-time controller, sensor fusion, or high-speed logging

---

### 2.4 Raspberry Pi (Linux SBC)

| Parameter | Pi 4 | Pi 5 | Source |
|-----------|------|------|--------|
| CPU | BCM2711 (4× Cortex-A72 @ 1.5 GHz) | BCM2712 (4× Cortex-A76 @ 2.4 GHz) | Raspberry Pi specs |
| RAM | 1–8 GB | 4–8 GB | Raspberry Pi specs |
| GPU | VideoCore VI | VideoCore VII | Raspberry Pi specs |
| GPIO | 40-pin header | 40-pin header | Raspberry Pi specs |
| USB | 2× USB 3.0 + 2× USB 2.0 | 2× USB 3.0 + 2× USB 2.0 | Raspberry Pi specs |
| Ethernet | Gigabit | Gigabit | Raspberry Pi specs |
| Price | ~$35–75 | ~$60–80 | Raspberry Pi store |

**Feasibility Boundaries:**
- Linux = non-real-time by default; requires PREEMPT_RT patch for soft real-time
- Excellent for high-level planning, vision, networking, logging
- GPIO is slow and non-deterministic compared to bare-metal MCUs
- Best paired with real-time MCUs for motor control (Pi does planning, MCU does FOC)

---

## 3. Communication Protocols

### 3.1 CAN (Controller Area Network)

| Parameter | Value | Source |
|-----------|-------|--------|
| Max data rate | **1 Mbps** (CAN 2.0), **5–8 Mbps** (CAN FD) | ISO 11898 |
| Max bus length @ 1 Mbps | **~40 m** | ISO 11898 |
| Max nodes | **~110** (theoretical), **~30–50** practical | ISO 11898 |
| Message size | **8 bytes** (CAN 2.0), **64 bytes** (CAN FD) | ISO 11898 |
| Topology | Linear bus, terminated at both ends | ISO 11898 |
| Termination resistor | **120 Ω** at each end | ISO 11898 |
| Cable | Twisted pair (CAN_H, CAN_L), shield optional | ISO 11898 |

**Verified Constants:**
- CAN bit timing requires sampling point at **75–87.5%** of bit time (source: ISO 11898, CiA recommendations)
- Standard CAN identifier: **11-bit** (0–2047); Extended: **29-bit** (source: ISO 11898)
- Common robotics frame rate: **100–1000 Hz** per joint on CAN (source: community practice)

**Feasibility Boundaries:**
- Robust, differential signaling → excellent noise immunity
- Bus arbitration handles collisions automatically
- Limited bandwidth: 12 joints × 8-byte frames × 1 kHz = ~768 kbps → approaches 1 Mbps limit with overhead
- CAN FD alleviates bandwidth but requires FD-compatible transceivers and controllers
- Needs termination resistors at both ends; missing termination = signal reflections

**Open Questions:**
1. What is the actual achievable bus utilization before latency becomes unacceptable?
2. How does CAN bus behave under EMI from motor PWM switching?

---

### 3.2 EtherCAT

| Parameter | Value | Source |
|-----------|-------|--------|
| Data rate | **100 Mbps** (Fast Ethernet) | EtherCAT Technology Group |
| Cycle time | **< 1 ms** typical, **< 100 µs** achievable | ETG docs |
| Topology | Line, tree, star, ring (with redundancy) | ETG docs |
| Max nodes | **65,535** (theoretical) | ETG docs |
| Cable | Standard Ethernet (CAT5e/CAT6) | ETG docs |
| Protocol | Master-slave, processed on the fly | ETG docs |

**Verified Constants:**
- EtherCAT frame processing time per slave: **~1 µs** (source: ETG docs, Beckhoff ESC datasheet)
- Distributed clocks (DC) synchronization accuracy: **< 100 ns** (source: ETG docs)
- Typical cycle time for servo drives: **250 µs – 1 ms** (source: industrial practice)

**Feasibility Boundaries:**
- Industrial standard for multi-axis motion control
- Requires EtherCAT master (Linux with IgH EtherCAT, Beckhoff TwinCAT, etc.)
- Slave controllers (ESC) add cost (~$10–50 per node in chip + magnetics)
- Overkill for <6 joints; scales beautifully for >12 joints or sub-millisecond sync
- Redundancy option (ring topology) for safety-critical applications

---

### 3.3 RS-485

| Parameter | Value | Source |
|-----------|-------|--------|
| Max data rate | **10 Mbps** (short distance), **100 kbps** @ 1200 m | TIA-485-A |
| Max distance | **1,200 m** @ 100 kbps | TIA-485-A |
| Max nodes | **32** (unit load), **256** (1/8 unit load transceivers) | TIA-485-A |
| Topology | Daisy chain (bus) | TIA-485-A |
| Termination | **120 Ω** at both ends | TIA-485-A |
| Biasing | Required for idle state stability | TIA-485-A |
| Cable | Twisted pair, shield recommended | TIA-485-A |

**Verified Constants:**
- Differential voltage: **±1.5 V** min (source: TIA-485-A)
- Receiver input sensitivity: **±200 mV** (source: TIA-485-A)
- Common-mode voltage range: **–7 V to +12 V** (source: TIA-485-A)

**Feasibility Boundaries:**
- Simple, cheap, robust for multi-drop networks
- Half-duplex requires direction control (TX enable pin)
- Slower than CAN for robotics; often used for sensor networks or legacy motor drives
- Modbus-RTU is common higher-layer protocol over RS-485

---

### 3.4 SPI (Serial Peripheral Interface)

| Parameter | Value | Source |
|-----------|-------|--------|
| Max clock | **10–50 MHz** (MCU-dependent) | MCU datasheets |
| Data width | **8-bit** typical, configurable | SPI spec |
| Mode | CPOL, CPHA (4 combinations) | SPI spec |
| Topology | Point-to-point or multi-slave (with chip select) | SPI spec |
| Lines | MOSI, MISO, SCLK, CS (per slave) | SPI spec |
| Max distance | **~0.3 m** on PCB, **~1–2 m** with care | Practice |

**Verified Constants:**
- SPI is **full-duplex**: data flows both ways simultaneously (source: Motorola SPI spec)
- Chip select (CS) must be asserted for entire transaction (source: SPI spec)

**Feasibility Boundaries:**
- Very fast, simple protocol
- Not suited for long distances or multi-master
- Chip select lines proliferate with many slaves
- Often used for: encoders (AS5048A), IMUs (SPI variant), displays, flash memory

---

### 3.5 I2C

| Parameter | Value | Source |
|-----------|-------|--------|
| Standard mode | **100 kbps** | NXP I2C spec |
| Fast mode | **400 kbps** | NXP I2C spec |
| Fast mode plus | **1 Mbps** | NXP I2C spec |
| High speed mode | **3.4 Mbps** | NXP I2C spec |
| Max bus capacitance | **400 pF** | NXP I2C spec |
| Max distance | **~1 m** at 400 kHz | Practice |
| Pull-up resistors | **1–10 kΩ** (depends on bus capacitance, speed) | NXP I2C spec |
| Topology | Multi-master, multi-slave (7-bit or 10-bit address) | NXP I2C spec |

**Verified Constants:**
- 7-bit address space: **112 usable addresses** (16 reserved) (source: NXP I2C spec)
- Clock stretching: slave can hold SCL low to slow master (source: NXP I2C spec)

**Feasibility Boundaries:**
- Simple 2-wire bus (SDA, SCL)
- Slower than SPI; susceptible to bus capacitance issues
- Good for: temperature sensors, EEPROMs, some IMUs, LED drivers
- Not recommended for motor encoders or high-bandwidth sensors

---

## 4. Power Electronics

### 4.1 DC-DC Converters

| Topology | Vin | Vout | Iout | Efficiency | Best For | Source |
|----------|-----|------|------|------------|----------|--------|
| Buck (step-down) | 48 V | 12 V | 10 A | **92–96%** | Main logic power | Datasheets |
| Buck | 48 V | 5 V | 5 A | **90–95%** | Peripherals, sensors | Datasheets |
| Buck | 12 V | 3.3 V | 2 A | **85–92%** | MCU, digital logic | Datasheets |
| Buck-Boost | 6–36 V | 12 V | 5 A | **85–93%** | Battery-powered, varying Vin | Datasheets |
| Isolated flyback | 48 V | 12 V | 2 A | **80–88%** | Isolated gate drives, safety | Datasheets |

**Verified Constants:**
- Switching frequency for buck converters: **100 kHz – 2 MHz** (source: TI, Analog Devices app notes)
- Inductor ripple current: typically **20–40%** of Iout (source: power electronics design guides)
- Output capacitor ESR dominates ripple at high frequencies (source: power electronics theory)

**Feasibility Boundaries:**
- Switching noise can couple into analog sensors; layout and filtering critical
- Higher frequency = smaller magnetics but higher switching losses
- Synchronous rectification (MOSFET instead of diode) improves efficiency 3–8%
- Hot-swap controllers needed for safe battery connection

---

### 4.2 Battery Systems

| Chemistry | Nominal V | Energy Density | Cycle Life | Max C-Rate | Cost ($/kWh) | Source |
|-----------|-----------|----------------|------------|------------|--------------|--------|
| LiPo (LCO/NMC) | 3.7 V/cell | **150–250 Wh/kg** | 300–500 | 5–30C | ~$150–300 | Datasheets |
| LiFePO4 | 3.2 V/cell | **90–160 Wh/kg** | 2,000–5,000 | 1–3C | ~$100–200 | Datasheets |
| Li-ion 18650 (NMC) | 3.6 V/cell | **200–250 Wh/kg** | 500–1,000 | 2–5C | ~$150–250 | Datasheets |
| Solid-state (emerging) | 3.5–4.0 V/cell | **300–500 Wh/kg** | 1,000+ | 1–5C | ~$500–1,000+ | Literature |

**Verified Constants:**
- LiPo cell voltage range: **3.0 V (empty) to 4.2 V (full)** (source: battery datasheets)
- LiFePO4 cell voltage range: **2.5 V (empty) to 3.65 V (full)** (source: battery datasheets)
- 6S LiPo pack: **22.2 V nominal**, **25.2 V full**, **18.0 V empty** (source: calculated)
- 12S LiPo pack: **44.4 V nominal**, **50.4 V full**, **36.0 V empty** (source: calculated)
- C-rate: 1C = capacity in A; 10C on 10 Ah battery = 100 A (source: battery convention)

**Feasibility Boundaries:**
- LiPo: high energy density, high discharge rate, but fire risk if damaged or overcharged
- LiFePO4: safer, longer life, lower energy density → heavier for same capacity
- BMS (battery management system) is mandatory: cell balancing, overvoltage, undervoltage, overcurrent, temperature
- Humanoid power budget: ~1–2 kW peak → 6S or 12S LiPo with 10–20 Ah typical
- Regenerative braking: battery must accept charge current; some BMS designs block charging during discharge

**Open Questions:**
1. What is the actual internal resistance and voltage sag of our chosen battery under 50A pulses?
2. How does battery temperature affect peak power output during high-torque maneuvers?

---

### 4.3 EMI / EMC Considerations

| Issue | Mitigation | Source |
|-------|------------|--------|
| Motor PWM switching noise | Shielded motor cables, ferrite cores, proper grounding | EMC design guides |
| CAN bus reflections | Proper termination, twisted pair, stub length < 0.3 m | ISO 11898 |
| Power supply ripple | Bulk capacitance, ceramic decoupling, LC filters | Power electronics practice |
| Radiated emissions | Shielded enclosures, gaskets, cable shielding | FCC/CE compliance guides |
| Ground loops | Single-point ground, star grounding, isolation | EMC design guides |

**Verified Constants:**
- Motor cable shield: **360° termination** at drive end, optional at motor end (source: drive manufacturer manuals)
- Decoupling capacitor placement: **< 10 mm** from IC power pins (source: high-speed PCB design guides)
- Switching frequency harmonics: significant EMI at **n × fsw** (source: EMC theory)

---

## 5. PCB Design Rules

### 5.1 General Guidelines

| Parameter | Value | Source |
|-----------|-------|--------|
| Trace width (1 oz copper, 10 °C rise) | **0.3 mm per amp** (external), **0.6 mm per amp** (internal) | IPC-2221 |
| Trace width (2 oz copper) | **0.15 mm per amp** (external) | IPC-2221 |
| Minimum trace/space (standard PCB) | **0.15 mm / 0.15 mm** (6/6 mil) | PCB fab houses |
| Minimum trace/space (advanced) | **0.075 mm / 0.075 mm** (3/3 mil) | Advanced PCB fabs |
| Via diameter | **0.3–0.5 mm** drill, **0.6–1.0 mm** pad | PCB design guides |
| Via current capacity | **~1 A** per standard via (0.3 mm drill, 1 oz) | IPC-2221 approx |
| Clearance (high voltage) | **1 mm per 500 V** (general guideline) | IPC-2221 |
| Component keepout from edge | **2–3 mm** | PCB assembly guidelines |

**Verified Constants:**
- Copper resistivity: **1.68 × 10⁻⁸ Ω·m** (source: NIST)
- 1 oz copper thickness: **35 µm** (source: IPC-4562)
- 2 oz copper thickness: **70 µm** (source: IPC-4562)
- FR-4 Tg: **130–180 °C** (standard to high-Tg) (source: IPC-4101)

### 5.2 High-Current Design

| Parameter | Guideline | Source |
|-----------|-----------|--------|
| Motor phase current traces | **2 oz copper minimum**, **≥2 mm per amp** | Power electronics practice |
| Via stitching | **multiple vias** for high-current paths (reduce inductance) | PCB design guides |
| Plane layers | **dedicated power and ground planes** for noise reduction | High-speed design guides |
| Current sense resistors | **Kelvin connection** (4-wire sensing) | Precision measurement practice |
| Gate drive loops | **minimize loop area** to reduce inductance | MOSFET driver app notes |

### 5.3 Thermal Management

| Parameter | Value | Source |
|-----------|-------|--------|
| Thermal vias under QFN/DFN | **0.3 mm drill, 1.0 mm pitch**, filled or tented | PCB design guides |
| Copper pour for heat spreading | **as large as practical**, multiple vias to inner planes | Thermal design guides |
| Thermal interface material (TIM) | **0.5–3.0 W/(m·K)** for gap pads | TIM datasheets |
| Junction-to-ambient (θJA) | **20–60 °C/W** (no heatsink), **2–10 °C/W** (with heatsink) | IC datasheets |
| Max junction temperature (MOSFET) | **150–175 °C** | MOSFET datasheets |
| Max junction temperature (IC) | **125–150 °C** | IC datasheets |

**Verified Constants:**
- MOSFET Rds(on) increases with temperature: **~0.4–0.7% per °C** (source: MOSFET datasheets)
- Conduction losses: P = I² × Rds(on) (source: power electronics theory)
- Switching losses: Psw = 0.5 × V × I × (tr + tf) × fsw (source: power electronics theory)

---

## Verified Data Summary

| Data Point | Source | Confidence |
|------------|--------|------------|
| ODrive specs | ODrive documentation, GitHub | High |
| SimpleFOC capabilities | SimpleFOC docs, GitHub | High |
| Elmo/Ingenia/Copley specs | Manufacturer datasheets | High |
| T-Motor/Unitree/MyActuator specs | Datasheets | High |
| STM32 family specs | ST reference manuals, datasheets | High |
| ESP32 specs | Espressif datasheets | High |
| Teensy specs | PJRC website, NXP datasheet | High |
| CAN / CAN FD specs | ISO 11898 | High |
| EtherCAT specs | ETG documentation | High |
| RS-485 specs | TIA-485-A | High |
| SPI / I2C specs | Motorola/NXP specifications | High |
| Battery chemistry specs | Manufacturer datasheets | High |
| PCB design rules | IPC-2221, IPC-4101 | High |
| Power electronics formulas | Standard theory | High |
| EMI/EMC guidelines | Industry practice, FCC/CE docs | Medium |
| Integration performance (latency, jitter) | Community reports, not yet measured | Low-Medium |

---

## Open Questions / Learning Targets

1. What is the end-to-end control latency (sensor → MCU → driver → motor) for our chosen stack?
2. How does CAN bus latency scale with 12+ joints + sensors + compute node?
3. What is the EMI signature of our motor PWM + cabling, and how does it affect IMU accuracy?
4. Can we achieve <1 ms cycle time for whole-body control with EtherCAT + STM32 + ODrive?
5. What is the thermal derating curve for our motor drivers under continuous humanoid walking loads?
6. How do we implement safe torque off (STO) for emergency stop compliance?
7. What is the optimal power distribution architecture (48V bus vs. 24V bus vs. dual bus)?
8. Can we design a custom motor driver PCB that matches ODrive performance at lower cost?
9. What is the battery voltage sag under peak humanoid power demands, and how does it affect motor torque?
10. How do we handle regenerative braking energy without overvoltage shutdown?

---

*Last verified: 2026-06-08. Next review: after first electronics bring-up.*
