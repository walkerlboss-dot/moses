# materials-selection.md — Moses Knowledge Base

> **Domain:** Materials Science for Humanoid Robotics
> **Status:** Seed document — will grow with testing and characterization
> **Last Updated:** 2026-06-08
> **Confidence:** Medium — mechanical properties from standards and datasheets; selection heuristics from literature and practice

---

## 1. Aluminum Alloys

### 1.1 6061-T6

**Composition:** Al-Mg-Si alloy, solution heat-treated and artificially aged (T6 temper).

| Property | Value | Source |
|----------|-------|--------|
| Density | **2,700 kg/m³** | ASTM B221 |
| Young's modulus (E) | **68.9 GPa** | ASM Handbook Vol. 2 |
| Yield strength (σy) | **276 MPa** | ASTM B221 |
| Ultimate tensile strength (σuts) | **310 MPa** | ASTM B221 |
| Elongation at break | **12%** | ASTM B221 |
| Hardness (Brinell) | **95 HB** | ASM Handbook |
| Thermal conductivity | **167 W/(m·K)** | ASM Handbook |
| CTE (20–100 °C) | **23.6 × 10⁻⁶ /°C** | ASM Handbook |
| Specific stiffness (E/ρ) | **25.5 × 10⁶ m²/s²** | Calculated |
| Specific strength (σy/ρ) | **102 × 10³ m²/s²** | Calculated |
| Fatigue strength (10⁷ cycles, R=-1) | **96 MPa** | ASM Handbook |
| Melting range | **582–652 °C** | ASM Handbook |
| Machinability rating | **Good** (90% relative to 2011) | Machinability databases |
| Weldability | **Good** (TIG, MIG) | AWS D1.2 |
| Anodizing | **Excellent** | Industry practice |
| Cost (sheet, 6 mm) | ~$4–8/kg | Market (volatile) |

**Best For:** General structural, frames, brackets, links, heat sinks, anodized cosmetic parts.

**Feasibility Boundaries:**
- Not as strong as 7075; for high-stress joints, 7075 or steel may be needed
- T6 temper loses strength in weld HAZ; weld zones effectively revert to T0 (~125 MPa yield)
- Stress corrosion cracking possible in some tempers/environments; T6 is generally resistant

---

### 1.2 7075-T6 / 7075-T651

**Composition:** Al-Zn-Mg-Cu alloy, highest strength of common aluminum alloys.

| Property | Value | Source |
|----------|-------|--------|
| Density | **2,810 kg/m³** | ASTM B209 |
| Young's modulus (E) | **71.7 GPa** | ASM Handbook Vol. 2 |
| Yield strength (σy) | **503 MPa** | ASTM B209 |
| Ultimate tensile strength (σuts) | **572 MPa** | ASTM B209 |
| Elongation at break | **11%** | ASTM B209 |
| Hardness (Brinell) | **150 HB** | ASM Handbook |
| Thermal conductivity | **130 W/(m·K)** | ASM Handbook |
| CTE (20–100 °C) | **23.2 × 10⁻⁶ /°C** | ASM Handbook |
| Specific stiffness (E/ρ) | **25.5 × 10⁶ m²/s²** | Calculated |
| Specific strength (σy/ρ) | **179 × 10³ m²/s²** | Calculated |
| Fatigue strength (10⁷ cycles, R=-1) | **159 MPa** | ASM Handbook |
| Melting range | **477–635 °C** | ASM Handbook |
| Machinability rating | **Good** | Machinability databases |
| Weldability | **Poor** (susceptible to cracking) | AWS D1.2 |
| Anodizing | **Fair** (not as uniform as 6061) | Industry practice |
| Cost (sheet, 6 mm) | ~$8–15/kg | Market (volatile) |

**Best For:** High-stress joints, bearing housings, gear carriers, anywhere 6061 would be marginal.

**Feasibility Boundaries:**
- Poor weldability; design for bolting or adhesive bonding
- Higher susceptibility to stress corrosion cracking than 6061; T73 temper improves SCC resistance at slight strength cost
- More expensive and harder to machine than 6061

---

### 1.3 5052-H32

**Composition:** Al-Mg alloy, strain-hardened and stabilized (H32 temper).

| Property | Value | Source |
|----------|-------|--------|
| Density | **2,680 kg/m³** | ASTM B209 |
| Yield strength (σy) | **193 MPa** | ASTM B209 |
| Ultimate tensile strength (σuts) | **228 MPa** | ASTM B209 |
| Elongation at break | **12%** | ASTM B209 |
| Specific stiffness (E/ρ) | **25.5 × 10⁶ m²/s²** | Calculated |
| Specific strength (σy/ρ) | **72 × 10³ m²/s²** | Calculated |

**Best For:** Sheet metal parts, enclosures, bent brackets (excellent formability), marine/corrosive environments.

---

### 1.4 2024-T3 / T351

**Composition:** Al-Cu-Mg alloy, high strength with good fatigue resistance.

| Property | Value | Source |
|----------|-------|--------|
| Density | **2,780 kg/m³** | ASTM B209 |
| Yield strength (σy) | **345 MPa** | ASTM B209 |
| Ultimate tensile strength (σuts) | **483 MPa** | ASTM B209 |
| Fatigue strength (10⁷ cycles) | **138 MPa** | ASM Handbook |

**Best For:** Aerospace-style structures, highly loaded tension members. Less common in robotics than 6061/7075.

---

## 2. Steel Alloys

### 2.1 Mild Steel / Low-Carbon Steel (AISI 1018, A36)

| Property | Value | Source |
|----------|-------|--------|
| Density | **7,850 kg/m³** | ASM Handbook |
| Young's modulus (E) | **200 GPa** | ASM Handbook |
| Yield strength (σy) | **250 MPa** (A36), **370 MPa** (1018 cold-drawn) | ASTM A36, AISI datasheets |
| Ultimate tensile strength (σuts) | **400–550 MPa** | ASTM A36 |
| Elongation at break | **20–25%** | ASTM A36 |
| Specific stiffness (E/ρ) | **25.5 × 10⁶ m²/s²** | Calculated |
| Specific strength (σy/ρ) | **32–47 × 10³ m²/s²** | Calculated |
| Thermal conductivity | **50 W/(m·K)** | ASM Handbook |
| CTE | **12 × 10⁻⁶ /°C** | ASM Handbook |
| Cost | ~$0.50–1.50/kg | Market (volatile) |

**Best For:** Low-cost structural, test fixtures, bases, anything where weight is not critical.

**Feasibility Boundaries:**
- 3× denser than aluminum → heavy; avoid in moving links
- Prone to rust; needs painting, plating, or coating
- Excellent weldability

---

### 2.2 Alloy Steel (4140, Pre-Hard 4140-HT)

| Property | Value | Source |
|----------|-------|--------|
| Density | **7,850 kg/m³** | ASM Handbook |
| Yield strength (σy), annealed | **460 MPa** | ASM Handbook |
| Yield strength (σy), HT (Rc 28–32) | **750–1,000 MPa** | ASM Handbook |
| Ultimate tensile strength (σuts), HT | **1,000–1,200 MPa** | ASM Handbook |
| Hardness, HT | **Rc 28–32** | ASM Handbook |
| Specific strength (σy/ρ), HT | **96–127 × 10³ m²/s²** | Calculated |

**Best For:** Shafts, gears, high-stress pins, bearing races, anything needing wear resistance.

**Feasibility Boundaries:**
- Requires heat treatment to achieve full properties
- HT condition is harder to machine; may need grinding for precision features
- Can be nitrided for extreme surface hardness

---

### 2.3 Stainless Steel (304, 316, 17-4 PH)

| Alloy | Yield Strength | UTS | Key Property | Best For |
|-------|---------------|-----|--------------|----------|
| 304 (A2-70) | **215 MPa** | **505 MPa** | Corrosion resistance, non-magnetic | Fasteners, housings, food/medical adjacent |
| 316 (A4-80) | **220 MPa** | **520 MPa** | Superior corrosion resistance (marine) | Marine, chemical exposure |
| 17-4 PH (H900) | **1,170 MPa** | **1,310 MPa** | High strength + corrosion resistance | High-stress corrosion-resistant parts |

**Verified Constants:**
- 304 stainless magnetic permeability: **~1.02** (nearly non-magnetic) (source: ASM Handbook)
- 17-4 PH can be precipitation hardened to **Rc 38–44** (source: ASM Handbook)

**Feasibility Boundaries:**
- 304/316: lower strength than carbon steels; work-hardens rapidly during machining
- 17-4 PH: expensive, requires heat treatment, but unique combination of strength and corrosion resistance
- All stainless: galling risk when self-mated; use anti-seize or different alloy pairs

---

### 2.4 Tool Steel (O1, A2, D2)

| Alloy | Hardness (Rc) | Best For |
|-------|---------------|----------|
| O1 (oil-hardening) | **Rc 60–64** | Punches, dies, knives |
| A2 (air-hardening) | **Rc 57–62** | Cold work tools, gauges |
| D2 (high-carbon, high-chromium) | **Rc 58–62** | Wear-resistant tools, forming dies |

**Best For:** Custom tooling, fixtures, wear plates, cutting tools for robot maintenance.

---

## 3. Titanium Alloys

### 3.1 Ti-6Al-4V (Grade 5)

**The workhorse titanium alloy for aerospace and high-performance applications.**

| Property | Value | Source |
|----------|-------|--------|
| Density | **4,430 kg/m³** | ASTM B265 |
| Young's modulus (E) | **113.8 GPa** | ASM Handbook |
| Yield strength (σy), annealed | **880 MPa** | ASTM B265 |
| Ultimate tensile strength (σuts) | **950 MPa** | ASTM B265 |
| Elongation at break | **14%** | ASTM B265 |
| Specific stiffness (E/ρ) | **25.7 × 10⁶ m²/s²** | Calculated |
| Specific strength (σy/ρ) | **199 × 10³ m²/s²** | Calculated |
| Fatigue strength (10⁷ cycles) | **510 MPa** (Kt=1) | ASM Handbook |
| Hardness | **Rc 36** | ASM Handbook |
| Thermal conductivity | **6.7 W/(m·K)** | ASM Handbook |
| CTE | **8.6 × 10⁻⁶ /°C** | ASM Handbook |
| Melting range | **1,600–1,660 °C** | ASM Handbook |
| Machinability | **Poor** (30% of steel) | Machinability databases |
| Weldability | **Good** (inert atmosphere required) | AWS D1.9 |
| Biocompatibility | **Excellent** | ISO 5832-3 |
| Cost | ~$30–80/kg (mill products) | Market (volatile) |

**Best For:** High-performance robot links where weight and strength are critical, prosthetic-inspired designs, anything where specific strength is the dominant figure of merit.

**Feasibility Boundaries:**
- Expensive and difficult to machine (low thermal conductivity causes heat buildup)
- Galling when self-mated; use coatings or dissimilar materials
- Requires inert gas shielding for welding
- Not cost-effective for most first-build robotics; consider for optimization phase

---

## 4. Plastics and Polymers

### 4.1 PLA (Polylactic Acid)

| Property | Value | Source |
|----------|-------|--------|
| Density | **1,240 kg/m³** | Material datasheets |
| Tensile strength | **30–60 MPa** | ASTM D638 |
| Young's modulus | **2.5–3.5 GPa** | ASTM D638 |
| Elongation at break | **2–10%** | ASTM D638 |
| Glass transition (Tg) | **55–60 °C** | DSC measurements |
| Melting point (Tm) | **150–160 °C** | DSC measurements |
| Thermal conductivity | **0.13 W/(m·K)** | Literature |
| CTE | **~70 × 10⁻⁶ /°C** | Literature |
| Biodegradable | **Yes** (industrial composting) | ISO 17088 |
| Cost | ~$20–30/kg (filament) | Market |

**Best For:** Prototypes, jigs, fixtures, non-structural parts, visual models.

**Feasibility Boundaries:**
- Low Tg → deforms above 55 °C (e.g., in direct sunlight, near motors)
- Brittle; poor impact resistance
- Creeps under sustained load
- Not suitable for structural robot parts

---

### 4.2 PETG (Polyethylene Terephthalate Glycol)

| Property | Value | Source |
|----------|-------|--------|
| Density | **1,270 kg/m³** | Material datasheets |
| Tensile strength | **50–75 MPa** | ASTM D638 |
| Young's modulus | **2.0–2.5 GPa** | ASTM D638 |
| Elongation at break | **20–50%** | ASTM D638 |
| Glass transition (Tg) | **75–80 °C** | DSC measurements |
| Impact strength (Izod) | **1.5–3.0 kJ/m²** | ASTM D256 |
| Cost | ~$25–35/kg (filament) | Market |

**Best For:** Structural brackets, housings, protective covers, anything needing more toughness than PLA.

**Feasibility Boundaries:**
- Hygroscopic (absorbs moisture); must be dried before printing for best results
- More flexible than PLA/ABS; may not be stiff enough for long spans
- Good chemical resistance

---

### 4.3 ABS (Acrylonitrile Butadiene Styrene)

| Property | Value | Source |
|----------|-------|--------|
| Density | **1,040 kg/m³** | Material datasheets |
| Tensile strength | **30–50 MPa** | ASTM D638 |
| Young's modulus | **1.8–2.5 GPa** | ASTM D638 |
| Elongation at break | **10–50%** | ASTM D638 |
| Glass transition (Tg) | **105 °C** | DSC measurements |
| Impact strength (Izod) | **2.0–4.0 kJ/m²** | ASTM D256 |
| Cost | ~$20–30/kg (filament) | Market |

**Best For:** Impact-resistant parts, enclosures, parts needing higher temperature resistance than PLA.

**Feasibility Boundaries:**
- Requires heated bed and enclosed printer to prevent warping
- Emits styrene fumes; needs ventilation
- Less stiff than PLA or PETG

---

### 4.4 Nylon (PA6, PA66, PA12)

| Property | PA6 | PA66 | PA12 | Source |
|----------|-----|------|------|--------|
| Density | 1,130 kg/m³ | 1,140 kg/m³ | 1,010 kg/m³ | Material datasheets |
| Tensile strength | 60–85 MPa | 70–85 MPa | 45–55 MPa | ASTM D638 |
| Young's modulus | 1.5–2.5 GPa | 2.0–3.0 GPa | 1.2–1.8 GPa | ASTM D638 |
| Elongation at break | 30–100% | 20–80% | 100–300% | ASTM D638 |
| Tg | 50–60 °C | 70–80 °C | 50–55 °C | DSC measurements |
| Tm | 215–225 °C | 255–265 °C | 175–180 °C | DSC measurements |
| Impact strength (Izod) | 3–8 kJ/m² | 3–6 kJ/m² | 10–20 kJ/m² | ASTM D256 |
| Moisture absorption | 2.5–3.0% | 2.5–3.0% | 0.5–1.5% | ISO 62 |
| Cost (filament) | ~$40–60/kg | ~$50–80/kg | ~$60–100/kg | Market |

**Best For:** Gears, bearings, bushings, tough structural parts, living hinges.

**Feasibility Boundaries:**
- Hygroscopic (especially PA6/PA66); moisture affects dimensions and properties
- PA12 absorbs less moisture → better dimensional stability
- Can be annealed to increase crystallinity and stiffness
- Excellent wear resistance and low coefficient of friction

---

### 4.5 PEEK (Polyether Ether Ketone)

**High-performance engineering thermoplastic.**

| Property | Value | Source |
|----------|-------|--------|
| Density | **1,320 kg/m³** | Material datasheets |
| Tensile strength | **90–100 MPa** | ASTM D638 |
| Young's modulus | **3.6 GPa** | ASTM D638 |
| Elongation at break | **30–50%** | ASTM D638 |
| Tg | **143 °C** | DSC measurements |
| Tm | **343 °C** | DSC measurements |
| Continuous use temperature | **250–260 °C** | UL 746B |
| Thermal conductivity | **0.25 W/(m·K)** | Literature |
| Cost | ~$300–600/kg (filament) | Market |

**Best For:** High-temperature bearings, insulators, aerospace-grade prototypes.

**Feasibility Boundaries:**
- Very expensive; requires high-temp printer (heated chamber to 150+ °C)
- Difficult to print; warping and poor bed adhesion
- Outstanding chemical resistance and mechanical properties at temperature

---

## 5. Composites

### 5.1 Carbon Fiber / Epoxy (see also manufacturing-methods.md)

| Property | Unidirectional | Quasi-Isotropic [0/±45/90]s | Source |
|----------|---------------|------------------------------|--------|
| Density | 1,500–1,600 kg/m³ | 1,500–1,600 kg/m³ | Calculated |
| Tensile strength (0°) | 1,500–2,500 MPa | 600–800 MPa | Laminate theory, datasheets |
| Tensile modulus (0°) | 150–300 GPa | 70–80 GPa | Laminate theory, datasheets |
| Compressive strength (0°) | 1,200–1,500 MPa | 500–600 MPa | Laminate theory |
| Interlaminar shear strength | 80–120 MPa | 80–120 MPa | ASTM D2344 |
| CTE (0°) | ~0 (slightly negative) | ~2 × 10⁻⁶ /°C | Literature |
| Cost (prepreg) | ~$50–150/kg | ~$50–150/kg | Market |

**Best For:** High-performance robot links, tubes, anything where specific stiffness and specific strength dominate.

**Feasibility Boundaries:**
- Anisotropic; must design with fiber direction in mind
- Brittle; impact damage causes delamination
- Conductive; galvanic corrosion when contacting aluminum (use fiberglass isolators)
- Difficult to repair; damage often requires replacement

---

### 5.2 Fiberglass / Epoxy (GFRP)

| Property | Value | Source |
|----------|-------|--------|
| Density | **1,800–2,000 kg/m³** | Material datasheets |
| Tensile strength | **300–600 MPa** | ASTM D3039 |
| Tensile modulus | **20–45 GPa** | ASTM D3039 |
| Specific stiffness | **11–25 × 10⁶ m²/s²** | Calculated |
| Cost | ~$5–20/kg (raw) | Market |

**Best For:** Electrical isolation between CF and aluminum, low-cost composite prototypes, non-conductive structures.

---

## 6. Bearing Materials

### 6.1 Bearing Steels

| Material | Hardness (Rc) | Application | Source |
|----------|---------------|-------------|--------|
| 52100 (AISI) | **60–65** | Ball and roller bearings | ASTM A295 |
| 440C stainless | **58–60** | Corrosion-resistant bearings | ASTM A756 |
| M50 (tool steel) | **62–65** | High-temp aerospace bearings | AMS 6491 |

**Verified Constants:**
- 52100 minimum hardness for bearings: **Rc 58** (source: SKF/NSK specifications)

---

### 6.2 Self-Lubricating Bearings

| Material | Max Load (MPa) | Max Speed (m/s) | Max Temp (°C) | Best For |
|----------|---------------|-----------------|---------------|----------|
| Oil-impregnated bronze (SAE 841) | 20–35 | 2.5 | 80–150 | Sintered bronze bushings |
| PTFE-lined steel (DU bushing) | 140 | 2.5 | -200 to +280 | Maintenance-free, dry running |
| IGUS iglide (various polymers) | 20–80 | 0.5–1.5 | -40 to +250 | Plastic bushings, light duty |
| Graphite-impregnated bronze | 70 | 1.0 | 350 | High-temp, no lubrication |

**Verified Constants:**
- SAE 841 bronze: porosity **18–25%** by volume, oil content **18–25%** (source: ASTM B438)
- IGUS iglide J: max PV value **0.34 MPa·m/s** (source: IGUS datasheet)

---

### 6.3 Ceramic Bearings (Si₃N₄)

| Property | Value | Source |
|----------|-------|--------|
| Density | **3,200 kg/m³** | Material datasheets |
| Young's modulus | **310 GPa** | Literature |
| Hardness (Vickers) | **1,500–1,700 HV** | Literature |
| Compressive strength | **3,000 MPa** | Literature |
| Thermal conductivity | **30 W/(m·K)** | Literature |
| CTE | **3.2 × 10⁻⁶ /°C** | Literature |
| Cost (hybrid bearing) | ~5–20× steel bearing | Market |

**Best For:** High-speed spindles, corrosive environments, electric motors (non-conductive, no arcing).

---

## 7. Fastener Materials

| Material | Strength Class | Yield Strength | UTS | Best For | Source |
|----------|---------------|---------------|-----|----------|--------|
| Carbon steel (plain) | 4.8 | **320 MPa** | **400 MPa** | Low-stress, indoor | ISO 898-1 |
| Carbon steel (zinc plated) | 8.8 | **640 MPa** | **800 MPa** | General structural | ISO 898-1 |
| Carbon steel (black oxide) | 10.9 | **900 MPa** | **1,000 MPa** | High-stress | ISO 898-1 |
| Carbon steel (alloy) | 12.9 | **1,080 MPa** | **1,200 MPa** | Critical joints | ISO 898-1 |
| Stainless steel (A2-70) | 70 (property class) | **450 MPa** | **700 MPa** | Corrosion resistance | ISO 3506-1 |
| Stainless steel (A4-80) | 80 (property class) | **600 MPa** | **800 MPa** | Marine/chemical | ISO 3506-1 |
| Titanium (Grade 5) | — | **880 MPa** | **950 MPa** | Weight-critical | ASTM F468 |

**Verified Constants:**
- Torque coefficient K for steel bolts: **0.20 (dry)**, **0.15 (lubricated)** (source: Shigley's Mechanical Engineering Design)
- Typical preload for reusable steel joints: **75% of proof load** (source: Shigley's)
- Proof load for 8.8 M6 bolt: **~12.1 kN**; preload: **~9.1 kN**; torque (dry, K=0.20): **~10.9 N·m** (source: calculated from ISO 898-1)

---

## 8. Material Selection Heuristics for Humanoid Robotics

### 8.1 Selection Matrix by Component

| Component | Primary Requirement | Recommended Material(s) | Alternative |
|-----------|--------------------|------------------------|-------------|
| Main frame / torso | Stiffness, machinability | 6061-T6 aluminum | 5052-H32 (sheet) |
| Thigh / shin links | Specific stiffness, strength | CF/epoxy tube, 7075-T6 | 6061-T6 (heavier) |
| Joint housings | Strength, fatigue, bearing seats | 7075-T6, 6061-T6 | Steel (4140-HT for high load) |
| Bearing shafts | Hardness, wear, fatigue | 4140-HT, 17-4 PH | 52100 (bearing steel) |
| Gears | Wear, fatigue, noise | 4140-HT (hardened), brass (pinion) | POM/nylon (light duty) |
| Foot sole | Impact, grip, wear | Urethane rubber, TPU | Sorbothane (damping) |
| Cable guides / guards | Flexibility, abrasion | Nylon (PA12), PETG | PTFE (low friction) |
| Sensor mounts | Vibration damping, light | 6061-T6, PETG, CF-PA | Sorbothane isolators |
| Battery tray | Fire resistance, strength | Steel, aluminum | FR-4 (PCB material) |
| Covers / cosmetics | Light, finish | ABS, ASA, PC | CF-PA (structural cosmetic) |

### 8.2 Figures of Merit

| Figure of Merit | Formula | Best Material | Notes |
|-----------------|---------|---------------|-------|
| Specific stiffness | E / ρ | CF/epoxy (quasi-iso: ~45 × 10⁶), Ti-6Al-4V (~25.7 × 10⁶), Al 6061 (~25.5 × 10⁶) | Stiffness per unit weight |
| Specific strength | σy / ρ | CF/epoxy (UD: ~1,000 × 10³), Ti-6Al-4V (~199 × 10³), Al 7075 (~179 × 10³) | Strength per unit weight |
| Specific toughness | KIC / ρ | Steel alloys, titanium | Fracture resistance per weight |
| Thermal conductivity / density | k / ρ | Aluminum alloys | Heat dissipation per weight |

---

## Verified Data Summary

| Data Point | Source | Confidence |
|------------|--------|------------|
| Al 6061-T6 mechanical properties | ASTM B221, ASM Handbook | High |
| Al 7075-T6 mechanical properties | ASTM B209, ASM Handbook | High |
| Steel 1018, A36, 4140 properties | ASTM A36, AISI, ASM Handbook | High |
| Stainless 304, 316, 17-4 PH properties | ASTM A240, ASM Handbook | High |
| Ti-6Al-4V properties | ASTM B265, ASM Handbook | High |
| PLA/PETG/ABS/nylon properties | ASTM D638, material datasheets | High |
| PEEK properties | ASTM D638, Victrex datasheet | High |
| CF/epoxy properties | Toray datasheets, laminate theory | High |
| Bearing steel properties | ASTM A295, SKF/NSK specs | High |
| Fastener strength classes | ISO 898-1, ISO 3506-1 | High |
| Self-lubricating bearing specs | SAE 841, IGUS datasheets | High |
| Ceramic bearing (Si₃N₄) properties | Literature, manufacturer specs | Medium |
| Cost estimates | Market prices (volatile) | Low-Medium |

---

## Open Questions / Learning Targets

1. What is the fatigue life of 3D-printed CF-PA (Markforged, Anisoprint) under robot joint cyclic loading?
2. How does moisture absorption affect PA6/PA66 gear dimensions and backlash over time?
3. What is the optimal material for a custom series elastic actuator spring (music wire vs. titanium vs. CF)?
4. Can we characterize the impact resistance of printed vs. machined vs. molded nylon gears?
5. What is the galvanic corrosion rate between CF and 6061-T6 in a humid environment?
6. How does annealing (salt, oven) affect the crystallinity and mechanical properties of printed PA6/PA66?
7. What is the wear rate of IGUS bushings vs. ball bearings in a humanoid hip joint under realistic loads?
8. Can we establish a standard material palette (3–5 materials) to simplify BOM and sourcing?

---

*Last verified: 2026-06-08. Next review: after material testing campaign.*
