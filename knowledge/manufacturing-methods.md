# manufacturing-methods.md — Moses Knowledge Base

> **Domain:** Manufacturing & Fabrication Methods for Humanoid Robotics
> **Status:** Seed document — will grow with build experience
> **Last Updated:** 2026-06-08
> **Confidence:** Medium — constants from industry standards and vendor datasheets; process parameters from literature and community practice

---

## 1. 3D Printing (Additive Manufacturing)

### 1.1 Fused Deposition Modeling (FDM)

**Principle:** Thermoplastic filament extruded through heated nozzle, deposited layer by layer.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Layer height | 0.1–0.3 mm | 0.2 mm standard; 0.1 mm for fine detail |
| Nozzle diameter | 0.4 mm (standard), 0.6–1.0 mm (fast) | Smaller = finer detail, slower |
| Print speed | 40–150 mm/s | Quality vs. speed tradeoff |
| Bed temperature | 60–110 °C | Material-dependent |
| Nozzle temperature | 190–300 °C | Material-dependent |
| Dimensional accuracy | ±0.2–0.5 mm | Depends on machine, material, geometry |
| Minimum feature size | ~2× nozzle diameter | ~0.8 mm for 0.4 mm nozzle |
| Build volume (common) | 220×220×250 mm to 350×350×400 mm | Larger = more expensive |

**Materials for Robotics:**

| Material | Tensile Strength | Flexural Modulus | Max Use Temp | Cost/kg | Best For |
|----------|------------------|------------------|--------------|---------|----------|
| PLA | 30–60 MPa | 2.5–3.5 GPa | 50–60 °C | $20–30 | Prototypes, jigs, non-structural |
| PETG | 50–75 MPa | 2.0–2.5 GPa | 70–80 °C | $25–35 | Structural brackets, housings |
| ABS | 30–50 MPa | 1.8–2.5 GPa | 90–100 °C | $20–30 | Impact-resistant parts (needs enclosure) |
| ASA | 40–55 MPa | 2.0–2.5 GPa | 90–100 °C | $30–40 | Outdoor/UV-stable parts |
| Nylon (PA6/PA12) | 60–85 MPa | 1.5–2.5 GPa | 100–150 °C | $40–80 | Gears, bearings, tough structural |
| Polycarbonate (PC) | 60–75 MPa | 2.0–2.4 GPa | 130–140 °C | $40–60 | High-temp, impact-resistant |
| Carbon-fiber filled (CF-PA, CF-PC) | 80–150 MPa | 4–8 GPa | 120–150 °C | $60–120 | Rigid structural, robot links |
| Glass-fiber filled (GF-PA) | 70–100 MPa | 3–5 GPa | 120–150 °C | $50–90 | Stiff, less abrasive than CF |

**Verified Constants:**
- PLA glass transition temperature Tg ≈ **55–60 °C** (source: material datasheets, literature)
- ABS Tg ≈ **105 °C** (source: ASTM standards)
- Nylon 6/6 Tg ≈ **50–60 °C**, melting point Tm ≈ **255–265 °C** (source: polymer handbooks)
- Annealing nylon in salt: 150–170 °C for 30 min increases crystallinity and stiffness (source: community practice, Prusa/MatterHackers guides)

**Feasibility Boundaries:**
- FDM parts are **anisotropic**: Z-axis strength is 60–80% of XY strength due to layer adhesion (source: ISO/ASTM 52900 test methods)
- Tight tolerances (>±0.1 mm) require post-machining or design compensation
- Internal stresses cause warping; large flat parts need brims/rafts/enclosures
- Threaded inserts (heat-set or ultrasonic) are preferred over tapped plastic threads for M3 and larger

**Open Questions:**
1. What is the fatigue life of CF-PA under cyclic loading (robot joint cycles)?
2. Can vapor smoothing (acetone for ABS, MEK for ASA) achieve sufficient surface finish for bearing seats?
3. What is the optimal infill pattern and density for robot links under bending + torsion?

---

### 1.2 Stereolithography (SLA / Resin)

**Principle:** UV laser or projector cures liquid photopolymer resin layer by layer.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Layer height | 0.025–0.1 mm | Much finer than FDM |
| XY resolution | 0.05–0.1 mm | Laser spot or pixel size |
| Dimensional accuracy | ±0.1–0.2 mm | Better than FDM |
| Surface finish | Ra 1–5 µm | Near-injection-mold quality |
| Build volume | 120×68×150 mm to 300×170×400 mm | Smaller than FDM typically |
| Print speed | 20–60 mm/h (Z) | Measured in vertical speed |

**Materials for Robotics:**

| Material Type | Tensile Strength | Elongation | Max Use Temp | Cost/L | Best For |
|---------------|------------------|------------|--------------|--------|----------|
| Standard resin | 30–60 MPa | 2–10% | 40–60 °C | $30–60 | Visual prototypes, molds |
| Tough/durable resin | 30–55 MPa | 20–80% | 40–60 °C | $50–100 | Functional prototypes, snap fits |
| Engineering (ABS-like) | 40–70 MPa | 10–50% | 50–70 °C | $60–120 | Structural prototypes |
| High-temp resin | 50–80 MPa | 2–10% | 200–300 °C (HDT) | $100–200 | Molds, fixtures, heat-resistant |
| Flexible/elastic resin | 5–30 MPa | 50–300% | 40–60 °C | $80–150 | Seals, gaskets, dampers |
| Ceramic-filled resin | 60–100 MPa | 1–3% | 1000+ °C (fired) | $100–200 | Investment casting patterns |

**Verified Constants:**
- Standard resin HDT (heat deflection temp) @ 0.45 MPa: **40–60 °C** (source: Formlabs, Anycubic datasheets)
- High-temp resin HDT: up to **238 °C** (Formlabs High Temp Resin v2, source: datasheet)
- Post-cure requirement: 30–60 min at 60 °C under UV for full properties (source: resin manufacturer protocols)

**Feasibility Boundaries:**
- Resins are **brittle** compared to FDM thermoplastics; poor for impact loads
- UV degradation over time; not suitable for long-term outdoor exposure without coating
- Requires washing and post-curing workflow
- Isotropic properties (no layer weakness like FDM)

**Open Questions:**
1. Can ceramic-filled resin be fired to produce custom ceramic bearings or insulators?
2. What is the long-term creep behavior of tough resins under constant load?

---

### 1.3 Selective Laser Sintering (SLS)

**Principle:** CO₂ laser fuses powdered material (typically nylon) layer by layer.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Layer height | 0.08–0.15 mm | |
| Dimensional accuracy | ±0.2–0.3 mm | Good for complex geometries |
| Surface finish | Ra 6–12 µm | Slightly grainy, can be dyed/tumbled |
| Build volume | 300×300×300 mm to 700×380×580 mm | Industrial machines |
| Material | PA12 (nylon 12) most common | Also PA11, TPU, PP |

**Verified Constants:**
- SLS PA12 tensile strength: **45–50 MPa** (source: EOS, HP datasheets)
- SLS PA12 elongation at break: **10–15%** (source: EOS PA2200 datasheet)
- SLS TPU (flexible) Shore A: **80–95** (source: material datasheets)

**Feasibility Boundaries:**
- No support structures needed (powder bed is self-supporting) → complex geometries, lattice structures
- Higher cost than FDM: service bureau ~$5–15/cm³; industrial machine $200K–$500K
- Excellent for production-grade robot end-use parts if budget allows
- Good chemical resistance, low moisture absorption (PA12)

**Open Questions:**
1. What is the fatigue performance of SLS PA12 vs. injection-molded PA12?
2. Can SLS produce sufficiently precise gear teeth for robotic transmissions?

---

## 2. CNC Machining (Subtractive)

### 2.1 Milling

**Principle:** Rotary cutting tool removes material from a workpiece.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Dimensional accuracy | ±0.025–0.125 mm (±0.001–0.005 in) | Tight tolerance = slower, more expensive |
| Surface finish (roughing) | Ra 3.2–6.3 µm | |
| Surface finish (finishing) | Ra 0.4–1.6 µm | Fine finishing passes |
| Minimum wall thickness | 0.5–1.0 mm (aluminum) | Thinner = vibration, deflection |
| Internal corner radius | ≥ tool radius (typically ≥1.5 mm for small mills) | Sharp internal corners impossible |

**Feeds and Speeds (Aluminum 6061-T6, carbide end mill):**

| Tool Diameter | Spindle Speed (RPM) | Feed Rate (mm/min) | Depth of Cut |
|---------------|---------------------|--------------------|--------------|
| 3 mm | 12,000–16,000 | 600–1,200 | 1× diameter (rough), 0.3× (finish) |
| 6 mm | 8,000–12,000 | 1,000–2,000 | 1× diameter (rough), 0.3× (finish) |
| 10 mm | 5,000–8,000 | 1,500–3,000 | 1× diameter (rough), 0.3× (finish) |

**Verified Constants:**
- Aluminum 6061-T6 specific cutting energy: **0.7–1.0 kW/(cm³/min)** (source: machining handbooks)
- Chip load per tooth (aluminum): **0.05–0.15 mm/tooth** for finishing, **0.1–0.3 mm/tooth** for roughing (source: Harvey Tool, Kennametal guides)

**Feasibility Boundaries:**
- 3-axis mills: limited to prismatic parts, no undercuts without fixturing gymnastics
- 5-axis mills: complex geometries, $100K–$500K+ machine cost; service bureaus charge $80–$200/hour
- Minimum feature size limited by tool diameter and spindle runout
- Thin walls and deep pockets are challenging (tool deflection, chatter)

**Open Questions:**
1. What is the cost-optimal batch size for machined vs. printed robot brackets?
2. Can high-speed machining (HSM) strategies reduce cycle time for aluminum robot links by 50%?

---

### 2.2 Turning / Lathe

**Principle:** Workpiece rotates while stationary cutting tool removes material.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Dimensional accuracy | ±0.01–0.05 mm | Excellent for cylindrical features |
| Surface finish | Ra 0.8–3.2 µm | |
| Best for | Shafts, pulleys, bearing housings, threaded features | Axially symmetric parts |

**Verified Constants:**
- Turning aluminum 6061-T6: cutting speed **200–400 m/min** (source: Sandvik Coromant guides)

**Feasibility Boundaries:**
- Limited to parts with rotational symmetry (or near-symmetry with live tooling)
- Live tooling (mill-turn) adds complexity and cost but enables off-axis features

---

### 2.3 Waterjet / Laser Cutting

| Process | Material Thickness | Kerf Width | Accuracy | Best For |
|---------|-------------------|------------|----------|----------|
| Waterjet | Up to 150 mm | 0.8–1.5 mm | ±0.1–0.2 mm | Thick plates, composites, heat-sensitive materials |
| CO₂ laser | Up to 25 mm (steel), 12 mm (aluminum) | 0.1–0.3 mm | ±0.05–0.1 mm | Sheet metal, plastics, wood |
| Fiber laser | Up to 25 mm (steel), 12 mm (aluminum) | 0.1–0.2 mm | ±0.05–0.1 mm | Sheet metal, faster than CO₂ |

**Verified Constants:**
- Fiber laser cutting speed (aluminum 3 mm): **~8–15 m/min** (source: Trumpf, Amada datasheets)
- Waterjet cutting speed (aluminum 6 mm): **~150–300 mm/min** (source: OMAX, Flow specs)

**Feasibility Boundaries:**
- Laser: heat-affected zone (HAZ) can alter material properties near cut edge
- Waterjet: no HAZ, but slower; edge quality rougher than laser
- Both are 2D processes; 3D requires 5-axis waterjet or tube cutting

---

## 3. Sheet Metal Fabrication

### 3.1 Bending

**Principle:** Sheet metal deformed plastically along a straight line.

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Minimum bend radius | 1× material thickness (aluminum) | Smaller radius = cracking risk |
| Bend allowance | Calculated from material thickness, bend radius, bend angle | K-factor typically 0.33–0.5 |
| Springback | 1–3° (aluminum), up to 10° (steel) | Must over-bend or use coining |
| Tolerance | ±0.25–0.5 mm on bend angle and location | |

**Verified Constants:**
- Aluminum 5052-H32 minimum bend radius: **0× thickness** (very ductile) to **1× thickness** (source: ASM Handbook, vendor bend tables)
- Aluminum 6061-T6 minimum bend radius: **1.5–3× thickness** (less ductile in T6 temper) (source: ASM Handbook)

**Feasibility Boundaries:**
- Press brake: simple bends, low tooling cost
- Roll forming: continuous profiles, high volume
- Hemming: edge folding for safety/stiffness
- Minimum flange length: typically 4× material thickness + bend radius

---

### 3.2 Punching / Stamping

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Hole diameter | ≥ material thickness | Smaller = tool breakage |
| Hole-to-edge distance | ≥ 1.5× material thickness | |
| Tolerance | ±0.1–0.25 mm | |

**Feasibility Boundaries:**
- High setup cost, economical only at volume (>100–1000 parts)
- Turret punch: flexible, lower volume; stamping die: dedicated, high volume

---

## 4. Carbon Fiber Composites

### 4.1 Manufacturing Methods

| Method | Best For | Labor Intensity | Surface Finish | Cost |
|--------|----------|-----------------|----------------|------|
| Hand layup | One-offs, prototypes, repairs | High | Fair | Low |
| Vacuum bagging | Improved consolidation over hand layup | Medium | Good | Medium |
| Resin infusion (RTM/VARTM) | Medium volume, complex shapes | Medium | Good | Medium |
| Prepreg + autoclave | Aerospace quality, high performance | Medium | Excellent | High |
| Filament winding | Tubes, pressure vessels, robot links | Low (automated) | Good | Medium-High |
| Compression molding | High volume, simple shapes | Low | Excellent | High (tooling) |

### 4.2 Material Properties

| Property | Standard Modulus CF/Epoxy | High Modulus CF/Epoxy | Notes |
|----------|---------------------------|------------------------|-------|
| Tensile strength | 1,500–2,000 MPa | 1,200–1,500 MPa | Along fiber direction |
| Tensile modulus | 150–230 GPa | 300–450 GPa | Along fiber direction |
| Density | 1,500–1,600 kg/m³ | 1,500–1,600 kg/m³ | |
| Specific strength | ~1,000 kN·m/kg | ~800 kN·m/kg | Strength/density |
| Specific stiffness | ~100 MN·m/kg | ~200 MN·m/kg | Modulus/density |
| Compressive strength | 1,200–1,500 MPa | 1,000–1,200 MPa | Often lower than tensile |
| Interlaminar shear | 80–120 MPa | 60–90 MPa | Weak direction |

**Verified Constants:**
- T300 standard modulus fiber: tensile strength **3,530 MPa**, modulus **230 GPa** (source: Toray datasheet)
- T800H intermediate modulus: tensile strength **5,490 MPa**, modulus **294 GPa** (source: Toray datasheet)
- M40J high modulus: tensile strength **4,410 MPa**, modulus **377 GPa** (source: Toray datasheet)
- Quasi-isotropic laminate [0/±45/90]s: in-plane modulus ~**70–80 GPa** (source: composite mechanics textbooks, rule of mixtures)

**Feasibility Boundaries:**
- CF is **anisotropic** and **brittle**; design must account for fiber direction
- Impact damage causes internal delamination not visible externally
- Drilling and machining CF produces conductive dust that damages electronics
- Moisture absorption degrades epoxy matrix over time
- CTE (coefficient of thermal expansion) near-zero in fiber direction; can cause thermal stress at metal interfaces

**Open Questions:**
1. What is the optimal laminate schedule for a humanoid robot thigh link (bending + torsion)?
2. Can 3D-printed continuous fiber (Markforged, Anisoprint) achieve sufficient properties for structural robot links?
3. What is the fatigue life of CF/epoxy under robot joint cyclic loading?

---

## 5. Aluminum Extrusion

### 5.1 T-Slot Extrusion Systems

**Principle:** Aluminum alloy forced through a die to create a constant cross-section profile with T-slots for modular assembly.

| Profile Series | Cross-Section | Typical Use | Weight/m | Cost/m |
|---------------|---------------|-------------|----------|--------|
| 20×20 mm | 20×20 mm | Small fixtures, sensors | ~0.4 kg/m | $3–6 |
| 30×30 mm | 30×30 mm | Medium frames, gantries | ~0.9 kg/m | $5–10 |
| 40×40 mm | 40×40 mm | Robot frames, test rigs | ~1.5 kg/m | $7–15 |
| 80×40 mm | 80×40 mm | Heavy structures, bases | ~3.0 kg/m | $15–25 |

**Verified Constants:**
- Common alloy: **6063-T5** or **6061-T6** (source: 80/20, Bosch Rexroth, Item datasheets)
- 6063-T5 yield strength: **145 MPa** (source: ASTM B221)
- 6061-T6 yield strength: **276 MPa** (source: ASTM B221)
- T-slot nut grip: typically **M4, M5, M6, M8** threads (source: vendor catalogs)

**Feasibility Boundaries:**
- Excellent for prototyping frames, test rigs, and non-structural assemblies
- Not suitable for high-stress robot links (joints, limbs) without significant reinforcement
- Connection joints are the weak point; bracket design matters more than extrusion strength
- Vibration can loosen T-slot connections without thread-locking compound or lock washers

**Open Questions:**
1. Can hybrid extrusion + machined end fittings produce viable robot links?
2. What is the optimal bracket design for a 40×40 mm extrusion knee joint?

---

## 6. Tolerances and Surface Finishes

### 6.1 Tolerance Grades (ISO 2768 / ASME Y14.5)

| Tolerance Grade | Linear (mm) | Angular | Typical Application |
|-----------------|-------------|---------|---------------------|
| IT6 | ±0.01–0.02 | — | Precision bearings, gauges |
| IT7 | ±0.015–0.03 | — | Ball bearing fits, precision shafts |
| IT8 | ±0.025–0.05 | — | General machining, gear fits |
| IT9 | ±0.04–0.1 | — | Standard machining, clearance fits |
| IT10 | ±0.06–0.15 | — | Rough machining, castings |
| IT11–IT16 | ±0.1–2.0 | — | Forging, welding, rough castings |

**For Humanoid Robotics:**
- Bearing bores: **IT6–IT7** (source: SKF/NSK bearing mounting guidelines)
- Gear bores and shafts: **IT7–IT8** (source: AGMA/ISO gear standards)
- General structural: **IT9–IT10**
- 3D printed parts: typically **IT11–IT13** unless post-machined

### 6.2 Surface Finish (Ra, µm)

| Ra (µm) | Process | Application |
|---------|---------|-------------|
| 0.025–0.1 | Lapping, superfinishing | High-precision bearings, seals |
| 0.1–0.4 | Grinding, honing | Bearing races, hydraulic cylinders |
| 0.4–1.6 | Precision machining, fine grinding | Gears, shafts, bearing seats |
| 1.6–6.3 | Standard machining | General mechanical parts |
| 6.3–25 | Rough machining, casting, forging | Non-critical structural |
| 25–100 | As-cast, as-forged, weld bead | Heavy structural, non-mating |

**Verified Constants:**
- Recommended Ra for rolling bearing seats: **0.8–1.6 µm** (source: SKF General Catalog)
- Recommended Ra for sliding bearings: **0.2–0.8 µm** (source: machinery design handbooks)

---

## 7. Joining Methods

| Method | Strength | Cost | Best For | Notes |
|--------|----------|------|----------|-------|
| Bolting (SHCS) | High (joint-dependent) | Low | Removable assemblies, prototyping | Most common in robotics |
| Threaded insert (heat-set) | Medium | Low | Plastic parts | M3–M8 common |
| Riveting | High | Low | Sheet metal, permanent | Blind rivets for access |
| Adhesive bonding (epoxy, acrylic) | Medium-High | Low | CF, mixed materials, sealing | Surface prep critical |
| Welding (TIG, MIG) | Very high | Medium | Steel, aluminum structures | Heat distortion risk |
| Brazing | High | Medium | Dissimilar metals, heat-sensitive nearby | Lower temp than welding |
| Press fit / interference fit | High | Low | Bearings, pins | Requires precise tolerances |
| Snap fit (plastic) | Low-Medium | Very low | Enclosures, covers | Design for plastic deformation |

**Verified Constants:**
- Typical bolt preload: **75% of proof load** for reusable joints, **90% for permanent** (source: Shigley's Mechanical Engineering Design)
- Torque-preload relationship: T = K·D·F, where K ≈ **0.2** for dry steel, **0.15** for lubricated (source: machinery handbooks)

---

## Verified Data Summary

| Data Point | Source | Confidence |
|------------|--------|------------|
| FDM layer height / accuracy ranges | ISO/ASTM 52900, printer specs | High |
| PLA/ABS/PETG/PC mechanical properties | Material datasheets, literature | High |
| SLS PA12 properties | EOS, HP datasheets | High |
| SLA resin properties | Formlabs, Anycubic datasheets | High |
| Aluminum 6061-T6 machining parameters | Sandvik, Kennametal, Harvey Tool | High |
| CF fiber properties (T300, T800, M40J) | Toray datasheets | High |
| Aluminum extrusion specs | 80/20, Bosch Rexroth, Item | High |
| ISO tolerance grades | ISO 2768, ASME Y14.5 | High |
| Surface finish recommendations | SKF, machinery design handbooks | High |
| Bolt preload / torque constants | Shigley's Mechanical Engineering Design | High |
| 3D printing anisotropy ratios | ASTM test methods, literature | Medium |
| Cost estimates | Market prices, service bureau quotes | Low-Medium (volatile) |

---

## Open Questions / Learning Targets

1. What is the optimal manufacturing method for each Titan subsystem (frame, links, joints, end effectors)?
2. Can topology-optimized 3D-printed titanium or aluminum parts replace machined brackets?
3. What is the cost and time tradeoff between in-house FDM vs. service bureau SLS/SLA/MJF?
4. How do we design for manufacturability (DFM) across FDM, CNC, and sheet metal in one assembly?
5. What post-processing (annealing, vapor smoothing, coating) is needed for each printed material?
6. Can we establish a standard fastener kit (M3, M4, M5, M6 SHCS + nuts + washers) to reduce BOM complexity?
7. What is the minimum viable surface finish for each joint type (revolute, prismatic, spherical)?

---

*Last verified: 2026-06-08. Next review: after first manufacturing run.*
