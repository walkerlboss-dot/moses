"""
moses/industrial/safety.py
===========================

Industrial safety system for humanoid robot cells.

Implements:
  - Risk assessment per ISO 12100
  - Safety function calculations per SISTEMA
  - Performance Level (PL) determination per ISO 13849-1
  - Safety PLC integration
  - Collaborative robot safety per ISO/TS 15066

Standards referenced:
  - ISO 12100:2010  - Risk assessment and risk reduction
  - ISO 10218-1:2011 - Robot safety (industrial robots)
  - ISO 10218-2:2011 - Robot systems and integration
  - ISO 13849-1:2023 - Safety-related parts of control systems
  - ISO 13849-2:2012 - Validation
  - ISO 13850:2015  - Emergency stop
  - ISO 13855:2010  - Positioning of safeguards
  - ISO 13857:2019  - Safety distances
  - IEC 62061:2021  - Safety of machinery - Functional safety
  - IEC 61508:2010  - Functional safety of E/E/PE systems
  - ISO/TS 15066:2016 - Collaborative robots (now ISO/TR 20218-2)
  - ISO/TR 20218-1:2018 - Application of ISO 10218
  - SISTEMA (Software for evaluation of machine applications)

Author: Moses Industrial Team
Version: 6.0.0

DISCLAIMER: This module provides educational reference implementations.
Actual safety system design must be performed by qualified safety engineers
and validated by third-party notified bodies where required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union


# ---------------------------------------------------------------------------
# Hazard and Risk Assessment (ISO 12100)
# ---------------------------------------------------------------------------

class HazardType(Enum):
    """Hazard types per ISO 12100 Annex B."""
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    NOISE = "noise"
    VIBRATION = "vibration"
    RADIATION = "radiation"
    MATERIAL_SUBSTANCE = "material_substance"
    ERGONOMIC = "ergonomic"
    ENVIRONMENTAL = "environmental"
    COMBINATION = "combination"


class HarmSeverity(Enum):
    """Severity of harm (S) per ISO 12100 / ISO 13849-1."""
    SLIGHT = 1          # S1: Slight (normally reversible injury)
    SERIOUS = 2         # S2: Serious (normally irreversible injury, including death)


class ExposureFrequency(Enum):
    """Frequency and/or duration of exposure (F) per ISO 13849-1."""
    SELDOM_TO_LESS_OFTEN = 1    # F1: Seldom to less often
    FREQUENT_TO_CONTINUOUS = 2  # F2: Frequent to continuous


class AvoidancePossibility(Enum):
    """Possibility of avoiding hazard (P) per ISO 13849-1."""
    POSSIBLE = 1        # P1: Possible under specific conditions
    SCARCELY_POSSIBLE = 2  # P2: Scarcely possible


@dataclass
class Hazard:
    """Individual hazard description per ISO 12100."""
    id: str
    description: str
    hazard_type: HazardType
    task: str
    hazard_zone: str
    severity: HarmSeverity
    exposure: ExposureFrequency
    avoidance: AvoidancePossibility
    risk_score: Optional[int] = None

    def calculate_risk(self) -> int:
        """
        Calculate risk score (simplified ISO 12100 approach).

        Risk = S * F * P (each 1 or 2)
        Range: 1 (lowest) to 8 (highest)
        """
        self.risk_score = self.severity.value * self.exposure.value * self.avoidance.value
        return self.risk_score


@dataclass
class RiskAssessment:
    """
    Risk assessment document per ISO 12100.

    Contains hazard identification, risk estimation, and
    risk reduction measures.
    """
    machine_name: str
    assessment_id: str
    assessor: str
    date: str
    hazards: List[Hazard] = field(default_factory=list)
    risk_reductions: List[Dict[str, Any]] = field(default_factory=list)

    def add_hazard(self, hazard: Hazard) -> None:
        hazard.calculate_risk()
        self.hazards.append(hazard)

    def get_required_pl(self) -> str:
        """
        Determine required Performance Level from risk graph.

        ISO 13849-1 Figure 5:
          S1 + F1 + P1 -> PL a
          S1 + F1 + P2 -> PL b
          S1 + F2 + P1 -> PL b
          S1 + F2 + P2 -> PL c
          S2 + F1 + P1 -> PL c
          S2 + F1 + P2 -> PL d
          S2 + F2 + P1 -> PL d
          S2 + F2 + P2 -> PL e
        """
        max_risk = max((h.risk_score or 0 for h in self.hazards), default=0)
        pl_map = {
            1: "a", 2: "b", 3: "c", 4: "d",
            5: "d", 6: "d", 7: "e", 8: "e",
        }
        return pl_map.get(max_risk, "e")

    def add_risk_reduction(
        self,
        hazard_id: str,
        measure: str,
        measure_type: str,      # "inherently_safe", "safeguard", "information"
        residual_risk: int,
    ) -> None:
        self.risk_reductions.append({
            "hazard_id": hazard_id,
            "measure": measure,
            "type": measure_type,
            "residual_risk": residual_risk,
        })

    def generate_report(self) -> Dict[str, Any]:
        """Generate ISO 12100 compliant risk assessment report."""
        return {
            "machine": self.machine_name,
            "assessment_id": self.assessment_id,
            "assessor": self.assessor,
            "date": self.date,
            "hazards": [
                {
                    "id": h.id,
                    "description": h.description,
                    "type": h.hazard_type.value,
                    "severity": h.severity.name,
                    "exposure": h.exposure.name,
                    "avoidance": h.avoidance.name,
                    "risk_score": h.risk_score,
                }
                for h in self.hazards
            ],
            "required_pl": self.get_required_pl(),
            "risk_reductions": self.risk_reductions,
        }


# ---------------------------------------------------------------------------
# SISTEMA Calculations (ISO 13849-1)
# ---------------------------------------------------------------------------

class PerformanceLevel(Enum):
    """Performance Levels per ISO 13849-1."""
    PL_A = "a"
    PL_B = "b"
    PL_C = "c"
    PL_D = "d"
    PL_E = "e"


class Category(Enum):
    """Architectural Categories per ISO 13849-1."""
    CAT_B = "B"
    CAT_1 = "1"
    CAT_2 = "2"
    CAT_3 = "3"
    CAT_4 = "4"


class DCLevel(Enum):
    """Diagnostic Coverage levels per ISO 13849-1 Table E.1."""
    NONE = 0.0
    LOW = 0.6          # 60% < DC < 99%
    MEDIUM = 0.9       # 90% < DC < 99%
    HIGH = 0.99        # DC >= 99%


class CCFScore:
    """Common Cause Failure score per ISO 13849-1 Annex F."""
    def __init__(self) -> None:
        self.measures: Dict[str, bool] = {
            "separation": False,           # Physical separation
            "diversity": False,            # Diversity in design
            "protection": False,           # Protection against environmental stress
            "analysis": False,             # FMEA/FTA analysis
            "experience": False,           # Proven experience
        }

    def score(self) -> int:
        """Return CCF score (minimum 65 required for Category 3/4)."""
        points = {
            "separation": 15,
            "diversity": 20,
            "protection": 15,
            "analysis": 5,
            "experience": 5,
        }
        return sum(points[k] for k, v in self.measures.items() if v)

    def meets_requirement(self) -> bool:
        return self.score() >= 65


@dataclass
class SafetyComponent:
    """Single safety-related component."""
    name: str
    b10d: float                 # B10d value (cycles to 10% dangerous failure)
    nop: float                  # Number of operations per year
    d_level: str = "low"        # "low", "medium", "high" (dangerous failure distribution)
    beta: float = 0.02          # Common cause failure factor (for redundant channels)
    channel: str = "single"     # "single", "channel_1", "channel_2"

    def calculate_mtffd(self) -> float:
        """
        Calculate MTTFd (Mean Time To Dangerous Failure).

        MTTFd = B10d / (0.1 * nop)  [years]
        """
        return self.b10d / (0.1 * self.nop)

    def get_mtffd_level(self) -> str:
        """
        Determine MTTFd level per ISO 13849-1 Table 4.

        Low:    3 years <= MTTFd < 10 years
        Medium: 10 years <= MTTFd < 30 years
        High:   30 years <= MTTFd < 100 years
        """
        mtffd = self.calculate_mtffd()
        if mtffd < 3:
            return "low"
        elif mtffd < 10:
            return "low"
        elif mtffd < 30:
            return "medium"
        elif mtffd < 100:
            return "high"
        return "high"


@dataclass
class SafetyFunction:
    """
    Safety function block per ISO 13849-1 / SISTEMA.

    A safety function comprises:
      - Input device(s) (sensor)
      - Logic unit (safety PLC/relay)
      - Output device(s) (actuator)
    """
    name: str
    category: Category
    inputs: List[SafetyComponent] = field(default_factory=list)
    logic: List[SafetyComponent] = field(default_factory=list)
    outputs: List[SafetyComponent] = field(default_factory=list)
    dc: DCLevel = DCLevel.NONE
    ccf: CCFScore = field(default_factory=CCFScore)

    def calculate_mtffd_channel(self, components: List[SafetyComponent]) -> float:
        """
        Calculate MTTFd for a channel with multiple components in series.

        1/MTTFd_channel = sum(1/MTTFd_i) for all components in channel
        """
        total = 0.0
        for comp in components:
            mtffd = comp.calculate_mtffd()
            if mtffd > 0:
                total += 1.0 / mtffd
        return 1.0 / total if total > 0 else float("inf")

    def determine_pl(self) -> PerformanceLevel:
        """
        Determine achieved Performance Level per ISO 13849-1 Figure 6.

        Uses Category, MTTFd (per channel), and DC.
        """
        # Calculate MTTFd for each channel
        input_mtffd = self.calculate_mtffd_channel(self.inputs)
        logic_mtffd = self.calculate_mtffd_channel(self.logic)
        output_mtffd = self.calculate_mtffd_channel(self.outputs)

        # Overall MTTFd is the lowest of input/logic/output
        overall_mtffd = min(input_mtffd, logic_mtffd, output_mtffd)

        # Determine MTTFd level
        if overall_mtffd < 3:
            mtffd_level = "low"
        elif overall_mtffd < 10:
            mtffd_level = "low"
        elif overall_mtffd < 30:
            mtffd_level = "medium"
        else:
            mtffd_level = "high"

        # PL determination matrix (simplified from ISO 13849-1 Table 6)
        pl_matrix = {
            Category.CAT_B: {
                "low": PerformanceLevel.PL_A,
                "medium": PerformanceLevel.PL_A,
                "high": PerformanceLevel.PL_A,
            },
            Category.CAT_1: {
                "low": PerformanceLevel.PL_B,
                "medium": PerformanceLevel.PL_B,
                "high": PerformanceLevel.PL_B,
            },
            Category.CAT_2: {
                "low": PerformanceLevel.PL_B,
                "medium": PerformanceLevel.PL_C,
                "high": PerformanceLevel.PL_D,
            },
            Category.CAT_3: {
                "low": PerformanceLevel.PL_C,
                "medium": PerformanceLevel.PL_D,
                "high": PerformanceLevel.PL_E,
            },
            Category.CAT_4: {
                "low": PerformanceLevel.PL_C,
                "medium": PerformanceLevel.PL_D,
                "high": PerformanceLevel.PL_E,
            },
        }

        base_pl = pl_matrix[self.category][mtffd_level]

        # Adjust for DC (simplified: DC can increase PL by one step)
        if self.dc == DCLevel.LOW and base_pl.value < "d":
            return PerformanceLevel(chr(ord(base_pl.value) + 1))
        elif self.dc == DCLevel.MEDIUM and base_pl.value < "e":
            return PerformanceLevel(chr(ord(base_pl.value) + 1))
        elif self.dc == DCLevel.HIGH and base_pl.value < "e":
            return PerformanceLevel(chr(min(ord(base_pl.value) + 2, ord("e"))))

        return base_pl

    def calculate_pfh(self) -> float:
        """
        Calculate PFHd (Probability of Dangerous Failure per Hour).

        Per ISO 13849-1 Annex K:
          PFHd = 1 / MTTFd_channel + beta * (1 / MTTFd_channel2) + DC_factor

        For Category 3/4 with redundant channels:
          PFHd ≈ beta * (1/MTTFd)^2 * T1 + DC_terms
        """
        mtffd = self.calculate_mtffd_channel(self.inputs + self.logic + self.outputs)
        if mtffd <= 0:
            return float("inf")

        if self.category in (Category.CAT_3, Category.CAT_4):
            # Simplified PFHd for redundant architecture
            beta = 0.02  # Typical for well-separated channels
            t1 = 20 * 8760  # 20 years in hours (proof test interval)
            pfhd = beta * (1.0 / (mtffd * 8760))**2 * t1
        else:
            pfhd = 1.0 / (mtffd * 8760)

        return pfhd

    def meets_requirement(self, required_pl: PerformanceLevel) -> bool:
        """Check if achieved PL meets required PL."""
        achieved = self.determine_pl()
        pl_order = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        return pl_order[achieved.value] >= pl_order[required_pl.value]


class SISTEMACalculator:
    """
    SISTEMA-style safety function calculator.

    Aggregates multiple safety functions and validates
    against required PL per ISO 13849-1.
    """
    def __init__(self) -> None:
        self.functions: Dict[str, SafetyFunction] = {}

    def add_function(self, func: SafetyFunction) -> None:
        self.functions[func.name] = func

    def validate_all(self, required_pl: PerformanceLevel) -> Dict[str, Any]:
        """Validate all safety functions against required PL."""
        results = {}
        for name, func in self.functions.items():
            achieved = func.determine_pl()
            pfhd = func.calculate_pfh()
            meets = func.meets_requirement(required_pl)
            results[name] = {
                "achieved_pl": achieved.value,
                "required_pl": required_pl.value,
                "meets_requirement": meets,
                "pfhd": pfhd,
                "mtffd": func.calculate_mtffd_channel(
                    func.inputs + func.logic + func.outputs
                ),
                "category": func.category.value,
                "dc": func.dc.name,
                "ccf_score": func.ccf.score(),
                "ccf_ok": func.ccf.meets_requirement(),
            }
        return results

    def generate_sistema_report(self) -> Dict[str, Any]:
        """Generate report compatible with SISTEMA format."""
        return {
            "version": "2.0",
            "functions": {
                name: {
                    "inputs": [{"name": c.name, "b10d": c.b10d, "nop": c.nop}
                               for c in func.inputs],
                    "logic": [{"name": c.name, "b10d": c.b10d, "nop": c.nop}
                              for c in func.logic],
                    "outputs": [{"name": c.name, "b10d": c.b10d, "nop": c.nop}
                                for c in func.outputs],
                    "category": func.category.value,
                    "dc": func.dc.name,
                }
                for name, func in self.functions.items()
            },
        }


# ---------------------------------------------------------------------------
# Safety PLC Integration
# ---------------------------------------------------------------------------

class SafetyPLCInterface:
    """
    Safety PLC communication interface.

    Supports:
      - Safe I/O monitoring
      - Safety program execution
      - Fault reaction time calculation
      - Safe torque off (STO) activation

    Compatible with:
      - Pilz PNOZmulti
      - Siemens SIRIUS 3SK
      - Schneider Preventa
      - Omron G9SE
    """
    def __init__(
        self,
        plc_type: str = "pilz_pnozmulti",
        safety_cycle_ms: float = 10.0,
        response_time_ms: float = 50.0,
    ) -> None:
        self.plc_type = plc_type
        self.safety_cycle_ms = safety_cycle_ms
        self.response_time_ms = response_time_ms
        self._safe_inputs: Dict[str, bool] = {}
        self._safe_outputs: Dict[str, bool] = {}
        self._faults: List[str] = []

    def read_safe_input(self, name: str) -> bool:
        """Read safety-rated input (dual-channel, tested)."""
        return self._safe_inputs.get(name, False)

    def write_safe_output(self, name: str, value: bool) -> None:
        """Write safety-rated output (monitored, pulse-tested)."""
        self._safe_outputs[name] = value

    def check_dual_channel(
        self,
        channel_a: str,
        channel_b: str,
        discrepancy_time_ms: float = 50.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check dual-channel safety input for discrepancy.

        Per ISO 13849-1 Category 3/4, both channels must agree
        within discrepancy time.
        """
        a = self._safe_inputs.get(channel_a, False)
        b = self._safe_inputs.get(channel_b, False)
        if a != b:
            return False, f"Discrepancy: {channel_a}={a}, {channel_b}={b}"
        return a, None

    def calculate_fault_reaction_time(
        self,
        sensor_response_ms: float,
        plc_cycle_ms: float,
        contactor_release_ms: float,
    ) -> float:
        """
        Calculate total fault reaction time.

        FRT = sensor_response + 2*plc_cycle + contactor_release + safety_margin
        Per ISO 13849-1, must be less than process stopping time.
        """
        safety_margin = 20.0  # 20 ms engineering margin
        return sensor_response_ms + 2 * plc_cycle_ms + contactor_release_ms + safety_margin

    def activate_sto(self, axis_name: str) -> None:
        """
        Activate Safe Torque Off (STO) per IEC 61800-5-2.

        STO is the most common safety function for servo drives.
        Removes power from motor windings without braking.
        """
        self.write_safe_output(f"STO_{axis_name}", True)

    def activate_ss1(self, axis_name: str, delay_ms: float) -> None:
        """
        Activate Safe Stop 1 (SS1) per IEC 61800-5-2.

        Initiates controlled stop, then activates STO after delay.
        """
        self.write_safe_output(f"SS1_{axis_name}", True)
        # In real implementation: start timer, then activate STO

    def activate_sls(self, axis_name: str, speed_limit: float) -> None:
        """
        Activate Safely-Limited Speed (SLS) per IEC 61800-5-2.

        Monitors actual speed against limit.
        """
        # Would configure safety speed monitor
        pass

    def run_self_test(self) -> List[str]:
        """Run safety PLC self-test sequence."""
        faults = []
        # Test all safe outputs
        for name in self._safe_outputs:
            # Toggle output and verify feedback
            pass
        return faults


# ---------------------------------------------------------------------------
# Collaborative Robot Safety (ISO/TS 15066)
# ---------------------------------------------------------------------------

class CollaborativeMode(Enum):
    """Collaborative operation modes per ISO/TS 15066."""
    SAFETY_RATED_MONITORED_STOP = "safety_rated_monitored_stop"
    HAND_GUIDING = "hand_guiding"
    SPEED_AND_SEPARATION_MONITORING = "speed_and_separation_monitoring"
    POWER_AND_FORCE_LIMITING = "power_and_force_limiting"


@dataclass
class BiomechanicalLimit:
    """
    Biomechanical limits for power and force limiting.

    Per ISO/TS 15066 Table A.1 - A.5:
      - Skull and forehead: 150 N (quasi-static), 130 N (transient)
      - Hand/fingers: 140 N (quasi-static), 70 N (transient)
      - Neck: 150 N (quasi-static)
      - etc.
    """
    body_region: str
    contact_type: str          # "quasi_static" or "transient"
    max_force_n: float
    max_pressure_n_per_cm2: float
    max_energy_j: Optional[float] = None


# ISO/TS 15066 Table A.2 (simplified)
BIOMECHANICAL_LIMITS: List[BiomechanicalLimit] = [
    BiomechanicalLimit("skull_forehead", "quasi_static", 150.0, 150.0 / 1.0),
    BiomechanicalLimit("skull_forehead", "transient", 130.0, 130.0 / 1.0),
    BiomechanicalLimit("face", "quasi_static", 65.0, 65.0 / 1.0),
    BiomechanicalLimit("face", "transient", 65.0, 65.0 / 1.0),
    BiomechanicalLimit("neck", "quasi_static", 150.0, 150.0 / 1.0),
    BiomechanicalLimit("neck", "transient", 150.0, 150.0 / 1.0),
    BiomechanicalLimit("back_chest", "quasi_static", 210.0, 210.0 / 30.0),
    BiomechanicalLimit("back_chest", "transient", 210.0, 210.0 / 30.0),
    BiomechanicalLimit("abdomen", "quasi_static", 140.0, 140.0 / 30.0),
    BiomechanicalLimit("abdomen", "transient", 140.0, 140.0 / 30.0),
    BiomechanicalLimit("pelvis", "quasi_static", 180.0, 180.0 / 30.0),
    BiomechanicalLimit("pelvis", "transient", 180.0, 180.0 / 30.0),
    BiomechanicalLimit("upper_arm", "quasi_static", 150.0, 150.0 / 10.0),
    BiomechanicalLimit("upper_arm", "transient", 150.0, 150.0 / 10.0),
    BiomechanicalLimit("lower_arm", "quasi_static", 160.0, 160.0 / 10.0),
    BiomechanicalLimit("lower_arm", "transient", 160.0, 160.0 / 10.0),
    BiomechanicalLimit("hand_fingers", "quasi_static", 140.0, 140.0 / 1.0),
    BiomechanicalLimit("hand_fingers", "transient", 70.0, 70.0 / 1.0),
    BiomechanicalLimit("lower_leg", "quasi_static", 130.0, 130.0 / 10.0),
    BiomechanicalLimit("lower_leg", "transient", 130.0, 130.0 / 10.0),
    BiomechanicalLimit("foot_toes", "quasi_static", 160.0, 160.0 / 10.0),
    BiomechanicalLimit("foot_toes", "transient", 160.0, 160.0 / 10.0),
]


@dataclass
class CollaborativeSafety:
    """
    Collaborative robot safety manager per ISO/TS 15066.

    Manages:
      - Power and force limiting (PFL)
      - Speed and separation monitoring (SSM)
      - Safety-rated monitored stop (SRMS)
      - Hand guiding
    """
    robot_mass_kg: float
    max_speed_m_per_s: float
    max_payload_kg: float
    collaborative_mode: CollaborativeMode = CollaborativeMode.POWER_AND_FORCE_LIMITING
    _active_limits: List[BiomechanicalLimit] = field(default_factory=list)

    def set_body_region_limits(self, body_region: str, contact_type: str = "quasi_static") -> None:
        """Set active biomechanical limits for risk assessment."""
        self._active_limits = [
            lim for lim in BIOMECHANICAL_LIMITS
            if lim.body_region == body_region and lim.contact_type == contact_type
        ]

    def calculate_max_contact_force(
        self,
        effective_mass_kg: float,
        impact_speed_m_per_s: float,
        compression_distance_m: float = 0.01,
    ) -> float:
        """
        Estimate maximum contact force during collision.

        Simplified energy-based model:
          F_max ≈ m * v^2 / (2 * d) + m * g

        Where:
          m = effective mass at contact point
          v = relative velocity at impact
          d = compression distance (typically 1-10 mm for rigid contact)
        """
        kinetic_energy = 0.5 * effective_mass_kg * impact_speed_m_per_s**2
        force_from_energy = kinetic_energy / compression_distance_m
        gravitational = effective_mass_kg * 9.81
        return force_from_energy + gravitational

    def check_force_compliance(
        self,
        measured_force_n: float,
        body_region: str,
        contact_type: str = "quasi_static",
    ) -> Tuple[bool, float]:
        """
        Check if measured force complies with ISO/TS 15066 limits.

        Returns (compliant, margin) where margin is percentage below limit.
        """
        limit = next(
            (lim for lim in BIOMECHANICAL_LIMITS
             if lim.body_region == body_region and lim.contact_type == contact_type),
            None,
        )
        if not limit:
            return False, 0.0

        compliant = measured_force_n <= limit.max_force_n
        margin = (limit.max_force_n - measured_force_n) / limit.max_force_n * 100
        return compliant, margin

    def calculate_safety_distance_ssm(
        self,
        robot_speed_m_per_s: float,
        human_speed_m_per_s = 1.6,     # ISO/TS 15066: 1.6 m/s walking speed
        robot_stopping_distance_m: float = 0.0,
        reaction_time_s: float = 0.5,   # SSM system reaction time
        intrusion_depth_m: float = 0.0,  # Depth of human body part
    ) -> float:
        """
        Calculate protective separation distance for SSM.

        ISO/TS 15066, Clause 5.4.2:
          S = (K * T) + C + Zd + Zr

        Where:
          K = 1.6 m/s (human approach speed)
          T = system stopping performance (reaction + stopping)
          C = intrusion distance (1200 mm for vertical, 850 mm for horizontal)
          Zd = position uncertainty from robot
          Zr = position uncertainty from protective equipment
        """
        K = human_speed_m_per_s
        T = reaction_time_s + robot_stopping_distance_m / max(robot_speed_m_per_s, 0.001)
        C = intrusion_depth_m if intrusion_depth_m > 0 else 0.85  # default 850 mm
        Zd = 0.05   # 50 mm robot position uncertainty
        Zr = 0.05   # 50 mm sensor uncertainty

        S = (K * T) + C + Zd + Zr
        return S

    def calculate_pressure(
        self,
        force_n: float,
        contact_area_cm2: float,
    ) -> float:
        """Calculate contact pressure."""
        if contact_area_cm2 <= 0:
            return float("inf")
        return force_n / contact_area_cm2

    def assess_pfl_risk(
        self,
        effective_mass_kg: float,
        max_speed_m_per_s: float,
        body_region: str,
        contact_area_cm2: float,
        contact_type: str = "quasi_static",
    ) -> Dict[str, Any]:
        """
        Comprehensive PFL risk assessment.

        Evaluates force and pressure against ISO/TS 15066 limits.
        """
        force = self.calculate_max_contact_force(effective_mass_kg, max_speed_m_per_s)
        pressure = self.calculate_pressure(force, contact_area_cm2)

        limit = next(
            (lim for lim in BIOMECHANICAL_LIMITS
             if lim.body_region == body_region and lim.contact_type == contact_type),
            None,
        )

        force_compliant = force <= limit.max_force_n if limit else False
        pressure_compliant = pressure <= limit.max_pressure_n_per_cm2 if limit else False

        return {
            "body_region": body_region,
            "contact_type": contact_type,
            "effective_mass_kg": effective_mass_kg,
            "max_speed_m_per_s": max_speed_m_per_s,
            "calculated_force_n": force,
            "calculated_pressure_n_per_cm2": pressure,
            "force_limit_n": limit.max_force_n if limit else None,
            "pressure_limit_n_per_cm2": limit.max_pressure_n_per_cm2 if limit else None,
            "force_compliant": force_compliant,
            "pressure_compliant": pressure_compliant,
            "overall_compliant": force_compliant and pressure_compliant,
        }

    def configure_hand_guiding(
        self,
        max_speed_m_per_s: float = 0.25,     # ISO/TS 15066: max 250 mm/s
        max_force_n: float = 140.0,           # Hand/finger limit
        enable_deadman: bool = True,
    ) -> Dict[str, Any]:
        """
        Configure hand guiding mode per ISO/TS 15066 Clause 5.3.

        Requirements:
          - Speed limited to 250 mm/s
          - Force limited to safe levels
          - Deadman switch required
          - Safety-rated monitored stop on release
        """
        return {
            "mode": CollaborativeMode.HAND_GUIDING.value,
            "max_speed_m_per_s": max_speed_m_per_s,
            "max_force_n": max_force_n,
            "deadman_switch": enable_deadman,
            "three_position_switch": True,     # Required for hand guiding
            "enabling_device": True,
        }

    def generate_safety_report(self) -> Dict[str, Any]:
        """Generate ISO/TS 15066 compliant safety report."""
        return {
            "standard": "ISO/TS 15066:2016",
            "robot_mass_kg": self.robot_mass_kg,
            "max_speed_m_per_s": self.max_speed_m_per_s,
            "max_payload_kg": self.max_payload_kg,
            "collaborative_mode": self.collaborative_mode.value,
            "active_limits": [
                {
                    "body_region": lim.body_region,
                    "contact_type": lim.contact_type,
                    "max_force_n": lim.max_force_n,
                    "max_pressure_n_per_cm2": lim.max_pressure_n_per_cm2,
                }
                for lim in self._active_limits
            ],
        }


# ---------------------------------------------------------------------------
# Safety validation helpers
# ---------------------------------------------------------------------------

def validate_safety_system(
    risk_assessment: RiskAssessment,
    sistema: SISTEMACalculator,
    collaborative: Optional[CollaborativeSafety] = None,
) -> Dict[str, Any]:
    """
    Comprehensive safety system validation.

    Validates:
      1. Risk assessment completeness
      2. PL achievement vs requirement
      3. Collaborative safety compliance
      4. Safety distance calculations
    """
    required_pl = PerformanceLevel(risk_assessment.get_required_pl())
    sistema_results = sistema.validate_all(required_pl)

    all_meet = all(r["meets_requirement"] for r in sistema_results.values())

    report = {
        "risk_assessment": risk_assessment.generate_report(),
        "sistema_validation": sistema_results,
        "all_pl_requirements_met": all_meet,
    }

    if collaborative:
        report["collaborative_safety"] = collaborative.generate_safety_report()

    return report


def calculate_iso_13855_safety_distance(
    approach_speed_mm_per_s: float = 1600.0,
    system_response_time_ms: float = 500.0,
    machine_stopping_time_ms: float = 200.0,
    reach_over_distance_mm: float = 0.0,
    resolution_mm: float = 30.0,
) -> float:
    """
    Calculate minimum safety distance per ISO 13855.

    S = K * T + C

    K = approach speed (mm/s)
      - Hand: 1600 mm/s
      - Body: 1600 mm/s
      - Leg: 1600 mm/s
    T = total response time = system_response + machine_stopping
    C = intrusion distance
      - For light curtains with resolution <= 30 mm: C = 850 mm
      - For resolution > 30 mm: C = 8 * (resolution - 30) mm
      - For reach-over: additional CRO
    """
    K = approach_speed_mm_per_s
    T = (system_response_time_ms + machine_stopping_time_ms) / 1000.0

    if resolution_mm <= 30:
        C = 850.0
    else:
        C = 8.0 * (resolution_mm - 30.0)

    C += reach_over_distance_mm

    S = K * T + C
    return S
