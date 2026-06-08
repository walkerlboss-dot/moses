"""
moses/design/biomechanics.py
Human biomechanics reference and scaling for humanoid robots.

Sources:
- Winter, D.A. (2009) Biomechanics and Motor Control of Human Movement
- NASA Anthropometric Source Book (1978)
- OpenSim / Stanford NMBL gait data
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Joint Range of Motion (ROM) database — degrees
# ---------------------------------------------------------------------------

@dataclass
class JointROM:
    """Range of motion for a single degree of freedom."""
    name: str
    min_deg: float
    max_deg: float
    nominal_deg: float = 0.0
    # Human average; robot target may differ


# Values from clinical / biomechanics literature (healthy adult male)
HUMAN_ROM: Dict[str, JointROM] = {
    # ---- Hip ----
    "hip_flexion":        JointROM("Hip Flexion",        -30,  120, 0),
    "hip_extension":      JointROM("Hip Extension",      -30,   30, 0),
    "hip_abduction":      JointROM("Hip Abduction",      -45,   45, 0),
    "hip_adduction":      JointROM("Hip Adduction",      -30,   30, 0),
    "hip_internal_rot":   JointROM("Hip Internal Rot",   -45,   45, 0),
    "hip_external_rot":   JointROM("Hip External Rot",   -45,   45, 0),

    # ---- Knee ----
    "knee_flexion":       JointROM("Knee Flexion",         0,  140, 0),

    # ---- Ankle ----
    "ankle_dorsiflex":    JointROM("Ankle Dorsiflexion", -20,   30, 0),
    "ankle_plantarflex":  JointROM("Ankle Plantarflex",  -45,   45, 0),
    "ankle_inversion":    JointROM("Ankle Inversion",    -20,   30, 0),
    "ankle_eversion":     JointROM("Ankle Eversion",     -30,   20, 0),

    # ---- Spine / Trunk ----
    "lumbar_flexion":     JointROM("Lumbar Flexion",     -60,   60, 0),
    "lumbar_extension":   JointROM("Lumbar Extension",   -35,   35, 0),
    "lumbar_lateral":     JointROM("Lumbar Lateral Bend",-35,   35, 0),
    "lumbar_rotation":    JointROM("Lumbar Rotation",    -45,   45, 0),

    # ---- Shoulder ----
    "shoulder_flexion":   JointROM("Shoulder Flexion",  -180,  180, 0),
    "shoulder_abduction": JointROM("Shoulder Abduction", -180,  180, 0),
    "shoulder_int_rot":   JointROM("Shoulder Int Rot",   -90,   90, 0),

    # ---- Elbow ----
    "elbow_flexion":      JointROM("Elbow Flexion",        0,  150, 0),
    "elbow_pronation":    JointROM("Elbow Pronation",    -90,   90, 0),

    # ---- Wrist ----
    "wrist_flexion":      JointROM("Wrist Flexion",      -80,   80, 0),
    "wrist_deviation":    JointROM("Wrist Deviation",    -30,   30, 0),
}


def robot_rom_scale(human_rom: JointROM,
                    safety_margin: float = 0.85) -> JointROM:
    """
    Scale human ROM to robot target.
    Robots typically limit ROM to avoid self-collision and cable wrap.
    """
    span = human_rom.max_deg - human_rom.min_deg
    new_span = span * safety_margin
    center = (human_rom.max_deg + human_rom.min_deg) / 2.0
    return JointROM(
        name=human_rom.name + " (Robot)",
        min_deg=center - new_span / 2.0,
        max_deg=center + new_span / 2.0,
        nominal_deg=human_rom.nominal_deg,
    )


# ---------------------------------------------------------------------------
# Muscle force curves (Hill-type simplified)
# ---------------------------------------------------------------------------

class MuscleCurve:
    """
    Simplified Hill-type muscle model.
    F = F_max · f_l(l) · f_v(v) · activation
    """
    def __init__(self, f_max: float, l_opt: float, l_slack: float,
                 v_max: float):
        """
        Args:
            f_max: Maximum isometric force [N]
            l_opt: Optimal fiber length [m]
            l_slack: Tendon slack length [m]
            v_max: Maximum contraction velocity [m/s]
        """
        self.f_max = f_max
        self.l_opt = l_opt
        self.l_slack = l_slack
        self.v_max = v_max

    def force_length(self, l_m: float) -> float:
        """
        Gaussian force-length relationship.
        f_l = exp( -((l_m/l_opt - 1)/0.5)² )
        """
        x = l_m / self.l_opt - 1.0
        return np.exp(-(x / 0.5) ** 2)

    def force_velocity(self, v_m: float) -> float:
        """
        Hyperbolic force-velocity (simplified).
        v_m > 0  shortening
        v_m < 0  lengthening
        """
        if v_m >= 0.0:
            # Shortening
            return (self.v_max - v_m) / (self.v_max + 2.5 * v_m)
        else:
            # Lengthening — eccentric, stronger
            return 1.5 + 1.0 * np.tanh(-v_m / (0.2 * self.v_max))

    def force(self, l_m: float, v_m: float, activation: float = 1.0) -> float:
        return self.f_max * self.force_length(l_m) * self.force_velocity(v_m) * activation


# Representative human muscle data (from OpenSim / literature)
HUMAN_MUSCLES: Dict[str, MuscleCurve] = {
    "rectus_femoris": MuscleCurve(f_max=1200.0, l_opt=0.10, l_slack=0.30, v_max=1.2),
    "vastus_lateralis": MuscleCurve(f_max=2500.0, l_opt=0.09, l_slack=0.15, v_max=1.0),
    "biceps_femoris": MuscleCurve(f_max=1100.0, l_opt=0.11, l_slack=0.28, v_max=1.1),
    "gastrocnemius": MuscleCurve(f_max=1500.0, l_opt=0.05, l_slack=0.35, v_max=0.8),
    "soleus": MuscleCurve(f_max=3500.0, l_opt=0.04, l_slack=0.22, v_max=0.4),
    "tibialis_anterior": MuscleCurve(f_max=800.0, l_opt=0.07, l_slack=0.20, v_max=1.0),
    "gluteus_maximus": MuscleCurve(f_max=2500.0, l_opt=0.14, l_slack=0.10, v_max=1.2),
    "iliopsoas": MuscleCurve(f_max=1200.0, l_opt=0.10, l_slack=0.08, v_max=1.5),
}


def scale_muscle_to_actuator(muscle: MuscleCurve,
                             scale_force: float = 1.0,
                             scale_length: float = 1.0) -> MuscleCurve:
    """
    Scale muscle parameters to robot actuator equivalents.
    scale_force: torque / force multiplier (gear ratio effect)
    scale_length: link length scale
    """
    return MuscleCurve(
        f_max=muscle.f_max * scale_force,
        l_opt=muscle.l_opt * scale_length,
        l_slack=muscle.l_slack * scale_length,
        v_max=muscle.v_max * scale_length,
    )


# ---------------------------------------------------------------------------
# Gait cycle analysis
# ---------------------------------------------------------------------------

@dataclass
class GaitCycle:
    """Normalized gait cycle parameters (0–100%)."""
    stride_length: float       # [m]
    stride_time: float         # [s]
    cadence: float             # steps / min
    walking_speed: float       # [m/s]
    duty_factor: float         # stance / total cycle

    def step_length(self) -> float:
        return self.stride_length / 2.0

    def step_time(self) -> float:
        return self.stride_time / 2.0

    def stance_time(self) -> float:
        return self.stride_time * self.duty_factor

    def swing_time(self) -> float:
        return self.stride_time * (1.0 - self.duty_factor)


# Representative gait data for 1.75 m adult at various speeds
GAIT_PRESETS: Dict[str, GaitCycle] = {
    "slow_walk": GaitCycle(
        stride_length=1.10, stride_time=1.20, cadence=50.0,
        walking_speed=0.92, duty_factor=0.65,
    ),
    "normal_walk": GaitCycle(
        stride_length=1.40, stride_time=1.05, cadence=57.0,
        walking_speed=1.33, duty_factor=0.62,
    ),
    "fast_walk": GaitCycle(
        stride_length=1.70, stride_time=0.95, cadence=63.0,
        walking_speed=1.79, duty_factor=0.58,
    ),
    "slow_run": GaitCycle(
        stride_length=2.20, stride_time=0.80, cadence=75.0,
        walking_speed=2.75, duty_factor=0.40,
    ),
}


def gait_cycle_percent(phase: float) -> str:
    """Map 0–100% to gait phase name."""
    if phase < 10:
        return "initial_contact"
    elif phase < 30:
        return "loading_response"
    elif phase < 50:
        return "mid_stance"
    elif phase < 60:
        return "terminal_stance"
    elif phase < 73:
        return "pre_swing"
    elif phase < 87:
        return "initial_swing"
    elif phase < 100:
        return "mid_swing"
    return "terminal_swing"


# ---------------------------------------------------------------------------
# Energy efficiency models
# ---------------------------------------------------------------------------

@dataclass
class EnergyModel:
    """Metabolic / electrical energy models for locomotion."""
    mass: float          # total mass [kg]
    height: float        # stature [m]

    def froude_number(self, v: float) -> float:
        """Fr = v² / (g·L) — dimensionless speed."""
        return v ** 2 / (9.80665 * self.height)

    def preferred_walking_speed(self) -> float:
        """Humans prefer ~0.5 sqrt(g·L)."""
        return 0.5 * np.sqrt(9.80665 * self.height)

    def metabolic_cost_walking(self, v: float) -> float:
        """
        Margaria / Pandolf approx: Cw [W/kg] = 2.3 + 0.32·(v-0.7)²
        Returns total metabolic power [W].
        """
        c = 2.3 + 0.32 * (v - 0.7) ** 2
        return c * self.mass

    def metabolic_cost_running(self, v: float) -> float:
        """
        Running cost ~ linear with speed: Cr = 3.8·v  [J/(kg·m)]
        Power = Cr · v · m
        """
        cr = 3.8  # J/(kg·m)
        return cr * v * self.mass

    def electrical_equivalent(self, metabolic_w: float,
                              muscle_efficiency: float = 0.25,
                              motor_efficiency: float = 0.90) -> float:
        """
        Convert metabolic power to electrical motor power.
        """
        mechanical = metabolic_w * muscle_efficiency
        electrical = mechanical / motor_efficiency
        return electrical


# ---------------------------------------------------------------------------
# Scaling laws — human to robot
# ---------------------------------------------------------------------------

@dataclass
class Anthropometry:
    """Segment lengths and masses for a human/robot."""
    total_mass: float
    total_height: float
    # Segment lengths as fraction of height
    head: float = 0.13
    trunk: float = 0.30
    upper_arm: float = 0.19
    forearm: float = 0.15
    hand: float = 0.11
    thigh: float = 0.24
    shank: float = 0.25
    foot: float =0.15

    def segment_lengths(self) -> Dict[str, float]:
        h = self.total_height
        return {
            "head": self.head * h,
            "trunk": self.trunk * h,
            "upper_arm": self.upper_arm * h,
            "forearm": self.forearm * h,
            "hand": self.hand * h,
            "thigh": self.thigh * h,
            "shank": self.shank * h,
            "foot": self.foot * h,
        }

    def segment_masses(self) -> Dict[str, float]:
        """
        Winter (2009) segment mass fractions.
        """
        m = self.total_mass
        return {
            "head": 0.081 * m,
            "trunk": 0.497 * m,
            "upper_arm": 0.028 * m,
            "forearm": 0.016 * m,
            "hand": 0.006 * m,
            "thigh": 0.100 * m,
            "shank": 0.0465 * m,
            "foot": 0.0145 * m,
        }

    def center_of_mass_height(self) -> float:
        """Whole-body CoM ~ 0.56 of stature (standing)."""
        return 0.56 * self.total_height


def scale_human_to_robot(human: Anthropometry,
                         target_height: float,
                         target_mass: float) -> Anthropometry:
    """
    Scale human proportions to robot dimensions.
    Uses geometric scaling (lengths ∝ height, masses ∝ mass).
    """
    scale_l = target_height / human.total_height
    scale_m = target_mass / human.total_mass
    return Anthropometry(
        total_mass=target_mass,
        total_height=target_height,
        head=human.head,
        trunk=human.trunk,
        upper_arm=human.upper_arm,
        forearm=human.forearm,
        hand=human.hand,
        thigh=human.thigh,
        shank=human.shank,
        foot=human.foot,
    )


# ---------------------------------------------------------------------------
# Convenience: full robot spec generator
# ---------------------------------------------------------------------------

def generate_robot_spec(mass: float = 80.0,
                        height: float = 1.75) -> Dict:
    """
    Generate a complete robot biomechanical specification.
    """
    human_ref = Anthropometry(total_mass=70.0, total_height=1.75)
    robot = scale_human_to_robot(human_ref, height, mass)

    roms = {k: robot_rom_scale(v) for k, v in HUMAN_ROM.items()}
    muscles = {k: scale_muscle_to_actuator(v,
                scale_force=1.0, scale_length=height/1.75)
                for k, v in HUMAN_MUSCLES.items()}

    energy = EnergyModel(mass=mass, height=height)

    return {
        "anthropometry": robot,
        "segment_lengths": robot.segment_lengths(),
        "segment_masses": robot.segment_masses(),
        "rom": roms,
        "muscle_actuators": muscles,
        "energy": energy,
        "gait_normal": GAIT_PRESETS["normal_walk"],
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MOSES v4.0 — Biomechanics Demo")
    print("=" * 60)

    spec = generate_robot_spec(mass=80.0, height=1.75)

    print("\n--- Segment Lengths ---")
    for seg, length in spec["segment_lengths"].items():
        print(f"  {seg:12s}: {length:.3f} m")

    print("\n--- Joint ROM (Robot, with safety margin) ---")
    for name, rom in list(spec["rom"].items())[:6]:
        print(f"  {rom.name:25s}: {rom.min_deg:6.1f}° to {rom.max_deg:6.1f}°")

    print("\n--- Gait: Normal Walk ---")
    g = spec["gait_normal"]
    print(f"  Speed: {g.walking_speed:.2f} m/s")
    print(f"  Step length: {g.step_length():.2f} m")
    print(f"  Stance time: {g.stance_time():.2f} s")
    print(f"  Swing time: {g.swing_time():.2f} s")

    print("\n--- Energy at preferred speed ---")
    em = spec["energy"]
    v_pref = em.preferred_walking_speed()
    p_met = em.metabolic_cost_walking(v_pref)
    p_elec = em.electrical_equivalent(p_met)
    print(f"  Preferred speed: {v_pref:.2f} m/s")
    print(f"  Metabolic power: {p_met:.1f} W")
    print(f"  Electrical equiv: {p_elec:.1f} W")
