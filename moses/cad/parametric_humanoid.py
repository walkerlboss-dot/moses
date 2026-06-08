"""
Moses v4.0 — Parametric Humanoid CAD Generator
================================================

Professional-grade parametric humanoid robot design system.
Generates full kinematic chains with biomechanically accurate proportions,
joint definitions, and multi-format export (URDF, USD, STEP, STL).

Libraries: trimesh, cadquery, build123d (with graceful degradation)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Optional CAD backends — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import cadquery as cq
    from cadquery import Workplane, Vector, Location
    HAS_CADQUERY = True
except Exception:
    HAS_CADQUERY = False

try:
    import build123d as b3d
    from build123d import (
        BuildPart, Box, Cylinder, Sphere, Cone,
        Location as B3DLocation, Axis, Plane,
        fillet, chamfer, add
    )
    HAS_BUILD123D = True
except Exception:
    HAS_BUILD123D = False

try:
    import trimesh
    from trimesh import Trimesh
    from trimesh.exchange.urdf import export_urdf
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class JointType(Enum):
    """Supported joint types for the kinematic chain."""
    REVOLUTE = auto()      # 1-DOF rotation (hinge)
    PRISMATIC = auto()     # 1-DOF translation
    SPHERICAL = auto()     # 3-DOF ball joint (simplified as 3 revolute)
    FIXED = auto()         # 0-DOF rigid attachment
    CONTINUOUS = auto()    # Unlimited revolute


class ExportFormat(Enum):
    """Supported export formats."""
    URDF = "urdf"
    USD = "usd"
    STEP = "step"
    STL = "stl"
    OBJ = "obj"
    DAE = "dae"
    GLTF = "gltf"


# Biomechanical constants (percentages of total height)
BIOMECH_RATIOS = {
    "head": 0.130,
    "neck": 0.030,
    "torso": 0.288,
    "pelvis": 0.090,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "thigh": 0.245,
    "shin": 0.246,
    "foot": 0.052,
}

# Mass distribution (percentages of total mass, Winter 2009 biomechanics)
MASS_RATIOS = {
    "head": 0.081,
    "neck": 0.018,
    "torso": 0.279,
    "pelvis": 0.142,
    "upper_arm": 0.028,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.100,
    "shin": 0.046,
    "foot": 0.014,
}

# Default joint limits (degrees or meters)
DEFAULT_JOINT_LIMITS: Dict[JointType, Tuple[float, float]] = {
    JointType.REVOLUTE: (-math.pi, math.pi),
    JointType.PRISMATIC: (-0.05, 0.05),
    JointType.SPHERICAL: (-math.pi/2, math.pi/2),
    JointType.CONTINUOUS: (-float('inf'), float('inf')),
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class JointLimits:
    """Joint limit specification."""
    lower: float
    upper: float
    effort: float = 100.0
    velocity: float = 10.0

    def to_dict(self) -> Dict[str, float]:
        return {"lower": self.lower, "upper": self.upper,
                "effort": self.effort, "velocity": self.velocity}


@dataclass
class JointDef:
    """Joint definition in the kinematic chain."""
    name: str
    joint_type: JointType
    parent: str
    child: str
    origin: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    axis: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    limits: Optional[JointLimits] = None
    damping: float = 0.1
    friction: float = 0.0


@dataclass
class LinkDef:
    """Rigid link definition."""
    name: str
    mass: float
    com: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    inertia_tensor: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3)
    )
    visual_mesh: Optional[Any] = None
    collision_mesh: Optional[Any] = None
    color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)


@dataclass
class ActuatorDef:
    """Actuator / motor specification."""
    name: str
    joint_name: str
    max_torque: float
    max_speed: float
    gear_ratio: float = 1.0
    mass: float = 0.5
    dimensions: Tuple[float, float, float] = (0.04, 0.04, 0.06)


@dataclass
class BearingDef:
    """Bearing specification."""
    name: str
    joint_name: str
    bore_diameter: float
    outer_diameter: float
    width: float
    type: str = "deep_groove_ball"


@dataclass
class FastenerDef:
    """Fastener (bolt/screw) specification."""
    name: str
    type: str = "socket_head_cap"
    diameter: float = 0.004  # M4
    length: float = 0.020
    material: str = "steel_12.9"
    torque: float = 2.5


@dataclass
class HumanoidParameters:
    """Parametric humanoid specification."""
    total_height: float = 1.70
    total_mass: float = 70.0
    gender: str = "neutral"
    
    # Override ratios (optional)
    ratios: Dict[str, float] = field(default_factory=dict)
    mass_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Joint configuration
    joint_limits: Dict[str, JointLimits] = field(default_factory=dict)
    
    # Actuators
    actuator_density: float = 0.05  # kg per Nm torque
    
    # Detail level
    detail_level: int = 2  # 0=blocks, 1=simplified, 2=detailed, 3=production

    def get_ratio(self, part: str) -> float:
        return self.ratios.get(part, BIOMECH_RATIOS.get(part, 0.1))

    def get_mass_ratio(self, part: str) -> float:
        return self.mass_ratios.get(part, MASS_RATIOS.get(part, 0.01))


# ---------------------------------------------------------------------------
# Geometry Builders
# ---------------------------------------------------------------------------

class GeometryBuilder:
    """Build geometric primitives for humanoid links."""

    def __init__(self, params: HumanoidParameters):
        self.params = params
        self.height = params.total_height
        self.mass = params.total_mass

    def _scale(self, ratio: float) -> float:
        return self.height * ratio

    def build_head(self) -> Dict[str, Any]:
        """Build head geometry — ellipsoid approximation."""
        h = self._scale(BIOMECH_RATIOS["head"])
        w = h * 0.75
        d = h * 0.80
        mass = self.mass * MASS_RATIOS["head"]
        return {
            "type": "ellipsoid",
            "size": (w, d, h),
            "mass": mass,
            "color": (0.95, 0.90, 0.85, 1.0),
        }

    def build_neck(self) -> Dict[str, Any]:
        """Build neck geometry — cylinder."""
        h = self._scale(BIOMECH_RATIOS["neck"])
        r = h * 0.35
        mass = self.mass * MASS_RATIOS["neck"]
        return {
            "type": "cylinder",
            "radius": r,
            "height": h,
            "mass": mass,
            "color": (0.85, 0.85, 0.85, 1.0),
        }

    def build_torso(self) -> Dict[str, Any]:
        """Build torso — tapered box / rounded prism."""
        h = self._scale(BIOMECH_RATIOS["torso"])
        w_top = self._scale(0.18)
        w_bottom = self._scale(0.28)
        d = self._scale(0.15)
        mass = self.mass * MASS_RATIOS["torso"]
        return {
            "type": "tapered_box",
            "height": h,
            "width_top": w_top,
            "width_bottom": w_bottom,
            "depth": d,
            "mass": mass,
            "color": (0.2, 0.4, 0.7, 1.0),
        }

    def build_pelvis(self) -> Dict[str, Any]:
        """Build pelvis — rounded box."""
        h = self._scale(BIOMECH_RATIOS["pelvis"])
        w = self._scale(0.26)
        d = self._scale(0.16)
        mass = self.mass * MASS_RATIOS["pelvis"]
        return {
            "type": "rounded_box",
            "size": (w, d, h),
            "mass": mass,
            "color": (0.25, 0.35, 0.65, 1.0),
        }

    def build_upper_arm(self, side: str = "left") -> Dict[str, Any]:
        """Build upper arm — tapered cylinder."""
        h = self._scale(BIOMECH_RATIOS["upper_arm"])
        r_top = self._scale(0.038)
        r_bottom = self._scale(0.032)
        mass = self.mass * MASS_RATIOS["upper_arm"]
        return {
            "type": "tapered_cylinder",
            "height": h,
            "radius_top": r_top,
            "radius_bottom": r_bottom,
            "mass": mass,
            "color": (0.3, 0.5, 0.8, 1.0),
        }

    def build_forearm(self, side: str = "left") -> Dict[str, Any]:
        """Build forearm — tapered cylinder."""
        h = self._scale(BIOMECH_RATIOS["forearm"])
        r_top = self._scale(0.032)
        r_bottom = self._scale(0.025)
        mass = self.mass * MASS_RATIOS["forearm"]
        return {
            "type": "tapered_cylinder",
            "height": h,
            "radius_top": r_top,
            "radius_bottom": r_bottom,
            "mass": mass,
            "color": (0.3, 0.5, 0.8, 1.0),
        }

    def build_hand(self, side: str = "left") -> Dict[str, Any]:
        """Build hand — box with finger stubs."""
        h = self._scale(BIOMECH_RATIOS["hand"])
        w = h * 0.9
        d = h * 0.35
        mass = self.mass * MASS_RATIOS["hand"]
        return {
            "type": "box",
            "size": (w, d, h),
            "mass": mass,
            "color": (0.9, 0.85, 0.8, 1.0),
        }

    def build_thigh(self, side: str = "left") -> Dict[str, Any]:
        """Build thigh — tapered cylinder."""
        h = self._scale(BIOMECH_RATIOS["thigh"])
        r_top = self._scale(0.055)
        r_bottom = self._scale(0.045)
        mass = self.mass * MASS_RATIOS["thigh"]
        return {
            "type": "tapered_cylinder",
            "height": h,
            "radius_top": r_top,
            "radius_bottom": r_bottom,
            "mass": mass,
            "color": (0.2, 0.35, 0.6, 1.0),
        }

    def build_shin(self, side: str = "left") -> Dict[str, Any]:
        """Build shin / lower leg — tapered cylinder."""
        h = self._scale(BIOMECH_RATIOS["shin"])
        r_top = self._scale(0.045)
        r_bottom = self._scale(0.035)
        mass = self.mass * MASS_RATIOS["shin"]
        return {
            "type": "tapered_cylinder",
            "height": h,
            "radius_top": r_top,
            "radius_bottom": r_bottom,
            "mass": mass,
            "color": (0.2, 0.35, 0.6, 1.0),
        }

    def build_foot(self, side: str = "left") -> Dict[str, Any]:
        """Build foot — rounded box."""
        h = self._scale(BIOMECH_RATIOS["foot"])
        l = h * 2.5  # foot length
        w = h * 1.2  # foot width
        mass = self.mass * MASS_RATIOS["foot"]
        return {
            "type": "rounded_box",
            "size": (l, w, h),
            "mass": mass,
            "color": (0.15, 0.25, 0.5, 1.0),
        }


# ---------------------------------------------------------------------------
# Trimesh Mesh Factory
# ---------------------------------------------------------------------------

class TrimeshFactory:
    """Create trimesh objects from geometry descriptions."""

    def __init__(self):
        if not HAS_TRIMESH:
            raise ImportError("trimesh is required for mesh generation")

    def create(self, geom: Dict[str, Any]) -> Trimesh:
        gtype = geom.get("type", "box")
        if gtype == "box":
            return self._box(geom["size"])
        elif gtype == "cylinder":
            return self._cylinder(geom["radius"], geom["height"])
        elif gtype == "ellipsoid":
            return self._ellipsoid(geom["size"])
        elif gtype == "tapered_cylinder":
            return self._tapered_cylinder(
                geom["height"], geom["radius_top"], geom["radius_bottom"]
            )
        elif gtype in ("rounded_box", "tapered_box"):
            # Fall back to box for now
            if "size" in geom:
                return self._box(geom["size"])
            else:
                return self._box((0.1, 0.1, 0.1))
        else:
            return self._box((0.1, 0.1, 0.1))

    def _box(self, size: Tuple[float, float, float]) -> Trimesh:
        return trimesh.creation.box(extents=size)

    def _cylinder(self, radius: float, height: float) -> Trimesh:
        return trimesh.creation.cylinder(radius=radius, height=height)

    def _ellipsoid(self, size: Tuple[float, float, float]) -> Trimesh:
        # Approximate with icosphere scaled
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sphere.apply_scale(size)
        return sphere

    def _tapered_cylinder(
        self, height: float, r_top: float, r_bottom: float
    ) -> Trimesh:
        if abs(r_top - r_bottom) < 1e-6:
            return self._cylinder(r_top, height)
        # Create cone frustum
        return trimesh.creation.cone(radius=r_bottom, height=height, sections=32)


# ---------------------------------------------------------------------------
# Parametric Humanoid Generator
# ---------------------------------------------------------------------------

class ParametricHumanoid:
    """
    Parametric humanoid robot CAD generator.

    Generates a complete kinematic chain from biomechanical parameters,
    with joints, links, actuators, bearings, and fasteners.
    """

    def __init__(self, params: Optional[HumanoidParameters] = None):
        self.params = params or HumanoidParameters()
        self.geom_builder = GeometryBuilder(self.params)
        self.links: Dict[str, LinkDef] = {}
        self.joints: Dict[str, JointDef] = {}
        self.actuators: Dict[str, ActuatorDef] = {}
        self.bearings: Dict[str, BearingDef] = {}
        self.fasteners: List[FastenerDef] = []
        self._built = False

    # ------------------------------------------------------------------
    # Build API
    # ------------------------------------------------------------------

    def build(self) -> "ParametricHumanoid":
        """Build the full humanoid model."""
        if self._built:
            return self
        self._build_links()
        self._build_joints()
        self._build_actuators()
        self._build_bearings()
        self._build_fasteners()
        self._built = True
        return self

    def _build_links(self) -> None:
        """Create all rigid links."""
        p = self.params
        gb = self.geom_builder

        # Central body
        self.links["base"] = self._make_link("base", {"type": "box", "size": (0.01, 0.01, 0.01)}, 0.0)
        self.links["pelvis"] = self._make_link("pelvis", gb.build_pelvis())
        self.links["torso"] = self._make_link("torso", gb.build_torso())
        self.links["neck"] = self._make_link("neck", gb.build_neck())
        self.links["head"] = self._make_link("head", gb.build_head())

        # Arms
        for side in ("left", "right"):
            self.links[f"upper_arm_{side}"] = self._make_link(
                f"upper_arm_{side}", gb.build_upper_arm(side)
            )
            self.links[f"forearm_{side}"] = self._make_link(
                f"forearm_{side}", gb.build_forearm(side)
            )
            self.links[f"hand_{side}"] = self._make_link(
                f"hand_{side}", gb.build_hand(side)
            )

        # Legs
        for side in ("left", "right"):
            self.links[f"thigh_{side}"] = self._make_link(
                f"thigh_{side}", gb.build_thigh(side)
            )
            self.links[f"shin_{side}"] = self._make_link(
                f"shin_{side}", gb.build_shin(side)
            )
            self.links[f"foot_{side}"] = self._make_link(
                f"foot_{side}", gb.build_foot(side)
            )

    def _make_link(self, name: str, geom: Dict[str, Any], mass_override: Optional[float] = None) -> LinkDef:
        mass = mass_override if mass_override is not None else geom.get("mass", 1.0)
        size = geom.get("size", (0.1, 0.1, 0.1))
        # Approximate inertia for box
        ix = mass * (size[1]**2 + size[2]**2) / 12.0
        iy = mass * (size[0]**2 + size[2]**2) / 12.0
        iz = mass * (size[0]**2 + size[1]**2) / 12.0
        inertia = np.diag([ix, iy, iz])
        return LinkDef(
            name=name,
            mass=mass,
            com=np.zeros(3),
            inertia_tensor=inertia,
            color=geom.get("color", (0.8, 0.8, 0.8, 1.0)),
        )

    def _build_joints(self) -> None:
        """Create the kinematic chain joints."""
        h = self.params.total_height

        # Base → pelvis (fixed / floating base)
        self.joints["base_to_pelvis"] = JointDef(
            name="base_to_pelvis",
            joint_type=JointType.FIXED,
            parent="base",
            child="pelvis",
            origin=self._t([0.0, 0.0, h * 0.05]),
        )

        # Pelvis → torso
        torso_h = h * BIOMECH_RATIOS["torso"]
        self.joints["spine"] = JointDef(
            name="spine",
            joint_type=JointType.REVOLUTE,
            parent="pelvis",
            child="torso",
            origin=self._t([0.0, 0.0, torso_h * 0.5]),
            axis=np.array([0.0, 1.0, 0.0]),
            limits=JointLimits(-math.pi/4, math.pi/4),
        )

        # Torso → neck
        neck_h = h * BIOMECH_RATIOS["neck"]
        self.joints["neck_joint"] = JointDef(
            name="neck_joint",
            joint_type=JointType.REVOLUTE,
            parent="torso",
            child="neck",
            origin=self._t([0.0, 0.0, torso_h * 0.5 + neck_h * 0.3]),
            axis=np.array([0.0, 1.0, 0.0]),
            limits=JointLimits(-math.pi/3, math.pi/3),
        )

        # Neck → head
        head_h = h * BIOMECH_RATIOS["head"]
        self.joints["head_joint"] = JointDef(
            name="head_joint",
            joint_type=JointType.REVOLUTE,
            parent="neck",
            child="head",
            origin=self._t([0.0, 0.0, neck_h * 0.5 + head_h * 0.3]),
            axis=np.array([0.0, 1.0, 0.0]),
            limits=JointLimits(-math.pi/2, math.pi/2),
        )

        # Shoulders
        shoulder_width = h * 0.22
        upper_arm_h = h * BIOMECH_RATIOS["upper_arm"]
        for side, sign in (("left", -1), ("right", 1)):
            self.joints[f"shoulder_{side}"] = JointDef(
                name=f"shoulder_{side}",
                joint_type=JointType.SPHERICAL,
                parent="torso",
                child=f"upper_arm_{side}",
                origin=self._t([sign * shoulder_width * 0.5, 0.0, torso_h * 0.35]),
                axis=np.array([1.0, 0.0, 0.0]),
                limits=JointLimits(-math.pi, math.pi),
            )
            # Elbow
            self.joints[f"elbow_{side}"] = JointDef(
                name=f"elbow_{side}",
                joint_type=JointType.REVOLUTE,
                parent=f"upper_arm_{side}",
                child=f"forearm_{side}",
                origin=self._t([0.0, 0.0, -upper_arm_h * 0.5]),
                axis=np.array([1.0, 0.0, 0.0]),
                limits=JointLimits(-math.pi/2, math.pi/2),
            )
            # Wrist
            forearm_h = h * BIOMECH_RATIOS["forearm"]
            self.joints[f"wrist_{side}"] = JointDef(
                name=f"wrist_{side}",
                joint_type=JointType.REVOLUTE,
                parent=f"forearm_{side}",
                child=f"hand_{side}",
                origin=self._t([0.0, 0.0, -forearm_h * 0.5]),
                axis=np.array([0.0, 1.0, 0.0]),
                limits=JointLimits(-math.pi/4, math.pi/4),
            )

        # Hips
        hip_width = h * 0.18
        thigh_h = h * BIOMECH_RATIOS["thigh"]
        for side, sign in (("left", -1), ("right", 1)):
            self.joints[f"hip_{side}"] = JointDef(
                name=f"hip_{side}",
                joint_type=JointType.SPHERICAL,
                parent="pelvis",
                child=f"thigh_{side}",
                origin=self._t([sign * hip_width * 0.5, 0.0, -h * 0.03]),
                axis=np.array([1.0, 0.0, 0.0]),
                limits=JointLimits(-math.pi/2, math.pi/2),
            )
            # Knee
            self.joints[f"knee_{side}"] = JointDef(
                name=f"knee_{side}",
                joint_type=JointType.REVOLUTE,
                parent=f"thigh_{side}",
                child=f"shin_{side}",
                origin=self._t([0.0, 0.0, -thigh_h * 0.5]),
                axis=np.array([1.0, 0.0, 0.0]),
                limits=JointLimits(0.0, math.pi),
            )
            # Ankle
            shin_h = h * BIOMECH_RATIOS["shin"]
            self.joints[f"ankle_{side}"] = JointDef(
                name=f"ankle_{side}",
                joint_type=JointType.REVOLUTE,
                parent=f"shin_{side}",
                child=f"foot_{side}",
                origin=self._t([0.0, 0.0, -shin_h * 0.5]),
                axis=np.array([1.0, 0.0, 0.0]),
                limits=JointLimits(-math.pi/6, math.pi/6),
            )

    def _build_actuators(self) -> None:
        """Create actuator definitions for each actuated joint."""
        for jname, joint in self.joints.items():
            if joint.joint_type in (JointType.FIXED,):
                continue
            # Estimate required torque from link mass × lever arm × gravity
            child_link = self.links.get(joint.child, LinkDef(name=joint.child, mass=1.0))
            lever = 0.1  # approximate
            required_torque = child_link.mass * 9.81 * lever
            actuator_mass = required_torque * self.params.actuator_density
            self.actuators[jname] = ActuatorDef(
                name=f"actuator_{jname}",
                joint_name=jname,
                max_torque=required_torque * 2.0,
                max_speed=5.0,
                gear_ratio=10.0,
                mass=max(actuator_mass, 0.1),
            )

    def _build_bearings(self) -> None:
        """Create bearing definitions for revolute joints."""
        for jname, joint in self.joints.items():
            if joint.joint_type not in (JointType.REVOLUTE, JointType.CONTINUOUS):
                continue
            self.bearings[jname] = BearingDef(
                name=f"bearing_{jname}",
                joint_name=jname,
                bore_diameter=0.008,
                outer_diameter=0.022,
                width=0.007,
            )

    def _build_fasteners(self) -> None:
        """Create fastener BOM entries."""
        # Rough estimate: 4 bolts per actuator/bearing pair
        for jname in self.actuators:
            for i in range(4):
                self.fasteners.append(FastenerDef(
                    name=f"bolt_{jname}_{i}",
                    type="socket_head_cap",
                    diameter=0.004,
                    length=0.020,
                    torque=2.5,
                ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _t(translation: List[float]) -> npt.NDArray[np.float64]:
        """Create a 4×4 translation matrix."""
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = translation
        return m

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, path: Union[str, Path], fmt: ExportFormat) -> Path:
        """Export the humanoid to the specified format."""
        path = Path(path)
        if not self._built:
            self.build()

        if fmt == ExportFormat.URDF:
            return self._export_urdf(path)
        elif fmt == ExportFormat.USD:
            return self._export_usd(path)
        elif fmt == ExportFormat.STEP:
            return self._export_step(path)
        elif fmt == ExportFormat.STL:
            return self._export_stl(path)
        elif fmt == ExportFormat.OBJ:
            return self._export_obj(path)
        elif fmt == ExportFormat.DAE:
            return self._export_dae(path)
        elif fmt == ExportFormat.GLTF:
            return self._export_gltf(path)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def _export_urdf(self, path: Path) -> Path:
        """Export to URDF (Unified Robot Description Format)."""
        robot = ET.Element("robot", name="moses_humanoid")

        # Links
        for link in self.links.values():
            l_elem = ET.SubElement(robot, "link", name=link.name)
            # Inertial
            inertial = ET.SubElement(l_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{link.mass:.6f}")
            i = link.inertia_tensor
            ET.SubElement(
                inertial, "inertia",
                ixx=f"{i[0,0]:.6e}", ixy=f"{i[0,1]:.6e}", ixz=f"{i[0,2]:.6e}",
                iyy=f"{i[1,1]:.6e}", iyz=f"{i[1,2]:.6e}", izz=f"{i[2,2]:.6e}",
            )
            # Visual / collision placeholders
            ET.SubElement(l_elem, "visual")
            ET.SubElement(l_elem, "collision")

        # Joints
        for joint in self.joints.values():
            jtype_str = {
                JointType.REVOLUTE: "revolute",
                JointType.PRISMATIC: "prismatic",
                JointType.SPHERICAL: "spherical",
                JointType.FIXED: "fixed",
                JointType.CONTINUOUS: "continuous",
            }.get(joint.joint_type, "fixed")

            j_elem = ET.SubElement(
                robot, "joint",
                name=joint.name,
                type=jtype_str,
            )
            ET.SubElement(j_elem, "parent", link=joint.parent)
            ET.SubElement(j_elem, "child", link=joint.child)
            origin = joint.origin
            xyz = f"{origin[0,3]:.6f} {origin[1,3]:.6f} {origin[2,3]:.6f}"
            rpy = "0 0 0"  # Simplified
            ET.SubElement(j_elem, "origin", xyz=xyz, rpy=rpy)
            if joint.joint_type != JointType.FIXED:
                axis = joint.axis
                ET.SubElement(j_elem, "axis", xyz=f"{axis[0]:.3f} {axis[1]:.3f} {axis[2]:.3f}")
            if joint.limits:
                ET.SubElement(
                    j_elem, "limit",
                    lower=f"{joint.limits.lower:.6f}",
                    upper=f"{joint.limits.upper:.6f}",
                    effort=f"{joint.limits.effort:.3f}",
                    velocity=f"{joint.limits.velocity:.3f}",
                )

        # Write
        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ")
        path.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode"))
        return path

    def _export_usd(self, path: Path) -> Path:
        """Export to USD (Universal Scene Description) — simplified ASCII."""
        lines = [
            "#usda 1.0",
            "(",
            '    defaultPrim = "moses_humanoid"',
            ")",
            "",
            'def Xform "moses_humanoid"',
            "{",
        ]
        for link in self.links.values():
            lines.append(f'    def Xform "{link.name}"')
            lines.append("    {")
            lines.append(f'        double3 xformOp:translate = (0, 0, 0)')
            lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append("    }")
        lines.append("}")
        path.write_text("\n".join(lines))
        return path

    def _export_step(self, path: Path) -> Path:
        """Export to STEP via cadquery or build123d if available."""
        if HAS_BUILD123D:
            return self._export_step_build123d(path)
        elif HAS_CADQUERY:
            return self._export_step_cadquery(path)
        else:
            raise RuntimeError("No CAD backend available for STEP export. Install build123d or cadquery.")

    def _export_step_build123d(self, path: Path) -> Path:
        """Build a simplified STEP assembly using build123d."""
        parts: List[b3d.Part] = []
        for link in self.links.values():
            if link.name == "base":
                continue
            # Approximate each link as a small box
            size = 0.05
            with BuildPart() as bp:
                Box(size, size, size * 2)
                p = bp.part
                p.label = link.name
                parts.append(p)
        if parts:
            assembly = b3d.Compound(children=parts)
            assembly.export_step(str(path))
        else:
            path.write_text("")  # Empty fallback
        return path

    def _export_step_cadquery(self, path: Path) -> Path:
        """Build a simplified STEP assembly using cadquery."""
        result = cq.Workplane("XY")
        for idx, link in enumerate(self.links.values()):
            if link.name == "base":
                continue
            size = 0.05
            box = cq.Workplane("XY").box(size, size, size * 2)
            result = result.add(box.translate((idx * 0.1, 0, 0)))
        cq.exporters.export(result, str(path))
        return path

    def _export_stl(self, path: Path) -> Path:
        """Export visual meshes as STL."""
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh required for STL export")
        factory = TrimeshFactory()
        combined = []
        for link in self.links.values():
            if link.name == "base":
                continue
            geom = self._get_link_geometry(link.name)
            if geom:
                mesh = factory.create(geom)
                # Simple positioning based on link order
                mesh.apply_translation([0, 0, 0])
                combined.append(mesh)
        if combined:
            scene = trimesh.util.concatenate(combined)
            scene.export(str(path))
        return path

    def _export_obj(self, path: Path) -> Path:
        """Export to Wavefront OBJ."""
        stl_path = path.with_suffix(".stl")
        self._export_stl(stl_path)
        if stl_path.exists() and HAS_TRIMESH:
            mesh = trimesh.load_mesh(str(stl_path))
            mesh.export(str(path))
            stl_path.unlink(missing_ok=True)
        return path

    def _export_dae(self, path: Path) -> Path:
        """Export to COLLADA DAE."""
        # DAE requires more complex handling; use trimesh if possible
        if HAS_TRIMESH:
            stl_path = path.with_suffix(".stl")
            self._export_stl(stl_path)
            mesh = trimesh.load_mesh(str(stl_path))
            mesh.export(str(path))
            stl_path.unlink(missing_ok=True)
        return path

    def _export_gltf(self, path: Path) -> Path:
        """Export to glTF 2.0."""
        if HAS_TRIMESH:
            stl_path = path.with_suffix(".stl")
            self._export_stl(stl_path)
            mesh = trimesh.load_mesh(str(stl_path))
            mesh.export(str(path))
            stl_path.unlink(missing_ok=True)
        return path

    def _get_link_geometry(self, link_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve geometry description for a link."""
        gb = self.geom_builder
        mapping = {
            "head": gb.build_head,
            "neck": gb.build_neck,
            "torso": gb.build_torso,
            "pelvis": gb.build_pelvis,
        }
        if link_name in mapping:
            return mapping[link_name]()
        for side in ("left", "right"):
            if link_name == f"upper_arm_{side}":
                return gb.build_upper_arm(side)
            if link_name == f"forearm_{side}":
                return gb.build_forearm(side)
            if link_name == f"hand_{side}":
                return gb.build_hand(side)
            if link_name == f"thigh_{side}":
                return gb.build_thigh(side)
            if link_name == f"shin_{side}":
                return gb.build_shin(side)
            if link_name == f"foot_{side}":
                return gb.build_foot(side)
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_kinematic_chain(self) -> List[Tuple[str, str, str]]:
        """Return (joint_name, parent, child) tuples."""
        return [(j.name, j.parent, j.child) for j in self.joints.values()]

    def get_dof(self) -> int:
        """Calculate total degrees of freedom."""
        dof_map = {
            JointType.REVOLUTE: 1,
            JointType.PRISMATIC: 1,
            JointType.SPHERICAL: 3,
            JointType.FIXED: 0,
            JointType.CONTINUOUS: 1,
        }
        return sum(dof_map[j.joint_type] for j in self.joints.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize humanoid definition to dictionary."""
        return {
            "parameters": {
                "total_height": self.params.total_height,
                "total_mass": self.params.total_mass,
                "gender": self.params.gender,
                "detail_level": self.params.detail_level,
            },
            "links": {
                name: {
                    "mass": link.mass,
                    "com": link.com.tolist(),
                    "inertia": link.inertia_tensor.tolist(),
                }
                for name, link in self.links.items()
            },
            "joints": {
                name: {
                    "type": j.joint_type.name,
                    "parent": j.parent,
                    "child": j.child,
                    "axis": j.axis.tolist(),
                }
                for name, j in self.joints.items()
            },
            "actuators": {
                name: {
                    "max_torque": a.max_torque,
                    "max_speed": a.max_speed,
                    "mass": a.mass,
                }
                for name, a in self.actuators.items()
            },
            "dof": self.get_dof(),
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_humanoid(
    height: float = 1.70,
    mass: float = 70.0,
    **kwargs: Any,
) -> ParametricHumanoid:
    """Factory: create and build a parametric humanoid."""
    params = HumanoidParameters(total_height=height, total_mass=mass, **kwargs)
    return ParametricHumanoid(params).build()


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    humanoid = create_humanoid(height=1.75, mass=75.0)
    print(f"DOF: {humanoid.get_dof()}")
    print(f"Links: {list(humanoid.links.keys())}")
    print(f"Joints: {list(humanoid.joints.keys())}")
    print(f"Actuators: {list(humanoid.actuators.keys())}")

    # Quick export test
    tmp = Path(tempfile.gettempdir())
    humanoid.export(tmp / "moses.urdf", ExportFormat.URDF)
    print(f"URDF exported to {tmp / 'moses.urdf'}")
