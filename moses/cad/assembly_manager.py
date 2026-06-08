"""
Moses v4.0 — Assembly Manager
=============================

Bill of Materials (BOM), assembly instructions, interference detection,
mass property calculation, and tolerance analysis.

Libraries: trimesh, numpy, scipy (optional)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import numpy.typing as npt

try:
    import trimesh
    from trimesh import Trimesh
    HAS_TRIMESH = True
except Exception as exc:
    HAS_TRIMESH = False
    logging.warning("trimesh not available: %s", exc)

try:
    from scipy.spatial import ConvexHull, cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class BOMCategory(Enum):
    """Category for BOM line items."""
    STRUCTURAL = "structural"
    ACTUATOR = "actuator"
    BEARING = "bearing"
    FASTENER = "fastener"
    ELECTRONIC = "electronic"
    SENSOR = "sensor"
    CABLE = "cable"
    FINISH = "finish"
    OTHER = "other"


@dataclass
class BOMItem:
    """Single line item in a Bill of Materials."""
    part_number: str
    description: str
    category: BOMCategory
    quantity: int
    unit: str = "ea"
    material: str = ""
    mass_kg: float = 0.0
    supplier: str = ""
    cost_usd: float = 0.0
    lead_time_days: int = 0
    notes: str = ""
    cad_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class AssemblyStep:
    """Single step in assembly instructions."""
    step_number: int
    title: str
    description: str
    parts: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    torque_specs: Dict[str, float] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    estimated_time_min: float = 5.0
    warnings: List[str] = field(default_factory=list)
    prerequisites: List[int] = field(default_factory=list)


@dataclass
class MassProperties:
    """Calculated mass properties for a link or assembly."""
    total_mass: float
    center_of_mass: npt.NDArray[np.float64]
    inertia_tensor: npt.NDArray[np.float64]  # 3×3 about CoM
    principal_moments: npt.NDArray[np.float64]
    principal_axes: npt.NDArray[np.float64]  # 3×3 rotation matrix
    bounding_box: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mass": self.total_mass,
            "center_of_mass": self.center_of_mass.tolist(),
            "inertia_tensor": self.inertia_tensor.tolist(),
            "principal_moments": self.principal_moments.tolist(),
            "principal_axes": self.principal_axes.tolist(),
            "bounding_box": [
                self.bounding_box[0].tolist(),
                self.bounding_box[1].tolist(),
            ],
            "volume": self.volume,
        }


@dataclass
class InterferenceReport:
    """Report of detected interferences between parts."""
    part_a: str
    part_b: str
    intersection_volume: float
    contact_points: List[npt.NDArray[np.float64]]
    severity: str  # "critical", "warning", "info"
    recommended_action: str


@dataclass
class ToleranceSpec:
    """Geometric tolerance specification."""
    dimension: str
    nominal: float
    upper: float
    lower: float
    tolerance_grade: str = "ISO_2768_m"


@dataclass
class ToleranceAnalysis:
    """Result of tolerance stack-up analysis."""
    dimension_chain: List[ToleranceSpec]
    nominal_total: float
    worst_case_upper: float
    worst_case_lower: float
    rss_upper: float
    rss_lower: float
    cpk_estimate: float


# ---------------------------------------------------------------------------
# Assembly Manager
# ---------------------------------------------------------------------------

class AssemblyManager:
    """
    Manage the full assembly lifecycle of a humanoid robot:

    - BOM generation from link/actuator/bearing/fastener definitions
    - Step-by-step assembly instructions
    - Interference detection between components
    - Mass property aggregation (CoM, inertia tensor)
    - Tolerance stack-up analysis
    """

    def __init__(self):
        self.bom_items: List[BOMItem] = []
        self.assembly_steps: List[AssemblyStep] = []
        self.mass_cache: Dict[str, MassProperties] = {}
        self.interference_cache: List[InterferenceReport] = []
        self.tolerance_specs: Dict[str, ToleranceSpec] = {}

    # ==================================================================
    # BOM Generation
    # ==================================================================

    def add_bom_item(self, item: BOMItem) -> None:
        """Add a single item to the BOM."""
        self.bom_items.append(item)

    def add_bom_items(self, items: List[BOMItem]) -> None:
        """Add multiple items."""
        self.bom_items.extend(items)

    def generate_bom_from_humanoid(
        self,
        humanoid: Any,  # ParametricHumanoid instance
    ) -> List[BOMItem]:
        """
        Auto-generate BOM from a ParametricHumanoid.
        """
        items: List[BOMItem] = []

        # Structural links
        for name, link in humanoid.links.items():
            if name == "base":
                continue
            items.append(BOMItem(
                part_number=f"LINK-{name.upper().replace('_', '-')}",
                description=f"Structural link: {name}",
                category=BOMCategory.STRUCTURAL,
                quantity=1,
                material="aluminum_6061",
                mass_kg=link.mass,
                notes=f"Inertia tensor diagonal: {np.diag(link.inertia_tensor)}",
            ))

        # Actuators
        for name, actuator in humanoid.actuators.items():
            items.append(BOMItem(
                part_number=f"ACT-{name.upper().replace('_', '-')}",
                description=f"Brushless DC actuator for {actuator.joint_name}",
                category=BOMCategory.ACTUATOR,
                quantity=1,
                material="steel_housing",
                mass_kg=actuator.mass,
                notes=f"Max torque: {actuator.max_torque:.2f} Nm, "
                      f"Max speed: {actuator.max_speed:.1f} rad/s",
            ))

        # Bearings
        for name, bearing in humanoid.bearings.items():
            items.append(BOMItem(
                part_number=f"BRG-{bearing.bore_diameter*1000:.0f}"
                            f"x{bearing.outer_diameter*1000:.0f}"
                            f"x{bearing.width*1000:.0f}",
                description=f"{bearing.type.replace('_', ' ').title()} bearing",
                category=BOMCategory.BEARING,
                quantity=2,  # Typically 2 per joint
                material="bearing_steel_52100",
                mass_kg=0.005,
                notes=f"Bore: {bearing.bore_diameter*1000:.1f} mm",
            ))

        # Fasteners
        fastener_counts: Dict[str, int] = {}
        for f in humanoid.fasteners:
            key = f"{f.type}_{f.diameter*1000:.0f}mmx{f.length*1000:.0f}mm"
            fastener_counts[key] = fastener_counts.get(key, 0) + 1

        for spec, qty in fastener_counts.items():
            items.append(BOMItem(
                part_number=f"FST-{spec.upper().replace(' ', '-')}",
                description=f"Fastener: {spec}",
                category=BOMCategory.FASTENER,
                quantity=qty,
                material=f.material,
                mass_kg=0.002 * qty,
            ))

        self.bom_items = items
        return items

    def get_bom_by_category(self) -> Dict[BOMCategory, List[BOMItem]]:
        """Group BOM items by category."""
        groups: Dict[BOMCategory, List[BOMItem]] = {}
        for item in self.bom_items:
            groups.setdefault(item.category, []).append(item)
        return groups

    def get_total_mass(self) -> float:
        """Sum mass of all BOM items (conservative estimate)."""
        return sum(item.mass_kg * item.quantity for item in self.bom_items)

    def get_total_cost(self) -> float:
        """Sum cost of all BOM items."""
        return sum(item.cost_usd * item.quantity for item in self.bom_items)

    def export_bom_csv(self, path: Union[str, Path]) -> Path:
        """Export BOM to CSV."""
        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Part Number", "Description", "Category", "Qty", "Unit",
                "Material", "Mass (kg)", "Supplier", "Cost (USD)",
                "Lead Time (days)", "Notes",
            ])
            for item in self.bom_items:
                writer.writerow([
                    item.part_number, item.description, item.category.value,
                    item.quantity, item.unit, item.material,
                    f"{item.mass_kg:.4f}", item.supplier,
                    f"{item.cost_usd:.2f}", item.lead_time_days, item.notes,
                ])
        return path

    def export_bom_json(self, path: Union[str, Path]) -> Path:
        """Export BOM to JSON."""
        path = Path(path)
        data = {
            "items": [item.to_dict() for item in self.bom_items],
            "summary": {
                "total_items": len(self.bom_items),
                "total_mass_kg": self.get_total_mass(),
                "total_cost_usd": self.get_total_cost(),
            },
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    # ==================================================================
    # Assembly Instructions
    # ==================================================================

    def generate_assembly_instructions(
        self,
        humanoid: Any,
    ) -> List[AssemblyStep]:
        """
        Generate step-by-step assembly instructions for the humanoid.
        """
        steps: List[AssemblyStep] = []
        step_num = 1

        # Step 1: Base frame
        steps.append(AssemblyStep(
            step_number=step_num,
            title="Mount Base Frame",
            description="Secure the base mounting plate to the test stand. "
                        "Ensure level and all bolt holes aligned.",
            parts=["base", "mounting_plate"],
            tools=["torque_wrench", "level", "M6_bolts"],
            torque_specs={"M6_bolt": 12.0},
            estimated_time_min=10.0,
        ))
        step_num += 1

        # Step 2: Pelvis
        steps.append(AssemblyStep(
            step_number=step_num,
            title="Attach Pelvis",
            description="Mount pelvis to base frame using shoulder bolts. "
                        "Install IMU inside pelvis cavity.",
            parts=["pelvis", "imu", "shoulder_bolts"],
            tools=["torque_wrench", "hex_keys"],
            torque_specs={"shoulder_bolt": 18.0},
            prerequisites=[1],
            estimated_time_min=15.0,
        ))
        step_num += 1

        # Step 3: Hip actuators and thighs
        for side in ("left", "right"):
            steps.append(AssemblyStep(
                step_number=step_num,
                title=f"Install {side.title()} Hip & Thigh",
                description=f"Press-fit hip bearings, install actuator, "
                            f"attach thigh link. Check backlash.",
                parts=[f"hip_actuator_{side}", f"thigh_{side}",
                       f"bearing_hip_{side}"],
                tools=["bearing_press", "torque_wrench", "feeler_gauge"],
                torque_specs={f"hip_bearing_cap": 8.0},
                prerequisites=[2],
                estimated_time_min=20.0,
            ))
            step_num += 1

        # Step 4: Knee and shin
        for side in ("left", "right"):
            steps.append(AssemblyStep(
                step_number=step_num,
                title=f"Install {side.title()} Knee & Shin",
                description="Assemble knee joint with crossed-roller bearing. "
                            "Attach shin and verify full range of motion.",
                parts=[f"knee_actuator_{side}", f"shin_{side}",
                       f"bearing_knee_{side}"],
                tools=["torque_wrench", "protractor"],
                torque_specs={f"knee_bearing_cap": 6.0},
                prerequisites=[step_num - 2],
                estimated_time_min=20.0,
            ))
            step_num += 1

        # Step 5: Ankle and foot
        for side in ("left", "right"):
            steps.append(AssemblyStep(
                step_number=step_num,
                title=f"Install {side.title()} Ankle & Foot",
                description="Mount ankle actuator and foot. "
                            "Install force-torque sensor.",
                parts=[f"ankle_actuator_{side}", f"foot_{side}",
                       f"fts_{side}"],
                tools=["torque_wrench", "multimeter"],
                prerequisites=[step_num - 2],
                estimated_time_min=15.0,
            ))
            step_num += 1

        # Step 6: Torso
        steps.append(AssemblyStep(
            step_number=step_num,
            title="Attach Torso",
            description="Mount torso to pelvis via spine actuator. "
                        "Route power and data cables.",
            parts=["torso", "spine_actuator", "cable_harness"],
            tools=["torque_wrench", "cable_ties"],
            prerequisites=[3],
            estimated_time_min=25.0,
        ))
        step_num += 1

        # Step 7: Shoulders and arms
        for side in ("left", "right"):
            steps.append(AssemblyStep(
                step_number=step_num,
                title=f"Install {side.title()} Shoulder & Arm",
                description="Assemble 3-DOF shoulder, attach upper arm, "
                            "elbow, forearm, wrist, and hand.",
                parts=[f"shoulder_actuator_{side}", f"upper_arm_{side}",
                       f"elbow_actuator_{side}", f"forearm_{side}",
                       f"wrist_actuator_{side}", f"hand_{side}"],
                tools=["torque_wrench", "soldering_iron"],
                prerequisites=[step_num - 1],
                estimated_time_min=30.0,
            ))
            step_num += 1

        # Step 8: Neck and head
        steps.append(AssemblyStep(
            step_number=step_num,
            title="Install Neck & Head",
            description="Mount neck actuator, attach head unit with cameras "
                        "and display. Calibrate neck IMU.",
            parts=["neck_actuator", "head", "cameras", "display"],
            tools=["torque_wrench", "calibration_jig"],
            prerequisites=[step_num - 1],
            estimated_time_min=20.0,
        ))
        step_num += 1

        # Step 9: Final checks
        steps.append(AssemblyStep(
            step_number=step_num,
            title="Final Inspection & Power-On",
            description="Verify all fasteners torqued, cables secured, "
                        "no interferences. Power on and run self-test.",
            parts=[],
            tools=["torque_wrench", "multimeter", "oscilloscope"],
            prerequisites=list(range(1, step_num)),
            estimated_time_min=30.0,
            warnings=[
                "Ensure emergency stop is accessible",
                "Verify ground fault protection",
            ],
        ))

        self.assembly_steps = steps
        return steps

    def export_instructions_markdown(self, path: Union[str, Path]) -> Path:
        """Export assembly instructions to Markdown."""
        path = Path(path)
        lines = ["# Moses Humanoid — Assembly Instructions\n"]
        for step in self.assembly_steps:
            lines.append(f"## Step {step.step_number}: {step.title}\n")
            lines.append(f"**Time:** {step.estimated_time_min:.0f} min\n")
            if step.prerequisites:
                lines.append(f"**Prerequisites:** {step.prerequisites}\n")
            lines.append(f"\n{step.description}\n")
            if step.parts:
                lines.append("\n**Parts:**")
                for p in step.parts:
                    lines.append(f"- {p}")
                lines.append("")
            if step.tools:
                lines.append("\n**Tools:**")
                for t in step.tools:
                    lines.append(f"- {t}")
                lines.append("")
            if step.torque_specs:
                lines.append("\n**Torque Specs:**")
                for k, v in step.torque_specs.items():
                    lines.append(f"- {k}: {v:.1f} Nm")
                lines.append("")
            if step.warnings:
                lines.append("\n**⚠️ Warnings:**")
                for w in step.warnings:
                    lines.append(f"- {w}")
                lines.append("")
            lines.append("---\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ==================================================================
    # Mass Properties
    # ==================================================================

    def calculate_mass_properties(
        self,
        meshes: Dict[str, Trimesh],
        densities: Optional[Dict[str, float]] = None,
    ) -> MassProperties:
        """
        Calculate aggregate mass properties from a dict of named meshes.

        Parameters
        ----------
        meshes : dict
            Mapping name -> trimesh object.
        densities : dict, optional
            Material density (kg/m³) per mesh name. Defaults to aluminum.
        """
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh required for mass property calculation")

        default_density = 2700.0  # Aluminum kg/m³
        densities = densities or {}

        total_mass = 0.0
        weighted_com = np.zeros(3)
        total_inertia = np.zeros((3, 3))
        all_verts: List[npt.NDArray[np.float64]] = []

        for name, mesh in meshes.items():
            rho = densities.get(name, default_density)
            if mesh.is_watertight:
                volume = mesh.volume
            else:
                # Approximate via convex hull
                volume = mesh.convex_hull.volume
            mass = volume * rho

            com = mesh.center_mass if mesh.is_watertight else mesh.centroid
            inertia = mesh.moment_inertia if mesh.is_watertight else np.eye(3) * mass * 0.01

            total_mass += mass
            weighted_com += mass * com
            # Parallel axis theorem to world origin
            r = com
            r_sq = np.dot(r, r)
            parallel = mass * (r_sq * np.eye(3) - np.outer(r, r))
            total_inertia += inertia + parallel
            all_verts.append(mesh.vertices)

        if total_mass <= 0:
            raise ValueError("Total mass must be > 0")

        com = weighted_com / total_mass

        # Shift inertia to CoM
        r_com = com
        r_com_sq = np.dot(r_com, r_com)
        correction = total_mass * (r_com_sq * np.eye(3) - np.outer(r_com, r_com))
        inertia_com = total_inertia - correction

        # Principal moments / axes
        eigvals, eigvecs = np.linalg.eigh(inertia_com)
        idx = np.argsort(eigvals)[::-1]
        principal_moments = eigvals[idx]
        principal_axes = eigvecs[:, idx]

        # Bounding box
        all_v = np.vstack(all_verts)
        bbox = (all_v.min(axis=0), all_v.max(axis=0))

        props = MassProperties(
            total_mass=total_mass,
            center_of_mass=com,
            inertia_tensor=inertia_com,
            principal_moments=principal_moments,
            principal_axes=principal_axes,
            bounding_box=bbox,
            volume=sum(m.volume if m.is_watertight else m.convex_hull.volume
                       for m in meshes.values()),
        )
        self.mass_cache["assembly"] = props
        return props

    def calculate_link_mass_properties(
        self,
        mesh: Trimesh,
        density: float = 2700.0,
    ) -> MassProperties:
        """Calculate mass properties for a single link mesh."""
        if not mesh.is_watertight:
            mesh = mesh.convex_hull
        volume = mesh.volume
        mass = volume * density
        com = mesh.center_mass
        inertia = mesh.moment_inertia
        eigvals, eigvecs = np.linalg.eigh(inertia)
        idx = np.argsort(eigvals)[::-1]
        return MassProperties(
            total_mass=mass,
            center_of_mass=com,
            inertia_tensor=inertia,
            principal_moments=eigvals[idx],
            principal_axes=eigvecs[:, idx],
            bounding_box=(mesh.bounds[0], mesh.bounds[1]),
            volume=volume,
        )

    # ==================================================================
    # Interference Detection
    # ==================================================================

    def detect_interferences(
        self,
        meshes: Dict[str, Trimesh],
        tolerance: float = 1e-5,
    ) -> List[InterferenceReport]:
        """
        Detect overlapping / interfering meshes.

        Uses bounding-box broadphase + mesh narrowphase.
        """
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh required for interference detection")

        reports: List[InterferenceReport] = []
        names = list(meshes.keys())

        for i, name_a in enumerate(names):
            mesh_a = meshes[name_a]
            for name_b in names[i + 1:]:
                mesh_b = meshes[name_b]

                # Broadphase: AABB overlap
                if not self._aabb_overlap(mesh_a.bounds, mesh_b.bounds):
                    continue

                # Narrowphase: collision check via proximity query
                try:
                    # Use trimesh proximity for distance
                    prox = trimesh.proximity.ProximityQuery(mesh_a)
                    signed_dist = prox.signed_distance(mesh_b.vertices)
                    min_dist = float(np.min(signed_dist))
                except Exception:
                    # Fallback: use bounding sphere distance
                    min_dist = float(np.linalg.norm(
                        mesh_a.centroid - mesh_b.centroid
                    )) - (mesh_a.bounding_sphere.primitive.radius +
                          mesh_b.bounding_sphere.primitive.radius)

                if min_dist < -tolerance:
                    # Intersection detected — estimate volume from overlap
                    try:
                        intersection = mesh_a.intersection(mesh_b)
                        vol = intersection.volume if hasattr(intersection, "volume") and hasattr(intersection, 'volume') else 0.0
                    except Exception:
                        vol = 0.0
                    severity = "critical" if vol > 1e-6 else "warning"
                    reports.append(InterferenceReport(
                        part_a=name_a,
                        part_b=name_b,
                        intersection_volume=float(vol),
                        contact_points=[],  # Could extract from proximity
                        severity=severity,
                        recommended_action=(
                            "Redesign overlapping geometry or add clearance."
                            if severity == "critical" else
                            "Verify tolerance stack-up."
                        ),
                    ))

        self.interference_cache = reports
        return reports

    @staticmethod
    def _aabb_overlap(
        a: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        b: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> bool:
        """Check AABB overlap."""
        return np.all(a[0] <= b[1]) and np.all(b[0] <= a[1])

    def has_critical_interferences(self) -> bool:
        """Return True if any critical interference exists."""
        return any(r.severity == "critical" for r in self.interference_cache)

    # ==================================================================
    # Tolerance Analysis
    # ==================================================================

    def add_tolerance(self, spec: ToleranceSpec) -> None:
        """Register a tolerance specification."""
        self.tolerance_specs[spec.dimension] = spec

    def analyze_tolerance_stack(
        self,
        dimension_chain: List[str],
    ) -> ToleranceAnalysis:
        """
        Perform worst-case and RSS tolerance stack-up analysis.

        Parameters
        ----------
        dimension_chain : list of str
            Ordered list of dimension names to stack.
        """
        specs = [self.tolerance_specs[d] for d in dimension_chain]
        nominal = sum(s.nominal for s in specs)
        wc_upper = nominal + sum(s.upper for s in specs)
        wc_lower = nominal + sum(s.lower for s in specs)

        # RSS (Root Sum Square)
        rss_tol = math.sqrt(sum(max(s.upper, abs(s.lower))**2 for s in specs))
        rss_upper = nominal + rss_tol
        rss_lower = nominal - rss_tol

        # Estimate Cpk assuming normal distribution and ±3σ = tolerance
        total_tol = wc_upper - wc_lower
        sigma = total_tol / 6.0
        cpk = (rss_tol / (3 * sigma)) if sigma > 0 else float('inf')

        return ToleranceAnalysis(
            dimension_chain=specs,
            nominal_total=nominal,
            worst_case_upper=wc_upper,
            worst_case_lower=wc_lower,
            rss_upper=rss_upper,
            rss_lower=rss_lower,
            cpk_estimate=cpk,
        )

    def export_tolerance_report(
        self, analysis: ToleranceAnalysis, path: Union[str, Path]
    ) -> Path:
        """Export tolerance analysis to JSON."""
        path = Path(path)
        data = {
            "nominal_total": analysis.nominal_total,
            "worst_case": {
                "upper": analysis.worst_case_upper,
                "lower": analysis.worst_case_lower,
            },
            "rss": {
                "upper": analysis.rss_upper,
                "lower": analysis.rss_lower,
            },
            "cpk_estimate": analysis.cpk_estimate,
            "dimensions": [
                {
                    "name": s.dimension,
                    "nominal": s.nominal,
                    "upper": s.upper,
                    "lower": s.lower,
                }
                for s in analysis.dimension_chain
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    # ==================================================================
    # Full Report
    # ==================================================================

    def generate_full_report(
        self,
        humanoid: Any,
        meshes: Optional[Dict[str, Trimesh]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive assembly report."""
        report: Dict[str, Any] = {
            "bom": {
                "items": [item.to_dict() for item in self.bom_items],
                "total_mass_kg": self.get_total_mass(),
                "total_cost_usd": self.get_total_cost(),
            },
            "assembly_steps": len(self.assembly_steps),
            "interferences": [
                {
                    "part_a": r.part_a,
                    "part_b": r.part_b,
                    "volume": r.intersection_volume,
                    "severity": r.severity,
                }
                for r in self.interference_cache
            ],
        }
        if meshes and "assembly" in self.mass_cache:
            mp = self.mass_cache["assembly"]
            report["mass_properties"] = mp.to_dict()
        return report


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------

def compute_center_of_mass(
    masses: List[float],
    positions: List[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Weighted average of positions."""
    masses_arr = np.array(masses)
    positions_arr = np.array(positions)
    return np.average(positions_arr, axis=0, weights=masses_arr)


def parallel_axis_theorem(
    inertia_local: npt.NDArray[np.float64],
    mass: float,
    displacement: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply parallel axis theorem to shift inertia tensor."""
    r = displacement
    r_sq = np.dot(r, r)
    return inertia_local + mass * (r_sq * np.eye(3) - np.outer(r, r))


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mgr = AssemblyManager()
    # Dummy BOM
    mgr.add_bom_item(BOMItem(
        part_number="LINK-TORSO-01",
        description="Torso structural link",
        category=BOMCategory.STRUCTURAL,
        quantity=1,
        mass_kg=5.5,
    ))
    print(f"Total mass: {mgr.get_total_mass():.2f} kg")
    print(f"Total cost: ${mgr.get_total_cost():.2f}")
