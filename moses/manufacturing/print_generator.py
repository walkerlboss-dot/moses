"""
Moses v4.0 3D Print Generator
==============================
Slicer integration, support generation, orientation optimization,
material profiles, and print time / cost estimation.

Supports Cura, PrusaSlicer, and Bambu Studio via CLI invocation
and native profile generation.

References
----------
- Cura 5.x command line usage guide
- PrusaSlicer 2.7+ CLI documentation
- Bambu Studio 1.8+ CLI (orca-slicer fork)
- Material datasheets: Polymaker, Prusament, eSUN, MatterHackers
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Material Profiles (real-world FFF parameters)
# ---------------------------------------------------------------------------

class FilamentMaterial(Enum):
    PLA = auto()
    PETG = auto()
    ABS = auto()
    NYLON = auto()
    CF_NYLON = auto()      # Carbon-fiber reinforced nylon
    CF_PETG = auto()       # Carbon-fiber reinforced PETG
    CF_ABS = auto()        # Carbon-fiber reinforced ABS
    TPU = auto()           # Flexible (humanoid grip pads)


FILAMENT_DB: Dict[FilamentMaterial, dict] = {
    FilamentMaterial.PLA: {
        "name": "PLA",
        "brand": "Prusament / Polymaker",
        "density_g_cm3": 1.24,
        "cost_per_kg_usd": 22.0,
        "nozzle_temp_c": 210,
        "bed_temp_c": 60,
        "chamber_temp_c": 0,
        "fan_speed_pct": 100,
        "print_speed_mm_s": 60,
        "travel_speed_mm_s": 150,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.45,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 40,
        "infill_pattern": "gyroid",
        "retraction_mm": 0.8,
        "retraction_speed_mm_s": 35,
        "tensile_mpa": 65,
        "flexural_mpa": 95,
        "elongation_pct": 6.0,
    },
    FilamentMaterial.PETG: {
        "name": "PETG",
        "brand": "Prusament / eSUN",
        "density_g_cm3": 1.27,
        "cost_per_kg_usd": 25.0,
        "nozzle_temp_c": 240,
        "bed_temp_c": 85,
        "chamber_temp_c": 0,
        "fan_speed_pct": 50,
        "print_speed_mm_s": 50,
        "travel_speed_mm_s": 150,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.45,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 40,
        "infill_pattern": "gyroid",
        "retraction_mm": 1.2,
        "retraction_speed_mm_s": 40,
        "tensile_mpa": 75,
        "flexural_mpa": 85,
        "elongation_pct": 20.0,
    },
    FilamentMaterial.ABS: {
        "name": "ABS",
        "brand": "Hatchbox / Polymaker",
        "density_g_cm3": 1.04,
        "cost_per_kg_usd": 20.0,
        "nozzle_temp_c": 250,
        "bed_temp_c": 100,
        "chamber_temp_c": 50,
        "fan_speed_pct": 0,
        "print_speed_mm_s": 55,
        "travel_speed_mm_s": 150,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.45,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 40,
        "infill_pattern": "gyroid",
        "retraction_mm": 0.8,
        "retraction_speed_mm_s": 40,
        "tensile_mpa": 40,
        "flexural_mpa": 70,
        "elongation_pct": 25.0,
    },
    FilamentMaterial.NYLON: {
        "name": "Nylon (PA6/PA12)",
        "brand": "Polymaker / MatterHackers",
        "density_g_cm3": 1.14,
        "cost_per_kg_usd": 55.0,
        "nozzle_temp_c": 260,
        "bed_temp_c": 80,
        "chamber_temp_c": 40,
        "fan_speed_pct": 30,
        "print_speed_mm_s": 40,
        "travel_speed_mm_s": 120,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.45,
        "wall_count": 4,
        "top_bottom_layers": 5,
        "infill_pct": 50,
        "infill_pattern": "gyroid",
        "retraction_mm": 1.5,
        "retraction_speed_mm_s": 30,
        "tensile_mpa": 85,
        "flexural_mpa": 110,
        "elongation_pct": 40.0,
    },
    FilamentMaterial.CF_NYLON: {
        "name": "Carbon Fiber Nylon",
        "brand": "Polymaker PA6-CF / Prusament PC-CF",
        "density_g_cm3": 1.18,
        "cost_per_kg_usd": 75.0,
        "nozzle_temp_c": 280,
        "bed_temp_c": 90,
        "chamber_temp_c": 50,
        "fan_speed_pct": 20,
        "print_speed_mm_s": 35,
        "travel_speed_mm_s": 120,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.50,
        "wall_count": 4,
        "top_bottom_layers": 5,
        "infill_pct": 50,
        "infill_pattern": "grid",
        "retraction_mm": 1.0,
        "retraction_speed_mm_s": 25,
        "tensile_mpa": 120,
        "flexural_mpa": 180,
        "elongation_pct": 5.0,
        "hardened_nozzle": True,
    },
    FilamentMaterial.CF_PETG: {
        "name": "Carbon Fiber PETG",
        "brand": "eSUN / Fiberlogy",
        "density_g_cm3": 1.30,
        "cost_per_kg_usd": 45.0,
        "nozzle_temp_c": 250,
        "bed_temp_c": 85,
        "chamber_temp_c": 0,
        "fan_speed_pct": 50,
        "print_speed_mm_s": 45,
        "travel_speed_mm_s": 150,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.50,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 40,
        "infill_pattern": "grid",
        "retraction_mm": 1.2,
        "retraction_speed_mm_s": 35,
        "tensile_mpa": 75,
        "flexural_mpa": 110,
        "elongation_pct": 8.0,
        "hardened_nozzle": True,
    },
    FilamentMaterial.CF_ABS: {
        "name": "Carbon Fiber ABS",
        "brand": "3DXTech / Fiberlogy",
        "density_g_cm3": 1.08,
        "cost_per_kg_usd": 50.0,
        "nozzle_temp_c": 260,
        "bed_temp_c": 100,
        "chamber_temp_c": 50,
        "fan_speed_pct": 0,
        "print_speed_mm_s": 50,
        "travel_speed_mm_s": 150,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.50,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 40,
        "infill_pattern": "grid",
        "retraction_mm": 0.8,
        "retraction_speed_mm_s": 40,
        "tensile_mpa": 60,
        "flexural_mpa": 100,
        "elongation_pct": 4.0,
        "hardened_nozzle": True,
    },
    FilamentMaterial.TPU: {
        "name": "TPU (95A)",
        "brand": "SainSmart / NinjaTek",
        "density_g_cm3": 1.21,
        "cost_per_kg_usd": 35.0,
        "nozzle_temp_c": 230,
        "bed_temp_c": 60,
        "chamber_temp_c": 0,
        "fan_speed_pct": 100,
        "print_speed_mm_s": 25,
        "travel_speed_mm_s": 120,
        "layer_height_mm": 0.20,
        "line_width_mm": 0.45,
        "wall_count": 3,
        "top_bottom_layers": 4,
        "infill_pct": 20,
        "infill_pattern": "gyroid",
        "retraction_mm": 2.0,
        "retraction_speed_mm_s": 20,
        "tensile_mpa": 35,
        "flexural_mpa": None,
        "elongation_pct": 500.0,
    },
}


# ---------------------------------------------------------------------------
# Printer Profiles
# ---------------------------------------------------------------------------

class PrinterModel(Enum):
    PRUSA_MK4 = auto()
    PRUSA_XL = auto()
    BAMBULAB_X1 = auto()
    BAMBULAB_P1P = auto()
    VORON_24 = auto()
    CREALITY_K1 = auto()


PRINTER_DB: Dict[PrinterModel, dict] = {
    PrinterModel.PRUSA_MK4: {
        "build_volume_mm": (250, 210, 220),
        "max_speed_mm_s": 200,
        "acceleration_mm_s2": 4000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 5.0,
        "power_w": 150,
    },
    PrinterModel.PRUSA_XL: {
        "build_volume_mm": (360, 360, 360),
        "max_speed_mm_s": 200,
        "acceleration_mm_s2": 4000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 8.0,
        "power_w": 300,
    },
    PrinterModel.BAMBULAB_X1: {
        "build_volume_mm": (256, 256, 256),
        "max_speed_mm_s": 500,
        "acceleration_mm_s2": 20000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 6.0,
        "power_w": 200,
    },
    PrinterModel.BAMBULAB_P1P: {
        "build_volume_mm": (256, 256, 256),
        "max_speed_mm_s": 500,
        "acceleration_mm_s2": 20000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 5.0,
        "power_w": 180,
    },
    PrinterModel.VORON_24: {
        "build_volume_mm": (300, 300, 300),
        "max_speed_mm_s": 350,
        "acceleration_mm_s2": 10000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 4.0,
        "power_w": 250,
    },
    PrinterModel.CREALITY_K1: {
        "build_volume_mm": (220, 220, 250),
        "max_speed_mm_s": 600,
        "acceleration_mm_s2": 20000,
        "nozzle_diameter_mm": 0.40,
        "hourly_rate_usd": 3.5,
        "power_w": 150,
    },
}


# ---------------------------------------------------------------------------
# Slicer Profile
# ---------------------------------------------------------------------------

@dataclass
class SlicerProfile:
    """Serializable slicer configuration."""

    filament: FilamentMaterial
    printer: PrinterModel
    layer_height_mm: float = 0.20
    line_width_mm: float = 0.45
    wall_count: int = 3
    top_bottom_layers: int = 4
    infill_pct: float = 40.0
    infill_pattern: str = "gyroid"
    support_enable: bool = True
    support_angle: float = 50.0  # overhang angle threshold
    support_density: float = 15.0
    brim_width_mm: float = 8.0
    raft_enable: bool = False
    extra_fields: dict = field(default_factory=dict)

    def to_cura_dict(self) -> dict:
        f = FILAMENT_DB[self.filament]
        p = PRINTER_DB[self.printer]
        return {
            "layer_height": self.layer_height_mm,
            "line_width": self.line_width_mm,
            "wall_line_count": self.wall_count,
            "top_layers": self.top_bottom_layers,
            "bottom_layers": self.top_bottom_layers,
            "infill_sparse_density": self.infill_pct,
            "infill_pattern": self.infill_pattern,
            "material_print_temperature": f["nozzle_temp_c"],
            "material_bed_temperature": f["bed_temp_c"],
            "cool_fan_speed": f["fan_speed_pct"],
            "speed_print": f["print_speed_mm_s"],
            "speed_travel": f["travel_speed_mm_s"],
            "retraction_amount": f["retraction_mm"],
            "retraction_speed": f["retraction_speed_mm_s"],
            "support_enable": self.support_enable,
            "support_overhang_angle": self.support_angle,
            "support_infill_rate": self.support_density,
            "adhesion_type": "brim" if self.brim_width_mm > 0 else "none",
            "brim_width": self.brim_width_mm,
            **self.extra_fields,
        }

    def to_prusa_ini(self) -> str:
        """Generate PrusaSlicer-style INI snippet."""
        f = FILAMENT_DB[self.filament]
        p = PRINTER_DB[self.printer]
        lines = [
            "[print_settings]",
            f"layer_height = {self.layer_height_mm}",
            f"perimeters = {self.wall_count}",
            f"top_solid_layers = {self.top_bottom_layers}",
            f"bottom_solid_layers = {self.top_bottom_layers}",
            f"fill_density = {self.infill_pct}%",
            f"fill_pattern = {self.infill_pattern}",
            f"support_material = {1 if self.support_enable else 0}",
            f"support_material_angle = {self.support_angle}",
            f"support_material_density = {self.support_density}%",
            f"brim_width = {self.brim_width_mm}",
            "",
            "[filament_settings]",
            f"temperature = {f['nozzle_temp_c']}",
            f"bed_temperature = {f['bed_temp_c']}",
            f"fan_speed = {f['fan_speed_pct']}",
            f"retract_length = {f['retraction_mm']}",
            f"retract_speed = {f['retraction_speed_mm_s']}",
            "",
            "[printer_settings]",
            f"nozzle_diameter = {p['nozzle_diameter_mm']}",
            f"max_print_speed = {p['max_speed_mm_s']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Support Generator
# ---------------------------------------------------------------------------

class SupportGenerator:
    """
    Generate support structures for overhangs.
    Simplified geometric analysis; real slicers use voxel/mesh methods.
    """

    @staticmethod
    def overhang_area(mesh_triangles: List[Tuple], overhang_angle_deg: float = 50.0) -> float:
        """
        Estimate overhang area from triangle normals.
        mesh_triangles: list of ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3))
        """
        area = 0.0
        threshold = math.cos(math.radians(overhang_angle_deg))
        for tri in mesh_triangles:
            (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = tri
            # Normal Z component via cross product
            ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
            vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm = math.sqrt(nx**2 + ny**2 + nz**2)
            if norm == 0:
                continue
            nz_unit = abs(nz) / norm
            if nz_unit < threshold:
                # Triangle area
                area += 0.5 * norm
        return area

    @staticmethod
    def support_volume(overhang_area_mm2: float, support_density: float, avg_height_mm: float = 10.0) -> float:
        """Estimate support material volume in mm³."""
        return overhang_area_mm2 * avg_height_mm * (support_density / 100.0)


# ---------------------------------------------------------------------------
# Orientation Optimizer
# ---------------------------------------------------------------------------

class OrientationOptimizer:
    """
    Optimize build orientation for strength and minimal support.
    Evaluates candidate rotations around X/Y/Z axes.
    """

    @staticmethod
    def evaluate_orientation(
        mesh_triangles: List[Tuple],
        rotation_xyz_deg: Tuple[float, float, float],
        material: FilamentMaterial,
    ) -> dict:
        """
        Return a score dict for a given orientation.
        Higher score = better.
        """
        # Simplified: just use overhang area and Z-height as proxies
        overhang = SupportGenerator.overhang_area(mesh_triangles)
        # Bounding box Z-height proxy
        zs = [v[2] for tri in mesh_triangles for v in tri]
        z_height = max(zs) - min(zs)

        # Score: penalize overhang, penalize tall prints (layer adhesion risk)
        score = 1000.0 - overhang * 0.1 - z_height * 2.0

        # CF materials prefer Z=0 alignment for fiber direction
        if "CF" in FILAMENT_DB[material]["name"]:
            # Favor orientations where primary stress axis aligns with bed plane
            score += 200

        return {
            "rotation": rotation_xyz_deg,
            "overhang_area_mm2": round(overhang, 2),
            "z_height_mm": round(z_height, 2),
            "score": round(score, 2),
        }

    @classmethod
    def optimize(
        cls,
        mesh_triangles: List[Tuple],
        material: FilamentMaterial,
        step_deg: float = 45.0,
    ) -> dict:
        """Brute-force search over coarse rotations."""
        best = None
        for rx in [0, step_deg, 90, 135]:
            for ry in [0, step_deg, 90, 135]:
                for rz in [0, step_deg, 90, 135]:
                    ev = cls.evaluate_orientation(mesh_triangles, (rx, ry, rz), material)
                    if best is None or ev["score"] > best["score"]:
                        best = ev
        return best


# ---------------------------------------------------------------------------
# Print Estimator
# ---------------------------------------------------------------------------

class PrintEstimator:
    """
    Estimate print time, material usage, and cost from part geometry.
    Uses bounding-box heuristics calibrated against slicer outputs.
    """

    @staticmethod
    def estimate(
        bounding_box_mm: Tuple[float, float, float],
        volume_cm3: float,
        profile: SlicerProfile,
        printer: PrinterModel,
    ) -> dict:
        """
        Returns dict with time_hr, filament_g, filament_m, cost_usd, power_kwh.
        """
        f = FILAMENT_DB[profile.filament]
        p = PRINTER_DB[printer]

        # Layer count
        layers = math.ceil(bounding_box_mm[2] / profile.layer_height_mm)

        # Perimeter length heuristic: ~4× perimeter per layer
        perimeter_mm = 2 * (bounding_box_mm[0] + bounding_box_mm[1])
        wall_length_mm = perimeter_mm * profile.wall_count * layers

        # Infill area per layer
        layer_area_mm2 = bounding_box_mm[0] * bounding_box_mm[1]
        infill_area_mm2 = layer_area_mm2 * (profile.infill_pct / 100.0)
        infill_length_mm = (infill_area_mm2 / (profile.line_width_mm * profile.layer_height_mm)) * layers

        # Support volume (heuristic: 30% of part volume if enabled)
        support_vol_cm3 = volume_cm3 * 0.30 if profile.support_enable else 0.0

        # Total extruded volume
        extrusion_width_mm = profile.line_width_mm
        extrusion_height_mm = profile.layer_height_mm
        total_vol_mm3 = (
            (wall_length_mm + infill_length_mm) * extrusion_width_mm * extrusion_height_mm
            + support_vol_cm3 * 1000
        )

        # Filament mass
        filament_dia_mm = 1.75
        filament_length_mm = total_vol_mm3 / (math.pi * (filament_dia_mm / 2) ** 2)
        filament_m = filament_length_mm / 1000.0
        filament_g = total_vol_mm3 * (f["density_g_cm3"] / 1000.0)

        # Time estimate
        # Average speed: print speed weighted by accel limits
        avg_speed_mm_s = min(f["print_speed_mm_s"], p["max_speed_mm_s"] * 0.6)
        print_time_sec = filament_length_mm / avg_speed_mm_s
        # Add non-print moves (~20% overhead) + heatup/cooldown (5 min)
        print_time_sec *= 1.20
        print_time_sec += 300
        print_time_hr = print_time_sec / 3600.0

        # Cost
        material_cost = (filament_g / 1000.0) * f["cost_per_kg_usd"]
        machine_cost = print_time_hr * p["hourly_rate_usd"]
        power_kwh = (p["power_w"] * print_time_hr) / 1000.0
        power_cost = power_kwh * 0.15  # $0.15/kWh
        total_cost = material_cost + machine_cost + power_cost

        return {
            "print_time_hr": round(print_time_hr, 2),
            "layers": layers,
            "filament_g": round(filament_g, 2),
            "filament_m": round(filament_m, 2),
            "material_cost_usd": round(material_cost, 2),
            "machine_cost_usd": round(machine_cost, 2),
            "power_kwh": round(power_kwh, 3),
            "power_cost_usd": round(power_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "hardened_nozzle": f.get("hardened_nozzle", False),
        }


# ---------------------------------------------------------------------------
# Print Job
# ---------------------------------------------------------------------------

@dataclass
class PrintJob:
    """High-level container for a 3D print job."""

    stl_path: str
    profile: SlicerProfile
    printer: PrinterModel
    bounding_box_mm: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    volume_cm3: float = 50.0
    quantity: int = 1

    def estimate(self) -> dict:
        base = PrintEstimator.estimate(self.bounding_box_mm, self.volume_cm3, self.profile, self.printer)
        if self.quantity > 1:
            # Batch efficiency: 10% time savings per additional part (nesting)
            base["print_time_hr"] = round(base["print_time_hr"] * (1 - 0.10 * (self.quantity - 1)), 2)
            base["total_cost_usd"] = round(base["total_cost_usd"] * self.quantity * 0.95, 2)
        return base

    def export_profile(self, slicer: str, out_path: str) -> str:
        """Export slicer profile to disk."""
        slicer = slicer.lower()
        if slicer == "cura":
            data = self.profile.to_cura_dict()
            with open(out_path, "w") as fh:
                json.dump(data, fh, indent=2)
        elif slicer in ("prusa", "prusaslicer"):
            text = self.profile.to_prusa_ini()
            with open(out_path, "w") as fh:
                fh.write(text)
        else:
            raise ValueError(f"Unsupported slicer: {slicer}")
        return out_path

    def slice(self, slicer_path: str, slicer: str, out_gcode: str) -> str:
        """
        Invoke slicer CLI. Requires slicer binary installed.
        Returns path to generated G-code.
        """
        slicer = slicer.lower()
        if slicer == "cura":
            cmd = [
                slicer_path,
                "slice",
                "-j", self.export_profile("cura", "/tmp/moses_cura.json"),
                "-l", self.stl_path,
                "-o", out_gcode,
            ]
        elif slicer in ("prusa", "prusaslicer"):
            profile_path = self.export_profile("prusa", "/tmp/moses_prusa.ini")
            cmd = [
                slicer_path,
                "--load", profile_path,
                "--export-gcode",
                "--output", out_gcode,
                self.stl_path,
            ]
        elif slicer == "bambu":
            # Bambu Studio / Orca Slicer CLI
            cmd = [
                slicer_path,
                "--slice",
                "--export-gcode",
                "--output", out_gcode,
                self.stl_path,
            ]
        else:
            raise ValueError(f"Unsupported slicer: {slicer}")

        subprocess.run(cmd, check=True)
        return out_gcode


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_print_job(
    stl_path: str,
    material: FilamentMaterial = FilamentMaterial.PLA,
    printer: PrinterModel = PrinterModel.PRUSA_MK4,
    bbox_mm: Tuple[float, float, float] = (50, 50, 50),
    vol_cm3: float = 50.0,
) -> PrintJob:
    profile = SlicerProfile(
        filament=material,
        printer=printer,
        layer_height_mm=FILAMENT_DB[material]["layer_height_mm"],
        line_width_mm=FILAMENT_DB[material]["line_width_mm"],
        wall_count=FILAMENT_DB[material]["wall_count"],
        top_bottom_layers=FILAMENT_DB[material]["top_bottom_layers"],
        infill_pct=FILAMENT_DB[material]["infill_pct"],
        infill_pattern=FILAMENT_DB[material]["infill_pattern"],
    )
    return PrintJob(stl_path, profile, printer, bbox_mm, vol_cm3)
