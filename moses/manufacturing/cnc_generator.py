"""
Moses v4.0 CNC G-Code Generator
================================
Toolpath generation for milling, turning, and drilling operations.
Supports aluminum (6061, 7075), steel (1018, 4140), and titanium (Ti-6Al-4V).
Exports to Fanuc, Haas, and LinuxCNC formats.

References
----------
- Sandvik Coromant: General Turning / Milling catalog (ISO 513)
- Kennametal: Speeds & Feeds pocket guide
- Haas VF-series operator manual (rev 2024)
- Fanuc 0i-MF programming manual
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Material Database (real-world cutting parameters)
# ---------------------------------------------------------------------------

class Material(Enum):
    AL_6061 = auto()
    AL_7075 = auto()
    STEEL_1018 = auto()
    STEEL_4140 = auto()
    TI_6AL4V = auto()


MATERIAL_DB = {
    Material.AL_6061: {
        "name": "Aluminum 6061-T6",
        "density_kg_m3": 2700,
        "hardness_hb": 95,
        "sfm_hss": 600,
        "sfm_carbide": 1200,
        "sfm_ceramic": 3000,
        "feed_per_tooth_in": 0.005,
        "max_doc_in": 0.250,
        "unit_hp": 0.25,  # hp / in^3/min
        "cost_per_kg": 4.50,
    },
    Material.AL_7075: {
        "name": "Aluminum 7075-T6",
        "density_kg_m3": 2810,
        "hardness_hb": 150,
        "sfm_hss": 400,
        "sfm_carbide": 900,
        "sfm_ceramic": 2000,
        "feed_per_tooth_in": 0.004,
        "max_doc_in": 0.200,
        "unit_hp": 0.33,
        "cost_per_kg": 12.00,
    },
    Material.STEEL_1018: {
        "name": "Steel 1018 (low-carbon)",
        "density_kg_m3": 7870,
        "hardness_hb": 126,
        "sfm_hss": 120,
        "sfm_carbide": 350,
        "sfm_ceramic": 800,
        "feed_per_tooth_in": 0.003,
        "max_doc_in": 0.150,
        "unit_hp": 1.0,
        "cost_per_kg": 1.20,
    },
    Material.STEEL_4140: {
        "name": "Steel 4140 (pre-hard)",
        "density_kg_m3": 7850,
        "hardness_hb": 285,
        "sfm_hss": 60,
        "sfm_carbide": 200,
        "sfm_ceramic": 500,
        "feed_per_tooth_in": 0.002,
        "max_doc_in": 0.100,
        "unit_hp": 1.4,
        "cost_per_kg": 2.80,
    },
    Material.TI_6AL4V: {
        "name": "Titanium Ti-6Al-4V",
        "density_kg_m3": 4430,
        "hardness_hb": 334,
        "sfm_hss": 40,
        "sfm_carbide": 150,
        "sfm_ceramic": 400,
        "feed_per_tooth_in": 0.002,
        "max_doc_in": 0.080,
        "unit_hp": 1.5,
        "cost_per_kg": 35.00,
    },
}


# ---------------------------------------------------------------------------
# Tool Database
# ---------------------------------------------------------------------------

class ToolType(Enum):
    END_MILL = auto()
    FACE_MILL = auto()
    BALL_MILL = auto()
    DRILL = auto()
    TURN_INSERT = auto()
    BORING_BAR = auto()


@dataclass
class Tool:
    """CNC cutting tool definition."""

    tool_number: int
    name: str
    tool_type: ToolType
    diameter_in: float
    flutes: int
    material: str = "carbide"  # hss, carbide, ceramic, cermet
    coating: str = "TiAlN"
    stickout_in: float = 3.0
    max_rpm: int = 10_000
    coolant: bool = True

    def __post_init__(self):
        assert self.diameter_in > 0
        assert self.flutes >= 1


# Preset tools for humanoid-scale parts (typical Haas VF-2 loadout)
TOOL_LIBRARY: List[Tool] = [
    Tool(1, "3/8 EM", ToolType.END_MILL, 0.375, 4, "carbide", "TiAlN", 3.0, 10_000, True),
    Tool(2, "1/2 EM", ToolType.END_MILL, 0.500, 4, "carbide", "TiAlN", 3.5, 10_000, True),
    Tool(3, "3/4 FM", ToolType.FACE_MILL, 0.750, 6, "carbide", "TiAlN", 2.5, 8_000, True),
    Tool(4, "1/4 BM", ToolType.BALL_MILL, 0.250, 2, "carbide", "TiAlN", 3.0, 12_000, True),
    Tool(5, "#7 DRILL", ToolType.DRILL, 0.2010, 2, "carbide", "TiN", 2.0, 8_000, True),
    Tool(6, "1/4 DRILL", ToolType.DRILL, 0.2500, 2, "carbide", "TiN", 2.0, 8_000, True),
    Tool(7, "3/8 DRILL", ToolType.DRILL, 0.3750, 2, "carbide", "TiN", 2.5, 8_000, True),
    Tool(8, "TNMG332", ToolType.TURN_INSERT, 0.500, 1, "carbide", "CVD", 1.0, 4_000, True),
]


# ---------------------------------------------------------------------------
# Feed / Speed Calculator
# ---------------------------------------------------------------------------

class FeedSpeedCalculator:
    """
    Calculate spindle speed (RPM) and feed rate (IPM) from
    SFM, tool diameter, and chipload per tooth.
    """

    @staticmethod
    def rpm(sfm: float, diameter_in: float) -> int:
        """RPM = (SFM × 3.82) / D"""
        return int((sfm * 3.82) / diameter_in)

    @staticmethod
    def ipm(rpm: int, feed_per_tooth: float, flutes: int) -> float:
        """IPM = RPM × chipload × flutes"""
        return rpm * feed_per_tooth * flutes

    @classmethod
    def for_material(
        cls,
        material: Material,
        tool: Tool,
        doc_in: Optional[float] = None,
        woc_in: Optional[float] = None,
        adjust: float = 1.0,
    ) -> Tuple[int, float, float]:
        """
        Returns (rpm, ipm, mrr_in3_min) for a given material and tool.
        Adjust factor scales SFM for conservative/aggressive cuts.
        """
        db = MATERIAL_DB[material]
        sfm_key = f"sfm_{tool.material}"
        sfm = db.get(sfm_key, db["sfm_carbide"]) * adjust
        rpm = min(cls.rpm(sfm, tool.diameter_in), tool.max_rpm)

        # Reduce chipload for deeper cuts (radial chip thinning heuristic)
        fpt = db["feed_per_tooth_in"]
        if doc_in and woc_in:
            ae_ratio = woc_in / tool.diameter_in
            if ae_ratio < 0.5:
                fpt *= (ae_ratio / 0.5) ** 0.5  # chip thinning correction

        ipm = cls.ipm(rpm, fpt, tool.flutes)

        # Material Removal Rate
        doc = doc_in or db["max_doc_in"] * 0.5
        woc = woc_in or tool.diameter_in * 0.5
        mrr = woc * doc * ipm

        return rpm, ipm, mrr

    @staticmethod
    def horsepower(mrr_in3_min: float, unit_hp: float) -> float:
        """Required spindle HP = MRR × unit horsepower."""
        return mrr_in3_min * unit_hp


# ---------------------------------------------------------------------------
# Surface Finish Predictor
# ---------------------------------------------------------------------------

class SurfaceFinishPredictor:
    """
    Predict theoretical surface finish (Ra, μin) from feed per revolution
    and nose radius (turning) or cusp height (milling).

    References
    ----------
    - Sandvik:  Ra ≈ f² / (32 × rε)   [turning]
    - Milling cusp:  h = f² / (8 × D)
    """

    @staticmethod
    def turning_ra(feed_per_rev_in: float, nose_radius_in: float) -> float:
        """Ra in micro-inches."""
        return (feed_per_rev_in**2) / (32.0 * nose_radius_in) * 1_000_000

    @staticmethod
    def milling_cusp(feed_per_tooth_in: float, tool_diameter_in: float) -> float:
        """Cusp height (peak-to-valley) in inches."""
        return (feed_per_tooth_in**2) / (8.0 * tool_diameter_in)

    @staticmethod
    def milling_ra(feed_per_tooth_in: float, tool_diameter_in: float) -> float:
        """Approximate Ra from cusp height (Ra ≈ h / 4)."""
        return SurfaceFinishPredictor.milling_cusp(feed_per_tooth_in, tool_diameter_in) / 4.0 * 1_000_000


# ---------------------------------------------------------------------------
# Toolpaths
# ---------------------------------------------------------------------------

@dataclass
class Toolpath:
    """Base class for a machining operation."""

    tool: Tool
    rpm: int
    feed_ipm: float
    rapid_height: float = 0.5  # G0 clearance above part
    plunge_feed_pct: float = 0.5  # % of feed for Z plunge

    def generate(self, controller: str) -> List[str]:
        raise NotImplementedError

    def _header(self, controller: str) -> List[str]:
        lines = [f"( Operation: {self.__class__.__name__} )"]
        lines.append(f"( Tool: T{self.tool.tool_number:02d} {self.tool.name} )")
        lines.append(f"( RPM: {self.rpm}, Feed: {self.feed_ipm:.1f} IPM )")
        lines.append(f"T{self.tool.tool_number:02d} M06")
        if controller in ("fanuc", "haas"):
            lines.append(f"G43 H{self.tool.tool_number:02d} Z{self.rapid_height:.4f}")
        else:  # linuxcnc
            lines.append(f"G43.1 Z{self.rapid_height:.4f}")
        lines.append(f"S{self.rpm} M03")
        if self.tool.coolant:
            lines.append("M08")
        lines.append(f"G54")
        return lines

    def _footer(self, controller: str) -> List[str]:
        lines = []
        if self.tool.coolant:
            lines.append("M09")
        lines.append("M05")
        lines.append("G91 G28 Z0")
        lines.append("G90")
        return lines


@dataclass
class FaceMill(Toolpath):
    """Face milling operation (rectangular stock)."""

    x0: float = 0.0
    y0: float = 0.0
    width: float = 2.0
    height: float = 2.0
    depth: float = 0.010
    stepover_pct: float = 75.0

    def generate(self, controller: str) -> List[str]:
        lines = self._header(controller)
        stepover = self.tool.diameter_in * (self.stepover_pct / 100.0)
        passes = int(math.ceil(self.height / stepover))
        z = -self.depth
        plunge = self.feed_ipm * self.plunge_feed_pct

        lines.append(f"G00 X{self.x0:.4f} Y{self.y0:.4f}")
        lines.append(f"G01 Z{z:.4f} F{plunge:.1f}")

        for i in range(passes):
            y = self.y0 + i * stepover
            y = min(y, self.y0 + self.height)
            x_start = self.x0 if i % 2 == 0 else self.x0 + self.width
            x_end = self.x0 + self.width if i % 2 == 0 else self.x0
            lines.append(f"G01 Y{y:.4f} F{self.feed_ipm:.1f}")
            lines.append(f"G01 X{x_end:.4f}")

        lines.append(f"G00 Z{self.rapid_height:.4f}")
        lines.extend(self._footer(controller))
        return lines


@dataclass
class PocketMill(Toolpath):
    """Rectangular pocket roughing."""

    x0: float = 0.0
    y0: float = 0.0
    width: float = 1.0
    height: float = 1.0
    depth: float = 0.5
    stepdown: float = 0.100
    stepover_pct: float = 40.0
    finish_allowance: float = 0.010

    def generate(self, controller: str) -> List[str]:
        lines = self._header(controller)
        stepover = self.tool.diameter_in * (self.stepover_pct / 100.0)
        plunge = self.feed_ipm * self.plunge_feed_pct
        z_levels = [round(-self.stepdown * (i + 1), 4) for i in range(int(math.ceil(self.depth / self.stepdown)))]
        z_levels[-1] = -self.depth

        # Spiral-in from center heuristic
        cx, cy = self.x0 + self.width / 2, self.y0 + self.height / 2
        lines.append(f"G00 X{cx:.4f} Y{cy:.4f}")

        for z in z_levels:
            lines.append(f"G01 Z{z:.4f} F{plunge:.1f}")
            # Simple zig-zag pocket at this Z
            inner_w = self.width - 2 * self.finish_allowance - self.tool.diameter_in
            inner_h = self.height - 2 * self.finish_allowance - self.tool.diameter_in
            passes_y = max(1, int(math.ceil(inner_h / stepover)))
            for i in range(passes_y):
                y = cy - inner_h / 2 + i * stepover
                y = min(y, cy + inner_h / 2)
                x_left = cx - inner_w / 2
                x_right = cx + inner_w / 2
                lines.append(f"G01 Y{y:.4f} F{self.feed_ipm:.1f}")
                lines.append(f"G01 X{x_right if i % 2 == 0 else x_left:.4f}")
            lines.append(f"G00 Z{self.rapid_height:.4f}")

        lines.extend(self._footer(controller))
        return lines


@dataclass
class DrillCycle(Toolpath):
    """Standard drilling cycle (G81/G83)."""

    x: float = 0.0
    y: float = 0.0
    z_bottom: float = -0.500
    z_top: float = 0.0
    peck_depth: Optional[float] = None  # None = G81, else G83
    retract: float = 0.050

    def generate(self, controller: str) -> List[str]:
        lines = self._header(controller)
        lines.append(f"G00 X{self.x:.4f} Y{self.y:.4f}")
        if self.peck_depth:
            lines.append(f"G83 Z{self.z_bottom:.4f} R{self.retract:.4f} Q{self.peck_depth:.4f} F{self.feed_ipm:.1f}")
        else:
            lines.append(f"G81 Z{self.z_bottom:.4f} R{self.retract:.4f} F{self.feed_ipm:.1f}")
        lines.append("G80")
        lines.append(f"G00 Z{self.rapid_height:.4f}")
        lines.extend(self._footer(controller))
        return lines


@dataclass
class TurnOperation(Toolpath):
    """Simple OD turning pass (lathe)."""

    x_start: float = 1.0  # diameter
    x_end: float = 0.9
    z_start: float = 0.0
    z_end: float = -2.0
    doc_radial: float = 0.010  # radial DOC

    def generate(self, controller: str) -> List[str]:
        lines = self._header(controller)
        # Lathe typically uses G18 (XZ plane)
        lines.append("G18 G99")  # feed per rev
        lines.append(f"G00 X{self.x_start:.4f} Z{self.z_start:.4f}")
        lines.append(f"G01 X{self.x_end:.4f} F{self.feed_ipm:.4f}")
        lines.append(f"G01 Z{self.z_end:.4f}")
        lines.append(f"G00 X{self.x_start + 0.1:.4f}")
        lines.append(f"G00 Z{self.z_start:.4f}")
        lines.extend(self._footer(controller))
        return lines


# ---------------------------------------------------------------------------
# CNC Job
# ---------------------------------------------------------------------------

class CNCJob:
    """Container for a sequence of CNC operations."""

    def __init__(self, controller: str, material: Material, stock_size: Tuple[float, float, float] = (4.0, 2.0, 1.0), machine: Optional[str] = None):
        self.controller = controller.lower()
        self.machine = (machine or controller).lower()
        self.material = material
        self.stock_size = stock_size  # (L, W, H) inches
        self.operations: List[Toolpath] = []
        self.setup_time_min: float = 15.0
        self.hourly_rate: float = 85.0  # USD/hr for 3-axis CNC

    def add_operation(self, op: Toolpath) -> "CNCJob":
        self.operations.append(op)
        return self

    def estimate_time(self) -> float:
        """Rough cycle time estimate (minutes) from MRR and distances."""
        total_sec = 0.0
        db = MATERIAL_DB[self.material]
        for op in self.operations:
            # heuristic: 30% rapid, 70% cutting
            if isinstance(op, FaceMill):
                area = op.width * op.height
                vol = area * op.depth
                _, _, mrr = FeedSpeedCalculator.for_material(self.material, op.tool, op.depth, op.tool.diameter_in * 0.75)
                cut_sec = (vol / mrr) * 60 if mrr > 0 else 60
                total_sec += cut_sec * 1.3
            elif isinstance(op, PocketMill):
                vol = op.width * op.height * op.depth
                _, _, mrr = FeedSpeedCalculator.for_material(self.material, op.tool, op.stepdown, op.tool.diameter_in * 0.4)
                cut_sec = (vol / mrr) * 60 if mrr > 0 else 120
                total_sec += cut_sec * 1.4
            elif isinstance(op, DrillCycle):
                # assume 0.5s positioning + drill time
                depth = abs(op.z_bottom)
                feed_per_rev = MATERIAL_DB[self.material]["feed_per_tooth_in"] * 2
                rpm = op.rpm
                ipm = rpm * feed_per_rev
                total_sec += 1.0 + (depth / ipm) * 60
            elif isinstance(op, TurnOperation):
                length = abs(op.z_end - op.z_start)
                # feed per rev -> ipm
                fpr = MATERIAL_DB[self.material]["feed_per_tooth_in"]
                ipm = op.rpm * fpr
                total_sec += (length / ipm) * 60 * 1.2
            else:
                total_sec += 60  # generic 1 min
        return total_sec / 60.0 + self.setup_time_min

    def estimate_cost(self) -> dict:
        """Return cost breakdown dict."""
        cycle_min = self.estimate_time()
        machine_cost = (cycle_min / 60.0) * self.hourly_rate
        stock_vol_in3 = math.prod(self.stock_size)
        stock_kg = stock_vol_in3 * 16.387 * MATERIAL_DB[self.material]["density_kg_m3"] / 1_000_000
        material_cost = stock_kg * MATERIAL_DB[self.material]["cost_per_kg"]
        overhead = machine_cost * 0.25  # 25% overhead
        return {
            "cycle_time_min": round(cycle_min, 2),
            "machine_cost_usd": round(machine_cost, 2),
            "material_cost_usd": round(material_cost, 2),
            "overhead_usd": round(overhead, 2),
            "total_usd": round(machine_cost + material_cost + overhead, 2),
        }

    def generate_gcode(self) -> str:
        """Generate complete G-code program."""
        exporter = GCodeExporter(self.controller)
        return exporter.export(self)


# ---------------------------------------------------------------------------
# G-Code Exporter
# ---------------------------------------------------------------------------

class GCodeExporter:
    CONTROLLER_HEADERS = {
        "fanuc": [
            "%",
            "O1001 (MOSES PART)",
            "G90 G17 G20 G40 G49 G80",
            "G91 G28 Z0",
            "G90",
        ],
        "haas": [
            "%",
            "O01001 (MOSES PART)",
            "G90 G17 G20 G40 G49 G80",
            "G91 G28 Z0",
            "G90",
        ],
        "linuxcnc": [
            "( MOSES PART )",
            "G90 G17 G20 G40 G49 G80",
            "G54",
        ],
    }

    CONTROLLER_FOOTERS = {
        "fanuc": ["M30", "%"],
        "haas": ["M30", "%"],
        "linuxcnc": ["M2", "%"],
    }

    def __init__(self, controller: str):
        self.controller = controller.lower()
        if self.controller not in self.CONTROLLER_HEADERS:
            raise ValueError(f"Unsupported controller: {controller}")

    def export(self, job: CNCJob) -> str:
        lines = list(self.CONTROLLER_HEADERS[self.controller])
        lines.append(f"( Material: {MATERIAL_DB[job.material]['name']} )")
        lines.append(f"( Machine: {job.machine.upper()} )")
        lines.append(f"( Stock: {job.stock_size[0]} x {job.stock_size[1]} x {job.stock_size[2]} in )")

        for op in job.operations:
            lines.extend(op.generate(self.controller))
            lines.append("")

        lines.extend(self.CONTROLLER_FOOTERS[self.controller])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_face_mill(
    material: Material,
    width: float = 2.0,
    height: float = 2.0,
    depth: float = 0.010,
    tool_dia: float = 0.750,
) -> FaceMill:
    """Create a face-mill operation with auto-calculated speeds/feeds."""
    tool = next((t for t in TOOL_LIBRARY if abs(t.diameter_in - tool_dia) < 0.001), TOOL_LIBRARY[0])
    rpm, ipm, _ = FeedSpeedCalculator.for_material(material, tool, depth, tool_dia * 0.75)
    return FaceMill(tool=tool, rpm=rpm, feed_ipm=ipm, width=width, height=height, depth=depth)


def quick_drill(
    material: Material,
    x: float,
    y: float,
    z_bottom: float,
    drill_dia: float = 0.250,
    peck: Optional[float] = None,
) -> DrillCycle:
    tool = next((t for t in TOOL_LIBRARY if t.tool_type == ToolType.DRILL and abs(t.diameter_in - drill_dia) < 0.001), TOOL_LIBRARY[5])
    rpm, ipm, _ = FeedSpeedCalculator.for_material(material, tool)
    return DrillCycle(tool=tool, rpm=rpm, feed_ipm=ipm, x=x, y=y, z_bottom=z_bottom, peck_depth=peck)
