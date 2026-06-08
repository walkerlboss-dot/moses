"""
Moses v4.0 Sheet Metal Design Module
======================================
Bend allowance, flat pattern generation, punch/die selection,
and DXF export for laser / waterjet cutting.

Supports aluminum, mild steel, and stainless steel.
All units are inches unless noted.

References
----------
- Machinery's Handbook, 31st ed. (bend allowance tables)
- Amada / Trumpf punch press tooling catalogs
- Bystronic / Mazak laser cutting parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Material Database
# ---------------------------------------------------------------------------

class SheetMaterial(Enum):
    AL_5052_H32 = auto()
    AL_6061_T6 = auto()
    STEEL_CRS = auto()      # Cold Rolled Steel
    STEEL_HRS = auto()      # Hot Rolled Steel
    SS_304 = auto()         # Stainless 304
    SS_316 = auto()         # Stainless 316


SHEET_DB = {
    SheetMaterial.AL_5052_H32: {
        "name": "Aluminum 5052-H32",
        "density_lb_in3": 0.098,
        "yield_ksi": 28,
        "uts_ksi": 33,
        "elastic_modulus_ksi": 10_200,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.062,
        "cost_per_lb_usd": 2.80,
        "laser_kerf_in": 0.008,
        "waterjet_kerf_in": 0.040,
    },
    SheetMaterial.AL_6061_T6: {
        "name": "Aluminum 6061-T6",
        "density_lb_in3": 0.098,
        "yield_ksi": 40,
        "uts_ksi": 45,
        "elastic_modulus_ksi": 10_000,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.125,
        "cost_per_lb_usd": 3.50,
        "laser_kerf_in": 0.008,
        "waterjet_kerf_in": 0.040,
    },
    SheetMaterial.STEEL_CRS: {
        "name": "Cold Rolled Steel (1008/1010)",
        "density_lb_in3": 0.284,
        "yield_ksi": 45,
        "uts_ksi": 55,
        "elastic_modulus_ksi": 29_000,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.062,
        "cost_per_lb_usd": 0.85,
        "laser_kerf_in": 0.006,
        "waterjet_kerf_in": 0.035,
    },
    SheetMaterial.STEEL_HRS: {
        "name": "Hot Rolled Steel (A36)",
        "density_lb_in3": 0.284,
        "yield_ksi": 36,
        "uts_ksi": 58,
        "elastic_modulus_ksi": 29_000,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.125,
        "cost_per_lb_usd": 0.65,
        "laser_kerf_in": 0.006,
        "waterjet_kerf_in": 0.035,
    },
    SheetMaterial.SS_304: {
        "name": "Stainless Steel 304",
        "density_lb_in3": 0.289,
        "yield_ksi": 30,
        "uts_ksi": 75,
        "elastic_modulus_ksi": 28_000,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.125,
        "cost_per_lb_usd": 2.20,
        "laser_kerf_in": 0.006,
        "waterjet_kerf_in": 0.035,
    },
    SheetMaterial.SS_316: {
        "name": "Stainless Steel 316",
        "density_lb_in3": 0.289,
        "yield_ksi": 30,
        "uts_ksi": 75,
        "elastic_modulus_ksi": 28_000,
        "k_factor": 0.5,
        "bend_radius_min_in": 0.156,
        "cost_per_lb_usd": 3.50,
        "laser_kerf_in": 0.006,
        "waterjet_kerf_in": 0.035,
    },
}


# ---------------------------------------------------------------------------
# Bend Allowance Calculator
# ---------------------------------------------------------------------------

class BendAllowanceCalculator:
    """
    Calculate bend allowance, bend deduction, and setback for press brake
    and punch press operations.

    Bend Allowance (BA) = (π/180) × Bend_Angle × (R + K×T)
    Bend Deduction (BD) = 2 × (Tan(Angle/2) × (R + T)) - BA
    """

    @staticmethod
    def bend_allowance(
        angle_deg: float,
        radius_in: float,
        thickness_in: float,
        k_factor: float = 0.5,
    ) -> float:
        """Bend allowance in inches."""
        rad = math.radians(angle_deg)
        return rad * (radius_in + k_factor * thickness_in)

    @staticmethod
    def bend_deduction(
        angle_deg: float,
        radius_in: float,
        thickness_in: float,
        k_factor: float = 0.5,
    ) -> float:
        """Bend deduction in inches."""
        ba = BendAllowanceCalculator.bend_allowance(angle_deg, radius_in, thickness_in, k_factor)
        leg = math.tan(math.radians(angle_deg / 2)) * (radius_in + thickness_in)
        return 2 * leg - ba

    @staticmethod
    def setback(
        angle_deg: float,
        radius_in: float,
        thickness_in: float,
    ) -> float:
        """Setback for bend layout."""
        return (radius_in + thickness_in) * math.tan(math.radians(angle_deg / 2))

    @staticmethod
    def k_factor_from_bend_allowance(
        ba: float,
        angle_deg: float,
        radius_in: float,
        thickness_in: float,
    ) -> float:
        """Reverse-calculate K-factor from measured BA."""
        rad = math.radians(angle_deg)
        return (ba / rad - radius_in) / thickness_in


# ---------------------------------------------------------------------------
# Punch / Die Selector
# ---------------------------------------------------------------------------

class PunchDieSelector:
    """
    Select punch and die tooling for press brake and turret punch press.
    Based on Amada / Trumpf standard tooling.
    """

    PUNCH_LIBRARY = {
        # name: (punch_radius_in, min_bend_length_in, max_thickness_in, tonnage_per_inch)
        "acute_30": (0.008, 0.5, 0.125, 8.0),
        "acute_45": (0.016, 0.75, 0.187, 10.0),
        "gooseneck_85": (0.032, 1.0, 0.250, 12.0),
        "gooseneck_90": (0.040, 1.5, 0.375, 15.0),
        "hemming": (0.003, 2.0, 0.060, 20.0),
    }

    DIE_LIBRARY = {
        # name: (die_opening_in, min_bend_radius_in, max_thickness_in)
        "v_die_4": (0.25, 0.032, 0.125),
        "v_die_6": (0.375, 0.047, 0.187),
        "v_die_8": (0.500, 0.063, 0.250),
        "v_die_12": (0.750, 0.094, 0.375),
        "v_die_16": (1.000, 0.125, 0.500),
    }

    @classmethod
    def select_punch(cls, thickness_in: float, bend_angle_deg: float, bend_length_in: float) -> Optional[str]:
        for name, (pr, min_len, max_t, _) in cls.PUNCH_LIBRARY.items():
            if thickness_in <= max_t and bend_length_in >= min_len:
                if bend_angle_deg <= 45 and "acute" in name:
                    return name
                elif bend_angle_deg > 45 and "gooseneck" in name:
                    return name
        return "gooseneck_90"  # default

    @classmethod
    def select_die(cls, thickness_in: float, material: SheetMaterial) -> Optional[str]:
        # Rule of thumb: die opening = 6–8× thickness
        target_opening = thickness_in * 8.0
        best = None
        best_diff = float("inf")
        for name, (opening, _, max_t) in cls.DIE_LIBRARY.items():
            if thickness_in <= max_t:
                diff = abs(opening - target_opening)
                if diff < best_diff:
                    best_diff = diff
                    best = name
        return best

    @classmethod
    def tonnage(
        cls,
        thickness_in: float,
        bend_length_in: float,
        material: SheetMaterial,
        die_opening_in: float,
    ) -> float:
        """
        Press brake tonnage (tons) = (K × UTS × L × T²) / D
        K ≈ 1.33 for V-die air bending.
        """
        uts_ksi = SHEET_DB[material]["uts_ksi"]
        k = 1.33
        return (k * uts_ksi * bend_length_in * (thickness_in**2)) / die_opening_in


# ---------------------------------------------------------------------------
# Bend & Feature Definitions
# ---------------------------------------------------------------------------

@dataclass
class Bend:
    """A single bend in a sheet metal part."""

    angle_deg: float = 90.0
    radius_in: float = 0.062
    thickness_in: float = 0.125
    length_in: float = 2.0
    direction: str = "up"  # up / down
    k_factor: float = 0.5

    def allowance(self) -> float:
        return BendAllowanceCalculator.bend_allowance(
            self.angle_deg, self.radius_in, self.thickness_in, self.k_factor
        )

    def deduction(self) -> float:
        return BendAllowanceCalculator.bend_deduction(
            self.angle_deg, self.radius_in, self.thickness_in, self.k_factor
        )

    def tonnage(self, material: SheetMaterial) -> float:
        die = PunchDieSelector.select_die(self.thickness_in, material)
        die_opening = PunchDieSelector.DIE_LIBRARY[die][0] if die else 0.5
        return PunchDieSelector.tonnage(
            self.thickness_in, self.length_in, material, die_opening
        )


@dataclass
class Hole:
    """Punched or drilled hole."""

    x: float = 0.0
    y: float = 0.0
    diameter_in: float = 0.250
    is_punched: bool = True


@dataclass
class Slot:
    """Punched slot."""

    x: float = 0.0
    y: float = 0.0
    width_in: float = 0.250
    length_in: float = 0.500
    angle_deg: float = 0.0


# ---------------------------------------------------------------------------
# Flat Pattern Generator
# ---------------------------------------------------------------------------

class FlatPatternGenerator:
    """
    Generate flat pattern (unfolded) geometry from a bent part.
    Produces a list of 2D line segments and arcs.
    """

    @staticmethod
    def unfold(
        bends: List[Bend],
        leg_lengths: List[float],
    ) -> List[Tuple[str, Tuple]]:
        """
        Returns list of (type, coords) where type is 'line' or 'arc'.
        leg_lengths: lengths of straight sections between bends.
        """
        if len(leg_lengths) != len(bends) + 1:
            raise ValueError("leg_lengths must have len(bends)+1 elements")

        segments = []
        current_x = 0.0
        current_y = 0.0
        angle_accum = 0.0

        # Start leg
        segments.append(("line", (current_x, current_y, current_x + leg_lengths[0], current_y)))
        current_x += leg_lengths[0]

        for i, bend in enumerate(bends):
            ba = bend.allowance()
            # Arc center is offset by (R + T/2) in the bend direction
            r_eff = bend.radius_in + bend.thickness_in / 2.0
            cx = current_x + r_eff * math.sin(math.radians(angle_accum))
            cy = current_y - r_eff * math.cos(math.radians(angle_accum))
            # Arc spans the external bend angle (180 - internal_angle for air bend)
            ext_angle = 180.0 - bend.angle_deg
            segments.append(("arc", (cx, cy, r_eff, angle_accum, angle_accum + ext_angle)))
            angle_accum += ext_angle

            # Move to next leg start
            dx = ba * math.cos(math.radians(angle_accum))
            dy = ba * math.sin(math.radians(angle_accum))
            current_x += dx
            current_y += dy

            next_leg = leg_lengths[i + 1]
            nx = current_x + next_leg * math.cos(math.radians(angle_accum))
            ny = current_y + next_leg * math.sin(math.radians(angle_accum))
            segments.append(("line", (current_x, current_y, nx, ny)))
            current_x, current_y = nx, ny

        return segments

    @staticmethod
    def flat_length(bends: List[Bend], leg_lengths: List[float]) -> float:
        """Total flat pattern length."""
        total = sum(leg_lengths)
        for b in bends:
            total += b.allowance()
        return total


# ---------------------------------------------------------------------------
# Sheet Metal Part
# ---------------------------------------------------------------------------

@dataclass
class SheetMetalPart:
    """A complete sheet metal part definition."""

    material: SheetMaterial
    thickness_in: float
    bends: List[Bend] = field(default_factory=list)
    holes: List[Hole] = field(default_factory=list)
    slots: List[Slot] = field(default_factory=list)
    stock_width_in: float = 12.0
    stock_length_in: float = 24.0
    cut_method: str = "laser"  # laser / waterjet / punch
    hourly_rate_usd: float = 75.0  # laser/waterjet hourly rate
    press_brake_rate_usd: float = 65.0  # press brake hourly rate

    def add_bend(self, bend: Bend) -> "SheetMetalPart":
        self.bends.append(bend)
        return self

    def add_hole(self, hole: Hole) -> "SheetMetalPart":
        self.holes.append(hole)
        return self

    def flat_pattern(self, leg_lengths: List[float]) -> List[Tuple]:
        return FlatPatternGenerator.unfold(self.bends, leg_lengths)

    def flat_length(self, leg_lengths: List[float]) -> float:
        return FlatPatternGenerator.flat_length(self.bends, leg_lengths)

    def tonnage_total(self) -> float:
        return sum(b.tonnage(self.material) for b in self.bends)

    def cutting_time_min(self, perimeter_in: float) -> float:
        """Estimate laser/waterjet cutting time from perimeter."""
        # Typical feed rates (in/min)
        rates = {
            "laser": {"aluminum": 400, "steel": 300, "stainless": 250},
            "waterjet": {"aluminum": 15, "steel": 10, "stainless": 8},
        }
        mat_name = SHEET_DB[self.material]["name"].lower()
        if "aluminum" in mat_name:
            key = "aluminum"
        elif "stainless" in mat_name:
            key = "stainless"
        else:
            key = "steel"
        rate = rates[self.cut_method][key]
        return perimeter_in / rate

    def estimate_cost(self, leg_lengths: List[float], perimeter_in: float) -> dict:
        """Full cost estimate."""
        db = SHEET_DB[self.material]
        flat_len = self.flat_length(leg_lengths)
        flat_width = max(self.stock_width_in, flat_len)  # simplified
        area_in2 = flat_len * flat_width
        volume_in3 = area_in2 * self.thickness_in
        weight_lb = volume_in3 * db["density_lb_in3"]

        material_cost = weight_lb * db["cost_per_lb_usd"]
        cut_time_hr = self.cutting_time_min(perimeter_in) / 60.0
        cut_cost = cut_time_hr * self.hourly_rate_usd

        # Bend time: ~30 sec per bend + setup 5 min
        bend_time_hr = (5.0 + len(self.bends) * 0.5) / 60.0
        bend_cost = bend_time_hr * self.press_brake_rate_usd

        overhead = (cut_cost + bend_cost) * 0.20
        total = material_cost + cut_cost + bend_cost + overhead

        return {
            "flat_length_in": round(flat_len, 3),
            "flat_width_in": round(flat_width, 3),
            "weight_lb": round(weight_lb, 3),
            "material_cost_usd": round(material_cost, 2),
            "cut_time_min": round(cut_time_hr * 60, 2),
            "cut_cost_usd": round(cut_cost, 2),
            "bend_time_min": round(bend_time_hr * 60, 2),
            "bend_cost_usd": round(bend_cost, 2),
            "overhead_usd": round(overhead, 2),
            "total_cost_usd": round(total, 2),
            "tonnage_required": round(self.tonnage_total(), 1),
        }


# ---------------------------------------------------------------------------
# DXF Exporter
# ---------------------------------------------------------------------------

class DXFExporter:
    """
    Minimal DXF R12 text exporter for 2D flat patterns.
    No external dependencies.
    """

    HEADER = """  0
SECTION
  2
HEADER
  9
$ACADVER
  1
AC1009
  0
ENDSEC
  0
SECTION
  2
TABLES
  0
ENDSEC
  0
SECTION
  2
BLOCKS
  0
ENDSEC
  0
SECTION
  2
ENTITIES
"""

    FOOTER = """  0
ENDSEC
  0
SECTION
  2
OBJECTS
  0
ENDSEC
  0
EOF
"""

    @classmethod
    def export_flat_pattern(
        cls,
        segments: List[Tuple[str, Tuple]],
        holes: List[Hole],
        slots: List[Slot],
        out_path: str,
    ) -> str:
        """Write flat pattern to DXF."""
        lines = [cls.HEADER]

        for seg_type, coords in segments:
            if seg_type == "line":
                x1, y1, x2, y2 = coords
                lines.append(cls._line(x1, y1, x2, y2))
            elif seg_type == "arc":
                cx, cy, r, start_deg, end_deg = coords
                lines.append(cls._arc(cx, cy, r, start_deg, end_deg))

        for h in holes:
            lines.append(cls._circle(h.x, h.y, h.diameter_in / 2.0))

        for s in slots:
            lines.append(cls._slot(s.x, s.y, s.width_in, s.length_in, s.angle_deg))

        lines.append(cls.FOOTER)

        with open(out_path, "w") as fh:
            fh.write("\n".join(lines))
        return out_path

    @staticmethod
    def _line(x1: float, y1: float, x2: float, y2: float) -> str:
        return f"""  0
LINE
  8
0
 10
{x1:.4f}
 20
{y1:.4f}
 11
{x2:.4f}
 21
{y2:.4f}"""

    @staticmethod
    def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
        return f"""  0
ARC
  8
0
 10
{cx:.4f}
 20
{cy:.4f}
 40
{r:.4f}
 50
{start_deg:.4f}
 51
{end_deg:.4f}"""

    @staticmethod
    def _circle(cx: float, cy: float, r: float) -> str:
        return f"""  0
CIRCLE
  8
0
 10
{cx:.4f}
 20
{cy:.4f}
 40
{r:.4f}"""

    @staticmethod
    def _slot(cx: float, cy: float, width: float, length: float, angle_deg: float) -> str:
        """Slot as two semicircles + two lines (simplified as polyline)."""
        # For DXF R12, approximate with a closed polyline
        rad = math.radians(angle_deg)
        dx = (length / 2) * math.cos(rad)
        dy = (length / 2) * math.sin(rad)
        r = width / 2
        # Endpoints
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy
        # Perpendicular offset for width
        px = -dy / (length / 2) * r
        py = dx / (length / 2) * r
        pts = [
            (x1 + px, y1 + py),
            (x2 + px, y2 + py),
            (x2 - px, y2 - py),
            (x1 - px, y1 - py),
        ]
        return DXFExporter._lwpolyline(pts, closed=True)

    @staticmethod
    def _lwpolyline(points: List[Tuple[float, float]], closed: bool = False) -> str:
        lines = ["  0", "LWPOLYLINE", "  8", "0", " 90", str(len(points)), " 70", "1" if closed else "0"]
        for x, y in points:
            lines.extend([" 10", f"{x:.4f}", " 20", f"{y:.4f}"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_bracket(
    material: SheetMaterial = SheetMaterial.AL_5052_H32,
    thickness_in: float = 0.125,
    width_in: float = 2.0,
    leg1_in: float = 3.0,
    leg2_in: float = 3.0,
    angle_deg: float = 90.0,
) -> SheetMetalPart:
    """Create a simple L-bracket."""
    part = SheetMetalPart(material=material, thickness_in=thickness_in)
    bend = Bend(angle_deg=angle_deg, radius_in=0.062, thickness_in=thickness_in, length_in=width_in)
    part.add_bend(bend)
    return part
