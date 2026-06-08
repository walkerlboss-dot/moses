"""
moses/design/__init__.py
Moses v4.0 Humanoid Design Optimizer

Modules:
    structural_analysis  — FEA-style stress, deflection, buckling
    biomechanics         — Human ROM, gait, muscle curves, scaling
    weight_optimizer     — SIMP topology optimization, mass minimization
    cost_model           — BOM, manufacturing, assembly cost estimation

Usage:
    from moses.design import structural_analysis, biomechanics
    from moses.design.weight_optimizer import optimize_humanoid_weight
    from moses.design.cost_model import build_humanoid_bom
"""

__version__ = "4.0.0"
__all__ = [
    "structural_analysis",
    "biomechanics",
    "weight_optimizer",
    "cost_model",
]
