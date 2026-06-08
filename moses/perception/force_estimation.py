"""
moses/perception/force_estimation.py
Force Estimation Module — Moses v6.0 Perception Stack

Estimates external forces from proprioception, contact force distribution,
collision detection from torque residuals, and force-control integration.

References:
- De Luca & Mattone, "Sensorless robot collision detection with nonlinear
  observers", ICRA 2004.
- Haddadin et al., "Collision detection and reaction", IJRR 2017.
- Ott et al., "Cartesian impedance control", ICRA 2010.
- Featherstone, "Rigid Body Dynamics Algorithms", Springer 2008.
- Siciliano & Khatib (eds.), "Springer Handbook of Robotics", 2016.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import deque

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# 1. Rigid-Body Dynamics Helpers
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """Joint-level state snapshot."""
    q: FloatArray      # positions (n,)
    dq: FloatArray     # velocities (n,)
    ddq: FloatArray    # accelerations (n,)
    tau: FloatArray    # measured torques (n,)
    timestamp: float = 0.0


@dataclass
class ExternalWrench:
    """6-DOF wrench at a specific link/frame."""
    force: FloatArray     # (3,)
    torque: FloatArray    # (3,)
    link_id: int = -1
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# 2. Inverse Dynamics & Torque Residuals
# ---------------------------------------------------------------------------

class InverseDynamics:
    """
    Recursive Newton-Euler Algorithm (RNEA).
    Featherstone, "Rigid Body Dynamics Algorithms", Springer 2008.

    For an n-link serial manipulator:
        tau = M(q) ddq + C(q, dq) dq + g(q) + tau_ext
    """

    def __init__(self, n_dof: int, link_masses: List[float], link_inertias: List[FloatArray]):
        self.n = n_dof
        self.m = np.array(link_masses, dtype=np.float64)
        self.I = link_inertias  # list of (3,3) inertia tensors

    def compute_rnea(self, state: RobotState, gravity: FloatArray = np.array([0, 0, -9.81])) -> FloatArray:
        """
        Simplified RNEA for a revolute-joint chain aligned with z-axis.
        Returns expected joint torques tau_model.
        """
        # This is a minimal symbolic implementation.
        # Production code uses Pinocchio / RBDL for full kinematics.
        tau_model = np.zeros(self.n, dtype=np.float64)
        # Gravity term only (simplified diagonal mass matrix)
        for i in range(self.n):
            tau_model[i] = self.m[i] * np.linalg.norm(gravity) * np.sin(state.q[i]) * 0.1
        # Coriolis / centrifugal (simplified)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                tau_model[i] += 0.05 * state.dq[j] ** 2 * np.sin(state.q[i] - state.q[j])
        # Inertial
        tau_model += self.m * 0.1 * state.ddq  # approximate diagonal M
        return tau_model


# ---------------------------------------------------------------------------
# 3. External Force Estimation from Proprioception
# ---------------------------------------------------------------------------

class ExternalForceEstimator:
    """
    Estimate external joint torques via momentum observer.
    De Luca & Mattone, ICRA 2004; Haddadin et al., IJRR 2017.

    Observer dynamics:
        r = K_I ( p - ∫(tau + tau_ext - g(q)) dt )
    where p = M(q) dq is generalized momentum.
    """

    def __init__(self, n_dof: int, observer_gain: float = 20.0, dt: float = 0.001):
        self.n = n_dof
        self.K = observer_gain
        self.dt = dt
        self.p_int = np.zeros(n_dof, dtype=np.float64)
        self.r = np.zeros(n_dof, dtype=np.float64)
        self.history: deque[FloatArray] = deque(maxlen=100)

    def update(self, state: RobotState, idyn: InverseDynamics, gravity: FloatArray = np.array([0, 0, -9.81])) -> FloatArray:
        """
        Returns estimated external joint torques tau_ext.
        """
        tau_model = idyn.compute_rnea(state, gravity)
        # Generalised momentum (simplified diagonal mass)
        M_diag = idyn.m * 0.1
        p = M_diag * state.dq
        # Observer integration
        self.p_int += (state.tau - tau_model + self.r) * self.dt
        self.r = self.K * (p - self.p_int)
        self.history.append(self.r.copy())
        return self.r.copy()

    def get_smoothed_estimate(self, window: int = 10) -> FloatArray:
        if len(self.history) < window:
            return self.r.copy()
        return np.mean(list(self.history)[-window:], axis=0)


# ---------------------------------------------------------------------------
# 4. Contact Force Distribution
# ---------------------------------------------------------------------------

class ContactForceDistribution:
    """
    Map estimated joint torques to end-effector / contact-space forces.
    Uses the manipulator Jacobian transpose mapping:
        tau = J^T F
    Solve for F via least-squares or QP with friction constraints.
    """

    def __init__(self, n_dof: int):
        self.n = n_dof

    def jacobian_transpose_solve(self, tau_ext: FloatArray, J: FloatArray) -> ExternalWrench:
        """
        F = (J J^T)^{-1} J tau_ext   (damped least-squares)
        """
        # Damped pseudo-inverse
        damping = 0.01 ** 2 * np.eye(6)
        JJT = J @ J.T + damping
        F = np.linalg.solve(JJT, J @ tau_ext)
        return ExternalWrench(force=F[:3], torque=F[3:6])

    def distribute_contact_forces(
        self,
        tau_ext: FloatArray,
        contact_jacobians: List[FloatArray],
        friction_cones: Optional[List[Tuple[float, float]]] = None,
    ) -> List[ExternalWrench]:
        """
        Multi-contact force distribution via QP:
            min || sum(J_i^T f_i) - tau_ext ||^2
            s.t.  f_i inside friction cone

        Simplified here to unconstrained least-squares stacking.
        """
        m = len(contact_jacobians)
        J_stack = np.hstack(contact_jacobians)  # (n, 6*m)
        # Damped least-squares for stacked forces
        damping = 0.001 * np.eye(J_stack.shape[1])
        f_all = np.linalg.solve(J_stack.T @ J_stack + damping, J_stack.T @ tau_ext)
        wrenches: List[ExternalWrench] = []
        for i in range(m):
            fi = f_all[i * 6:(i + 1) * 6]
            wrenches.append(ExternalWrench(force=fi[:3], torque=fi[3:6], link_id=i))
        return wrenches


# ---------------------------------------------------------------------------
# 5. Collision Detection from Torque Residuals
# ---------------------------------------------------------------------------

class CollisionDetector:
    """
    Detect and localise collisions using momentum-observer residuals.
    Haddadin et al., IJRR 2017.
    """

    def __init__(self, n_dof: int, threshold: float = 2.0, dt: float = 0.001):
        self.n = n_dof
        self.threshold = threshold  # Nm
        self.dt = dt
        self.estimator = ExternalForceEstimator(n_dof, observer_gain=20.0, dt=dt)
        self.collision_active = False
        self.collision_joint: Optional[int] = None
        self.collision_start_time: Optional[float] = None

    def update(self, state: RobotState, idyn: InverseDynamics) -> Dict[str, any]:
        tau_ext = self.estimator.update(state, idyn)
        mag = np.abs(tau_ext)
        max_mag = float(mag.max())
        max_joint = int(np.argmax(mag))

        detected = max_mag > self.threshold
        if detected and not self.collision_active:
            self.collision_active = True
            self.collision_joint = max_joint
            self.collision_start_time = state.timestamp

        if not detected and self.collision_active:
            self.collision_active = False
            self.collision_joint = None

        return {
            "collision": detected,
            "joint": max_joint,
            "magnitude": max_mag,
            "tau_ext": tau_ext.copy(),
            "active": self.collision_active,
        }


# ---------------------------------------------------------------------------
# 6. Force Control Integration
# ---------------------------------------------------------------------------

class AdmittanceController:
    """
    Admittance control: desired dynamics in Cartesian space.
    Ott et al., ICRA 2010.

    M_d ddX + D_d dX + K_d dX = F_ext
    where dX = X - X_des.
    """

    def __init__(
        self,
        mass: FloatArray = np.eye(3) * 5.0,
        damping: FloatArray = np.eye(3) * 100.0,
        stiffness: FloatArray = np.eye(3) * 500.0,
        dt: float = 0.001,
    ):
        self.M = mass
        self.D = damping
        self.K = stiffness
        self.dt = dt
        self.x = np.zeros(3, dtype=np.float64)
        self.dx = np.zeros(3, dtype=np.float64)

    def step(self, F_ext: FloatArray, x_des: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Integrate admittance dynamics one step.
        Returns (x, dx) — desired pose and velocity corrections.
        """
        # Error state
        e = self.x - x_des
        de = self.dx
        # M ddq + D dq + K q = F
        # Explicit Euler for demo; production uses implicit integration
        ddx = np.linalg.solve(self.M, F_ext - self.D @ de - self.K @ e)
        self.dx += ddx * self.dt
        self.x += self.dx * self.dt
        return self.x.copy(), self.dx.copy()


class ImpedanceController:
    """
    Impedance control: command torque to realise desired Cartesian impedance.
    tau = J^T ( K (x_des - x) + D (dx_des - dx) - F_ext ) + tau_gravity
    """

    def __init__(
        self,
        K: FloatArray = np.eye(6) * 500.0,
        D: FloatArray = np.eye(6) * 50.0,
    ):
        self.K = K
        self.D = D

    def compute_torque(
        self,
        x: FloatArray,
        dx: FloatArray,
        x_des: FloatArray,
        dx_des: FloatArray,
        F_ext: ExternalWrench,
        J: FloatArray,
        tau_gravity: FloatArray,
    ) -> FloatArray:
        """
        Returns joint torque command.
        """
        x_err = np.concatenate([x_des[:3] - x[:3], np.zeros(3)])  # orientation simplified
        dx_err = np.concatenate([dx_des[:3] - dx[:3], np.zeros(3)])
        F_cmd = self.K @ x_err + self.D @ dx_err
        F_cmd[:3] -= F_ext.force
        F_cmd[3:6] -= F_ext.torque
        tau = J.T @ F_cmd + tau_gravity
        return tau
