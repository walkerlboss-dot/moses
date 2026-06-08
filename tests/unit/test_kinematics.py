"""
test_kinematics.py — Unit tests for 7-DOF arm kinematics.

Verifies:
  - Forward kinematics consistency
  - Inverse kinematics convergence
  - Jacobian correctness via finite differences
  - Singularity detection near known singular configurations
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    """Tests for FK: joint angles → end-effector pose."""

    def test_fk_identity_at_zero(self, seven_dof_arm):
        """
        Verify that all-zero joint angles produce a deterministic end-effector pose.
        """
        q = np.zeros(7)
        T = seven_dof_arm.fk(q)
        assert T.shape == (4, 4)
        # Last row of homogeneous transform must be [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])
        # Rotation part should be identity when all angles are zero
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)

    def test_fk_translation_accumulation(self, seven_dof_arm):
        """
        Verify that FK accumulates link lengths along the x-axis for zero angles.
        """
        q = np.zeros(7)
        T = seven_dof_arm.fk(q)
        expected_x = sum(seven_dof_arm.link_lengths)
        assert np.isclose(T[0, 3], expected_x, atol=1e-6)
        assert np.isclose(T[1, 3], 0.0, atol=1e-6)

    def test_fk_rotation_nonzero(self, seven_dof_arm):
        """
        Verify that non-zero joint angles produce a non-identity rotation.
        """
        q = np.array([0.1, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0])
        T = seven_dof_arm.fk(q)
        assert not np.allclose(T[:3, :3], np.eye(3), atol=1e-3)

    def test_fk_batch_consistency(self, seven_dof_arm):
        """
        Verify that repeated FK calls with the same angles yield identical results.
        """
        q = np.random.randn(7) * 0.5
        T1 = seven_dof_arm.fk(q)
        T2 = seven_dof_arm.fk(q)
        assert np.allclose(T1, T2, atol=1e-12)


# ---------------------------------------------------------------------------
# Inverse Kinematics
# ---------------------------------------------------------------------------

class TestInverseKinematics:
    """Tests for IK: target pose → joint angles."""

    def test_ik_convergence_identity(self, seven_dof_arm):
        """
        Verify IK converges when the target is the FK of a known configuration.
        """
        q_true = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.0, -0.05])
        T_target = seven_dof_arm.fk(q_true)
        q_sol = seven_dof_arm.ik(T_target, initial_guess=np.zeros(7))
        T_sol = seven_dof_arm.fk(q_sol)
        pos_error = np.linalg.norm(T_target[:3, 3] - T_sol[:3, 3])
        assert pos_error < 1e-3, f"IK position error too large: {pos_error}"

    def test_ik_respects_joint_limits(self, seven_dof_arm):
        """
        Verify that IK solutions stay within joint limits.
        """
        q_true = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        T_target = seven_dof_arm.fk(q_true)
        q_sol = seven_dof_arm.ik(T_target, initial_guess=np.zeros(7))
        for i, (lo, hi) in enumerate(seven_dof_arm.joint_limits):
            assert lo <= q_sol[i] <= hi, f"Joint {i} out of bounds: {q_sol[i]} not in [{lo}, {hi}]"

    def test_ik_different_initial_guesses(self, seven_dof_arm):
        """
        Verify IK can converge from multiple initial guesses to the same target.
        """
        q_true = np.array([0.2, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1])
        T_target = seven_dof_arm.fk(q_true)
        guesses = [np.zeros(7), np.ones(7) * 0.1, np.ones(7) * -0.1]
        for guess in guesses:
            q_sol = seven_dof_arm.ik(T_target, initial_guess=guess)
            T_sol = seven_dof_arm.fk(q_sol)
            pos_error = np.linalg.norm(T_target[:3, 3] - T_sol[:3, 3])
            assert pos_error < 1e-2, f"IK failed from guess {guess[:3]}... error={pos_error}"


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

class TestJacobian:
    """Tests for geometric Jacobian computation."""

    def test_jacobian_shape(self, seven_dof_arm):
        """
        Verify Jacobian has shape (6, 7) for a 7-DOF arm.
        """
        q = np.zeros(7)
        J = seven_dof_arm.jacobian(q)
        assert J.shape == (6, 7)

    def test_jacobian_finite_difference_position(self, seven_dof_arm):
        """
        Verify the position part of the Jacobian via central finite differences.
        """
        q = np.array([0.1, -0.1, 0.2, 0.0, -0.05, 0.05, 0.0])
        J = seven_dof_arm.jacobian(q)
        J_pos = J[:3, :]  # 3x7
        eps = 1e-5
        for i in range(7):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            p_plus = seven_dof_arm.fk(q_plus)[:3, 3]
            p_minus = seven_dof_arm.fk(q_minus)[:3, 3]
            fd = (p_plus - p_minus) / (2 * eps)
            assert np.allclose(J_pos[:, i], fd, atol=1e-4), f"Jacobian column {i} mismatch"

    def test_jacobian_rank_at_nonsingular(self, seven_dof_arm):
        """
        Verify full rank (3) for the position Jacobian at a generic configuration.
        """
        q = np.array([0.2, -0.1, 0.3, 0.0, -0.1, 0.1, 0.0])
        J = seven_dof_arm.jacobian(q)
        rank = np.linalg.matrix_rank(J[:3, :], tol=1e-3)
        assert rank == 3, f"Expected rank 3, got {rank}"


# ---------------------------------------------------------------------------
# Singularity Detection
# ---------------------------------------------------------------------------

class TestSingularityDetection:
    """Tests for detecting kinematic singularities."""

    def test_nonsingular_generic_config(self, seven_dof_arm):
        """
        A generic configuration should NOT be flagged as singular.
        """
        q = np.array([0.2, -0.1, 0.3, 0.0, -0.1, 0.1, 0.0])
        assert not seven_dof_arm.is_singular(q, threshold=1e-3)

    def test_singular_collinear_joints(self, seven_dof_arm):
        """
        When all joint axes align (all zeros), the arm is in a singular configuration.
        """
        q = np.zeros(7)
        # All revolute joints about z-axis are collinear at zero → singular
        assert seven_dof_arm.is_singular(q, threshold=1e-2)

    def test_singular_threshold_sensitivity(self, seven_dof_arm):
        """
        Verify that raising the threshold makes more configurations appear singular.
        """
        q = np.zeros(7)
        assert not seven_dof_arm.is_singular(q, threshold=1e-6)
        assert seven_dof_arm.is_singular(q, threshold=1e-2)

    def test_singular_near_workspace_boundary(self, seven_dof_arm):
        """
        Near full extension the manipulability drops; verify detection.
        """
        # Push joints to nearly align
        q = np.array([0.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0])
        s_low = seven_dof_arm.is_singular(q, threshold=1e-4)
        s_high = seven_dof_arm.is_singular(q, threshold=1e-1)
        # With a very high threshold it should be flagged
        assert s_high is True
