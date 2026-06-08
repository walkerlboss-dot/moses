"""
test_checkpoint.py — Unit tests for the human-in-the-loop checkpoint system.

Verifies:
  - Checkpoint save/load roundtrip preserves all fields
  - Resuming from a checkpoint restores correct state
  - Corrupted checkpoint files are handled gracefully
  - Timeout detection and safe-default actions
  - Override permissions and audit trails
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Import the checkpoint module from the integration package
sys_path_hack = str(Path(__file__).resolve().parents[2] / "integration")
import sys
if sys_path_hack not in sys.path:
    sys.path.insert(0, sys_path_hack)

from human_checkpoint import (
    Checkpoint,
    CheckpointEngine,
    GateType,
    GateStatus,
    Urgency,
    Tier3Rule,
    TIER3_RULEBOOK,
    HITLGuard,
)


# ---------------------------------------------------------------------------
# Save / Load Roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    """Tests for checkpoint persistence roundtrip."""

    def test_save_and_load(self, tmp_checkpoint_dir):
        """
        Creating a checkpoint, saving it, and loading from disk must preserve
        all fields including gate_type, status, urgency, and context.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        evidence = {"sim_report.json": Path("/tmp/fake_sim.json")}
        context = {"target_environment": "physical", "sim_success_rate": 87.0}

        cp = engine.create_checkpoint(
            gate_type=GateType.PHYSICAL_DEPLOY,
            requested_by="moses",
            context=context,
            evidence=evidence,
            urgency=Urgency.URGENT,
            timeout_hours=12.0,
        )

        # Force reload by creating a new engine instance
        engine2 = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp2 = engine2.active.get(cp.checkpoint_id)
        assert cp2 is not None
        assert cp2.gate_type == GateType.PHYSICAL_DEPLOY
        assert cp2.status == GateStatus.PENDING
        assert cp2.urgency == Urgency.URGENT
        assert cp2.requested_by == "moses"
        assert cp2.agent_context == context
        assert cp2.evidence_paths == evidence

    def test_json_serialization(self):
        """
        Checkpoint.to_json() and Checkpoint.from_json() must be exact inverses.
        """
        now = datetime.now(timezone.utc)
        cp = Checkpoint(
            checkpoint_id="abc12345",
            gate_type=GateType.SAFETY_OVERRIDE,
            status=GateStatus.PENDING,
            urgency=Urgency.NORMAL,
            requested_by="titan",
            requested_at=now,
            timeout_at=now + timedelta(hours=24),
            evidence_paths={"report.json": Path("/shared/report.json")},
            agent_context={"risk_score": 85},
        )
        data = cp.to_json()
        cp2 = Checkpoint.from_json(data)
        assert cp2.checkpoint_id == cp.checkpoint_id
        assert cp2.gate_type == cp.gate_type
        assert cp2.status == cp.status
        assert cp2.urgency == cp.urgency
        assert cp2.requested_at == cp.requested_at
        assert cp2.timeout_at == cp.timeout_at
        assert cp2.evidence_paths == cp.evidence_paths
        assert cp2.agent_context == cp.agent_context

    def test_load_ignores_non_pending(self, tmp_checkpoint_dir):
        """
        Only PENDING and TIMED_OUT checkpoints are loaded into active on startup.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.BUDGET_EXCEED,
            requested_by="moses",
            context={"projected_spend": 600.0},
            evidence={},
        )
        engine.process_response(cp.checkpoint_id, "APPROVE")

        engine2 = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        assert cp.checkpoint_id not in engine2.active


# ---------------------------------------------------------------------------
# Resume from Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    """Tests for resuming workflow from a saved checkpoint."""

    def test_resume_pending(self, tmp_checkpoint_dir):
        """
        A checkpoint in PENDING status can be resumed (queried) by a new engine.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.POLICY_UNTESTED,
            requested_by="moses",
            context={"policy_sim_runs": 0},
            evidence={},
        )
        engine2 = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        guard = HITLGuard(engine2)
        assert guard.is_pending(cp.checkpoint_id) is True
        assert guard.is_approved(cp.checkpoint_id) is False

    def test_resume_after_approval(self, tmp_checkpoint_dir):
        """
        After approval, the checkpoint file on disk must reflect APPROVED status.
        Note: _load_active only loads PENDING/TIMED_OUT, so we verify via direct file read.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.HARDWARE_PROCURE,
            requested_by="moses",
            context={"item_cost_usd": 750},
            evidence={},
        )
        engine.process_response(cp.checkpoint_id, "APPROVE")

        # Verify by reading the file directly
        cp_path = tmp_checkpoint_dir / f"{cp.checkpoint_id}.json"
        assert cp_path.exists()
        data = json.loads(cp_path.read_text())
        assert data["status"] == "approved"
        assert data["alex_response"] == "APPROVE"

    def test_safe_default_on_timeout(self, tmp_checkpoint_dir):
        """
        The safe default action for a timed-out checkpoint must match the rulebook.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.EMERGENCY_STOP,
            requested_by="moses",
            context={"estop_triggered": True},
            evidence={},
            timeout_hours=0.0,  # immediate timeout for test
        )
        # Manually set timeout in the past
        cp.timeout_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        engine._save(cp)

        engine2 = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        timed_out = engine2.check_timeouts()
        assert len(timed_out) == 1
        assert timed_out[0].status == GateStatus.TIMED_OUT

        guard = HITLGuard(engine2)
        default = guard.safe_default_action(cp.checkpoint_id)
        rule = next(r for r in TIER3_RULEBOOK if r.gate_type == GateType.EMERGENCY_STOP)
        assert default == rule.safe_default_on_timeout


# ---------------------------------------------------------------------------
# Corruption Handling
# ---------------------------------------------------------------------------

class TestCheckpointCorruption:
    """Tests for graceful handling of corrupted checkpoint files."""

    def test_corrupt_json_skipped(self, tmp_checkpoint_dir):
        """
        A corrupted JSON file in the checkpoint directory must not crash the engine;
        it should be logged and skipped during load.
        """
        bad_file = tmp_checkpoint_dir / "bad.json"
        bad_file.write_text("this is not json {{{")

        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        # Should not raise; active dict may be empty
        assert isinstance(engine.active, dict)

    def test_missing_checkpoint_raises(self, tmp_checkpoint_dir):
        """
        Processing a response for a non-existent checkpoint must raise ValueError.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        with pytest.raises(ValueError, match="not found"):
            engine.process_response("nonexistent", "APPROVE")

    def test_invalid_response_format(self, tmp_checkpoint_dir):
        """
        An unrecognized response string must raise ValueError.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.AGENT_CONFLICT,
            requested_by="moses",
            context={"conflict_iterations": 2},
            evidence={},
        )
        with pytest.raises(ValueError, match="Unknown response format"):
            engine.process_response(cp.checkpoint_id, "MAYBE")

    def test_override_not_allowed(self, tmp_checkpoint_dir):
        """
        Attempting to override a checkpoint that does not allow overrides must raise PermissionError.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.SAFETY_OVERRIDE,
            requested_by="moses",
            context={"risk_score": 85},
            evidence={},
        )
        with pytest.raises(PermissionError, match="Override not allowed"):
            engine.apply_override(cp.checkpoint_id, override_by="alex", justification="test")

    def test_override_allowed(self, tmp_checkpoint_dir):
        """
        Overriding a checkpoint that explicitly allows overrides must succeed and log audit data.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.BUDGET_EXCEED,
            requested_by="moses",
            context={"projected_spend": 600.0},
            evidence={},
        )
        result = engine.apply_override(
            cp.checkpoint_id,
            override_by="alex",
            justification="Critical hardware needed for milestone",
        )
        assert result.status == GateStatus.OVERRIDDEN
        assert result.override_record is not None
        assert result.override_record["overridden_by"] == "alex"
        assert "Critical hardware" in result.override_record["justification"]


# ---------------------------------------------------------------------------
# Engine Summary
# ---------------------------------------------------------------------------

class TestCheckpointSummary:
    """Tests for checkpoint engine summary statistics."""

    def test_summary_counts(self, tmp_checkpoint_dir):
        """
        get_summary() must return accurate counts per status.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp1 = engine.create_checkpoint(
            gate_type=GateType.PHYSICAL_DEPLOY,
            requested_by="moses",
            context={},
            evidence={},
        )
        cp2 = engine.create_checkpoint(
            gate_type=GateType.BUDGET_EXCEED,
            requested_by="moses",
            context={},
            evidence={},
        )
        engine.process_response(cp1.checkpoint_id, "APPROVE")
        engine.process_response(cp2.checkpoint_id, "DENY")

        summary = engine.get_summary()
        assert summary.get("approved", 0) == 1
        assert summary.get("denied", 0) == 1
        assert summary.get("pending", 0) == 0

    def test_get_pending_filter(self, tmp_checkpoint_dir):
        """
        get_pending() must return only checkpoints with PENDING status.
        """
        engine = CheckpointEngine(storage_path=tmp_checkpoint_dir)
        cp = engine.create_checkpoint(
            gate_type=GateType.CROSS_SILO_WRITE,
            requested_by="moses",
            context={},
            evidence={},
        )
        pending = engine.get_pending()
        assert len(pending) == 1
        engine.process_response(cp.checkpoint_id, "APPROVE")
        pending = engine.get_pending()
        assert len(pending) == 0
