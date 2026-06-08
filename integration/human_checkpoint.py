#!/usr/bin/env python3
"""
Human-in-the-Loop System (HITL)

Enforces Tier 3 gates for the Moses-Titan robotics pipeline.
Defines when Alex MUST review before proceeding, formats review requests,
handles timeouts, and provides override capabilities.

Reference: integration_spec.md §3.2, AGENTS.md §Safety Levels & Approvals
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("human_checkpoint")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_ROOT = Path("/shared/checkpoints")
APPROVALS_TOPIC = "approvals"  # Telegram Thread ID 8
DEFAULT_TIMEOUT_HOURS = 24.0
MAX_TIMEOUT_HOURS = 72.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GateType(Enum):
    """Categories of human checkpoint."""
    PHYSICAL_DEPLOY = "physical_deploy"          # Deploy to real robot
    SAFETY_OVERRIDE = "safety_override"          # Override Titan INFEASIBLE
    BUDGET_EXCEED = "budget_exceed"              # Spend > approved envelope
    DESIGN_BREAKING = "design_breaking"          # Change that breaks prior cert
    POLICY_UNTESTED = "policy_untested"          # Policy never run in sim
    HARDWARE_PROCURE = "hardware_procure"        # Buy parts > $500
    EMERGENCY_STOP = "emergency_stop"            # E-stop reset / resume
    AGENT_CONFLICT = "agent_conflict"            # Titan vs Moses unresolved
    CROSS_SILO_WRITE = "cross_silo_write"        # Write across silo boundary

class GateStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    MODIFIED = "modified"
    TIMED_OUT = "timed_out"
    OVERRIDDEN = "overridden"
    ABORTED = "aborted"

class Urgency(Enum):
    LOW = "low"
    NORMAL = "normal"
    URGENT = "urgent"
    EMERGENCY = "emergency"

# ---------------------------------------------------------------------------
# Tier 3 Gate Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier3Rule:
    """A single rule that triggers a Tier 3 gate."""
    gate_type: GateType
    description: str
    auto_conditions: List[str]  # Conditions that auto-trigger
    required_evidence: List[str]  # What must be attached
    default_timeout_hours: float
    allow_override: bool
    safe_default_on_timeout: str  # "abort", "hold", "degraded"


TIER3_RULEBOOK: List[Tier3Rule] = [
    Tier3Rule(
        gate_type=GateType.PHYSICAL_DEPLOY,
        description="Deploy trained policy or firmware to physical humanoid robot",
        auto_conditions=[
            "target_environment == 'physical'",
            "sim_success_rate < 95%",
        ],
        required_evidence=[
            "sim_report.json",
            "safety_analysis.json",
            "rollback_plan.md",
        ],
        default_timeout_hours=24.0,
        allow_override=False,
        safe_default_on_timeout="abort",
    ),
    Tier3Rule(
        gate_type=GateType.SAFETY_OVERRIDE,
        description="Proceed with design despite Titan INFEASIBLE or risk_score >= 80",
        auto_conditions=[
            "titan_status == 'INFEASIBLE'",
            "risk_score >= 80",
            "fail_modes contains 'structural_collapse'",
        ],
        required_evidence=[
            "titan_feasibility_report.json",
            "moses_counter_argument.md",
            "mitigation_plan.json",
        ],
        default_timeout_hours=24.0,
        allow_override=False,  # Safety never overridden without human
        safe_default_on_timeout="abort",
    ),
    Tier3Rule(
        gate_type=GateType.BUDGET_EXCEED,
        description="Exceed approved compute or hardware budget by >10%",
        auto_conditions=[
            "projected_spend > approved_amount * 1.10",
        ],
        required_evidence=[
            "budget_request.json",
            "spend_forecast.json",
            "justification.md",
        ],
        default_timeout_hours=12.0,
        allow_override=True,  # Alex can approve overage
        safe_default_on_timeout="hold",
    ),
    Tier3Rule(
        gate_type=GateType.DESIGN_BREAKING,
        description="Design change invalidates prior safety certification",
        auto_conditions=[
            "certified_design_hash != current_design_hash",
            "component_in_cert_changed == true",
        ],
        required_evidence=[
            "design_diff.json",
            "prior_cert.json",
            "re_cert_plan.md",
        ],
        default_timeout_hours=24.0,
        allow_override=False,
        safe_default_on_timeout="hold",
    ),
    Tier3Rule(
        gate_type=GateType.POLICY_UNTESTED,
        description="Deploy policy that has never passed full sim suite",
        auto_conditions=[
            "policy_sim_runs == 0",
            "policy_eval_score is null",
        ],
        required_evidence=[
            "policy_metadata.json",
            "training_log.json",
            "test_plan.md",
        ],
        default_timeout_hours=24.0,
        allow_override=False,
        safe_default_on_timeout="abort",
    ),
    Tier3Rule(
        gate_type=GateType.HARDWARE_PROCURE,
        description="Purchase robotics hardware or components > $500",
        auto_conditions=[
            "item_cost_usd > 500",
        ],
        required_evidence=[
            "vendor_quote.pdf",
            "bom.json",
            "procurement_justification.md",
        ],
        default_timeout_hours=48.0,
        allow_override=True,
        safe_default_on_timeout="hold",
    ),
    Tier3Rule(
        gate_type=GateType.EMERGENCY_STOP,
        description="Reset or resume after emergency stop triggered",
        auto_conditions=[
            "estop_triggered == true",
            "resume_requested == true",
        ],
        required_evidence=[
            "incident_report.json",
            "root_cause.md",
            "safe_resume_checklist.json",
        ],
        default_timeout_hours=4.0,
        allow_override=False,
        safe_default_on_timeout="hold",
    ),
    Tier3Rule(
        gate_type=GateType.AGENT_CONFLICT,
        description="Unresolved conflict between Titan and Moses after 2 iterations",
        auto_conditions=[
            "conflict_iterations >= 2",
            "resolver_output == 'ESCALATE'",
        ],
        required_evidence=[
            "titan_report.json",
            "moses_counter.md",
            "iteration_log.jsonl",
        ],
        default_timeout_hours=24.0,
        allow_override=True,
        safe_default_on_timeout="hold",
    ),
    Tier3Rule(
        gate_type=GateType.CROSS_SILO_WRITE,
        description="Write operation across silo boundary (e.g., robotics → finance)",
        auto_conditions=[
            "operation_type == 'write'",
            "source_silo != target_silo",
        ],
        required_evidence=[
            "operation_description.md",
            "data_classification.json",
            "risk_assessment.md",
        ],
        default_timeout_hours=24.0,
        allow_override=True,
        safe_default_on_timeout="abort",
    ),
]

# ---------------------------------------------------------------------------
# Checkpoint Data Model
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """A single human checkpoint instance."""
    checkpoint_id: str
    gate_type: GateType
    status: GateStatus
    urgency: Urgency
    requested_by: str  # agent_id
    requested_at: datetime
    timeout_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None  # "alex", "system_timeout", "override"
    evidence_paths: Dict[str, Path] = field(default_factory=dict)
    agent_context: Dict[str, Any] = field(default_factory=dict)
    alex_response: Optional[str] = None  # "APPROVE", "DENY", "MODIFY: ..."
    modification_notes: Optional[str] = None
    override_record: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "urgency": self.urgency.value,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "evidence_paths": {k: str(v) for k, v in self.evidence_paths.items()},
            "agent_context": self.agent_context,
            "alex_response": self.alex_response,
            "modification_notes": self.modification_notes,
            "override_record": self.override_record,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Checkpoint:
        return cls(
            checkpoint_id=data["checkpoint_id"],
            gate_type=GateType(data["gate_type"]),
            status=GateStatus(data["status"]),
            urgency=Urgency(data["urgency"]),
            requested_by=data["requested_by"],
            requested_at=datetime.fromisoformat(data["requested_at"]),
            timeout_at=datetime.fromisoformat(data["timeout_at"]),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolved_by=data.get("resolved_by"),
            evidence_paths={k: Path(v) for k, v in data.get("evidence_paths", {}).items()},
            agent_context=data.get("agent_context", {}),
            alex_response=data.get("alex_response"),
            modification_notes=data.get("modification_notes"),
            override_record=data.get("override_record"),
        )


# ---------------------------------------------------------------------------
# Review Request Formatter
# ---------------------------------------------------------------------------

class ReviewFormatter:
    """Formats checkpoint requests for human consumption (Telegram, email, dashboard)."""

    @staticmethod
    def telegram_message(cp: Checkpoint, rule: Tier3Rule) -> str:
        urgency_emoji = {
            Urgency.LOW: "🟢",
            Urgency.NORMAL: "🔵",
            Urgency.URGENT: "🟠",
            Urgency.EMERGENCY: "🔴",
        }
        lines = [
            f"{urgency_emoji.get(cp.urgency, '⚪')} **TIER 3 GATE — {cp.gate_type.value.upper()}**",
            f"",
            f"**ID:** `{cp.checkpoint_id}`",
            f"**Requested by:** {cp.requested_by}",
            f"**Time:** {cp.requested_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Timeout:** {cp.timeout_at.strftime('%Y-%m-%d %H:%M UTC')} ({(cp.timeout_at - cp.requested_at).total_seconds() / 3600:.1f}h)",
            f"",
            f"**Description:** {rule.description}",
            f"",
            f"**Required Evidence:**",
        ]
        for ev in rule.required_evidence:
            path = cp.evidence_paths.get(ev, "NOT ATTACHED")
            lines.append(f"  • `{ev}` → {path}")
        lines.append("")
        lines.append("**Agent Context:**")
        for k, v in cp.agent_context.items():
            lines.append(f"  • {k}: {v}")
        lines.append("")
        lines.append("**Reply with:**")
        lines.append("  `APPROVE <id>` — approve as-is")
        lines.append("  `DENY <id> [reason]` — deny and abort")
        lines.append("  `MODIFY <id> <notes>` — approve with changes")
        lines.append("")
        lines.append(f"_Safe default on timeout: {rule.safe_default_on_timeout.upper()}_")
        return "\n".join(lines)

    @staticmethod
    def email_subject(cp: Checkpoint) -> str:
        return f"[TIER-3] {cp.gate_type.value.upper()} — {cp.checkpoint_id} — {cp.urgency.value.upper()}"

    @staticmethod
    def dashboard_card(cp: Checkpoint) -> Dict[str, Any]:
        return {
            "id": cp.checkpoint_id,
            "type": cp.gate_type.value,
            "status": cp.status.value,
            "urgency": cp.urgency.value,
            "requested_at": cp.requested_at.isoformat(),
            "timeout_at": cp.timeout_at.isoformat(),
            "progress_pct": 0 if cp.status == GateStatus.PENDING else 100,
            "evidence_count": len(cp.evidence_paths),
        }


# ---------------------------------------------------------------------------
# Checkpoint Engine
# ---------------------------------------------------------------------------

class CheckpointEngine:
    """
    Core engine: evaluates conditions, creates checkpoints,
    handles responses, enforces timeouts.
    """

    def __init__(self, storage_path: Path = CHECKPOINT_ROOT):
        self.storage = storage_path
        self.storage.mkdir(parents=True, exist_ok=True)
        self.active: Dict[str, Checkpoint] = {}
        self._load_active()

    def _checkpoint_path(self, cp_id: str) -> Path:
        return self.storage / f"{cp_id}.json"

    def _load_active(self):
        for path in self.storage.glob("*.json"):
            try:
                cp = Checkpoint.from_json(json.loads(path.read_text()))
                if cp.status in (GateStatus.PENDING, GateStatus.TIMED_OUT):
                    self.active[cp.checkpoint_id] = cp
            except Exception as e:
                logger.error(f"Failed to load checkpoint {path}: {e}")

    def _save(self, cp: Checkpoint):
        self._checkpoint_path(cp.checkpoint_id).write_text(
            json.dumps(cp.to_json(), indent=2)
        )

    def evaluate_conditions(
        self,
        context: Dict[str, Any],
    ) -> List[Tier3Rule]:
        """
        Evaluate agent context against all Tier3 rules.
        Returns list of triggered rules.
        """
        triggered: List[Tier3Rule] = []
        for rule in TIER3_RULEBOOK:
            if self._check_rule(rule, context):
                triggered.append(rule)
        return triggered

    def _check_rule(self, rule: Tier3Rule, context: Dict[str, Any]) -> bool:
        # Simple condition evaluator — in production, use a safe expression engine
        for cond in rule.auto_conditions:
            if not self._eval_condition(cond, context):
                return False
        return True

    def _eval_condition(self, cond: str, context: Dict[str, Any]) -> bool:
        # Naive eval — replace with proper safe eval in production
        try:
            # Map common patterns
            if "==" in cond:
                left, right = cond.split("==", 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                return str(context.get(left, "")) == right
            if ">=" in cond:
                left, right = cond.split(">=", 1)
                left = left.strip()
                right = float(right.strip())
                return float(context.get(left, 0)) >= right
            if ">" in cond:
                left, right = cond.split(">", 1)
                left = left.strip()
                right = float(right.strip())
                return float(context.get(left, 0)) > right
            if "contains" in cond:
                # e.g., "fail_modes contains 'structural_collapse'"
                parts = cond.split("contains")
                key = parts[0].strip()
                val = parts[1].strip().strip("'\"")
                arr = context.get(key, [])
                return val in arr
        except Exception:
            return False
        return False

    def create_checkpoint(
        self,
        gate_type: GateType,
        requested_by: str,
        context: Dict[str, Any],
        evidence: Dict[str, Path],
        urgency: Urgency = Urgency.NORMAL,
        timeout_hours: Optional[float] = None,
    ) -> Checkpoint:
        rule = next(r for r in TIER3_RULEBOOK if r.gate_type == gate_type)
        to = timeout_hours or rule.default_timeout_hours
        to = min(to, MAX_TIMEOUT_HOURS)

        now = datetime.now(timezone.utc)
        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            gate_type=gate_type,
            status=GateStatus.PENDING,
            urgency=urgency,
            requested_by=requested_by,
            requested_at=now,
            timeout_at=now + timedelta(hours=to),
            evidence_paths=evidence,
            agent_context=context,
        )
        self.active[cp.checkpoint_id] = cp
        self._save(cp)
        logger.info(f"[Checkpoint] Created {cp.checkpoint_id} ({gate_type.value}) for {requested_by}")
        return cp

    def process_response(
        self,
        checkpoint_id: str,
        response: str,  # "APPROVE", "DENY reason...", "MODIFY notes..."
        responder: str = "alex",
    ) -> Checkpoint:
        """Process Alex's (or override) response to a checkpoint."""
        cp = self.active.get(checkpoint_id)
        if not cp:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        if cp.status != GateStatus.PENDING:
            raise ValueError(f"Checkpoint {checkpoint_id} already resolved ({cp.status.value})")

        now = datetime.now(timezone.utc)
        cp.resolved_at = now
        cp.resolved_by = responder
        cp.alex_response = response

        upper = response.strip().upper()
        if upper.startswith("APPROVE"):
            cp.status = GateStatus.APPROVED
        elif upper.startswith("DENY"):
            cp.status = GateStatus.DENIED
            cp.modification_notes = response[4:].strip() if len(response) > 4 else None
        elif upper.startswith("MODIFY"):
            cp.status = GateStatus.MODIFIED
            cp.modification_notes = response[6:].strip() if len(response) > 6 else None
        else:
            raise ValueError(f"Unknown response format: {response}")

        self._save(cp)
        logger.info(f"[Checkpoint] {checkpoint_id} → {cp.status.value} by {responder}")
        return cp

    def check_timeouts(self) -> List[Checkpoint]:
        """Poll for timed-out checkpoints. Returns list of timed-out checkpoints."""
        now = datetime.now(timezone.utc)
        timed_out: List[Checkpoint] = []
        for cp in list(self.active.values()):
            if cp.status != GateStatus.PENDING:
                continue
            if now >= cp.timeout_at:
                rule = next(r for r in TIER3_RULEBOOK if r.gate_type == cp.gate_type)
                cp.status = GateStatus.TIMED_OUT
                cp.resolved_at = now
                cp.resolved_by = "system_timeout"
                self._save(cp)
                timed_out.append(cp)
                logger.warning(
                    f"[Checkpoint] {cp.checkpoint_id} TIMED OUT → safe_default={rule.safe_default_on_timeout}"
                )
        return timed_out

    def apply_override(
        self,
        checkpoint_id: str,
        override_by: str,
        justification: str,
        new_status: GateStatus = GateStatus.OVERRIDDEN,
    ) -> Checkpoint:
        """
        Apply an administrative override to a checkpoint.
        Requires justification. Logs permanently.
        """
        cp = self.active.get(checkpoint_id)
        if not cp:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        rule = next(r for r in TIER3_RULEBOOK if r.gate_type == cp.gate_type)
        if not rule.allow_override:
            raise PermissionError(f"Override not allowed for {cp.gate_type.value}")

        now = datetime.now(timezone.utc)
        cp.status = new_status
        cp.resolved_at = now
        cp.resolved_by = override_by
        cp.override_record = {
            "overridden_by": override_by,
            "justification": justification,
            "original_status": cp.status.value,
            "timestamp": now.isoformat(),
        }
        self._save(cp)
        logger.warning(f"[Checkpoint] {checkpoint_id} OVERRIDDEN by {override_by}: {justification}")
        return cp

    def get_pending(self) -> List[Checkpoint]:
        return [cp for cp in self.active.values() if cp.status == GateStatus.PENDING]

    def get_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cp in self.active.values():
            counts[cp.status.value] = counts.get(cp.status.value, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------

class HITLGuard:
    """
    Drop-in guard for Moses / Titan / Atlas to check before acting.
    Usage:
        guard = HITLGuard()
        triggered = guard.check_before_deploy(context)
        if triggered:
            cp = guard.create_checkpoint(...)
            # Block until approved
            while guard.is_pending(cp.checkpoint_id):
                time.sleep(5)
            if not guard.is_approved(cp.checkpoint_id):
                raise AbortOperation("Denied by Tier 3 gate")
    """

    def __init__(self, engine: Optional[CheckpointEngine] = None):
        self.engine = engine or CheckpointEngine()
        self.formatter = ReviewFormatter()

    def check_before_action(
        self,
        action_type: GateType,
        context: Dict[str, Any],
        evidence: Dict[str, Path],
        requested_by: str,
        urgency: Urgency = Urgency.NORMAL,
    ) -> Optional[Checkpoint]:
        """
        Check if action requires human approval.
        Returns Checkpoint if gate created, None if no rules triggered.
        """
        triggered = self.engine.evaluate_conditions(context)
        matching = [r for r in triggered if r.gate_type == action_type]
        if not matching:
            return None

        cp = self.engine.create_checkpoint(
            gate_type=action_type,
            requested_by=requested_by,
            context=context,
            evidence=evidence,
            urgency=urgency,
        )
        return cp

    def is_pending(self, checkpoint_id: str) -> bool:
        cp = self.engine.active.get(checkpoint_id)
        return cp is not None and cp.status == GateStatus.PENDING

    def is_approved(self, checkpoint_id: str) -> bool:
        cp = self.engine.active.get(checkpoint_id)
        return cp is not None and cp.status in (GateStatus.APPROVED, GateStatus.MODIFIED, GateStatus.OVERRIDDEN)

    def is_denied(self, checkpoint_id: str) -> bool:
        cp = self.engine.active.get(checkpoint_id)
        return cp is not None and cp.status in (GateStatus.DENIED, GateStatus.ABORTED, GateStatus.TIMED_OUT)

    def safe_default_action(self, checkpoint_id: str) -> str:
        """Return the safe default action for a timed-out checkpoint."""
        cp = self.engine.active.get(checkpoint_id)
        if not cp:
            return "abort"
        rule = next(r for r in TIER3_RULEBOOK if r.gate_type == cp.gate_type)
        return rule.safe_default_on_timeout


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

def demo():
    logging.basicConfig(level=logging.INFO)

    engine = CheckpointEngine(storage_path=Path("/tmp/checkpoints_demo"))
    guard = HITLGuard(engine)

    # Scenario: Moses wants to deploy an untested policy to physical robot
    context = {
        "target_environment": "physical",
        "sim_success_rate": 87.0,
        "policy_sim_runs": 0,
        "policy_eval_score": None,
    }
    evidence = {
        "sim_report.json": Path("/tmp/fake_sim.json"),
        "safety_analysis.json": Path("/tmp/fake_safety.json"),
    }

    cp = guard.check_before_action(
        action_type=GateType.PHYSICAL_DEPLOY,
        context=context,
        evidence=evidence,
        requested_by="moses",
        urgency=Urgency.URGENT,
    )

    if cp:
        print("\n=== TELEGRAM REVIEW REQUEST ===")
        rule = next(r for r in TIER3_RULEBOOK if r.gate_type == cp.gate_type)
        print(guard.formatter.telegram_message(cp, rule))

        # Simulate Alex approving
        time.sleep(1)
        engine.process_response(cp.checkpoint_id, "APPROVE", responder="alex")
        print(f"\n=== RESOLVED ===")
        print(f"Status: {cp.status.value}")
        print(f"Safe default would have been: {rule.safe_default_on_timeout}")
    else:
        print("No checkpoint triggered (unexpected)")


if __name__ == "__main__":
    demo()
