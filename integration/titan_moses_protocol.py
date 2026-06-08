#!/usr/bin/env python3
"""
Titan-Moses Collaboration Protocol

Implements the bidirectional design-simulation loop between:
  - TITAN: Physics constraints, feasibility studies, safety analysis
  - MOSES: Designs, test results, trained policies, failure data

Features:
  - Shared knowledge corpus management
  - Conflict resolution (Titan says INFEASIBLE, Moses wants to test)
  - Async message bus integration
  - Telemetry streaming and artifact persistence

Reference: integration_spec.md §3.1
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("titan_moses_protocol")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_ROOT = Path("/shared")
INBOX_ROOT = SHARED_ROOT / "inbox"
OUTBOX_ROOT = SHARED_ROOT / "outbox"
SIM_ARTIFACT_ROOT = SHARED_ROOT / "sim" / "titan"
DESIGN_ROOT = SHARED_ROOT / "designs" / "moses"
KNOWLEDGE_ROOT = SHARED_ROOT / "knowledge" / "robotics" / "corpus"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds
SIM_TIMEOUT_SEC = 1800  # 30 minutes

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeasibilityStatus(Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    NEEDS_WORK = "NEEDS_WORK"
    SIM_ERROR = "SIM_ERROR"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConflictResolution(Enum):
    AUTO_ACCEPT_TITAN = auto()      # Titan wins — physics is law
    AUTO_ACCEPT_MOSES = auto()      # Moses wins — innovation push
    ESCALATE_TO_HUMAN = auto()      # Tier 3 gate
    COMPROMISE_ITERATE = auto()     # Both agents iterate

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignArtifact:
    """A design file produced by Moses."""
    format: str  # urdf, sdf, onshape_url, mjcf
    uri: Path
    checksum: str
    version: str = "1.0"

    def verify(self) -> bool:
        if not self.uri.exists():
            return False
        h = hashlib.sha256(self.uri.read_bytes()).hexdigest()
        return h == self.checksum


@dataclass
class TestScenario:
    """Scenario parameters for Titan simulation."""
    scenario_type: str  # walk, manipulate, balance, fall_recovery
    duration_sec: float = 60.0
    terrain: str = "flat"
    perturbations: List[str] = field(default_factory=list)


@dataclass
class DesignSubmission:
    """Moses → Titan: Request to evaluate a design."""
    message_id: str
    design: DesignArtifact
    scenario: TestScenario
    priority: str = "normal"
    deadline: Optional[datetime] = None
    moses_notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "message_type": "DESIGN_SUBMISSION",
            "message_id": self.message_id,
            "from": "moses",
            "to": "titan",
            "payload": {
                "design": {
                    "format": self.design.format,
                    "uri": str(self.design.uri),
                    "checksum": self.design.checksum,
                    "version": self.design.version,
                },
                "test_scenario": {
                    "type": self.scenario.scenario_type,
                    "duration_sec": self.scenario.duration_sec,
                    "terrain": self.scenario.terrain,
                    "perturbations": self.scenario.perturbations,
                },
                "priority": self.priority,
                "deadline": self.deadline.isoformat() if self.deadline else None,
                "moses_notes": self.moses_notes,
            },
            "tier": 2,
        }


@dataclass
class PhysicsValidation:
    """Physics check results from Titan."""
    com_position: List[float]
    stability_margin: float
    pass_: bool


@dataclass
class SafetyAnalysis:
    """Safety scoring from Titan."""
    risk_score: int  # 0-100
    max_joint_torque: float
    collision_probability: float
    fail_modes: List[str]


@dataclass
class Recommendation:
    """Titan's suggested fix."""
    severity: str
    component: str
    issue: str
    suggestion: str


@dataclass
class FeasibilityReport:
    """Titan → Moses: Results of simulation / analysis."""
    message_id: str
    in_reply_to: str
    status: FeasibilityStatus
    physics_validation: PhysicsValidation
    safety_analysis: SafetyAnalysis
    recommendations: List[Recommendation]
    sim_artifacts: Dict[str, Path]
    titan_notes: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> FeasibilityReport:
        p = data["payload"]
        return cls(
            message_id=data["message_id"],
            in_reply_to=data["in_reply_to"],
            status=FeasibilityStatus(p["status"]),
            physics_validation=PhysicsValidation(**p["physics_validation"]),
            safety_analysis=SafetyAnalysis(**p["safety_analysis"]),
            recommendations=[Recommendation(**r) for r in p.get("recommendations", [])],
            sim_artifacts={k: Path(v) for k, v in p.get("sim_artifacts", {}).items()},
            titan_notes=p.get("titan_notes", ""),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "message_type": "FEASIBILITY_REPORT",
            "message_id": self.message_id,
            "in_reply_to": self.in_reply_to,
            "from": "titan",
            "to": "moses",
            "payload": {
                "status": self.status.value,
                "physics_validation": asdict(self.physics_validation),
                "safety_analysis": asdict(self.safety_analysis),
                "recommendations": [asdict(r) for r in self.recommendations],
                "sim_artifacts": {k: str(v) for k, v in self.sim_artifacts.items()},
                "titan_notes": self.titan_notes,
            },
            "tier": 2,
        }


@dataclass
class KnowledgeEntry:
    """Single entry in the shared robotics knowledge corpus."""
    entry_id: str
    timestamp: datetime
    source_agent: str
    entry_type: str  # design, sim_result, policy, failure, lesson
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return json.dumps({
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "entry_type": self.entry_type,
            "content": self.content,
            "embedding": self.embedding,
            "tags": self.tags,
        })


# ---------------------------------------------------------------------------
# Message Bus (Filesystem-based)
# ---------------------------------------------------------------------------

class FilesystemMessageBus:
    """
    Durable, filesystem-backed message bus.
    At-least-once delivery via idempotency keys.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inbox = INBOX_ROOT / agent_id
        self.outbox = OUTBOX_ROOT / agent_id
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()

    def send(self, to: str, payload: Dict[str, Any]) -> str:
        msg_id = payload.get("message_id", str(uuid.uuid4()))
        payload["message_id"] = msg_id
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        target = OUTBOX_ROOT / to / f"{msg_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.rename(target)
        logger.info(f"[{self.agent_id}] sent {payload['message_type']} → {to} ({msg_id})")
        return msg_id

    def poll(self, max_age_sec: float = 3600.0) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        cutoff = time.time() - max_age_sec

        for path in sorted(self.inbox.glob("*.json")):
            if path.stat().st_mtime < cutoff:
                continue
            msg_id = path.stem
            if msg_id in self._processed:
                continue
            try:
                data = json.loads(path.read_text())
                messages.append(data)
                self._processed.add(msg_id)
                # Archive after processing
                archive = self.inbox / "archive"
                archive.mkdir(exist_ok=True)
                shutil.move(str(path), str(archive / path.name))
            except Exception as e:
                logger.error(f"Failed to process message {path}: {e}")
        return messages

    def wait_for_reply(
        self,
        correlation_id: str,
        timeout_sec: float = 300.0,
        poll_interval: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            for msg in self.poll(max_age_sec=timeout_sec):
                if msg.get("in_reply_to") == correlation_id:
                    return msg
            time.sleep(poll_interval)
        logger.warning(f"Timeout waiting for reply to {correlation_id}")
        return None


# ---------------------------------------------------------------------------
# Shared Knowledge Corpus
# ---------------------------------------------------------------------------

class KnowledgeCorpus:
    """
    Append-only shared knowledge graph for robotics.
    Both agents read/write. Embeddings optional (computed lazily).
    """

    def __init__(self, path: Path = KNOWLEDGE_ROOT):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "index.jsonl"

    def append(self, entry: KnowledgeEntry) -> None:
        with self.index_file.open("a") as f:
            f.write(entry.to_jsonl() + "\n")
        logger.info(f"[Corpus] appended {entry.entry_type} from {entry.source_agent}")

    def query(
        self,
        entry_type: Optional[str] = None,
        source_agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[KnowledgeEntry]:
        results: List[KnowledgeEntry] = []
        if not self.index_file.exists():
            return results

        for line in self.index_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if entry_type and data["entry_type"] != entry_type:
                    continue
                if source_agent and data["source_agent"] != source_agent:
                    continue
                if tags and not any(t in data.get("tags", []) for t in tags):
                    continue
                if since and datetime.fromisoformat(data["timestamp"]) < since:
                    continue
                results.append(KnowledgeEntry(**data))
                if len(results) >= limit:
                    break
            except Exception:
                continue
        return results

    def get_latest_design_version(self, design_name: str) -> Optional[str]:
        """Return latest version string for a design name."""
        entries = self.query(entry_type="design", tags=[design_name], limit=1)
        if not entries:
            return None
        return entries[0].content.get("version")


# ---------------------------------------------------------------------------
# Conflict Resolution Engine
# ---------------------------------------------------------------------------

class ConflictResolver:
    """
    When Titan says INFEASIBLE but Moses wants to test anyway.

    Rules (in priority order):
    1. If safety_analysis.risk_score >= 80 → ALWAYS escalate to human (Tier 3)
    2. If fail_modes contains "structural_collapse" or "uncontrolled_fall" → ALWAYS escalate
    3. If Moses has run >2 prior iterations on same component → Escalate (prevent loops)
    4. If Titan status == INFEASIBLE and Moses counter-argument provided → Compromise iterate
    5. If Titan status == INFEASIBLE, no counter → Accept Titan
    6. If Titan status == NEEDS_WORK → Compromise iterate
    """

    def __init__(self, corpus: KnowledgeCorpus):
        self.corpus = corpus

    def resolve(
        self,
        report: FeasibilityReport,
        moses_counter: Optional[str] = None,
        iteration_count: int = 0,
    ) -> ConflictResolution:
        safety = report.safety_analysis

        # Rule 1: High risk score
        if safety.risk_score >= 80:
            logger.warning(f"[Conflict] Risk score {safety.risk_score} ≥ 80 → ESCALATE")
            return ConflictResolution.ESCALATE_TO_HUMAN

        # Rule 2: Catastrophic failure modes
        catastrophic = {"structural_collapse", "uncontrolled_fall", "power_loss_mid_stride", "head_impact_fatal"}
        if catastrophic & set(safety.fail_modes):
            logger.warning(f"[Conflict] Catastrophic fail mode detected → ESCALATE")
            return ConflictResolution.ESCALATE_TO_HUMAN

        # Rule 3: Iteration loop prevention
        if iteration_count >= 2:
            logger.warning(f"[Conflict] {iteration_count} iterations → ESCALATE")
            return ConflictResolution.ESCALATE_TO_HUMAN

        # Rule 4: Moses has a reasoned counter-argument
        if report.status == FeasibilityStatus.INFEASIBLE and moses_counter:
            logger.info("[Conflict] Moses counter-argument present → COMPROMISE_ITERATE")
            return ConflictResolution.COMPROMISE_ITERATE

        # Rule 5: Titan wins by default
        if report.status == FeasibilityStatus.INFEASIBLE:
            logger.info("[Conflict] No counter-argument → AUTO_ACCEPT_TITAN")
            return ConflictResolution.AUTO_ACCEPT_TITAN

        # Rule 6: Needs work
        if report.status == FeasibilityStatus.NEEDS_WORK:
            logger.info("[Conflict] NEEDS_WORK → COMPROMISE_ITERATE")
            return ConflictResolution.COMPROMISE_ITERATE

        # Default: feasible, no conflict
        return ConflictResolution.AUTO_ACCEPT_TITAN

    def build_escalation_payload(
        self,
        report: FeasibilityReport,
        moses_counter: Optional[str],
        iteration_count: int,
    ) -> Dict[str, Any]:
        """Build the escalation message for Walker / human checkpoint."""
        return {
            "message_type": "ESCALATION",
            "message_id": str(uuid.uuid4()),
            "from": "titan_moses_protocol",
            "to": "walker",
            "payload": {
                "escalation_type": "CONFLICT",
                "reason": f"Titan {report.status.value} on design {report.in_reply_to}",
                "context": {
                    "titan_report": str(SIM_ARTIFACT_ROOT / report.in_reply_to / "report.json"),
                    "moses_counter": moses_counter or "None provided",
                    "iteration_count": iteration_count,
                    "risk_score": report.safety_analysis.risk_score,
                    "fail_modes": report.safety_analysis.fail_modes,
                },
                "requested_action": "HUMAN_REVIEW",
                "urgency": "urgent" if report.safety_analysis.risk_score >= 80 else "normal",
            },
            "tier": 3,
        }


# ---------------------------------------------------------------------------
# Titan Interface
# ---------------------------------------------------------------------------

class TitanInterface:
    """
    Moses-side client for talking to Titan.
    Handles submission, polling, retries, and conflict resolution.
    """

    def __init__(self, bus: FilesystemMessageBus, corpus: KnowledgeCorpus):
        self.bus = bus
        self.corpus = corpus
        self.resolver = ConflictResolver(corpus)

    def submit_design(
        self,
        design: DesignArtifact,
        scenario: TestScenario,
        priority: str = "normal",
        moses_notes: str = "",
    ) -> str:
        if not design.verify():
            raise ValueError(f"Design checksum mismatch: {design.uri}")

        msg_id = str(uuid.uuid4())
        sub = DesignSubmission(
            message_id=msg_id,
            design=design,
            scenario=scenario,
            priority=priority,
            moses_notes=moses_notes,
        )
        self.bus.send("titan", sub.to_json())

        # Log to corpus
        self.corpus.append(KnowledgeEntry(
            entry_id=msg_id,
            timestamp=datetime.now(timezone.utc),
            source_agent="moses",
            entry_type="design",
            content={
                "design_uri": str(design.uri),
                "format": design.format,
                "version": design.version,
                "scenario": asdict(scenario),
            },
            tags=["design", design.format, scenario.scenario_type],
        ))
        return msg_id

    def await_report(
        self,
        submission_id: str,
        timeout_sec: float = SIM_TIMEOUT_SEC,
    ) -> Optional[FeasibilityReport]:
        raw = self.bus.wait_for_reply(submission_id, timeout_sec=timeout_sec)
        if raw is None:
            return None
        return FeasibilityReport.from_json(raw)

    def handle_report(
        self,
        report: FeasibilityReport,
        moses_counter: Optional[str] = None,
        iteration_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Process Titan's report. Returns action dict:
          {action: "ACCEPT|REJECT|ITERATE|ESCALATE", reason: "...", payload: {...}}
        """
        # Log to corpus
        self.corpus.append(KnowledgeEntry(
            entry_id=report.message_id,
            timestamp=datetime.now(timezone.utc),
            source_agent="titan",
            entry_type="sim_result",
            content={
                "in_reply_to": report.in_reply_to,
                "status": report.status.value,
                "risk_score": report.safety_analysis.risk_score,
                "recommendations": [asdict(r) for r in report.recommendations],
            },
            tags=["sim", report.status.value],
        ))

        resolution = self.resolver.resolve(report, moses_counter, iteration_count)

        if resolution == ConflictResolution.AUTO_ACCEPT_TITAN:
            return {
                "action": "REJECT" if report.status == FeasibilityStatus.INFEASIBLE else "ACCEPT",
                "reason": f"Titan verdict: {report.status.value}. No counter-argument.",
                "payload": report.to_json(),
            }

        if resolution == ConflictResolution.COMPROMISE_ITERATE:
            return {
                "action": "ITERATE",
                "reason": f"Titan says {report.status.value}; Moses counter present. Iterate.",
                "payload": {
                    "report": report.to_json(),
                    "recommendations": [asdict(r) for r in report.recommendations],
                },
            }

        if resolution == ConflictResolution.ESCALATE_TO_HUMAN:
            esc = self.resolver.build_escalation_payload(report, moses_counter, iteration_count)
            self.bus.send("walker", esc)
            return {
                "action": "ESCALATE",
                "reason": "Safety threshold or iteration limit exceeded. Human review required.",
                "payload": esc,
            }

        # Fallback
        return {"action": "REJECT", "reason": "Unhandled resolution state.", "payload": {}}


# ---------------------------------------------------------------------------
# Moses Interface (Titan-side)
# ---------------------------------------------------------------------------

class MosesInterface:
    """
    Titan-side server for receiving design submissions and sending reports.
    """

    def __init__(self, bus: FilesystemMessageBus, corpus: KnowledgeCorpus):
        self.bus = bus
        self.corpus = corpus
        self._handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def on_design(self, handler: Callable[[DesignSubmission], FeasibilityReport]):
        """Register a handler that takes a DesignSubmission and returns a FeasibilityReport."""
        self._handlers["DESIGN_SUBMISSION"] = lambda msg: self._wrap_handler(msg, handler)

    def _wrap_handler(
        self,
        msg: Dict[str, Any],
        handler: Callable[[DesignSubmission], FeasibilityReport],
    ) -> None:
        p = msg["payload"]
        design = DesignArtifact(
            format=p["design"]["format"],
            uri=Path(p["design"]["uri"]),
            checksum=p["design"]["checksum"],
            version=p["design"].get("version", "1.0"),
        )
        scenario = TestScenario(
            scenario_type=p["test_scenario"]["type"],
            duration_sec=p["test_scenario"]["duration_sec"],
            terrain=p["test_scenario"]["terrain"],
            perturbations=p["test_scenario"].get("perturbations", []),
        )
        sub = DesignSubmission(
            message_id=msg["message_id"],
            design=design,
            scenario=scenario,
            priority=p.get("priority", "normal"),
            deadline=datetime.fromisoformat(p["deadline"]) if p.get("deadline") else None,
            moses_notes=p.get("moses_notes", ""),
        )

        report = handler(sub)
        self.bus.send("moses", report.to_json())

    def poll_and_process(self) -> int:
        """Poll inbox and process messages. Returns count processed."""
        count = 0
        for msg in self.bus.poll():
            msg_type = msg.get("message_type")
            if msg_type in self._handlers:
                try:
                    self._handlers[msg_type](msg)
                    count += 1
                except Exception as e:
                    logger.error(f"Error handling {msg_type}: {e}")
            else:
                logger.warning(f"No handler for message type: {msg_type}")
        return count


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

def demo():
    logging.basicConfig(level=logging.INFO)

    # Setup
    corpus = KnowledgeCorpus()
    moses_bus = FilesystemMessageBus("moses")
    titan_bus = FilesystemMessageBus("titan")

    # --- Titan side ---
    titan = MosesInterface(titan_bus, corpus)

    def handle_design(sub: DesignSubmission) -> FeasibilityReport:
        logger.info(f"[Titan] received design {sub.design.uri}")
        # Simulate physics check
        stable = sub.design.format == "urdf"  # naive mock
        return FeasibilityReport(
            message_id=str(uuid.uuid4()),
            in_reply_to=sub.message_id,
            status=FeasibilityStatus.FEASIBLE if stable else FeasibilityStatus.INFEASIBLE,
            physics_validation=PhysicsValidation(
                com_position=[0.1, 0.0, 0.9],
                stability_margin=0.05 if stable else 0.01,
                pass_=stable,
            ),
            safety_analysis=SafetyAnalysis(
                risk_score=45 if stable else 85,
                max_joint_torque=95.0,
                collision_probability=0.05,
                fail_modes=[] if stable else ["ankle_roll_overload"],
            ),
            recommendations=[],
            sim_artifacts={},
        )

    titan.on_design(handle_design)

    # --- Moses side ---
    moses = TitanInterface(moses_bus, corpus)

    # Create a dummy design
    dummy_urdf = DESIGN_ROOT / "demo_arm.urdf"
    dummy_urdf.parent.mkdir(parents=True, exist_ok=True)
    dummy_urdf.write_text("<robot name='demo'><link name='base'/></robot>")
    design = DesignArtifact(
        format="urdf",
        uri=dummy_urdf,
        checksum=hashlib.sha256(dummy_urdf.read_bytes()).hexdigest(),
    )
    scenario = TestScenario(scenario_type="balance", duration_sec=10.0)

    # Submit
    sub_id = moses.submit_design(design, scenario, moses_notes="First test")

    # Titan processes
    titan.poll_and_process()

    # Moses awaits report
    report = moses.await_report(sub_id, timeout_sec=5.0)
    if report:
        result = moses.handle_report(report, iteration_count=0)
        print(f"\n=== RESULT ===")
        print(f"Action: {result['action']}")
        print(f"Reason: {result['reason']}")
    else:
        print("No report received (timeout)")


if __name__ == "__main__":
    demo()
