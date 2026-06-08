"""
approval_gates.py — Human oversight for self-modification in Moses v4.0.

Tier system for self-modification:
  Tier 1: Hyperparameter changes (auto-approved)
  Tier 2: Code mutations (auto-approved, logged)
  Tier 3: Core file changes (requires Alex approval)
  Tier 4: Self-architecture changes (requires Alex approval + 24h wait)

Fail-safe: if unsure, stop and ask.
All events logged immutably.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "alex_telegram_id": "6819661992",      # Alex's Telegram ID
    "telegram_bot_token": None,            # Set via env or config
    "pending_approvals_dir": ".moses_pending_approvals",
    "approval_log": ".moses_approval_log.jsonl",
    "tier3_core_files": [
        "moses/safety/__init__.py",
        "moses/safety/bounds_checker.py",
        "moses/safety/approval_gates.py",
        "moses/safety/drift_detector.py",
        "moses/safety/integrity_checker.py",
        "moses/core/engine.py",
        "moses/core/learner.py",
    ],
    "tier4_patterns": [
        r".*architecture.*",
        r".*self_modify.*",
        r".*recursive.*",
        r".*bootstrap.*",
        r"safety/.*",           # Any safety file change is Tier 3/4
    ],
    "auto_approve_tier1": True,
    "auto_approve_tier2": True,
    "require_tier3_approval": True,
    "require_tier4_approval": True,
    "tier4_wait_hours": 24,
}

# ─── Tier Definitions ────────────────────────────────────────────────────────

class Tier(Enum):
    TIER_1_HYPERPARAM = auto()
    TIER_2_CODE_MUTATION = auto()
    TIER_3_CORE_FILE = auto()
    TIER_4_ARCHITECTURE = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.name


TIER_NAMES = {
    Tier.TIER_1_HYPERPARAM: "Tier 1: Hyperparameter",
    Tier.TIER_2_CODE_MUTATION: "Tier 2: Code Mutation",
    Tier.TIER_3_CORE_FILE: "Tier 3: Core File",
    Tier.TIER_4_ARCHITECTURE: "Tier 4: Architecture",
    Tier.UNKNOWN: "Unknown",
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ChangeRequest:
    request_id: str
    timestamp: str
    tier: str
    files: List[str]
    diff_summary: str
    predicted_impact: str
    proposed_by: str
    status: str          # PENDING, APPROVED, DENIED, EXPIRED
    alex_approved: bool
    approval_time: Optional[str]
    scheduled_time: Optional[str]  # For Tier 4 delay
    checksum: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def create(cls, tier: Tier, files: List[str], diff: str,
               impact: str, proposed_by: str = "moses") -> "ChangeRequest":
        timestamp = datetime.now().isoformat()
        req_id = hashlib.sha256(
            f"{timestamp}:{':'.join(files)}:{diff[:100]}".encode()
        ).hexdigest()[:16]
        
        scheduled = None
        if tier == Tier.TIER_4_ARCHITECTURE:
            scheduled = (datetime.now() +
                        timedelta(hours=DEFAULT_CONFIG["tier4_wait_hours"])).isoformat()
        
        payload = {
            "request_id": req_id,
            "timestamp": timestamp,
            "tier": tier.name,
            "files": files,
            "diff_summary": diff[:2000],  # Truncate for storage
            "predicted_impact": impact,
            "proposed_by": proposed_by,
            "status": "PENDING",
            "alex_approved": False,
            "approval_time": None,
            "scheduled_time": scheduled,
        }
        payload["checksum"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return cls(**payload)


@dataclass
class ApprovalEvent:
    timestamp: str
    event_type: str
    request_id: str
    tier: str
    message: str
    details: Dict
    checksum: str

    def to_line(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "tier": self.tier,
            "message": self.message,
            "details": self.details,
        }
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return json.dumps({"payload": payload, "checksum": checksum})


# ─── Telegram Integration ────────────────────────────────────────────────────

class TelegramNotifier:
    """Sends approval requests to Alex via Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.environ.get("MOSES_TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("MOSES_TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        if not self.is_configured():
            print(f"[TELEGRAM] Would send:\n{text}\n", file=sys.stderr)
            return False
        
        try:
            import urllib.request
            import urllib.parse
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()
            
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"Telegram send failed: {e}", file=sys.stderr)
            return False

    def send_approval_request(self, request: ChangeRequest) -> bool:
        tier_name = TIER_NAMES.get(Tier[request.tier], request.tier)
        wait_notice = ""
        if request.scheduled_time:
            wait_notice = f"\n⏰ *Earliest execution:* `{request.scheduled_time}`"
        
        text = f"""🚨 *Moses Self-Modification Request*

*ID:* `{request.request_id}`
*Tier:* {tier_name}
*Files:* {', '.join(request.files)}
*Proposed by:* {request.proposed_by}
*Time:* {request.timestamp}
{wait_notice}

*Predicted Impact:*
```
{request.predicted_impact[:500]}
```

*Diff Summary:*
```diff
{request.diff_summary[:800]}
```

Reply with:
• `APPROVE {request.request_id}` to approve
• `DENY {request.request_id}` to deny
"""
        return self.send_message(text)

    def send_notification(self, title: str, message: str) -> bool:
        text = f"*{title}*\n{message}"
        return self.send_message(text)


# ─── Approval Gates ──────────────────────────────────────────────────────────

class ApprovalGates:
    """
    Manages the tiered approval system for self-modification.
    
    Principles:
    - Defense in depth: tier classification is conservative
    - Fail-safe: ambiguous changes default to higher tier
    - Immutable audit trail: every decision logged
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.notifier = TelegramNotifier(
            self.config.get("telegram_bot_token"),
            self.config.get("alex_telegram_id")
        )
        self._pending_dir = Path(self.config["pending_approvals_dir"])
        self._pending_dir.mkdir(parents=True, exist_ok=True)

    # ─── Tier Classification ─────────────────────────────────────────────────

    def classify_change(self, files: List[str], diff: str = "") -> Tier:
        """
        Classify a proposed change into a tier.
        Conservative: when in doubt, escalate.
        """
        # Check Tier 4 patterns first (most restrictive)
        tier4_patterns = [re.compile(p) for p in self.config["tier4_patterns"]]
        for f in files:
            for pattern in tier4_patterns:
                if pattern.match(f):
                    return Tier.TIER_4_ARCHITECTURE
        
        # Check Tier 3 core files
        core_files = set(self.config["tier3_core_files"])
        for f in files:
            if f in core_files or any(f.endswith(c.split("/")[-1]) for c in core_files):
                return Tier.TIER_3_CORE_FILE
        
        # Check for safety-related keywords in diff
        safety_keywords = ["safety", "bounds", "approval", "integrity", "drift"]
        diff_lower = diff.lower()
        if any(kw in diff_lower for kw in safety_keywords):
            # Fail-safe: safety-related changes are at least Tier 3
            return Tier.TIER_3_CORE_FILE
        
        # Check if it's a hyperparameter change only
        hyperparam_patterns = [r".*\.json$", r".*\.yaml$", r".*\.toml$",
                               r".*config.*", r".*settings.*"]
        if all(any(re.match(p, f) for p in hyperparam_patterns) for f in files):
            return Tier.TIER_1_HYPERPARAM
        
        # Default to Tier 2 for code changes
        code_extensions = {".py", ".rs", ".cpp", ".c", ".go", ".js", ".ts"}
        if any(Path(f).suffix in code_extensions for f in files):
            return Tier.TIER_2_CODE_MUTATION
        
        # Unknown changes are treated as Tier 3 (fail-safe)
        return Tier.TIER_3_CORE_FILE

    # ─── Logging ─────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, request_id: str, tier: str,
                   message: str, details: Dict):
        event = ApprovalEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            request_id=request_id,
            tier=tier,
            message=message,
            details=details,
            checksum="",
        )
        log_path = Path(self.config["approval_log"])
        try:
            with open(log_path, "a") as f:
                f.write(event.to_line() + "\n")
        except Exception as e:
            print(f"APPROVAL LOG FAILURE: {e}", file=sys.stderr)

    # ─── Approval Storage ────────────────────────────────────────────────────

    def _save_request(self, request: ChangeRequest):
        path = self._pending_dir / f"{request.request_id}.json"
        with open(path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)

    def _load_request(self, request_id: str) -> Optional[ChangeRequest]:
        path = self._pending_dir / f"{request_id}.json"
        if path.exists():
            with open(path, "r") as f:
                return ChangeRequest(**json.load(f))
        return None

    def _update_request_status(self, request_id: str, status: str,
                               approved: bool = False):
        req = self._load_request(request_id)
        if req:
            req.status = status
            req.alex_approved = approved
            if approved:
                req.approval_time = datetime.now().isoformat()
            self._save_request(req)

    # ─── Request Processing ──────────────────────────────────────────────────

    def request_change(self, files: List[str], diff: str = "",
                       impact: str = "", proposed_by: str = "moses") -> Dict:
        """
        Process a change request through the approval gates.
        Returns dict with 'approved', 'tier', 'request_id', 'message'.
        """
        tier = self.classify_change(files, diff)
        request = ChangeRequest.create(tier, files, diff, impact, proposed_by)
        
        self._save_request(request)
        self._log_event("REQUEST_CREATED", request.request_id, tier.name,
                       f"Change request created: {tier.name}",
                       {"files": files, "impact": impact})

        if tier == Tier.TIER_1_HYPERPARAM and self.config["auto_approve_tier1"]:
            request.status = "APPROVED"
            request.alex_approved = True
            request.approval_time = datetime.now().isoformat()
            self._save_request(request)
            self._log_event("AUTO_APPROVED", request.request_id, tier.name,
                           "Tier 1 auto-approved", {})
            return {
                "approved": True,
                "tier": tier.name,
                "request_id": request.request_id,
                "message": "Tier 1 hyperparameter change auto-approved.",
            }

        if tier == Tier.TIER_2_CODE_MUTATION and self.config["auto_approve_tier2"]:
            request.status = "APPROVED"
            request.alex_approved = True
            request.approval_time = datetime.now().isoformat()
            self._save_request(request)
            self._log_event("AUTO_APPROVED", request.request_id, tier.name,
                           "Tier 2 auto-approved", {})
            return {
                "approved": True,
                "tier": tier.name,
                "request_id": request.request_id,
                "message": "Tier 2 code mutation auto-approved (logged).",
            }

        # Tier 3 and 4 require Alex approval
        self.notifier.send_approval_request(request)
        self._log_event("AWAITING_APPROVAL", request.request_id, tier.name,
                       "Sent to Alex for approval", {})
        
        return {
            "approved": False,
            "tier": tier.name,
            "request_id": request.request_id,
            "message": f"{TIER_NAMES[tier]} change requires Alex approval. "
                      f"Request sent: {request.request_id}",
        }

    def check_approval(self, request_id: str) -> Dict:
        """Check if a pending request has been approved."""
        req = self._load_request(request_id)
        if not req:
            return {"approved": False, "error": "Request not found"}
        
        if req.status == "APPROVED":
            # For Tier 4, check the waiting period
            if req.tier == Tier.TIER_4_ARCHITECTURE.name and req.scheduled_time:
                scheduled = datetime.fromisoformat(req.scheduled_time)
                if datetime.now() < scheduled:
                    remaining = scheduled - datetime.now()
                    return {
                        "approved": True,
                        "executable": False,
                        "wait_remaining_hours": remaining.total_seconds() / 3600,
                        "message": f"Approved but waiting period active. "
                                  f"Execute after {req.scheduled_time}",
                    }
            return {
                "approved": True,
                "executable": True,
                "message": "Approved and ready to execute.",
            }
        
        if req.status == "DENIED":
            return {"approved": False, "message": "Request denied by Alex."}
        
        return {"approved": False, "status": req.status,
                "message": "Awaiting Alex approval."}

    def process_alex_response(self, response_text: str) -> Dict:
        """
        Process Alex's response from Telegram.
        Expected formats:
          APPROVE <request_id>
          DENY <request_id>
        """
        response_text = response_text.strip().upper()
        
        approve_match = re.match(r"APPROVE\s+(\w+)", response_text)
        deny_match = re.match(r"DENY\s+(\w+)", response_text)
        
        if approve_match:
            req_id = approve_match.group(1)
            req = self._load_request(req_id)
            if not req:
                return {"error": f"Request {req_id} not found"}
            
            self._update_request_status(req_id, "APPROVED", approved=True)
            self._log_event("MANUALLY_APPROVED", req_id, req.tier,
                           "Approved by Alex", {})
            self.notifier.send_notification(
                "✅ Approval Granted",
                f"Request `{req_id}` approved.\n"
                f"Tier: {TIER_NAMES.get(Tier[req.tier], req.tier)}"
            )
            return {"success": True, "request_id": req_id, "action": "APPROVED"}
        
        if deny_match:
            req_id = deny_match.group(1)
            req = self._load_request(req_id)
            if not req:
                return {"error": f"Request {req_id} not found"}
            
            self._update_request_status(req_id, "DENIED")
            self._log_event("MANUALLY_DENIED", req_id, req.tier,
                           "Denied by Alex", {})
            self.notifier.send_notification(
                "❌ Approval Denied",
                f"Request `{req_id}` denied."
            )
            return {"success": True, "request_id": req_id, "action": "DENIED"}
        
        return {"error": "Unrecognized command. Use APPROVE <id> or DENY <id>"}

    # ─── Predicted Impact Analysis ───────────────────────────────────────────

    def analyze_impact(self, files: List[str], diff: str) -> str:
        """
        Generate a predicted impact summary for the change.
        This is a heuristic analysis — not a guarantee.
        """
        impacts = []
        
        # Count lines changed
        added = diff.count("\n+")
        removed = diff.count("\n-")
        impacts.append(f"Lines: +{added}/-{removed}")
        
        # Check for risky patterns
        risky_patterns = {
            r"import\s+os\s*;.*system": "System command execution",
            r"eval\s*\(": "Dynamic code evaluation",
            r"exec\s*\(": "Code execution",
            r"subprocess\.call": "Subprocess invocation",
            r"__import__": "Dynamic importing",
            r"open\s*\(.*['\"]w": "File write operations",
            r"rm\s+-rf": "Dangerous deletion",
            r"shutdown|reboot|halt": "System shutdown",
        }
        
        risks_found = []
        for pattern, description in risky_patterns.items():
            if re.search(pattern, diff, re.IGNORECASE):
                risks_found.append(description)
        
        if risks_found:
            impacts.append(f"⚠️ Risk patterns detected: {', '.join(risks_found)}")
        
        # File-specific impacts
        if any("test" in f for f in files):
            impacts.append("Affects test suite")
        if any("safety" in f for f in files):
            impacts.append("⚠️ MODIFIES SAFETY SYSTEM")
        if any("config" in f for f in files):
            impacts.append("Configuration changes")
        
        return "\n".join(impacts) if impacts else "No significant impact predicted."

    def request_with_impact(self, files: List[str], diff: str = "",
                           proposed_by: str = "moses") -> Dict:
        """Convenience method that auto-generates impact analysis."""
        impact = self.analyze_impact(files, diff)
        return self.request_change(files, diff, impact, proposed_by)


# ─── Singleton & Helpers ─────────────────────────────────────────────────────

_default_gates: Optional[ApprovalGates] = None


def get_gates(config: Optional[Dict] = None) -> ApprovalGates:
    global _default_gates
    if _default_gates is None or config is not None:
        _default_gates = ApprovalGates(config)
    return _default_gates


def classify(files: List[str], diff: str = "") -> str:
    return get_gates().classify_change(files, diff).name


def request(files: List[str], diff: str = "", impact: str = "") -> Dict:
    return get_gates().request_change(files, diff, impact)


def check(request_id: str) -> Dict:
    return get_gates().check_approval(request_id)


def process_response(text: str) -> Dict:
    return get_gates().process_alex_response(text)
