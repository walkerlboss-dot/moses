"""
moses.safety — Safety system for Moses v4.0 recursive self-learning.

This package provides defense-in-depth safety mechanisms to prevent
runaway self-modification, ensure human oversight, and maintain
system integrity.

Modules:
    bounds_checker    — Hard limits on resources and performance
    approval_gates    — Tiered human oversight for self-modification
    drift_detector    — Detects undesirable code quality drift
    integrity_checker — Verifies system integrity and detects corruption

Usage:
    from moses.safety import SafetySystem
    
    safety = SafetySystem()
    safety.initialize()      # Set up checksums and baselines
    safety.pre_change()      # Call before any self-modification
    safety.post_change()     # Call after self-modification
    safety.periodic_check()  # Call periodically (e.g., hourly)

Principles:
    - Defense in depth: multiple independent checks
    - Fail-safe: if unsure, stop and ask
    - Immutable logging: all safety events logged with checksums
    - Human oversight: Alex retains control over high-risk changes
"""

import sys
from datetime import datetime
from typing import Dict, List, Optional

# Import all safety modules
from . import bounds_checker
from . import approval_gates
from . import drift_detector
from . import integrity_checker


__version__ = "4.0.0"
__all__ = [
    "SafetySystem",
    "bounds_checker",
    "approval_gates",
    "drift_detector",
    "integrity_checker",
]


class SafetySystem:
    """
    Unified safety system orchestrating all safety checks.
    
    This is the main interface for the Moses safety subsystem.
    It coordinates the four independent safety modules and ensures
    they operate with defense-in-depth principles.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._initialized = False
        self._last_periodic_check: Optional[datetime] = None

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def initialize(self):
        """
        Initialize the safety system.
        
        This should be called once at system startup.
        It establishes baselines, registers checksums, and verifies
        initial system integrity.
        """
        print("[SAFETY] Initializing Moses v4.0 Safety System...")
        
        # Initialize integrity checksums
        integrity_checker.init_checksums()
        
        # Take initial drift snapshot
        drift_detector.snapshot()
        
        # Verify initial integrity
        result = integrity_checker.verify()
        if not result["passed"]:
            print("[SAFETY] FATAL: Initial integrity check failed!")
            sys.exit(79)
        
        self._initialized = True
        print("[SAFETY] Safety system initialized successfully.")
        return {"status": "initialized", "integrity": result}

    # ─── Change Gate ─────────────────────────────────────────────────────────

    def pre_change(self, files: List[str], diff: str = "",
                   impact: str = "", proposed_by: str = "moses") -> Dict:
        """
        Gate that must be passed BEFORE any self-modification.
        
        This method:
        1. Checks resource bounds
        2. Classifies the change tier
        3. Requires approval for Tier 3/4
        4. Verifies system integrity
        
        Returns a dict with 'proceed' (bool) and details.
        """
        if not self._initialized:
            print("[SAFETY] WARNING: Safety system not initialized!")
        
        print(f"[SAFETY] Pre-change check for {len(files)} files...")
        
        # 1. Check bounds
        bounds_ok = bounds_checker.check_all()
        if not bounds_ok:
            return {
                "proceed": False,
                "reason": "Resource bounds exceeded",
                "action": "STOP",
            }
        
        # 2. Check integrity before change
        integrity_result = integrity_checker.verify()
        if not integrity_result["passed"]:
            return {
                "proceed": False,
                "reason": "Integrity check failed",
                "details": integrity_result,
                "action": "STOP",
            }
        
        # 3. Request approval through gates
        approval = approval_gates.request(files, diff, impact)
        
        if not approval["approved"]:
            return {
                "proceed": False,
                "reason": approval["message"],
                "request_id": approval.get("request_id"),
                "tier": approval.get("tier"),
                "action": "AWAIT_APPROVAL",
            }
        
        # 4. For Tier 4, check waiting period
        if approval.get("tier") == "TIER_4_ARCHITECTURE":
            check = approval_gates.check(approval["request_id"])
            if not check.get("executable", True):
                return {
                    "proceed": False,
                    "reason": check["message"],
                    "request_id": approval["request_id"],
                    "action": "WAIT",
                }
        
        print(f"[SAFETY] Pre-change check PASSED. Tier: {approval.get('tier')}")
        return {
            "proceed": True,
            "request_id": approval.get("request_id"),
            "tier": approval.get("tier"),
            "action": "PROCEED",
        }

    def post_change(self, files: List[str], lines_changed: int = 0):
        """
        Call AFTER any self-modification to update tracking.
        
        This records the change in bounds and triggers follow-up checks.
        """
        print(f"[SAFETY] Post-change recording for {len(files)} files...")
        
        # Record code change in bounds checker
        bounds_checker.record_code_change(lines_changed, len(files))
        
        # Re-verify integrity (some files changed)
        integrity_checker.verify()
        
        # Take new drift snapshot
        drift_detector.snapshot()
        
        print("[SAFETY] Post-change recording complete.")

    # ─── Periodic Monitoring ─────────────────────────────────────────────────

    def periodic_check(self) -> Dict:
        """
        Run periodic safety checks.
        
        Should be called regularly (e.g., every hour or after each
        learning iteration) to detect drift and bound violations.
        """
        print("[SAFETY] Running periodic safety check...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "bounds": None,
            "drift": None,
            "integrity": None,
        }
        
        # Check bounds
        bounds_status = bounds_checker.get_status()
        results["bounds"] = bounds_status
        
        # Check drift
        drift_results = drift_detector.check()
        results["drift"] = drift_results
        
        # Check integrity
        integrity_results = integrity_checker.verify()
        results["integrity"] = integrity_results
        
        # Determine overall status
        critical = (
            bounds_status.get("shutdown_triggered") or
            drift_results.get("critical_count", 0) > 0 or
            not integrity_results.get("passed", True)
        )
        
        warnings = (
            drift_results.get("warning_count", 0) > 0 or
            integrity_results.get("warnings", False)
        )
        
        if critical:
            results["status"] = "CRITICAL"
            results["action"] = "STOP"
        elif warnings:
            results["status"] = "WARNING"
            results["action"] = "MONITOR"
        else:
            results["status"] = "HEALTHY"
            results["action"] = "CONTINUE"
        
        self._last_periodic_check = datetime.now()
        print(f"[SAFETY] Periodic check complete: {results['status']}")
        return results

    # ─── Status & Utilities ──────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get comprehensive safety system status."""
        return {
            "initialized": self._initialized,
            "version": __version__,
            "bounds": bounds_checker.get_status(),
            "drift": drift_detector.summary(),
            "integrity": integrity_checker.status(),
            "last_periodic_check": self._last_periodic_check.isoformat() if self._last_periodic_check else None,
        }

    def emergency_stop(self, reason: str):
        """
        Trigger an emergency stop.
        
        This is a manual override that immediately halts the system.
        """
        print(f"\n{'='*60}")
        print("EMERGENCY STOP TRIGGERED")
        print(f"Reason: {reason}")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")
        sys.exit(80)  # Custom exit code for emergency stop


# ─── Convenience Functions ───────────────────────────────────────────────────

def initialize() -> Dict:
    """Initialize the safety system."""
    return SafetySystem().initialize()


def pre_change(files: List[str], diff: str = "", impact: str = "") -> Dict:
    """Pre-change gate."""
    return SafetySystem().pre_change(files, diff, impact)


def post_change(files: List[str], lines_changed: int = 0):
    """Post-change recording."""
    SafetySystem().post_change(files, lines_changed)


def periodic_check() -> Dict:
    """Run periodic safety checks."""
    return SafetySystem().periodic_check()


def get_status() -> Dict:
    """Get safety system status."""
    return SafetySystem().get_status()
