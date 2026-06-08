"""
drift_detector.py — Detects undesirable drift in Moses v4.0.

Monitors:
- Code complexity trends
- Test coverage trends
- Documentation freshness
- Code bloat (files growing without purpose)

Alerts if metrics trend negative over time.
Fail-safe: negative trends trigger warnings, then escalation.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "history_file": ".moses_drift_history.json",
    "log_file": ".moses_drift_log.jsonl",
    "scan_dirs": ["moses"],
    "exclude_patterns": ["__pycache__", ".git", "*.pyc", ".pytest_cache"],
    "complexity_threshold": 15,           # McCabe complexity warning
    "complexity_trend_window": 7,         # Days to track
    "coverage_floor": 0.60,               # 60% minimum coverage
    "coverage_decline_threshold": 0.05,   # 5% decline triggers alert
    "docstring_ratio_floor": 0.30,        # 30% of functions need docstrings
    "bloat_threshold_lines": 50,          # Lines added without test coverage
    "bloat_file_threshold": 200,          # File size warning (lines)
    "alert_cooldown_hours": 24,
    "trend_samples_required": 3,          # Min samples for trend analysis
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class FileMetrics:
    path: str
    lines: int
    functions: int
    classes: int
    docstring_count: int
    complexity_score: float
    has_tests: bool
    last_modified: str
    checksum: str

    def docstring_ratio(self) -> float:
        return self.docstring_count / max(self.functions, 1)

    def to_dict(self):
        return asdict(self)


@dataclass
class DriftSnapshot:
    timestamp: str
    files: Dict[str, FileMetrics]
    overall_coverage: Optional[float]
    avg_complexity: float
    total_lines: int
    docstring_ratio: float
    checksum: str

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "overall_coverage": self.overall_coverage,
            "avg_complexity": self.avg_complexity,
            "total_lines": self.total_lines,
            "docstring_ratio": self.docstring_ratio,
            "checksum": self.checksum,
        }


@dataclass
class DriftEvent:
    timestamp: str
    event_type: str
    severity: str      # INFO, WARNING, CRITICAL
    message: str
    metric: str
    current_value: float
    baseline_value: Optional[float]
    trend: str         # IMPROVING, STABLE, DEGRADING
    checksum: str

    def to_line(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "trend": self.trend,
        }
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return json.dumps({"payload": payload, "checksum": checksum})


# ─── Metrics Extraction ──────────────────────────────────────────────────────

class MetricsExtractor:
    """Extracts code metrics from Python files."""

    @staticmethod
    def extract_file_metrics(filepath: Path) -> Optional[FileMetrics]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
                lines = source.count("\n") + 1
        except Exception:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return FileMetrics(
                path=str(filepath),
                lines=lines,
                functions=0, classes=0, docstring_count=0,
                complexity_score=0, has_tests=False,
                last_modified=datetime.fromtimestamp(
                    os.path.getmtime(filepath)).isoformat(),
                checksum=hashlib.sha256(source.encode()).hexdigest()[:16],
            )

        functions = 0
        classes = 0
        docstrings = 0
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
                if ast.get_docstring(node):
                    docstrings += 1
                complexity += MetricsExtractor._cyclomatic_complexity(node)
            elif isinstance(node, ast.ClassDef):
                classes += 1
                if ast.get_docstring(node):
                    docstrings += 1

        has_tests = "test" in filepath.name or "test" in str(filepath.parent)

        return FileMetrics(
            path=str(filepath),
            lines=lines,
            functions=functions,
            classes=classes,
            docstring_count=docstrings,
            complexity_score=complexity,
            has_tests=has_tests,
            last_modified=datetime.fromtimestamp(
                os.path.getmtime(filepath)).isoformat(),
            checksum=hashlib.sha256(source.encode()).hexdigest()[:16],
        )

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        """Simplified McCabe complexity calculation."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For,
                                ast.ExceptHandler, ast.With,
                                ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    @staticmethod
    def get_coverage() -> Optional[float]:
        """Attempt to get test coverage. Returns None if unavailable."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=moses", "--cov-report=json",
                 "-q", "--tb=no"],
                capture_output=True, text=True, timeout=120
            )
            cov_file = Path("coverage.json")
            if cov_file.exists():
                with open(cov_file, "r") as f:
                    data = json.load(f)
                totals = data.get("totals", {})
                percent = totals.get("percent_covered")
                cov_file.unlink(missing_ok=True)
                return percent / 100 if percent else None
        except Exception:
            pass
        return None


# ─── Drift Detector ──────────────────────────────────────────────────────────

class DriftDetector:
    """
    Detects undesirable drift in code quality metrics.
    
    Principles:
    - Trend-based: single bad metric isn't enough; we track trends
    - Conservative: alert on sustained negative trends
    - Actionable: provide specific files/metrics causing drift
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.history: List[DriftSnapshot] = self._load_history()
        self._last_alert_time: Optional[datetime] = None

    # ─── History Management ──────────────────────────────────────────────────

    def _load_history(self) -> List[DriftSnapshot]:
        path = Path(self.config["history_file"])
        if not path.exists():
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            snapshots = []
            for item in data:
                files = {k: FileMetrics(**v) for k, v in item.get("files", {}).items()}
                snapshots.append(DriftSnapshot(
                    timestamp=item["timestamp"],
                    files=files,
                    overall_coverage=item.get("overall_coverage"),
                    avg_complexity=item["avg_complexity"],
                    total_lines=item["total_lines"],
                    docstring_ratio=item["docstring_ratio"],
                    checksum=item.get("checksum", ""),
                ))
            return snapshots
        except Exception as e:
            self._log_event("HISTORY_LOAD_ERROR", "WARNING",
                           f"Failed to load history: {e}", "history", 0, None, "UNKNOWN")
            return []

    def _save_history(self):
        path = Path(self.config["history_file"])
        data = [s.to_dict() for s in self.history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _log_event(self, event_type: str, severity: str, message: str,
                   metric: str, current: float, baseline: Optional[float],
                   trend: str):
        event = DriftEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            message=message,
            metric=metric,
            current_value=current,
            baseline_value=baseline,
            trend=trend,
            checksum="",
        )
        log_path = Path(self.config["log_file"])
        try:
            with open(log_path, "a") as f:
                f.write(event.to_line() + "\n")
        except Exception as e:
            print(f"DRIFT LOG FAILURE: {e}", file=sys.stderr)

    # ─── Snapshot Creation ───────────────────────────────────────────────────

    def take_snapshot(self) -> DriftSnapshot:
        """Capture current state of all monitored files."""
        files: Dict[str, FileMetrics] = {}
        total_lines = 0
        total_complexity = 0
        total_functions = 0
        total_docstrings = 0

        exclude = [re.compile(p.replace("*", ".*")) for p in self.config["exclude_patterns"]]

        for scan_dir in self.config["scan_dirs"]:
            path = Path(scan_dir)
            if not path.exists():
                continue
            for pyfile in path.rglob("*.py"):
                rel_path = str(pyfile.relative_to(Path.cwd()))
                if any(p.match(rel_path) or p.match(pyfile.name) for p in exclude):
                    continue
                
                metrics = MetricsExtractor.extract_file_metrics(pyfile)
                if metrics:
                    files[rel_path] = metrics
                    total_lines += metrics.lines
                    total_complexity += metrics.complexity_score
                    total_functions += metrics.functions
                    total_docstrings += metrics.docstring_count

        coverage = MetricsExtractor.get_coverage()
        avg_complexity = total_complexity / max(len(files), 1)
        doc_ratio = total_docstrings / max(total_functions, 1)

        snapshot = DriftSnapshot(
            timestamp=datetime.now().isoformat(),
            files=files,
            overall_coverage=coverage,
            avg_complexity=avg_complexity,
            total_lines=total_lines,
            docstring_ratio=doc_ratio,
            checksum="",
        )
        
        # Compute checksum
        payload = json.dumps(snapshot.to_dict(), sort_keys=True)
        snapshot.checksum = hashlib.sha256(payload.encode()).hexdigest()[:16]
        
        return snapshot

    def record_snapshot(self):
        """Take a snapshot and add to history."""
        snapshot = self.take_snapshot()
        self.history.append(snapshot)
        
        # Keep only recent history (configurable window)
        cutoff = datetime.now() - timedelta(days=self.config["complexity_trend_window"] * 2)
        self.history = [
            h for h in self.history
            if datetime.fromisoformat(h.timestamp) > cutoff
        ]
        
        self._save_history()
        self._log_event("SNAPSHOT_RECORDED", "INFO",
                       f"Recorded snapshot: {len(snapshot.files)} files, "
                       f"{snapshot.total_lines} lines",
                       "snapshot", float(len(snapshot.files)), None, "STABLE")
        return snapshot

    # ─── Trend Analysis ──────────────────────────────────────────────────────

    def _get_trend(self, metric_extractor: callable) -> Tuple[str, Optional[float], Optional[float]]:
        """Analyze trend for a given metric. Returns (trend, current, baseline)."""
        if len(self.history) < self.config["trend_samples_required"]:
            return "INSUFFICIENT_DATA", None, None
        
        values = [metric_extractor(h) for h in self.history]
        values = [v for v in values if v is not None]
        
        if len(values) < self.config["trend_samples_required"]:
            return "INSUFFICIENT_DATA", None, None
        
        current = values[-1]
        baseline = values[0]
        
        # Simple linear trend: compare first half avg vs second half avg
        mid = len(values) // 2
        first_avg = sum(values[:mid]) / max(mid, 1)
        second_avg = sum(values[mid:]) / max(len(values) - mid, 1)
        
        # Determine if metric is "good" (higher is better) or "bad"
        # For coverage, docstring_ratio: higher is better
        # For complexity: lower is better
        
        return current, baseline, first_avg, second_avg

    def analyze_complexity_trend(self) -> List[Dict]:
        """Check if complexity is trending upward."""
        alerts = []
        if len(self.history) < self.config["trend_samples_required"]:
            return alerts
        
        current = self.history[-1]
        baseline = self.history[0]
        
        # Check overall average
        if current.avg_complexity > baseline.avg_complexity * 1.1:
            alerts.append({
                "metric": "avg_complexity",
                "current": current.avg_complexity,
                "baseline": baseline.avg_complexity,
                "trend": "DEGRADING",
                "message": f"Average complexity increased from "
                          f"{baseline.avg_complexity:.1f} to {current.avg_complexity:.1f}",
            })
        
        # Check individual files
        for path, metrics in current.files.items():
            if metrics.complexity_score > self.config["complexity_threshold"]:
                old = baseline.files.get(path)
                if old and metrics.complexity_score > old.complexity_score * 1.2:
                    alerts.append({
                        "metric": f"complexity:{path}",
                        "current": metrics.complexity_score,
                        "baseline": old.complexity_score,
                        "trend": "DEGRADING",
                        "message": f"{path} complexity increased significantly "
                                  f"({old.complexity_score} -> {metrics.complexity_score})",
                    })
        
        return alerts

    def analyze_coverage_trend(self) -> List[Dict]:
        """Check if test coverage is declining."""
        alerts = []
        coverage_history = [h.overall_coverage for h in self.history if h.overall_coverage is not None]
        
        if len(coverage_history) < self.config["trend_samples_required"]:
            return alerts
        
        current = coverage_history[-1]
        baseline = coverage_history[0]
        
        if current < self.config["coverage_floor"]:
            alerts.append({
                "metric": "coverage",
                "current": current,
                "baseline": baseline,
                "trend": "DEGRADING",
                "message": f"Coverage {current:.1%} below floor {self.config['coverage_floor']:.1%}",
            })
        
        if baseline > 0 and (baseline - current) > self.config["coverage_decline_threshold"]:
            alerts.append({
                "metric": "coverage_decline",
                "current": current,
                "baseline": baseline,
                "trend": "DEGRADING",
                "message": f"Coverage declined by {baseline - current:.1%} "
                          f"(threshold: {self.config['coverage_decline_threshold']:.1%})",
            })
        
        return alerts

    def analyze_documentation_trend(self) -> List[Dict]:
        """Check if documentation is decaying."""
        alerts = []
        if len(self.history) < self.config["trend_samples_required"]:
            return alerts
        
        current = self.history[-1]
        baseline = self.history[0]
        
        if current.docstring_ratio < self.config["docstring_ratio_floor"]:
            alerts.append({
                "metric": "docstring_ratio",
                "current": current.docstring_ratio,
                "baseline": baseline.docstring_ratio,
                "trend": "DEGRADING",
                "message": f"Docstring ratio {current.docstring_ratio:.1%} below "
                          f"floor {self.config['docstring_ratio_floor']:.1%}",
            })
        
        if current.docstring_ratio < baseline.docstring_ratio * 0.9:
            alerts.append({
                "metric": "docstring_decline",
                "current": current.docstring_ratio,
                "baseline": baseline.docstring_ratio,
                "trend": "DEGRADING",
                "message": f"Docstring ratio declined from {baseline.docstring_ratio:.1%} "
                          f"to {current.docstring_ratio:.1%}",
            })
        
        return alerts

    def analyze_bloat(self) -> List[Dict]:
        """Detect code bloat: files growing without tests or purpose."""
        alerts = []
        if len(self.history) < 2:
            return alerts
        
        current = self.history[-1]
        previous = self.history[-2]
        
        for path, metrics in current.files.items():
            old = previous.files.get(path)
            if not old:
                continue
            
            line_growth = metrics.lines - old.lines
            if line_growth > self.config["bloat_threshold_lines"]:
                # Check if growth is accompanied by tests
                if not metrics.has_tests:
                    alerts.append({
                        "metric": f"bloat:{path}",
                        "current": metrics.lines,
                        "baseline": old.lines,
                        "trend": "DEGRADING",
                        "message": f"{path} grew by {line_growth} lines without test coverage",
                    })
            
            if metrics.lines > self.config["bloat_file_threshold"]:
                alerts.append({
                    "metric": f"large_file:{path}",
                    "current": metrics.lines,
                    "baseline": old.lines,
                    "trend": "DEGRADING",
                    "message": f"{path} is {metrics.lines} lines (threshold: "
                              f"{self.config['bloat_file_threshold']})",
                })
        
        # Also check total growth rate
        total_growth = current.total_lines - previous.total_lines
        if total_growth > self.config["bloat_threshold_lines"] * 5:
            alerts.append({
                "metric": "total_growth",
                "current": current.total_lines,
                "baseline": previous.total_lines,
                "trend": "DEGRADING",
                "message": f"Total codebase grew by {total_growth} lines since last snapshot",
            })
        
        return alerts

    # ─── Master Analysis ─────────────────────────────────────────────────────

    def check_all(self) -> Dict:
        """Run all drift checks and return consolidated report."""
        if not self.history:
            self.record_snapshot()
        
        all_alerts = []
        all_alerts.extend(self.analyze_complexity_trend())
        all_alerts.extend(self.analyze_coverage_trend())
        all_alerts.extend(self.analyze_documentation_trend())
        all_alerts.extend(self.analyze_bloat())
        
        # Categorize by severity
        critical = [a for a in all_alerts if "safety" in a.get("message", "").lower()]
        warnings = [a for a in all_alerts if a not in critical]
        
        # Log all alerts
        for alert in all_alerts:
            severity = "CRITICAL" if alert in critical else "WARNING"
            self._log_event(
                "DRIFT_DETECTED", severity, alert["message"],
                alert["metric"], alert["current"], alert.get("baseline"), alert["trend"]
            )
        
        # Cooldown check
        can_alert = True
        if self._last_alert_time:
            cooldown = timedelta(hours=self.config["alert_cooldown_hours"])
            if datetime.now() - self._last_alert_time < cooldown:
                can_alert = False
        
        if critical and can_alert:
            self._last_alert_time = datetime.now()
        
        return {
            "drift_detected": len(all_alerts) > 0,
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "alerts": all_alerts,
            "snapshot": self.history[-1].to_dict() if self.history else None,
            "history_length": len(self.history),
        }

    def get_summary(self) -> Dict:
        """Get current metrics summary without trend analysis."""
        if not self.history:
            return {"error": "No snapshots recorded yet"}
        
        latest = self.history[-1]
        return {
            "timestamp": latest.timestamp,
            "files_tracked": len(latest.files),
            "total_lines": latest.total_lines,
            "avg_complexity": latest.avg_complexity,
            "coverage": latest.overall_coverage,
            "docstring_ratio": latest.docstring_ratio,
        }


# ─── Singleton & Helpers ─────────────────────────────────────────────────────

_default_detector: Optional[DriftDetector] = None


def get_detector(config: Optional[Dict] = None) -> DriftDetector:
    global _default_detector
    if _default_detector is None or config is not None:
        _default_detector = DriftDetector(config)
    return _default_detector


def snapshot() -> DriftSnapshot:
    return get_detector().record_snapshot()


def check() -> Dict:
    return get_detector().check_all()


def summary() -> Dict:
    return get_detector().get_summary()
