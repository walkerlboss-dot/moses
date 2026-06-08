"""
integrity_checker.py — System integrity verification for Moses v4.0.

Verifies:
- Core files haven't been corrupted
- Checksummed critical configs
- All imports resolve
- No circular dependencies introduced

Fail-safe: any integrity failure triggers alert and optionally stops execution.
"""

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "checksums_file": ".moses_checksums.json",
    "integrity_log": ".moses_integrity_log.jsonl",
    "core_files": [
        "moses/safety/__init__.py",
        "moses/safety/bounds_checker.py",
        "moses/safety/approval_gates.py",
        "moses/safety/drift_detector.py",
        "moses/safety/integrity_checker.py",
    ],
    "critical_configs": [
        "config.json",
        "moses_config.yaml",
        ".env",
    ],
    "scan_dirs": ["moses"],
    "exclude_patterns": ["__pycache__", ".git", "*.pyc", ".pytest_cache"],
    "auto_stop_on_failure": True,
    "allow_missing_core": False,  # Fail-safe: core files must exist
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ChecksumRecord:
    path: str
    sha256: str
    size: int
    modified: str
    verified: bool

    def to_dict(self):
        return asdict(self)


@dataclass
class IntegrityEvent:
    timestamp: str
    event_type: str
    severity: str      # INFO, WARNING, CRITICAL
    message: str
    file_path: Optional[str]
    expected_checksum: Optional[str]
    actual_checksum: Optional[str]
    checksum: str

    def to_line(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "expected_checksum": self.expected_checksum,
            "actual_checksum": self.actual_checksum,
        }
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return json.dumps({"payload": payload, "checksum": checksum})


@dataclass
class ImportCheck:
    file_path: str
    import_path: str
    is_relative: bool
    resolved: bool
    error: Optional[str]


@dataclass
class DependencyGraph:
    nodes: Set[str]
    edges: List[Tuple[str, str]]  # (from, to)
    cycles: List[List[str]]


# ─── Integrity Checker ───────────────────────────────────────────────────────

class IntegrityChecker:
    """
    Verifies system integrity through multiple independent checks.
    
    Principles:
    - Defense in depth: checksums, imports, and dependency checks are independent
    - Fail-safe: any critical failure can stop execution
    - Immutable audit: all checks logged
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.checksums: Dict[str, ChecksumRecord] = self._load_checksums()

    # ─── Logging ─────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, severity: str, message: str,
                   file_path: Optional[str] = None,
                   expected: Optional[str] = None,
                   actual: Optional[str] = None):
        event = IntegrityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            message=message,
            file_path=file_path,
            expected_checksum=expected,
            actual_checksum=actual,
            checksum="",
        )
        log_path = Path(self.config["integrity_log"])
        try:
            with open(log_path, "a") as f:
                f.write(event.to_line() + "\n")
        except Exception as e:
            print(f"INTEGRITY LOG FAILURE: {e}", file=sys.stderr)

    # ─── Checksum Management ─────────────────────────────────────────────────

    def _compute_checksum(self, filepath: Path) -> Optional[str]:
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def _load_checksums(self) -> Dict[str, ChecksumRecord]:
        path = Path(self.config["checksums_file"])
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return {k: ChecksumRecord(**v) for k, v in data.items()}
        except Exception as e:
            self._log_event("CHECKSUM_LOAD_ERROR", "WARNING",
                           f"Failed to load checksums: {e}")
            return {}

    def _save_checksums(self):
        path = Path(self.config["checksums_file"])
        data = {k: v.to_dict() for k, v in self.checksums.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def register_checksum(self, filepath: str) -> Optional[ChecksumRecord]:
        """Register or update checksum for a file."""
        path = Path(filepath)
        if not path.exists():
            self._log_event("FILE_NOT_FOUND", "WARNING",
                           f"Cannot checksum missing file: {filepath}", filepath)
            return None
        
        checksum = self._compute_checksum(path)
        if not checksum:
            self._log_event("CHECKSUM_COMPUTE_ERROR", "WARNING",
                           f"Failed to compute checksum: {filepath}", filepath)
            return None
        
        record = ChecksumRecord(
            path=filepath,
            sha256=checksum,
            size=path.stat().st_size,
            modified=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            verified=True,
        )
        self.checksums[filepath] = record
        self._save_checksums()
        self._log_event("CHECKSUM_REGISTERED", "INFO",
                       f"Registered checksum for {filepath}", filepath,
                       expected=checksum)
        return record

    def verify_checksum(self, filepath: str) -> bool:
        """Verify a file against its registered checksum."""
        path = Path(filepath)
        if not path.exists():
            self._log_event("FILE_MISSING", "CRITICAL",
                           f"Core file missing: {filepath}", filepath)
            return False
        
        record = self.checksums.get(filepath)
        if not record:
            # Fail-safe: unregistered core file is a problem
            self._log_event("CHECKSUM_NOT_FOUND", "WARNING",
                           f"No checksum on record for {filepath}", filepath)
            return False
        
        current = self._compute_checksum(path)
        if current != record.sha256:
            self._log_event("CHECKSUM_MISMATCH", "CRITICAL",
                           f"Checksum mismatch for {filepath}", filepath,
                           expected=record.sha256, actual=current)
            return False
        
        self._log_event("CHECKSUM_VERIFIED", "INFO",
                       f"Checksum verified: {filepath}", filepath,
                       expected=record.sha256, actual=current)
        return True

    def verify_all_core_files(self) -> Dict:
        """Verify all configured core files."""
        results = {"verified": [], "failed": [], "missing": []}
        
        for filepath in self.config["core_files"]:
            path = Path(filepath)
            if not path.exists():
                results["missing"].append(filepath)
                self._log_event("CORE_FILE_MISSING", "CRITICAL",
                               f"Core file missing: {filepath}", filepath)
                if self.config["allow_missing_core"]:
                    continue
                else:
                    return results
            
            if self.verify_checksum(filepath):
                results["verified"].append(filepath)
            else:
                results["failed"].append(filepath)
        
        return results

    def initialize_checksums(self):
        """Register checksums for all core files and critical configs."""
        all_files = self.config["core_files"] + self.config["critical_configs"]
        for filepath in all_files:
            if Path(filepath).exists():
                self.register_checksum(filepath)
        self._log_event("CHECKSUMS_INITIALIZED", "INFO",
                       f"Initialized {len(self.checksums)} checksums")

    # ─── Import Resolution ───────────────────────────────────────────────────

    def _extract_imports(self, filepath: Path) -> List[ImportCheck]:
        """Extract all imports from a Python file."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception:
            return []

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportCheck(
                        file_path=str(filepath),
                        import_path=alias.name,
                        is_relative=False,
                        resolved=False,
                        error=None,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level > 0:
                    # Relative import
                    imports.append(ImportCheck(
                        file_path=str(filepath),
                        import_path=f"{'.' * node.level}{module}",
                        is_relative=True,
                        resolved=False,
                        error=None,
                    ))
                else:
                    imports.append(ImportCheck(
                        file_path=str(filepath),
                        import_path=module,
                        is_relative=False,
                        resolved=False,
                        error=None,
                    ))
        return imports

    def _resolve_import(self, imp: ImportCheck, base_dir: Path) -> bool:
        """Attempt to resolve an import to an actual file or installed package."""
        if imp.is_relative:
            # Resolve relative to the importing file
            file_dir = Path(imp.file_path).parent
            parts = imp.import_path.lstrip(".").split(".")
            dots = len(imp.import_path) - len(imp.import_path.lstrip("."))
            
            # Walk up directories for each dot
            current_dir = file_dir
            for _ in range(dots - 1):
                current_dir = current_dir.parent
            
            # Try to find the module
            for part in parts:
                candidate = current_dir / part
                if (candidate / "__init__.py").exists():
                    current_dir = candidate
                    continue
                elif candidate.with_suffix(".py").exists():
                    return True
                current_dir = candidate
            
            return (current_dir / "__init__.py").exists()
        else:
            # Absolute import — check if it's a local module or installed package
            parts = imp.import_path.split(".")
            
            # Check if it's in our scan dirs
            for scan_dir in self.config["scan_dirs"]:
                base = Path(scan_dir)
                current = base
                for part in parts:
                    candidate = current / part
                    if (candidate / "__init__.py").exists():
                        current = candidate
                        continue
                    elif candidate.with_suffix(".py").exists():
                        return True
                    current = candidate
                if (current / "__init__.py").exists():
                    return True
            
            # Check installed packages
            try:
                spec = importlib.util.find_spec(imp.import_path.split(".")[0])
                return spec is not None
            except Exception:
                return False

    def check_imports(self) -> Dict:
        """Verify all imports in scanned files resolve correctly."""
        results = {"resolved": [], "unresolved": [], "errors": []}
        
        exclude = [p.replace("*", ".*") for p in self.config["exclude_patterns"]]
        
        for scan_dir in self.config["scan_dirs"]:
            base = Path(scan_dir)
            if not base.exists():
                continue
            for pyfile in base.rglob("*.py"):
                rel_path = str(pyfile.relative_to(Path.cwd()))
                if any(p in rel_path for p in exclude):
                    continue
                
                imports = self._extract_imports(pyfile)
                for imp in imports:
                    if self._resolve_import(imp, base):
                        results["resolved"].append({
                            "file": imp.file_path,
                            "import": imp.import_path,
                        })
                    else:
                        results["unresolved"].append({
                            "file": imp.file_path,
                            "import": imp.import_path,
                        })
                        self._log_event("UNRESOLVED_IMPORT", "WARNING",
                                       f"Unresolved import: {imp.import_path} "
                                       f"in {imp.file_path}", imp.file_path)
        
        return results

    # ─── Circular Dependency Detection ───────────────────────────────────────

    def _build_dependency_graph(self) -> DependencyGraph:
        """Build a dependency graph from imports."""
        nodes: Set[str] = set()
        edges: List[Tuple[str, str]] = []
        
        exclude = [p.replace("*", ".*") for p in self.config["exclude_patterns"]]
        
        for scan_dir in self.config["scan_dirs"]:
            base = Path(scan_dir)
            if not base.exists():
                continue
            for pyfile in base.rglob("*.py"):
                rel_path = str(pyfile.relative_to(Path.cwd()))
                if any(p in rel_path for p in exclude):
                    continue
                
                module_name = str(pyfile.relative_to(Path.cwd())).replace(os.sep, ".")[:-3]
                nodes.add(module_name)
                
                imports = self._extract_imports(pyfile)
                for imp in imports:
                    if imp.is_relative:
                        # Resolve to absolute module name
                        file_dir = Path(imp.file_path).parent
                        dots = len(imp.import_path) - len(imp.import_path.lstrip("."))
                        current_dir = file_dir
                        for _ in range(dots - 1):
                            current_dir = current_dir.parent
                        rel_module = str(current_dir.relative_to(Path.cwd())).replace(os.sep, ".")
                        target = f"{rel_module}.{imp.import_path.lstrip('.').replace('/', '.')}" if imp.import_path.lstrip(".") else rel_module
                    else:
                        target = imp.import_path
                    
                    edges.append((module_name, target))
                    nodes.add(target)
        
        return DependencyGraph(nodes=nodes, edges=edges, cycles=[])

    def _find_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """Find all cycles in the dependency graph using DFS."""
        adj = defaultdict(list)
        for src, dst in graph.edges:
            adj[src].append(dst)
        
        cycles = []
        visited = set()
        rec_stack = []
        rec_set = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_set:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.append(node)
            rec_set.add(node)
            
            for neighbor in adj[node]:
                if neighbor in graph.nodes:
                    dfs(neighbor, path + [neighbor])
            
            rec_stack.pop()
            rec_set.remove(node)
        
        for node in graph.nodes:
            visited.clear()
            rec_stack.clear()
            rec_set.clear()
            dfs(node, [node])
        
        # Deduplicate cycles
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            normalized = tuple(sorted(cycle))
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(cycle)
        
        return unique_cycles

    def check_circular_dependencies(self) -> Dict:
        """Detect circular dependencies in the codebase."""
        graph = self._build_dependency_graph()
        cycles = self._find_cycles(graph)
        
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            self._log_event("CIRCULAR_DEPENDENCY", "CRITICAL",
                           f"Circular dependency detected: {cycle_str}")
        
        return {
            "cycles_found": len(cycles) > 0,
            "cycle_count": len(cycles),
            "cycles": cycles,
            "modules_checked": len(graph.nodes),
            "dependencies_tracked": len(graph.edges),
        }

    # ─── Master Check ────────────────────────────────────────────────────────

    def check_all(self) -> Dict:
        """Run all integrity checks."""
        self._log_event("INTEGRITY_CHECK_START", "INFO",
                       "Starting full integrity check")
        
        # 1. Check core file checksums
        checksum_results = self.verify_all_core_files()
        
        # 2. Check import resolution
        import_results = self.check_imports()
        
        # 3. Check circular dependencies
        cycle_results = self.check_circular_dependencies()
        
        # Determine overall health
        critical_failures = (
            len(checksum_results.get("failed", [])) > 0 or
            len(checksum_results.get("missing", [])) > 0 or
            cycle_results["cycles_found"]
        )
        
        warnings = len(import_results.get("unresolved", [])) > 0
        
        if critical_failures:
            self._log_event("INTEGRITY_CHECK_FAILED", "CRITICAL",
                           "Critical integrity failures detected")
            if self.config["auto_stop_on_failure"]:
                print("\n" + "="*60, file=sys.stderr)
                print("INTEGRITY CHECK FAILED — STOPPING", file=sys.stderr)
                print("="*60 + "\n", file=sys.stderr)
                sys.exit(78)  # Custom exit code for integrity failure
        elif warnings:
            self._log_event("INTEGRITY_CHECK_WARNINGS", "WARNING",
                           "Integrity check passed with warnings")
        else:
            self._log_event("INTEGRITY_CHECK_PASSED", "INFO",
                           "All integrity checks passed")
        
        return {
            "passed": not critical_failures,
            "critical": critical_failures,
            "warnings": warnings,
            "checksums": checksum_results,
            "imports": import_results,
            "circular_dependencies": cycle_results,
        }

    def get_status(self) -> Dict:
        """Get current integrity status."""
        return {
            "checksums_registered": len(self.checksums),
            "core_files_configured": len(self.config["core_files"]),
            "last_check": self.checksums.get(
                self.config["core_files"][0], ChecksumRecord("", "", 0, "", False)
            ).modified if self.checksums else None,
        }


# ─── Singleton & Helpers ─────────────────────────────────────────────────────

_default_checker: Optional[IntegrityChecker] = None


def get_checker(config: Optional[Dict] = None) -> IntegrityChecker:
    global _default_checker
    if _default_checker is None or config is not None:
        _default_checker = IntegrityChecker(config)
    return _default_checker


def verify() -> Dict:
    return get_checker().check_all()


def init_checksums():
    get_checker().initialize_checksums()


def status() -> Dict:
    return get_checker().get_status()
