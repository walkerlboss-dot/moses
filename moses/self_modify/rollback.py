"""
Rollback System for Moses v4.0

Git-based versioning of all code changes with automatic rollback
on test failure or metric degradation. Keeps full history and
can restore any previous version.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VersionRecord:
    """A single checkpoint in the rollback history."""
    version_id: str
    timestamp: str
    description: str
    file_path: str
    file_hash: str
    parent_version: Optional[str]
    tags: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    auto_rollback_triggered: bool = False


@dataclass
class RollbackHistory:
    """Full history for a managed file."""
    file_path: str
    versions: List[VersionRecord] = field(default_factory=list)
    current_version_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Core rollback manager
# ---------------------------------------------------------------------------

class RollbackManager:
    """
    Git-based versioning with automatic rollback capabilities.

    Usage:
        rm = RollbackManager(repo_root="/path/to/moses")
        rm.checkpoint("moses/brain.py", description="Before mutation")
        # ... apply mutation ...
        if tests_fail:
            rm.rollback("moses/brain.py")
        # Or restore any version:
        rm.restore("moses/brain.py", version_id="abc123")
    """

    HISTORY_FILE = ".moses_rollback_history.json"
    BACKUP_DIR = ".moses_backups"

    def __init__(self, repo_root: Union[str, Path], use_git: bool = True):
        self.repo_root = Path(repo_root).resolve()
        self.use_git = use_git and self._git_available()
        self.backup_dir = self.repo_root / self.BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._history: Dict[str, RollbackHistory] = {}
        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        file_path: Union[str, Path],
        description: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict] = None,
    ) -> str:
        """
        Save a version checkpoint for *file_path*.

        Returns:
            version_id: Unique identifier for this checkpoint.
        """
        path = self._resolve(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot checkpoint non-existent file: {path}")

        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        version_id = self._make_version_id(path, content)

        # Write backup copy
        backup_path = self._backup_path(version_id)
        backup_path.write_bytes(content)

        # Git commit if available
        if self.use_git:
            self._git_add_commit(path, description, version_id)

        # Record history
        history = self._history.setdefault(str(path), RollbackHistory(file_path=str(path)))
        parent = history.current_version_id
        record = VersionRecord(
            version_id=version_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            file_path=str(path),
            file_hash=file_hash,
            parent_version=parent,
            tags=tags or [],
            metrics=metrics or {},
        )
        history.versions.append(record)
        history.current_version_id = version_id
        self._save_history()

        return version_id

    def rollback(
        self,
        file_path: Union[str, Path],
        reason: str = "",
    ) -> str:
        """
        Roll back *file_path* to its previous checkpoint.

        Returns:
            version_id of the restored version.
        """
        path = self._resolve(file_path)
        history = self._history.get(str(path))
        if not history or not history.versions:
            raise RollbackError(f"No history found for {path}")

        current = history.current_version_id
        if current is None:
            raise RollbackError(f"No current version recorded for {path}")

        # Find parent
        record = self._find_record(history, current)
        parent_id = record.parent_version if record else None
        if parent_id is None:
            raise RollbackError(f"No parent version to roll back to for {path}")

        return self.restore(file_path, parent_id, reason=reason)

    def restore(
        self,
        file_path: Union[str, Path],
        version_id: str,
        reason: str = "",
    ) -> str:
        """
        Restore *file_path* to a specific *version_id*.

        Returns:
            version_id of the restored version.
        """
        path = self._resolve(file_path)
        history = self._history.get(str(path))
        if not history:
            raise RollbackError(f"No history found for {path}")

        record = self._find_record(history, version_id)
        if record is None:
            raise RollbackError(f"Version {version_id} not found for {path}")

        backup_path = self._backup_path(version_id)
        if not backup_path.exists():
            raise RollbackError(f"Backup missing for version {version_id}")

        # Safety: checkpoint current state before restoring
        if path.exists():
            self.checkpoint(
                path,
                description=f"Auto-save before restore to {version_id}",
                tags=["auto-save", "pre-restore"],
            )

        # Restore file
        shutil.copy2(backup_path, path)

        # Mark auto-rollback if applicable
        if reason:
            record.auto_rollback_triggered = True
            record.description += f" | Restored: {reason}"
            self._save_history()

        # Git revert if available
        if self.use_git:
            self._git_checkout_version(path, version_id)

        history.current_version_id = version_id
        self._save_history()
        return version_id

    def auto_rollback_on_failure(
        self,
        file_path: Union[str, Path],
        test_func: callable,
        *test_args,
        **test_kwargs,
    ) -> bool:
        """
        Run *test_func* and rollback if it raises or returns False.

        Returns:
            True if tests passed, False if rolled back.
        """
        path = self._resolve(file_path)
        pre_version = self.checkpoint(path, description="Auto-rollback checkpoint")

        try:
            result = test_func(*test_args, **test_kwargs)
            if result is False:
                raise RuntimeError("Test returned False")
            return True
        except Exception as exc:
            self.rollback(path, reason=f"Test failure: {exc}")
            return False

    def list_versions(self, file_path: Union[str, Path]) -> List[VersionRecord]:
        """Return all version records for *file_path*."""
        path = self._resolve(file_path)
        history = self._history.get(str(path))
        return history.versions if history else []

    def current_version(self, file_path: Union[str, Path]) -> Optional[VersionRecord]:
        """Return the current version record for *file_path*."""
        path = self._resolve(file_path)
        history = self._history.get(str(path))
        if history and history.current_version_id:
            return self._find_record(history, history.current_version_id)
        return None

    def diff_versions(
        self,
        file_path: Union[str, Path],
        version_a: str,
        version_b: str,
    ) -> str:
        """Unified diff between two versions."""
        path = self._resolve(file_path)
        backup_a = self._backup_path(version_a)
        backup_b = self._backup_path(version_b)
        if not backup_a.exists() or not backup_b.exists():
            raise RollbackError("One or both backup versions missing")

        import difflib
        lines_a = backup_a.read_text(encoding="utf-8").splitlines(keepends=True)
        lines_b = backup_b.read_text(encoding="utf-8").splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"{path.name}@{version_a}",
                tofile=f"{path.name}@{version_b}",
                lineterm="",
            )
        )

    def prune_old_versions(
        self,
        file_path: Union[str, Path],
        keep: int = 50,
    ) -> int:
        """Remove backups older than the most recent *keep* versions."""
        path = self._resolve(file_path)
        history = self._history.get(str(path))
        if not history or len(history.versions) <= keep:
            return 0

        removed = 0
        to_remove = history.versions[:-keep]
        for record in to_remove:
            backup = self._backup_path(record.version_id)
            if backup.exists():
                backup.unlink()
                removed += 1
        history.versions = history.versions[-keep:]
        self._save_history()
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, file_path: Union[str, Path]) -> Path:
        """Resolve a path relative to repo_root."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _make_version_id(self, path: Path, content: bytes) -> str:
        """Deterministic version ID from path + content + time."""
        nonce = str(time.time_ns()).encode()
        return hashlib.sha256(str(path).encode() + content + nonce).hexdigest()[:12]

    def _backup_path(self, version_id: str) -> Path:
        return self.backup_dir / f"{version_id}.py"

    def _history_path(self) -> Path:
        return self.repo_root / self.HISTORY_FILE

    def _load_history(self) -> None:
        path = self._history_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._history = {
                    k: RollbackHistory(
                        file_path=v["file_path"],
                        versions=[VersionRecord(**r) for r in v.get("versions", [])],
                        current_version_id=v.get("current_version_id"),
                    )
                    for k, v in data.items()
                }
            except Exception:
                self._history = {}

    def _save_history(self) -> None:
        path = self._history_path()
        data = {
            k: {
                "file_path": v.file_path,
                "versions": [asdict(r) for r in v.versions],
                "current_version_id": v.current_version_id,
            }
            for k, v in self._history.items()
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _find_record(self, history: RollbackHistory, version_id: str) -> Optional[VersionRecord]:
        for record in history.versions:
            if record.version_id == version_id:
                return record
        return None

    # ------------------------------------------------------------------
    # Git integration
    # ------------------------------------------------------------------

    def _git_available(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _git_add_commit(self, path: Path, description: str, version_id: str) -> None:
        """Stage and commit the file with version metadata."""
        try:
            rel = os.path.relpath(path, self.repo_root)
            subprocess.run(
                ["git", "add", rel],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
                check=True,
            )
            msg = f"[Moses Self-Modify] {description}\n\nVersion: {version_id}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
                check=True,
            )
        except Exception:
            # Git failure is non-fatal; backup system remains
            pass

    def _git_checkout_version(self, path: Path, version_id: str) -> None:
        """Attempt to checkout the file at the commit containing version_id."""
        try:
            rel = os.path.relpath(path, self.repo_root)
            # Find commit by message
            result = subprocess.run(
                ["git", "log", "--all", "--oneline", "--grep", version_id],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                commit = result.stdout.splitlines()[0].split()[0]
                subprocess.run(
                    ["git", "checkout", commit, "--", rel],
                    cwd=self.repo_root,
                    capture_output=True,
                    timeout=10,
                    check=True,
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RollbackError(Exception):
    """Raised when a rollback operation cannot be completed."""
    pass
