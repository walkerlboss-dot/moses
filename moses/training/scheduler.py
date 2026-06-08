"""
Moses Training Scheduler
========================
Cron-like scheduling, priority queues, GPU-aware resource allocation, and
preemption for continuous training workloads.

The scheduler can be run as a long-lived service (see :meth:`run_forever`) or
invoked ad-hoc to pop the next eligible job.

Dependencies
------------
- ``schedule`` for cron-like triggers.
- ``celery`` (optional) for distributed task execution.
- ``pydantic`` for config validation.

Example
-------
>>> from moses.training.scheduler import TrainingScheduler, JobSpec
>>> sched = TrainingScheduler.from_yaml("scheduler_config.yaml")
>>> sched.submit(JobSpec(name="nightly-ppo", priority=10, gpu_count=2))
>>> sched.run_forever()
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger("moses.training.scheduler")

# Optional Celery integration
try:
    from celery import Celery  # type: ignore[import-untyped]

    _CELERY_AVAILABLE = True
except Exception:
    Celery = None  # type: ignore[misc,assignment]
    _CELERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enums & Data models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


class PreemptionPolicy(str, Enum):
    NONE = "none"
    YIELD = "yield"  # Low-priority pauses/resumes
    KILL = "kill"  # Low-priority is killed


@dataclass(order=True)
class JobSpec:
    """
    Specification for a training job.

    The ``priority`` field is inverted for the min-heap (higher number =
    higher priority). ``gpu_count`` and ``memory_gib`` are used for
    resource-aware scheduling.
    """

    # Internal heap key — do not set manually
    _heap_priority: int = field(init=False, repr=False)

    name: str
    priority: int = 0  # Higher is more urgent
    gpu_count: int = 1
    memory_gib: float = 16.0
    command: Optional[List[str]] = None
    preemption_policy: PreemptionPolicy = PreemptionPolicy.YIELD
    metadata: Dict[str, Any] = field(default_factory=dict)
    cron: Optional[str] = None  # schedule syntax, e.g. "0 2 * * *"
    max_retries: int = 3
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Invert priority so heapq (min-heap) pops highest priority first
        self._heap_priority = -self.priority


@dataclass
class JobRecord:
    """Mutable runtime state for a scheduled job."""

    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Resource tracker
# ---------------------------------------------------------------------------

class ResourcePool:
    """
    Simple GPU / RAM resource tracker.

    In a real cluster this would query Kubernetes (``kubectl``), Ray, or
    SLURM. Here we maintain a local reservation table.
    """

    def __init__(
        self,
        total_gpus: int = 0,
        total_memory_gib: float = 0.0,
    ) -> None:
        self.total_gpus = total_gpus
        self.total_memory_gib = total_memory_gib
        self._lock = threading.Lock()
        self._reserved_gpus: Dict[str, int] = {}  # job_name -> count
        self._reserved_memory: Dict[str, float] = {}  # job_name -> GiB

    def available(self) -> Tuple[int, float]:
        with self._lock:
            used_gpus = sum(self._reserved_gpus.values())
            used_mem = sum(self._reserved_memory.values())
        return self.total_gpus - used_gpus, self.total_memory_gib - used_mem

    def reserve(self, job_name: str, gpus: int, memory_gib: float) -> bool:
        with self._lock:
            used_gpus = sum(self._reserved_gpus.values())
            used_mem = sum(self._reserved_memory.values())
            if (used_gpus + gpus > self.total_gpus) or (
                used_mem + memory_gib > self.total_memory_gib
            ):
                return False
            self._reserved_gpus[job_name] = gpus
            self._reserved_memory[job_name] = memory_gib
            return True

    def release(self, job_name: str) -> None:
        with self._lock:
            self._reserved_gpus.pop(job_name, None)
            self._reserved_memory.pop(job_name, None)

    def preemptible_jobs(
        self, min_priority: int
    ) -> List[Tuple[str, JobSpec]]:
        """Return currently running jobs with priority < *min_priority*."""
        # To be wired by the scheduler
        return []


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TrainingScheduler:
    """
    Priority-queue-based training scheduler with cron triggers and
    optional Celery dispatch.

    Parameters
    ----------
    resource_pool :
        Shared resource tracker (GPUs, memory).
    celery_app :
        Optional Celery application for distributed execution.
    preemption_policy :
        Global default when a high-priority job arrives.
    state_file :
        Path to JSON file for durable queue state (optional).
    """

    def __init__(
        self,
        resource_pool: ResourcePool,
        celery_app: Optional[Any] = None,
        preemption_policy: PreemptionPolicy = PreemptionPolicy.YIELD,
        state_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self._pool = resource_pool
        self._celery = celery_app
        self._preemption_policy = preemption_policy
        self._state_file = Path(state_file) if state_file else None

        self._queue: List[Tuple[int, float, JobSpec]] = []  # heapq
        self._running: Dict[str, JobRecord] = {}
        self._history: List[JobRecord] = []
        self._lock = threading.Lock()
        self._counter = 0  # tie-breaker for heapq
        self._shutdown = threading.Event()

        if self._state_file and self._state_file.exists():
            self._load_state()

    # -- Factory --------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingScheduler":
        """Instantiate from a YAML configuration file."""
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        pool_cfg = cfg.get("resources", {})
        pool = ResourcePool(
            total_gpus=pool_cfg.get("gpus", 0),
            total_memory_gib=pool_cfg.get("memory_gib", 0.0),
        )

        sched = cls(
            resource_pool=pool,
            preemption_policy=PreemptionPolicy(
                cfg.get("preemption_policy", "yield")
            ),
            state_file=cfg.get("state_file"),
        )

        # Pre-load recurring jobs
        for job_cfg in cfg.get("recurring_jobs", []):
            sched.submit(JobSpec(**job_cfg))

        return sched

    # -- Public API ----------------------------------------------------------

    def submit(self, spec: JobSpec) -> str:
        """Enqueue a new job. Returns the job name."""
        with self._lock:
            self._counter += 1
            heapq.heappush(
                self._queue,
                (spec._heap_priority, self._counter, spec),
            )
        logger.info("Submitted job %s (priority=%d)", spec.name, spec.priority)
        self._persist_state()
        return spec.name

    def cancel(self, job_name: str) -> bool:
        """Remove a pending job by name."""
        with self._lock:
            new_queue = []
            found = False
            for prio, cnt, spec in self._queue:
                if spec.name == job_name:
                    found = True
                    continue
                new_queue.append((prio, cnt, spec))
            if found:
                self._queue = new_queue
                heapq.heapify(self._queue)
                logger.info("Cancelled pending job %s", job_name)
                self._persist_state()
                return True
        return False

    def pop_next(self) -> Optional[JobSpec]:
        """
        Pop the highest-priority job that fits current resources.
        If a higher-priority job cannot fit, attempts preemption of
        lower-priority running jobs according to policy.
        """
        with self._lock:
            while self._queue:
                prio, cnt, spec = self._queue[0]
                if self._pool.reserve(spec.name, spec.gpu_count, spec.memory_gib):
                    heapq.heappop(self._queue)
                    return spec

                # Try preemption
                if self._preemption_policy != PreemptionPolicy.NONE:
                    preempted = self._attempt_preemption(spec)
                    if preempted:
                        heapq.heappop(self._queue)
                        return spec

                # Cannot run now — leave in queue
                break
        return None

    def start_job(self, spec: JobSpec, worker_id: Optional[str] = None) -> JobRecord:
        """Mark a job as running and dispatch via Celery or local thread."""
        record = JobRecord(
            spec=spec,
            status=JobStatus.RUNNING,
            started_at=time.time(),
            worker_id=worker_id,
        )
        with self._lock:
            self._running[spec.name] = record
        logger.info("Started job %s on worker %s", spec.name, worker_id)

        if self._celery and spec.command:
            # Dispatch to Celery — assumes a task named "moses.training.run_command"
            self._celery.send_task(
                "moses.training.run_command",
                args=[spec.command],
                task_id=spec.name,
            )
        return record

    def finish_job(
        self,
        job_name: str,
        status: JobStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Transition a running job to terminal state."""
        with self._lock:
            record = self._running.pop(job_name, None)
            if record is None:
                logger.warning("finish_job called for unknown job %s", job_name)
                return
            record.status = status
            record.finished_at = time.time()
            record.error_message = error_message
            self._history.append(record)
            self._pool.release(job_name)
        logger.info("Job %s finished with status %s", job_name, status.value)
        self._persist_state()

    def list_pending(self) -> List[JobSpec]:
        with self._lock:
            return [spec for _, _, spec in sorted(self._queue)]

    def list_running(self) -> List[JobRecord]:
        with self._lock:
            return list(self._running.values())

    def run_forever(self, poll_interval: float = 5.0) -> None:
        """
        Blocking loop that continuously schedules jobs.
        Intended to run in a dedicated thread or process.
        """
        logger.info("Scheduler loop started (poll=%.1fs)", poll_interval)
        while not self._shutdown.is_set():
            spec = self.pop_next()
            if spec:
                self.start_job(spec, worker_id=f"worker-{threading.current_thread().ident}")
            time.sleep(poll_interval)

    def shutdown(self) -> None:
        """Signal the scheduler loop to exit."""
        self._shutdown.set()
        logger.info("Scheduler shutdown requested")

    # -- Preemption ----------------------------------------------------------

    def _attempt_preemption(self, incoming: JobSpec) -> bool:
        """
        Attempt to free resources by preempting lower-priority running jobs.
        Returns True if enough resources were freed.
        """
        victims = [
            (name, rec)
            for name, rec in self._running.items()
            if rec.spec.priority < incoming.priority
        ]
        victims.sort(key=lambda x: x[1].spec.priority)

        freed_gpus = 0
        freed_mem = 0.0
        to_preempt: List[str] = []

        for name, rec in victims:
            freed_gpus += rec.spec.gpu_count
            freed_mem += rec.spec.memory_gib
            to_preempt.append(name)
            if (
                freed_gpus >= incoming.gpu_count
                and freed_mem >= incoming.memory_gib
            ):
                break

        if freed_gpus < incoming.gpu_count or freed_mem < incoming.memory_gib:
            return False

        for name in to_preempt:
            rec = self._running[name]
            if rec.spec.preemption_policy == PreemptionPolicy.KILL:
                logger.warning("Preempting (kill) job %s for %s", name, incoming.name)
                self.finish_job(name, JobStatus.PREEMPTED, error_message="killed by preemption")
            else:
                logger.warning("Preempting (yield) job %s for %s", name, incoming.name)
                # Yield: stop and re-enqueue
                self._pool.release(name)
                rec.status = JobStatus.PENDING
                rec.started_at = None
                rec.worker_id = None
                with self._lock:
                    self._counter += 1
                    heapq.heappush(
                        self._queue,
                        (rec.spec._heap_priority, self._counter, rec.spec),
                    )
                    del self._running[name]
        return True

    # -- Persistence ---------------------------------------------------------

    def _persist_state(self) -> None:
        if self._state_file is None:
            return
        with self._lock:
            payload = {
                "queue": [
                    {"priority": p, "counter": c, "spec": asdict(s)}
                    for p, c, s in self._queue
                ],
                "running": {
                    name: {
                        "spec": asdict(rec.spec),
                        "status": rec.status.value,
                        "attempts": rec.attempts,
                        "started_at": rec.started_at,
                        "worker_id": rec.worker_id,
                        "error_message": rec.error_message,
                    }
                    for name, rec in self._running.items()
                },
            }
        tmp = self._state_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(self._state_file)

    def _load_state(self) -> None:
        if self._state_file is None or not self._state_file.exists():
            return
        with open(self._state_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        with self._lock:
            for item in payload.get("queue", []):
                spec = JobSpec(**item["spec"])
                self._counter = max(self._counter, item["counter"])
                heapq.heappush(
                    self._queue,
                    (spec._heap_priority, item["counter"], spec),
                )
            for name, rec_data in payload.get("running", {}).items():
                spec = JobSpec(**rec_data["spec"])
                record = JobRecord(
                    spec=spec,
                    status=JobStatus(rec_data["status"]),
                    attempts=rec_data["attempts"],
                    started_at=rec_data.get("started_at"),
                    worker_id=rec_data.get("worker_id"),
                    error_message=rec_data.get("error_message"),
                )
                self._running[name] = record
        logger.info("Restored scheduler state from %s", self._state_file)


# ---------------------------------------------------------------------------
# Celery task stub (to be imported by Celery worker)
# ---------------------------------------------------------------------------

if _CELERY_AVAILABLE:

    def make_celery_app(broker_url: str, backend_url: Optional[str] = None) -> Celery:
        app = Celery("moses_training", broker=broker_url, backend=backend_url)
        app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
        )

        @app.task(bind=True, max_retries=3)
        def run_command(self, command: List[str]) -> Dict[str, Any]:
            """Celery task that runs a shell command and reports results."""
            import subprocess

            logger.info("Celery task executing: %s", " ".join(command))
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.CalledProcessError as exc:
                logger.error("Command failed: %s", exc)
                raise self.retry(exc=exc, countdown=60)

        return app
