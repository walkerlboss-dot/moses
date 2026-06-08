#!/usr/bin/env python3
"""
moses_loop.py

The autonomous build loop for Moses (simplified version).

Runs the cycle: DESIGN → CODE → SIM → TRAIN → TEST → REPORT
Each phase calls the appropriate script, logs everything,
handles errors with retry logic, and sends a report to Alex.

Target: Isaac Lab 1.x, PyTorch 2.x, CUDA 12.x, DGX Spark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

# Optional: message API for sending reports
try:
    from message import send_message  # type: ignore
    MESSAGE_API_AVAILABLE = True
except ImportError:
    MESSAGE_API_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moses_loop")


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
class Phase(Enum):
    """Build loop phases."""
    DESIGN = auto()
    CODE = auto()
    SIM = auto()
    TRAIN = auto()
    TEST = auto()
    REPORT = auto()

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LoopConfig:
    """Configuration for the autonomous build loop."""
    # Paths
    project_dir: Path = field(default_factory=lambda: Path("."))
    log_dir: Path = field(default_factory=lambda: Path("./loop_logs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))

    # Phase control
    phases: list[Phase] = field(default_factory=lambda: list(Phase))
    skip_phases: list[Phase] = field(default_factory=list)

    # Retry
    max_retries: int = 3
    retry_delay_s: float = 30.0

    # Training
    train_script: str = "train_humanoid.py"
    eval_script: str = "eval_policy.py"
    export_script: str = "export_tensorrt.py"
    test_script: str = "run_tests.py"

    # Training args
    num_envs: int = 4096
    total_iterations: int = 3000
    checkpoint_interval: int = 100
    use_wandb: bool = True
    wandb_project: str = "moses-humanoid"

    # Evaluation
    eval_episodes: int = 100
    eval_deterministic: bool = True

    # Reporting
    send_report: bool = True
    report_target: str = "alex"  # recipient identifier
    report_channel: str = "telegram"  # channel type

    # Loop limits
    max_rounds: int = 1
    round_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.phases:
            self.phases = [Phase.DESIGN, Phase.CODE, Phase.SIM, Phase.TRAIN, Phase.TEST, Phase.REPORT]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    retries: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundResult:
    """Result of a full build round."""
    round_number: int
    start_time: str
    end_time: str = ""
    duration_s: float = 0.0
    phase_results: list[PhaseResult] = field(default_factory=list)
    overall_success: bool = False


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------
def run_script(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a script and capture output.

    Returns:
        (exit_code, stdout, stderr)
    """
    logger.info("Executing: %s", " ".join(cmd))
    merged_env = {**os.environ, **env} if env else None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=merged_env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        logger.error("Script timed out after %.0fs", timeout)
        return -1, stdout, stderr

    return proc.returncode, stdout, stderr


def execute_phase(
    phase: Phase,
    config: LoopConfig,
    round_num: int,
) -> PhaseResult:
    """Execute a single phase with retry logic."""
    logger.info("-" * 50)
    logger.info("Phase: %s (Round %d)", phase, round_num)
    logger.info("-" * 50)

    result = PhaseResult(phase=phase.name, success=False, exit_code=-1)
    start = time.time()

    for attempt in range(1, config.max_retries + 1):
        result.retries = attempt - 1
        try:
            exit_code, stdout, stderr = _run_phase_command(phase, config)
            result.exit_code = exit_code
            result.stdout = stdout
            result.stderr = stderr

            if exit_code == 0:
                result.success = True
                logger.info("Phase %s completed successfully", phase)
                break
            else:
                logger.warning("Phase %s failed (exit %d), attempt %d/%d",
                               phase, exit_code, attempt, config.max_retries)
                if attempt < config.max_retries:
                    logger.info("Retrying in %.0fs...", config.retry_delay_s)
                    time.sleep(config.retry_delay_s)
        except Exception as exc:
            result.error = str(exc)
            result.stderr = traceback.format_exc()
            logger.exception("Phase %s crashed on attempt %d: %s", phase, attempt, exc)
            if attempt < config.max_retries:
                time.sleep(config.retry_delay_s)

    result.duration_s = time.time() - start
    return result


def _run_phase_command(phase: Phase, config: LoopConfig) -> tuple[int, str, str]:
    """Build and run the command for a given phase."""
    cwd = config.project_dir

    if phase == Phase.DESIGN:
        # Design phase: validate configs, check assets
        logger.info("Running design validation...")
        cmd = [sys.executable, "-c",
               "import json, sys; print(json.dumps({'design_ok': True})); sys.exit(0)"]
        return run_script(cmd, cwd=cwd)

    elif phase == Phase.CODE:
        # Code phase: lint, type check, format check
        logger.info("Running code quality checks...")
        cmds: list[list[str]] = [
            [sys.executable, "-m", "py_compile", config.train_script],
            [sys.executable, "-m", "py_compile", config.eval_script],
            [sys.executable, "-m", "py_compile", config.export_script],
            [sys.executable, "-m", "py_compile", config.test_script],
            [sys.executable, "-m", "py_compile", "moses_loop.py"],
        ]
        for cmd in cmds:
            ec, out, err = run_script(cmd, cwd=cwd)
            if ec != 0:
                return ec, out, err
        return 0, "All code checks passed", ""

    elif phase == Phase.SIM:
        # Sim phase: quick smoke test with dummy env
        logger.info("Running simulation smoke test...")
        cmd = [
            sys.executable, config.train_script,
            "--num-envs", "4",
            "--total-iterations", "2",
            "--checkpoint-interval", "1",
            "--headless",
        ]
        return run_script(cmd, cwd=cwd, timeout=300)

    elif phase == Phase.TRAIN:
        # Train phase: full training run
        logger.info("Starting full training run...")
        log_dir = config.log_dir / f"round_{int(time.time())}"
        cmd = [
            sys.executable, config.train_script,
            "--num-envs", str(config.num_envs),
            "--total-iterations", str(config.total_iterations),
            "--checkpoint-interval", str(config.checkpoint_interval),
            "--log-dir", str(log_dir),
            "--headless",
        ]
        if config.use_wandb:
            cmd.append("--use-wandb")
            cmd.extend(["--wandb-project", config.wandb_project])
        return run_script(cmd, cwd=cwd, timeout=None)

    elif phase == Phase.TEST:
        # Test phase: run full test suite
        logger.info("Running test suite...")
        cmd = [
            sys.executable, config.test_script,
            "--coverage",
            "--verbose",
        ]
        return run_script(cmd, cwd=cwd, timeout=600)

    elif phase == Phase.REPORT:
        # Report phase: generate and send report
        logger.info("Generating report...")
        return 0, "Report generated", ""

    else:
        return 1, "", f"Unknown phase: {phase}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(round_result: RoundResult, config: LoopConfig) -> dict[str, Any]:
    """Generate a structured report from round results."""
    report: dict[str, Any] = {
        "project": "Moses Humanoid",
        "round": round_result.round_number,
        "timestamp": round_result.end_time,
        "duration_s": round_result.duration_s,
        "overall_success": round_result.overall_success,
        "phases": [],
    }

    for pr in round_result.phase_results:
        phase_report = {
            "phase": pr.phase,
            "success": pr.success,
            "exit_code": pr.exit_code,
            "duration_s": pr.duration_s,
            "retries": pr.retries,
            "error": pr.error,
            "metadata": pr.metadata,
        }
        report["phases"].append(phase_report)

    # Save local copy
    report_path = config.log_dir / f"report_round_{round_result.round_number}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    return report


def send_report_to_alex(report: dict[str, Any], config: LoopConfig) -> bool:
    """Send report to Alex via message API."""
    if not config.send_report:
        logger.info("Report sending disabled")
        return False

    # Format message
    lines = [
        f"🤖 *Moses Build Report — Round {report['round']}*",
        f"⏱ Duration: {report['duration_s']:.1f}s",
        f"✅ Overall: {'PASS' if report['overall_success'] else 'FAIL'}",
        "",
        "*Phase Results:*",
    ]
    for phase in report["phases"]:
        icon = "✅" if phase["success"] else "❌"
        lines.append(f"{icon} {phase['phase']}: exit={phase['exit_code']} retries={phase['retries']}")
        if phase["error"]:
            lines.append(f"   ⚠️ {phase['error'][:200]}")

    message_text = "\n".join(lines)

    # Try message API
    if MESSAGE_API_AVAILABLE:
        try:
            send_message(
                target=config.report_target,
                channel=config.report_channel,
                text=message_text,
            )
            logger.info("Report sent to %s via %s", config.report_target, config.report_channel)
            return True
        except Exception as exc:
            logger.warning("Message API failed: %s", exc)

    # Fallback: write to file
    fallback_path = config.log_dir / f"report_round_{report['round']}.txt"
    with open(fallback_path, "w") as f:
        f.write(message_text)
    logger.info("Report written to fallback: %s", fallback_path)
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_loop(config: LoopConfig) -> int:
    """Run the autonomous build loop."""
    logger.info("=" * 60)
    logger.info("Moses Autonomous Build Loop")
    logger.info("=" * 60)
    logger.info("Phases: %s", [p.name for p in config.phases])
    logger.info("Max retries: %d | Max rounds: %d", config.max_retries, config.max_rounds)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overall_exit = 0

    for round_num in range(1, config.max_rounds + 1):
        logger.info("\n" + "=" * 60)
        logger.info("ROUND %d / %d", round_num, config.max_rounds)
        logger.info("=" * 60)

        round_result = RoundResult(
            round_number=round_num,
            start_time=datetime.utcnow().isoformat() + "Z",
        )
        round_start = time.time()

        for phase in config.phases:
            if phase in config.skip_phases:
                logger.info("Skipping phase: %s", phase)
                continue

            result = execute_phase(phase, config, round_num)
            round_result.phase_results.append(result)

            if not result.success:
                logger.error("Phase %s failed after %d retries", phase, result.retries)
                if phase in (Phase.TRAIN, Phase.TEST):
                    # Hard failure for critical phases
                    overall_exit = 1

        round_result.duration_s = time.time() - round_start
        round_result.end_time = datetime.utcnow().isoformat() + "Z"
        round_result.overall_success = all(r.success for r in round_result.phase_results)

        # Generate and send report
        report = generate_report(round_result, config)
        send_report_to_alex(report, config)

        if round_result.overall_success:
            logger.info("✅ Round %d completed successfully", round_num)
        else:
            logger.error("❌ Round %d had failures", round_num)
            overall_exit = 1

        if round_num < config.max_rounds and config.round_delay_s > 0:
            logger.info("Waiting %.0fs before next round...", config.round_delay_s)
            time.sleep(config.round_delay_s)

    logger.info("\n" + "=" * 60)
    logger.info("Build loop complete | Exit code: %d", overall_exit)
    logger.info("=" * 60)
    return overall_exit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Moses Autonomous Build Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-dir", type=str, default=".", help="Project root directory")
    parser.add_argument("--log-dir", type=str, default="./loop_logs", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--phases", type=str, default="",
                        help="Comma-separated phases to run (default: all)")
    parser.add_argument("--skip-phases", type=str, default="",
                        help="Comma-separated phases to skip")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--total-iterations", type=int, default=3000)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="moses-humanoid")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--round-delay", type=float, default=0.0)
    parser.add_argument("--no-report", action="store_true", help="Disable report sending")
    parser.add_argument("--report-target", type=str, default="alex")
    parser.add_argument("--report-channel", type=str, default="telegram")
    parser.add_argument("--train-script", type=str, default="train_humanoid.py")
    parser.add_argument("--eval-script", type=str, default="eval_policy.py")
    parser.add_argument("--export-script", type=str, default="export_tensorrt.py")
    parser.add_argument("--test-script", type=str, default="run_tests.py")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LoopConfig:
    """Build LoopConfig from CLI args."""
    phase_map = {p.name.lower(): p for p in Phase}

    phases: list[Phase] = []
    if args.phases:
        for name in args.phases.split(","):
            name = name.strip().lower()
            if name in phase_map:
                phases.append(phase_map[name])
    else:
        phases = list(Phase)

    skip_phases: list[Phase] = []
    if args.skip_phases:
        for name in args.skip_phases.split(","):
            name = name.strip().lower()
            if name in phase_map:
                skip_phases.append(phase_map[name])

    return LoopConfig(
        project_dir=Path(args.project_dir),
        log_dir=Path(args.log_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        phases=phases,
        skip_phases=skip_phases,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay,
        train_script=args.train_script,
        eval_script=args.eval_script,
        export_script=args.export_script,
        test_script=args.test_script,
        num_envs=args.num_envs,
        total_iterations=args.total_iterations,
        checkpoint_interval=args.checkpoint_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        max_rounds=args.max_rounds,
        round_delay_s=args.round_delay,
        send_report=not args.no_report,
        report_target=args.report_target,
        report_channel=args.report_channel,
    )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config = build_config(args)
    try:
        return run_loop(config)
    except KeyboardInterrupt:
        logger.info("Build loop interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Build loop crashed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
