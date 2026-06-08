#!/usr/bin/env python3
"""
run_tests.py

Test runner for the Moses humanoid project.

Discovers all tests in the tests/ directory, runs unit + integration +
simulation tests, generates a coverage report, and exits with code 0
if all tests pass.

Target: Python 3.10+, pytest, coverage.py
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_tests")


# ---------------------------------------------------------------------------
# Test discovery helpers
# ---------------------------------------------------------------------------
TEST_CATEGORIES = ["unit", "integration", "sim"]


def discover_tests(tests_dir: Path, category: str | None = None) -> list[Path]:
    """Discover test files in the tests directory.

    Args:
        tests_dir: Root tests directory.
        category: Optional category filter (unit, integration, sim).

    Returns:
        List of test file paths.
    """
    if not tests_dir.exists():
        logger.warning("Tests directory not found: %s", tests_dir)
        return []

    test_files: list[Path] = []

    if category:
        cat_dir = tests_dir / category
        if cat_dir.exists():
            test_files.extend(sorted(cat_dir.glob("test_*.py")))
            test_files.extend(sorted(cat_dir.glob("*_test.py")))
        else:
            logger.warning("Category directory not found: %s", cat_dir)
    else:
        for cat in TEST_CATEGORIES:
            cat_dir = tests_dir / cat
            if cat_dir.exists():
                test_files.extend(sorted(cat_dir.glob("test_*.py")))
                test_files.extend(sorted(cat_dir.glob("*_test.py")))

    return test_files


def check_pytest() -> bool:
    """Check if pytest is available."""
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        return False


def check_coverage() -> bool:
    """Check if coverage.py is available."""
    try:
        import coverage  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_pytest(
    test_paths: Sequence[Path | str],
    coverage_enabled: bool,
    coverage_dir: Path,
    verbose: bool,
    junit_path: Path | None,
    markers: Sequence[str] | None,
    extra_args: Sequence[str] | None,
) -> int:
    """Run tests via pytest.

    Returns:
        pytest exit code.
    """
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if coverage_enabled:
        cmd.extend(["--cov=.", "--cov-report=term-missing", f"--cov-report=html:{coverage_dir}"])
        cmd.append("--cov-branch")

    if junit_path:
        cmd.extend([f"--junitxml={junit_path}"])

    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])

    if extra_args:
        cmd.extend(extra_args)

    cmd.extend(str(p) for p in test_paths)

    logger.info("Running: %s", " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    logger.info("Test run completed in %.2fs | Exit code: %d", elapsed, result.returncode)
    return result.returncode


def run_unittest(
    test_paths: Sequence[Path | str],
    verbose: bool,
) -> int:
    """Fallback test runner using unittest discovery.

    Returns:
        unittest exit code.
    """
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for path in test_paths:
        p = Path(path)
        if p.is_file():
            # Load from file
            spec = __import__("importlib.util").util.spec_from_file_location(p.stem, p)
            if spec and spec.loader:
                mod = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                suite.addTests(loader.loadTestsFromModule(mod))
        elif p.is_dir():
            suite.addTests(loader.discover(str(p), pattern="test_*.py"))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def generate_coverage_report(coverage_dir: Path) -> None:
    """Generate a standalone coverage report if not done by pytest."""
    if not check_coverage():
        logger.warning("coverage.py not installed; skipping coverage report")
        return

    try:
        import coverage
        cov = coverage.Coverage()
        cov.load()
        cov.html_report(directory=str(coverage_dir))
        logger.info("Coverage HTML report: %s/index.html", coverage_dir)
    except Exception as exc:
        logger.warning("Coverage report generation failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Moses humanoid test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tests-dir", type=str, default="./tests", help="Tests root directory"
    )
    parser.add_argument(
        "--category", type=str, default="",
        help="Test category: unit, integration, sim (default: all)"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--coverage-dir", type=str, default="./htmlcov", help="Coverage output directory"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--junit-xml", type=str, default="", help="Write JUnit XML report"
    )
    parser.add_argument(
        "--marker", type=str, action="append", default=None,
        help="pytest marker filter (can be used multiple times)"
    )
    parser.add_argument(
        "--no-pytest", action="store_true",
        help="Force unittest fallback even if pytest is available"
    )
    parser.add_argument(
        "--failfast", action="store_true", help="Stop on first failure"
    )
    parser.add_argument(
        "--parallel", "-n", type=int, default=0,
        help="Number of parallel workers (requires pytest-xdist)"
    )
    parser.add_argument(
        "--extra-args", type=str, default="",
        help="Extra arguments passed to pytest (space-separated)"
    )
    return parser.parse_args()


def run_tests(args: argparse.Namespace) -> int:
    """Main test runner entry point."""
    logger.info("=" * 60)
    logger.info("Moses Humanoid Test Runner")
    logger.info("=" * 60)

    tests_dir = Path(args.tests_dir)
    coverage_dir = Path(args.coverage_dir)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    # Discover tests
    category = args.category if args.category else None
    test_files = discover_tests(tests_dir, category)

    if not test_files:
        logger.error("No test files discovered in %s", tests_dir)
        return 1

    logger.info("Discovered %d test files:", len(test_files))
    for tf in test_files:
        logger.info("  %s", tf)

    # Determine runner
    use_pytest = check_pytest() and not args.no_pytest
    if use_pytest:
        logger.info("Using pytest runner")
    else:
        logger.info("Using unittest fallback runner")

    # Extra args
    extra: list[str] = []
    if args.failfast:
        extra.append("-x")
    if args.parallel > 0:
        extra.extend(["-n", str(args.parallel)])
    if args.extra_args:
        extra.extend(args.extra_args.split())

    junit_path = Path(args.junit_xml) if args.junit_xml else None

    # Run
    if use_pytest:
        exit_code = run_pytest(
            test_paths=test_files,
            coverage_enabled=args.coverage,
            coverage_dir=coverage_dir,
            verbose=args.verbose,
            junit_path=junit_path,
            markers=args.marker,
            extra_args=extra,
        )
    else:
        exit_code = run_unittest(
            test_paths=test_files,
            verbose=args.verbose,
        )
        if args.coverage:
            generate_coverage_report(coverage_dir)

    # Summary
    if exit_code == 0:
        logger.info("✅ All tests passed")
    else:
        logger.error("❌ Tests failed with exit code %d", exit_code)

    if args.coverage and coverage_dir.exists():
        logger.info("Coverage report: %s/index.html", coverage_dir.absolute())

    return exit_code


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    try:
        return run_tests(args)
    except KeyboardInterrupt:
        logger.info("Test run interrupted")
        return 130
    except Exception as exc:
        logger.exception("Test runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
