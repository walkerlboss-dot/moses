# ═══════════════════════════════════════════════════════════════════════════
#  Moses v3.0 — Task Runner (GNU Make)
# ═══════════════════════════════════════════════════════════════════════════
# Usage:
#   make install      Install production + dev dependencies
#   make lint         Run all linters (black, ruff, mypy)
#   make test         Run pytest (CPU tests)
#   make train        Launch training job
#   make eval         Run evaluation
#   make docker-build Build the Docker image
#   make docker-run   Run the training container
#   make clean        Remove build artifacts, caches, and temp files
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: help install install-dev lint format test train eval docker-build docker-run clean

# Default target: show help
help:
	@echo "Moses v3.0 — Available targets:"
	@echo "  install      Install production + dev dependencies"
	@echo "  install-dev  Install dev dependencies only"
	@echo "  lint         Run all linters (black --check, ruff, mypy)"
	@echo "  format       Auto-format code with black and ruff"
	@echo "  test         Run pytest (excludes gpu/isaac tests)"
	@echo "  test-all     Run all tests (requires GPU)"
	@echo "  train        Launch training job via Hydra"
	@echo "  eval         Run evaluation on latest checkpoint"
	@echo "  docker-build Build the Docker image"
	@echo "  docker-run   Run training inside Docker (GPU)"
	@echo "  docker-up    Start full docker-compose stack"
	@echo "  docker-down  Stop docker-compose stack"
	@echo "  clean        Remove build artifacts, caches, pyc files"

# ── Installation ──────────────────────────────────────────────────────────

install:
	@echo "→ Installing production + dev dependencies..."
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e .

install-dev:
	@echo "→ Installing dev dependencies..."
	pip install --upgrade pip
	pip install -r requirements-dev.txt

# ── Linting & Formatting ──────────────────────────────────────────────────

lint:
	@echo "→ Running black (format check)..."
	black --check --diff src tests
	@echo "→ Running ruff..."
	ruff check src tests
	@echo "→ Running mypy..."
	mypy src

format:
	@echo "→ Auto-formatting with black..."
	black src tests
	@echo "→ Auto-fixing with ruff..."
	ruff check --fix src tests

# ── Testing ───────────────────────────────────────────────────────────────

test:
	@echo "→ Running CPU tests..."
	pytest tests/ -m "not gpu and not isaac" --cov=src/moses --cov-report=term-missing -v

test-all:
	@echo "→ Running ALL tests (requires GPU)..."
	pytest tests/ --cov=src/moses --cov-report=term-missing -v

# ── Training & Evaluation ─────────────────────────────────────────────────

train:
	@echo "→ Launching training..."
	python -m moses.train config=default

eval:
	@echo "→ Running evaluation..."
	python -m moses.eval checkpoint=checkpoints/latest.pt

# ── Docker ────────────────────────────────────────────────────────────────

docker-build:
	@echo "→ Building Docker image..."
	docker build -t moses:latest .

docker-run:
	@echo "→ Running training container..."
	docker run --rm --gpus all \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/logs:/app/logs \
		moses:latest \
		python -m moses.train config=default

docker-up:
	@echo "→ Starting docker-compose stack..."
	docker compose up -d

docker-down:
	@echo "→ Stopping docker-compose stack..."
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────────

clean:
	@echo "→ Cleaning build artifacts..."
	rm -rf build/ dist/ .eggs/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "→ Clean complete."
