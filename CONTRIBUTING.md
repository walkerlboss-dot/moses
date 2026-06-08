# Contributing to Moses

> **Thank you for your interest in Moses!** This is the autonomous humanoid robotics builder for Boss Industries. We welcome contributions that push the frontier.

---

## Code of Conduct

This project adheres to a standard of professionalism and respect. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 12.3+ (for GPU training)
- Docker (for containerized development)
- Git LFS (for large model files)

### Installation

```bash
git clone https://github.com/walkerlboss-dot/moses.git
cd moses

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Code Style

We enforce code quality automatically:

| Tool | Purpose | Command |
|------|---------|---------|
| **black** | Code formatting | `make lint` |
| **ruff** | Fast linting | `make lint` |
| **mypy** | Type checking | `make type-check` |
| **pytest** | Testing | `make test` |

**Pre-commit hooks** run these checks automatically on every commit.

---

## Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests (requires Isaac Sim)
pytest tests/integration -v

# Simulation tests (slow)
pytest tests/integration -m slow -v
```

### Writing Tests

- All new code must include tests
- Use `pytest` fixtures for setup/teardown
- Mock external dependencies (Isaac Sim, W&B, etc.)
- Name tests descriptively: `test_[function]_[scenario]_[expected_result]`

---

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature`
3. **Make changes** with tests
4. **Run checks**: `make lint && make test`
5. **Commit** with descriptive messages
6. **Push** and open a PR
7. **Fill out the PR template** completely

### PR Requirements

- [ ] Code follows style guide (`make lint` passes)
- [ ] Tests pass (`make test` passes)
- [ ] Type hints included
- [ ] Docstrings for public APIs
- [ ] CHANGELOG.md updated
- [ ] README updated if user-facing changes

---

## Commit Message Format

```
type(scope): short description

Longer explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `chore`: Maintenance

---

## Architecture Decisions

Major architectural decisions are documented in `docs/ADRs/`. If your PR changes architecture, add or update an ADR.

---

## Questions?

- Open a [Discussion](https://github.com/walkerlboss-dot/moses/discussions)
- Or reach out to Alex Walk (alex@boss.industries)

---

*Built with 💪 by the Boss Industries team.*
