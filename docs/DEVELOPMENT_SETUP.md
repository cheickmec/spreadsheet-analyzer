# Development Setup Guide

This guide covers the development environment setup for the Spreadsheet Analyzer project.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd spreadsheet-analyzer
```

### 2. Install Dependencies

```bash
# Install project dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### 3. Install Pre-commit Hooks

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Pre-commit Configuration

This project uses comprehensive pre-commit hooks to ensure code quality:

- **Code Formatting**: Ruff (replaces Black, isort, and many flake8 plugins)
- **Type Checking**: mypy
- **Security Scanning**: bandit and safety
- **File Checks**: YAML/JSON/TOML validation, large file detection
- **Commit Messages**: Conventional commits enforced by commitizen

### Running Pre-commit

```bash
# Run all hooks manually
uv run pre-commit run --all-files

# Run specific hooks
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files

# Update hooks to latest versions
uv run pre-commit autoupdate

# Skip specific hooks
SKIP=bandit,mypy uv run pre-commit run --all-files
```

## Code Style

- Line length: 120 characters
- Python 3.12+ syntax
- Type hints encouraged but not required
- Docstrings for public functions/classes
- Follow the patterns in CLAUDE.md

## Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/spreadsheet_analyzer

# Run specific test file
uv run pytest tests/test_specific.py

# Run tests matching pattern
uv run pytest -k "test_pattern"
```

## Commit Message Convention

This project uses conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks
- `perf:` Performance improvements
- `style:` Code style changes

Examples:

- `feat: add Excel formula parser`
- `fix: correct circular reference detection`
- `docs: update API documentation`
- `test: add unit tests for agent system`

The commit message format is enforced by pre-commit hooks.
