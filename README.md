# Spreadsheet Analyzer

A Python project for analyzing spreadsheet data.

## Development Setup

This project uses `uv` for dependency management and `pre-commit` for code quality checks.

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd spreadsheet-analyzer
```

2. Install dependencies:

```bash
uv sync
```

3. Install development dependencies:

```bash
uv sync --dev
```

4. Install pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

### Pre-commit Hooks

This project uses comprehensive pre-commit hooks to ensure code quality:

- **Code Formatting**: Ruff (replaces Black, isort, and many flake8 plugins)
- **Type Checking**: mypy
- **Security Scanning**: bandit and safety
- **File Checks**: YAML/JSON/TOML validation, large file detection, etc.
- **Commit Messages**: Conventional commits enforced by commitizen

#### Running Pre-commit

To run all hooks manually:

```bash
uv run pre-commit run --all-files
```

To run specific hooks:

```bash
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files
```

To update hooks to latest versions:

```bash
uv run pre-commit autoupdate
```

To skip specific hooks:

```bash
SKIP=bandit,mypy uv run pre-commit run --all-files
```

### Code Style

This project follows:

- Line length: 120 characters
- Python 3.12+ syntax
- Type hints encouraged but not required
- Docstrings for public functions/classes

### Testing

Run tests with:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov
```

### Commit Messages

This project uses conventional commits. Examples:

- `feat: add new analysis function`
- `fix: correct data parsing error`
- `docs: update README`
- `refactor: simplify data processing`
- `test: add unit tests for analyzer`
- `chore: update dependencies`

The commit message format is enforced by pre-commit hooks.

## Project Structure

```
spreadsheet-analyzer/
├── .pre-commit-config.yaml  # Pre-commit configuration
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # This file
├── main.py               # Main entry point
└── tests/               # Test files (when added)
```

## License

[Add your license here]
