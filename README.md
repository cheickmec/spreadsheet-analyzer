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
├── .pre-commit-config.yaml     # Pre-commit configuration
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── main.py                     # Main entry point
├── docs/                       # 📚 Comprehensive Documentation Hub
│   ├── README.md              # Documentation overview and navigation
│   ├── complete-guide/        # 📖 Complete Implementation Guide
│   │   └── building-intelligent-spreadsheet-analyzers.md  # 318KB comprehensive guide
│   └── research/              # 🔬 Detailed Research Documentation
│       ├── 1-llm-agentic-fundamentals/
│       ├── 2-engineering-techniques/
│       ├── 3-workflow-orchestration/
│       ├── 4-implementation-optimization/
│       └── 5-broader-considerations/
├── comprehensive-system-design.md  # System design document
└── tests/                      # Test files (when added)
```

## 📚 Documentation

This project includes extensive documentation for building AI-powered spreadsheet analysis systems:

### 🎯 **Complete Implementation Guide**

**[Building Intelligent Spreadsheet Analyzers: A Complete Guide to AI Agents, RAG Systems, and Multi-Agent Orchestration](./docs/complete-guide/building-intelligent-spreadsheet-analyzers.md)**

A comprehensive 318KB guide covering:

- 🏗️ **AI Agent Architectures** with enhanced Mermaid diagrams
- 🔧 **Practical Implementation** strategies and code examples
- 📊 **Multi-Agent Orchestration** frameworks and patterns
- 🚀 **Production Deployment** considerations and optimization
- 🎯 **Domain-Specific Guidance** for spreadsheet/Excel analysis

### 🔬 **Research Documentation**

**[Detailed Research Hub](./docs/research/)**

In-depth analysis of:

- Latest LLM agentic systems research (2023-2024)
- Framework comparisons (LangGraph, CrewAI, AutoGen)
- Implementation patterns and best practices
- Performance optimization and security considerations

See **[docs/README.md](./docs/README.md)** for complete navigation guide.

## License

[Add your license here]
