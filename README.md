# Spreadsheet Analyzer

**AI-Powered Excel Analysis System**\
*Proprietary Software - Yiriden LLC*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

## 🎯 Overview

Spreadsheet Analyzer is an intelligent system that automatically analyzes Excel files to reveal hidden structures, relationships, and potential issues. By combining deterministic parsing with AI-powered insights, it transforms complex spreadsheets from opaque data containers into transparent, well-documented systems.

### Key Features

- 📌 **Deep Structural Analysis**: Maps every element from cells to pivot tables
- 🏗️ **AI-Powered Intelligence**: Multi-agent system for semantic understanding
- ✅ **Validation-First**: Verifies all findings through actual calculations
- 🚀 **High Performance**: Analyzes files in seconds, not minutes
- 🔐 **Enterprise Security**: Sandboxed execution with comprehensive audit trails
- 📊 **Comprehensive Reporting**: Detailed insights in multiple formats

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- 8GB+ RAM recommended
- macOS, Linux, or Windows with WSL2

### Installation

```bash
# Clone the repository
git clone https://github.com/yiriden/spreadsheet-analyzer.git
cd spreadsheet-analyzer

# Install dependencies using uv
uv sync

# Install CLI tool
uv pip install -e .

# Run a simple analysis
spreadsheet-analyzer analyze test-files/data-analysis/advanced_excel_formulas.xlsx
```

### Basic Usage

```bash
# Install the package with CLI dependencies
uv sync
uv pip install -e .

# Analyze a single Excel file
spreadsheet-analyzer analyze financial-model.xlsx

# Verbose output with progress tracking
spreadsheet-analyzer -vv analyze data.xlsx

# Fast mode with JSON output
spreadsheet-analyzer analyze data.xlsx --mode fast --format json

# Save results to file
spreadsheet-analyzer analyze sensitive.xlsx -o results.yaml

# Skip formula analysis for faster results
spreadsheet-analyzer analyze large-file.xlsx --no-formulas

# Help and available options
spreadsheet-analyzer --help
spreadsheet-analyzer analyze --help
```

## 🔍 What It Does

### 1. **Structural Analysis** (Deterministic)

- Sheet enumeration and classification
- Formula parsing and dependency mapping
- Named range detection and validation
- Chart and visual element extraction
- External reference identification

### 2. **Intelligent Analysis** (AI-Powered)

- Pattern recognition across sheets
- Semantic understanding of business logic
- Anomaly and error detection
- Optimization recommendations
- Cross-sheet relationship mapping

### 3. **Validation & Verification**

- Formula calculation verification
- Data integrity checks
- Circular reference detection
- Security vulnerability scanning
- Performance bottleneck identification

## 🏗️ Architecture

The system implements a 5-stage deterministic analysis pipeline with hybrid FP/OOP design:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Stage 0: File   │────▶│ Stage 1:        │────▶│ Stage 2:        │
│ Integrity (FP)  │     │ Security (FP)   │     │ Structure(Hybrid)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Stage 3: Formula│     │ Stage 4: Content│     │ Pipeline         │
│ Analysis (OOP)  │────▶│ Intelligence(FP)│────▶│ Orchestrator    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Key Components:**

- **Stage 0-1**: Pure functional programming for stateless validation
- **Stage 2**: Hybrid approach for complex structural mapping
- **Stage 3**: Object-oriented for graph-based formula analysis
- **Stage 4**: Functional programming for content intelligence
- **Pipeline**: Progress tracking with observer pattern

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Check code quality
uv run pre-commit run --all-files
```

### Project Structure

```
spreadsheet-analyzer/
│├──── src/spreadsheet_analyzer/     # Main application code
│   │├──── pipeline/                 # 5-stage analysis pipeline
│   │   │├──── stages/               # Individual stage implementations
│   │   │├──── types.py              # Immutable data structures
│   │   │└── pipeline.py           # Main orchestrator
│   │├──── cli/                      # Command-line interface
│   │   │├──── commands/             # CLI commands (analyze, batch, etc.)
│   │   │└── __init__.py           # CLI entry point
│   │├──── services/                 # Business logic layer
│   │   │└── analysis_service.py   # Core analysis service
│   │└── graph_db/                 # Graph database integration
│├──── test-files/                   # Example Excel files
│├──── tests/                        # Test suite
│└── docs/                         # Documentation
    │├──── design/                   # System design documents
    │├──── progress/                 # Implementation tracking
    │└── research/                 # AI/LLM research
```

## 📚 Documentation

- **[Implementation Status](docs/progress/implementation-status.md)**: 📊 Current progress and roadmap tracking
- **[Pipeline Design](docs/design/deterministic-analysis-pipeline.md)**: 5-stage pipeline architecture
- **[System Design](docs/design/comprehensive-system-design.md)**: Complete technical specification
- **[CLI Architecture](docs/design/cli-architecture-design.md)**: Terminal interface design
- **[Contributing](CONTRIBUTING.md)**: Development practices and testing philosophy

## ⚡ Performance

Designed for enterprise-scale analysis:

| Operation                      | Target Performance |
| ------------------------------ | ------------------ |
| File Upload (< 10MB)           | < 2 seconds        |
| Basic Analysis (< 10 sheets)   | < 5 seconds        |
| Deep AI Analysis (< 50 sheets) | < 30 seconds       |
| Memory Usage                   | < 512MB per agent  |

## 🔐 Security

- **Sandboxed Execution**: All analysis runs in isolated Jupyter kernels
- **No Macro Execution**: VBA/macros analyzed statically only
- **File Validation**: Comprehensive format and content validation
- **Audit Logging**: Complete trail of all operations
- **Data Privacy**: No data persistence without explicit consent

## 🤝 Contributing

This is proprietary software owned by Yiriden LLC. External contributions require:

1. Signed Contributor License Agreement (CLA)
1. Adherence to coding standards in CLAUDE.md
1. Passing all tests and security checks

## 📜 License

Proprietary Software - Yiriden LLC. All rights reserved.

**Contact**:\
Cheick Berthe\
Email: cab25004@vt.edu\
Organization: Yiriden LLC

______________________________________________________________________

*Built with Python, powered by AI, designed for analysts.*
