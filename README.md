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
- ÄŸÅ¸Â§Â  **AI-Powered Intelligence**: Multi-agent system for semantic understanding
- âœ… **Validation-First**: Verifies all findings through actual calculations
- ğŸš€ **High Performance**: Analyzes files in seconds, not minutes
- ÄŸÅ¸â€�â€™ **Enterprise Security**: Sandboxed execution with comprehensive audit trails
- ğŸ“Š **Comprehensive Reporting**: Detailed insights in multiple formats

## ğŸš€ Quick Start

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

# Run a simple analysis
uv run python scripts/analyze_excel.py test-files/data-analysis/advanced_excel_formulas.xlsx
```

### Basic Usage

```bash
# Analyze a single Excel file
uv run python scripts/analyze_excel.py financial-model.xlsx

# Quick analysis with fast mode
uv run python scripts/analyze_excel.py data.xlsx --mode fast

# Strict security analysis
uv run python scripts/analyze_excel.py sensitive.xlsx --mode strict --detailed

# Batch analyze directory
uv run python scripts/batch_analyze.py /path/to/excel/files --recursive

# Run comprehensive test suite
uv run python scripts/run_test_suite.py
```

## ğŸ“‹ What It Does

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

## ğŸ�—ï¸� Architecture

The system implements a 5-stage deterministic analysis pipeline with hybrid FP/OOP design:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ Stage 0: File   â”‚â”€â”€â”€â”€â–¶â”‚ Stage 1:        â”‚â”€â”€â”€â”€â–¶â”‚ Stage 2:        â”‚
â”‚ Integrity (FP)  â”‚     â”‚ Security (FP)   â”‚     â”‚ Structure(Hybrid)â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚                       â”‚                        â”‚
         â–¼                       â–¼                        â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ Stage 3: Formulaâ”‚     â”‚ Stage 4: Contentâ”‚     â”‚ Pipeline         â”‚
â”‚ Analysis (OOP)  â”‚â”€â”€â”€â”€â–¶â”‚ Intelligence(FP)â”‚â”€â”€â”€â”€â–¶â”‚ Orchestrator    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Key Components:**

- **Stage 0-1**: Pure functional programming for stateless validation
- **Stage 2**: Hybrid approach for complex structural mapping
- **Stage 3**: Object-oriented for graph-based formula analysis
- **Stage 4**: Functional programming for content intelligence
- **Pipeline**: Progress tracking with observer pattern

## ğŸ”§ Development

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
Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ src/spreadsheet_analyzer/     # Main application code
Ã¢â€�   Ã¢â€�â€�Ã¢â€�â‚¬Ã¢â€�â‚¬ pipeline/                 # 5-stage analysis pipeline
Ã¢â€�       Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ stages/               # Individual stage implementations
Ã¢â€�       Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ types.py              # Immutable data structures
Ã¢â€�       Ã¢â€�â€�Ã¢â€�â‚¬Ã¢â€�â‚¬ pipeline.py           # Main orchestrator
Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ scripts/                      # Analysis utilities
Ã¢â€�   Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ analyze_excel.py          # Single file analyzer
Ã¢â€�   Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ batch_analyze.py          # Batch processing
Ã¢â€�   Ã¢â€�â€�Ã¢â€�â‚¬Ã¢â€�â‚¬ run_test_suite.py         # Comprehensive testing
Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ test-files/                   # Example Excel files
Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ tests/                        # Test suite
Ã¢â€�â€�Ã¢â€�â‚¬Ã¢â€�â‚¬ docs/                         # Documentation
    Ã¢â€�Å“Ã¢â€�â‚¬Ã¢â€�â‚¬ design/                   # System design documents
    Ã¢â€�â€�Ã¢â€�â‚¬Ã¢â€�â‚¬ research/                 # AI/LLM research
```

## ğŸ“š Documentation

- **[Pipeline Design](docs/design/deterministic-analysis-pipeline.md)**: 5-stage pipeline architecture
- **[System Design](docs/design/comprehensive-system-design.md)**: Complete technical specification
- **[Script Usage](scripts/README.md)**: Guide to analysis utilities
- **[Contributing](CONTRIBUTING.md)**: Development practices and testing philosophy

## âš¡ Performance

Designed for enterprise-scale analysis:

| Operation                      | Target Performance |
| ------------------------------ | ------------------ |
| File Upload (< 10MB)           | < 2 seconds        |
| Basic Analysis (< 10 sheets)   | < 5 seconds        |
| Deep AI Analysis (< 50 sheets) | < 30 seconds       |
| Memory Usage                   | < 512MB per agent  |

## ğŸ”’ Security

- **Sandboxed Execution**: All analysis runs in isolated Jupyter kernels
- **No Macro Execution**: VBA/macros analyzed statically only
- **File Validation**: Comprehensive format and content validation
- **Audit Logging**: Complete trail of all operations
- **Data Privacy**: No data persistence without explicit consent

## ğŸ¤� Contributing

This is proprietary software owned by Yiriden LLC. External contributions require:

1. Signed Contributor License Agreement (CLA)
1. Adherence to coding standards in CLAUDE.md
1. Passing all tests and security checks

## ğŸ“„ License

Proprietary Software - Yiriden LLC. All rights reserved.

**Contact**:\
Cheick Berthe\
Email: cab25004@vt.edu\
Organization: Yiriden LLC

______________________________________________________________________

*Built with Python, powered by AI, designed for analysts.*
