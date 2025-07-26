# 🚀 CLI Redesign Plan

## 🎯 Executive Summary

**STATUS: 🚧 ARCHITECTURE RESTORATION IN PROGRESS**

This document outlines the complete redesign of the spreadsheet-analyzer CLI from a complex multi-file Click-based interface to a streamlined, unified command-line tool that leverages our three-tier architecture (`core_exec`, `plugins`, `workflows`).

**BACKWARD COMPATIBILITY**: Completely eliminated. All legacy CLI interfaces, entry points, and compatibility layers have been removed in favor of the unified `analyze-spreadsheet` command.

**ARCHITECTURE ISSUE IDENTIFIED**: The utils module was incorrectly added, creating duplicate implementations of core_exec functionality. This violates the three-tier architecture and must be corrected.

## ⚠️ CRITICAL: Architecture Principles

### The Three-Tier Architecture is MANDATORY:

1. **core_exec/**: Generic notebook & kernel primitives (NO domain logic)

   - NotebookBuilder: For ALL notebook construction
   - NotebookIO: For ALL notebook I/O operations
   - KernelService: For ALL kernel management
   - ExecutionBridge: For orchestrating execution
   - QualityInspector: For generic quality checks

1. **plugins/**: Domain-specific functionality (spreadsheet, formulas, etc.)

   - spreadsheet/tasks.py: Spreadsheet analysis tasks
   - spreadsheet/quality.py: Spreadsheet-specific quality checks
   - spreadsheet/io/: Excel/CSV I/O utilities
   - spreadsheet/analysis/: Formula errors, data profiling

1. **workflows/**: High-level orchestration using core_exec + plugins

   - NotebookWorkflow: Main orchestration API

### FORBIDDEN Practices:

- ❌ NEVER create utils/ modules that duplicate core_exec functionality
- ❌ NEVER bypass the architecture layers
- ❌ NEVER simplify tests without documented justification
- ❌ NEVER import from utils/ for notebook operations (use core_exec)

### Required Practices:

- ✅ Domain utilities go in plugins/{domain}/
- ✅ Use core_exec.NotebookBuilder for ALL notebook operations
- ✅ Use core_exec.NotebookIO for ALL notebook I/O
- ✅ Maintain comprehensive async/integration tests (2500+ lines)
- ✅ All tests must use real kernels, not mocks

## 📋 Architecture Restoration Tasks

### Phase 0: Architecture Restoration (🚧 IN PROGRESS)

- [ ] 🚧 Remove duplicate utils implementations
  - [ ] Delete utils/notebook_builder.py (duplicates core_exec)
  - [ ] Delete utils/notebook_io.py (duplicates core_exec)
- [ ] ⏳ Relocate domain utilities to plugins
  - [ ] Move utils/excel_io.py → plugins/spreadsheet/io/
  - [ ] Move utils/formula_errors.py → plugins/spreadsheet/analysis/
  - [ ] Move utils/data_profiling.py → plugins/spreadsheet/analysis/
- [ ] ⏳ Update all imports to use proper architecture
- [ ] ⏳ Restore comprehensive test coverage (from ~627 to ~2700 lines)

### Phase 1: Core CLI (✅ = Done, 🚧 = In Progress, ⏳ = Pending)

- [x] ✅ Create new unified `analyze.py` CLI entry point
- [x] ✅ Remove old `analyze_sheet.py` and `analyze_sheet_langchain.py`
- [x] ✅ Update CLI to use LLM by default with `--no-llm` flag
- [x] ✅ Integrate with three-tier architecture (core_exec, plugins, workflows)

### Phase 2: Test Data Creation

- [ ] ⏳ Create `test_data/` directory with sample files:
  - [ ] `simple_sales.xlsx` - Multi-sheet sales data
  - [ ] `financial_model.xlsx` - Complex formulas
  - [ ] `inventory_tracking.csv` - CSV example
  - [ ] `employee_records.xlsx` - Data quality issues

### Phase 3: Test Infrastructure

- [ ] ⏳ Create `reference_notebooks/` directory structure
- [ ] ⏳ Create `tests/cli/` test structure:
  - [ ] `conftest.py` - Test fixtures and utilities
  - [ ] `test_deterministic_generation.py` - Generate reference notebooks
  - [ ] `test_notebook_validation.py` - Validate notebook quality

### Phase 4: Reference Notebook Generation

- [ ] ⏳ Implement tests that generate notebooks for each test file/sheet
- [ ] ⏳ Store notebooks permanently in Git for regression testing
- [ ] ⏳ Add validation for notebook structure and output quality

### Phase 5: Integration & Documentation

- [ ] ⏳ Add comprehensive docstrings and type hints
- [ ] ⏳ Update project documentation

## 🗂️ Directory Structure (Target)

```
spreadsheet-analyzer/
├── test_data/                      # Test input files (Git tracked)
│   ├── simple_sales.xlsx
│   ├── financial_model.xlsx
│   ├── inventory_tracking.csv
│   └── employee_records.xlsx
│
├── reference_notebooks/            # Generated outputs (Git tracked)
│   ├── simple_sales/
│   │   ├── Sheet1.ipynb
│   │   ├── Sheet2.ipynb
│   │   └── Summary.ipynb
│   ├── financial_model/
│   │   └── ...
│   └── ...
│
├── src/spreadsheet_analyzer/
│   ├── core_exec/                  # Generic notebook/kernel primitives
│   │   ├── __init__.py
│   │   ├── bridge.py               # Execution orchestration
│   │   ├── kernel_service.py       # Kernel management
│   │   ├── notebook_builder.py     # Notebook construction
│   │   ├── notebook_io.py          # Notebook I/O
│   │   └── quality.py              # Quality inspection
│   │
│   ├── plugins/                    # Domain-specific functionality
│   │   ├── __init__.py
│   │   ├── base.py                 # Plugin protocols
│   │   └── spreadsheet/
│   │       ├── __init__.py
│   │       ├── tasks.py            # Analysis tasks
│   │       ├── quality.py          # Quality checks
│   │       ├── io/                 # I/O utilities
│   │       │   ├── __init__.py
│   │       │   └── excel_io.py
│   │       └── analysis/           # Analysis utilities
│   │           ├── __init__.py
│   │           ├── formula_errors.py
│   │           └── data_profiling.py
│   │
│   ├── workflows/                  # High-level orchestration
│   │   ├── __init__.py
│   │   └── notebook_workflow.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── analyze.py              # THE main entry point
│   │
│   └── utils/                      # Minimal generic utilities ONLY
│       ├── __init__.py
│       └── cost.py                 # LLM cost calculation ONLY
│
└── tests/
    ├── core_exec/                  # Comprehensive async tests
    │   ├── test_bridge.py          # ~500 lines
    │   ├── test_kernel_service.py  # ~500 lines
    │   ├── test_notebook_builder.py # ~700 lines
    │   ├── test_notebook_io.py     # ~600 lines
    │   └── test_quality.py         # ~400 lines
    └── cli/
        ├── conftest.py
        ├── test_deterministic_generation.py
        └── test_notebook_validation.py
```

## 🎨 CLI Design

### New Simple Interface:

```bash
# Default: Use LLM
analyze-spreadsheet data.xlsx

# Deterministic only (for tests/cost control)  
analyze-spreadsheet data.xlsx --no-llm

# Sheet selection
analyze-spreadsheet data.xlsx --sheet "Revenue" --no-llm

# Task selection  
analyze-spreadsheet data.xlsx --tasks "profile,outliers" --no-llm
```

### Key Features:

- **LLM by Default**: `--use-llm` is the default behavior
- **Deterministic Option**: `--no-llm` for testing/cost control
- **One Output Format**: Jupyter notebooks only
- **Smart Defaults**: Minimal configuration required
- **Plugin Integration**: Uses three-tier architecture

## 🧪 Testing Strategy

### Deterministic Generation Tests:

- Run CLI with `--no-llm` on all test files
- Generate notebooks for every sheet
- Store results in `reference_notebooks/` (Git tracked)
- Verify reproducibility (same input = same output)

### Notebook Validation Tests:

- Validate nbformat structure
- Check for presence of outputs in code cells
- Verify no error outputs
- Validate markdown formatting
- Check task coverage (all plugins ran)

### Benefits:

- **No Network Costs**: Tests use `--no-llm` flag
- **Visual Regression**: Notebook changes visible in PRs
- **Living Documentation**: Reference notebooks show real outputs
- **Comprehensive Coverage**: Every plugin and feature tested

## 📝 Notes

This plan focuses on radical simplification while maintaining all functionality through the underlying three-tier architecture. The CLI becomes a thin wrapper that orchestrates the workflow layer.

**BACKWARD COMPATIBILITY**: Completely eliminated. All legacy CLI interfaces, entry points, and compatibility layers have been removed in favor of the unified `analyze-spreadsheet` command.
