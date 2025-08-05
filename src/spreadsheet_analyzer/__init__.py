"""
Spreadsheet Analyzer - Excel analysis using deterministic pipeline and LLM tools.

This package provides intelligent Excel file analysis through:

1. **Deterministic Pipeline** (`pipeline`):
   - Security scanning
   - Structure analysis
   - Formula dependency analysis
   - Content intelligence

2. **Notebook Interface** (`notebook_cli`):
   - Automated LLM-powered analysis
   - Interactive Jupyter notebook generation
   - Formula evaluation and graph queries

Usage:
    # Run automated analysis via CLI
    python -m spreadsheet_analyzer.notebook_cli data.xlsx

    # Direct pipeline usage
    from spreadsheet_analyzer.pipeline import DeterministicPipeline

    pipeline = DeterministicPipeline()
    result = pipeline.run("data.xlsx")
"""

# Core execution primitives (domain-agnostic)
from .core_exec import (
    ExecutionBridge,
    ExecutionResult,
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookIO,
    QualityInspector,
    QualityMetrics,
)

__version__ = "1.0.0"

__all__ = [
    # Core execution layer
    "ExecutionBridge",
    "ExecutionResult",
    "KernelProfile",
    "KernelService",
    "NotebookBuilder",
    "NotebookIO",
    "QualityInspector",
    "QualityMetrics",
]


def quick_start() -> str:
    """
    Return a quick start guide for new users.

    Returns:
        Multi-line string with usage examples
    """
    return """
# Spreadsheet Analyzer Quick Start

## 1. Automated Analysis with LLM
```bash
# Basic usage
python -m spreadsheet_analyzer.notebook_cli data.xlsx

# With options
python -m spreadsheet_analyzer.notebook_cli data.xlsx \\
    --model claude-3-5-sonnet-20241022 \\
    --max-rounds 5 \\
    --sheet-index 0 \\
    --notebook-path analysis.ipynb
```

## 2. Direct Pipeline Usage
```python
from spreadsheet_analyzer.pipeline import DeterministicPipeline

# Run deterministic analysis
pipeline = DeterministicPipeline()
result = pipeline.run("data.xlsx")

if result.success:
    print(f"Analysis complete!")
    print(f"Security risk: {result.security.risk_level}")
    print(f"Sheets: {result.structure.sheet_count}")
    print(f"Formula complexity: {result.formulas.formula_complexity_score}/100")
    print(f"Circular references: {result.formulas.has_circular_references}")
```

## 3. Programmatic Notebook Usage
```python
from spreadsheet_analyzer.notebook_session import notebook_session
from spreadsheet_analyzer.notebook_llm_interface import get_notebook_tools

async with notebook_session("my_session", "output.ipynb") as session:
    # Execute code
    result = await session.toolkit.execute_code("import pandas as pd")

    # Add markdown
    result = await session.toolkit.render_markdown("# Analysis Results")

    # Save notebook
    await session.toolkit.save_notebook("output.ipynb", overwrite=True)
```

See notebook_cli.py for the complete implementation.
"""
