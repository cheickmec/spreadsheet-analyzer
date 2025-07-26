"""
Spreadsheet Analyzer - A modular system for analyzing Excel and CSV files.

This package provides a three-tier architecture:

1. **Core Execution Layer** (`core_exec`):
   - Generic notebook building and execution primitives
   - Kernel management and I/O operations
   - Domain-agnostic quality assessment

2. **Plugin System** (`plugins`):
   - Domain-specific analysis tasks (spreadsheet, CSV, etc.)
   - Extensible quality inspection
   - Task-based cell generation

3. **Workflow Orchestration** (`workflows`):
   - High-level APIs combining core + plugins
   - CLI/API integration points
   - Complete analysis workflows

Usage:
    # Simple analysis notebook generation
    from spreadsheet_analyzer.workflows import create_analysis_notebook

    result = await create_analysis_notebook(
        file_path="data.xlsx",
        output_path="analysis.ipynb",
        execute=True
    )

    # Advanced workflow configuration
    from spreadsheet_analyzer.workflows import NotebookWorkflow, WorkflowConfig

    config = WorkflowConfig(
        file_path="data.xlsx",
        tasks=["data_profiling", "outlier_detection"],
        quality_checks=True
    )

    workflow = NotebookWorkflow()
    result = await workflow.run(config)
"""

# Core execution primitives (domain-agnostic)
from .core_exec import (
    CellType,
    ExecutionBridge,
    ExecutionResult,
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookCell,
    NotebookIO,
    QualityInspector,
    QualityMetrics,
)

# Plugin system
from .plugins.base import PluginRegistry, Task, registry
from .plugins.base import QualityInspector as PluginQualityInspector

# Convenience imports
from .plugins.spreadsheet import register_all_plugins as register_spreadsheet_plugins

# Workflow orchestration (main user API)
from .workflows import (
    NotebookWorkflow,
    WorkflowConfig,
    WorkflowMode,
    WorkflowResult,
    create_analysis_notebook,
    execute_notebook,
)

__version__ = "1.0.0"

__all__ = [
    # Core execution layer
    "KernelService",
    "KernelProfile",
    "ExecutionResult",
    "NotebookBuilder",
    "NotebookCell",
    "CellType",
    "NotebookIO",
    "ExecutionBridge",
    "QualityMetrics",
    "QualityInspector",
    # Plugin system
    "Task",
    "PluginQualityInspector",
    "PluginRegistry",
    "registry",
    # Workflow orchestration
    "NotebookWorkflow",
    "WorkflowConfig",
    "WorkflowResult",
    "WorkflowMode",
    "create_analysis_notebook",
    "execute_notebook",
    # Convenience
    "register_spreadsheet_plugins",
]


def quick_start() -> str:
    """
    Return a quick start guide for new users.

    Returns:
        Multi-line string with usage examples
    """
    return """
# Spreadsheet Analyzer Quick Start

## 1. Simple Analysis (Async)
```python
import asyncio
from spreadsheet_analyzer import create_analysis_notebook

async def analyze():
    result = await create_analysis_notebook(
        file_path="your_data.xlsx",
        output_path="analysis.ipynb",
        execute=True
    )

    if result.success:
        print(f"Analysis complete! Quality score: {result.quality_metrics.overall_score}")
    else:
        print(f"Errors: {result.errors}")

asyncio.run(analyze())
```

## 2. Advanced Configuration
```python
from spreadsheet_analyzer import NotebookWorkflow, WorkflowConfig, WorkflowMode

config = WorkflowConfig(
    file_path="data.xlsx",
    output_path="custom_analysis.ipynb",
    sheet_name="Sales Data",
    mode=WorkflowMode.BUILD_AND_EXECUTE,
    tasks=["data_profiling", "formula_analysis", "outlier_detection"],
    quality_checks=True,
    execute_timeout=600
)

workflow = NotebookWorkflow()
result = await workflow.run(config)
```

## 3. Plugin Development
```python
from spreadsheet_analyzer.plugins.base import BaseTask
from spreadsheet_analyzer import registry

class CustomAnalysisTask(BaseTask):
    def __init__(self):
        super().__init__("custom_analysis", "My custom analysis")

    def build_initial_cells(self, context):
        # Return list of NotebookCell objects
        pass

# Register your plugin
registry.register_task(CustomAnalysisTask())
```

See documentation for more examples and advanced usage.
"""
