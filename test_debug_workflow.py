#!/usr/bin/env python3
"""Debug workflow to see why no cells are generated."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask, FormulaAnalysisTask
from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_debug_workflow():
    """Debug the workflow."""
    # First test tasks directly
    print("Testing tasks directly...")
    task = DataProfilingTask()
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }

    # Check if task can validate context
    issues = task.validate_context(context)
    print(f"Task validation issues: {issues}")

    # Generate cells
    cells = task.build_initial_cells(context)
    print(f"Task generated {len(cells)} cells")

    # Now test through workflow
    print("\nTesting through workflow...")
    workflow = NotebookWorkflow()

    # Create config - try specifying tasks explicitly
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        output_path="debug_output.ipynb",
        sheet_name="Yiriden Transactions 2025",
        mode=WorkflowMode.BUILD_ONLY,  # Just build, don't execute
        tasks=[DataProfilingTask, FormulaAnalysisTask],  # Explicit tasks
    )

    # Check config
    print(f"Config tasks: {config.tasks}")
    print(f"Config mode: {config.mode}")

    result = await workflow.run(config)

    print("\nWorkflow result:")
    print(f"Success: {result.success}")
    print(f"Notebook cells: {len(result.notebook.cells) if result.notebook else 0}")
    print(f"Warnings: {result.warnings}")
    print(f"Errors: {result.errors}")

    # Cleanup
    Path("debug_output.ipynb").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_debug_workflow())
