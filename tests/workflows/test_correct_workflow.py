#!/usr/bin/env python3
"""Test workflow with correct task specification."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_correct_workflow():
    """Test workflow with proper configuration."""
    # Create workflow
    workflow = NotebookWorkflow()

    # Create config - let auto-selection work
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        output_path="test_correct_output.ipynb",
        sheet_name="Yiriden Transactions 2025",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        # Don't specify tasks - let it auto-select based on file type
    )

    print("Running workflow with auto-selected tasks...")
    result = await workflow.run(config)

    print("\nWorkflow completed!")
    print(f"Success: {result.success}")
    print(f"Output path: {result.output_path}")

    if result.notebook:
        print(f"\nNotebook has {len(result.notebook.cells)} cells")

        # Check for outputs in code cells
        cells_with_outputs = 0
        total_outputs = 0

        for i, cell in enumerate(result.notebook.cells):
            if cell.cell_type.value == "code":
                outputs = cell.outputs if cell.outputs else []
                if outputs:
                    cells_with_outputs += 1
                    total_outputs += len(outputs)

        print(f"Code cells with outputs: {cells_with_outputs}")
        print(f"Total outputs: {total_outputs}")

        # Check specific cells
        for i, cell in enumerate(result.notebook.cells[:6]):  # First 6 cells
            if cell.cell_type.value == "code":
                source = "".join(cell.source)[:50]
                outputs = cell.outputs if cell.outputs else []
                print(f"\nCell {i}: {source}...")
                print(f"  Outputs: {len(outputs)}")

                if outputs and len(outputs) > 0:
                    for j, output in enumerate(outputs[:2]):  # First 2 outputs
                        if output.get("output_type") == "stream":
                            text = "".join(output.get("text", []))[:80]
                            print(f"    Output {j}: {text.strip()}...")

    if result.errors:
        print(f"\nErrors: {result.errors}")
    if result.warnings:
        print(f"\nWarnings: {result.warnings}")

    # Cleanup
    Path("test_correct_output.ipynb").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_correct_workflow())
