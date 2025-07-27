#!/usr/bin/env python3
"""Test a fresh execution with all fixes to verify outputs are captured."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask, FormulaAnalysisTask
from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_fresh_execution():
    """Test fresh notebook generation and execution."""
    # Create unique output name
    timestamp = datetime.now().strftime("%H%M%S")
    output_path = f"test_fresh_output_{timestamp}.ipynb"

    # Create workflow
    workflow = NotebookWorkflow()

    # Create config for build and execute
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        output_path=output_path,
        sheet_name="Yiriden Transactions 2025",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=[DataProfilingTask, FormulaAnalysisTask],
    )

    print("Running fresh workflow with all fixes...")
    print(f"Output will be saved to: {output_path}")

    result = await workflow.run(config)

    print("\nWorkflow completed!")
    print(f"Success: {result.success}")
    print(f"Output path: {result.output_path}")

    if result.notebook:
        print(f"\nNotebook has {len(result.notebook.cells)} cells:")

        # Check each cell for outputs
        import_cell_found = False
        load_cell_found = False

        for i, cell in enumerate(result.notebook.cells):
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                outputs = cell.get("outputs", [])

                # Check if this is the import cell
                if "Data profiling imports" in source and "import pandas" in source:
                    import_cell_found = True
                    print(f"\n✅ Import cell found (Cell {i}):")
                    print(f"   Outputs: {len(outputs)}")
                    if outputs:
                        for output in outputs:
                            if output.get("output_type") == "stream":
                                text = "".join(output.get("text", []))
                                print(f"   Output text: {text.strip()}")

                # Check if this is the data loading cell
                elif "Load Excel data" in source and "Successfully loaded" in source:
                    load_cell_found = True
                    print(f"\n✅ Data loading cell found (Cell {i}):")
                    print(f"   Outputs: {len(outputs)}")
                    if outputs:
                        for output in outputs:
                            if output.get("output_type") == "stream":
                                text = "".join(output.get("text", []))
                                print(f"   Output text: {text.strip()}")
                    else:
                        print("   ❌ No outputs captured!")

        if not import_cell_found:
            print("\n❌ Import cell not found!")
        if not load_cell_found:
            print("\n❌ Data loading cell not found!")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    # Read the saved file to double-check
    print("\nReading saved notebook from disk...")
    import json

    with open(output_path) as f:
        saved_notebook = json.load(f)

    print(f"Saved notebook has {len(saved_notebook['cells'])} cells")

    # Cleanup
    Path(output_path).unlink(missing_ok=True)
    print(f"\nCleaned up test file: {output_path}")


if __name__ == "__main__":
    asyncio.run(test_fresh_execution())
