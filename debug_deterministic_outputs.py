#!/usr/bin/env python
"""Debug why deterministic analysis doesn't capture outputs."""

import asyncio
import logging
from pathlib import Path

from spreadsheet_analyzer.core_exec import KernelProfile, NotebookIO
from spreadsheet_analyzer.plugins.base import registry
from spreadsheet_analyzer.plugins.spreadsheet import register_all_plugins as register_spreadsheet_plugins
from spreadsheet_analyzer.workflows import NotebookWorkflow, WorkflowConfig, WorkflowMode

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Debug deterministic analysis output capture."""
    # Register plugins
    register_spreadsheet_plugins()

    # Configuration
    excel_path = Path("test_assets/collection/business-accounting/Business Accounting.xlsx")

    # Get the first available sheet
    from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets

    sheets = list_sheets(excel_path)
    sheet_name = sheets[0] if sheets else "Sheet1"

    output_dir = Path("analysis_results/Business Accounting")
    output_path = output_dir / f"{sheet_name.replace(' ', '_')}_debug.ipynb"

    print(f"🔍 Debugging {excel_path.name}")
    print(f"   📄 Sheet: {sheet_name}")
    print("-" * 60)

    # Get available tasks - use only the first few for testing
    tasks = registry.list_tasks()
    deterministic_tasks = [task for task in tasks if hasattr(task, "is_deterministic") and task.is_deterministic]
    if not deterministic_tasks:
        deterministic_tasks = tasks[:3]  # Use first 3 tasks for testing

    task_names = [task.name for task in deterministic_tasks[:3]]  # Limit to 3 for debugging

    print(f"📋 Using {len(task_names)} tasks for debugging:")
    for name in task_names:
        print(f"   - {name}")

    # Create workflow config with explicit kernel settings
    config = WorkflowConfig(
        file_path=str(excel_path),
        output_path=str(output_path),
        sheet_name=sheet_name,
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=task_names,
        kernel_profile=KernelProfile(
            name="python3",
            max_execution_time=120,
            idle_timeout_seconds=60,
            wait_for_shell_reply=True,
            output_drain_timeout_ms=1000,  # Increased
            output_drain_max_timeout_ms=3000,  # Increased
            output_drain_max_attempts=10,  # Increased
        ),
        execute_timeout=120,
    )

    # Run workflow
    workflow = NotebookWorkflow()
    try:
        result = await workflow.run(config)
    finally:
        await workflow.cleanup()

    print("\n" + "=" * 60)
    print("CHECKING OUTPUTS IN SAVED NOTEBOOK")
    print("=" * 60)

    # Read the saved notebook and check outputs
    if result.output_path and Path(result.output_path).exists():
        notebook = NotebookIO.read_notebook(Path(result.output_path))

        print(f"\nTotal cells: {len(notebook.cells)}")

        code_cells = [cell for cell in notebook.cells if cell.cell_type.value == "code"]
        print(f"Code cells: {len(code_cells)}")

        for i, cell in enumerate(code_cells):
            print(f"\n📋 Cell {i + 1}:")
            print(f"   Source: {cell.source[0][:60]}..." if cell.source else "   Source: (empty)")
            print(f"   Execution count: {cell.execution_count}")
            print(f"   Number of outputs: {len(cell.outputs) if cell.outputs else 0}")

            if cell.outputs:
                for j, output in enumerate(cell.outputs):
                    output_type = output.get("output_type", output.get("type", "unknown"))
                    print(f"   Output {j + 1}: type={output_type}")

                    if output_type == "stream":
                        text = output.get("text", [])
                        if isinstance(text, list):
                            text = "".join(text)
                        print(f"      Text: {text.strip()[:100]}...")
                    elif output_type == "execute_result":
                        data = output.get("data", {})
                        print(f"      Data keys: {list(data.keys())}")
                    elif output_type == "error":
                        print(f"      Error: {output.get('ename')} - {output.get('evalue')}")

    # Let's also check the actual notebook generation code
    print("\n" + "=" * 60)
    print("CHECKING NOTEBOOK GENERATION")
    print("=" * 60)

    # Import the specific plugin to see what it generates
    from spreadsheet_analyzer.plugins.spreadsheet.tasks.integrity_checks import IntegrityChecks

    integrity_plugin = IntegrityChecks()

    # Create a simple test dataframe
    import pandas as pd

    test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Generate notebook cells
    print("\nGenerating cells from IntegrityChecks plugin...")
    cells = integrity_plugin.generate_notebook_cells(test_df, excel_path, sheet_name)

    print(f"Generated {len(cells)} cells")
    for i, cell in enumerate(cells[:3]):  # Show first 3 cells
        print(f"\nGenerated Cell {i + 1}:")
        print(f"  Type: {cell.cell_type}")
        print(f"  Source preview: {cell.source[0][:80]}..." if cell.source else "  Source: (empty)")


if __name__ == "__main__":
    asyncio.run(main())
