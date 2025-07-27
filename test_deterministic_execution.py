#!/usr/bin/env python
"""Test deterministic notebook execution directly."""

import asyncio
from pathlib import Path

from src.spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookIO,
)


async def test_deterministic_execution():
    """Test executing the already generated deterministic notebook."""

    # Path to the generated notebook
    notebook_path = Path("outputs/Business Accounting/Sheet1.ipynb")

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        print("Please run: uv run src/spreadsheet_analyzer/cli/analyze.py --no-llm first")
        return

    print(f"Loading notebook from: {notebook_path}")

    # Load the notebook
    notebook = NotebookIO.read_notebook(notebook_path)
    print(f"Loaded notebook with {notebook.cell_count()} cells")

    # Count code cells
    code_cells = [cell for cell in notebook.cells if cell.cell_type.value == "code"]
    print(f"Found {len(code_cells)} code cells")

    # Show first few code cells
    print("\n=== First 3 Code Cells ===")
    for i, cell in enumerate(code_cells[:3]):
        print(f"\nCell {i + 1}:")
        print(f"  Code preview: {cell.source[0][:60]}...")
        print(f"  Current outputs: {len(cell.outputs) if cell.outputs else 0}")

    # Set up execution with more aggressive timing
    profile = KernelProfile(
        wait_for_shell_reply=True,
        output_drain_timeout_ms=500,  # Increased from 150
        output_drain_max_timeout_ms=2000,  # Increased from 1000
        output_drain_max_attempts=5,  # Increased from 3
    )

    kernel_service = KernelService(profile)
    execution_bridge = ExecutionBridge(kernel_service, execution_timeout=30)

    print("\n=== Starting Execution ===")

    async with kernel_service:
        # Create session
        session_id = await kernel_service.create_session("deterministic-test")
        print(f"Created kernel session: {session_id}")

        # Add a warmup delay
        print("Warming up kernel...")
        await asyncio.sleep(1.0)

        # Test with a simple command first
        print("\nTesting kernel with simple command...")
        test_result = await kernel_service.execute(session_id, "print('Kernel is ready!')")
        print(f"Test result: {test_result.status}, outputs: {len(test_result.outputs)}")
        if test_result.outputs:
            print(f"Test output: {test_result.outputs[0]}")

        # Execute the notebook
        print("\nExecuting notebook cells...")
        executed_notebook = await execution_bridge.execute_notebook(session_id, notebook)

        # Check results
        print("\n=== Execution Results ===")
        executed_code_cells = [cell for cell in executed_notebook.cells if cell.cell_type.value == "code"]

        outputs_found = 0
        for i, cell in enumerate(executed_code_cells[:5]):  # Check first 5
            print(f"\nCell {i + 1}:")
            print(f"  Execution count: {cell.execution_count}")
            print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
            if cell.outputs:
                outputs_found += 1
                for output in cell.outputs[:1]:  # Show first output
                    output_type = output.get("output_type", output.get("type", "unknown"))
                    print(f"  Output type: {output_type}")

        print(f"\n✅ Total cells with outputs: {outputs_found}/{len(executed_code_cells)}")

        # Save the executed notebook
        output_path = Path("test_deterministic_executed.ipynb")
        NotebookIO.write_notebook(executed_notebook, output_path, overwrite=True)
        print(f"\n💾 Saved executed notebook to: {output_path}")

        # Close session
        await kernel_service.close_session(session_id)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_deterministic_execution())
