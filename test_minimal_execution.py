#!/usr/bin/env python3
"""Test minimal notebook execution with print outputs."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec import ExecutionBridge, KernelProfile, KernelService, NotebookBuilder


async def test_minimal_execution():
    """Test minimal notebook execution."""
    profile = KernelProfile(
        output_drain_timeout_ms=150,
        output_drain_max_timeout_ms=1000,
        output_drain_max_attempts=3,
        wait_for_shell_reply=True,
    )

    # Create a simple notebook
    notebook = NotebookBuilder()
    notebook.add_code_cell("print('Hello World')")
    notebook.add_code_cell("""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(f'✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns')
print(f'📊 Data shape: {df.shape}')
""")

    print("Original notebook:")
    for i, cell in enumerate(notebook.cells):
        print(f"\nCell {i}:")
        print("".join(cell.source))
        print(f"Outputs: {cell.outputs}")

    # Execute the notebook
    async with KernelService(profile) as service:
        session_id = await service.create_session("test")
        bridge = ExecutionBridge(service)

        print("\n" + "=" * 60)
        print("Executing notebook...")

        executed_notebook = await bridge.execute_notebook(session_id, notebook)

        print("\nExecuted notebook:")
        for i, cell in enumerate(executed_notebook.cells):
            print(f"\nCell {i}:")
            print("".join(cell.source))
            print(f"Outputs: {cell.outputs}")
            if cell.outputs:
                for output in cell.outputs:
                    if output.get("output_type") == "stream":
                        print(f"  Stream output: {output.get('text', [])}")

    # Also save the notebook to inspect
    from src.spreadsheet_analyzer.core_exec import NotebookIO

    io = NotebookIO()
    io.write_notebook(executed_notebook, "test_minimal_executed.ipynb", overwrite=True)
    print("\nSaved executed notebook to test_minimal_executed.ipynb")


if __name__ == "__main__":
    asyncio.run(test_minimal_execution())
