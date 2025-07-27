#!/usr/bin/env python
"""Test different output methods in kernel execution."""

import asyncio

from src.spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookIO,
)


async def test_display_methods():
    """Test different display methods to see which produce outputs."""

    print("=== Testing Display Methods ===\n")

    # Create notebook with different output methods
    builder = NotebookBuilder()
    builder.add_markdown_cell("# Testing Display Methods")

    # Test 1: Simple print
    builder.add_code_cell("print('Test 1: Simple print works!')")

    # Test 2: IPython display
    builder.add_code_cell("""
from IPython.display import display
display('Test 2: IPython display')
""")

    # Test 3: DataFrame display
    builder.add_code_cell("""
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print('Test 3a: DataFrame with print:')
print(df)
print('\\nTest 3b: DataFrame with display:')
display(df)
""")

    # Test 4: Matplotlib
    builder.add_code_cell("""
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Test 4: Matplotlib plot')
print('Before plt.show()')
plt.show()
print('After plt.show()')
""")

    # Test 5: Just returning a value
    builder.add_code_cell("""
# Test 5: Just returning a value
df = pd.DataFrame({'X': [7, 8, 9]})
df  # This should display in Jupyter
""")

    print(f"Created notebook with {builder.cell_count()} cells")

    # Execute with aggressive timing
    profile = KernelProfile(
        wait_for_shell_reply=True,
        output_drain_timeout_ms=500,
        output_drain_max_timeout_ms=2000,
        output_drain_max_attempts=5,
    )

    kernel_service = KernelService(profile)
    execution_bridge = ExecutionBridge(kernel_service, execution_timeout=30)

    async with kernel_service:
        session_id = await kernel_service.create_session("display-test")
        print(f"Created session: {session_id}")

        # Warm up
        await asyncio.sleep(0.5)

        # Execute notebook
        print("\nExecuting notebook...")
        executed_notebook = await execution_bridge.execute_notebook(session_id, builder)

        # Check outputs
        print("\n=== Results ===")
        code_cells = [cell for cell in executed_notebook.cells if cell.cell_type.value == "code"]

        for i, cell in enumerate(code_cells):
            print(f"\nCell {i + 1}:")
            print(f"  Code: {cell.source[0].split(':')[0]}...")
            print(f"  Execution count: {cell.execution_count}")
            print(f"  Number of outputs: {len(cell.outputs) if cell.outputs else 0}")

            if cell.outputs:
                for j, output in enumerate(cell.outputs):
                    output_type = output.get("output_type", output.get("type", "unknown"))
                    print(f"  Output {j + 1}: type={output_type}")

                    if output_type == "stream":
                        text = output.get("text", [])
                        if isinstance(text, list):
                            text = "".join(text)
                        print(f"    Preview: {text.strip()[:60]}...")
                    elif output_type == "execute_result":
                        data = output.get("data", {})
                        print(f"    Data types: {list(data.keys())}")
                    elif output_type == "display_data":
                        data = output.get("data", {})
                        print(f"    Display types: {list(data.keys())}")

        # Save notebook
        output_path = "test_display_methods.ipynb"
        NotebookIO.write_notebook(executed_notebook, output_path, overwrite=True)
        print(f"\n💾 Saved to: {output_path}")

        await kernel_service.close_session(session_id)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_display_methods())
