#!/usr/bin/env python3
"""Test kernel execution and notebook generation directly."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager
from spreadsheet_analyzer.notebook_llm.notebook_document import NotebookDocument


async def test_kernel_execution():
    """Test if kernel execution and notebook saving works."""

    # Create a simple notebook with test code
    notebook = NotebookDocument(title="Test Kernel Execution")

    # Add some test cells
    notebook.add_markdown_cell("# Test Kernel Execution", cell_id="title")
    notebook.add_code_cell("print('Hello from kernel!')", cell_id="hello")
    notebook.add_code_cell("import pandas as pd\nprint(f'Pandas version: {pd.__version__}')", cell_id="pandas_test")
    notebook.add_code_cell("x = 2 + 2\nprint(f'2 + 2 = {x}')", cell_id="math_test")

    print("Created notebook with test cells")
    print(f"Cell count: {len(notebook.cells)}")

    # Execute the notebook using kernel manager
    async with AgentKernelManager() as manager:
        session = await manager.create_session("test-session")
        print(f"Created kernel session: {session.session_id}")

        # Execute each code cell
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        print(f"Executing {len(code_cells)} code cells...")

        for cell in code_cells:
            if cell.source.strip():  # Skip empty cells
                print(f"Executing cell {cell.id}: {cell.source[:50]}...")
                result = await manager.execute_code(session, cell.source)
                print(f"  Result type: {type(result)}")
                print(f"  Has stdout: {'stdout' in result and bool(result['stdout'])}")
                print(f"  Has stderr: {'stderr' in result and bool(result['stderr'])}")
                if result.get("stdout"):
                    print(f"  Stdout: {result['stdout'][:100]}")
                if result.get("stderr"):
                    print(f"  Stderr: {result['stderr'][:100]}")

        # Save the session as a notebook
        output_path = Path("test_kernel_output.ipynb")
        manager.save_session_as_notebook(session, output_path, "Test Execution Results")
        print(f"Saved notebook to: {output_path}")

        # Check the saved file
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"Saved notebook size: {size} bytes")

            # Read first few lines to verify content
            with open(output_path) as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                print("First 5 lines of saved notebook:")
                for i, line in enumerate(first_lines, 1):
                    print(f"  {i}: {line}")
        else:
            print("ERROR: Notebook file was not created!")


if __name__ == "__main__":
    asyncio.run(test_kernel_execution())
