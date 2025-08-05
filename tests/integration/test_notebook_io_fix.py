#!/usr/bin/env python3
"""Test the fixed save_notebook function in notebook_io.py with raw kernel execution results."""

from pathlib import Path

from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO
from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument


def test_notebook_io_with_raw_execution_results():
    """Test that save_notebook properly converts raw kernel execution results."""

    print("Testing notebook_io.py save_notebook function with raw execution results...")

    # Create a notebook document similar to what business accounting workflow creates
    notebook = NotebookDocument(
        id="test_raw_results",
        cells=[],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        kernel_spec={"name": "python3", "display_name": "Python 3"},
        language_info={"name": "python", "version": "3.12"},
    )

    # Add a markdown cell
    title_cell = Cell(
        id="title", cell_type=CellType.MARKDOWN, source="# Test Notebook with Raw Execution Results", metadata={}
    )
    notebook.cells.append(title_cell)

    # Create a code cell with raw kernel execution result as output
    # This simulates what happens in the business accounting workflow
    raw_execution_result = {
        "msg_id": "2bdbeedf-e50f0a3027c0948da328296b_65275_0",
        "status": "ok",
        "outputs": [{"type": "stream", "text": "2 + 2 = 4\n"}, {"type": "execute_result", "data": {"text/plain": "4"}}],
        "error": None,
    }

    # Add code cell with this raw execution result
    cell = Cell(
        id="raw_result_test",
        cell_type=CellType.CODE,
        source="x = 2 + 2\nprint(f'2 + 2 = {x}')\nx",
        metadata={},
        outputs=[raw_execution_result],  # This is the problematic raw result
    )
    notebook.cells.append(cell)

    # Add another cell with proper individual outputs (should work fine)
    proper_outputs = [{"type": "stream", "text": "Hello World!\n"}]

    cell2 = Cell(
        id="proper_outputs_test",
        cell_type=CellType.CODE,
        source="print('Hello World!')",
        metadata={},
        outputs=proper_outputs,
    )
    notebook.cells.append(cell2)

    # Save using the NotebookIO class
    output_path = Path("test_notebook_io_output.ipynb")
    try:
        io = NotebookIO()
        io.save_notebook(notebook.to_nbformat(), output_path)
        print(f"✅ Notebook saved successfully to {output_path}")

        # Check the output
        if output_path.exists():
            import json

            with Path(output_path).open() as f:
                notebook_data = json.load(f)

            print(f"📊 Notebook contains {len(notebook_data['cells'])} cells")

            for i, cell_data in enumerate(notebook_data["cells"]):
                cell_type = cell_data.get("cell_type", "unknown")
                outputs = cell_data.get("outputs", [])

                print(f"📝 Cell {i + 1}: {cell_type}")

                if outputs:
                    print(f"   {len(outputs)} outputs:")
                    for j, output in enumerate(outputs):
                        output_type = output.get("output_type", "unknown")
                        print(f"      Output {j + 1}: {output_type}")

                        # Check if this is a clean output or raw execution result
                        if "msg_id" in output and "status" in output:
                            print(f"         ❌ STILL RAW EXECUTION RESULT: {output}")
                        else:
                            if output_type == "stream":
                                text = output.get("text", "")[:50]
                                print(f"         ✅ Clean stream: {text}...")
                            elif output_type == "execute_result":
                                data = output.get("data", {})
                                print(f"         ✅ Clean execute_result: {data}")
                            else:
                                print(f"         ✅ Clean {output_type}: {output}")
                else:
                    print("   No outputs")
        else:
            print("❌ Notebook file was not created!")

    except Exception as e:
        print(f"❌ Error saving notebook: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_notebook_io_with_raw_execution_results()
