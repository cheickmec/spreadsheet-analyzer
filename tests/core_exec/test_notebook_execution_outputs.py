"""
Tests for notebook execution with actual output generation.

This test suite consolidates previous output-related tests and validates
that various output types (print, display, plots, errors) are correctly
captured during notebook execution.
"""

from pathlib import Path

import pytest

from spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookIO,
)


@pytest.mark.asyncio
async def test_various_output_types(tmp_path: Path) -> None:
    """Test execution of a notebook with various output types."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_markdown_cell("# Output Types Test")
    notebook.add_code_cell("print('This is a stream output.')")
    notebook.add_code_cell("'This is an execute_result.'")
    notebook.add_code_cell("from IPython.display import display, HTML; display(HTML('<b>This is display_data</b>'))")
    notebook.add_code_cell("import matplotlib.pyplot as plt; plt.plot([1, 2]); plt.show()")
    notebook.add_code_cell("1 / 0")

    async with KernelService(profile) as service:
        session_id = await service.create_session("output-types-session")
        bridge = ExecutionBridge(service)
        results = await bridge.execute_notebook(session_id, notebook, stop_on_error=False)

        # Save for inspection
        output_path = tmp_path / "output_types.ipynb"

        # The notebook was already executed and outputs were added during execution
        # So we can just save the executed notebook directly
        NotebookIO.write_notebook(notebook, output_path, overwrite=True)

        # Assertions
        # Cell 1: stream
        assert results[0].status == "ok"
        assert results[0].outputs[0]["type"] == "stream"
        assert "This is a stream output." in results[0].outputs[0]["text"]

        # Cell 2: execute_result
        assert results[1].status == "ok"
        assert results[1].outputs[0]["type"] == "execute_result"
        assert "This is an execute_result." in results[1].outputs[0]["data"]["text/plain"]

        # Cell 3: display_data
        assert results[2].status == "ok"
        assert results[2].outputs[0]["type"] == "display_data"
        assert "<b>This is display_data</b>" in results[2].outputs[0]["data"]["text/html"]

        # Cell 4: plot
        assert results[3].status == "ok"
        assert any(o["type"] == "display_data" and "image/png" in o["data"] for o in results[3].outputs)

        # Cell 5: error
        assert results[4].status == "error"
        # When there's an error, the error information is in the error field, not outputs
        assert results[4].error is not None
        assert "ZeroDivisionError" in results[4].error["ename"]
