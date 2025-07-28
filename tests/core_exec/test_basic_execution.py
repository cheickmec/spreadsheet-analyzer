"""
Tests for basic notebook execution functionality.

This test suite consolidates previous simple execution tests into one
file with parameterized tests for different scenarios.
"""

import pytest

from spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookBuilder,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, cells, expected_outputs",
    [
        (
            "simple_print",
            ["print('Hello World')"],
            [["Hello World"]],
        ),
        (
            "multiple_prints",
            ["print('Line 1')", "print('Line 2')"],
            [["Line 1"], ["Line 2"]],
        ),
        (
            "calculation",
            ["x = 5 + 3", "print(x)"],
            [[], ["8"]],
        ),
        (
            "pandas_dataframe",
            [
                "import pandas as pd",
                "df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})",
                "print(df.shape)",
            ],
            [[], [], ["(2, 2)"]],
        ),
    ],
)
async def test_basic_execution_scenarios(name, cells, expected_outputs):
    """Test various basic execution scenarios."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    for cell_code in cells:
        notebook.add_code_cell(cell_code)

    async with KernelService(profile) as service:
        session_id = await service.create_session(f"test-{name}")
        bridge = ExecutionBridge(service)
        results = await bridge.execute_notebook(session_id, notebook)

        assert len(results) == len(cells)
        for i, result in enumerate(results):
            assert result.status == "ok"
            # A simple check to see if the expected text is in the output
            if expected_outputs[i]:
                assert any(expected in str(output) for output in result.outputs for expected in expected_outputs[i])
