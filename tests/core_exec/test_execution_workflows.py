"""
Tests for different notebook execution workflows.

This test suite consolidates previous workflow-related tests and validates
different execution strategies like sequential and deterministic execution.
"""

import pytest

from spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookBuilder,
)


@pytest.mark.asyncio
async def test_sequential_execution_workflow() -> None:
    """Test a sequential execution workflow."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_code_cell("a = 1")
    notebook.add_code_cell("b = a + 1")
    notebook.add_code_cell("c = b + 1; print(c)")

    async with KernelService(profile) as service:
        session_id = await service.create_session("sequential-workflow-session")
        bridge = ExecutionBridge(service)
        results = await bridge.execute_notebook(session_id, notebook)

        assert len(results) == 3
        assert results[0].status == "ok"
        assert results[1].status == "ok"
        assert results[2].status == "ok"
        assert any("3" in str(o) for o in results[2].outputs)


@pytest.mark.asyncio
async def test_deterministic_execution_workflow() -> None:
    """Test a deterministic execution workflow."""
    # This test ensures that given the same notebook, the execution yields
    # the same results.
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_code_cell("import random; random.seed(42); print(random.random())")

    async with KernelService(profile) as service:
        session_id = await service.create_session("deterministic-workflow-session")
        bridge = ExecutionBridge(service)

        # First execution
        results1 = await bridge.execute_notebook(session_id, notebook)
        output1 = results1[0].outputs[0]["text"]

        # Second execution (in a new session to ensure no state leakage)
        session_id2 = await service.create_session("deterministic-workflow-session-2")
        results2 = await bridge.execute_notebook(session_id2, notebook)
        output2 = results2[0].outputs[0]["text"]

        assert output1 == output2
