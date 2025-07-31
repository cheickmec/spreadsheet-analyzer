"""
Tests for the ExecutionBridge module.

This test suite validates the execution bridge functionality including:
- Orchestration between notebooks and kernels
- Sequential cell execution with state management
- Output attachment and formatting
- Error handling and recovery
- Data persistence with scrapbook
"""

import pytest

from spreadsheet_analyzer.core_exec import (
    ExecutionBridge,
    KernelProfile,
    KernelService,
    NotebookBuilder,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_simple_code_notebook() -> None:
    """Test executing a notebook with simple code cells."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_markdown_cell("# Simple Calculation")
    notebook.add_code_cell("x = 5")
    notebook.add_code_cell("y = 10")
    notebook.add_code_cell("result = x + y; print(f'Result: {result}')")

    async with KernelService(profile) as service:
        session_id = await service.create_session("calc-session")
        bridge = ExecutionBridge(service)
        results = await bridge.execute_notebook(session_id, notebook)

        assert len(results) == 3
        final_result = results[-1]
        assert final_result.status == "ok"
        assert any("Result: 15" in str(o) for o in final_result.outputs)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_notebook_with_error() -> None:
    """Test executing a notebook with code that produces errors."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_code_cell("good_code = 'this works'")
    notebook.add_code_cell("1 / 0")
    notebook.add_code_cell("print('This should not execute if stop_on_error=True')")

    async with KernelService(profile) as service:
        session_id = await service.create_session("error-session")
        bridge = ExecutionBridge(service)

        with pytest.raises(RuntimeError):
            await bridge.execute_notebook(session_id, notebook, stop_on_error=True)

        # Test without stop_on_error
        results = await bridge.execute_notebook(session_id, notebook, stop_on_error=False)
        assert len(results) == 3
        assert results[0].status == "ok"
        assert results[1].status == "error"
        assert "ZeroDivisionError" in results[1].error.get("ename", "")
        assert results[2].status == "ok"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_scrapbook_persistence() -> None:
    """Test data persistence using scrapbook."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_code_cell("my_data = {'a': 1, 'b': 'hello'}")

    async with KernelService(profile) as service:
        session_id = await service.create_session("scrapbook-session")
        bridge = ExecutionBridge(service, enable_persistence=True)

        # First, define the data in the kernel
        await bridge.execute_cell(session_id, "my_data = {'a': 1, 'b': 'hello'}")

        # Test retrieval of the persisted data
        retrieved_data = await bridge.get_persisted_data(session_id, "my_data")
        assert "'a': 1" in retrieved_data
        assert "'b': 'hello'" in retrieved_data


@pytest.mark.slow
@pytest.mark.asyncio
async def test_streaming_execution() -> None:
    """Test streaming execution of a notebook."""
    profile = KernelProfile()
    notebook = NotebookBuilder()
    notebook.add_code_cell("import time; print('start'); time.sleep(0.1); print('end')")

    async with KernelService(profile) as service:
        session_id = await service.create_session("streaming-session")
        bridge = ExecutionBridge(service)

        results = []
        async for result in bridge.execute_notebook_streaming(session_id, notebook):
            results.append(result)

        assert len(results) == 1
        assert results[0].status == "ok"
        outputs_str = "".join(str(o) for o in results[0].outputs)
        assert "start" in outputs_str
        assert "end" in outputs_str
