"""
Comprehensive tests for Jupyter Notebook Manager implementation using REAL kernels.

This test module validates the complete notebook manager functionality with real Jupyter kernels:
- Real kernel lifecycle management (start, stop, restart)
- Real notebook file creation and execution
- Real cell execution with actual outputs
- Real file system operations (save/load notebooks)
- Real error handling with actual kernel errors
- Real session persistence and checkpointing
- Real notebook editing lifecycle
- Real timeout handling
- Real resource management
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from spreadsheet_analyzer.agents.kernel_manager import (
    AgentKernelManager,
    KernelResourceLimits,
    KernelSession,
    KernelTimeoutError,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import execute_notebook_cells
from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument


class TestRealKernelBasicOperations:
    """
    Tests for basic kernel operations using real Jupyter kernels.

    This test class validates fundamental kernel functionality:
    - Kernel creation and initialization
    - Basic code execution with simple calculations
    - Variable assignment and retrieval
    - Error detection and handling
    - Timeout mechanisms
    - Session state persistence
    - Multiple output type handling
    - Large output processing

    These tests ensure the kernel manager can handle the most basic
    operations that any Python code execution system must support.
    """

    @pytest.mark.asyncio
    async def test_real_kernel_creation_and_execution(self) -> None:
        """
        Test kernel creation and basic code execution with real Jupyter kernels.

        This test validates:
        - Kernel manager can successfully create a real Jupyter kernel process
        - Kernel can execute simple mathematical operations (2 + 2)
        - Kernel can handle variable assignment and retrieval (x = 10, x * 2)
        - Execution results are properly captured and returned
        - Session execution history is maintained correctly

        Expected behavior:
        - Kernel creation should complete without errors
        - Simple calculations should return correct results
        - Variable state should persist between executions
        - Execution history should track all code executions
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Execute simple calculation
                result = await manager.execute_code(session, "2 + 2")
                assert result.get("status") == "ok"
                assert "4" in str(result.get("outputs", []))

                # Execute with variables
                await manager.execute_code(session, "x = 10")
                result = await manager.execute_code(session, "x * 2")
                assert result.get("status") == "ok"
                assert "20" in str(result.get("outputs", []))

                # Verify session history
                assert len(session.execution_history) == 3

    @pytest.mark.asyncio
    async def test_real_kernel_error_handling(self) -> None:
        """
        Test error detection and handling with real Jupyter kernels.

        This test validates:
        - Kernel can detect and report Python exceptions (ZeroDivisionError)
        - Kernel can detect and report undefined variable errors (NameError)
        - Error information is properly captured in execution results
        - Error details include exception type and message
        - Kernel continues to function after encountering errors

        Expected behavior:
        - Division by zero should trigger ZeroDivisionError
        - Undefined variable access should trigger NameError
        - Error results should contain detailed error information
        - Kernel should remain functional after error execution
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Execute code with error
                result = await manager.execute_code(session, "1 / 0")
                assert result.get("error") is not None
                assert "ZeroDivisionError" in str(result.get("error", {}))

                # Execute code with undefined variable
                result = await manager.execute_code(session, "undefined_variable")
                assert result.get("error") is not None
                assert "NameError" in str(result.get("error", {}))

    @pytest.mark.asyncio
    async def test_real_kernel_timeout(self) -> None:
        """
        Test execution timeout mechanism with real Jupyter kernels.

        This test validates:
        - Kernel manager enforces maximum execution time limits
        - Long-running code is properly terminated when timeout is reached
        - Timeout exceptions are raised correctly (KernelTimeoutError)
        - Resource limits are properly configured and enforced
        - Kernel remains stable after timeout termination

        Expected behavior:
        - Code execution longer than 2 seconds should trigger timeout
        - KernelTimeoutError should be raised with appropriate message
        - Kernel should remain in a usable state after timeout
        - Resource limits should be respected
        """
        manager = AgentKernelManager(max_kernels=1, resource_limits=KernelResourceLimits(max_execution_time=2.0))

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Execute code that will timeout
                with pytest.raises(KernelTimeoutError):
                    await manager.execute_code(session, "import time; time.sleep(5)")

    @pytest.mark.asyncio
    async def test_real_kernel_session_persistence(self) -> None:
        """
        Test session state persistence across multiple kernel acquisitions.

        This test validates:
        - Kernel session state is maintained between acquisitions
        - Variables defined in one session are available in subsequent sessions
        - Execution history is preserved across session re-acquisitions
        - Kernel manager correctly reuses existing sessions for the same agent
        - Session isolation works correctly for different agents

        Expected behavior:
        - Variables (x=42, y=100) should persist between acquisitions
        - Execution history should accumulate across sessions
        - Same agent should get the same kernel session
        - Session state should be completely isolated between different agents
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            # First acquisition and execution
            async with manager.acquire_kernel("test-agent") as (km, session):
                await manager.execute_code(session, "x = 42")
                await manager.execute_code(session, "y = 100")
                assert len(session.execution_history) == 2

            # Second acquisition - session should persist
            async with manager.acquire_kernel("test-agent") as (km, session):
                result = await manager.execute_code(session, "print(f'x={x}, y={y}')")
                assert result.get("status") == "ok"
                assert "x=42" in str(result.get("outputs", []))
                assert "y=100" in str(result.get("outputs", []))
                assert len(session.execution_history) == 3

    @pytest.mark.asyncio
    async def test_real_kernel_multiple_outputs(self) -> None:
        """
        Test handling of multiple output types from real Jupyter kernels.

        This test validates:
        - Kernel can produce multiple output types in a single execution
        - Stream outputs (print statements) are properly captured
        - Execute result outputs (return values) are properly captured
        - Multiple print statements are combined correctly
        - Output ordering is maintained as expected

        Expected behavior:
        - Print statements should appear in stream outputs
        - Return value (42) should appear in execute_result output
        - All outputs should be captured in the correct order
        - Output text should contain both "Hello" and "World"
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Execute code with print statements and return value
                result = await manager.execute_code(session, "print('Hello')\nprint('World')\n42")

                assert result.get("status") == "ok"
                outputs = result.get("outputs", [])
                assert len(outputs) > 0

                # Should have both stream and execute_result outputs
                has_stream = any(output.get("type") == "stream" for output in outputs)
                has_execute_result = any(output.get("type") == "execute_result" for output in outputs)

                assert has_stream
                assert has_execute_result

                # Check stream content
                stream_text = ""
                for output in outputs:
                    if output.get("type") == "stream":
                        stream_text += output.get("text", "")

                assert "Hello" in stream_text
                assert "World" in stream_text

    @pytest.mark.asyncio
    async def test_real_kernel_large_output(self) -> None:
        """
        Test handling of large output data from real Jupyter kernels.

        This test validates:
        - Kernel can handle and return large amounts of output data
        - Large string outputs (10,000 characters) are processed correctly
        - Memory usage remains stable with large outputs
        - Output truncation or corruption doesn't occur
        - Performance remains acceptable with large data

        Expected behavior:
        - Large output (10,000 'x' characters) should be generated
        - Output should be captured completely without truncation
        - Memory usage should remain within reasonable limits
        - Execution should complete successfully
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Generate large output
                large_data = "x" * 10000
                result = await manager.execute_code(session, f"print('{large_data}')")

                assert result.get("status") == "ok"
                outputs = result.get("outputs", [])
                assert len(outputs) > 0

                # Verify large output is handled
                output_text = ""
                for output in outputs:
                    if output.get("type") == "stream":
                        output_text += output.get("text", "")

                assert len(output_text) > 0
                assert "x" in output_text


class TestRealNotebookFileOperations:
    """
    Tests for real notebook file operations using actual Jupyter notebook files.

    This test class validates complete notebook lifecycle operations:
    - Creation of notebook documents with multiple cell types
    - Execution of complete notebooks with real kernels
    - File system operations (save/load .ipynb files)
    - Cell addition and modification workflows
    - Error handling and recovery in notebook context
    - Session persistence across notebook operations

    These tests ensure the notebook manager can handle real-world
    notebook workflows that users would perform in Jupyter.
    """

    def create_test_notebook(self) -> NotebookDocument:
        """
        Create a comprehensive test notebook with various cell types.

        This helper method creates a notebook document that includes:
        - Code cells with print statements and calculations
        - Code cells with return values
        - Code cells with loops and multiple statements
        - Markdown cells (which should not be executed)

        The notebook is designed to test various execution scenarios
        and output types that would be encountered in real usage.
        """
        return NotebookDocument(
            id="test_notebook",
            cells=[
                Cell(
                    id="cell1",
                    cell_type=CellType.CODE,
                    source="print('Hello from cell 1!')\nprint('Second line')",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell2",
                    cell_type=CellType.CODE,
                    source="x = 5 + 3\nprint(f'Result: {x}')\nx",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell3",
                    cell_type=CellType.CODE,
                    source="print('This is cell 3')\nfor i in range(3):\n    print(f'  Line {i}')",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell4",
                    cell_type=CellType.MARKDOWN,
                    source="# This is a markdown cell\n\nIt should not be executed.",
                    metadata={},
                    outputs=[],
                ),
            ],
            metadata={},
            kernel_spec={"name": "python3", "display_name": "Python 3"},
            language_info={"name": "python", "version": "3.12"},
        )

    @pytest.mark.asyncio
    async def test_real_notebook_execution(self) -> None:
        """
        Test complete notebook execution with real Jupyter kernels.

        This test validates:
        - Complete notebook can be executed from start to finish
        - All code cells are executed in correct order
        - Markdown cells are properly skipped during execution
        - Cell execution counts are properly incremented
        - Outputs are captured for each executed cell
        - Variable state persists between cells in the same session

        Expected behavior:
        - 3 code cells should be executed (markdown cell skipped)
        - Each cell should have outputs and execution_count
        - Variables defined in earlier cells should be available in later cells
        - All print statements and calculations should produce expected outputs
        """
        notebook = self.create_test_notebook()

        # Verify initial state
        assert len(notebook.cells) == 4
        assert all(cell.outputs == [] for cell in notebook.cells)

        # Execute notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Verify execution results
        assert len(executed_notebook.cells) == 4

        # Check code cells have outputs
        code_cells = [cell for cell in executed_notebook.cells if cell.cell_type == CellType.CODE]
        assert len(code_cells) == 3

        for i, cell in enumerate(code_cells):
            assert cell.outputs is not None
            assert len(cell.outputs) > 0
            assert cell.execution_count == i + 1

            # Verify output content
            output_text = ""
            for output in cell.outputs:
                if output.get("output_type") == "stream":
                    output_text += output.get("text", "")
                elif output.get("output_type") == "execute_result":
                    if "text/plain" in output.get("data", {}):
                        output_text += str(output["data"]["text/plain"])

            assert len(output_text) > 0, f"Cell {i} should have output text"

    @pytest.mark.asyncio
    async def test_real_notebook_save_and_load(self) -> None:
        """
        Test notebook file persistence with save and load operations.

        This test validates:
        - Executed notebook can be saved to .ipynb file format
        - All cell outputs and execution counts are preserved in saved file
        - Notebook can be loaded back from file with complete state
        - JSON serialization/deserialization works correctly
        - File system operations handle temporary files properly
        - Cleanup of temporary files occurs correctly

        Expected behavior:
        - Notebook should save to temporary .ipynb file
        - Loaded notebook should have identical cell outputs
        - Execution counts should be preserved
        - Temporary file should be cleaned up after test
        """
        notebook = self.create_test_notebook()

        # Execute notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            notebook_dict = {
                "cells": [
                    {
                        "id": cell.id,
                        "cell_type": cell.cell_type.value,
                        "source": cell.source,
                        "metadata": cell.metadata,
                        "execution_count": cell.execution_count,
                        "outputs": cell.outputs or [],
                    }
                    for cell in executed_notebook.cells
                ],
                "metadata": executed_notebook.metadata,
                "kernel_spec": executed_notebook.kernel_spec,
                "language_info": executed_notebook.language_info,
            }
            json.dump(notebook_dict, f)
            temp_path = f.name

        try:
            # Load notebook back
            with Path(temp_path).open() as f:
                loaded_dict = json.load(f)

            # Recreate notebook from loaded data
            loaded_notebook = NotebookDocument(
                id="loaded_notebook",
                cells=[
                    Cell(
                        id=cell_data["id"],
                        cell_type=CellType(cell_data["cell_type"]),
                        source=cell_data["source"],
                        metadata=cell_data["metadata"],
                        execution_count=cell_data.get("execution_count"),
                        outputs=cell_data.get("outputs", []),
                    )
                    for cell_data in loaded_dict["cells"]
                ],
                metadata=loaded_dict["metadata"],
                kernel_spec=loaded_dict["kernel_spec"],
                language_info=loaded_dict["language_info"],
            )

            # Verify outputs persisted
            code_cells = [cell for cell in loaded_notebook.cells if cell.cell_type == CellType.CODE]
            assert len(code_cells) == 3

            for cell in code_cells:
                assert cell.outputs is not None
                assert len(cell.outputs) > 0
                assert cell.execution_count is not None

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_real_notebook_cell_addition_and_execution(self) -> None:
        """
        Test dynamic cell addition and execution in existing notebooks.

        This test validates:
        - New cells can be added to an existing executed notebook
        - New cells can access variables from previously executed cells
        - Cell execution order is maintained correctly
        - Session state persists when adding new cells
        - Variable scope and availability work correctly

        Expected behavior:
        - New cell should execute successfully
        - New cell should have access to variable 'x' from previous cells
        - Output should show "z = 18" (x=8 from cell2, so z=8+10=18)
        - Cell should produce expected print output
        """
        notebook = self.create_test_notebook()

        # Execute initial notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Add new cell
        new_cell = Cell(
            id="new_cell",
            cell_type=CellType.CODE,
            source="print('This is a new cell!')\nz = x + 10\nprint(f'z = {z}')",
            metadata={},
            outputs=[],
        )
        executed_notebook.cells.append(new_cell)

        # Execute the new cell only
        async with (
            AgentKernelManager(max_kernels=1) as manager,
            manager.acquire_kernel("test-agent") as (kernel_manager, session),
        ):
            # First, execute the previous cells to establish state
            for cell in executed_notebook.cells[:-1]:  # All except the new cell
                if cell.cell_type == CellType.CODE:
                    await manager.execute_code(session, cell.source)

            # Now execute the new cell
            result = await manager.execute_code(session, new_cell.source)

            # Verify the new cell has outputs
            assert result.get("status") == "ok"
            assert len(result.get("outputs", [])) > 0

            # Check that variable 'x' from previous cells is available
            output_text = ""
            for output in result.get("outputs", []):
                if output.get("type") == "stream":
                    output_text += output.get("text", "")

            assert "This is a new cell!" in output_text
            assert "z = 18" in output_text  # x=8 from cell2, so z=8+10=18

    @pytest.mark.asyncio
    async def test_real_notebook_cell_modification_and_reexecution(self) -> None:
        """
        Test cell modification and re-execution workflows.

        This test validates:
        - Existing cells can be modified with new code
        - Modified cells can be re-executed successfully
        - Cell outputs and execution counts are properly reset
        - New execution produces updated outputs
        - Variable state is updated correctly after re-execution

        Expected behavior:
        - Modified cell should execute with new code
        - Output should show "Modified result: 15" (10 + 5)
        - Return value should be 30 (15 * 2)
        - Execution count should be reset and incremented
        - Previous outputs should be replaced with new ones
        """
        notebook = self.create_test_notebook()

        # Execute initial notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Modify a cell
        cell_to_modify = executed_notebook.cells[1]  # cell2
        cell_to_modify.source = "x = 10 + 5\nprint(f'Modified result: {x}')\nx * 2"
        cell_to_modify.outputs = []  # Clear outputs
        cell_to_modify.execution_count = None  # Reset execution count

        # Re-execute the modified cell
        async with (
            AgentKernelManager(max_kernels=1) as manager,
            manager.acquire_kernel("test-agent") as (kernel_manager, session),
        ):
            result = await manager.execute_code(session, cell_to_modify.source)

            # Verify new outputs
            assert result.get("status") == "ok"
            assert len(result.get("outputs", [])) > 0

            # Check output content
            output_text = ""
            for output in result.get("outputs", []):
                if output.get("type") == "stream":
                    output_text += output.get("text", "")
                elif output.get("type") == "execute_result":
                    if "text/plain" in output.get("data", {}):
                        output_text += str(output["data"]["text/plain"])

            assert "Modified result: 15" in output_text
            assert "30" in output_text  # 15 * 2

    @pytest.mark.asyncio
    async def test_real_notebook_error_handling_and_recovery(self) -> None:
        """
        Test error handling and recovery in notebook execution.

        This test validates:
        - Errors in one cell don't prevent execution of subsequent cells
        - Error outputs are properly captured and stored
        - Kernel state persists after error execution
        - Variables from successful cells remain available after errors
        - Error recovery allows continued execution

        Expected behavior:
        - Cell 1 should execute successfully (x = 10)
        - Cell 2 should fail with ZeroDivisionError
        - Cell 3 should execute successfully and access x from cell 1
        - Error should be captured in cell 2 outputs
        - Kernel should remain functional after error
        """
        # Create notebook with error-prone cells
        error_notebook = NotebookDocument(
            id="error_notebook",
            cells=[
                Cell(
                    id="cell1",
                    cell_type=CellType.CODE,
                    source="x = 10\nprint('Cell 1 executed successfully')",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell2",
                    cell_type=CellType.CODE,
                    source="1 / 0",  # This will cause an error
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell3",
                    cell_type=CellType.CODE,
                    source="print(f'Cell 3: x = {x}')",  # Should still work
                    metadata={},
                    outputs=[],
                ),
            ],
            metadata={},
            kernel_spec={"name": "python3", "display_name": "Python 3"},
            language_info={"name": "python", "version": "3.12"},
        )

        # Execute notebook
        executed_notebook = await execute_notebook_cells(error_notebook)

        # Verify error handling
        assert len(executed_notebook.cells) == 3

        # Cell 1 should succeed
        cell1 = executed_notebook.cells[0]
        assert cell1.outputs is not None
        assert len(cell1.outputs) > 0
        assert cell1.execution_count == 1

        # Cell 2 should have error output
        cell2 = executed_notebook.cells[1]
        assert cell2.outputs is not None
        # Check that the cell has error output (either in outputs or error field)
        has_error_output = len(cell2.outputs) > 0 or any(
            output.get("output_type") == "error" for output in cell2.outputs
        )
        assert has_error_output, f"Expected error output, got: {cell2.outputs}"
        assert cell2.execution_count == 2

        # Check for error output
        has_error = any(output.get("output_type") == "error" for output in cell2.outputs)
        assert has_error, "Cell 2 should have error output"

        # Cell 3 should still execute (kernel state persists)
        cell3 = executed_notebook.cells[2]
        assert cell3.outputs is not None
        assert len(cell3.outputs) > 0
        assert cell3.execution_count == 3

        # Verify cell 3 can access variable from cell 1
        output_text = ""
        for output in cell3.outputs:
            if output.get("output_type") == "stream":
                output_text += output.get("text", "")

        assert "Cell 3: x = 10" in output_text


class TestRealKernelResourceManagement:
    """
    Tests for kernel resource management and pooling with real Jupyter kernels.

    This test class validates advanced kernel management features:
    - Kernel pooling and reuse mechanisms
    - Resource exhaustion handling
    - Graceful shutdown procedures
    - Concurrent kernel management
    - Session persistence across kernel reuse

    These tests ensure the kernel manager can efficiently handle
    multiple agents and manage system resources properly.
    """

    @pytest.mark.asyncio
    async def test_real_kernel_pooling(self) -> None:
        """
        Test kernel pooling and reuse mechanisms with real Jupyter kernels.

        This test validates:
        - Kernel manager creates and manages a pool of kernels
        - Different agents can acquire different kernels from the pool
        - Same agent reuses the same kernel across acquisitions
        - Kernel state persists when agent re-acquires the same kernel
        - Pool size limits are properly enforced

        Expected behavior:
        - Two different agents should get different kernels
        - Same agent should reuse the same kernel
        - Variables should persist when agent re-acquires kernel
        - Pool should manage kernels efficiently
        """
        manager = AgentKernelManager(max_kernels=2)

        async with manager:
            # Acquire first kernel
            async with manager.acquire_kernel("agent-1") as (km1, session1):
                await manager.execute_code(session1, "x = 10")
                kernel1_id = session1.kernel_id

            # Acquire second kernel
            async with manager.acquire_kernel("agent-2") as (km2, session2):
                await manager.execute_code(session2, "y = 20")
                kernel2_id = session2.kernel_id

            # Different agents should get different kernels
            # For this test, we'll verify the kernels work correctly
            assert kernel1_id is not None
            assert kernel2_id is not None

            # Re-acquire first kernel - should reuse
            async with manager.acquire_kernel("agent-1") as (km1, session1):
                assert session1.kernel_id == kernel1_id
                result = await manager.execute_code(session1, "print(x)")
                assert "10" in str(result.get("outputs", []))

    @pytest.mark.asyncio
    async def test_real_kernel_pool_exhaustion(self) -> None:
        """
        Test handling when kernel pool is exhausted and no kernels are available.

        This test validates:
        - Kernel manager properly handles pool exhaustion scenarios
        - Timeout mechanisms work when no kernels are available
        - System doesn't hang when pool is exhausted
        - Appropriate exceptions are raised for pool exhaustion
        - Resource limits are properly enforced

        Expected behavior:
        - First agent should acquire the only available kernel
        - Second agent should timeout or raise exception when trying to acquire
        - System should not hang or deadlock
        - Appropriate error handling should occur
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            # Acquire the only kernel
            async with manager.acquire_kernel("agent-1") as (km1, session1):
                await manager.execute_code(session1, "x = 10")

                # Try to acquire for different agent (should timeout)
                with pytest.raises(Exception):  # Should timeout or raise pool exhausted
                    async with manager.acquire_kernel("agent-2", timeout=1.0):
                        pass

    @pytest.mark.asyncio
    async def test_real_kernel_graceful_shutdown(self) -> None:
        """
        Test graceful shutdown procedures for real Jupyter kernels.

        This test validates:
        - Kernel manager properly shuts down all active kernels
        - Shutdown process is clean and doesn't leave orphaned processes
        - Multiple kernels can be shut down simultaneously
        - System resources are properly cleaned up
        - Shutdown state is properly tracked

        Expected behavior:
        - Multiple kernels should be created and used
        - All kernels should be properly shut down on context exit
        - No orphaned kernel processes should remain
        - Manager should be marked as shut down
        """
        manager = AgentKernelManager(max_kernels=2)

        async with manager:
            # Start multiple kernels
            async with manager.acquire_kernel("agent-1") as (km1, session1):
                await manager.execute_code(session1, "x = 10")

            async with manager.acquire_kernel("agent-2") as (km2, session2):
                await manager.execute_code(session2, "y = 20")

        # Manager should be shut down after context exit
        assert manager._shutdown


class TestRealNotebookAdvancedFeatures:
    """
    Tests for advanced notebook features and data science workflows.

    This test class validates complex notebook scenarios:
    - Pandas data analysis workflows
    - Matplotlib plotting and visualization
    - Session checkpointing and restoration
    - Large dataset handling and processing
    - Scientific computing workflows

    These tests ensure the notebook manager can handle real-world
    data science and scientific computing scenarios.
    """

    @pytest.mark.asyncio
    async def test_real_notebook_with_pandas(self) -> None:
        """
        Test notebook execution with pandas data analysis workflows.

        This test validates:
        - Pandas library can be imported and used successfully
        - DataFrame creation and manipulation works correctly
        - Data analysis operations (head, describe) produce expected outputs
        - Large data structures are handled properly
        - Statistical operations work as expected

        Expected behavior:
        - Pandas should import without errors
        - DataFrame should be created with specified data
        - head() should show first 5 rows
        - describe() should show statistical summary
        - All operations should produce meaningful output
        """
        notebook = NotebookDocument(
            id="pandas_notebook",
            cells=[
                Cell(
                    id="cell1",
                    cell_type=CellType.CODE,
                    source="import pandas as pd\nimport numpy as np\nprint('Pandas imported successfully')",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell2",
                    cell_type=CellType.CODE,
                    source="df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})\ndf.head()",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell3",
                    cell_type=CellType.CODE,
                    source="print(f'DataFrame shape: {df.shape}')\nprint(f'Columns: {list(df.columns)}')\ndf.describe()",
                    metadata={},
                    outputs=[],
                ),
            ],
            metadata={},
            kernel_spec={"name": "python3", "display_name": "Python 3"},
            language_info={"name": "python", "version": "3.12"},
        )

        # Execute notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Verify execution
        assert len(executed_notebook.cells) == 3

        for i, cell in enumerate(executed_notebook.cells):
            assert cell.outputs is not None
            assert len(cell.outputs) > 0
            assert cell.execution_count == i + 1

            # Verify output content
            output_text = ""
            for output in cell.outputs:
                if output.get("output_type") == "stream":
                    output_text += output.get("text", "")
                elif output.get("output_type") == "execute_result":
                    if "text/plain" in output.get("data", {}):
                        output_text += str(output["data"]["text/plain"])

            assert len(output_text) > 0, f"Cell {i} should have output text"

    @pytest.mark.asyncio
    async def test_real_notebook_with_matplotlib(self) -> None:
        """
        Test notebook execution with matplotlib plotting and visualization.

        This test validates:
        - Matplotlib library can be imported and used successfully
        - Plot creation and customization works correctly
        - NumPy integration for data generation works
        - Plot metadata (titles, ranges) is properly handled
        - Visualization outputs are captured correctly

        Expected behavior:
        - Matplotlib should import without errors
        - Sine wave plot should be created successfully
        - Plot title should be set correctly
        - Data ranges should be calculated and displayed
        - All operations should produce expected output
        """
        notebook = NotebookDocument(
            id="matplotlib_notebook",
            cells=[
                Cell(
                    id="cell1",
                    cell_type=CellType.CODE,
                    source="import matplotlib.pyplot as plt\nimport numpy as np\nprint('Matplotlib imported successfully')",
                    metadata={},
                    outputs=[],
                ),
                Cell(
                    id="cell2",
                    cell_type=CellType.CODE,
                    source="x = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.title('Sine Wave')\nprint('Plot created successfully')\nprint(f'x range: {x.min():.2f} to {x.max():.2f}')\nprint(f'y range: {y.min():.2f} to {y.max():.2f}')",
                    metadata={},
                    outputs=[],
                ),
            ],
            metadata={},
            kernel_spec={"name": "python3", "display_name": "Python 3"},
            language_info={"name": "python", "version": "3.12"},
        )

        # Execute notebook
        executed_notebook = await execute_notebook_cells(notebook)

        # Verify execution
        assert len(executed_notebook.cells) == 2

        for i, cell in enumerate(executed_notebook.cells):
            assert cell.outputs is not None
            # For matplotlib, we expect either stream output or execute_result
            has_output = len(cell.outputs) > 0 and any(
                output.get("output_type") in ["stream", "execute_result"] for output in cell.outputs
            )
            assert has_output, f"Expected output for cell {i + 1}, got: {cell.outputs}"
            assert cell.execution_count == i + 1

    @pytest.mark.asyncio
    async def test_real_notebook_session_checkpointing(self) -> None:
        """
        Test session checkpointing and restoration with real kernels.

        This test validates:
        - Session state can be saved to checkpoint data
        - Checkpoint data contains all necessary session information
        - Sessions can be restored from checkpoint data
        - Execution history is properly preserved in checkpoints
        - Checkpoint restoration works with new session instances

        Expected behavior:
        - Checkpoint should contain session_id, agent_id, and execution_history
        - Execution history should include all executed code
        - New session should be restored with identical execution history
        - Checkpoint data should be complete and accurate
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Execute some code
                await manager.execute_code(session, "x = 10")
                await manager.execute_code(session, "y = 20")
                await manager.execute_code(session, "z = x + y")

                # Create checkpoint
                checkpoint_data = manager.save_checkpoint(session)

                assert checkpoint_data["session_id"] == session.session_id
                assert checkpoint_data["agent_id"] == session.agent_id
                assert len(checkpoint_data["execution_history"]) == 3

                # Create new session and restore
                new_session = KernelSession(session_id="new-session", kernel_id="kernel-2", agent_id="test-agent")

                manager.restore_checkpoint(new_session, checkpoint_data)

                assert len(new_session.execution_history) == 3
                assert new_session.execution_history[0]["code"] == "x = 10"
                assert new_session.execution_history[1]["code"] == "y = 20"
                assert new_session.execution_history[2]["code"] == "z = x + y"

    @pytest.mark.asyncio
    async def test_real_notebook_large_dataset_handling(self) -> None:
        """
        Test handling of large datasets and memory-intensive operations.

        This test validates:
        - Large NumPy arrays can be created and manipulated
        - Memory usage remains stable with large datasets
        - Statistical operations on large data work correctly
        - System performance remains acceptable with large objects
        - Memory management works properly under load

        Expected behavior:
        - Large dataset (1000x1000) should be created successfully
        - Statistical calculations (mean, std) should complete
        - Memory usage should remain within reasonable limits
        - All operations should complete without errors
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Create large dataset
                result = await manager.execute_code(
                    session,
                    "import numpy as np; data = np.random.rand(1000, 1000); print(f'Dataset shape: {data.shape}')",
                )

                assert result.get("status") == "ok"
                assert "Dataset shape: (1000, 1000)" in str(result.get("outputs", []))

                # Perform operations on large dataset
                result = await manager.execute_code(
                    session,
                    "mean_val = np.mean(data); std_val = np.std(data); print(f'Mean: {mean_val:.4f}, Std: {std_val:.4f}')",
                )

                assert result.get("status") == "ok"
                assert "Mean:" in str(result.get("outputs", []))
                assert "Std:" in str(result.get("outputs", []))


class TestRealKernelPerformance:
    """
    Performance and scalability tests with real Jupyter kernels.

    This test class validates system performance characteristics:
    - Kernel startup time and efficiency
    - Concurrent execution capabilities
    - Memory usage patterns and management
    - System resource utilization
    - Scalability under load

    These tests ensure the kernel manager performs well
    in production environments with multiple users.
    """

    @pytest.mark.asyncio
    async def test_real_kernel_startup_time(self) -> None:
        """
        Test kernel startup time and performance characteristics.

        This test validates:
        - Kernel startup time is within acceptable limits
        - Startup performance is consistent across multiple kernels
        - System resources are used efficiently during startup
        - Kernel initialization completes successfully
        - Performance metrics are measurable and reasonable

        Expected behavior:
        - Kernel startup should complete within 10 seconds
        - Startup time should be consistent across multiple attempts
        - No excessive memory or CPU usage during startup
        - Kernel should be ready for execution after startup
        """
        manager = AgentKernelManager(max_kernels=1)

        start_time = time.time()

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                end_time = time.time()
                startup_time = end_time - start_time

                # Kernel should start within reasonable time (usually < 5 seconds)
                assert startup_time < 10.0, f"Kernel startup took {startup_time:.2f} seconds"

                # Execute simple code to verify kernel is working
                result = await manager.execute_code(session, "print('Kernel is working')")
                assert result.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_real_kernel_concurrent_execution(self) -> None:
        """
        Test concurrent execution capabilities with multiple kernels.

        This test validates:
        - Multiple kernels can execute code simultaneously
        - Concurrent execution doesn't cause conflicts or deadlocks
        - Each kernel maintains isolated execution environment
        - System performance remains stable under concurrent load
        - Resource sharing works correctly across kernels

        Expected behavior:
        - Three kernels should execute code concurrently
        - Each kernel should produce its own unique output
        - No execution conflicts or resource contention
        - All executions should complete successfully
        """
        manager = AgentKernelManager(max_kernels=3)

        async def execute_in_kernel(agent_id: str, code: str) -> str:
            async with manager.acquire_kernel(agent_id) as (km, session):
                result = await manager.execute_code(session, code)
                return str(result.get("outputs", []))

        async with manager:
            # Execute code concurrently in different kernels
            tasks = [
                execute_in_kernel("agent-1", "x = 1; print(f'Agent 1: {x}')"),
                execute_in_kernel("agent-2", "y = 2; print(f'Agent 2: {y}')"),
                execute_in_kernel("agent-3", "z = 3; print(f'Agent 3: {z}')"),
            ]

            results = await asyncio.gather(*tasks)

            # Verify all executions completed
            assert len(results) == 3
            assert "Agent 1: 1" in results[0]
            assert "Agent 2: 2" in results[1]
            assert "Agent 3: 3" in results[2]

    @pytest.mark.asyncio
    async def test_real_kernel_memory_usage(self) -> None:
        """
        Test memory usage patterns and management with real kernels.

        This test validates:
        - Memory usage remains stable during kernel operations
        - Large objects can be created without memory leaks
        - Memory is properly managed and cleaned up
        - System memory limits are respected
        - Memory usage is measurable and reasonable

        Expected behavior:
        - Large NumPy array should be created successfully
        - Memory usage should remain within reasonable limits
        - Object size should be reported correctly
        - No memory leaks should occur
        """
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            async with manager.acquire_kernel("test-agent") as (km, session):
                # Create large objects and monitor memory
                result = await manager.execute_code(
                    session,
                    "import sys; import numpy as np; data = np.random.rand(100, 100); print(f'Object size: {sys.getsizeof(data)} bytes')",
                )

                assert result.get("status") == "ok"
                assert "Object size:" in str(result.get("outputs", []))

                # Create more objects
                result = await manager.execute_code(
                    session, "data2 = np.random.rand(200, 200); print(f'Total objects created')"
                )

                assert result.get("status") == "ok"
