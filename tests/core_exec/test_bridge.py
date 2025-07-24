"""
Tests for the ExecutionBridge module.

This test suite validates the execution bridge functionality including:
- ExecutionStats data structure
- ExecutionBridge orchestration between notebooks and kernels
- Sequential cell execution with state management
- Output attachment and formatting
- Error handling and recovery
- Execution count synchronization

Following TDD principles with functional tests - no mocking used.
All tests use real KernelService and NotebookBuilder integration.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from spreadsheet_analyzer.core_exec.bridge import (
    ExecutionBridge,
    ExecutionStats,
)
from spreadsheet_analyzer.core_exec.kernel_service import (
    KernelService,
    KernelProfile,
)
from spreadsheet_analyzer.core_exec.notebook_builder import (
    NotebookBuilder,
    CellType,
)
from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO


class TestExecutionStats:
    """Test ExecutionStats data structure."""

    def test_execution_stats_creation(self) -> None:
        """Test creating ExecutionStats with all fields."""
        stats = ExecutionStats(
            total_cells=10,
            executed_cells=8,
            skipped_cells=1,
            error_cells=1,
            total_duration_seconds=45.5
        )
        
        assert stats.total_cells == 10
        assert stats.executed_cells == 8
        assert stats.skipped_cells == 1
        assert stats.error_cells == 1
        assert stats.total_duration_seconds == 45.5

    def test_execution_stats_defaults(self) -> None:
        """Test ExecutionStats with default values."""
        stats = ExecutionStats(
            total_cells=0,
            executed_cells=0,
            skipped_cells=0,
            error_cells=0,
            total_duration_seconds=0.0
        )
        
        assert stats.total_cells == 0
        assert stats.executed_cells == 0
        assert stats.skipped_cells == 0
        assert stats.error_cells == 0
        assert stats.total_duration_seconds == 0.0


class TestExecutionBridge:
    """Test ExecutionBridge functionality."""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self) -> None:
        """Test creating an ExecutionBridge."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        assert bridge.kernel_service is kernel_service
        assert bridge._execution_callbacks == []

    @pytest.mark.asyncio
    async def test_execute_empty_notebook(self) -> None:
        """Test executing an empty notebook."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create empty notebook
            notebook = NotebookBuilder()
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should succeed with no executions
            assert result is notebook
            assert stats.total_cells == 0
            assert stats.executed_cells == 0
            assert stats.successful_cells == 0
            assert stats.failed_cells == 0

    @pytest.mark.asyncio
    async def test_execute_markdown_only_notebook(self) -> None:
        """Test executing a notebook with only markdown cells."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with markdown cells
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Introduction")
            notebook.add_markdown_cell("This is a description.")
            notebook.add_markdown_cell("## Conclusion")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should succeed without executing markdown cells
            assert result is notebook
            assert stats.total_cells == 3
            assert stats.executed_cells == 0  # Markdown cells not executed
            assert stats.successful_cells == 0
            assert stats.failed_cells == 0

    @pytest.mark.asyncio
    async def test_execute_simple_code_notebook(self) -> None:
        """Test executing a notebook with simple code cells."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with code cells
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Simple Calculation")
            notebook.add_code_cell("x = 5")
            notebook.add_code_cell("y = 10")
            notebook.add_code_cell("result = x + y\nprint(f'Result: {result}')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should execute all code cells
            assert result is notebook
            assert stats.total_cells == 4
            assert stats.executed_cells == 3  # 3 code cells
            assert stats.successful_cells == 3
            assert stats.failed_cells == 0
            assert stats.total_execution_time > 0
            
            # Check that outputs were attached
            code_cells = [cell for cell in notebook.cells if cell.cell_type == CellType.CODE]
            assert len(code_cells) == 3
            
            # Last cell should have output
            last_cell = code_cells[2]
            assert len(last_cell.outputs) > 0
            
            # Check for expected output
            output_found = False
            for output in last_cell.outputs:
                if isinstance(output, dict) and "Result: 15" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected 'Result: 15' in outputs: {last_cell.outputs}"

    @pytest.mark.asyncio
    async def test_execute_notebook_with_error(self) -> None:
        """Test executing a notebook with code that produces errors."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with error-producing code
            notebook = NotebookBuilder()
            notebook.add_code_cell("good_code = 'this works'")
            notebook.add_code_cell("1 / 0  # This will cause ZeroDivisionError")
            notebook.add_code_cell("print('This should still execute')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should execute all cells despite error
            assert result is notebook
            assert stats.total_cells == 3
            assert stats.executed_cells == 3
            assert stats.successful_cells == 2  # First and third cells succeed
            assert stats.failed_cells == 1     # Second cell fails
            
            # Check that error is captured in outputs
            error_cell = notebook.cells[1]
            assert len(error_cell.outputs) > 0
            
            # Should have error output
            error_found = False
            for output in error_cell.outputs:
                if (isinstance(output, dict) and 
                    output.get("output_type") == "error" and
                    "ZeroDivisionError" in str(output)):
                    error_found = True
                    break
            assert error_found, f"Expected ZeroDivisionError in outputs: {error_cell.outputs}"

    @pytest.mark.asyncio
    async def test_execute_notebook_variable_persistence(self) -> None:
        """Test that variables persist across cell executions."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with dependent cells
            notebook = NotebookBuilder()
            notebook.add_code_cell("data = [1, 2, 3, 4, 5]")
            notebook.add_code_cell("total = sum(data)")
            notebook.add_code_cell("average = total / len(data)")
            notebook.add_code_cell("print(f'Average: {average}')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # All cells should succeed
            assert stats.executed_cells == 4
            assert stats.successful_cells == 4
            assert stats.failed_cells == 0
            
            # Last cell should have correct output
            last_cell = notebook.cells[3]
            output_found = False
            for output in last_cell.outputs:
                if "Average: 3.0" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected 'Average: 3.0' in outputs: {last_cell.outputs}"

    @pytest.mark.asyncio
    async def test_execute_notebook_with_imports(self) -> None:
        """Test executing notebook with library imports."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with imports and usage
            notebook = NotebookBuilder()
            notebook.add_code_cell("import json\nimport math")
            notebook.add_code_cell("data = {'name': 'test', 'value': 42}")
            notebook.add_code_cell("json_str = json.dumps(data)")
            notebook.add_code_cell("sqrt_value = math.sqrt(data['value'])")
            notebook.add_code_cell("print(f'JSON: {json_str}')\nprint(f'Square root: {sqrt_value}')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # All cells should succeed
            assert stats.executed_cells == 5
            assert stats.successful_cells == 5
            assert stats.failed_cells == 0
            
            # Check outputs
            last_cell = notebook.cells[4]
            outputs_text = str(last_cell.outputs)
            assert "JSON:" in outputs_text
            assert "Square root:" in outputs_text

    @pytest.mark.asyncio
    async def test_execute_notebook_with_timeout(self) -> None:
        """Test executing notebook with timeout on slow cells."""
        # Create service with short timeout
        profile = KernelProfile(max_execution_time=1.0)
        kernel_service = KernelService(profile=profile)
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with slow-running code
            notebook = NotebookBuilder()
            notebook.add_code_cell("quick_result = 2 + 2")
            notebook.add_code_cell("import time\ntime.sleep(5)  # This will timeout")
            notebook.add_code_cell("final_result = 'should still execute'")
            
            # Execute notebook - should handle timeout gracefully
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should execute all cells, but second one should fail with timeout
            assert stats.executed_cells == 3
            assert stats.successful_cells == 2  # First and third succeed
            assert stats.failed_cells == 1     # Second fails with timeout

    @pytest.mark.asyncio
    async def test_execution_count_management(self) -> None:
        """Test that execution counts are properly managed during execution."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook with mixed cell types
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Start")
            notebook.add_code_cell("a = 1")
            notebook.add_markdown_cell("## Middle")
            notebook.add_code_cell("b = 2")
            notebook.add_code_cell("c = a + b")
            notebook.add_markdown_cell("## End")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Check execution counts are properly set
            code_cells = [cell for cell in notebook.cells if cell.cell_type == CellType.CODE]
            assert len(code_cells) == 3
            
            # Execution counts should be sequential
            assert code_cells[0].execution_count == 1
            assert code_cells[1].execution_count == 2
            assert code_cells[2].execution_count == 3
            
            # Markdown cells should not have execution counts
            markdown_cells = [cell for cell in notebook.cells if cell.cell_type == CellType.MARKDOWN]
            for cell in markdown_cells:
                assert cell.execution_count is None

    @pytest.mark.asyncio
    async def test_execution_callbacks(self) -> None:
        """Test execution callbacks for monitoring progress."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        # Track callback invocations
        callback_data = []
        
        def progress_callback(cell_index: int, cell_type: CellType, status: str, **kwargs) -> None:
            callback_data.append({
                'cell_index': cell_index,
                'cell_type': cell_type,
                'status': status,
                'kwargs': kwargs
            })
        
        bridge.add_execution_callback(progress_callback)
        
        async with kernel_service:
            # Create notebook
            notebook = NotebookBuilder()
            notebook.add_code_cell("x = 1")
            notebook.add_code_cell("y = 2")
            notebook.add_code_cell("print(x + y)")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should have received callbacks
            assert len(callback_data) >= 6  # At least start/end for each cell
            
            # Check callback structure
            start_callbacks = [cb for cb in callback_data if cb['status'] == 'start']
            end_callbacks = [cb for cb in callback_data if cb['status'] == 'end']
            
            assert len(start_callbacks) == 3
            assert len(end_callbacks) == 3
            
            # Check cell indices are correct
            for i, cb in enumerate(start_callbacks):
                assert cb['cell_index'] == i
                assert cb['cell_type'] == CellType.CODE

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self) -> None:
        """Test that memory usage is tracked during execution."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook that uses memory
            notebook = NotebookBuilder()
            notebook.add_code_cell("import sys")
            notebook.add_code_cell("large_list = list(range(100000))")
            notebook.add_code_cell("memory_usage = sys.getsizeof(large_list)")
            notebook.add_code_cell("print(f'Memory used: {memory_usage} bytes')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Should track some memory usage
            assert stats.memory_peak_mb >= 0
            assert stats.total_execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_and_save_notebook(self, tmp_path: Path) -> None:
        """Test executing notebook and saving result to file."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create notebook
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Data Analysis")
            notebook.add_code_cell("data = [1, 2, 3, 4, 5]")
            notebook.add_code_cell("mean = sum(data) / len(data)")
            notebook.add_code_cell("print(f'Mean: {mean}')")
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=30.0)
            
            # Save executed notebook
            output_path = tmp_path / "executed_notebook.ipynb"
            NotebookIO.write_notebook(result, output_path)
            
            # Read back and verify
            loaded_notebook = NotebookIO.read_notebook(output_path)
            
            # Should have outputs attached
            code_cells = [cell for cell in loaded_notebook.cells if cell.cell_type == CellType.CODE]
            
            # Last cell should have output with mean
            last_cell = code_cells[2]
            assert len(last_cell.outputs) > 0
            
            output_found = False
            for output in last_cell.outputs:
                if "Mean: 3.0" in str(output):
                    output_found = True
                    break
            assert output_found

    @pytest.mark.asyncio
    async def test_concurrent_notebook_execution(self) -> None:
        """Test executing multiple notebooks concurrently."""
        kernel_service = KernelService(max_concurrent_kernels=3)
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create multiple notebooks
            notebooks = []
            for i in range(3):
                notebook = NotebookBuilder()
                notebook.add_code_cell(f"notebook_id = {i}")
                notebook.add_code_cell(f"result = notebook_id * 10")
                notebook.add_code_cell("print(f'Notebook {notebook_id}: {result}')")
                notebooks.append(notebook)
            
            # Execute concurrently
            tasks = [
                bridge.execute_notebook(notebook, timeout=30.0)
                for notebook in notebooks
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for i, (result_notebook, stats) in enumerate(results):
                assert stats.executed_cells == 3
                assert stats.successful_cells == 3
                assert stats.failed_cells == 0
                
                # Check outputs contain correct notebook ID
                code_cells = [cell for cell in result_notebook.cells if cell.cell_type == CellType.CODE]
                last_cell = code_cells[2]
                
                output_found = False
                for output in last_cell.outputs:
                    if f"Notebook {i}:" in str(output):
                        output_found = True
                        break
                assert output_found, f"Expected notebook {i} output in: {last_cell.outputs}"

    @pytest.mark.asyncio
    async def test_complex_data_analysis_workflow(self) -> None:
        """Test a complex data analysis workflow execution."""
        kernel_service = KernelService()
        bridge = ExecutionBridge(kernel_service)
        
        async with kernel_service:
            # Create comprehensive analysis notebook
            notebook = NotebookBuilder()
            
            # Setup
            notebook.add_markdown_cell("# Complex Data Analysis")
            notebook.add_code_cell("import json\nimport statistics")
            
            # Data creation
            notebook.add_markdown_cell("## Data Creation")
            notebook.add_code_cell("""
# Create sample dataset
data = {
    'sales': [100, 150, 120, 180, 200, 170, 190],
    'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
}
print(f'Created dataset with {len(data["sales"])} data points')
            """)
            
            # Analysis
            notebook.add_markdown_cell("## Statistical Analysis")
            notebook.add_code_cell("""
# Calculate statistics
sales = data['sales']
mean_sales = statistics.mean(sales)
median_sales = statistics.median(sales)
max_sales = max(sales)
min_sales = min(sales)

print(f'Mean sales: {mean_sales:.2f}')
print(f'Median sales: {median_sales}')
print(f'Max sales: {max_sales}')
print(f'Min sales: {min_sales}')
            """)
            
            # Results
            notebook.add_markdown_cell("## Results Summary")
            notebook.add_code_cell("""
# Create summary report
summary = {
    'total_days': len(data['sales']),
    'mean': mean_sales,
    'median': median_sales,
    'range': max_sales - min_sales
}

print('Analysis Summary:')
for key, value in summary.items():
    print(f'  {key}: {value}')
            """)
            
            # Execute notebook
            result, stats = await bridge.execute_notebook(notebook, timeout=60.0)
            
            # Should execute all code cells successfully
            code_cell_count = len([cell for cell in notebook.cells if cell.cell_type == CellType.CODE])
            assert stats.executed_cells == code_cell_count
            assert stats.successful_cells == code_cell_count
            assert stats.failed_cells == 0
            
            # Check that outputs contain expected analysis results
            code_cells = [cell for cell in notebook.cells if cell.cell_type == CellType.CODE]
            
            # Last cell should have summary output
            summary_cell = code_cells[-1]
            summary_output = str(summary_cell.outputs)
            assert "Analysis Summary:" in summary_output
            assert "total_days:" in summary_output
            assert "mean:" in summary_output 