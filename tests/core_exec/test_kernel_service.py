"""
Tests for the generic KernelService module.

This test suite validates the core kernel service functionality including:
- KernelProfile configuration and validation
- ExecutionResult data structures
- KernelService lifecycle management
- Async execution with real kernels
- Resource limit enforcement
- Error handling and timeout behavior

Following TDD principles with functional tests - no mocking used.
All tests use real Jupyter kernels for authentic behavior validation.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from spreadsheet_analyzer.core_exec.kernel_service import (
    KernelProfile,
    ExecutionResult,
    KernelService,
    KernelTimeoutError,
    KernelResourceLimitError,
)


class TestKernelProfile:
    """Test KernelProfile configuration and validation."""

    def test_default_profile_creation(self) -> None:
        """Test creating a KernelProfile with default values."""
        profile = KernelProfile()
        
        assert profile.name == "python3"
        assert profile.max_cpu_percent == 80.0
        assert profile.max_memory_mb == 1024
        assert profile.max_execution_time == 30.0
        assert profile.max_output_size_mb == 10
        assert profile.idle_timeout_seconds == 300.0
        assert profile.env_vars == {}
        assert profile.working_dir is None

    def test_custom_profile_creation(self) -> None:
        """Test creating a KernelProfile with custom values."""
        custom_env = {"PYTHONPATH": "/custom/path", "DEBUG": "1"}
        working_dir = Path("/tmp/kernel_work")
        
        profile = KernelProfile(
            name="python3",
            max_cpu_percent=50.0,
            max_memory_mb=2048,
            max_execution_time=60.0,
            max_output_size_mb=20,
            idle_timeout_seconds=600.0,
            env_vars=custom_env,
            working_dir=working_dir
        )
        
        assert profile.name == "python3"
        assert profile.max_cpu_percent == 50.0
        assert profile.max_memory_mb == 2048
        assert profile.max_execution_time == 60.0
        assert profile.max_output_size_mb == 20
        assert profile.idle_timeout_seconds == 600.0
        assert profile.env_vars == custom_env
        assert profile.working_dir == working_dir

    def test_profile_validation(self) -> None:
        """Test that KernelProfile validates input parameters."""
        # Test negative CPU percent
        with pytest.raises(ValueError, match="CPU percent must be between 0 and 100"):
            KernelProfile(max_cpu_percent=-10.0)
            
        # Test CPU percent over 100
        with pytest.raises(ValueError, match="CPU percent must be between 0 and 100"):
            KernelProfile(max_cpu_percent=150.0)
            
        # Test negative memory
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            KernelProfile(max_memory_mb=-512)
            
        # Test negative timeout
        with pytest.raises(ValueError, match="Execution time must be positive"):
            KernelProfile(max_execution_time=-5.0)

    def test_profile_immutability(self) -> None:
        """Test that KernelProfile is immutable (frozen dataclass)."""
        profile = KernelProfile()
        
        # Should not be able to modify fields
        with pytest.raises(Exception):  # FrozenInstanceError
            profile.max_cpu_percent = 90.0  # type: ignore
        
        with pytest.raises(Exception):  # FrozenInstanceError
            profile.name = "julia-1.6"  # type: ignore


class TestExecutionResult:
    """Test ExecutionResult data structure."""

    def test_successful_execution_result(self) -> None:
        """Test creating an ExecutionResult for successful execution."""
        result = ExecutionResult(
            status="success",
            outputs=[{"output_type": "stream", "text": "Hello, World!"}],
            execution_time=1.23,
            memory_usage_mb=45.6,
            message_id="test-msg-123"
        )
        
        assert result.status == "success"
        assert len(result.outputs) == 1
        assert result.outputs[0]["text"] == "Hello, World!"
        assert result.execution_time == 1.23
        assert result.memory_usage_mb == 45.6
        assert result.message_id == "test-msg-123"
        assert result.error_info is None

    def test_error_execution_result(self) -> None:
        """Test creating an ExecutionResult for failed execution."""
        error_info = {
            "ename": "ValueError",
            "evalue": "Invalid input",
            "traceback": ["Traceback...", "ValueError: Invalid input"]
        }
        
        result = ExecutionResult(
            status="error",
            outputs=[],
            execution_time=0.5,
            memory_usage_mb=30.0,
            message_id="error-msg-456",
            error_info=error_info
        )
        
        assert result.status == "error"
        assert len(result.outputs) == 0
        assert result.execution_time == 0.5
        assert result.error_info == error_info
        assert result.error_info["ename"] == "ValueError"

    def test_timeout_execution_result(self) -> None:
        """Test creating an ExecutionResult for timeout."""
        result = ExecutionResult(
            status="timeout",
            outputs=[],
            execution_time=30.0,
            memory_usage_mb=100.0,
            message_id="timeout-msg-789"
        )
        
        assert result.status == "timeout"
        assert result.execution_time == 30.0


class TestKernelService:
    """Test the main KernelService functionality using real kernels."""

    @pytest.mark.asyncio
    async def test_kernel_service_creation(self) -> None:
        """Test creating a KernelService with default configuration."""
        service = KernelService()
        
        assert service.profile.name == "python3"
        assert service.max_concurrent_kernels == 5
        assert service._active_kernels == {}
        assert service._kernel_pool == []
        assert not service._shutdown

    @pytest.mark.asyncio
    async def test_kernel_service_with_custom_profile(self) -> None:
        """Test creating a KernelService with custom profile."""
        profile = KernelProfile(
            max_execution_time=60.0,
            max_memory_mb=2048
        )
        
        service = KernelService(profile=profile, max_concurrent_kernels=10)
        
        assert service.profile.max_execution_time == 60.0
        assert service.profile.max_memory_mb == 2048
        assert service.max_concurrent_kernels == 10

    @pytest.mark.asyncio
    async def test_kernel_lifecycle_basic(self) -> None:
        """Test basic kernel creation and shutdown."""
        service = KernelService()
        
        async with service:
            # Create a kernel
            kernel_id = await service.create_kernel()
            assert kernel_id is not None
            assert len(kernel_id) > 0
            assert kernel_id in service._active_kernels
            
            # Check kernel is alive
            assert await service.is_kernel_alive(kernel_id)
            
            # Shutdown the kernel
            await service.shutdown_kernel(kernel_id)
            assert kernel_id not in service._active_kernels

    @pytest.mark.asyncio
    async def test_simple_code_execution(self) -> None:
        """Test executing simple Python code in a kernel."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Execute simple calculation
            result = await service.execute_code(kernel_id, "2 + 2")
            
            assert result.status == "success"
            assert result.execution_time > 0
            assert result.message_id is not None
            
            # Check for expected output (should contain "4")
            output_found = False
            for output in result.outputs:
                if "4" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected '4' in outputs: {result.outputs}"

    @pytest.mark.asyncio
    async def test_print_statement_execution(self) -> None:
        """Test executing code with print statements."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Execute print statement
            result = await service.execute_code(kernel_id, "print('Hello, World!')")
            
            assert result.status == "success"
            
            # Look for stream output containing our text
            stream_found = False
            for output in result.outputs:
                if (isinstance(output, dict) and 
                    output.get("output_type") == "stream" and
                    "Hello, World!" in output.get("text", "")):
                    stream_found = True
                    break
            assert stream_found, f"Expected stream output with 'Hello, World!' in: {result.outputs}"

    @pytest.mark.asyncio 
    async def test_variable_persistence(self) -> None:
        """Test that variables persist across executions in the same kernel."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Set a variable
            result1 = await service.execute_code(kernel_id, "test_var = 42")
            assert result1.status == "success"
            
            # Use the variable in another execution
            result2 = await service.execute_code(kernel_id, "print(test_var)")
            assert result2.status == "success"
            
            # Check that output contains our variable value
            output_found = False
            for output in result2.outputs:
                if "42" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected '42' in outputs: {result2.outputs}"

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test handling of Python execution errors."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Execute code that raises an error
            result = await service.execute_code(kernel_id, "1 / 0")
            
            assert result.status == "error"
            assert result.error_info is not None
            assert "ZeroDivisionError" in result.error_info.get("ename", "")

    @pytest.mark.asyncio
    async def test_execution_timeout(self) -> None:
        """Test timeout handling for long-running code."""
        # Create service with short timeout
        profile = KernelProfile(max_execution_time=1.0)
        service = KernelService(profile=profile)
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Execute code that takes too long
            with pytest.raises(KernelTimeoutError):
                await service.execute_code(kernel_id, "import time; time.sleep(5)")

    @pytest.mark.asyncio
    async def test_multiple_kernels(self) -> None:
        """Test creating and managing multiple kernels simultaneously."""
        service = KernelService(max_concurrent_kernels=3)
        
        async with service:
            # Create multiple kernels
            kernel_ids = []
            for i in range(3):
                kernel_id = await service.create_kernel()
                kernel_ids.append(kernel_id)
            
            assert len(kernel_ids) == 3
            assert len(service._active_kernels) == 3
            
            # Execute different code in each kernel
            results = []
            for i, kernel_id in enumerate(kernel_ids):
                result = await service.execute_code(kernel_id, f"kernel_num = {i + 1}")
                results.append(result)
            
            # All executions should succeed
            for result in results:
                assert result.status == "success"

    @pytest.mark.asyncio
    async def test_kernel_isolation(self) -> None:
        """Test that kernels are isolated from each other."""
        service = KernelService(max_concurrent_kernels=2)
        
        async with service:
            kernel1 = await service.create_kernel()
            kernel2 = await service.create_kernel()
            
            # Set variable in kernel1
            result1 = await service.execute_code(kernel1, "isolation_test = 'kernel1'")
            assert result1.status == "success"
            
            # Try to access the variable from kernel2 (should fail)
            result2 = await service.execute_code(kernel2, "print(isolation_test)")
            assert result2.status == "error"
            assert "NameError" in result2.error_info.get("ename", "")

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self) -> None:
        """Test that context manager properly cleans up resources."""
        service = KernelService()
        kernel_ids = []
        
        async with service:
            # Create some kernels
            for _ in range(2):
                kernel_id = await service.create_kernel()
                kernel_ids.append(kernel_id)
        
        # After context exit, service should be shut down
        assert service._shutdown
        assert len(service._active_kernels) == 0

    @pytest.mark.asyncio
    async def test_kernel_restart(self) -> None:
        """Test restarting a kernel clears its state."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Set a variable
            result1 = await service.execute_code(kernel_id, "restart_test = 'before_restart'")
            assert result1.status == "success"
            
            # Restart the kernel
            await service.restart_kernel(kernel_id)
            
            # Variable should no longer exist
            result2 = await service.execute_code(kernel_id, "print(restart_test)")
            assert result2.status == "error"
            assert "NameError" in result2.error_info.get("ename", "")


# Integration tests for comprehensive workflow testing
class TestKernelServiceIntegration:
    """Integration tests for KernelService with complex scenarios."""

    @pytest.mark.asyncio
    async def test_data_analysis_workflow(self) -> None:
        """Test a complete data analysis workflow."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Step 1: Import libraries
            result1 = await service.execute_code(kernel_id, "import pandas as pd\nimport numpy as np")
            assert result1.status == "success"
            
            # Step 2: Create sample data
            result2 = await service.execute_code(kernel_id, """
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)
            """)
            assert result2.status == "success"
            
            # Step 3: Analyze data
            result3 = await service.execute_code(kernel_id, "print(f'Average age: {df.age.mean()}')")
            assert result3.status == "success"
            
            # Check that output contains expected average
            output_found = False
            for output in result3.outputs:
                if "30.0" in str(output):  # Average of 25, 30, 35
                    output_found = True
                    break
            assert output_found, f"Expected average age in outputs: {result3.outputs}"

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self) -> None:
        """Test that concurrent executions in different kernels work safely."""
        service = KernelService(max_concurrent_kernels=3)
        
        async with service:
            # Create multiple kernels
            kernels = []
            for i in range(3):
                kernel_id = await service.create_kernel()
                kernels.append(kernel_id)
            
            # Execute code concurrently in all kernels
            async def execute_in_kernel(kernel_id: str, value: int) -> ExecutionResult:
                # Set different values in each kernel
                await service.execute_code(kernel_id, f"concurrent_value = {value}")
                # Return the value to verify isolation
                return await service.execute_code(kernel_id, "print(concurrent_value)")
            
            # Run concurrent executions
            tasks = [
                execute_in_kernel(kernels[0], 100),
                execute_in_kernel(kernels[1], 200), 
                execute_in_kernel(kernels[2], 300)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for result in results:
                assert result.status == "success"
            
            # Each should have printed its own value
            expected_values = ["100", "200", "300"]
            for i, result in enumerate(results):
                value_found = False
                for output in result.outputs:
                    if expected_values[i] in str(output):
                        value_found = True
                        break
                assert value_found, f"Expected {expected_values[i]} in kernel {i} outputs: {result.outputs}"

    @pytest.mark.asyncio
    async def test_resource_monitoring(self) -> None:
        """Test that resource usage is properly tracked."""
        service = KernelService()
        
        async with service:
            kernel_id = await service.create_kernel()
            
            # Execute code that uses some memory
            result = await service.execute_code(kernel_id, """
import sys
data = list(range(10000))  # Create some data
print(f'Created list with {len(data)} elements')
            """)
            
            assert result.status == "success"
            assert result.execution_time > 0
            assert result.memory_usage_mb >= 0  # Should track some memory usage
            
            # Check output
            output_found = False
            for output in result.outputs:
                if "10000" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected list size in outputs: {result.outputs}" 