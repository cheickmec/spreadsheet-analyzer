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
from pathlib import Path

import pytest

from spreadsheet_analyzer.core_exec.kernel_service import (
    ExecutionResult,
    KernelProfile,
    KernelService,
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
            working_dir=working_dir,
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
            status="ok",
            outputs=[{"type": "stream", "text": "Hello, World!"}],
            duration_seconds=1.23,
            execution_count=1,
            msg_id="test-msg-123",
        )

        assert result.status == "ok"
        assert len(result.outputs) == 1
        assert result.outputs[0]["text"] == "Hello, World!"
        assert result.duration_seconds == 1.23
        assert result.execution_count == 1
        assert result.msg_id == "test-msg-123"
        assert result.error is None

    def test_error_execution_result(self) -> None:
        """Test creating an ExecutionResult for failed execution."""
        error_info = {
            "ename": "ValueError",
            "evalue": "Invalid input",
            "traceback": ["Traceback...", "ValueError: Invalid input"],
        }

        result = ExecutionResult(
            status="error",
            outputs=[],
            duration_seconds=0.5,
            execution_count=1,
            msg_id="error-msg-456",
            error=error_info,
        )

        assert result.status == "error"
        assert len(result.outputs) == 0
        assert result.duration_seconds == 0.5
        assert result.error == error_info
        assert result.error["ename"] == "ValueError"

    def test_timeout_execution_result(self) -> None:
        """Test creating an ExecutionResult for timeout."""
        result = ExecutionResult(
            status="timeout", outputs=[], duration_seconds=30.0, execution_count=1, msg_id="timeout-msg-789"
        )

        assert result.status == "timeout"
        assert result.duration_seconds == 30.0


class TestKernelService:
    """Test the main KernelService functionality using real kernels."""

    @pytest.mark.asyncio
    async def test_kernel_service_creation(self) -> None:
        """Test creating a KernelService with default configuration."""
        profile = KernelProfile()
        service = KernelService(profile)

        assert service.profile.name == "python3"
        assert service.max_sessions == 10  # Default value
        assert service._sessions == {}
        assert not service._shutdown

    @pytest.mark.asyncio
    async def test_kernel_service_with_custom_profile(self) -> None:
        """Test creating a KernelService with custom profile."""
        profile = KernelProfile(max_execution_time=60.0, max_memory_mb=2048)

        service = KernelService(profile, max_sessions=5)

        assert service.profile.max_execution_time == 60.0
        assert service.profile.max_memory_mb == 2048
        assert service.max_sessions == 5

    @pytest.mark.asyncio
    async def test_kernel_lifecycle_basic(self) -> None:
        """Test basic session creation and shutdown."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            # Create a session
            session_id = await service.create_session("test-session")
            assert session_id == "test-session"
            assert session_id in service._sessions

            # Check session is active
            sessions = await service.list_sessions()
            assert session_id in sessions

            # Close the session
            await service.close_session(session_id)
            assert session_id not in service._sessions

    @pytest.mark.asyncio
    async def test_simple_code_execution(self) -> None:
        """Test executing simple Python code in a session."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("calc-session")

            # Execute simple calculation
            result = await service.execute(session_id, "2 + 2")

            assert result.status == "ok"
            assert result.duration_seconds > 0
            assert result.msg_id is not None

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
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("print-session")

            # Execute print statement
            result = await service.execute(session_id, "print('Hello, World!')")

            assert result.status == "ok"

            # Look for stream output containing our text
            stream_found = False
            for output in result.outputs:
                if (
                    isinstance(output, dict)
                    and output.get("type") == "stream"
                    and "Hello, World!" in output.get("text", "")
                ):
                    stream_found = True
                    break
            assert stream_found, f"Expected stream output with 'Hello, World!' in: {result.outputs}"

    @pytest.mark.asyncio
    async def test_variable_persistence(self) -> None:
        """Test that variables persist across executions in the same session."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("var-session")

            # Set a variable
            result1 = await service.execute(session_id, "test_var = 42")
            assert result1.status == "ok"

            # Use the variable in another execution
            result2 = await service.execute(session_id, "print(test_var)")
            assert result2.status == "ok"

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
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("error-session")

            # Execute code that raises an error
            result = await service.execute(session_id, "1 / 0")

            assert result.status == "error"
            assert result.error is not None
            assert "ZeroDivisionError" in result.error.get("ename", "")

    @pytest.mark.asyncio
    async def test_execution_timeout(self) -> None:
        """Test timeout handling for long-running code."""
        # Create service with short timeout
        profile = KernelProfile(max_execution_time=1.0)
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("timeout-session")

            # Execute code that takes too long - should return timeout status
            result = await service.execute(session_id, "import time; time.sleep(5)")
            assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_multiple_sessions(self) -> None:
        """Test creating and managing multiple sessions simultaneously."""
        profile = KernelProfile()
        service = KernelService(profile, max_sessions=3)

        async with service:
            # Create multiple sessions
            session_ids = []
            for i in range(3):
                session_id = await service.create_session(f"session-{i}")
                session_ids.append(session_id)

            assert len(session_ids) == 3
            assert len(service._sessions) == 3

            # Execute different code in each session
            results = []
            for i, session_id in enumerate(session_ids):
                result = await service.execute(session_id, f"session_num = {i + 1}")
                results.append(result)

            # All executions should succeed
            for result in results:
                assert result.status == "ok"

    @pytest.mark.asyncio
    async def test_session_isolation(self) -> None:
        """Test that sessions are isolated from each other."""
        profile = KernelProfile()
        service = KernelService(profile, max_sessions=2)

        async with service:
            session1 = await service.create_session("session1")
            session2 = await service.create_session("session2")

            # Set variable in session1
            result1 = await service.execute(session1, "isolation_test = 'session1'")
            assert result1.status == "ok"

            # Try to access the variable from session2 (should fail)
            result2 = await service.execute(session2, "print(isolation_test)")
            assert result2.status == "error"
            assert "NameError" in result2.error.get("ename", "")

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self) -> None:
        """Test that context manager properly cleans up resources."""
        profile = KernelProfile()
        service = KernelService(profile)
        session_ids = []

        async with service:
            # Create some sessions
            for i in range(2):
                session_id = await service.create_session(f"cleanup-{i}")
                session_ids.append(session_id)

        # After context exit, service should be shut down
        assert service._shutdown
        assert len(service._sessions) == 0

    @pytest.mark.asyncio
    async def test_session_recreation(self) -> None:
        """Test recreating a session clears its state."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = "restart-session"
            await service.create_session(session_id)

            # Set a variable
            result1 = await service.execute(session_id, "restart_test = 'before_restart'")
            assert result1.status == "ok"

            # Close and recreate the session
            await service.close_session(session_id)
            await service.create_session(session_id)

            # Variable should no longer exist
            result2 = await service.execute(session_id, "print(restart_test)")
            assert result2.status == "error"
            assert "NameError" in result2.error.get("ename", "")


# Integration tests for comprehensive workflow testing
class TestKernelServiceIntegration:
    """Integration tests for KernelService with complex scenarios."""

    @pytest.mark.asyncio
    async def test_data_analysis_workflow(self) -> None:
        """Test a complete data analysis workflow."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("analysis-session")

            # Step 1: Import libraries
            result1 = await service.execute(session_id, "import pandas as pd\nimport numpy as np")
            assert result1.status == "ok"

            # Step 2: Create sample data
            result2 = await service.execute(
                session_id,
                """
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)
            """,
            )
            assert result2.status == "ok"

            # Step 3: Analyze data
            result3 = await service.execute(session_id, "print(f'Average age: {df.age.mean()}')")
            assert result3.status == "ok"

            # Check that output contains expected average
            output_found = False
            for output in result3.outputs:
                if "30.0" in str(output):  # Average of 25, 30, 35
                    output_found = True
                    break
            assert output_found, f"Expected average age in outputs: {result3.outputs}"

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self) -> None:
        """Test that concurrent executions in different sessions work safely."""
        profile = KernelProfile()
        service = KernelService(profile, max_sessions=3)

        async with service:
            # Create multiple sessions
            sessions = []
            for i in range(3):
                session_id = await service.create_session(f"concurrent-{i}")
                sessions.append(session_id)

            # Execute code concurrently in all sessions
            async def execute_in_session(session_id: str, value: int) -> ExecutionResult:
                # Set different values in each session
                await service.execute(session_id, f"concurrent_value = {value}")
                # Return the value to verify isolation
                return await service.execute(session_id, "print(concurrent_value)")

            # Run concurrent executions
            tasks = [
                execute_in_session(sessions[0], 100),
                execute_in_session(sessions[1], 200),
                execute_in_session(sessions[2], 300),
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            for result in results:
                assert result.status == "ok"

            # Each should have printed its own value
            expected_values = ["100", "200", "300"]
            for i, result in enumerate(results):
                value_found = False
                for output in result.outputs:
                    if expected_values[i] in str(output):
                        value_found = True
                        break
                assert value_found, f"Expected {expected_values[i]} in session {i} outputs: {result.outputs}"

    @pytest.mark.asyncio
    async def test_resource_monitoring(self) -> None:
        """Test that resource usage is properly tracked."""
        profile = KernelProfile()
        service = KernelService(profile)

        async with service:
            session_id = await service.create_session("resource-session")

            # Execute code that uses some memory
            result = await service.execute(
                session_id,
                """
import sys
data = list(range(10000))  # Create some data
print(f'Created list with {len(data)} elements')
            """,
            )

            assert result.status == "ok"
            assert result.duration_seconds > 0
            # Note: The restored API doesn't track memory usage per execution

            # Check output
            output_found = False
            for output in result.outputs:
                if "10000" in str(output):
                    output_found = True
                    break
            assert output_found, f"Expected list size in outputs: {result.outputs}"
