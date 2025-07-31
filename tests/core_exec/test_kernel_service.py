"""
Tests for the generic KernelService module.

This test suite validates the core kernel service functionality including:
- KernelProfile configuration and validation
- ExecutionResult data structures
- KernelService lifecycle management
- Async execution with real kernels
- Resource limit enforcement
- Error handling and timeout behavior
- Different kernel profiles and output collection
"""

import pytest

from spreadsheet_analyzer.core_exec.kernel_service import (
    KernelProfile,
    KernelService,
)


class TestKernelProfile:
    """Test KernelProfile configuration and validation."""

    def test_default_profile_creation(self) -> None:
        """Test creating a KernelProfile with default values."""
        profile = KernelProfile()
        assert profile.name == "python3"
        assert profile.max_execution_time == 30.0

    def test_custom_profile_creation(self) -> None:
        """Test creating a KernelProfile with custom values."""
        profile = KernelProfile(
            name="python3",
            max_execution_time=60.0,
            idle_timeout_seconds=600.0,
        )
        assert profile.max_execution_time == 60.0
        assert profile.idle_timeout_seconds == 600.0

    def test_profile_validation(self) -> None:
        """Test that KernelProfile validates input parameters."""
        with pytest.raises(ValueError):
            KernelProfile(max_cpu_percent=-10.0)
        with pytest.raises(ValueError):
            KernelProfile(max_memory_mb=-512)
        with pytest.raises(ValueError):
            KernelProfile(max_execution_time=-5.0)


class TestKernelService:
    """Test the main KernelService functionality using real kernels."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_kernel_lifecycle_basic(self) -> None:
        """Test basic session creation and shutdown."""
        profile = KernelProfile()
        async with KernelService(profile) as service:
            session_id = await service.create_session("test-session")
            assert session_id in await service.list_sessions()
            await service.close_session(session_id)
            assert session_id not in await service.list_sessions()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_simple_code_execution(self) -> None:
        """Test executing simple Python code in a session."""
        profile = KernelProfile()
        async with KernelService(profile) as service:
            session_id = await service.create_session("calc-session")
            result = await service.execute(session_id, "2 + 2")
            assert result.status == "ok"
            assert any("4" in str(o) for o in result.outputs)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_print_statement_execution(self) -> None:
        """Test executing code with print statements."""
        profile = KernelProfile()
        async with KernelService(profile) as service:
            session_id = await service.create_session("print-session")
            result = await service.execute(session_id, "print('Hello, Kernel!')")
            assert result.status == "ok"
            assert any("Hello, Kernel!" in o.get("text", "") for o in result.outputs if o.get("type") == "stream")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_variable_persistence(self) -> None:
        """Test that variables persist across executions in the same session."""
        profile = KernelProfile()
        async with KernelService(profile) as service:
            session_id = await service.create_session("var-session")
            await service.execute(session_id, "test_var = 123")
            result = await service.execute(session_id, "print(test_var)")
            assert result.status == "ok"
            assert any("123" in str(o) for o in result.outputs)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test handling of Python execution errors."""
        profile = KernelProfile()
        async with KernelService(profile) as service:
            session_id = await service.create_session("error-session")
            result = await service.execute(session_id, "1 / 0")
            assert result.status == "error"
            assert result.error is not None
            assert "ZeroDivisionError" in result.error.get("ename", "")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_execution_timeout(self) -> None:
        """Test timeout handling for long-running code."""
        profile = KernelProfile(max_execution_time=1.0)
        async with KernelService(profile) as service:
            session_id = await service.create_session("timeout-session")
            result = await service.execute(session_id, "import time; time.sleep(3)")
            assert result.status == "error"  # nbclient raises an exception on timeout
            assert "Timeout" in result.error.get("ename", "")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_session_isolation(self) -> None:
        """Test that sessions are isolated from each other."""
        profile = KernelProfile()
        async with KernelService(profile, max_sessions=2) as service:
            session1 = await service.create_session("session1")
            session2 = await service.create_session("session2")
            await service.execute(session1, "iso_var = 'session1_val'")
            result = await service.execute(session2, "print(iso_var)")
            assert result.status == "error"
            assert "NameError" in result.error.get("ename", "")

    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "profile_name, profile",
        [
            ("default", KernelProfile()),
            (
                "aggressive_timing",
                KernelProfile(
                    output_drain_timeout_ms=500,
                    output_drain_max_timeout_ms=2000,
                    output_drain_max_attempts=5,
                ),
            ),
        ],
    )
    async def test_kernel_profiles(self, profile_name, profile) -> None:
        """Test kernel execution with different profiles."""
        async with KernelService(profile) as service:
            session_id = await service.create_session(f"test-{profile_name}")
            result = await service.execute(
                session_id,
                "import pandas as pd; print(f'pandas version: {pd.__version__}')",
            )
            assert result.status == "ok"
            assert any("pandas version" in str(o) for o in result.outputs)
