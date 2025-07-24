"""
Tests for Jupyter Kernel Manager implementation.

Following TDD principles: Red -> Green -> Refactor

This test module validates the kernel manager functionality including:
- Kernel lifecycle management (start, stop, restart)
- Kernel pooling with max capacity limits
- Async context manager for resource acquisition/release
- Resource limits enforcement (CPU, memory, execution time)
- Session persistence and checkpointing
- Security sandboxing configuration
"""

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_asyncio import CoroutineFunction

from spreadsheet_analyzer.agents.kernel_manager import (
    AgentKernelManager,
    KernelPool,
    KernelPoolExhaustedError,
    KernelResource,
    KernelResourceLimits,
    KernelSession,
    KernelTimeoutError,
)


class TestKernelResourceLimits:
    """Test kernel resource limit configuration."""

    def test_default_limits(self) -> None:
        """Test that default resource limits are sensible."""
        limits = KernelResourceLimits()

        assert limits.max_cpu_percent == 80.0
        assert limits.max_memory_mb == 1024
        assert limits.max_execution_time == 30.0
        assert limits.max_output_size_mb == 10

    def test_custom_limits(self) -> None:
        """Test creating custom resource limits."""
        limits = KernelResourceLimits(
            max_cpu_percent=50.0, max_memory_mb=2048, max_execution_time=60.0, max_output_size_mb=20
        )

        assert limits.max_cpu_percent == 50.0
        assert limits.max_memory_mb == 2048
        assert limits.max_execution_time == 60.0
        assert limits.max_output_size_mb == 20

    def test_limits_validation(self) -> None:
        """Test that resource limits are validated."""
        # Negative values should raise errors
        with pytest.raises(ValueError, match="CPU percent must be positive"):
            KernelResourceLimits(max_cpu_percent=-10)

        with pytest.raises(ValueError, match="Memory limit must be positive"):
            KernelResourceLimits(max_memory_mb=-100)

        with pytest.raises(ValueError, match="Execution time must be positive"):
            KernelResourceLimits(max_execution_time=-5)

        # CPU percent over 100 should raise error
        with pytest.raises(ValueError, match="CPU percent cannot exceed 100"):
            KernelResourceLimits(max_cpu_percent=150.0)


class TestKernelResource:
    """Test individual kernel resource management."""

    @pytest.mark.asyncio
    async def test_kernel_creation(self) -> None:
        """Test creating a kernel resource."""
        kernel = KernelResource(kernel_id="test-kernel-1")

        assert kernel.kernel_id == "test-kernel-1"
        assert kernel.is_available
        assert not kernel.is_shutting_down
        assert kernel.created_at <= time.time()
        assert kernel.last_used_at is None

    @pytest.mark.asyncio
    async def test_kernel_acquire_release(self) -> None:
        """Test acquiring and releasing a kernel."""
        kernel = KernelResource(kernel_id="test-kernel-2")

        # Initial state
        assert kernel.is_available
        assert kernel.last_used_at is None

        # Acquire kernel
        kernel.acquire()
        assert not kernel.is_available
        assert kernel.last_used_at is not None

        # Cannot acquire when already in use
        with pytest.raises(RuntimeError, match="Kernel is not available"):
            kernel.acquire()  # type: ignore[unreachable]

        # Release kernel
        kernel.release()
        assert kernel.is_available

        # Cannot release when already available
        with pytest.raises(RuntimeError, match="Kernel is already available"):
            kernel.release()

    @pytest.mark.asyncio
    async def test_kernel_context_manager(self) -> None:
        """Test using kernel as async context manager."""
        kernel = KernelResource(kernel_id="test-kernel-3")

        assert kernel.is_available

        async with kernel:
            assert not kernel.is_available
            assert kernel.last_used_at is not None

        assert kernel.is_available


class TestKernelPool:
    """Test kernel pool management."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self) -> None:
        """Test creating a kernel pool."""
        pool = KernelPool(max_kernels=5)

        assert pool.max_kernels == 5
        assert len(pool.kernels) == 0
        assert pool.available_count == 0

    @pytest.mark.asyncio
    async def test_add_kernel_to_pool(self) -> None:
        """Test adding kernels to the pool."""
        pool = KernelPool(max_kernels=3)

        # Add first kernel
        kernel1 = KernelResource("kernel-1")
        pool.add_kernel(kernel1)
        assert len(pool.kernels) == 1
        assert pool.available_count == 1

        # Add second kernel
        kernel2 = KernelResource("kernel-2")
        pool.add_kernel(kernel2)
        assert len(pool.kernels) == 2
        assert pool.available_count == 2

        # Cannot add duplicate kernel ID
        with pytest.raises(ValueError, match="Kernel with ID kernel-1 already exists"):
            pool.add_kernel(KernelResource("kernel-1"))

    @pytest.mark.asyncio
    async def test_acquire_kernel_from_pool(self) -> None:
        """Test acquiring kernels from the pool."""
        pool = KernelPool(max_kernels=2)

        # Add two kernels
        kernel1 = KernelResource("kernel-1")
        kernel2 = KernelResource("kernel-2")
        pool.add_kernel(kernel1)
        pool.add_kernel(kernel2)

        # Acquire first kernel
        acquired = await pool.acquire()
        assert acquired in [kernel1, kernel2]
        assert not acquired.is_available
        assert pool.available_count == 1

        # Acquire second kernel
        acquired2 = await pool.acquire()
        assert acquired2 in [kernel1, kernel2]
        assert acquired2 != acquired
        assert not acquired2.is_available
        assert pool.available_count == 0

        # Pool exhausted
        with pytest.raises(KernelPoolExhaustedError):
            await pool.acquire(timeout=0.1)

    @pytest.mark.asyncio
    async def test_release_kernel_to_pool(self) -> None:
        """Test releasing kernels back to the pool."""
        pool = KernelPool(max_kernels=1)
        kernel = KernelResource("kernel-1")
        pool.add_kernel(kernel)

        # Acquire and release
        acquired = await pool.acquire()
        assert pool.available_count == 0

        pool.release(acquired)
        assert pool.available_count == 1
        assert acquired.is_available

    @pytest.mark.asyncio
    async def test_pool_wait_for_available(self) -> None:
        """Test waiting for a kernel to become available."""
        pool = KernelPool(max_kernels=1)
        kernel = KernelResource("kernel-1")
        pool.add_kernel(kernel)

        # Acquire the only kernel
        acquired = await pool.acquire()

        # Start a task to release after delay
        async def release_after_delay():
            await asyncio.sleep(0.2)
            pool.release(acquired)

        release_task = asyncio.create_task(release_after_delay())

        # Wait for kernel to become available
        start_time = time.time()
        reacquired = await pool.acquire(timeout=1.0)
        wait_time = time.time() - start_time

        assert reacquired == acquired
        assert 0.1 < wait_time < 0.5  # Should wait approximately 0.2 seconds

        await release_task


class TestKernelSession:
    """Test kernel session management."""

    def test_session_creation(self) -> None:
        """Test creating a kernel session."""
        session = KernelSession(session_id="test-session", kernel_id="kernel-1", agent_id="agent-1")

        assert session.session_id == "test-session"
        assert session.kernel_id == "kernel-1"
        assert session.agent_id == "agent-1"
        assert session.created_at <= time.time()
        assert session.last_checkpoint is None
        assert len(session.execution_history) == 0

    def test_add_execution_to_history(self) -> None:
        """Test adding executions to session history."""
        session = KernelSession(session_id="test-session", kernel_id="kernel-1", agent_id="agent-1")

        # Add first execution
        session.add_execution(code="x = 1 + 2", result={"output": "3", "status": "ok"})

        assert len(session.execution_history) == 1
        assert session.execution_history[0]["code"] == "x = 1 + 2"
        assert session.execution_history[0]["result"]["output"] == "3"

        # Add second execution
        session.add_execution(code="print(x)", result={"output": "3\n", "status": "ok"})

        assert len(session.execution_history) == 2

    def test_session_checkpointing(self) -> None:
        """Test session checkpoint functionality."""
        session = KernelSession(session_id="test-session", kernel_id="kernel-1", agent_id="agent-1")

        # Add some executions
        session.add_execution("x = 10", {"status": "ok"})
        session.add_execution("y = 20", {"status": "ok"})

        # Create checkpoint
        checkpoint_time = time.time()
        session.checkpoint()

        assert session.last_checkpoint is not None
        assert session.last_checkpoint >= checkpoint_time
        assert session.checkpoint_data is not None
        assert len(session.checkpoint_data["execution_history"]) == 2


class TestAgentKernelManager:
    """Test the main kernel manager implementation using real kernels."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self) -> None:
        """Test creating a kernel manager."""
        manager = AgentKernelManager(max_kernels=5)

        assert manager.max_kernels == 5
        assert isinstance(manager.pool, KernelPool)
        assert len(manager.sessions) == 0
        assert isinstance(manager.resource_limits, KernelResourceLimits)

    @pytest.mark.asyncio
    async def test_manager_context_manager(self) -> None:
        """Test using manager as async context manager."""
        manager = AgentKernelManager(max_kernels=2)

        async with manager as mgr:
            assert mgr == manager
            # Manager should be initialized

        # Manager should be cleaned up
        assert manager._shutdown

    @pytest.mark.asyncio
    async def test_acquire_kernel_for_agent(self) -> None:
        """Test acquiring a real kernel for an agent."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            # Acquire kernel
            async with manager.acquire_kernel("agent-1") as (kernel_manager, session):
                assert kernel_manager is not None
                assert session.agent_id == "agent-1"
                # Kernel ID is auto-generated UUID, just check it exists
                assert session.kernel_id is not None
                assert len(session.kernel_id) > 0
                assert "agent-1" in manager.sessions

                # Test that the kernel is actually working
                result = await manager.execute_code(session, "2 + 2")
                assert "4" in str(result)

            # Session should still exist after release
            assert "agent-1" in manager.sessions

    @pytest.mark.asyncio
    async def test_kernel_reuse_for_same_agent(self) -> None:
        """Test that the same agent reuses its kernel."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            # First acquisition
            async with manager.acquire_kernel("agent-1") as (km1, session1):
                first_kernel_id = session1.kernel_id
                # Set a variable to test persistence
                await manager.execute_code(session1, "test_var = 'persistent'")

            # Second acquisition by same agent
            async with manager.acquire_kernel("agent-1") as (km2, session2):
                assert session2.kernel_id == first_kernel_id
                assert session2 == session1  # Same session object

                # Test that variable persists
                result = await manager.execute_code(session2, "test_var")
                assert "persistent" in str(result)

    @pytest.mark.asyncio
    async def test_pool_exhaustion_handling(self) -> None:
        """Test handling when kernel pool is exhausted."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager, manager.acquire_kernel("agent-1") as (km1, session1):
            # Try to acquire for different agent (should timeout)
            with pytest.raises(KernelPoolExhaustedError):
                async with manager.acquire_kernel("agent-2", timeout=0.1):
                    pass

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self) -> None:
        """Test executing code with timeout enforcement."""
        manager = AgentKernelManager(max_kernels=1, resource_limits=KernelResourceLimits(max_execution_time=1.0))

        async with manager, manager.acquire_kernel("agent-1") as (km, session):
            # Execute code that takes too long
            with pytest.raises(KernelTimeoutError):
                await manager.execute_code(session, "import time; time.sleep(5)")

    @pytest.mark.asyncio
    async def test_session_persistence(self) -> None:
        """Test that session state persists across kernel acquisitions."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager:
            # First execution
            async with manager.acquire_kernel("agent-1") as (km, session):
                await manager.execute_code(session, "x = 42")
                assert len(session.execution_history) == 1

            # Second execution - session should persist
            async with manager.acquire_kernel("agent-1") as (km, session):
                result = await manager.execute_code(session, "print(x)")
                assert len(session.execution_history) == 2
                assert "print(x)" in session.execution_history[1]["code"]
                assert "42" in str(result)

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self) -> None:
        """Test graceful shutdown of all kernels."""
        manager = AgentKernelManager(max_kernels=3)

        async with manager:
            # Start kernels for different agents
            sessions = []
            for i in range(3):
                async with manager.acquire_kernel(f"agent-{i}") as (km, session):
                    sessions.append(session)
                    # Execute something to ensure kernel is active
                    await manager.execute_code(session, f"agent_id = '{session.agent_id}'")

        # Manager is shut down automatically via context manager
        assert manager._shutdown

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self) -> None:
        """Test session checkpointing and restoration."""
        manager = AgentKernelManager(max_kernels=1)

        # Create a session with some history
        session = KernelSession(session_id="test-session", kernel_id="kernel-1", agent_id="agent-1")
        session.add_execution("x = 10", {"status": "ok"})
        session.add_execution("y = 20", {"status": "ok"})

        # Save checkpoint
        checkpoint_data = manager.save_checkpoint(session)

        assert checkpoint_data["session_id"] == "test-session"
        assert checkpoint_data["agent_id"] == "agent-1"
        assert len(checkpoint_data["execution_history"]) == 2

        # Create new session and restore
        new_session = KernelSession(session_id="new-session", kernel_id="kernel-2", agent_id="agent-1")

        manager.restore_checkpoint(new_session, checkpoint_data)

        assert len(new_session.execution_history) == 2
        assert new_session.execution_history[0]["code"] == "x = 10"
        assert new_session.execution_history[1]["code"] == "y = 20"


# Integration test with real jupyter kernel (marked as slow)
@pytest.mark.slow
@pytest.mark.integration
class TestAgentKernelManagerIntegration:
    """Integration tests with real Jupyter kernels."""

    @pytest.mark.asyncio
    async def test_real_kernel_execution(self) -> None:
        """Test executing code in a real kernel."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager, manager.acquire_kernel("test-agent") as (km, session):
            # Execute simple calculation
            result = await manager.execute_code(session, "2 + 2")
            assert "4" in str(result)

            # Execute with variables
            await manager.execute_code(session, "x = 10")
            result = await manager.execute_code(session, "x * 2")
            assert "20" in str(result)

            # Verify session history
            assert len(session.execution_history) == 3

    @pytest.mark.asyncio
    async def test_real_kernel_error_handling(self) -> None:
        """Test error handling in real kernel."""
        manager = AgentKernelManager(max_kernels=1)

        async with manager, manager.acquire_kernel("test-agent") as (km, session):
            # Execute code with error
            result = await manager.execute_code(session, "1 / 0")
            assert "error" in result.get("status", "")
            assert "ZeroDivisionError" in str(result)

    @pytest.mark.asyncio
    async def test_real_kernel_timeout(self) -> None:
        """Test timeout handling with real kernel."""
        manager = AgentKernelManager(max_kernels=1, resource_limits=KernelResourceLimits(max_execution_time=1.0))

        async with manager, manager.acquire_kernel("test-agent") as (km, session):
            # Execute code that takes too long
            with pytest.raises(KernelTimeoutError):
                await manager.execute_code(session, "import time; time.sleep(5)")
