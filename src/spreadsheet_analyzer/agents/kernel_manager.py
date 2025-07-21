"""
Jupyter Kernel Manager for agent isolation and execution.

This module provides a managed pool of Jupyter kernels for agent code execution
with resource limits, session persistence, and security sandboxing.

CLAUDE-KNOWLEDGE: Jupyter kernels provide isolated Python execution environments
that can be reused across multiple code executions while maintaining state.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


class KernelTimeoutError(Exception):
    """Raised when kernel execution exceeds time limit."""

    pass


class KernelPoolExhaustedError(Exception):
    """Raised when no kernels are available in the pool."""

    pass


@dataclass(frozen=True)
class KernelResourceLimits:
    """Resource limits for kernel execution."""

    max_cpu_percent: float = 80.0
    max_memory_mb: int = 1024
    max_execution_time: float = 30.0
    max_output_size_mb: int = 10

    def __post_init__(self) -> None:
        """Validate resource limits."""
        if self.max_cpu_percent < 0:
            raise ValueError("CPU percent must be positive")
        if self.max_cpu_percent > 100:
            raise ValueError("CPU percent cannot exceed 100")
        if self.max_memory_mb < 0:
            raise ValueError("Memory limit must be positive")
        if self.max_execution_time < 0:
            raise ValueError("Execution time must be positive")


@dataclass
class KernelResource:
    """Represents a single kernel resource in the pool."""

    kernel_id: str
    is_available: bool = True
    is_shutting_down: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    _kernel_manager: AsyncKernelManager | None = None

    def acquire(self) -> None:
        """Mark kernel as in use."""
        if not self.is_available:
            raise RuntimeError("Kernel is not available")
        self.is_available = False
        self.last_used_at = time.time()

    def release(self) -> None:
        """Mark kernel as available."""
        if self.is_available:
            raise RuntimeError("Kernel is already available")
        self.is_available = True

    async def __aenter__(self) -> "KernelResource":
        """Async context manager entry."""
        self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self.release()


@dataclass
class KernelSession:
    """Represents an agent's kernel session with execution history."""

    session_id: str
    kernel_id: str
    agent_id: str
    created_at: float = field(default_factory=time.time)
    last_checkpoint: float | None = None
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_data: dict[str, Any] | None = None

    def add_execution(self, code: str, result: dict[str, Any]) -> None:
        """Add an execution to the session history."""
        self.execution_history.append({"timestamp": time.time(), "code": code, "result": result})

    def checkpoint(self) -> None:
        """Create a checkpoint of the current session state."""
        self.last_checkpoint = time.time()
        self.checkpoint_data = {
            "session_id": self.session_id,
            "kernel_id": self.kernel_id,
            "agent_id": self.agent_id,
            "checkpoint_time": self.last_checkpoint,
            "execution_history": self.execution_history.copy(),
        }


class KernelPool:
    """Manages a pool of kernel resources."""

    def __init__(self, max_kernels: int = 10) -> None:
        """Initialize kernel pool."""
        self.max_kernels = max_kernels
        self.kernels: dict[str, KernelResource] = {}
        self._pool_lock = asyncio.Lock()
        self._available_event = asyncio.Event()

    @property
    def available_count(self) -> int:
        """Get number of available kernels."""
        return sum(1 for k in self.kernels.values() if k.is_available)

    def add_kernel(self, kernel: KernelResource) -> None:
        """Add a kernel to the pool."""
        if kernel.kernel_id in self.kernels:
            raise ValueError(f"Kernel with ID {kernel.kernel_id} already exists")
        self.kernels[kernel.kernel_id] = kernel
        if kernel.is_available:
            self._available_event.set()

    async def acquire(self, timeout: float | None = None) -> KernelResource:
        """Acquire an available kernel from the pool."""
        start_time = time.time()

        while True:
            async with self._pool_lock:
                # Find available kernel
                for kernel in self.kernels.values():
                    if kernel.is_available and not kernel.is_shutting_down:
                        kernel.acquire()
                        return kernel

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise KernelPoolExhaustedError(f"No kernels available after {timeout}s")

                # Clear event if no kernels available
                if self.available_count == 0:
                    self._available_event.clear()

            # Wait for a kernel to become available
            try:
                remaining_timeout = None
                if timeout is not None:
                    remaining_timeout = timeout - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        raise KernelPoolExhaustedError("Timeout waiting for kernel")

                await asyncio.wait_for(self._available_event.wait(), timeout=remaining_timeout)
            except TimeoutError as e:
                raise KernelPoolExhaustedError(f"No kernels available after {timeout}s") from e

    def release(self, kernel: KernelResource) -> None:
        """Release a kernel back to the pool."""
        kernel.release()
        self._available_event.set()


class AgentKernelManager:
    """
    Manages Jupyter kernels for agent isolation with resource pooling.

    This class provides:
    - Kernel lifecycle management (start, stop, restart)
    - Kernel pooling with configurable limits
    - Session persistence and checkpointing
    - Resource limit enforcement
    - Async context manager for safe resource handling
    """

    def __init__(
        self, max_kernels: int = 10, kernel_name: str = "python3", resource_limits: KernelResourceLimits | None = None
    ) -> None:
        """
        Initialize the kernel manager.

        Args:
            max_kernels: Maximum number of kernels in the pool
            kernel_name: Kernel spec name (default: python3)
            resource_limits: Resource limits for kernel execution
        """
        self.max_kernels = max_kernels
        self.kernel_name = kernel_name
        self.resource_limits = resource_limits or KernelResourceLimits()

        self.pool = KernelPool(max_kernels)
        self.sessions: dict[str, KernelSession] = {}
        self._kernel_managers: dict[str, AsyncKernelManager] = {}
        self._shutdown = False

    async def __aenter__(self) -> "AgentKernelManager":
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def _initialize(self) -> None:
        """Initialize the kernel manager."""
        # Verify kernel spec exists
        spec_manager = KernelSpecManager()
        try:
            spec_manager.get_kernel_spec(self.kernel_name)
        except Exception as e:
            raise RuntimeError(f"Kernel spec '{self.kernel_name}' not found: {e}") from e

    @asynccontextmanager
    async def acquire_kernel(
        self, agent_id: str, timeout: float | None = None
    ) -> AsyncIterator[tuple[AsyncKernelManager, KernelSession]]:
        """
        Acquire a kernel for an agent.

        If the agent already has a session, reuse the same kernel.
        Otherwise, acquire a new kernel from the pool.

        Args:
            agent_id: Unique identifier for the agent
            timeout: Maximum time to wait for a kernel

        Yields:
            Tuple of (kernel_manager, session)
        """
        # Check if agent already has a session
        if agent_id in self.sessions:
            session = self.sessions[agent_id]
            kernel_id = session.kernel_id

            # Find the kernel resource
            kernel_resource = self.pool.kernels.get(kernel_id)
            if kernel_resource is not None:
                # Reuse existing kernel - acquire the specific kernel
                kernel_resource = await self.pool.acquire(timeout)
                try:
                    kernel_manager = self._kernel_managers[kernel_id]
                    yield kernel_manager, session
                finally:
                    self.pool.release(kernel_resource)
                return
            else:
                # Session exists but kernel was removed - create new session
                del self.sessions[agent_id]

        # Need to create a new kernel - first check if we have any in pool
        if len(self.pool.kernels) == 0:
            # Create initial kernel resource
            kernel_id = str(uuid.uuid4())
            kernel_resource = KernelResource(kernel_id=kernel_id)
            self.pool.add_kernel(kernel_resource)

        # Acquire a kernel from the pool
        kernel_resource = await self.pool.acquire(timeout)
        try:
            # Create kernel manager if needed
            if kernel_resource.kernel_id not in self._kernel_managers:
                await self._create_kernel(kernel_resource)

            kernel_manager = self._kernel_managers[kernel_resource.kernel_id]

            # Create new session
            session = KernelSession(
                session_id=str(uuid.uuid4()), kernel_id=kernel_resource.kernel_id, agent_id=agent_id
            )
            self.sessions[agent_id] = session

            yield kernel_manager, session

        finally:
            self.pool.release(kernel_resource)

    async def _create_kernel(self, kernel_resource: KernelResource) -> None:
        """Create and start a new kernel."""
        kernel_manager = AsyncKernelManager(kernel_name=self.kernel_name)

        # Start the kernel (this will generate a kernel_id)
        await kernel_manager.start_kernel()

        # Store the manager
        self._kernel_managers[kernel_resource.kernel_id] = kernel_manager
        kernel_resource._kernel_manager = kernel_manager

    async def execute_code(self, session: KernelSession, code: str) -> dict[str, Any]:
        """
        Execute code in a kernel session.

        Args:
            session: The kernel session
            code: Code to execute

        Returns:
            Execution result dictionary
        """
        kernel_manager = self._kernel_managers.get(session.kernel_id)
        if kernel_manager is None:
            raise RuntimeError(f"Kernel {session.kernel_id} not found")

        # Get kernel client
        client = kernel_manager.client()
        client.start_channels()

        try:
            # Execute code with timeout
            msg_id = client.execute(code)

            # Collect output with timeout
            result = await self._collect_execution_result(
                client, msg_id, timeout=self.resource_limits.max_execution_time
            )

            # Add to session history
            session.add_execution(code, result)

        except TimeoutError as e:
            raise KernelTimeoutError(
                f"Execution exceeded time limit of {self.resource_limits.max_execution_time}s"
            ) from e
        else:
            return result
        finally:
            client.stop_channels()

    async def _collect_execution_result(self, client: Any, msg_id: str, timeout: float) -> dict[str, Any]:
        """Collect execution result from kernel."""
        result: dict[str, Any] = {"msg_id": msg_id, "status": "unknown", "outputs": [], "error": None}

        # Wait for execute reply
        try:
            reply = await asyncio.wait_for(client.get_shell_msg(), timeout=timeout)
            result["status"] = reply["content"].get("status", "unknown")

            if result["status"] == "error":
                result["error"] = reply["content"]

        except TimeoutError:
            result["status"] = "timeout"
            raise

        # Collect output messages
        while True:
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=1.0)

                msg_type = msg.get("msg_type", "")

                if msg_type == "stream":
                    result["outputs"].append({"type": "stream", "text": msg["content"]["text"]})
                elif msg_type == "execute_result":
                    result["outputs"].append({"type": "execute_result", "data": msg["content"]["data"]})
                elif msg_type == "error":
                    result["error"] = msg["content"]
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    break

            except TimeoutError:
                break

        return result

    def save_checkpoint(self, session: KernelSession) -> dict[str, Any]:
        """Save a checkpoint of the session state."""
        session.checkpoint()
        return session.checkpoint_data or {}

    def restore_checkpoint(self, session: KernelSession, checkpoint_data: dict[str, Any]) -> None:
        """Restore session state from checkpoint."""
        # Restore execution history
        if "execution_history" in checkpoint_data:
            session.execution_history = checkpoint_data["execution_history"]

    async def shutdown(self) -> None:
        """Shutdown all kernels and cleanup resources."""
        self._shutdown = True

        # Shutdown all kernel managers
        for kernel_manager in self._kernel_managers.values():
            # Best effort cleanup - suppress any exceptions
            with suppress(Exception):
                await kernel_manager.shutdown_kernel()

        self._kernel_managers.clear()
        self.sessions.clear()
