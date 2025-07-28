"""
Generic Jupyter Kernel Service.

This module provides domain-agnostic kernel management functionality:
- Kernel lifecycle management (create, execute, shutdown)
- Resource limits and monitoring
- Session management and isolation
- Async execution with timeout handling

No domain-specific logic - pure kernel execution primitives.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from jupyter_client import AsyncKernelManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelProfile:
    """
    Configuration profile for kernel creation and resource management.

    This is the generic version of the original KernelResourceLimits,
    expanded to include kernel environment configuration.

    Args:
        name: Kernel spec name (e.g., 'python3', 'julia-1.6')
        max_cpu_percent: Maximum CPU usage as percentage (0-100)
        max_memory_mb: Maximum memory usage in megabytes
        max_execution_time: Maximum time for a single code execution in seconds
        max_output_size_mb: Maximum size of output data in megabytes
        idle_timeout_seconds: How long to keep idle kernels before shutting them down
        env_vars: Additional environment variables for the kernel
        working_dir: Working directory for the kernel process
        output_drain_timeout_ms: Initial timeout for post-idle output collection (milliseconds)
        output_drain_max_timeout_ms: Maximum timeout for post-idle output collection (milliseconds)
        output_drain_max_attempts: Maximum empty reads before stopping output collection
        wait_for_shell_reply: Whether to wait for shell reply after execution
    """

    name: str = "python3"
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 1024
    max_execution_time: float = 30.0
    max_output_size_mb: int = 10
    idle_timeout_seconds: float = 300.0
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: Path | None = None
    # Output collection tuning
    output_drain_timeout_ms: int = 150  # Initial drain timeout in milliseconds
    output_drain_max_timeout_ms: int = 1000  # Maximum drain timeout in milliseconds
    output_drain_max_attempts: int = 3  # Maximum consecutive empty reads before stopping
    wait_for_shell_reply: bool = True  # Whether to wait for shell reply confirmation

    def __post_init__(self) -> None:
        """Validate resource limits are reasonable."""
        if self.max_cpu_percent < 0 or self.max_cpu_percent > 100:
            raise ValueError("CPU percent must be between 0 and 100")
        if self.max_memory_mb < 0:
            raise ValueError("Memory limit must be positive")
        if self.max_execution_time < 0:
            raise ValueError("Execution time must be positive")


@dataclass
class ExecutionResult:
    """
    Result of code execution in a kernel.

    Args:
        status: Execution status - 'ok', 'error', or 'timeout'
        outputs: List of output objects (stdout, display_data, etc.)
        error: Error information if status is 'error'
        execution_count: Kernel execution counter
        msg_id: Message ID for debugging
        duration_seconds: How long the execution took
    """

    status: str
    outputs: list[dict[str, Any]]
    error: dict[str, Any] | None = None
    execution_count: int = 1
    msg_id: str = ""
    duration_seconds: float = 0.0


class KernelTimeoutError(Exception):
    """Raised when kernel execution exceeds time limit."""

    pass


class KernelResourceLimitError(Exception):
    """Raised when kernel exceeds resource limits."""

    pass


class KernelService:
    """
    Generic kernel service for managing Jupyter kernels.

    This service provides domain-agnostic kernel management with:
    - Session-based kernel isolation
    - Resource monitoring and limits
    - Async execution with timeout handling
    - Automatic cleanup of idle sessions

    No domain-specific logic - pure kernel execution primitives.
    """

    def __init__(self, profile: KernelProfile, max_sessions: int = 10):
        """
        Initialize the kernel service.

        Args:
            profile: Kernel configuration profile
            max_sessions: Maximum number of concurrent sessions
        """
        self.profile = profile
        self.max_sessions = max_sessions
        self._sessions: dict[str, Any] = {}
        self._session_last_used: dict[str, float] = {}
        self._kernel_manager: AsyncKernelManager | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False

    async def __aenter__(self) -> "KernelService":
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def _initialize(self) -> None:
        """Initialize the kernel manager and start cleanup task."""
        self._kernel_manager = AsyncKernelManager(kernel_name=self.profile.name)

        # Set environment variables if provided
        if self.profile.env_vars:
            self._kernel_manager.env = self.profile.env_vars

        # Set working directory if provided
        if self.profile.working_dir:
            self._kernel_manager.cwd = str(self.profile.working_dir)

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_sessions())

    async def create_session(self, session_id: str) -> str:
        """
        Create a new kernel session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Kernel ID for the session

        Raises:
            ValueError: If session_id already exists
            RuntimeError: If max sessions exceeded
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self.max_sessions}) exceeded")

        if not self._kernel_manager:
            raise RuntimeError("Kernel service not initialized")

        # Start kernel
        await self._kernel_manager.start_kernel()
        client = self._kernel_manager.client()
        await client.wait_for_ready(timeout=30)

        # Store session
        self._sessions[session_id] = {
            "kernel_manager": self._kernel_manager,
            "client": client,
            "created_at": time.time(),
        }
        self._session_last_used[session_id] = time.time()

        logger.info(f"Created kernel session {session_id}")
        return session_id

    async def execute(self, session_id: str, code: str) -> ExecutionResult:
        """
        Execute code in a kernel session using nbclient.

        Args:
            session_id: Kernel session to use
            code: Python code to execute

        Returns:
            ExecutionResult with outputs and status

        Raises:
            ValueError: If session_id doesn't exist
            KernelTimeoutError: If execution times out
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]
        self._session_last_used[session_id] = time.time()

        start_time = time.time()

        try:
            # Get the kernel client from the session
            kernel_client = session["client"]

            # Execute code directly on the kernel
            msg_id = kernel_client.execute(code)

            # Collect outputs
            outputs = []
            execution_count = None
            error = None

            # Process messages until execution is complete
            timeout_remaining = self.profile.max_execution_time
            while True:
                try:
                    msg = await kernel_client.get_iopub_msg(timeout=timeout_remaining)
                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "stream":
                        outputs.append(
                            {"type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")}
                        )
                    elif msg_type == "execute_result":
                        execution_count = content.get("execution_count", 1)
                        outputs.append(
                            {
                                "type": "execute_result",
                                "execution_count": execution_count,
                                "data": content.get("data", {}),
                                "metadata": content.get("metadata", {}),
                            }
                        )
                    elif msg_type == "display_data":
                        outputs.append(
                            {
                                "type": "display_data",
                                "data": content.get("data", {}),
                                "metadata": content.get("metadata", {}),
                            }
                        )
                    elif msg_type == "error":
                        error = {
                            "ename": content.get("ename", "Error"),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", []),
                        }
                    elif msg_type == "status" and content.get("execution_state") == "idle":
                        # Execution complete
                        break

                except TimeoutError:
                    raise KernelTimeoutError(f"Execution timed out after {self.profile.max_execution_time}s")

            # Get execution reply
            reply = await kernel_client.get_shell_msg(timeout=self.profile.max_execution_time)
            status = reply["content"]["status"]

            duration = time.time() - start_time

            return ExecutionResult(
                status="error" if error else status,
                outputs=outputs,
                error=error,
                duration_seconds=duration,
                execution_count=execution_count or 1,
            )

        except KernelTimeoutError as e:
            duration = time.time() - start_time
            logger.error(f"Execution timed out for session {session_id}: {e}")
            return ExecutionResult(
                status="error",
                outputs=[],
                error={"ename": "TimeoutError", "evalue": str(e), "traceback": []},
                duration_seconds=duration,
            )
        except (TimeoutError, Empty):
            # Handle both asyncio.TimeoutError and queue.Empty as timeout
            duration = time.time() - start_time
            logger.error(f"Execution timed out for session {session_id}")
            return ExecutionResult(
                status="error",
                outputs=[],
                error={
                    "ename": "TimeoutError",
                    "evalue": f"Execution timed out after {self.profile.max_execution_time}s",
                    "traceback": [],
                },
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed for session {session_id}: {type(e).__name__}: {e}")

            # Return error result
            return ExecutionResult(
                status="error",
                outputs=[],
                error={"ename": type(e).__name__, "evalue": str(e), "traceback": []},
                duration_seconds=duration,
            )

    def _convert_nbclient_results(self, cell: Any, duration: float) -> ExecutionResult:
        """
        Convert nbclient cell results to ExecutionResult format.

        Args:
            cell: Executed notebook cell from nbclient
            duration: Execution duration in seconds

        Returns:
            ExecutionResult with converted outputs
        """
        outputs = []
        error = None
        status = "ok"

        # Process cell outputs
        for output in cell.outputs:
            output_type = output.get("output_type", "unknown")

            if output_type == "stream":
                outputs.append({"type": "stream", "name": output.get("name", "stdout"), "text": output.get("text", "")})
            elif output_type == "execute_result":
                outputs.append(
                    {
                        "type": "execute_result",
                        "execution_count": output.get("execution_count", 1),
                        "data": output.get("data", {}),
                        "metadata": output.get("metadata", {}),
                    }
                )
            elif output_type == "display_data":
                outputs.append(
                    {"type": "display_data", "data": output.get("data", {}), "metadata": output.get("metadata", {})}
                )
            elif output_type == "error":
                error = {
                    "ename": output.get("ename", "Error"),
                    "evalue": output.get("evalue", "Unknown error"),
                    "traceback": output.get("traceback", []),
                }
                status = "error"
                # Also add error to outputs for notebook compatibility
                outputs.append(
                    {
                        "type": "error",
                        "ename": output.get("ename", "Error"),
                        "evalue": output.get("evalue", "Unknown error"),
                        "traceback": output.get("traceback", []),
                    }
                )

        return ExecutionResult(
            status=status, outputs=outputs, error=error, execution_count=cell.execution_count, duration_seconds=duration
        )

    async def execute_for_notebook(self, session_id: str, code: str) -> list[dict[str, Any]]:
        """
        Execute code and return notebook-compatible outputs.

        Args:
            session_id: Kernel session to use
            code: Python code to execute

        Returns:
            List of output dictionaries in notebook format
        """
        result = await self.execute(session_id, code)
        return result.outputs

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        """
        Get information about a kernel session.

        Args:
            session_id: Session identifier

        Returns:
            Session information dictionary
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_used": self._session_last_used[session_id],
            "kernel_id": session["kernel_manager"].kernel_id,
        }

    async def close_session(self, session_id: str) -> None:
        """
        Close a kernel session.

        Args:
            session_id: Session identifier
        """
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]

        try:
            # Shutdown kernel
            await session["kernel_manager"].shutdown_kernel()
            logger.info(f"Closed kernel session {session_id}")
        except Exception as e:
            logger.warning(f"Error closing session {session_id}: {e}")
        finally:
            # Remove from tracking
            del self._sessions[session_id]
            if session_id in self._session_last_used:
                del self._session_last_used[session_id]

    async def list_sessions(self) -> list[str]:
        """
        List all active session IDs.

        Returns:
            List of session identifiers
        """
        return list(self._sessions.keys())

    async def _cleanup_idle_sessions(self) -> None:
        """Background task to cleanup idle sessions."""
        while not self._shutdown:
            try:
                current_time = time.time()
                idle_sessions = []

                for session_id, last_used in self._session_last_used.items():
                    if current_time - last_used > self.profile.idle_timeout_seconds:
                        idle_sessions.append(session_id)

                for session_id in idle_sessions:
                    await self.close_session(session_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception:
                # Don't let cleanup task crash the service
                await asyncio.sleep(60)

    async def shutdown(self) -> None:
        """Shutdown the service and all sessions."""
        self._shutdown = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Close all sessions
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)
