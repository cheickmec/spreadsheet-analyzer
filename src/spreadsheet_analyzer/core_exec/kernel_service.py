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
from typing import Any

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbformat.v4 import new_output

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
    Generic Jupyter kernel management service.

    Provides low-level kernel execution without domain-specific logic.
    Handles kernel lifecycle, resource monitoring, and execution isolation.

    Usage:
        async with KernelService(profile) as service:
            session_id = await service.create_session("agent-1")
            result = await service.execute(session_id, "print('hello')")
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

        # Session management
        self._sessions: dict[str, AsyncKernelManager] = {}
        self._clients: dict[str, Any] = {}  # Kernel clients per session
        self._session_last_used: dict[str, float] = {}
        self._execution_counts: dict[str, int] = {}

        # State management
        self._shutdown = False
        self._cleanup_task: asyncio.Task | None = None

    async def __aenter__(self) -> "KernelService":
        """Initialize the service."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shutdown the service and cleanup resources."""
        await self.shutdown()

    async def _initialize(self) -> None:
        """Initialize the kernel service."""
        # Verify kernel spec exists
        spec_manager = KernelSpecManager()
        try:
            spec_manager.get_kernel_spec(self.profile.name)
        except Exception as e:
            raise RuntimeError(f"Kernel spec '{self.profile.name}' not found: {e}") from e

        # Start cleanup task for idle sessions
        if self.profile.idle_timeout_seconds > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_sessions())

    async def create_session(self, session_id: str) -> str:
        """
        Create a new kernel session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            The session ID that was created

        Raises:
            RuntimeError: If max sessions exceeded or session already exists
        """
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self.max_sessions}) exceeded")

        if session_id in self._sessions:
            raise RuntimeError(f"Session '{session_id}' already exists")

        # Create kernel manager
        kernel_manager = AsyncKernelManager(kernel_name=self.profile.name)

        # Configure kernel environment
        if self.profile.working_dir:
            kernel_manager.kernel_cwd = str(self.profile.working_dir)

        if self.profile.env_vars:
            if kernel_manager.kernel_env is None:
                kernel_manager.kernel_env = {}
            kernel_manager.kernel_env.update(self.profile.env_vars)

        # Start the kernel
        await kernel_manager.start_kernel()

        # Wait for kernel to be ready
        await asyncio.sleep(0.2)

        # Verify kernel is alive
        if not await kernel_manager.is_alive():
            raise RuntimeError(f"Failed to start kernel for session '{session_id}'")

        # Create and start client
        client = kernel_manager.client()
        client.start_channels()

        # Wait for channels to be ready
        await asyncio.sleep(0.5)  # Increased from 0.1 to match working example

        # Warm up the kernel with a simple execution
        warmup_msg_id = client.execute("pass")
        warmup_deadline = time.time() + 2.0
        while time.time() < warmup_deadline:
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.1)
                if (
                    msg.get("parent_header", {}).get("msg_id") == warmup_msg_id
                    and msg.get("msg_type") == "status"
                    and msg.get("content", {}).get("execution_state") == "idle"
                ):
                    break
            except TimeoutError:
                pass

        # Store session info
        self._sessions[session_id] = kernel_manager
        self._clients[session_id] = client
        self._session_last_used[session_id] = time.time()
        self._execution_counts[session_id] = 0

        return session_id

    async def execute(self, session_id: str, code: str) -> ExecutionResult:
        """
        Execute code in a kernel session.

        Args:
            session_id: Session to execute code in
            code: Python code to execute

        Returns:
            ExecutionResult with outputs and status

        Raises:
            RuntimeError: If session not found
            KernelTimeoutError: If execution times out
            KernelResourceLimitError: If resource limits exceeded
        """
        if session_id not in self._sessions:
            raise RuntimeError(f"Session '{session_id}' not found")

        kernel_manager = self._sessions[session_id]
        client = self._clients.get(session_id)

        if not client:
            raise RuntimeError(f"No client found for session '{session_id}'")

        start_time = time.time()

        try:
            # Verify kernel is alive
            if not await kernel_manager.is_alive():
                raise RuntimeError(f"Kernel for session '{session_id}' is not alive")

            # Clear pending messages
            await self._clear_pending_messages(client)

            # Execute code
            logger.debug(f"Executing code in session {session_id}: {code[:50]}...")
            msg_id = client.execute(code)
            logger.debug(f"Execution started with msg_id: {msg_id}")

            # Collect results with timeout
            result = await self._collect_execution_result(client, msg_id, self.profile.max_execution_time)

            # Update session tracking
            self._session_last_used[session_id] = time.time()
            self._execution_counts[session_id] += 1

            # Create result object
            return ExecutionResult(
                status=result.get("status", "unknown"),
                outputs=result.get("outputs", []),
                error=result.get("error"),
                execution_count=self._execution_counts[session_id],
                msg_id=msg_id,
                duration_seconds=time.time() - start_time,
            )

        except TimeoutError as e:
            raise KernelTimeoutError(f"Execution exceeded time limit of {self.profile.max_execution_time}s") from e

    async def execute_for_notebook(self, session_id: str, code: str) -> list[dict[str, Any]]:
        """
        Execute code and return notebook-ready output objects.

        Args:
            session_id: Session to execute code in
            code: Python code to execute

        Returns:
            List of nbformat-compatible output objects
        """
        result = await self.execute(session_id, code)

        # Convert to notebook format
        notebook_outputs = []

        if result.status == "ok":
            for output in result.outputs:
                if isinstance(output, dict):
                    output_type = output.get("type")

                    if output_type == "stream":
                        nb_output = new_output(output_type="stream", name="stdout", text=[output.get("text", "")])
                        notebook_outputs.append(nb_output)

                    elif output_type == "execute_result":
                        nb_output = new_output(
                            output_type="execute_result",
                            execution_count=result.execution_count,
                            data=output.get("data", {}),
                            metadata={},
                        )
                        notebook_outputs.append(nb_output)

                    elif output_type == "display_data":
                        nb_output = new_output(
                            output_type="display_data", data=output.get("data", {}), metadata=output.get("metadata", {})
                        )
                        notebook_outputs.append(nb_output)

                    elif output_type == "error":
                        nb_output = new_output(
                            output_type="error",
                            ename=output.get("ename", "Error"),
                            evalue=output.get("evalue", "Unknown error"),
                            traceback=output.get("traceback", []),
                        )
                        notebook_outputs.append(nb_output)

        elif result.status == "error" and result.error:
            nb_output = new_output(
                output_type="error",
                ename=result.error.get("ename", "ExecutionError"),
                evalue=result.error.get("evalue", "Code execution failed"),
                traceback=result.error.get("traceback", []),
            )
            notebook_outputs.append(nb_output)

        return notebook_outputs

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get information about a session."""
        if session_id not in self._sessions:
            raise RuntimeError(f"Session '{session_id}' not found")

        return {
            "session_id": session_id,
            "execution_count": self._execution_counts.get(session_id, 0),
            "last_used": self._session_last_used.get(session_id, 0),
            "is_alive": await self._sessions[session_id].is_alive(),
        }

    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        if session_id not in self._sessions:
            return

        # Stop client channels
        if session_id in self._clients:
            client = self._clients[session_id]
            client.stop_channels()
            del self._clients[session_id]

        kernel_manager = self._sessions[session_id]
        await kernel_manager.shutdown_kernel()

        # Cleanup tracking
        del self._sessions[session_id]
        self._session_last_used.pop(session_id, None)
        self._execution_counts.pop(session_id, None)

    async def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    async def _clear_pending_messages(self, client: Any) -> None:
        """Clear any pending messages from previous executions."""
        while True:
            try:
                await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
            except TimeoutError:
                break

    async def _collect_execution_result(self, client: Any, msg_id: str, timeout: float) -> dict[str, Any]:
        """Collect execution results from kernel with robust output capture."""
        outputs = []
        error = None
        status = "ok"
        idle_received = False
        late_message_count = 0
        all_messages_seen = []  # Track all messages for debugging

        # Wait for execution to complete
        start_time = time.time()
        deadline = start_time + timeout

        # Brief initial delay to allow kernel to start processing
        await asyncio.sleep(0.05)  # Increased from 0.01

        # Phase 1: Collect messages until idle status
        first_message_seen = False
        while time.time() < deadline:
            try:
                # Use profile-configured timeout for initial messages
                if not first_message_seen:
                    # Use the configured initial drain timeout
                    timeout_val = self.profile.output_drain_timeout_ms / 1000.0
                else:
                    timeout_val = min(1.0, deadline - time.time())
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=timeout_val)
                first_message_seen = True

                # Debug logging
                msg_type = msg.get("msg_type", "unknown")
                parent_msg_id = msg.get("parent_header", {}).get("msg_id", "none")
                all_messages_seen.append(
                    {
                        "msg_type": msg_type,
                        "parent_msg_id": parent_msg_id,
                        "our_msg_id": msg_id,
                        "matches": parent_msg_id == msg_id,
                    }
                )

                # Check if this message is for our execution
                if parent_msg_id != msg_id:
                    logger.debug(f"Skipping message with parent_msg_id={parent_msg_id}, our msg_id={msg_id}")
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    outputs.append(
                        {"type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")}
                    )

                elif msg_type == "execute_result":
                    outputs.append(
                        {
                            "type": "execute_result",
                            "execution_count": content.get("execution_count", 1),
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
                        "evalue": content.get("evalue", "Unknown error"),
                        "traceback": content.get("traceback", []),
                    }
                    status = "error"
                    # Also add error to outputs for notebook compatibility
                    outputs.append(
                        {
                            "type": "error",
                            "ename": content.get("ename", "Error"),
                            "evalue": content.get("evalue", "Unknown error"),
                            "traceback": content.get("traceback", []),
                        }
                    )

                elif msg_type == "status" and content.get("execution_state") == "idle":
                    # Execution completed - but don't break yet
                    idle_received = True
                    break

            except TimeoutError:
                if not first_message_seen and len(all_messages_seen) == 0:
                    elapsed = time.time() - start_time
                    logger.warning(f"No messages received at all for msg_id={msg_id} after {elapsed:.2f}s")
                if not idle_received:
                    status = "timeout"
                break

        # Phase 2: Post-idle drain with exponential backoff
        if idle_received and time.time() < deadline:
            empty_reads = 0
            drain_timeout = self.profile.output_drain_timeout_ms / 1000.0  # Convert to seconds
            max_drain_timeout = self.profile.output_drain_max_timeout_ms / 1000.0
            attempts = 0
            min_drain_attempts = 2  # Always try at least 2 drain attempts

            while (
                empty_reads < self.profile.output_drain_max_attempts or attempts < min_drain_attempts
            ) and time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=drain_timeout)

                    # Check if this message is for our execution
                    if msg.get("parent_header", {}).get("msg_id") != msg_id:
                        # Still reset the timeout since we got a message
                        empty_reads = 0
                        drain_timeout = self.profile.output_drain_timeout_ms / 1000.0
                        continue

                    # Process any late-arriving message
                    msg_type = msg["msg_type"]
                    content = msg["content"]
                    late_message_count += 1

                    if msg_type == "stream":
                        outputs.append(
                            {"type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")}
                        )
                    elif msg_type == "execute_result":
                        outputs.append(
                            {
                                "type": "execute_result",
                                "execution_count": content.get("execution_count", 1),
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
                        # Display data (like plots) might have more outputs coming
                        # Give extra time for matplotlib/plotting outputs
                        if "image/png" in content.get("data", {}):
                            drain_timeout = max(drain_timeout, 0.2)  # At least 200ms for plots

                    # Reset counter since we got a message
                    empty_reads = 0
                    drain_timeout = self.profile.output_drain_timeout_ms / 1000.0

                except TimeoutError:
                    empty_reads += 1
                    attempts += 1
                    # Exponential backoff
                    drain_timeout = min(drain_timeout * 2, max_drain_timeout)

            if late_message_count > 0:
                logger.debug(
                    f"Collected {late_message_count} messages after idle status "
                    f"(msg_id: {msg_id}, total outputs: {len(outputs)})"
                )

        # Phase 3: Wait for shell reply if configured
        if idle_received and self.profile.wait_for_shell_reply and time.time() < deadline:
            try:
                shell_reply = await asyncio.wait_for(client.get_shell_msg(), timeout=min(0.5, deadline - time.time()))
                # Shell reply provides additional execution metadata
                if shell_reply.get("content", {}).get("status") == "error" and not error:
                    # Update error info from shell reply if not already set
                    reply_content = shell_reply.get("content", {})
                    error = {
                        "ename": reply_content.get("ename", "Error"),
                        "evalue": reply_content.get("evalue", "Unknown error"),
                        "traceback": reply_content.get("traceback", []),
                    }
                    status = "error"
            except TimeoutError:
                # Shell reply timeout is not critical
                logger.debug(f"Shell reply timeout for msg_id: {msg_id}")

        # Log debugging info for empty outputs
        if len(outputs) == 0:
            logger.warning(
                f"No outputs collected for msg_id={msg_id}. "
                f"Total messages seen: {len(all_messages_seen)}, "
                f"Matching messages: {sum(1 for m in all_messages_seen if m['matches'])}"
            )
            for i, msg_info in enumerate(all_messages_seen):
                logger.debug(f"Message {i}: {msg_info}")

        return {"status": status, "outputs": outputs, "error": error, "msg_id": msg_id}

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
