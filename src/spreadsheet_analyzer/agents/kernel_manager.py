"""
Jupyter Kernel Manager for Agent Isolation and Execution

This module provides a sophisticated system for managing Jupyter kernels to enable
isolated Python code execution for multiple AI agents. It's designed to handle
concurrent execution, resource management, and session persistence in a production
environment.

================================================================================
BACKGROUND INFORMATION FOR JUNIOR DEVELOPERS
================================================================================

What is a Jupyter Kernel?
-------------------------
A Jupyter kernel is essentially a separate Python process that can execute code
and maintain state (variables, imports, etc.) across multiple code executions.
Think of it like having a Python interpreter running in the background that
remembers everything you've done.

Why do we need this?
--------------------
In our spreadsheet analyzer application, we have multiple AI agents that need to:
1. Execute Python code to analyze Excel files
2. Maintain state between different code executions (e.g., variables, imports)
3. Run concurrently without interfering with each other
4. Handle errors gracefully without crashing the entire system

The Problem We're Solving:
--------------------------
Without proper kernel management, we'd face several issues:
- All agents would share the same Python process, causing conflicts
- One agent's error could crash the entire system
- No way to limit resource usage (CPU, memory, execution time)
- No persistence of execution state between runs
- Poor performance due to starting new Python processes for each execution

Our Solution:
-------------
This kernel manager creates and manages a pool of Jupyter kernels, where:
- Each agent gets its own isolated kernel
- Kernels can be reused across multiple executions
- Resource limits prevent runaway processes
- Sessions persist state between executions
- Concurrent execution is handled safely

================================================================================
KEY CONCEPTS EXPLAINED
================================================================================

1. Kernel Pool:
   A collection of available kernels that can be assigned to agents.
   Like a car rental service - you have a limited number of cars (kernels)
   that can be checked out by customers (agents).

2. Kernel Session:
   A connection between an agent and a specific kernel. It tracks:
   - Which kernel the agent is using
   - Execution history (what code was run)
   - State persistence across executions

3. Resource Limits:
   Safety mechanisms to prevent kernels from consuming too many resources:
   - CPU usage limits
   - Memory limits
   - Execution time limits
   - Output size limits

4. Async Context Managers:
   Python's way of ensuring resources are properly cleaned up.
   The `async with` syntax automatically handles setup and cleanup.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

AgentKernelManager (Main Class)
├── KernelPool (Manages available kernels)
│   └── KernelResource (Individual kernel wrapper)
├── KernelSession (Agent-kernel connection)
└── KernelResourceLimits (Safety constraints)

Flow:
1. Agent requests a kernel via `acquire_kernel()`
2. KernelPool provides an available kernel
3. Agent executes code via `execute_code()`
4. Results are collected and returned
5. Kernel is returned to pool for reuse

================================================================================
USAGE EXAMPLE
================================================================================

```python
# Create a kernel manager with limits
manager = AgentKernelManager(
    max_kernels=5,  # Maximum 5 kernels in pool
    resource_limits=KernelResourceLimits(
        max_execution_time=30.0,  # 30 second timeout
        max_memory_mb=1024,       # 1GB memory limit
    )
)

# Use as async context manager for automatic cleanup
async with manager:
    # Acquire a kernel for an agent
    async with manager.acquire_kernel("agent-123") as (kernel, session):
        # Execute code
        result = await manager.execute_code(session, "print('Hello, World!')")
        print(result["outputs"])  # See the output

        # Execute more code (state persists)
        result = await manager.execute_code(session, "x = 42; print(x)")
        print(result["outputs"])  # Prints "42"
```

================================================================================
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
from structlog import get_logger

logger = get_logger(__name__)


class KernelTimeoutError(Exception):
    """
    Raised when kernel execution exceeds the configured time limit.

    This is a safety mechanism to prevent infinite loops or very long-running
    code from consuming resources indefinitely.

    Example:
        If you set max_execution_time=30.0 and code runs for 31 seconds,
        this exception will be raised.
    """

    pass


class KernelPoolExhaustedError(Exception):
    """
    Raised when no kernels are available in the pool.

    This happens when:
    1. All kernels are currently in use by other agents
    2. The maximum number of kernels has been reached
    3. A timeout occurs while waiting for a kernel to become available

    This is different from KernelTimeoutError because it's about resource
    availability, not execution time.
    """

    pass


@dataclass(frozen=True)
class KernelResourceLimits:
    """
    Configuration for resource limits to prevent runaway processes.

    These limits act as safety mechanisms to ensure that no single kernel
    can consume excessive system resources. Think of them as circuit breakers.

    Attributes:
        max_cpu_percent: Maximum CPU usage as percentage (0-100)
        max_memory_mb: Maximum memory usage in megabytes
        max_execution_time: Maximum time for a single code execution in seconds
        max_output_size_mb: Maximum size of output data in megabytes
        idle_timeout_seconds: How long to keep idle kernels before shutting them down

    Example:
        KernelResourceLimits(
            max_cpu_percent=80.0,      # Don't use more than 80% CPU
            max_memory_mb=1024,        # Don't use more than 1GB RAM
            max_execution_time=30.0,   # Stop execution after 30 seconds
            max_output_size_mb=10,     # Limit output to 10MB
            idle_timeout_seconds=300.0 # Shutdown idle kernels after 5 minutes
        )
    """

    max_cpu_percent: float = 80.0
    max_memory_mb: int = 1024
    max_execution_time: float = 30.0
    max_output_size_mb: int = 10
    idle_timeout_seconds: float = 300.0  # 5 minutes default

    def __post_init__(self) -> None:
        """
        Validate that resource limits are reasonable.

        This method runs automatically after the dataclass is created.
        It ensures that the limits make sense and prevents configuration errors.
        """
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
    """
    Represents a single kernel resource in the pool.

    This class wraps a Jupyter kernel and tracks its state (available, in use,
    shutting down, etc.). Think of it as a "car" in our car rental analogy.

    Key Concepts:
    - is_available: Whether this kernel can be assigned to an agent
    - is_shutting_down: Whether this kernel is being terminated
    - _kernel_manager: The actual Jupyter kernel process
    - _acquisition_lock: Prevents race conditions when multiple agents try to use the same kernel

    Thread Safety:
    The _acquisition_lock ensures that only one agent can acquire a kernel at a time,
    preventing race conditions that could lead to multiple agents using the same kernel.
    """

    kernel_id: str
    is_available: bool = True
    is_shutting_down: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    _kernel_manager: AsyncKernelManager | None = None
    _acquisition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def acquire(self) -> None:
        """
        Mark this kernel as in use.

        This is like checking out a car from the rental service.
        If the kernel is already in use, this raises an error.
        """
        if not self.is_available:
            raise RuntimeError("Kernel is not available")
        self.is_available = False
        self.last_used_at = time.time()

    def release(self) -> None:
        """
        Mark this kernel as available for reuse.

        This is like returning a car to the rental service.
        If the kernel is already available, this raises an error.
        """
        if self.is_available:
            raise RuntimeError("Kernel is already available")
        self.is_available = True

    async def __aenter__(self) -> "KernelResource":
        """
        Async context manager entry - automatically acquires the kernel.

        This allows you to use the kernel with the `async with` syntax:

        async with kernel_resource:
            # Kernel is automatically acquired here
            # ... use the kernel ...
        # Kernel is automatically released here
        """
        self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Async context manager exit - automatically releases the kernel.

        This ensures the kernel is always released, even if an exception occurs.
        """
        self.release()


@dataclass
class KernelSession:
    """
    Represents an agent's connection to a specific kernel.

    This tracks the relationship between an agent and a kernel, including:
    - Which kernel the agent is using
    - History of all code executions
    - Checkpoint data for state restoration

    Think of this as a "rental agreement" between an agent and a kernel.
    The session persists even when the agent isn't actively using the kernel,
    allowing state to be maintained across multiple interactions.

    Key Features:
    - execution_history: Complete log of all code executions
    - checkpoint_data: Snapshot of session state for restoration
    - agent_id: Links the session to a specific agent
    """

    session_id: str
    kernel_id: str
    agent_id: str
    created_at: float = field(default_factory=time.time)
    last_checkpoint: float | None = None
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_data: dict[str, Any] | None = None

    def add_execution(self, code: str, result: dict[str, Any]) -> None:
        """
        Add an execution to the session history.

        This maintains a complete audit trail of what code was executed
        and what the results were. This is useful for:
        - Debugging issues
        - Understanding agent behavior
        - Restoring session state
        """
        self.execution_history.append({"timestamp": time.time(), "code": code, "result": result})

    def checkpoint(self) -> None:
        """
        Create a checkpoint of the current session state.

        A checkpoint is like taking a snapshot of the session that can be
        restored later. This is useful for:
        - Saving session state before shutting down
        - Migrating sessions between different kernel managers
        - Debugging and analysis
        """
        self.last_checkpoint = time.time()
        self.checkpoint_data = {
            "session_id": self.session_id,
            "kernel_id": self.kernel_id,
            "agent_id": self.agent_id,
            "checkpoint_time": self.last_checkpoint,
            "execution_history": self.execution_history.copy(),
        }


class KernelPool:
    """
    Manages a pool of available kernel resources.

    This class implements the "car rental service" pattern where:
    - A limited number of kernels are available
    - Agents can check out kernels when needed
    - Kernels are returned to the pool when done
    - The pool handles waiting when no kernels are available

    Thread Safety:
    This class uses asyncio.Lock to ensure thread-safe operations when
    multiple agents are trying to acquire kernels simultaneously.

    Performance Optimizations:
    - Uses asyncio.Event for efficient waiting when no kernels are available
    - Implements timeout handling to prevent indefinite waiting
    - Tracks kernel availability to avoid unnecessary waiting
    """

    def __init__(self, max_kernels: int = 10) -> None:
        """
        Initialize the kernel pool.

        Args:
            max_kernels: Maximum number of kernels that can be in the pool
        """
        self.max_kernels = max_kernels
        self.kernels: dict[str, KernelResource] = {}
        self._pool_lock = asyncio.Lock()
        self._available_event = asyncio.Event()

    @property
    def available_count(self) -> int:
        """
        Get the number of kernels currently available.

        This is useful for monitoring pool utilization and debugging.
        """
        return sum(1 for k in self.kernels.values() if k.is_available)

    def add_kernel(self, kernel: KernelResource) -> None:
        """
        Add a kernel to the pool.

        Args:
            kernel: The kernel resource to add

        Raises:
            ValueError: If a kernel with the same ID already exists
        """
        if kernel.kernel_id in self.kernels:
            raise ValueError(f"Kernel with ID {kernel.kernel_id} already exists")
        self.kernels[kernel.kernel_id] = kernel
        if kernel.is_available:
            self._available_event.set()

    async def acquire(self, timeout: float | None = None) -> KernelResource:
        """
        Acquire an available kernel from the pool.

        This method will wait until a kernel becomes available, or until
        the timeout is reached. It implements a fair waiting mechanism
        where multiple agents waiting for kernels will be served in order.

        Args:
            timeout: Maximum time to wait for a kernel (None = wait forever)

        Returns:
            An available kernel resource

        Raises:
            KernelPoolExhaustedError: If no kernel becomes available within timeout
        """
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
        """
        Release a kernel back to the pool.

        This makes the kernel available for other agents to use.
        It also signals any waiting agents that a kernel is now available.

        Args:
            kernel: The kernel resource to release
        """
        kernel.release()
        self._available_event.set()


class AgentKernelManager:
    """
    Main class for managing Jupyter kernels for agent isolation.

    This is the primary interface for the kernel management system. It provides:
    - Kernel lifecycle management (start, stop, restart)
    - Kernel pooling with configurable limits
    - Session persistence and checkpointing
    - Resource limit enforcement
    - Async context manager for safe resource handling

    Architecture:
    The manager coordinates between:
    - KernelPool: Manages available kernels
    - KernelSession: Tracks agent-kernel relationships
    - AsyncKernelManager: Actual Jupyter kernel processes

    Usage Pattern:
    ```python
    async with AgentKernelManager(max_kernels=5) as manager:
        async with manager.acquire_kernel("agent-123") as (kernel, session):
            result = await manager.execute_code(session, "print('Hello')")
    ```

    Key Features:
    1. Automatic Resource Management: Uses async context managers for cleanup
    2. Session Persistence: Agents can reuse the same kernel across executions
    3. Resource Limits: Prevents runaway processes
    4. Concurrent Execution: Multiple agents can run simultaneously
    5. Error Isolation: One agent's error doesn't affect others
    """

    def __init__(
        self, max_kernels: int = 10, kernel_name: str = "python3", resource_limits: KernelResourceLimits | None = None
    ) -> None:
        """
        Initialize the kernel manager.

        Args:
            max_kernels: Maximum number of kernels in the pool
            kernel_name: Name of the Jupyter kernel to use (e.g., "python3", "python2")
            resource_limits: Resource limits for kernel execution

        Note:
            The kernel_name must correspond to an installed Jupyter kernel.
            Common values are "python3", "python2", or custom kernel names.
            You can list available kernels with: jupyter kernelspec list
        """
        self.max_kernels = max_kernels
        self.kernel_name = kernel_name
        self.resource_limits = resource_limits or KernelResourceLimits()

        # Core components
        self.pool = KernelPool(max_kernels)
        self.sessions: dict[str, KernelSession] = {}
        self._kernel_managers: dict[str, AsyncKernelManager] = {}

        # State management
        self._shutdown = False
        self._eviction_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "AgentKernelManager":
        """
        Async context manager entry - initialize the manager.

        This automatically:
        1. Verifies the kernel spec exists
        2. Starts the idle kernel eviction task
        3. Prepares the manager for use
        """
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Async context manager exit - shutdown the manager.

        This ensures all kernels are properly shut down and resources are cleaned up,
        even if an exception occurs during execution.
        """
        await self.shutdown()

    async def _initialize(self) -> None:
        """
        Initialize the kernel manager.

        This method:
        1. Verifies that the specified kernel is available
        2. Starts background tasks for resource management
        3. Prepares the system for kernel creation

        Raises:
            RuntimeError: If the kernel spec is not found
        """
        # Verify kernel spec exists
        spec_manager = KernelSpecManager()
        try:
            spec_manager.get_kernel_spec(self.kernel_name)
        except Exception as e:
            raise RuntimeError(f"Kernel spec '{self.kernel_name}' not found: {e}") from e

        # Start idle kernel eviction task
        if self.resource_limits.idle_timeout_seconds > 0:
            self._eviction_task = asyncio.create_task(self._evict_idle_kernels())

    @asynccontextmanager
    async def acquire_kernel(
        self, agent_id: str, timeout: float | None = None
    ) -> AsyncIterator[tuple[AsyncKernelManager, KernelSession]]:
        """
        Acquire a kernel for an agent.

        This is the main method for getting a kernel to execute code. It implements
        intelligent kernel reuse:

        1. If the agent already has a session, reuse the same kernel
        2. If no session exists, acquire a new kernel from the pool
        3. If the agent's kernel is in use, wait for it to become available

        Args:
            agent_id: Unique identifier for the agent (e.g., "agent-123")
            timeout: Maximum time to wait for a kernel (None = wait forever)

        Yields:
            Tuple of (kernel_manager, session) for use in async context

        Raises:
            TimeoutError: If waiting for kernel exceeds timeout
            KernelPoolExhaustedError: If no kernels are available

        Example:
            ```python
            async with manager.acquire_kernel("agent-123") as (kernel, session):
                result = await manager.execute_code(session, "print('Hello')")
            ```
        """
        # Check if agent already has a session
        if agent_id in self.sessions:
            session = self.sessions[agent_id]
            kernel_id = session.kernel_id

            # Find the kernel resource
            kernel_resource = self.pool.kernels.get(kernel_id)
            if kernel_resource is not None:
                # Reuse existing kernel - acquire the specific kernel
                if kernel_resource.is_available:
                    kernel_resource.acquire()
                    try:
                        kernel_manager = self._kernel_managers[kernel_id]
                        yield kernel_manager, session
                    finally:
                        kernel_resource.release()
                    return
                else:
                    # Kernel is in use, wait for it to become available
                    # Wait for the kernel to become available with timeout
                    start_wait = time.time()
                    while not kernel_resource.is_available:
                        if timeout and (time.time() - start_wait) > timeout:
                            raise TimeoutError(f"Timeout waiting for kernel {kernel_id} to become available")
                        await asyncio.sleep(0.1)

                    # Now acquire the kernel
                    kernel_resource.acquire()
                    try:
                        kernel_manager = self._kernel_managers[kernel_id]
                        yield kernel_manager, session
                    finally:
                        kernel_resource.release()
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
            # Use per-kernel lock to prevent race conditions
            async with kernel_resource._acquisition_lock:
                # Create kernel manager if needed
                if kernel_resource.kernel_id not in self._kernel_managers:
                    await self._create_kernel(kernel_resource)

                kernel_manager = self._kernel_managers[kernel_resource.kernel_id]

                # Create new session - protected by kernel lock
                session = KernelSession(
                    session_id=str(uuid.uuid4()), kernel_id=kernel_resource.kernel_id, agent_id=agent_id
                )
                self.sessions[agent_id] = session

            yield kernel_manager, session

        finally:
            self.pool.release(kernel_resource)

    async def _create_kernel(self, kernel_resource: KernelResource) -> None:
        """
        Create and start a new kernel.

        This method:
        1. Creates a new AsyncKernelManager
        2. Starts the actual kernel process
        3. Updates the kernel resource with the correct kernel ID
        4. Warms up the kernel to ensure it's ready for execution

        Args:
            kernel_resource: The kernel resource to initialize

        Note:
            Kernel creation is expensive (takes several seconds), so we try to
            reuse kernels when possible. This method is only called when a new
            kernel is actually needed.
        """
        kernel_manager = AsyncKernelManager(kernel_name=self.kernel_name)

        # Start the kernel (this will generate a kernel_id)
        await kernel_manager.start_kernel()

        # Update kernel_resource with actual kernel ID to prevent mismatch
        actual_kernel_id = kernel_manager.kernel_id
        old_kernel_id = kernel_resource.kernel_id

        # Update the pool's kernel dictionary with correct ID
        if old_kernel_id != actual_kernel_id:
            # Remove old entry and add with correct ID
            del self.pool.kernels[old_kernel_id]
            kernel_resource.kernel_id = actual_kernel_id
            self.pool.kernels[actual_kernel_id] = kernel_resource

        # Store the manager with correct ID
        self._kernel_managers[actual_kernel_id] = kernel_manager
        kernel_resource._kernel_manager = kernel_manager

        # Warm up the kernel to ensure it's ready for execution
        # The first execution in a new kernel often fails to capture outputs
        # due to kernel initialization timing issues
        await self._warm_up_kernel(kernel_manager)

    async def execute_code(self, session: KernelSession, code: str) -> dict[str, Any]:
        """
        Execute code in a kernel session.

        This is the main method for running Python code in a kernel. It handles:
        - Code execution with timeout
        - Output collection and size limiting
        - Error handling and reporting
        - Session history tracking

        Args:
            session: The kernel session to execute code in
            code: Python code to execute (can be multiple lines)

        Returns:
            Dictionary containing execution results:
            {
                "status": "ok" | "error" | "timeout",
                "outputs": [list of output messages],
                "error": error details (if status is "error"),
                "msg_id": message ID for debugging
            }

        Raises:
            KernelTimeoutError: If execution exceeds time limit
            RuntimeError: If kernel is not found

        Example:
            ```python
            result = await manager.execute_code(session, "print('Hello, World!')")
            if result["status"] == "ok":
                print("Output:", result["outputs"])
            else:
                print("Error:", result["error"])
            ```
        """
        kernel_manager = self._kernel_managers.get(session.kernel_id)
        if kernel_manager is None:
            raise RuntimeError(f"Kernel {session.kernel_id} not found")

        # Get kernel client
        client = kernel_manager.client()
        client.start_channels()

        try:
            # Clear any pending messages from previous executions
            # This prevents message pollution between cell executions
            while True:
                try:
                    await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
                except TimeoutError:
                    # No more pending messages
                    break

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
            # Use synchronous stop_channels as async version not available in jupyter_client
            # This is safe as it's non-blocking cleanup
            client.stop_channels()

    async def _collect_execution_result(self, client: Any, msg_id: str, timeout: float) -> dict[str, Any]:
        """
        Collect execution result from kernel with output size limits.

        This method handles the complex task of collecting all outputs from a kernel
        execution. It must handle:
        - Multiple types of output messages (stream, execute_result, error)
        - Timeout handling for long-running executions
        - Output size limiting to prevent memory issues
        - Message filtering to only process relevant messages

        Args:
            client: The kernel client for communication
            msg_id: The message ID of the execution request
            timeout: Maximum time to wait for completion

        Returns:
            Dictionary with execution results

        Technical Details:
        - Uses asyncio.create_task for concurrent shell reply and output collection
        - Implements output size tracking to prevent memory exhaustion
        - Handles different message types (stream, execute_result, error, status)
        - Uses parent_msg_id filtering to only process relevant messages
        """
        result: dict[str, Any] = {"msg_id": msg_id, "status": "unknown", "outputs": [], "error": None}

        # Track cumulative output size to prevent flooding
        output_size_bytes = 0
        max_output_bytes = self.resource_limits.max_output_size_mb * 1024 * 1024

        # Start collecting outputs immediately (before waiting for shell reply)
        # This ensures we don't miss early outputs
        start_time = time.time()
        shell_reply_received = False

        # Create tasks for both shell reply and output collection
        async def get_shell_reply() -> None:
            nonlocal shell_reply_received
            try:
                reply = await asyncio.wait_for(client.get_shell_msg(), timeout=timeout)
                result["status"] = reply["content"].get("status", "unknown")
                if result["status"] == "error":
                    result["error"] = reply["content"]
                shell_reply_received = True
            except TimeoutError:
                result["status"] = "timeout"
                # Re-raise the TimeoutError so it gets caught by the main execute_code method
                raise

        # Start shell reply task
        shell_task = asyncio.create_task(get_shell_reply())

        # Collect output messages with size tracking
        while True:
            try:
                # Respect overall timeout
                elapsed = time.time() - start_time
                remaining_timeout = max(0.1, min(1.0, timeout - elapsed))

                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=remaining_timeout)

                msg_type = msg.get("msg_type", "")

                # Debug logging
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")
                if msg_type in ["stream", "execute_result", "error", "status"]:
                    logger.debug(
                        "Received message",
                        msg_type=msg_type,
                        parent_msg_id=parent_msg_id,
                        expected_msg_id=msg_id,
                        matches=parent_msg_id == msg_id,
                        content_keys=list(msg.get("content", {}).keys())
                        if msg_type != "status"
                        else msg.get("content", {}).get("execution_state"),
                    )

                # Only process messages for this execution
                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    text = msg["content"]["text"]
                    text_bytes = len(text.encode("utf-8"))

                    # Check if adding this output would exceed limit
                    if output_size_bytes + text_bytes > max_output_bytes:
                        result["outputs"].append(
                            {
                                "type": "output_truncated",
                                "reason": "max_output_size_exceeded",
                                "size_bytes": output_size_bytes,
                            }
                        )
                        break

                    output_size_bytes += text_bytes
                    result["outputs"].append({"type": "stream", "text": text})

                elif msg_type == "execute_result":
                    # Estimate size of data representation
                    data_str = str(msg["content"]["data"])
                    data_bytes = len(data_str.encode("utf-8"))

                    if output_size_bytes + data_bytes > max_output_bytes:
                        result["outputs"].append(
                            {
                                "type": "output_truncated",
                                "reason": "max_output_size_exceeded",
                                "size_bytes": output_size_bytes,
                            }
                        )
                        break

                    output_size_bytes += data_bytes
                    result["outputs"].append({"type": "execute_result", "data": msg["content"]["data"]})

                elif msg_type == "error":
                    result["error"] = msg["content"]
                    # Also add error to outputs for compatibility with notebook execution
                    result["outputs"].append(
                        {
                            "type": "error",
                            "ename": msg["content"].get("ename", "Error"),
                            "evalue": msg["content"].get("evalue", "Unknown error"),
                            "traceback": msg["content"].get("traceback", []),
                        }
                    )

                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    # Don't break immediately on idle - wait for shell reply
                    # and collect any remaining outputs
                    if shell_reply_received:
                        # Both shell reply received and execution idle, safe to exit
                        # But first, try to collect any remaining outputs
                        try:
                            await asyncio.wait_for(client.get_iopub_msg(), timeout=0.1)
                            # If we got here, there are more messages, continue
                            continue
                        except TimeoutError:
                            # No more messages, safe to break
                            break

            except TimeoutError:
                # Normal timeout on iopub messages - kernel is idle
                break

            # Check if we've exceeded overall timeout
            if time.time() - start_time > timeout:
                result["outputs"].append(
                    {"type": "timeout_during_output_collection", "collected_bytes": output_size_bytes}
                )
                # Raise TimeoutError to trigger the timeout handling in execute_code
                raise TimeoutError(f"Execution exceeded time limit of {timeout}s")

        # Ensure shell task completes
        await shell_task
        return result

    async def _warm_up_kernel(self, kernel_manager: AsyncKernelManager) -> None:
        """
        Warm up a newly created kernel to ensure it's ready for execution.

        The first execution in a fresh kernel often fails to capture outputs
        due to kernel initialization timing. This warm-up ensures the kernel
        is fully initialized and ready to handle executions properly.

        Args:
            kernel_manager: The kernel manager to warm up

        Note:
            This method executes a simple command (1 + 1) to ensure the kernel
            is fully initialized. While this adds a small delay to kernel creation,
            it prevents issues with the first real execution.
        """
        client = kernel_manager.client()
        client.start_channels()

        try:
            # Clear any startup messages from the kernel
            clear_start = time.time()
            while time.time() - clear_start < 0.5:
                try:
                    await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
                except TimeoutError:
                    break

            # Execute a simple command to warm up the kernel
            warm_up_code = "# Kernel warm-up\n1 + 1"
            msg_id = client.execute(warm_up_code)

            # Use the existing _collect_execution_result method for consistency
            # but with a shorter timeout since this is just a warm-up
            result = await self._collect_execution_result(client, msg_id, timeout=2.0)

            if result.get("status") == "ok":
                logger.debug("Kernel warm-up successful", kernel_id=kernel_manager.kernel_id)
            else:
                logger.warning("Kernel warm-up had issues but proceeding", status=result.get("status"))

        except Exception as e:
            logger.warning("Kernel warm-up failed, proceeding anyway", error=str(e))
        finally:
            client.stop_channels()

    async def get_kernel_resource_usage(self, kernel_id: str) -> dict[str, Any]:
        """
        Get resource usage for a specific kernel.

        This method provides monitoring capabilities for kernel resource usage.
        Currently returns placeholder data, but can be extended to provide
        actual CPU and memory usage metrics.

        Args:
            kernel_id: The ID of the kernel to monitor

        Returns:
            Dictionary with resource usage metrics

        Note:
            To enable actual resource monitoring, install psutil and use it
            to track the kernel process's CPU and memory usage.
        """
        kernel_manager = self._kernel_managers.get(kernel_id)
        if not kernel_manager:
            return {"error": "Kernel not found"}

        # Placeholder for resource usage
        # In production, use psutil to get actual process metrics
        return {
            "kernel_id": kernel_id,
            "cpu_percent": 0.0,  # Would use psutil.Process(pid).cpu_percent()
            "memory_mb": 0.0,  # Would use psutil.Process(pid).memory_info().rss / 1024 / 1024
            "status": "running" if kernel_manager.is_alive() else "dead",
        }

    def save_checkpoint(self, session: KernelSession) -> dict[str, Any]:
        """
        Save a checkpoint of the session state.

        This creates a snapshot of the session that can be restored later.
        Useful for:
        - Saving state before shutting down
        - Migrating sessions between systems
        - Debugging and analysis

        Args:
            session: The session to checkpoint

        Returns:
            Dictionary containing checkpoint data
        """
        session.checkpoint()
        return session.checkpoint_data or {}

    def restore_checkpoint(self, session: KernelSession, checkpoint_data: dict[str, Any]) -> None:
        """
        Restore session state from checkpoint.

        This restores the execution history and other session data from
        a previously saved checkpoint.

        Args:
            session: The session to restore
            checkpoint_data: The checkpoint data to restore from
        """
        # Restore execution history
        if "execution_history" in checkpoint_data:
            session.execution_history = checkpoint_data["execution_history"]

    async def _evict_idle_kernels(self) -> None:
        """
        Background task to evict idle kernels.

        This method runs continuously in the background to:
        1. Monitor kernel usage patterns
        2. Shutdown kernels that have been idle for too long
        3. Free up system resources

        The eviction process:
        1. Checks each kernel's last_used_at timestamp
        2. If a kernel has been idle longer than idle_timeout_seconds, marks it for eviction
        3. Gracefully shuts down the kernel process
        4. Removes the kernel from the pool
        5. Cleans up associated sessions

        This prevents resource leaks from kernels that are no longer needed.
        """
        check_interval = min(60.0, self.resource_limits.idle_timeout_seconds / 5)

        while not self._shutdown:
            try:
                await asyncio.sleep(check_interval)

                current_time = time.time()
                kernels_to_evict = []

                # Check each kernel for idle timeout
                for kernel_id, kernel_resource in list(self.pool.kernels.items()):
                    if kernel_resource.is_available and kernel_resource.last_used_at is not None:
                        idle_time = current_time - kernel_resource.last_used_at

                        if idle_time > self.resource_limits.idle_timeout_seconds:
                            kernels_to_evict.append((kernel_id, kernel_resource))

                # Evict idle kernels
                for kernel_id, kernel_resource in kernels_to_evict:
                    try:
                        # Mark as shutting down to prevent new acquisitions
                        kernel_resource.is_shutting_down = True

                        # Shutdown the kernel
                        if kernel_id in self._kernel_managers:
                            kernel_manager = self._kernel_managers[kernel_id]
                            await kernel_manager.shutdown_kernel(now=False, restart=False)
                            del self._kernel_managers[kernel_id]

                        # Remove from pool
                        del self.pool.kernels[kernel_id]

                        # Remove any sessions using this kernel
                        sessions_to_remove = [
                            agent_id for agent_id, session in self.sessions.items() if session.kernel_id == kernel_id
                        ]
                        for agent_id in sessions_to_remove:
                            del self.sessions[agent_id]

                    except (OSError, RuntimeError):
                        # Continue with other evictions on error
                        # In production, log the error for monitoring
                        continue

            except asyncio.CancelledError:
                break
            except Exception:
                # Prevent eviction task from crashing on unexpected errors
                await asyncio.sleep(check_interval)

    async def shutdown(self) -> None:
        """
        Shutdown all kernels and cleanup resources.

        This method ensures graceful shutdown of the entire kernel management system:
        1. Cancels the background eviction task
        2. Shuts down all kernel processes
        3. Clears all internal state
        4. Releases all resources

        This method is automatically called when using the manager as an async
        context manager (async with AgentKernelManager() as manager:).
        """
        self._shutdown = True

        # Cancel eviction task
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._eviction_task

        # Shutdown all kernel managers
        for kernel_manager in self._kernel_managers.values():
            # Best effort cleanup - suppress any exceptions
            with suppress(Exception):
                await kernel_manager.shutdown_kernel()

        self._kernel_managers.clear()
        self.sessions.clear()
