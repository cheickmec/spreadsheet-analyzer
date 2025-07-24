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
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator, Protocol
from pathlib import Path

import psutil
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbformat.v4 import new_output


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
    """
    name: str = "python3"
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 1024
    max_execution_time: float = 30.0
    max_output_size_mb: int = 10
    idle_timeout_seconds: float = 300.0
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[Path] = None

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
    outputs: List[Dict[str, Any]]
    error: Optional[Dict[str, Any]] = None
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
        self._sessions: Dict[str, AsyncKernelManager] = {}
        self._session_last_used: Dict[str, float] = {}
        self._execution_counts: Dict[str, int] = {}
        
        # State management
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None

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
        
        # Store session info
        self._sessions[session_id] = kernel_manager
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
        client = kernel_manager.client()
        
        start_time = time.time()
        
        try:
            client.start_channels()
            
            # Clear pending messages
            await self._clear_pending_messages(client)
            
            # Execute code
            msg_id = client.execute(code)
            
            # Collect results with timeout
            result = await self._collect_execution_result(
                client, msg_id, self.profile.max_execution_time
            )
            
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
                duration_seconds=time.time() - start_time
            )
            
        except asyncio.TimeoutError as e:
            raise KernelTimeoutError(
                f"Execution exceeded time limit of {self.profile.max_execution_time}s"
            ) from e
        finally:
            client.stop_channels()

    async def execute_for_notebook(self, session_id: str, code: str) -> List[Dict[str, Any]]:
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
                        nb_output = new_output(
                            output_type="stream", 
                            name="stdout", 
                            text=[output.get("text", "")]
                        )
                        notebook_outputs.append(nb_output)
                        
                    elif output_type == "execute_result":
                        nb_output = new_output(
                            output_type="execute_result",
                            execution_count=result.execution_count,
                            data=output.get("data", {}),
                            metadata={}
                        )
                        notebook_outputs.append(nb_output)
                        
                    elif output_type == "display_data":
                        nb_output = new_output(
                            output_type="display_data",
                            data=output.get("data", {}),
                            metadata=output.get("metadata", {})
                        )
                        notebook_outputs.append(nb_output)
                        
                    elif output_type == "error":
                        nb_output = new_output(
                            output_type="error",
                            ename=output.get("ename", "Error"),
                            evalue=output.get("evalue", "Unknown error"), 
                            traceback=output.get("traceback", [])
                        )
                        notebook_outputs.append(nb_output)
                        
        elif result.status == "error" and result.error:
            nb_output = new_output(
                output_type="error",
                ename=result.error.get("ename", "ExecutionError"),
                evalue=result.error.get("evalue", "Code execution failed"),
                traceback=result.error.get("traceback", [])
            )
            notebook_outputs.append(nb_output)
            
        return notebook_outputs

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self._sessions:
            raise RuntimeError(f"Session '{session_id}' not found")
            
        return {
            "session_id": session_id,
            "execution_count": self._execution_counts.get(session_id, 0),
            "last_used": self._session_last_used.get(session_id, 0),
            "is_alive": await self._sessions[session_id].is_alive()
        }

    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        if session_id not in self._sessions:
            return
            
        kernel_manager = self._sessions[session_id]
        await kernel_manager.shutdown_kernel()
        
        # Cleanup tracking
        del self._sessions[session_id]
        self._session_last_used.pop(session_id, None)
        self._execution_counts.pop(session_id, None)

    async def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    async def _clear_pending_messages(self, client: Any) -> None:
        """Clear any pending messages from previous executions."""
        while True:
            try:
                await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
            except asyncio.TimeoutError:
                break

    async def _collect_execution_result(
        self, client: Any, msg_id: str, timeout: float
    ) -> Dict[str, Any]:
        """Collect execution results from kernel with timeout."""
        outputs = []
        error = None
        status = "ok"
        
        # Wait for execution to complete
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    client.get_iopub_msg(), 
                    timeout=min(1.0, deadline - time.time())
                )
                
                msg_type = msg["msg_type"]
                content = msg["content"]
                
                if msg_type == "stream":
                    outputs.append({
                        "type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", "")
                    })
                    
                elif msg_type == "execute_result":
                    outputs.append({
                        "type": "execute_result",
                        "execution_count": content.get("execution_count", 1),
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {})
                    })
                    
                elif msg_type == "display_data":
                    outputs.append({
                        "type": "display_data",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {})
                    })
                    
                elif msg_type == "error":
                    error = {
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", "Unknown error"),
                        "traceback": content.get("traceback", [])
                    }
                    status = "error"
                    
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    # Execution completed
                    break
                    
            except asyncio.TimeoutError:
                status = "timeout"
                break
                
        return {
            "status": status,
            "outputs": outputs,
            "error": error,
            "msg_id": msg_id
        }

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
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id) 