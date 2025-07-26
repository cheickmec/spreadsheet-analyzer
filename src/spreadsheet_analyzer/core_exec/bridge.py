"""
Execution bridge between notebooks and kernels.

This module provides the orchestration layer that executes notebook cells
using kernel services and attaches the results. It handles:
- Sequential cell execution with proper state management
- Output attachment and formatting
- Error handling and recovery
- Execution count synchronization

No domain-specific logic - pure execution orchestration.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .kernel_service import ExecutionResult, KernelService, KernelTimeoutError
from .notebook_builder import CellType, NotebookBuilder
from .notebook_io import NotebookIO


@dataclass
class ExecutionStats:
    """Statistics from notebook execution."""

    total_cells: int
    executed_cells: int
    skipped_cells: int
    error_cells: int
    total_duration_seconds: float


class ExecutionError(Exception):
    """Raised when execution encounters an unrecoverable error."""

    pass


class ExecutionBridge:
    """
    Bridge between notebook builders and kernel services.

    This class orchestrates the execution of notebook cells using kernel services,
    handling output attachment, error recovery, and execution state management.

    Key features:
    - Sequential execution with proper state management
    - Automatic output attachment to cells
    - Error isolation and recovery
    - Execution statistics and monitoring
    - Customizable execution hooks

    Usage:
        async with KernelService(profile) as kernel_service:
            bridge = ExecutionBridge(kernel_service)
            result_builder = await bridge.execute_notebook(
                session_id, notebook_builder
            )
    """

    def __init__(
        self, kernel_service: KernelService, stop_on_error: bool = False, execution_timeout: float | None = None
    ):
        """
        Initialize the execution bridge.

        Args:
            kernel_service: KernelService instance for code execution
            stop_on_error: Whether to stop execution on first error
            execution_timeout: Override timeout for individual cells
        """
        self.kernel_service = kernel_service
        self.stop_on_error = stop_on_error
        self.execution_timeout = execution_timeout

        # Hooks for customization
        self._pre_execution_hooks: list[Callable] = []
        self._post_execution_hooks: list[Callable] = []
        self._error_hooks: list[Callable] = []

    async def execute_notebook(
        self, session_id: str, notebook: NotebookBuilder, skip_empty_cells: bool = True
    ) -> NotebookBuilder:
        """
        Execute all code cells in a notebook.

        Args:
            session_id: Kernel session to use for execution
            notebook: NotebookBuilder with cells to execute
            skip_empty_cells: Whether to skip cells with no code

        Returns:
            New NotebookBuilder with execution results attached

        Raises:
            ExecutionError: If execution fails and stop_on_error=True
        """
        # Clone the notebook to avoid modifying the original
        result_notebook = notebook.clone()

        # Execution tracking
        stats = ExecutionStats(
            total_cells=notebook.code_cell_count(),
            executed_cells=0,
            skipped_cells=0,
            error_cells=0,
            total_duration_seconds=0.0,
        )

        # Execute code cells sequentially
        for i, cell in enumerate(result_notebook.cells):
            if cell.cell_type != CellType.CODE:
                continue

            # Skip empty cells if requested
            cell_code = "".join(cell.source).strip()
            if skip_empty_cells and not cell_code:
                stats.skipped_cells += 1
                continue

            try:
                # Run pre-execution hooks
                await self._run_hooks(
                    self._pre_execution_hooks, {"cell_index": i, "cell": cell, "session_id": session_id}
                )

                # Execute the cell
                execution_result = await self._execute_cell(session_id, cell_code)

                # Update cell with results
                cell.outputs = NotebookIO.convert_outputs_to_nbformat(execution_result.outputs)
                cell.execution_count = execution_result.execution_count

                # Update stats
                stats.executed_cells += 1
                stats.total_duration_seconds += execution_result.duration_seconds

                # Run post-execution hooks
                await self._run_hooks(
                    self._post_execution_hooks,
                    {"cell_index": i, "cell": cell, "execution_result": execution_result, "session_id": session_id},
                )

                # Check for errors
                if execution_result.status == "error":
                    stats.error_cells += 1

                    # Run error hooks
                    await self._run_hooks(
                        self._error_hooks,
                        {"cell_index": i, "cell": cell, "execution_result": execution_result, "session_id": session_id},
                    )

                    if self.stop_on_error:
                        raise ExecutionError(f"Cell {i} execution failed: {execution_result.error}")

            except Exception as e:
                stats.error_cells += 1

                # Create error output for the cell
                error_output = {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)],
                }
                cell.outputs = [error_output]

                # Run error hooks
                await self._run_hooks(
                    self._error_hooks, {"cell_index": i, "cell": cell, "error": e, "session_id": session_id}
                )

                if self.stop_on_error:
                    raise ExecutionError(f"Cell {i} execution failed: {e}") from e

        # Store execution stats in notebook metadata
        result_notebook.to_dict()["metadata"]["execution_stats"] = {
            "total_cells": stats.total_cells,
            "executed_cells": stats.executed_cells,
            "skipped_cells": stats.skipped_cells,
            "error_cells": stats.error_cells,
            "total_duration_seconds": stats.total_duration_seconds,
        }

        return result_notebook

    async def execute_cell(self, session_id: str, code: str, attach_outputs: bool = True) -> ExecutionResult:
        """
        Execute a single code cell.

        Args:
            session_id: Kernel session to use
            code: Code to execute
            attach_outputs: Whether to format outputs for notebook

        Returns:
            ExecutionResult with execution details
        """
        return await self._execute_cell(session_id, code)

    async def execute_and_create_cell(
        self, session_id: str, code: str, metadata: dict[str, Any] | None = None
    ) -> NotebookBuilder:
        """
        Execute code and create a new notebook with the result.

        Args:
            session_id: Kernel session to use
            code: Code to execute
            metadata: Optional metadata for the cell

        Returns:
            NotebookBuilder with single executed cell
        """
        # Execute the code
        result = await self._execute_cell(session_id, code)

        # Create notebook with the result
        builder = NotebookBuilder()
        outputs = NotebookIO.convert_outputs_to_nbformat(result.outputs)

        builder.add_code_cell(code=code, outputs=outputs, metadata=metadata or {}, increment_execution_count=False)

        # Set the actual execution count
        if builder.cells:
            builder.cells[0].execution_count = result.execution_count

        return builder

    def add_pre_execution_hook(self, hook: Callable) -> None:
        """Add a hook to run before each cell execution."""
        self._pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: Callable) -> None:
        """Add a hook to run after each cell execution."""
        self._post_execution_hooks.append(hook)

    def add_error_hook(self, hook: Callable) -> None:
        """Add a hook to run when cell execution fails."""
        self._error_hooks.append(hook)

    async def _execute_cell(self, session_id: str, code: str) -> ExecutionResult:
        """Execute a single cell with timeout handling."""
        try:
            if self.execution_timeout:
                # Use custom timeout if provided
                original_timeout = self.kernel_service.profile.max_execution_time
                self.kernel_service.profile = self.kernel_service.profile._replace(
                    max_execution_time=self.execution_timeout
                )

                try:
                    result = await self.kernel_service.execute(session_id, code)
                finally:
                    # Restore original timeout
                    self.kernel_service.profile = self.kernel_service.profile._replace(
                        max_execution_time=original_timeout
                    )

                return result
            else:
                return await self.kernel_service.execute(session_id, code)

        except KernelTimeoutError as e:
            # Convert timeout to execution result
            return ExecutionResult(
                status="timeout",
                outputs=[],
                error={"ename": "TimeoutError", "evalue": str(e), "traceback": [str(e)]},
                execution_count=1,
                msg_id="",
                duration_seconds=self.execution_timeout or self.kernel_service.profile.max_execution_time,
            )

    async def _run_hooks(self, hooks: list[Callable], context: dict[str, Any]) -> None:
        """Run a list of hooks with the given context."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context)
                else:
                    hook(context)
            except Exception:
                # Don't let hook failures break execution
                pass


class StreamingExecutionBridge(ExecutionBridge):
    """
    Execution bridge with real-time output streaming.

    Extends ExecutionBridge to provide real-time output streaming
    during execution, useful for long-running cells or interactive scenarios.
    """

    def __init__(self, kernel_service: KernelService, output_callback: Callable[[str], None], **kwargs):
        """
        Initialize streaming execution bridge.

        Args:
            kernel_service: KernelService instance
            output_callback: Callback for streaming output
            **kwargs: Additional arguments for ExecutionBridge
        """
        super().__init__(kernel_service, **kwargs)
        self.output_callback = output_callback

    async def execute_notebook(self, session_id: str, notebook: NotebookBuilder, **kwargs) -> NotebookBuilder:
        """Execute notebook with streaming output."""

        def stream_hook(context: dict[str, Any]) -> None:
            """Hook to stream output during execution."""
            if "execution_result" in context:
                result = context["execution_result"]
                for output in result.outputs:
                    if output.get("type") == "stream":
                        self.output_callback(output.get("text", ""))

        # Add streaming hook
        self.add_post_execution_hooks(stream_hook)

        # Execute normally
        return await super().execute_notebook(session_id, notebook, **kwargs)
