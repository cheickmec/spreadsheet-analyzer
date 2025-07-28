"""
Generic execution bridge for orchestrating notebook cell execution.

This module provides domain-agnostic execution orchestration functionality:
- Cell-by-cell execution with proper sequencing
- Output collection and formatting
- Error handling and recovery
- Data persistence between executions using scrapbook
- Session management integration

No domain-specific logic - pure execution orchestration primitives.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .kernel_service import ExecutionResult, KernelService
from .notebook_builder import NotebookBuilder

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics from notebook execution."""

    total_cells: int
    executed_cells: int
    skipped_cells: int
    error_cells: int
    total_duration_seconds: float


class ExecutionBridge:
    """
    Generic execution bridge for orchestrating notebook execution.

    This class provides domain-agnostic execution orchestration with:
    - Cell-by-cell execution using kernel services
    - Automatic data persistence between executions using scrapbook
    - Error handling and recovery
    - Session management integration
    - Output collection and formatting

    Key features:
    - Integrates with KernelService for robust execution
    - Uses scrapbook for structured data persistence
    - Maintains execution state across cells
    - Provides flexible execution strategies

    Usage:
        bridge = ExecutionBridge(kernel_service, enable_persistence=True)
        result = await bridge.execute_notebook(session_id, notebook_builder)
    """

    def __init__(
        self,
        kernel_service: KernelService,
        enable_persistence: bool = True,
        auto_glue_results: bool = True,
        glue_prefix: str = "cell_",
    ):
        """
        Initialize execution bridge.

        Args:
            kernel_service: Kernel service for code execution
            enable_persistence: Whether to enable scrapbook data persistence
            auto_glue_results: Whether to automatically glue cell results
            glue_prefix: Prefix for auto-glued result names
        """
        self.kernel_service = kernel_service
        self.enable_persistence = enable_persistence
        self.auto_glue_results = auto_glue_results
        self.glue_prefix = glue_prefix

    async def execute_notebook(
        self,
        session_id: str,
        notebook: NotebookBuilder,
        stop_on_error: bool = True,
    ) -> list[ExecutionResult]:
        """
        Execute all cells in a notebook sequentially.

        Args:
            session_id: Kernel session to use for execution
            notebook: NotebookBuilder containing cells to execute
            stop_on_error: Whether to stop execution on first error

        Returns:
            List of execution results (one per code cell)

        Raises:
            ValueError: If session_id doesn't exist
            RuntimeError: If execution fails and stop_on_error is True
        """
        results = []
        nb_notebook = notebook.to_notebook()

        for i, cell in enumerate(nb_notebook.cells):
            if cell.cell_type != "code":
                continue

            try:
                # Execute the cell
                result = await self.kernel_service.execute(session_id, cell.source)

                # Update cell with execution results
                cell.execution_count = result.execution_count
                cell.outputs = result.outputs

                # Auto-persist results if enabled
                if self.enable_persistence and self.auto_glue_results and result.status == "ok":
                    await self._persist_cell_result(session_id, i, result)

                results.append(result)

                # Check for errors
                if result.status == "error" and stop_on_error:
                    error_msg = f"Cell {i} execution failed: {result.error}"
                    logger.error(error_msg)
                    if stop_on_error:
                        raise RuntimeError(error_msg)

            except Exception as e:
                logger.error(f"Error executing cell {i}: {e}")
                if stop_on_error:
                    raise

        return results

    async def execute_notebook_streaming(
        self,
        session_id: str,
        notebook: NotebookBuilder,
        stop_on_error: bool = True,
    ):
        """
        Execute all cells in a notebook with streaming results.

        Args:
            session_id: Kernel session to use for execution
            notebook: NotebookBuilder containing cells to execute
            stop_on_error: Whether to stop execution on first error

        Yields:
            ExecutionResult for each cell as it completes
        """
        nb_notebook = notebook.to_notebook()

        for i, cell in enumerate(nb_notebook.cells):
            if cell.cell_type != "code":
                continue

            try:
                # Execute the cell
                result = await self.kernel_service.execute(session_id, cell.source)

                # Update cell with execution results
                cell.execution_count = result.execution_count
                cell.outputs = result.outputs

                # Auto-persist results if enabled
                if self.enable_persistence and self.auto_glue_results and result.status == "ok":
                    await self._persist_cell_result(session_id, i, result)

                yield result

                # Check for errors
                if result.status == "error" and stop_on_error:
                    error_msg = f"Cell {i} execution failed: {result.error}"
                    logger.error(error_msg)
                    if stop_on_error:
                        raise RuntimeError(error_msg)

            except Exception as e:
                logger.error(f"Error executing cell {i}: {e}")
                if stop_on_error:
                    raise

    async def execute_cell(
        self,
        session_id: str,
        code: str,
        cell_index: int | None = None,
        persist_result: bool = True,
    ) -> ExecutionResult:
        """
        Execute a single code cell.

        Args:
            session_id: Kernel session to use
            code: Python code to execute
            cell_index: Optional cell index for persistence naming
            persist_result: Whether to persist the result using scrapbook

        Returns:
            ExecutionResult with outputs and status
        """
        result = await self.kernel_service.execute(session_id, code)

        # Persist result if enabled
        if self.enable_persistence and persist_result and result.status == "ok":
            index = cell_index if cell_index is not None else 0
            await self._persist_cell_result(session_id, index, result)

        return result

    async def execute_cells(
        self,
        session_id: str,
        cells: list[str],
        stop_on_error: bool = True,
    ) -> list[ExecutionResult]:
        """
        Execute multiple code cells sequentially.

        Args:
            session_id: Kernel session to use
            cells: List of Python code strings to execute
            stop_on_error: Whether to stop on first error

        Returns:
            List of execution results
        """
        results = []

        for i, code in enumerate(cells):
            try:
                result = await self.execute_cell(session_id, code, cell_index=i)
                results.append(result)

                if result.status == "error" and stop_on_error:
                    break

            except Exception as e:
                logger.error(f"Error executing cell {i}: {e}")
                if stop_on_error:
                    raise

        return results

    async def _persist_cell_result(self, session_id: str, cell_index: int, result: ExecutionResult) -> None:
        """
        Persist cell execution result using scrapbook.

        Args:
            session_id: Kernel session identifier
            cell_index: Index of the cell
            result: Execution result to persist
        """
        try:
            # Create a unique name for this result
            result_name = f"{self.glue_prefix}{cell_index}_result"

            # Extract the most relevant output for persistence
            data_to_persist = self._extract_persistable_data(result.outputs)

            if data_to_persist is not None:
                # Use scrapbook to glue the data
                glue_code = f"""
import scrapbook as sb
sb.glue('{result_name}', {data_to_persist!r})
"""
                # Execute the glue command
                await self.kernel_service.execute(session_id, glue_code)
                logger.debug(f"Persisted result for cell {cell_index} as '{result_name}'")

        except Exception as e:
            logger.warning(f"Failed to persist cell {cell_index} result: {e}")

    def _extract_persistable_data(self, outputs: list[dict[str, Any]]) -> Any:
        """
        Extract data that can be persisted from cell outputs.

        Args:
            outputs: List of output dictionaries

        Returns:
            Data suitable for persistence, or None if no suitable data found
        """
        for output in outputs:
            output_type = output.get("type")

            if output_type == "execute_result":
                data = output.get("data", {})
                # Prefer text/plain, then text/html, then any other format
                if "text/plain" in data:
                    return data["text/plain"]
                elif "text/html" in data:
                    return data["text/html"]
                elif data:
                    # Return the first available data
                    return next(iter(data.values()))

            elif output_type == "stream":
                text = output.get("text", "")
                if text.strip():  # Only persist non-empty streams
                    return text

            elif output_type == "display_data":
                data = output.get("data", {})
                if data:
                    return next(iter(data.values()))

        return None

    async def get_persisted_data(self, session_id: str, result_name: str) -> Any:
        """
        Retrieve persisted data from a previous execution.

        Args:
            session_id: Kernel session identifier
            result_name: Name of the persisted result

        Returns:
            Retrieved data or None if not found
        """
        try:
            # Execute code to retrieve the scrapbook data
            retrieve_code = f"""
import scrapbook as sb
try:
    result = sb.read_notebook().scraps.get('{result_name}')
    if result:
        print(result.data)
    else:
        print("None")
except Exception as e:
    print(f"Error retrieving data: {{e}}")
"""
            result = await self.kernel_service.execute(session_id, retrieve_code)

            # Extract the printed result
            for output in result.outputs:
                if output.get("type") == "stream" and output.get("name") == "stdout":
                    text = output.get("text", "").strip()
                    if text and text != "None":
                        return text

            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve persisted data '{result_name}': {e}")
            return None

    async def list_persisted_data(self, session_id: str) -> list[str]:
        """
        List all available persisted data names.

        Args:
            session_id: Kernel session identifier

        Returns:
            List of persisted data names
        """
        try:
            list_code = """
import scrapbook as sb
try:
    scraps = sb.read_notebook().scraps
    for name in scraps.keys():
        print(name)
except Exception as e:
    print(f"Error listing scraps: {e}")
"""
            result = await self.kernel_service.execute(session_id, list_code)

            names = []
            for output in result.outputs:
                if output.get("type") == "stream" and output.get("name") == "stdout":
                    text = output.get("text", "").strip()
                    if text and not text.startswith("Error"):
                        names.extend(text.splitlines())

            return names

        except Exception as e:
            logger.warning(f"Failed to list persisted data: {e}")
            return []

    async def clear_persisted_data(self, session_id: str, result_name: str | None = None) -> None:
        """
        Clear persisted data.

        Args:
            session_id: Kernel session identifier
            result_name: Specific result name to clear, or None to clear all
        """
        try:
            if result_name:
                clear_code = f"""
import scrapbook as sb
try:
    sb.read_notebook().scraps.pop('{result_name}', None)
    print(f"Cleared {result_name}")
except Exception as e:
    print(f"Error clearing {result_name}: {{e}}")
"""
            else:
                clear_code = """
import scrapbook as sb
try:
    sb.read_notebook().scraps.clear()
    print("Cleared all persisted data")
except Exception as e:
    print(f"Error clearing data: {e}")
"""

            await self.kernel_service.execute(session_id, clear_code)

        except Exception as e:
            logger.warning(f"Failed to clear persisted data: {e}")

    async def execute_with_parameters(
        self,
        session_id: str,
        notebook: NotebookBuilder,
        parameters: dict[str, Any],
        parameter_cell_index: int = 0,
    ) -> list[ExecutionResult]:
        """
        Execute notebook with parameter injection.

        Args:
            session_id: Kernel session to use
            notebook: NotebookBuilder containing cells to execute
            parameters: Dictionary of parameters to inject
            parameter_cell_index: Index of cell to inject parameters into

        Returns:
            List of execution results
        """
        # Create parameter injection code
        param_code = "\n".join([f"{k} = {v!r}" for k, v in parameters.items()])

        # Execute parameter injection first
        param_result = await self.kernel_service.execute(session_id, param_code)

        if param_result.status == "error":
            raise RuntimeError(f"Parameter injection failed: {param_result.error}")

        # Execute the rest of the notebook
        return await self.execute_notebook(session_id, notebook)

    async def validate_notebook(self, notebook: NotebookBuilder) -> list[str]:
        """
        Validate notebook structure before execution.

        Args:
            notebook: NotebookBuilder to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate notebook structure
        notebook_errors = notebook.validate()
        errors.extend(notebook_errors)

        # Check for code cells
        if notebook.code_cell_count() == 0:
            errors.append("Notebook contains no code cells")

        return errors
