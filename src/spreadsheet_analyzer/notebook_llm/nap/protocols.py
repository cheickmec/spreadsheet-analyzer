"""NAP Protocol definitions for notebook operations.

This module defines the core protocols and data structures for notebook
manipulation operations, providing a consistent interface for interacting
with Jupyter notebooks.
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class CellType(Enum):
    """Types of cells in a notebook."""

    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


@dataclass
class Cell:
    """Represents a single cell in a notebook."""

    id: str
    cell_type: CellType
    source: str
    metadata: dict[str, Any]
    execution_count: int | None = None
    outputs: list[dict[str, Any]] | None = None


@dataclass
class CellExecutionResult:
    """Result of executing a cell."""

    cell_id: str
    success: bool
    outputs: list[dict[str, Any]]
    error: dict[str, Any] | None = None
    execution_count: int | None = None
    execution_time_ms: float | None = None


@dataclass
class CellSelector:
    """Selector for retrieving cells from a notebook."""

    cell_ids: list[str] | None = None
    cell_types: list[CellType] | None = None
    tag_filter: list[str] | None = None
    metadata_filter: dict[str, Any] | None = None
    index_range: tuple[int, int] | None = None  # (start, end) inclusive


@dataclass
class NotebookDocument:
    """Represents a complete notebook document."""

    id: str
    cells: list[Cell]
    metadata: dict[str, Any]
    kernel_spec: dict[str, Any]
    language_info: dict[str, Any]


class NotebookProtocol(Protocol):
    """Core protocol for notebook manipulation operations.

    This protocol defines the essential operations for interacting with
    Jupyter notebooks, including cell execution, retrieval, and updates.
    All implementations must be thread-safe and handle errors gracefully.
    """

    @abstractmethod
    async def create_notebook(self, kernel_name: str = "python3", metadata: dict[str, Any] | None = None) -> str:
        """Create a new notebook with specified kernel.

        Args:
            kernel_name: Name of the kernel to use (default: python3)
            metadata: Optional notebook metadata

        Returns:
            Unique identifier for the created notebook

        Raises:
            KernelNotFoundError: If specified kernel is not available
            NotebookCreationError: If notebook creation fails
        """
        ...

    @abstractmethod
    async def execute_cell(
        self, notebook_id: str, cell_content: str, cell_type: CellType = CellType.CODE, store_history: bool = True
    ) -> CellExecutionResult:
        """Execute a cell in the specified notebook.

        Args:
            notebook_id: Unique identifier of the notebook
            cell_content: Content to execute in the cell
            cell_type: Type of cell to create and execute
            store_history: Whether to store in execution history

        Returns:
            Execution result with outputs and status

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
            KernelError: If kernel execution fails
        """
        ...

    @abstractmethod
    async def get_cells(self, notebook_id: str, selector: CellSelector | None = None) -> list[Cell]:
        """Retrieve cells from a notebook based on selector criteria.

        Args:
            notebook_id: Unique identifier of the notebook
            selector: Criteria for selecting cells (None = all cells)

        Returns:
            List of cells matching the selector criteria

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
        """
        ...

    @abstractmethod
    async def update_cell(self, notebook_id: str, cell_id: str, new_content: str, clear_outputs: bool = False) -> Cell:
        """Update the content of an existing cell.

        Args:
            notebook_id: Unique identifier of the notebook
            cell_id: Identifier of the cell to update
            new_content: New content for the cell
            clear_outputs: Whether to clear existing outputs

        Returns:
            Updated cell object

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
            CellNotFoundError: If cell doesn't exist
        """
        ...

    @abstractmethod
    async def add_cell(
        self,
        notebook_id: str,
        cell_content: str,
        cell_type: CellType = CellType.CODE,
        position: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Cell:
        """Add a new cell to the notebook.

        Args:
            notebook_id: Unique identifier of the notebook
            cell_content: Content for the new cell
            cell_type: Type of cell to create
            position: Position to insert (None = append)
            metadata: Optional cell metadata

        Returns:
            Newly created cell object

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
            InvalidPositionError: If position is out of bounds
        """
        ...

    @abstractmethod
    async def delete_cell(self, notebook_id: str, cell_id: str) -> None:
        """Delete a cell from the notebook.

        Args:
            notebook_id: Unique identifier of the notebook
            cell_id: Identifier of the cell to delete

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
            CellNotFoundError: If cell doesn't exist
        """
        ...

    @abstractmethod
    async def get_notebook(self, notebook_id: str) -> NotebookDocument:
        """Retrieve complete notebook document.

        Args:
            notebook_id: Unique identifier of the notebook

        Returns:
            Complete notebook document

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
        """
        ...

    @abstractmethod
    async def save_notebook(self, notebook_id: str, path: str) -> None:
        """Save notebook to disk.

        Args:
            notebook_id: Unique identifier of the notebook
            path: File path to save the notebook

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
            IOError: If save operation fails
        """
        ...

    @abstractmethod
    async def close_notebook(self, notebook_id: str, save: bool = True) -> None:
        """Close notebook and cleanup resources.

        Args:
            notebook_id: Unique identifier of the notebook
            save: Whether to save before closing

        Raises:
            NotebookNotFoundError: If notebook doesn't exist
        """
        ...
