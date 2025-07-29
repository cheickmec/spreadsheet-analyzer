"""
Generic notebook builder for creating Jupyter notebooks programmatically.

This module provides domain-agnostic notebook construction functionality:
- Cell creation and management with proper formatting
- Metadata handling and validation
- nbformat compliance with source formatting
- Execution count management

No domain-specific logic - pure notebook construction primitives.
"""

import json
from typing import Any

import nbformat


class NotebookBuilder:
    """
    Generic notebook builder for programmatic notebook creation.

    This class provides a convenient facade over nbformat's native functions
    for building Jupyter notebooks without any domain-specific logic. It handles
    proper source formatting, metadata management, and nbformat compliance.

    Key features:
    - Direct use of nbformat.v4 factory functions
    - Proper execution count management
    - Metadata validation and handling
    - Generic cell creation without domain assumptions

    Usage:
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Analysis", {"tags": ["header"]})
        builder.add_code_cell("print('hello')")
        notebook = builder.to_notebook()
    """

    def __init__(self, kernel_name: str = "python3", kernel_display_name: str = "Python 3"):
        """
        Initialize notebook builder.

        Args:
            kernel_name: Kernel spec name (e.g., 'python3', 'julia-1.6')
            kernel_display_name: Human-readable kernel name
        """
        self.notebook = nbformat.v4.new_notebook(
            metadata={"kernelspec": {"name": kernel_name, "display_name": kernel_display_name}}
        )
        self._execution_count = 0

    def add_markdown_cell(self, content: str, metadata: dict[str, Any] | None = None) -> "NotebookBuilder":
        """
        Add a markdown cell to the notebook.

        Args:
            content: Markdown content
            metadata: Cell metadata dictionary

        Returns:
            Self for method chaining
        """
        cell = nbformat.v4.new_markdown_cell(source=self._format_source(content), metadata=metadata or {})
        self.notebook.cells.append(cell)
        return self

    def add_code_cell(
        self,
        code: str,
        outputs: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        increment_execution_count: bool = True,
    ) -> "NotebookBuilder":
        """
        Add a code cell to the notebook.

        Args:
            code: Python code to execute
            outputs: List of output objects
            metadata: Cell metadata dictionary
            increment_execution_count: Whether to increment execution counter

        Returns:
            Self for method chaining
        """
        if increment_execution_count:
            self._execution_count += 1
            execution_count = self._execution_count
        else:
            execution_count = None

        # Convert outputs to nbformat if needed
        if outputs:
            from .notebook_io import NotebookIO

            formatted_outputs = NotebookIO.convert_outputs_to_nbformat(outputs)
        else:
            formatted_outputs = []

        cell = nbformat.v4.new_code_cell(
            source=self._format_source(code),
            outputs=formatted_outputs,
            metadata=metadata or {},
            execution_count=execution_count,
        )
        self.notebook.cells.append(cell)
        return self

    def add_raw_cell(self, content: str, metadata: dict[str, Any] | None = None) -> "NotebookBuilder":
        """
        Add a raw cell to the notebook.

        Args:
            content: Raw cell content
            metadata: Cell metadata dictionary

        Returns:
            Self for method chaining
        """
        cell = nbformat.v4.new_raw_cell(source=self._format_source(content), metadata=metadata or {})
        self.notebook.cells.append(cell)
        return self

    def add_cell(self, cell: Any) -> "NotebookBuilder":
        """
        Add an existing nbformat cell to the notebook.

        Args:
            cell: nbformat cell object

        Returns:
            Self for method chaining
        """
        self.notebook.cells.append(cell)
        return self

    def insert_cell(self, index: int, cell: Any) -> "NotebookBuilder":
        """
        Insert a cell at a specific index.

        Args:
            index: Position to insert the cell
            cell: nbformat cell object

        Returns:
            Self for method chaining
        """
        self.notebook.cells.insert(index, cell)
        return self

    def remove_cell(self, index: int) -> "NotebookBuilder":
        """
        Remove a cell at the specified index.

        Args:
            index: Index of cell to remove

        Returns:
            Self for method chaining
        """
        if 0 <= index < len(self.notebook.cells):
            del self.notebook.cells[index]
        return self

    def get_cell(self, index: int) -> Any | None:
        """
        Get a cell at the specified index.

        Args:
            index: Index of cell to retrieve

        Returns:
            nbformat cell object or None if index out of range
        """
        if 0 <= index < len(self.notebook.cells):
            return self.notebook.cells[index]
        return None

    def update_execution_count(self, cell_index: int, execution_count: int) -> "NotebookBuilder":
        """
        Update execution count for a specific code cell.

        Args:
            cell_index: Index of the code cell
            execution_count: New execution count

        Returns:
            Self for method chaining
        """
        if 0 <= cell_index < len(self.notebook.cells):
            cell = self.notebook.cells[cell_index]
            if cell.cell_type == "code":
                cell.execution_count = execution_count
        return self

    def set_execution_count(self, count: int) -> "NotebookBuilder":
        """
        Set the next execution count for code cells.

        Args:
            count: Next execution count to use

        Returns:
            Self for method chaining
        """
        self._execution_count = count
        return self

    def _format_source(self, content: str) -> str:
        """
        Format source content for notebook cells.

        Args:
            content: Raw content string

        Returns:
            Formatted source string
        """
        # nbformat expects source as a string, not a list
        # Just return the content as-is to preserve formatting
        return content if content else ""

    import json

    # ... (rest of the file)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert notebook to dictionary format.

        Returns:
            Dictionary representation of the notebook
        """
        return json.loads(nbformat.writes(self.notebook))

    def to_notebook(self) -> Any:
        """
        Get the nbformat notebook object.

        Returns:
            nbformat notebook object
        """
        return self.notebook

    def clear(self) -> "NotebookBuilder":
        """
        Clear all cells from the notebook.

        Returns:
            Self for method chaining
        """
        self.notebook.cells = []
        self._execution_count = 0
        return self

    def cell_count(self) -> int:
        """Get the total number of cells."""
        return len(self.notebook.cells)

    def code_cell_count(self) -> int:
        """Get the number of code cells."""
        return sum(1 for cell in self.notebook.cells if cell.cell_type == "code")

    def markdown_cell_count(self) -> int:
        """Get the number of markdown cells."""
        return sum(1 for cell in self.notebook.cells if cell.cell_type == "markdown")

    def raw_cell_count(self) -> int:
        """Get the number of raw cells."""
        return sum(1 for cell in self.notebook.cells if cell.cell_type == "raw")

    def get_execution_count(self) -> int:
        """Get the current execution count."""
        return self._execution_count

    def validate(self) -> list[str]:
        """
        Validate the notebook structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            nbformat.validate(self.notebook)
        except Exception as e:
            errors.append(f"Notebook validation failed: {e}")

        # Check for empty cells
        for i, cell in enumerate(self.notebook.cells):
            if not cell.source or (isinstance(cell.source, list) and not any(cell.source)):
                errors.append(f"Cell {i} is empty")

        return errors

    def clone(self) -> "NotebookBuilder":
        """
        Create a deep copy of the notebook builder.

        Returns:
            New NotebookBuilder with copied content
        """
        import copy

        new_builder = NotebookBuilder()
        new_builder.notebook = copy.deepcopy(self.notebook)
        new_builder._execution_count = self._execution_count
        return new_builder
