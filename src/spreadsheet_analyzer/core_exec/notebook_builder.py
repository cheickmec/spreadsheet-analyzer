"""
Generic notebook builder for creating Jupyter notebooks programmatically.

This module provides domain-agnostic notebook construction functionality:
- Cell creation and management with proper formatting
- Metadata handling and validation
- nbformat compliance with source formatting
- Execution count management

No domain-specific logic - pure notebook construction primitives.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CellType(Enum):
    """Supported notebook cell types."""

    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


@dataclass
class NotebookCell:
    """
    Represents a single notebook cell with proper typing.

    Args:
        cell_type: Type of cell (code, markdown, raw)
        source: Cell content as properly formatted list of lines
        metadata: Cell metadata dictionary
        outputs: List of output objects (code cells only)
        execution_count: Execution counter (code cells only)
    """

    cell_type: CellType
    source: list[str]
    metadata: dict[str, Any]
    outputs: list[dict[str, Any]] | None = None
    execution_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to nbformat dictionary representation."""
        # Generate deterministic ID based on cell content
        # This ensures the same content always gets the same ID
        # Sort metadata keys to ensure consistent ordering
        sorted_metadata = json.dumps(self.metadata, sort_keys=True) if self.metadata else ""
        content_str = f"{self.cell_type.value}:{''.join(self.source)}:{sorted_metadata}"
        cell_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        cell_dict = {"cell_type": self.cell_type.value, "metadata": self.metadata, "source": self.source, "id": cell_id}

        if self.cell_type == CellType.CODE:
            cell_dict["execution_count"] = self.execution_count
            cell_dict["outputs"] = self.outputs or []

        return cell_dict


class NotebookBuilder:
    """
    Generic notebook builder for programmatic notebook creation.

    This class provides the core functionality for building Jupyter notebooks
    without any domain-specific logic. It handles proper source formatting,
    metadata management, and nbformat compliance.

    Key features:
    - Battle-tested source formatting from original NotebookBuilder
    - Proper execution count management
    - Metadata validation and handling
    - Generic cell creation without domain assumptions

    Usage:
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Analysis", {"tags": ["header"]})
        builder.add_code_cell("print('hello')")
        notebook_dict = builder.to_dict()
    """

    def __init__(self, kernel_name: str = "python3", kernel_display_name: str = "Python 3"):
        """
        Initialize notebook builder.

        Args:
            kernel_name: Kernel spec name (e.g., 'python3', 'julia-1.6')
            kernel_display_name: Human-readable kernel name
        """
        self.cells: list[NotebookCell] = []
        self.kernel_name = kernel_name
        self.kernel_display_name = kernel_display_name
        self._execution_count = 0

    def add_markdown_cell(self, content: str, metadata: dict[str, Any] | None = None) -> "NotebookBuilder":
        """
        Add a markdown cell to the notebook.

        Args:
            content: Markdown content for the cell
            metadata: Optional metadata for the cell

        Returns:
            Self for method chaining
        """
        cell = NotebookCell(cell_type=CellType.MARKDOWN, source=self._format_source(content), metadata=metadata or {})
        self.cells.append(cell)
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
            code: Python code for the cell
            outputs: Optional list of output objects
            metadata: Optional metadata for the cell
            increment_execution_count: Whether to auto-increment execution count

        Returns:
            Self for method chaining
        """
        if increment_execution_count:
            self._execution_count += 1

        cell = NotebookCell(
            cell_type=CellType.CODE,
            source=self._format_source(code),
            metadata=metadata or {},
            outputs=outputs or [],
            execution_count=self._execution_count,
        )
        self.cells.append(cell)
        return self

    def add_raw_cell(self, content: str, metadata: dict[str, Any] | None = None) -> "NotebookBuilder":
        """
        Add a raw cell to the notebook.

        Args:
            content: Raw content for the cell
            metadata: Optional metadata for the cell

        Returns:
            Self for method chaining
        """
        cell = NotebookCell(cell_type=CellType.RAW, source=self._format_source(content), metadata=metadata or {})
        self.cells.append(cell)
        return self

    def add_cell(self, cell: NotebookCell) -> "NotebookBuilder":
        """
        Add a pre-constructed cell to the notebook.

        Args:
            cell: NotebookCell to add

        Returns:
            Self for method chaining
        """
        self.cells.append(cell)
        return self

    def insert_cell(self, index: int, cell: NotebookCell) -> "NotebookBuilder":
        """
        Insert a cell at a specific position.

        Args:
            index: Position to insert at
            cell: NotebookCell to insert

        Returns:
            Self for method chaining
        """
        self.cells.insert(index, cell)
        return self

    def remove_cell(self, index: int) -> "NotebookBuilder":
        """
        Remove a cell by index.

        Args:
            index: Index of cell to remove

        Returns:
            Self for method chaining
        """
        if 0 <= index < len(self.cells):
            self.cells.pop(index)
        return self

    def get_cell(self, index: int) -> NotebookCell | None:
        """
        Get a cell by index.

        Args:
            index: Index of cell to retrieve

        Returns:
            NotebookCell or None if index is invalid
        """
        if 0 <= index < len(self.cells):
            return self.cells[index]
        return None

    def update_execution_count(self, cell_index: int, execution_count: int) -> "NotebookBuilder":
        """
        Update execution count for a specific code cell.

        This is useful when cells are executed out of order or when
        integrating with external execution systems.

        Args:
            cell_index: Index of the code cell to update
            execution_count: New execution count

        Returns:
            Self for method chaining
        """
        if 0 <= cell_index < len(self.cells):
            cell = self.cells[cell_index]
            if cell.cell_type == CellType.CODE:
                cell.execution_count = execution_count
        return self

    def set_execution_count(self, count: int) -> "NotebookBuilder":
        """
        Set the internal execution counter.

        Args:
            count: New execution count value

        Returns:
            Self for method chaining
        """
        self._execution_count = count
        return self

    def _format_source(self, content: str) -> list[str]:
        """
        Format content as list of lines for Jupyter format.

        This is the battle-tested formatting logic from the original NotebookBuilder.
        The Jupyter notebook format expects cell source to be a list of strings,
        where each string represents a line of content. All lines except the last
        should end with a newline character.

        Args:
            content: Raw string content

        Returns:
            List of formatted lines ready for notebook format
        """
        if not content:
            return [""]

        lines = content.split("\n")
        # Add newlines to all lines except the last one
        formatted = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                formatted.append(line + "\n")
            else:
                formatted.append(line)
        return formatted

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to Jupyter notebook format dictionary.

        Returns:
            Dictionary representing a complete Jupyter notebook in nbformat v4
        """
        # Determine kernel language based on kernel name
        language = "python"  # Default
        if "julia" in self.kernel_name.lower():
            language = "julia"
        elif "r" in self.kernel_name.lower():
            language = "r"
        elif "scala" in self.kernel_name.lower():
            language = "scala"

        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": {
                "kernelspec": {
                    "display_name": self.kernel_display_name,
                    "language": language,
                    "name": self.kernel_name,
                },
                "language_info": {"name": language, "version": "3.12" if language == "python" else "unknown"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def clear(self) -> "NotebookBuilder":
        """
        Clear all cells from the notebook.

        Returns:
            Self for method chaining
        """
        self.cells.clear()
        self._execution_count = 0
        return self

    def cell_count(self) -> int:
        """Get the total number of cells in the notebook."""
        return len(self.cells)

    def code_cell_count(self) -> int:
        """Get the number of code cells in the notebook."""
        return len([c for c in self.cells if c.cell_type == CellType.CODE])

    def markdown_cell_count(self) -> int:
        """Get the number of markdown cells in the notebook."""
        return len([c for c in self.cells if c.cell_type == CellType.MARKDOWN])

    def raw_cell_count(self) -> int:
        """Get the number of raw cells in the notebook."""
        return len([c for c in self.cells if c.cell_type == CellType.RAW])

    def get_execution_count(self) -> int:
        """Get the current execution counter value."""
        return self._execution_count

    def validate(self) -> list[str]:
        """
        Validate the notebook structure and return any issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        if not self.cells:
            issues.append("Notebook has no cells")

        for i, cell in enumerate(self.cells):
            if not cell.source:
                issues.append(f"Cell {i} has empty source")

            # Allow execution_count=None for unexecuted cells - this is valid in Jupyter
            # Only check that execution_count exists as an attribute
            if cell.cell_type == CellType.CODE and not hasattr(cell, "execution_count"):
                issues.append(f"Code cell {i} missing execution_count attribute")

        return issues

    def clone(self) -> "NotebookBuilder":
        """
        Create a deep copy of the notebook builder.

        Returns:
            New NotebookBuilder instance with same content
        """
        clone = NotebookBuilder(self.kernel_name, self.kernel_display_name)

        # Deep copy cells
        for cell in self.cells:
            clone_cell = NotebookCell(
                cell_type=cell.cell_type,
                source=cell.source.copy(),
                metadata=cell.metadata.copy(),
                outputs=cell.outputs.copy() if cell.outputs else None,
                execution_count=cell.execution_count,
            )
            clone.cells.append(clone_cell)

        clone._execution_count = self._execution_count
        return clone
