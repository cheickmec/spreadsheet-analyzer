"""
Notebook Tools - Core Functional Interface

Pragmatic, composable tools for notebook interaction.
Built for incremental development with functional programming principles.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from result import Err, Ok, Result

from spreadsheet_analyzer.core_exec import KernelService, NotebookBuilder


class CellType(Enum):
    """Types of notebook cells."""

    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


@dataclass(frozen=True)
class CellOutput:
    """Immutable cell output data."""

    output_type: str  # 'stream', 'execute_result', 'error', 'display_data'
    content: Any
    metadata: dict[str, Any] = None


@dataclass(frozen=True)
class CellExecution:
    """Immutable cell execution result."""

    cell_id: str
    cell_type: CellType
    content: str
    outputs: list[CellOutput] = field(default_factory=list)
    execution_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotebookState:
    """Immutable notebook state."""

    session_id: str
    cells: list[CellExecution] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NotebookToolkit:
    """Main toolkit for notebook operations."""

    def __init__(self, kernel_service: KernelService, session_id: str, notebook_path: Path | None = None):
        self.kernel_service = kernel_service
        self.session_id = session_id
        self.notebook_path = notebook_path or Path(f"notebook_{session_id}.ipynb")
        self._execution_count = 0
        # Initialize notebook builder for persistent state
        self._notebook_builder = NotebookBuilder()

    def _next_execution_count(self) -> int:
        """Get next execution count."""
        self._execution_count += 1
        return self._execution_count

    async def execute_code(self, code: str, cell_id: str | None = None) -> Result[CellExecution, str]:
        """Execute code in a code cell."""
        try:
            cell_id = cell_id or str(uuid.uuid4())

            # Execute the code through kernel service
            execution_result = await self.kernel_service.execute(session_id=self.session_id, code=code)

            # Convert execution result to cell outputs
            outputs = []

            # Process outputs from the execution result
            for output in execution_result.outputs:
                output_type = output.get("type", "stream")  # KernelService uses "type", not "output_type"

                if output_type == "stream":
                    content = output.get("text", "")  # Stream outputs have "text" field
                    outputs.append(
                        CellOutput(
                            output_type="stream", content=content, metadata={"name": output.get("name", "stdout")}
                        )
                    )
                elif output_type == "execute_result":
                    # Execute results have data with text/plain representation
                    content = output.get("data", {}).get("text/plain", "")
                    outputs.append(
                        CellOutput(
                            output_type="execute_result",
                            content=content,
                            metadata={"execution_count": self._next_execution_count()},
                        )
                    )
                elif output_type == "display_data":
                    # Display data also has data with text/plain representation
                    content = output.get("data", {}).get("text/plain", "")
                    outputs.append(
                        CellOutput(
                            output_type="display_data",
                            content=content,
                            metadata=output.get("metadata", {}),
                        )
                    )
                elif output_type == "error":
                    content = "\n".join(output.get("traceback", []))
                    outputs.append(
                        CellOutput(
                            output_type="error",
                            content=content,
                            metadata={"ename": output.get("ename", ""), "evalue": output.get("evalue", "")},
                        )
                    )

            # Check for execution errors
            if execution_result.status == "error" and execution_result.error:
                return Err(f"Execution error: {execution_result.error.get('evalue', 'Unknown error')}")

            # Create cell execution result
            cell_execution = CellExecution(
                cell_id=cell_id,
                cell_type=CellType.CODE,
                content=code,
                outputs=outputs,
                execution_count=self._execution_count,
            )

            # Add to notebook builder for persistence
            self._notebook_builder.add_code_cell(
                code=code,
                outputs=execution_result.outputs,  # Use raw outputs directly for proper nbformat conversion
                metadata={"cell_id": cell_id},
            )

            return Ok(cell_execution)

        except Exception as e:
            return Err(f"Code execution error: {e!s}")

    async def render_markdown(self, markdown: str, cell_id: str | None = None) -> Result[CellExecution, str]:
        """Render markdown content (not execute)."""
        try:
            cell_id = cell_id or str(uuid.uuid4())

            # Create cell execution result
            cell_execution = CellExecution(
                cell_id=cell_id,
                cell_type=CellType.MARKDOWN,
                content=markdown,
                outputs=[],  # Markdown doesn't produce outputs
                execution_count=None,  # Markdown doesn't have execution count
            )

            # Add to notebook builder for persistence
            self._notebook_builder.add_markdown_cell(content=markdown, metadata={"cell_id": cell_id})

            return Ok(cell_execution)

        except Exception as e:
            return Err(f"Markdown rendering error: {e!s}")

    async def add_raw_cell(self, content: str, cell_id: str | None = None) -> Result[CellExecution, str]:
        """Add raw cell content (no processing)."""
        try:
            cell_id = cell_id or str(uuid.uuid4())

            # Create cell execution result
            cell_execution = CellExecution(
                cell_id=cell_id,
                cell_type=CellType.RAW,
                content=content,
                outputs=[],  # Raw cells don't produce outputs
                execution_count=None,  # Raw cells don't have execution count
            )

            # Add to notebook builder for persistence
            self._notebook_builder.add_raw_cell(content=content, metadata={"cell_id": cell_id})

            return Ok(cell_execution)

        except Exception as e:
            return Err(f"Raw cell error: {e!s}")

    async def execute_cell_by_type(
        self, content: str, cell_type: CellType, cell_id: str | None = None
    ) -> Result[CellExecution, str]:
        """Execute or process a cell based on its type."""
        if cell_type == CellType.CODE:
            return await self.execute_code(content, cell_id)
        elif cell_type == CellType.MARKDOWN:
            return await self.render_markdown(content, cell_id)
        elif cell_type == CellType.RAW:
            return await self.add_raw_cell(content, cell_id)
        else:
            return Err(f"Unknown cell type: {cell_type}")

    def get_state(self) -> NotebookState:
        """Get current notebook state."""
        # This would need to be implemented to track all cells
        # For now, return a basic state
        return NotebookState(
            session_id=self.session_id,
            cells=[],  # Would need to track cells in practice
            metadata={"notebook_path": str(self.notebook_path)},
        )

    def save_notebook(self, file_path: Path | None = None, overwrite: bool = False) -> Result[Path, str]:
        """Save the notebook to disk."""
        try:
            from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO

            save_path = file_path or self.notebook_path
            saved_path = NotebookIO.write_notebook(self._notebook_builder, save_path, overwrite)
            return Ok(saved_path)

        except Exception as e:
            return Err(f"Failed to save notebook: {e!s}")

    def export_to_percent_format(self) -> str:
        """
        Export notebook to py:percent format for LLM context.

        This format is optimal for LLMs because it:
        - Reduces token usage by ~94% compared to JSON
        - Avoids complex JSON structures
        - Strips base64-encoded images
        - Provides clean, readable text

        Returns:
            Notebook content in py:percent format
        """
        lines = []

        for i, cell in enumerate(self._notebook_builder.notebook.cells):
            # Add cell metadata comment if useful
            cell_id = cell.metadata.get("cell_id", f"cell_{i}")

            if cell.cell_type == "markdown":
                lines.append(f"# %% [markdown] id={cell_id}")
                # Prefix each line with # for markdown
                if isinstance(cell.source, str):
                    source = cell.source
                elif isinstance(cell.source, list):
                    source = "".join(cell.source)
                else:
                    source = str(cell.source)
                for line in source.split("\n"):
                    lines.append(f"# {line}")

            elif cell.cell_type == "code":
                lines.append(f"# %% id={cell_id}")
                if isinstance(cell.source, str):
                    source = cell.source
                elif isinstance(cell.source, list):
                    source = "".join(cell.source)
                else:
                    source = str(cell.source)
                lines.append(source)

                # Add outputs as comments
                if hasattr(cell, "outputs") and cell.outputs:
                    lines.append("# Output:")
                    for output in cell.outputs:
                        if output.output_type == "stream":
                            text = output.text if hasattr(output, "text") else output.get("text", "")
                            # Handle if text is a list
                            if isinstance(text, list):
                                text = "".join(text)
                            # Limit output length to save tokens
                            if len(text) > 1000:
                                text = text[:1000] + "... (truncated)"
                            for line in text.split("\n"):
                                if line.strip():  # Skip empty lines
                                    lines.append(f"# {line}")

                        elif output.output_type == "execute_result":
                            # Extract text representation, avoid base64 images
                            data = output.data if hasattr(output, "data") else output.get("data", {})
                            if isinstance(data, dict) and "text/plain" in data:
                                text = data["text/plain"]
                                # Handle if text is a list
                                if isinstance(text, list):
                                    text = "".join(text)
                                # Limit output length
                                if len(text) > 1000:
                                    text = text[:1000] + "... (truncated)"
                                for line in text.split("\n"):
                                    if line.strip():
                                        lines.append(f"# {line}")

                        elif output.output_type == "error":
                            # Include error information
                            ename = output.ename if hasattr(output, "ename") else output.get("ename", "Error")
                            evalue = output.evalue if hasattr(output, "evalue") else output.get("evalue", "")
                            lines.append(f"# Error: {ename}: {evalue}")

            elif cell.cell_type == "raw":
                lines.append(f"# %% [raw] id={cell_id}")
                if isinstance(cell.source, str):
                    source = cell.source
                elif isinstance(cell.source, list):
                    source = "".join(cell.source)
                else:
                    source = str(cell.source)
                for line in source.split("\n"):
                    lines.append(f"# {line}")

            lines.append("")  # Empty line between cells

        return "\n".join(lines)


def create_toolkit(
    kernel_service: KernelService, session_id: str, notebook_path: Path | None = None
) -> NotebookToolkit:
    """Factory function for creating notebook toolkits."""
    return NotebookToolkit(kernel_service, session_id, notebook_path)
