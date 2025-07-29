"""
Notebook LLM Interface

LangChain tools for notebook operations that can be presented to LLMs.
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from result import Err, Ok

import spreadsheet_analyzer.notebook_session
from spreadsheet_analyzer.notebook_llm.graph_query_tools import get_graph_query_tools
from spreadsheet_analyzer.notebook_tools import CellType


# CLAUDE-KNOWLEDGE: Consistent output truncation for token efficiency
def truncate_output(text: str, max_length: int = 1000) -> str:
    """Truncate output text for token efficiency.

    Args:
        text: The output text to truncate
        max_length: Maximum length before truncation

    Returns:
        Original text if under max_length, otherwise truncated with indicator
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n... (output truncated)"


class ExecuteCodeInput(BaseModel):
    code: str = Field(description="Python code to execute")
    cell_id: str | None = Field(description="Optional cell ID for tracking", default=None)


class EditAndExecuteInput(BaseModel):
    cell_id: str = Field(description="Cell ID to edit and execute")
    new_code: str = Field(description="New code content")


class AddCellInput(BaseModel):
    content: str = Field(description="Cell content")
    cell_type: str = Field(description="Type of cell: 'code', 'markdown', or 'raw'")
    position: int | None = Field(description="Position to insert cell (optional)", default=None)


class DeleteCellInput(BaseModel):
    cell_id: str = Field(description="Cell ID to delete")


class ReadCellInput(BaseModel):
    cell_id: str = Field(description="Cell ID to read")


class GetStateInput(BaseModel):
    session_id: str = Field(description="Session ID to get state for")


class SaveNotebookInput(BaseModel):
    session_id: str = Field(description="Session ID to save")
    file_path: str | None = Field(description="File path to save to (optional)", default=None)


# Global session manager
_session_manager = None


def get_session_manager() -> spreadsheet_analyzer.notebook_session.SessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = spreadsheet_analyzer.notebook_session.SessionManager()
    return _session_manager


@tool
async def execute_code(input_data: ExecuteCodeInput) -> str:
    """Execute Python code in a new cell."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session("default_session")

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value
        result = await session.toolkit.execute_code(code=input_data.code, cell_id=input_data.cell_id)

        if isinstance(result, Err):
            return f"Execution failed: {result.value}"

        # Format the output for the LLM
        outputs = []
        for output in result.value.outputs:
            if output.output_type == "stream":
                # CLAUDE-KNOWLEDGE: Truncate stream outputs to manage token usage
                content = truncate_output(output.content)
                outputs.append(f"[{output.metadata.get('name', 'output')}] {content}")
            elif output.output_type == "execute_result":
                # CLAUDE-KNOWLEDGE: Truncate execution results for consistency
                content = truncate_output(output.content)
                outputs.append(f"Result: {content}")
            elif output.output_type == "error":
                # CLAUDE-KNOWLEDGE: Include error outputs but truncate long tracebacks
                content = truncate_output(output.content)
                outputs.append(f"Error: {content}")

        return f"✅ Code executed successfully\nCell ID: {result.value.cell_id}\nOutputs:\n" + "\n".join(outputs)

    except Exception as e:
        return f"Error executing code: {e!s}"


@tool
async def edit_and_execute(input_data: EditAndExecuteInput) -> str:
    """Edit an existing code cell and execute it immediately."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session("default_session")

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value
        # For now, we'll just execute the new code (edit + execute in one step)
        result = await session.toolkit.execute_code(code=input_data.new_code, cell_id=input_data.cell_id)

        if isinstance(result, Err):
            return f"Edit and execute failed: {result.value}"

        # Format the output for the LLM
        outputs = []
        for output in result.value.outputs:
            if output.output_type == "stream":
                # CLAUDE-KNOWLEDGE: Consistent truncation across all tool outputs
                content = truncate_output(output.content)
                outputs.append(f"[{output.metadata.get('name', 'output')}] {content}")
            elif output.output_type == "execute_result":
                content = truncate_output(output.content)
                outputs.append(f"Result: {content}")
            elif output.output_type == "error":
                content = truncate_output(output.content)
                outputs.append(f"Error: {content}")

        return f"✅ Cell edited and executed successfully\nCell ID: {result.value.cell_id}\nOutputs:\n" + "\n".join(
            outputs
        )

    except Exception as e:
        return f"Error editing and executing: {e!s}"


@tool
async def add_cell(input_data: AddCellInput) -> str:
    """Add a new cell to the notebook (code, markdown, or raw)."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session("default_session")

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value

        # Convert string to CellType enum
        cell_type_map = {"code": CellType.CODE, "markdown": CellType.MARKDOWN, "raw": CellType.RAW}

        if input_data.cell_type not in cell_type_map:
            return f"Error: Invalid cell type '{input_data.cell_type}'. Must be 'code', 'markdown', or 'raw'"

        cell_type = cell_type_map[input_data.cell_type]

        # Handle different cell types appropriately
        if cell_type == CellType.CODE:
            result = await session.toolkit.execute_code(input_data.content)
        elif cell_type == CellType.MARKDOWN:
            result = await session.toolkit.render_markdown(input_data.content)
        elif cell_type == CellType.RAW:
            result = await session.toolkit.add_raw_cell(input_data.content)

        if isinstance(result, Err):
            return f"Failed to add cell: {result.value}"

        cell_type_name = input_data.cell_type.capitalize()
        return f"✅ {cell_type_name} cell added successfully\nCell ID: {result.value.cell_id}\nContent: {input_data.content[:100]}{'...' if len(input_data.content) > 100 else ''}"

    except Exception as e:
        return f"Error adding cell: {e!s}"


@tool
async def add_markdown_cell(input_data: AddCellInput) -> str:
    """Add a markdown cell for documentation and formatting."""
    try:
        # Override the cell type to ensure it's markdown
        input_data.cell_type = "markdown"
        return await add_cell(input_data)
    except Exception as e:
        return f"Error adding markdown cell: {e!s}"


@tool
async def delete_cell(input_data: DeleteCellInput) -> str:
    """Delete a cell from the notebook."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session("default_session")

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value

        result = await session.toolkit.delete_cell(input_data.cell_id)

        if isinstance(result, Err):
            return f"Failed to delete cell: {result.value}"

        return f"✅ Cell {input_data.cell_id} deleted successfully"

    except Exception as e:
        return f"Error deleting cell: {e!s}"


@tool
async def read_cell(input_data: ReadCellInput) -> str:
    """Read the content and outputs of a specific cell."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session("default_session")

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value

        result = await session.toolkit.get_cell(input_data.cell_id)

        if isinstance(result, Err):
            return f"Failed to read cell: {result.value}"

        cell = result.value
        outputs = []
        for output in cell.outputs:
            # CLAUDE-KNOWLEDGE: Truncate outputs when reading cells too
            content = truncate_output(output.content)
            outputs.append(f"[{output.output_type}] {content}")

        return (
            f"Cell ID: {cell.cell_id}\nType: {cell.cell_type.value}\nContent: {cell.content}\nOutputs:\n"
            + "\n".join(outputs)
        )

    except Exception as e:
        return f"Error reading cell: {e!s}"


@tool
async def get_notebook_state(input_data: GetStateInput) -> str:
    """Get the current state of the notebook."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session(input_data.session_id)

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value

        state = await session.toolkit.get_notebook_state()

        if isinstance(state, Err):
            return f"Failed to get notebook state: {state.value}"

        # Format the state information
        cells_info = []
        for i, cell in enumerate(state.value.cells):
            cells_info.append(
                f"{i + 1}. [{cell.cell_type.value}] {cell.cell_id}: {cell.content[:50]}{'...' if len(cell.content) > 50 else ''}"
            )

        return f"Notebook State:\nTotal cells: {len(state.value.cells)}\nCells:\n" + "\n".join(cells_info)

    except Exception as e:
        return f"Error getting notebook state: {e!s}"


@tool
async def save_notebook(input_data: SaveNotebookInput) -> str:
    """Save the notebook to a file."""
    try:
        session_manager = get_session_manager()
        session_result = await session_manager.get_session(input_data.session_id)

        if session_result.is_err():
            return f"Error: {session_result.err_value}"

        session = session_result.ok_value

        file_path = Path(input_data.file_path) if input_data.file_path else None

        save_result = session.toolkit.save_notebook(file_path, overwrite=True)

        if isinstance(save_result, Ok):
            return f"✅ Notebook saved successfully to: {save_result.ok_value}"
        else:
            return f"❌ Failed to save notebook: {save_result.err_value}"

    except Exception as e:
        return f"Error saving notebook: {e!s}"


def get_notebook_tools() -> list[Any]:
    """Get all notebook tools for LLM use."""
    notebook_tools = [
        execute_code,
        edit_and_execute,
        add_cell,
        add_markdown_cell,
        delete_cell,
        read_cell,
        get_notebook_state,
        save_notebook,
    ]

    # Add graph query tools
    graph_tools = get_graph_query_tools()

    return notebook_tools + graph_tools


def create_notebook_tool_descriptions() -> str:
    """Create descriptions of all available tools for LLM prompting."""
    tools = get_notebook_tools()
    descriptions = []

    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")

    return "\n".join(descriptions)
