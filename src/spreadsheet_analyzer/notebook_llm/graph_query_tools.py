"""
Graph Query Tools for LLM Interface

Provides LangChain tools for querying the formula dependency graph.
Each tool creates a markdown cell documenting the query and results.
"""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from spreadsheet_analyzer.graph_db.query_interface import DependencyQueryResult
from spreadsheet_analyzer.notebook_session import NotebookSession


class CellDependencyInput(BaseModel):
    sheet: str = Field(description="Sheet name (e.g., 'Summary', 'Details')")
    cell_ref: str = Field(description="Cell reference (e.g., 'A1', 'B5')")


class RangeAnalysisInput(BaseModel):
    sheet: str = Field(description="Sheet name")
    start_cell: str = Field(description="Start cell of range (e.g., 'A1')")
    end_cell: str = Field(description="End cell of range (e.g., 'D10')")


class SheetAnalysisInput(BaseModel):
    sheet: str = Field(description="Sheet name to analyze")


class NoInput(BaseModel):
    """Empty input for tools that don't need parameters."""

    pass


def get_current_session() -> NotebookSession:
    """Get the current notebook session with query interface."""
    from spreadsheet_analyzer.notebook_llm_interface import get_session_manager

    session_manager = get_session_manager()
    session_result = session_manager.get_session_sync("default_session")
    if session_result.is_err():
        raise RuntimeError(f"No active session: {session_result.err_value}")
    return session_result.ok_value


def format_dependency_result(result: DependencyQueryResult) -> str:
    """Format dependency query result as markdown."""
    md = f"""**Cell:** `{result.sheet}!{result.cell_ref}`
**Has Formula:** {"Yes" if result.has_formula else "No"}"""

    if result.formula:
        md += f"\n**Formula:** `{result.formula}`"

    if result.direct_dependencies:
        md += f"\n\n**Direct Dependencies ({len(result.direct_dependencies)}):**"
        for dep in result.direct_dependencies:
            md += f"\n- `{dep}`"

    if result.range_dependencies:
        md += f"\n\n**Range Dependencies ({len(result.range_dependencies)}):**"
        for dep in result.range_dependencies:
            md += f"\n- `{dep}`"

    if result.direct_dependents:
        md += f"\n\n**Cells that depend on this ({len(result.direct_dependents)}):**"
        for dep in result.direct_dependents:
            md += f"\n- `{dep}`"

    if result.range_dependents:
        md += f"\n\n**Formulas using this via ranges ({len(result.range_dependents)}):**"
        for dep in result.range_dependents:
            md += f"\n- `{dep}`"

    if result.is_in_ranges:
        md += "\n\n**Part of ranges:**"
        for range_ref in result.is_in_ranges:
            md += f"\n- `{range_ref}`"

    return md


async def add_query_markdown(session: NotebookSession, title: str, query: str, result: str) -> None:
    """Add a markdown cell documenting the query and result."""
    markdown_content = f"""### 🔍 {title}

**Query:** `{query}`

**Result:**
{result}"""

    toolkit = session.toolkit
    await toolkit.render_markdown(markdown_content)


@tool
async def get_cell_dependencies(input_data: CellDependencyInput) -> str:
    """
    Get complete dependency information for a specific cell.
    Shows what the cell depends on and what depends on it.
    """
    try:
        session = get_current_session()

        # Check if query interface exists
        if not hasattr(session, "query_interface") or session.query_interface is None:
            return (
                "Error: Formula analysis not available. The deterministic pipeline may not have completed successfully."
            )

        # Execute query
        result = session.query_interface.get_cell_dependencies(input_data.sheet, input_data.cell_ref)

        # Format for markdown
        formatted_result = format_dependency_result(result)

        # Add to notebook
        query_str = f"get_cell_dependencies(sheet='{input_data.sheet}', cell_ref='{input_data.cell_ref}')"
        await add_query_markdown(session, "Cell Dependencies Analysis", query_str, formatted_result)

        # Return structured result to LLM
        return f"Analysis complete for {input_data.sheet}!{input_data.cell_ref}. Found {result.total_dependencies} dependencies and {result.total_dependents} dependents."

    except Exception as e:
        return f"Error analyzing cell dependencies: {e!s}"


@tool
async def find_cells_affecting_range(input_data: RangeAnalysisInput) -> str:
    """
    Find all cells that affect any cell within the specified range.
    Useful for understanding what impacts a specific area of the spreadsheet.
    """
    try:
        session = get_current_session()

        if not hasattr(session, "query_interface") or session.query_interface is None:
            return "Error: Formula analysis not available."

        # Execute query
        affecting_cells = session.query_interface.find_cells_affecting_range(
            input_data.sheet, input_data.start_cell, input_data.end_cell
        )

        # Format result
        if affecting_cells:
            md_result = f"Found {len(affecting_cells)} cells with dependencies in range:\n\n"
            for cell, deps in affecting_cells.items():
                md_result += f"**{cell}** depends on:\n"
                for dep in deps[:5]:  # Show first 5
                    md_result += f"  - `{dep}`\n"
                if len(deps) > 5:
                    md_result += f"  - ...and {len(deps) - 5} more\n"
                md_result += "\n"
        else:
            md_result = "No cells in this range have dependencies."

        # Add to notebook
        query_str = f"find_cells_affecting_range(sheet='{input_data.sheet}', start_cell='{input_data.start_cell}', end_cell='{input_data.end_cell}')"
        await add_query_markdown(session, "Range Impact Analysis", query_str, md_result)

        return f"Found {len(affecting_cells)} cells with dependencies in the range {input_data.start_cell}:{input_data.end_cell}."

    except Exception as e:
        return f"Error analyzing range: {e!s}"


@tool
async def find_empty_cells_in_formula_ranges(input_data: SheetAnalysisInput) -> str:
    """
    Find empty cells that are part of formula ranges.
    These might be data gaps that affect calculations.
    """
    try:
        session = get_current_session()

        if not hasattr(session, "query_interface") or session.query_interface is None:
            return "Error: Formula analysis not available."

        # Execute query
        empty_cells = session.query_interface.find_empty_cells_in_formula_ranges(input_data.sheet)

        # Format result
        if empty_cells:
            md_result = f"Found {len(empty_cells)} empty cells within formula ranges:\n\n"
            # Group by rows for better readability
            rows = {}
            for cell in empty_cells[:50]:  # Limit display
                row_num = "".join(filter(str.isdigit, cell))
                if row_num not in rows:
                    rows[row_num] = []
                rows[row_num].append(cell)

            for row, cells in sorted(rows.items(), key=lambda x: int(x[0]))[:10]:
                md_result += f"Row {row}: {', '.join(cells)}\n"

            if len(empty_cells) > 50:
                md_result += f"\n...and {len(empty_cells) - 50} more empty cells"
        else:
            md_result = "No empty cells found in formula ranges."

        # Add to notebook
        query_str = f"find_empty_cells_in_formula_ranges(sheet='{input_data.sheet}')"
        await add_query_markdown(session, "Empty Cells in Formula Ranges", query_str, md_result)

        return (
            f"Found {len(empty_cells)} empty cells that are included in formula ranges on sheet '{input_data.sheet}'."
        )

    except Exception as e:
        return f"Error finding empty cells: {e!s}"


@tool
async def get_formula_statistics() -> str:
    """
    Get comprehensive statistics about formulas in the workbook.
    Includes counts, complexity metrics, and range information.
    """
    try:
        session = get_current_session()

        if not hasattr(session, "query_interface") or session.query_interface is None:
            return "Error: Formula analysis not available."

        # Execute query
        stats = session.query_interface.get_formula_statistics_with_ranges()

        # Format result
        md_result = f"""**Total Formulas:** {stats["total_formulas"]:,}
**Formulas with Dependencies:** {stats["formulas_with_dependencies"]:,}
**Unique Cells Referenced:** {stats["unique_cells_referenced"]:,}
**Circular Reference Chains:** {stats["circular_reference_chains"]}
**Volatile Formulas:** {stats["volatile_formulas"]}
**External References:** {stats["external_references"]}
**Max Dependency Depth:** {stats["max_dependency_depth"]} levels
**Average Dependencies per Formula:** {stats["average_dependencies_per_formula"]}
**Complexity Score:** {stats["complexity_score"]}/100"""

        if stats.get("has_range_index"):
            md_result += "\n\n**Range Analysis:**\n"
            md_result += f"**Unique Ranges:** {stats.get('unique_ranges', 0):,}  \n"
            md_result += f"**Total Cells in Ranges:** {stats.get('total_cells_in_ranges', 0):,}"

        # Add to notebook
        await add_query_markdown(session, "Formula Statistics", "get_formula_statistics()", md_result)

        return f"Retrieved formula statistics: {stats['total_formulas']} formulas analyzed with complexity score {stats['complexity_score']}/100."

    except Exception as e:
        return f"Error getting statistics: {e!s}"


@tool
async def find_circular_references() -> str:
    """
    Find all circular reference chains in the workbook.
    Circular references can cause calculation errors.
    """
    try:
        session = get_current_session()

        if not hasattr(session, "query_interface") or session.query_interface is None:
            return "Error: Formula analysis not available."

        # Access the formula analysis directly for circular references
        if not hasattr(session.query_interface, "analysis"):
            return "Error: Formula analysis data not available."

        circular_refs = session.query_interface.analysis.circular_references

        # Format result
        if circular_refs:
            md_result = f"Found {len(circular_refs)} circular reference chains:\n\n"
            for i, chain in enumerate(circular_refs, 1):
                chain_str = " → ".join(f"`{cell}`" for cell in list(chain)[:10])
                if len(chain) > 10:
                    chain_str += " → ..."
                md_result += f"{i}. {chain_str}\n"
        else:
            md_result = "No circular references found."

        # Add to notebook
        await add_query_markdown(session, "Circular References", "find_circular_references()", md_result)

        return f"Found {len(circular_refs)} circular reference chains in the workbook."

    except Exception as e:
        return f"Error finding circular references: {e!s}"


def get_graph_query_tools() -> list[Any]:
    """Get all graph query tools for LLM use."""
    return [
        get_cell_dependencies,
        find_cells_affecting_range,
        find_empty_cells_in_formula_ranges,
        get_formula_statistics,
        find_circular_references,
    ]
