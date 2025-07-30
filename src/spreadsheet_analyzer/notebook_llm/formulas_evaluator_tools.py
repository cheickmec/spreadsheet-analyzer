"""
Excel Formulas Evaluator Tools for LLM

LangChain tools for Excel formula evaluation using the 'formulas' library.
This provides robust formula evaluation that works with complex Excel files.

CLAUDE-PERFORMANCE: This module primarily uses simple string parameters because:
1. Most operations take 1-2 simple inputs (cell addresses, file paths)
2. String parameters minimize LLM context window overhead
3. Input parsing is straightforward and errors are handled gracefully
4. Performance research shows context clutter impacts agent reliability

Only complex multi-parameter tools (like set_cell_and_recalculate) benefit from
Pydantic validation. See graph_query_tools.py for contrast where structured
validation is essential for graph traversal operations.
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from spreadsheet_analyzer.excel_formulas_evaluator import (
    HAS_FORMULAS,
    ExcelFormulasEvaluator,
)

# Global evaluator instance
_formulas_evaluator: ExcelFormulasEvaluator | None = None


# CLAUDE-PERFORMANCE: Only this complex tool uses Pydantic due to its 3 parameters
class WhatIfAnalysisInput(BaseModel):
    """Input for what-if analysis with multiple parameters."""

    cell_address: str = Field(description="Cell to change (e.g., 'Sheet1!A1')")
    value: str = Field(description="New value (will be converted to appropriate type: number, boolean, or string)")
    cells_to_check: str = Field(description="Comma-separated cells to recalculate (e.g., 'Sheet1!B1,Sheet1!C1')")


@tool
def load_excel_with_formulas(file_path: str) -> str:
    """Load an Excel file for formula evaluation and analysis using the formulas library.

    This creates an ExcelFormulasEvaluator that can:
    - Evaluate complex formulas accurately
    - Track dependencies between cells
    - Perform what-if analysis
    - Handle cross-sheet references
    - Export models to JSON

    This is more robust than xlcalculator and handles complex Excel files better.
    Uses simple string parameter to minimize context overhead."""
    global _formulas_evaluator

    if not HAS_FORMULAS:
        return "Error: formulas library is not installed. Install with: pip install formulas[all]"

    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        _formulas_evaluator = ExcelFormulasEvaluator(path)

        # Get summary statistics
        summary = _formulas_evaluator.get_formulas_summary()
        stats = _formulas_evaluator.get_dependency_graph_stats()

        return (
            f"✅ Excel file loaded successfully with formulas library!\n"
            f"File: {_formulas_evaluator.filename}\n"
            f"Sheets: {', '.join(summary['sheets'])}\n"
            f"Total cells: {stats['total_cells']:,}\n"
            f"Formula cells: {stats['formula_cells']:,}\n"
            f"Cells with dependencies: {stats['cells_with_dependencies']:,}\n"
            f"\nThis evaluator handles complex Excel files robustly."
        )

    except Exception as e:
        return f"Error loading Excel file: {e!s}"


@tool
def evaluate_cell(cell_address: str) -> str:
    """Evaluate a cell and get its calculated value.

    Returns the cell value and formula (if any), formatted with dependencies.
    Uses simple string parameter to minimize context overhead."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        value = _formulas_evaluator.get_cell_value(cell_address)
        formula = _formulas_evaluator.get_cell_formula(cell_address)

        result = f"Cell {cell_address}:\n"

        # Format the value nicely
        if hasattr(value, "__iter__") and not isinstance(value, str):
            # It's a range/array
            result += f"Value: [Range with {len(value)} items]\n"
        else:
            result += f"Value: {value}\n"

        if formula:
            result += f"Formula: {formula}\n"

        # Add dependencies if it's a formula
        if formula:
            deps = _formulas_evaluator.get_cell_dependencies(cell_address)
            if deps:
                result += f"Depends on: {', '.join(deps[:5])}"
                if len(deps) > 5:
                    result += f" ... and {len(deps) - 5} more"

        return result

    except Exception as e:
        return f"Error evaluating cell: {e!s}"


@tool
def get_cell_dependencies_formulas(cell_address: str) -> str:
    """Get all cells that a given cell depends on (using formulas evaluator).

    Returns list of dependencies with their values and formulas.
    Uses simple string parameter to minimize context overhead."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        deps = _formulas_evaluator.get_cell_dependencies(cell_address)
        formula = _formulas_evaluator.get_cell_formula(cell_address)

        if not deps:
            if formula:
                return f"Cell {cell_address} has a formula but no direct cell dependencies."
            else:
                return f"Cell {cell_address} has no dependencies (constant value)."

        result = f"Cell {cell_address} dependencies:\n"
        if formula:
            result += f"Formula: {formula}\n"

        result += f"\nDepends on {len(deps)} cells:\n"

        # Show up to 10 dependencies with their values
        for dep in deps[:10]:
            try:
                dep_value = _formulas_evaluator.get_cell_value(dep)
                dep_formula = _formulas_evaluator.get_cell_formula(dep)

                if dep_formula:
                    result += f"  {dep} = {dep_value} (formula)\n"
                else:
                    result += f"  {dep} = {dep_value}\n"
            except:
                result += f"  {dep} (error reading)\n"

        if len(deps) > 10:
            result += f"  ... and {len(deps) - 10} more dependencies"

        return result

    except Exception as e:
        return f"Error getting dependencies: {e!s}"


@tool
def get_cell_dependents_formulas(cell_address: str) -> str:
    """Get all cells that depend on a given cell (using formulas evaluator).

    Returns list of dependent cells with their formulas.
    Uses simple string parameter to minimize context overhead."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        deps = _formulas_evaluator.get_cell_dependents(cell_address)

        if not deps:
            return f"No cells depend on {cell_address}."

        result = f"Cells that depend on {cell_address}:\n"
        result += f"Total: {len(deps)} cells\n\n"

        # Show up to 10 dependents
        for dep in deps[:10]:
            dep_formula = _formulas_evaluator.get_cell_formula(dep)
            if dep_formula:
                result += f"  {dep}: {dep_formula}\n"
            else:
                result += f"  {dep}\n"

        if len(deps) > 10:
            result += f"  ... and {len(deps) - 10} more cells"

        return result

    except Exception as e:
        return f"Error getting dependents: {e!s}"


@tool(args_schema=WhatIfAnalysisInput)
def set_cell_and_recalculate(input_data: WhatIfAnalysisInput) -> str:
    """Set a cell value and recalculate specific cells to see the impact.

    This is useful for what-if analysis to understand how changes propagate
    through the spreadsheet. Uses Pydantic model due to 3 parameters - justified by complexity."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        # Parse cells to check
        cells_to_check_list = [c.strip() for c in input_data.cells_to_check.split(",")]

        # Get original values
        original_values = {}
        for cell in cells_to_check_list:
            try:
                original_values[cell] = _formulas_evaluator.get_cell_value(cell)
            except:
                original_values[cell] = "Error"

        # Convert value to appropriate type
        try:
            typed_value = float(input_data.value)
        except ValueError:
            if input_data.value.lower() in ("true", "false"):
                typed_value = input_data.value.lower() == "true"
            else:
                typed_value = input_data.value

        # Set the new value
        _formulas_evaluator.set_cell_value(input_data.cell_address, typed_value)

        # Recalculate
        new_values = _formulas_evaluator.calculate(outputs=cells_to_check_list)

        # Format results
        result = "✅ What-if analysis completed\n"
        result += f"Changed: {input_data.cell_address} = {typed_value}\n\n"
        result += "Impact on other cells:\n"

        for cell in cells_to_check_list:
            old_val = original_values.get(cell, "?")
            new_val = new_values.get(cell, "Error")

            if old_val != new_val:
                result += f"  {cell}: {old_val} → {new_val} (changed)\n"
            else:
                result += f"  {cell}: {new_val} (unchanged)\n"

        return result

    except Exception as e:
        return f"Error in what-if analysis: {e!s}"


@tool
def get_formula_statistics_formulas() -> str:
    """Get comprehensive statistics about formulas in the workbook (using formulas evaluator).

    Returns detailed formula statistics including counts, dependencies, and complexity.
    No parameters needed - analyzes the entire loaded workbook."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        summary = _formulas_evaluator.get_formulas_summary()
        stats = _formulas_evaluator.get_dependency_graph_stats()

        result = f"Formula Statistics for {_formulas_evaluator.filename}:\n\n"

        result += "Overall Stats:\n"
        result += f"  Total cells: {stats['total_cells']:,}\n"
        result += f"  Formula cells: {stats['formula_cells']:,}\n"
        result += f"  Non-formula cells: {stats['total_cells'] - stats['formula_cells']:,}\n"
        result += f"  Sheets: {stats['sheets']}\n"

        result += "\nDependency Stats:\n"
        result += f"  Cells with dependencies: {stats['cells_with_dependencies']:,}\n"
        result += f"  Total dependencies: {stats['total_dependencies']:,}\n"
        result += f"  Max dependencies per cell: {stats['max_dependencies_per_cell']}\n"
        result += f"  Total dependent relationships: {stats['total_dependents']:,}\n"
        result += f"  Max dependents per cell: {stats['max_dependents_per_cell']}\n"

        if summary["examples"]:
            result += "\n\nExample formula cells:\n"
            for ex in summary["examples"][:5]:
                result += f"  {ex['address']}\n"

        return result

    except Exception as e:
        return f"Error getting statistics: {e!s}"


@tool
def export_formulas_model(output_path: str) -> str:
    """Export the Excel model to JSON format for analysis or version control.

    Returns success or error message with export details.
    Uses simple string parameter to minimize context overhead."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        path = Path(output_path)
        _formulas_evaluator.export_to_json(path)

        return (
            f"✅ Model exported successfully to: {output_path}\n"
            f"This JSON file contains the complete Excel model and can be:\n"
            f"- Version controlled\n"
            f"- Analyzed externally\n"
            f"- Reimported later"
        )

    except Exception as e:
        return f"Error exporting model: {e!s}"


@tool
def list_sheets_formulas() -> str:
    """List all sheets in the Excel file (using formulas evaluator).

    Returns list of sheet names with formula counts.
    No parameters needed - lists sheets from the loaded workbook."""
    if _formulas_evaluator is None:
        return "Error: No Excel file loaded. Use load_excel_with_formulas first."

    try:
        sheets = _formulas_evaluator.sheets
        result = f"Sheets in {_formulas_evaluator.filename}:\n"

        for i, sheet in enumerate(sheets, 1):
            # Count formulas in each sheet
            formula_count = sum(
                1
                for cell_id in _formulas_evaluator.formula_cells
                if f"]{sheet}!" in cell_id or f"]'{sheet}'!" in cell_id
            )

            result += f"  {i}. {sheet} ({formula_count} formulas)\n"

        return result

    except Exception as e:
        return f"Error listing sheets: {e!s}"


@tool
def get_formulas_help() -> str:
    """Get help on using the formulas evaluator.

    Returns comprehensive help text with examples and workflow guidance.
    No parameters needed - provides static help information."""
    return """Excel Formulas Evaluator Help:

This evaluator uses the 'formulas' library which is MORE ROBUST than xlcalculator.

Features:
✅ Handles complex Excel files with thousands of formulas
✅ Supports advanced Excel functions (VLOOKUP, INDEX, MATCH, etc.)
✅ Accurate dependency tracking
✅ What-if analysis with recalculation
✅ JSON export for version control
✅ Better error handling

Workflow:
1. load_excel_with_formulas("path/to/file.xlsx") - Load the Excel file
2. list_sheets_formulas() - See all sheets and formula counts
3. evaluate_cell("Sheet1!A1") - Get cell value and formula
4. get_cell_dependencies_formulas("Sheet1!B2") - See what a cell depends on
5. get_cell_dependents_formulas("Sheet1!A1") - See what depends on a cell
6. set_cell_and_recalculate("Sheet1!A1", "100", "Sheet1!B1,Sheet1!C1") - What-if analysis
7. get_formula_statistics() - Overall statistics
8. export_formulas_model("model.json") - Export for analysis

Tips:
- This evaluator is recommended for complex Excel files
- Use graph-based tools for quick dependency queries
- Use this evaluator for accurate formula evaluation
- Combines well with pandas for data analysis
"""


def get_formulas_evaluator_tools() -> list[Any]:
    """Get all formulas evaluator tools for LLM use."""
    return [
        load_excel_with_formulas,
        evaluate_cell,
        get_cell_dependencies_formulas,
        get_cell_dependents_formulas,
        set_cell_and_recalculate,
        get_formula_statistics_formulas,
        export_formulas_model,
        list_sheets_formulas,
        get_formulas_help,
    ]
