"""
Excel Formulas Evaluator using the 'formulas' library

Provides a robust interface for Excel formula evaluation, dependency analysis,
and calculation. This replaces the limited xlcalculator approach with a more
capable solution that handles complex Excel files.
"""

import logging
from pathlib import Path
from typing import Any

try:
    import formulas

    HAS_FORMULAS = True
except ImportError:
    HAS_FORMULAS = False
    formulas = None

logger = logging.getLogger(__name__)


class ExcelFormulasEvaluator:
    """
    Robust Excel formula evaluator using the 'formulas' library.

    This class provides comprehensive formula evaluation, dependency analysis,
    and calculation capabilities for Excel files. It handles complex formulas,
    cross-sheet references, and various Excel functions.
    """

    def __init__(self, excel_path: Path):
        """
        Initialize the Excel formulas evaluator.

        Args:
            excel_path: Path to the Excel file

        Raises:
            ImportError: If formulas library is not installed
            FileNotFoundError: If the Excel file doesn't exist
        """
        if not HAS_FORMULAS:
            raise ImportError("formulas library is not installed. Install it with: pip install formulas[all]")

        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        self.excel_path = excel_path
        self.filename = excel_path.name

        # Load and compile the Excel model
        logger.info(f"Loading Excel file with formulas library: {excel_path}")
        self.model = formulas.ExcelModel()
        self.model.loads(str(excel_path))
        self.model.finish()

        # Cache useful information
        self._cells = None
        self._formula_cells = None
        self._sheets = None

    @property
    def cells(self) -> dict[str, Any]:
        """Get all cells in the workbook."""
        if self._cells is None:
            self._cells = dict(self.model.cells)
        return self._cells

    @property
    def formula_cells(self) -> dict[str, Any]:
        """Get only cells containing formulas."""
        if self._formula_cells is None:
            self._formula_cells = {}
            for cell_id, cell in self.cells.items():
                if hasattr(cell, "inputs") and cell.inputs:
                    self._formula_cells[cell_id] = cell
        return self._formula_cells

    @property
    def sheets(self) -> list[str]:
        """Get list of sheet names."""
        if self._sheets is None:
            # Extract unique sheet names from cell addresses
            sheet_names = set()
            for cell_id in self.cells:
                # Cell ID format: '[filename]SheetName'!CellRef
                if "]" in cell_id and "!" in cell_id:
                    sheet_part = cell_id.split("]")[1].split("!")[0]
                    sheet_names.add(sheet_part.strip("'"))
            self._sheets = sorted(sheet_names)
        return self._sheets

    def get_cell_value(self, cell_address: str) -> Any:
        """
        Get the calculated value of a cell.

        Args:
            cell_address: Cell address like "Sheet1!A1" or full format

        Returns:
            The calculated value of the cell
        """
        # Normalize cell address
        cell_id = self._normalize_cell_id(cell_address)

        # Calculate and return value
        result = self.model.calculate(outputs=[cell_id])
        return result.get(cell_id)

    def get_cell_formula(self, cell_address: str) -> str | None:
        """
        Get the formula of a cell if it exists.

        Args:
            cell_address: Cell address

        Returns:
            Formula string or None if cell has no formula
        """
        cell_id = self._normalize_cell_id(cell_address)
        cell = self.cells.get(cell_id)

        if cell and hasattr(cell, "inputs") and cell.inputs:
            # This is a formula cell
            # Try to get the original formula representation
            if hasattr(cell, "func") and hasattr(cell.func, "__name__"):
                return f"={cell.func.__name__}"
            else:
                return "=[Formula]"
        return None

    def get_cell_dependencies(self, cell_address: str) -> list[str]:
        """
        Get the cells that this cell depends on.

        Args:
            cell_address: Cell address

        Returns:
            List of cell addresses this cell depends on
        """
        cell_id = self._normalize_cell_id(cell_address)
        cell = self.cells.get(cell_id)

        if cell and hasattr(cell, "inputs"):
            # Extract dependencies from inputs
            deps = []
            for inp in cell.inputs:
                if isinstance(inp, str) and "!" in inp:
                    deps.append(inp)
            return deps
        return []

    def get_cell_dependents(self, cell_address: str) -> list[str]:
        """
        Get cells that depend on this cell.

        Args:
            cell_address: Cell address

        Returns:
            List of cell addresses that depend on this cell
        """
        cell_id = self._normalize_cell_id(cell_address)
        dependents = []

        # Search through all formula cells
        for dep_id, dep_cell in self.formula_cells.items():
            if hasattr(dep_cell, "inputs"):
                for inp in dep_cell.inputs:
                    if inp == cell_id:
                        dependents.append(dep_id)
                        break

        return dependents

    def set_cell_value(self, cell_address: str, value: Any) -> None:
        """
        Set a cell value and prepare for recalculation.

        Args:
            cell_address: Cell address
            value: New value for the cell
        """
        cell_id = self._normalize_cell_id(cell_address)

        # Update the input value
        # Note: formulas library handles this through calculate()
        self._pending_inputs = getattr(self, "_pending_inputs", {})
        self._pending_inputs[cell_id] = value

    def calculate(self, outputs: list[str] | None = None) -> dict[str, Any]:
        """
        Calculate cell values, optionally with pending input changes.

        Args:
            outputs: Optional list of cells to calculate

        Returns:
            Dictionary of calculated values
        """
        inputs = getattr(self, "_pending_inputs", {})

        if outputs:
            outputs = [self._normalize_cell_id(addr) for addr in outputs]

        result = self.model.calculate(inputs=inputs, outputs=outputs)

        # Clear pending inputs after calculation
        self._pending_inputs = {}

        return result

    def get_formulas_summary(self) -> dict[str, Any]:
        """
        Get a summary of all formulas in the workbook.

        Returns:
            Dictionary with formula statistics
        """
        formula_types = {}
        examples = []

        for cell_id, _cell in self.formula_cells.items():
            # Extract sheet and cell reference
            if "]" in cell_id and "!" in cell_id:
                sheet = cell_id.split("]")[1].split("!")[0].strip("'")
                cell_ref = cell_id.split("!")[-1]

                if len(examples) < 10:
                    examples.append(
                        {"sheet": sheet, "cell": cell_ref, "address": f"{sheet}!{cell_ref}", "has_formula": True}
                    )

                # Count formula types (simplified)
                formula_types["[Formula]"] = formula_types.get("[Formula]", 0) + 1

        return {
            "total_cells": len(self.cells),
            "total_formulas": len(self.formula_cells),
            "formula_types": formula_types,
            "sheets": self.sheets,
            "examples": examples,
        }

    def get_dependency_graph_stats(self) -> dict[str, Any]:
        """
        Get statistics about the dependency graph.

        Returns:
            Dictionary with graph statistics
        """
        # Calculate dependency statistics
        total_dependencies = 0
        max_dependencies = 0
        cells_with_deps = 0

        for cell_id, _cell in self.formula_cells.items():
            deps = self.get_cell_dependencies(cell_id)
            if deps:
                cells_with_deps += 1
                total_dependencies += len(deps)
                max_dependencies = max(max_dependencies, len(deps))

        # Calculate dependent statistics
        total_dependents = 0
        max_dependents = 0

        for cell_id in self.cells:
            deps = self.get_cell_dependents(cell_id)
            if deps:
                total_dependents += len(deps)
                max_dependents = max(max_dependents, len(deps))

        return {
            "total_cells": len(self.cells),
            "formula_cells": len(self.formula_cells),
            "cells_with_dependencies": cells_with_deps,
            "total_dependencies": total_dependencies,
            "max_dependencies_per_cell": max_dependencies,
            "total_dependents": total_dependents,
            "max_dependents_per_cell": max_dependents,
            "sheets": len(self.sheets),
        }

    def export_to_json(self, output_path: Path) -> None:
        """
        Export the model to JSON format.

        Args:
            output_path: Path for the output JSON file
        """
        # formulas library supports JSON export
        self.model.save(str(output_path))
        logger.info(f"Model exported to: {output_path}")

    def _normalize_cell_id(self, cell_address: str) -> str:
        """
        Normalize a cell address to the format expected by formulas library.

        Args:
            cell_address: Cell address like "Sheet1!A1"

        Returns:
            Normalized cell ID like "[filename]Sheet1!A1"
        """
        # If already in full format, return as is
        if cell_address.startswith("[") and "]" in cell_address:
            return cell_address

        # Add filename prefix
        if "!" in cell_address:
            sheet, cell = cell_address.split("!", 1)
            return f"[{self.filename}]{sheet}!{cell}"
        else:
            # Assume first sheet if no sheet specified
            if self.sheets:
                return f"[{self.filename}]{self.sheets[0]}!{cell_address}"
            else:
                return f"[{self.filename}]Sheet1!{cell_address}"

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_dependency_graph_stats()
        return (
            f"ExcelFormulasEvaluator("
            f"file='{self.filename}', "
            f"sheets={stats['sheets']}, "
            f"cells={stats['total_cells']}, "
            f"formulas={stats['formula_cells']})"
        )


def create_formulas_evaluator(excel_path: Path) -> ExcelFormulasEvaluator:
    """
    Factory function to create an ExcelFormulasEvaluator instance.

    Args:
        excel_path: Path to the Excel file

    Returns:
        ExcelFormulasEvaluator instance
    """
    return ExcelFormulasEvaluator(excel_path)


# Example usage in notebook:
# Load Excel file with formulas evaluator
# from pathlib import Path
# from spreadsheet_analyzer.excel_formulas_evaluator import create_formulas_evaluator
#
# evaluator = create_formulas_evaluator(Path("data/spreadsheet.xlsx"))
#
# # Get summary
# summary = evaluator.get_formulas_summary()
# print(f"Total formulas: {summary['total_formulas']}")
#
# # Get cell value
# value = evaluator.get_cell_value("Sheet1!A1")
#
# # Set value and recalculate
# evaluator.set_cell_value("Sheet1!A1", 100)
# result = evaluator.calculate(outputs=["Sheet1!B1", "Sheet1!C1"])
#
# # Get dependencies
# deps = evaluator.get_cell_dependencies("Sheet1!B1")
#
# # Export to JSON
# evaluator.export_to_json(Path("model.json"))
