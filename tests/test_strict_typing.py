"""
Tests demonstrating the benefits of strict typing.

This module shows how stricter mypy settings help catch common
errors that would otherwise only be found at runtime.
"""

from typing import Any

import pytest

from spreadsheet_analyzer.pipeline.stages.stage_3_formulas_typed import (
    TypedCellReference,
    TypedDependencyAnalyzer,
    TypedFormulaParser,
    analyze_worksheet_formulas,
)
from spreadsheet_analyzer.typing import (
    CellAddress,
    ColumnIndex,
    Failure,
    Formula,
    RowIndex,
    SheetName,
    Success,
    is_formula,
    is_valid_cell_address,
    validate_column_index,
    validate_row_index,
)


class TestStrictTyping:
    """Test suite demonstrating strict typing benefits."""

    def test_typed_cell_reference_validation(self) -> None:
        """Test that TypedCellReference enforces valid indices."""
        # Valid reference
        ref = TypedCellReference(sheet=SheetName("Sheet1"), start_col=ColumnIndex(1), start_row=RowIndex(1))
        assert ref.to_address() == "A1"

        # Invalid indices are caught at creation time
        with pytest.raises(ValueError, match="Invalid column index"):
            validate_column_index(0)  # Columns are 1-based

        with pytest.raises(ValueError, match="Invalid column index"):
            validate_column_index(16385)  # Exceeds Excel limit

        with pytest.raises(ValueError, match="Invalid row index"):
            validate_row_index(0)  # Rows are 1-based

        with pytest.raises(ValueError, match="Invalid row index"):
            validate_row_index(1048577)  # Exceeds Excel limit

    def test_formula_parser_type_safety(self) -> None:
        """Test that formula parser handles types correctly."""
        parser = TypedFormulaParser()

        # Valid formula
        result = parser.parse_formula("=A1+B2", SheetName("Sheet1"))
        assert isinstance(result, Success)
        parsed = result.value
        assert len(parsed.references) == 2
        assert parsed.original == "=A1+B2"

        # Empty formula - caught by validation
        result = parser.parse_formula("", SheetName("Sheet1"))
        assert isinstance(result, Failure)
        assert "Empty formula" in result.error

        # Not a formula - caught by type guard
        result = parser.parse_formula("Just text", SheetName("Sheet1"))
        assert isinstance(result, Failure)
        assert "Not a valid formula" in result.error

    def test_type_guards_work_correctly(self) -> None:
        """Test that type guards narrow types as expected."""
        # Test is_formula guard
        valid_formula = "=SUM(A1:A10)"
        invalid_formula = "Not a formula"

        assert is_formula(valid_formula) is True
        assert is_formula(invalid_formula) is False

        # When guard passes, type is narrowed to Formula
        if is_formula(valid_formula):
            formula: Formula = valid_formula  # Type checker knows this is safe
            assert formula.startswith("=")

        # Test is_valid_cell_address guard
        valid_addr = "A1"
        invalid_addr = "1A"  # Wrong format

        assert is_valid_cell_address(valid_addr) is True
        assert is_valid_cell_address(invalid_addr) is False

        # Type narrowing with guard
        if is_valid_cell_address(valid_addr):
            addr: CellAddress = valid_addr  # Safe assignment
            assert len(addr) >= 2

    def test_result_type_error_handling(self) -> None:
        """Test that Result types provide better error handling."""
        parser = TypedFormulaParser()

        # Parse multiple formulas, some invalid
        formulas = [
            "=A1+B2",  # Valid
            "",  # Empty
            "=SUM(A:A)",  # Valid with range
            "Not formula",  # Invalid
            "=1/0",  # Valid but problematic
        ]

        results = []
        errors = []

        for formula in formulas:
            result = parser.parse_formula(formula, SheetName("Test"))
            if isinstance(result, Success):
                results.append(result.value)
            else:
                errors.append(result.error)

        # We can handle errors gracefully
        assert len(results) == 3  # Three valid formulas
        assert len(errors) == 2  # Two failures

        # Error messages are informative
        assert any("Empty formula" in e for e in errors)
        assert any("Not a valid formula" in e for e in errors)

    def test_dependency_analyzer_type_constraints(self) -> None:
        """Test that dependency analyzer enforces type constraints."""
        analyzer = TypedDependencyAnalyzer()
        parser = TypedFormulaParser()

        # Parse a formula
        result = parser.parse_formula("=SUM(A1:A10)", SheetName("Data"))
        assert isinstance(result, Success)
        parsed = result.value

        # Add to analyzer with valid address
        analyzer.add_formula(CellAddress("B1"), parsed, SheetName("Data"))

        # Invalid address is caught
        with pytest.raises(ValueError, match="Invalid cell address"):
            analyzer.add_formula(CellAddress("InvalidAddress"), parsed, SheetName("Data"))

    def test_circular_reference_detection_typed(self) -> None:
        """Test circular reference detection with typed structures."""
        analyzer = TypedDependencyAnalyzer()
        parser = TypedFormulaParser()

        # Create circular references: A1 -> B1 -> C1 -> A1
        formulas = [
            ("A1", "=B1"),
            ("B1", "=C1"),
            ("C1", "=A1"),
        ]

        for cell, formula in formulas:
            result = parser.parse_formula(formula, SheetName("Sheet1"))
            assert isinstance(result, Success)
            analyzer.add_formula(CellAddress(cell), result.value, SheetName("Sheet1"))

        # Find cycles
        cycles = analyzer.find_circular_references()
        assert len(cycles) > 0

        # Verify cycle contains expected cells
        cycle_cells = set()
        for cycle in cycles:
            cycle_cells.update(cycle)

        expected = {"Sheet1!A1", "Sheet1!B1", "Sheet1!C1"}
        assert expected.issubset(cycle_cells)

    def test_edge_type_determination(self) -> None:
        """Test that edge types are determined correctly."""
        analyzer = TypedDependencyAnalyzer()
        parser = TypedFormulaParser()

        # Different formula types
        test_cases = [
            ("=SUM(A1:A10)", "SUMS_OVER"),
            ("=AVERAGE(B1:B5)", "AVERAGES_OVER"),
            ("=VLOOKUP(A1,C:D,2,FALSE)", "LOOKS_UP_IN"),
            ("=IF(A1>0,B1,C1)", "CONDITIONALLY_USES"),
            ("=A1+B1", "DEPENDS_ON"),
        ]

        for formula_str, expected_type in test_cases:
            result = parser.parse_formula(formula_str, SheetName("Test"))
            assert isinstance(result, Success)

            # Get the first reference
            ref = next(iter(result.value.references))

            # Check edge type
            edge_type = analyzer._determine_edge_type(result.value, ref)
            assert edge_type == expected_type

    def test_optional_handling(self) -> None:
        """Test that optional values are handled correctly."""
        # With strict_optional, None must be handled explicitly

        class MockCell:
            def __init__(self, value: Any | None) -> None:
                self.value = value
                self.coordinate = "A1"
                self.data_type = "f" if value and str(value).startswith("=") else "s"

        # Test with None value
        cell = MockCell(None)

        # Without proper handling, this would fail at runtime
        # But with types, we're forced to handle it
        if cell.value is not None and is_formula(str(cell.value)):
            formula = Formula(str(cell.value))
        else:
            formula = None

        assert formula is None

        # Test with valid value
        cell = MockCell("=A1+B1")
        if cell.value is not None and is_formula(str(cell.value)):
            formula = Formula(str(cell.value))
        else:
            formula = None

        assert formula == "=A1+B1"

    def test_type_safety_prevents_common_errors(self) -> None:
        """Test that type safety prevents common runtime errors."""
        parser = TypedFormulaParser()

        # These would be runtime errors without types:

        # 1. Passing wrong type to parse_formula
        # With types, this is caught at development time
        # parser.parse_formula(123, "Sheet1")  # Type error!

        # 2. Forgetting to check Result type
        result = parser.parse_formula("=A1", SheetName("Sheet1"))
        # Can't access .value without checking type first
        # value = result.value  # Type error without isinstance check!

        if isinstance(result, Success):
            value = result.value  # Now safe
            assert value.original == "=A1"

        # 3. Using wrong attribute names
        # With protocols, IDE catches these
        # if result.sucess:  # Typo caught by type checker!

        # 4. Mixing up row/column indices
        ref = TypedCellReference(
            sheet=SheetName("Test"),
            start_col=ColumnIndex(1),  # Can't accidentally use row here
            start_row=RowIndex(1),  # Can't accidentally use col here
        )
        assert ref.start_col == 1
        assert ref.start_row == 1


class TestProtocolUsage:
    """Test that protocols work correctly for duck typing."""

    def test_worksheet_protocol(self) -> None:
        """Test that any object matching WorksheetProtocol works."""

        class MockWorksheet:
            """Mock that satisfies WorksheetProtocol."""

            def __init__(self) -> None:
                self.title = "TestSheet"
                self._cells: list[Any] = []

            def iter_rows(self, **kwargs: Any) -> list[list[Any]]:
                return [self._cells]

            def add_cell(self, coord: str, value: str, data_type: str = "s") -> None:
                cell = type("Cell", (), {"coordinate": coord, "value": value, "data_type": data_type})()
                self._cells.append(cell)

        # Create mock worksheet
        ws = MockWorksheet()
        ws.add_cell("A1", "=B1+C1", "f")
        ws.add_cell("B1", "10", "n")
        ws.add_cell("C1", "20", "n")

        # Should work with analyze function
        result = analyze_worksheet_formulas(ws)
        assert isinstance(result, Success)

        analyzer = result.value
        assert len(analyzer.nodes) == 1  # One formula
        assert CellAddress("TestSheet!A1") in analyzer.nodes
