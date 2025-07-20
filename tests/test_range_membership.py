"""
Tests for range membership tracking and empty cell queries.

This module tests the RangeMembershipIndex and related functionality
for tracking which empty cells are part of formula ranges.
"""

import pytest

from spreadsheet_analyzer.graph_db.range_membership import (
    RangeBounds,
    RangeMembershipIndex,
    col_to_num,
    num_to_col,
    parse_cell_ref,
    parse_range_ref,
)


class TestColumnConversion:
    """Test Excel column conversion utilities."""

    def test_col_to_num(self):
        """Test column letter to number conversion."""
        assert col_to_num("A") == 1
        assert col_to_num("B") == 2
        assert col_to_num("Z") == 26
        assert col_to_num("AA") == 27
        assert col_to_num("AB") == 28
        assert col_to_num("AZ") == 52
        assert col_to_num("BA") == 53
        assert col_to_num("ZZ") == 702

    def test_num_to_col(self):
        """Test column number to letter conversion."""
        assert num_to_col(1) == "A"
        assert num_to_col(2) == "B"
        assert num_to_col(26) == "Z"
        assert num_to_col(27) == "AA"
        assert num_to_col(28) == "AB"
        assert num_to_col(52) == "AZ"
        assert num_to_col(53) == "BA"
        assert num_to_col(702) == "ZZ"

    def test_col_conversion_roundtrip(self):
        """Test that conversions are reversible."""
        for num in [1, 10, 26, 27, 100, 500, 702]:
            col = num_to_col(num)
            assert col_to_num(col) == num


class TestCellParsing:
    """Test cell and range reference parsing."""

    def test_parse_cell_ref_simple(self):
        """Test parsing simple cell references."""
        sheet, row, col = parse_cell_ref("A1")
        assert sheet == ""
        assert row == 1
        assert col == 1

        sheet, row, col = parse_cell_ref("Z100")
        assert sheet == ""
        assert row == 100
        assert col == 26

    def test_parse_cell_ref_with_sheet(self):
        """Test parsing cell references with sheet names."""
        sheet, row, col = parse_cell_ref("Sheet1!B5")
        assert sheet == "Sheet1"
        assert row == 5
        assert col == 2

        sheet, row, col = parse_cell_ref("'My Sheet'!C10")
        assert sheet == "My Sheet"
        assert row == 10
        assert col == 3

    def test_parse_range_ref_simple(self):
        """Test parsing simple range references."""
        sheet, bounds = parse_range_ref("A1:B2")
        assert sheet == ""
        assert bounds.start_row == 1
        assert bounds.start_col == 1
        assert bounds.end_row == 2
        assert bounds.end_col == 2

    def test_parse_range_ref_with_sheet(self):
        """Test parsing range references with sheet names."""
        sheet, bounds = parse_range_ref("Sheet1!B5:D10")
        assert sheet == "Sheet1"
        assert bounds.start_row == 5
        assert bounds.start_col == 2
        assert bounds.end_row == 10
        assert bounds.end_col == 4

    def test_parse_range_ref_reordering(self):
        """Test that range bounds are properly ordered."""
        # Reversed row order
        sheet, bounds = parse_range_ref("A10:A1")
        assert bounds.start_row == 1
        assert bounds.end_row == 10

        # Reversed column order
        sheet, bounds = parse_range_ref("D5:A5")
        assert bounds.start_col == 1
        assert bounds.end_col == 4


class TestRangeBounds:
    """Test RangeBounds functionality."""

    def test_contains(self):
        """Test cell containment in range."""
        bounds = RangeBounds("Sheet1", 5, 2, 10, 4)  # B5:D10

        # Inside range
        assert bounds.contains(5, 2)  # B5
        assert bounds.contains(7, 3)  # C7
        assert bounds.contains(10, 4)  # D10

        # Outside range
        assert not bounds.contains(4, 2)  # B4
        assert not bounds.contains(11, 2)  # B11
        assert not bounds.contains(7, 1)  # A7
        assert not bounds.contains(7, 5)  # E7

    def test_intersects(self):
        """Test range intersection."""
        bounds1 = RangeBounds("Sheet1", 5, 2, 10, 4)  # B5:D10

        # Fully contained
        bounds2 = RangeBounds("Sheet1", 6, 3, 8, 3)  # C6:C8
        assert bounds1.intersects(bounds2)
        assert bounds2.intersects(bounds1)

        # Partial overlap
        bounds3 = RangeBounds("Sheet1", 8, 3, 12, 5)  # C8:E12
        assert bounds1.intersects(bounds3)

        # No overlap
        bounds4 = RangeBounds("Sheet1", 1, 1, 3, 1)  # A1:A3
        assert not bounds1.intersects(bounds4)


class TestRangeMembershipIndex:
    """Test the RangeMembershipIndex class."""

    @pytest.fixture
    def index(self):
        """Create a fresh index for testing."""
        return RangeMembershipIndex()

    def test_add_range(self, index):
        """Test adding ranges to the index."""
        index.add_range("B1:B10", "Sheet1!A1", "Sheet1")
        index.add_range("D5:F8", "Sheet1!C2", "Sheet1")

        # Check that ranges were added
        all_ranges = index.get_all_ranges()
        assert "Sheet1" in all_ranges
        assert len(all_ranges["Sheet1"]) == 2

    def test_cell_membership_queries(self, index):
        """Test querying which cells are in ranges."""
        # Add some ranges
        index.add_range("B1:B10", "Sheet1!A1", "Sheet1")
        index.add_range("D5:F8", "Sheet1!C2", "Sheet1")

        # Test cells in first range
        assert index.is_cell_in_any_range("Sheet1", "B5")
        assert index.is_cell_in_any_range("Sheet1", "B1")
        assert index.is_cell_in_any_range("Sheet1", "B10")

        # Test cells in second range
        assert index.is_cell_in_any_range("Sheet1", "D5")
        assert index.is_cell_in_any_range("Sheet1", "E6")
        assert index.is_cell_in_any_range("Sheet1", "F8")

        # Test cells not in any range
        assert not index.is_cell_in_any_range("Sheet1", "A1")
        assert not index.is_cell_in_any_range("Sheet1", "C3")
        assert not index.is_cell_in_any_range("Sheet1", "G10")

    def test_get_ranges_containing_cell(self, index):
        """Test getting all formulas that reference a cell through ranges."""
        # Add overlapping ranges
        index.add_range("B1:B10", "Sheet1!A1", "Sheet1")
        index.add_range("A5:C15", "Sheet1!D1", "Sheet1")
        index.add_range("B8:B12", "Sheet1!E1", "Sheet1")

        # Cell in one range
        formulas = index.get_ranges_containing_cell("Sheet1", "B3")
        assert len(formulas) == 1
        assert "Sheet1!A1" in formulas

        # Cell in overlapping ranges
        formulas = index.get_ranges_containing_cell("Sheet1", "B9")
        assert len(formulas) == 3
        assert "Sheet1!A1" in formulas
        assert "Sheet1!D1" in formulas
        assert "Sheet1!E1" in formulas

        # Cell not in any range
        formulas = index.get_ranges_containing_cell("Sheet1", "Z99")
        assert len(formulas) == 0

    def test_range_intersection_queries(self, index):
        """Test finding ranges that intersect with a query range."""
        # Add some ranges
        index.add_range("B1:B10", "Sheet1!A1", "Sheet1")
        index.add_range("D5:F8", "Sheet1!C2", "Sheet1")
        index.add_range("A20:Z30", "Sheet1!M1", "Sheet1")

        # Query range that intersects first two
        intersections = index.get_ranges_intersecting_range("Sheet1", "B5", "E6")
        assert len(intersections) == 2
        formula_keys = [item[0] for item in intersections]
        assert "Sheet1!A1" in formula_keys
        assert "Sheet1!C2" in formula_keys

        # Query range that intersects only the large range
        intersections = index.get_ranges_intersecting_range("Sheet1", "M25", "N26")
        assert len(intersections) == 1
        assert intersections[0][0] == "Sheet1!M1"

        # Query range with no intersections
        intersections = index.get_ranges_intersecting_range("Sheet1", "AA1", "AB2")
        assert len(intersections) == 0

    def test_empty_cell_scenario(self, index):
        """Test the motivating scenario: finding dependencies of empty cells."""
        # Simulate a SUM formula that references a range including empty cells
        index.add_range("B1:B1000", "Sheet1!C1", "Sheet1")  # =SUM(B1:B1000)

        # Now we can query any cell in that range, even if it's empty
        assert index.is_cell_in_any_range("Sheet1", "B500")

        # Get the formula that depends on this empty cell
        formulas = index.get_ranges_containing_cell("Sheet1", "B500")
        assert len(formulas) == 1
        assert formulas[0] == "Sheet1!C1"

        # This tells us that if we modify B500, we need to recalculate C1

    def test_multiple_sheets(self, index):
        """Test that ranges are properly isolated by sheet."""
        # Add ranges to different sheets
        index.add_range("A1:A10", "Sheet1!B1", "Sheet1")
        index.add_range("A1:A10", "Sheet2!B1", "Sheet2")

        # Query should only return results from the correct sheet
        formulas1 = index.get_ranges_containing_cell("Sheet1", "A5")
        assert len(formulas1) == 1
        assert formulas1[0] == "Sheet1!B1"

        formulas2 = index.get_ranges_containing_cell("Sheet2", "A5")
        assert len(formulas2) == 1
        assert formulas2[0] == "Sheet2!B1"

        # No results from wrong sheet
        assert not index.is_cell_in_any_range("Sheet3", "A5")

    def test_performance_with_many_ranges(self, index):
        """Test that the index performs well with many ranges."""
        # Add 100 ranges
        for i in range(100):
            start_row = i * 10 + 1
            end_row = start_row + 9
            range_ref = f"A{start_row}:A{end_row}"
            formula_key = f"Sheet1!B{i + 1}"
            index.add_range(range_ref, formula_key, "Sheet1")

        # Query should still be fast
        assert index.is_cell_in_any_range("Sheet1", "A555")
        formulas = index.get_ranges_containing_cell("Sheet1", "A555")
        assert len(formulas) == 1
        assert formulas[0] == "Sheet1!B56"  # Range 55 covers rows 551-560


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_financial_model_scenario(self):
        """Test a typical financial model with subtotals and totals."""
        index = RangeMembershipIndex()

        # Revenue subtotals
        index.add_range("B2:B13", "Sheet1!B14", "Sheet1")  # Q1 total
        index.add_range("C2:C13", "Sheet1!C14", "Sheet1")  # Q2 total
        index.add_range("D2:D13", "Sheet1!D14", "Sheet1")  # Q3 total
        index.add_range("E2:E13", "Sheet1!E14", "Sheet1")  # Q4 total

        # Annual total
        index.add_range("B14:E14", "Sheet1!F14", "Sheet1")  # Year total

        # Empty cell B7 affects Q1 total
        assert index.is_cell_in_any_range("Sheet1", "B7")
        q1_deps = index.get_ranges_containing_cell("Sheet1", "B7")
        assert "Sheet1!B14" in q1_deps

        # Q1 total cell affects year total
        assert index.is_cell_in_any_range("Sheet1", "B14")
        year_deps = index.get_ranges_containing_cell("Sheet1", "B14")
        assert "Sheet1!F14" in year_deps

    def test_data_validation_scenario(self):
        """Test finding all cells that need validation in a range."""
        index = RangeMembershipIndex()

        # Multiple validation formulas checking ranges
        index.add_range("A2:A100", "Validation!B1", "Data")  # Check column A
        index.add_range("B2:B100", "Validation!B2", "Data")  # Check column B
        index.add_range("A2:E100", "Validation!B3", "Data")  # Check all data

        # Cell A50 is checked by two validation formulas
        validation_formulas = index.get_ranges_containing_cell("Data", "A50")
        assert len(validation_formulas) == 2
        assert "Validation!B1" in validation_formulas
        assert "Validation!B3" in validation_formulas

        # Cell C50 is only checked by the all-data validation
        validation_formulas = index.get_ranges_containing_cell("Data", "C50")
        assert len(validation_formulas) == 1
        assert "Validation!B3" in validation_formulas
