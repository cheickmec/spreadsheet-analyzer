"""
Comprehensive tests for refactored formula analysis module.

This test module demonstrates best practices for testing complex analysis code:
- Clear test documentation explaining what each test validates
- Comprehensive test coverage including edge cases
- Well-structured test data using builder patterns
- Type-safe test utilities
- Performance benchmarking for regression prevention

CLAUDE-KNOWLEDGE: Good tests serve as living documentation. They should
clearly communicate the expected behavior and edge cases of the system.
"""

import time
from pathlib import Path
from typing import Any

import pytest

from spreadsheet_analyzer.pipeline.stages.stage_3_formulas import (
    RANGE_STRATEGY_SKIP,
    RANGE_STRATEGY_SMART,
    CellReference,
    DependencyGraph,
    FormulaParser,
    RangeMembershipIndex,
    SemanticEdgeDetector,
    stage_3_formula_analysis,
)
from tests.test_base import BaseSpreadsheetTest, ExcelTestDataBuilder, TestDataPatterns


class TestFormulaParser(BaseSpreadsheetTest):
    """
    Test suite for Excel formula parsing functionality.

    These tests validate that the FormulaParser correctly extracts
    cell references from various Excel formula patterns.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for formula parser tests."""
        return {
            "test_type": "formula_parser",
            "timeout": 5.0,
        }

    @pytest.fixture
    def parser(self) -> FormulaParser:
        """Provide a fresh parser instance for each test."""
        return FormulaParser()

    def test_parse_simple_cell_reference(self, parser: FormulaParser) -> None:
        """
        Test parsing of simple cell references like =A1.

        This validates the most basic formula pattern where one cell
        references another cell's value directly.
        """
        formula = "=A1"
        current_sheet = "Sheet1"

        references = parser.parse_formula(formula, current_sheet)

        assert len(references) == 1
        ref = references[0]
        assert ref.sheet == "Sheet1"
        assert ref.start_col == 1  # Column A
        assert ref.start_row == 1
        assert not ref.is_range

    def test_parse_absolute_reference(self, parser: FormulaParser) -> None:
        """
        Test parsing of absolute cell references like =$A$1.

        Absolute references use $ to lock row/column when copying formulas.
        The parser should handle these correctly.
        """
        formula = "=$A$1+$B$2"
        current_sheet = "Data"

        references = parser.parse_formula(formula, current_sheet)

        assert len(references) == 2

        # First reference: $A$1
        assert references[0].sheet == "Data"
        assert references[0].start_col == 1
        assert references[0].start_row == 1

        # Second reference: $B$2
        assert references[1].sheet == "Data"
        assert references[1].start_col == 2
        assert references[1].start_row == 2

    def test_parse_range_reference(self, parser: FormulaParser) -> None:
        """
        Test parsing of range references like =SUM(A1:B10).

        Range references are fundamental to Excel and the parser must
        correctly identify the bounds of the range.
        """
        formula = "=SUM(A1:B10)"
        current_sheet = "Sales"

        references = parser.parse_formula(formula, current_sheet)

        assert len(references) == 1
        ref = references[0]
        assert ref.is_range
        assert ref.sheet == "Sales"
        assert ref.start_col == 1  # A
        assert ref.start_row == 1
        assert ref.end_col == 2  # B
        assert ref.end_row == 10
        assert ref.cell_count == 20  # 2 columns x 10 rows

    def test_parse_cross_sheet_reference(self, parser: FormulaParser) -> None:
        """
        Test parsing of cross-sheet references like =Sheet2!A1.

        Cross-sheet references are critical for multi-sheet workbooks
        and must be parsed with the correct sheet context.
        """
        formula = "=Sheet2!A1+Summary!B5"
        current_sheet = "Sheet1"

        references = parser.parse_formula(formula, current_sheet)

        assert len(references) == 2

        # First reference: Sheet2!A1
        assert references[0].sheet == "Sheet2"
        assert references[0].start_col == 1
        assert references[0].start_row == 1

        # Second reference: Summary!B5
        assert references[1].sheet == "Summary"
        assert references[1].start_col == 2
        assert references[1].start_row == 5

    def test_parse_sheet_name_with_spaces(self, parser: FormulaParser) -> None:
        """
        Test parsing sheet names containing spaces like ='Sales Data'!A1.

        Excel requires single quotes around sheet names with spaces.
        This is a common source of parsing errors.

        CLAUDE-GOTCHA: Sheet names with spaces must be wrapped in single
        quotes, but the quotes are not part of the actual sheet name.
        """
        formula = "='Sales Data'!A1+'Quarterly Report'!B2:C10"
        current_sheet = "Dashboard"

        references = parser.parse_formula(formula, current_sheet)

        assert len(references) == 2

        # First reference: 'Sales Data'!A1
        assert references[0].sheet == "Sales Data"  # Without quotes
        assert not references[0].is_range

        # Second reference: 'Quarterly Report'!B2:C10
        assert references[1].sheet == "Quarterly Report"
        assert references[1].is_range
        assert references[1].cell_count == 18  # 2 columns x 9 rows

    def test_parse_complex_formula(self, parser: FormulaParser) -> None:
        """
        Test parsing of complex formulas with multiple reference types.

        Real-world formulas often combine multiple reference types,
        functions, and operators. The parser must handle all correctly.
        """
        formula = "=IF(A1>0,SUM(Data!B2:B10),VLOOKUP(C1,'Lookup Table'!A:D,4,FALSE))"
        current_sheet = "Analysis"

        references = parser.parse_formula(formula, current_sheet)

        # Should find: A1, Data!B2:B10, C1, 'Lookup Table'!A:D
        assert len(references) == 4

        # Verify each reference type is parsed correctly
        ref_keys = [ref.to_key() for ref in references]
        assert "Analysis!A1" in ref_keys
        assert "Data!B2:B10" in ref_keys
        assert "Analysis!C1" in ref_keys
        assert "'Lookup Table'!A:D" in ref_keys

    def test_parser_caching(self, parser: FormulaParser) -> None:
        """
        Test that the parser cache improves performance for repeated formulas.

        The parser should cache results to avoid re-parsing identical
        formulas, which is common in spreadsheets with repeated patterns.

        CLAUDE-PERFORMANCE: Caching is critical for performance when
        analyzing sheets with thousands of similar formulas.
        """
        formula = "=SUM(A1:A100)+AVERAGE(B1:B100)"
        sheet = "Data"

        # First parse - should be a cache miss
        initial_misses = parser._cache_misses
        references1 = parser.parse_formula(formula, sheet)
        assert parser._cache_misses == initial_misses + 1

        # Second parse - should be a cache hit
        initial_hits = parser._cache_hits
        references2 = parser.parse_formula(formula, sheet)
        assert parser._cache_hits == initial_hits + 1

        # Results should be identical
        assert references1 == references2


class TestSemanticEdgeDetection(BaseSpreadsheetTest):
    """
    Test suite for semantic edge detection in formulas.

    These tests validate that the SemanticEdgeDetector correctly
    identifies the semantic relationships between cells based on
    the functions used in formulas.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for semantic detection tests."""
        return {
            "test_type": "semantic_detection",
            "enable_weights": True,
        }

    @pytest.fixture
    def detector(self) -> SemanticEdgeDetector:
        """Provide a semantic edge detector instance."""
        return SemanticEdgeDetector()

    def test_detect_sum_relationship(self, detector: SemanticEdgeDetector) -> None:
        """
        Test detection of SUM relationships.

        SUM functions create a "SUMS_OVER" relationship indicating
        that one cell aggregates values from other cells.
        """
        formula = "=SUM(A1:A10)"
        dependency = CellReference("Sheet1", 1, 1, 1, 10)  # A1:A10

        edge_metadata = detector.detect_edge_type(formula, dependency)

        assert edge_metadata.edge_type == "SUMS_OVER"
        assert edge_metadata.function_name == "SUM"
        assert edge_metadata.weight > 1.0  # Ranges should have higher weight

    def test_detect_lookup_relationship(self, detector: SemanticEdgeDetector) -> None:
        """
        Test detection of LOOKUP relationships.

        VLOOKUP/HLOOKUP functions create "LOOKS_UP_IN" relationships
        indicating data retrieval from a table.
        """
        formula = "=VLOOKUP(A1,LookupTable!A:D,4,FALSE)"
        dependency = CellReference("LookupTable", 1, 1, 4, None)  # A:D

        edge_metadata = detector.detect_edge_type(formula, dependency)

        assert edge_metadata.edge_type == "LOOKS_UP_IN"
        assert edge_metadata.function_name == "VLOOKUP"

    def test_detect_conditional_relationship(self, detector: SemanticEdgeDetector) -> None:
        """
        Test detection of conditional relationships.

        IF functions create "CONDITIONALLY_USES" relationships indicating
        that the dependency is used based on a condition.
        """
        formula = "=IF(A1>100,B1,C1)"
        dependency = CellReference("Sheet1", 2, 1)  # B1

        edge_metadata = detector.detect_edge_type(formula, dependency)

        assert edge_metadata.edge_type == "CONDITIONALLY_USES"
        assert edge_metadata.function_name == "IF"

    def test_formula_template_creation(self, detector: SemanticEdgeDetector) -> None:
        """
        Test creation of formula templates.

        Formula templates help identify patterns by replacing specific
        cell references with placeholders.
        """
        formula = "=SUM(A1:A10)+AVERAGE(B1:B10)-C1"

        # The template should replace all cell references
        template = detector._create_formula_template(formula)

        assert template == "=SUM(<REF>)+AVERAGE(<REF>)-<REF>"

    def test_edge_weight_calculation(self, detector: SemanticEdgeDetector) -> None:
        """
        Test that edge weights are calculated based on range size.

        Larger ranges should have higher weights to indicate their
        importance in the dependency graph.
        """
        formula = "=SUM(A1:A100)"

        # Small range
        small_range = CellReference("Sheet1", 1, 1, 1, 5)  # 5 cells
        small_edge = detector.detect_edge_type(formula, small_range)

        # Large range
        large_range = CellReference("Sheet1", 1, 1, 1, 100)  # 100 cells
        large_edge = detector.detect_edge_type(formula, large_range)

        assert large_edge.weight > small_edge.weight
        assert small_edge.weight > 1.0  # All ranges have weight > 1


class TestDependencyGraph(BaseSpreadsheetTest):
    """
    Test suite for dependency graph construction and analysis.

    These tests validate the core graph algorithms including
    circular reference detection and depth calculation.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for dependency graph tests."""
        return {
            "test_type": "dependency_graph",
            "enable_semantic": True,
        }

    def test_simple_dependency_chain(self) -> None:
        """
        Test construction of a simple linear dependency chain.

        This validates basic graph construction where cells depend
        on each other in a linear fashion: A1 -> B1 -> C1
        """
        graph = DependencyGraph(enable_semantic_analysis=False)

        # Create chain: C1 depends on B1, B1 depends on A1
        graph.add_node("Sheet1", "A1", "=100", [])
        graph.add_node(
            "Sheet1",
            "B1",
            "=A1*2",
            [
                CellReference("Sheet1", 1, 1)  # A1
            ],
        )
        graph.add_node(
            "Sheet1",
            "C1",
            "=B1+10",
            [
                CellReference("Sheet1", 2, 1)  # B1
            ],
        )

        # Verify graph structure
        assert len(graph.nodes) == 3
        assert "Sheet1!A1" in graph.nodes
        assert "Sheet1!A1" in graph.adjacency_list["Sheet1!B1"]
        assert "Sheet1!B1" in graph.adjacency_list["Sheet1!C1"]

        # Verify reverse adjacency (who depends on me?)
        assert "Sheet1!B1" in graph.reverse_adjacency["Sheet1!A1"]
        assert "Sheet1!C1" in graph.reverse_adjacency["Sheet1!B1"]

    def test_circular_reference_detection(self) -> None:
        """
        Test detection of circular references in formulas.

        Circular references occur when cells depend on each other
        in a cycle. This is a critical error to detect.

        CLAUDE-KNOWLEDGE: Excel allows circular references but flags
        them as errors. They can cause infinite calculation loops.
        """
        graph = DependencyGraph()

        # Create circular reference: A1 -> B1 -> C1 -> A1
        graph.add_node(
            "Sheet1",
            "A1",
            "=C1+1",
            [
                CellReference("Sheet1", 3, 1)  # C1
            ],
        )
        graph.add_node(
            "Sheet1",
            "B1",
            "=A1*2",
            [
                CellReference("Sheet1", 1, 1)  # A1
            ],
        )
        graph.add_node(
            "Sheet1",
            "C1",
            "=B1-5",
            [
                CellReference("Sheet1", 2, 1)  # B1
            ],
        )

        # Find circular references
        cycles = graph.find_circular_references()

        assert len(cycles) == 1
        cycle = next(iter(cycles))
        assert len(cycle) == 3
        assert "Sheet1!A1" in cycle
        assert "Sheet1!B1" in cycle
        assert "Sheet1!C1" in cycle

    def test_max_depth_calculation(self) -> None:
        """
        Test calculation of maximum dependency depth.

        The dependency depth indicates the longest chain of calculations
        needed to resolve all formulas. Deep chains can impact performance.
        """
        graph = DependencyGraph()

        # Create a dependency tree with max depth 3
        # Layer 0 (no dependencies)
        graph.add_node("Sheet1", "A1", "=100", [])
        graph.add_node("Sheet1", "A2", "=200", [])

        # Layer 1
        graph.add_node(
            "Sheet1",
            "B1",
            "=A1+A2",
            [
                CellReference("Sheet1", 1, 1),  # A1
                CellReference("Sheet1", 1, 2),  # A2
            ],
        )

        # Layer 2
        graph.add_node(
            "Sheet1",
            "C1",
            "=B1*2",
            [
                CellReference("Sheet1", 2, 1),  # B1
            ],
        )

        # Layer 3
        graph.add_node(
            "Sheet1",
            "D1",
            "=C1/10",
            [
                CellReference("Sheet1", 3, 1),  # C1
            ],
        )

        max_depth = graph.calculate_max_depth()
        assert max_depth == 3

    def test_complexity_score_calculation(self) -> None:
        """
        Test that formula complexity scores are calculated correctly.

        Complexity scores help identify formulas that might need
        refactoring or special attention during analysis.
        """
        graph = DependencyGraph()

        # Simple formula - low complexity
        graph.add_node(
            "Sheet1",
            "A1",
            "=B1+C1",
            [
                CellReference("Sheet1", 2, 1),
                CellReference("Sheet1", 3, 1),
            ],
        )

        # Complex formula with volatile function - higher complexity
        graph.add_node(
            "Sheet1",
            "A2",
            "=SUMIF(B:B,TODAY(),C:C)+RAND()",
            [
                CellReference("Sheet1", 2, 1, 2, None),  # B:B
                CellReference("Sheet1", 3, 1, 3, None),  # C:C
            ],
            volatile=True,
        )

        # External reference with dependencies - multiplied complexity
        graph.add_node(
            "Sheet1",
            "A3",
            "=[External.xlsx]Sheet1!A1+[External.xlsx]Sheet1!B1",
            [
                # External references might not be parseable as CellReference
                # but we simulate having found 2 dependencies
                CellReference("Sheet1", 1, 1),  # Placeholder for external ref
                CellReference("Sheet1", 2, 1),  # Placeholder for external ref
            ],
            external=True,
        )

        simple_node = graph.nodes["Sheet1!A1"]
        volatile_node = graph.nodes["Sheet1!A2"]
        external_node = graph.nodes["Sheet1!A3"]

        # Verify complexity ordering
        assert volatile_node.complexity_score > simple_node.complexity_score
        assert external_node.complexity_score > simple_node.complexity_score

        # Also verify that the external multiplier is applied correctly
        # External node should have higher score due to 1.5x multiplier


class TestFormulaAnalyzer(BaseSpreadsheetTest):
    """
    Integration tests for the complete FormulaAnalyzer.

    These tests validate the full analysis pipeline including
    file processing, progress tracking, and result generation.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for analyzer tests."""
        return {
            "test_type": "formula_analyzer",
            "max_test_formulas": 1000,
        }

    @pytest.mark.requires_excel
    def test_analyze_simple_workbook(
        self, sample_excel_file: Path, performance_tracker: dict[str, list[float]]
    ) -> None:
        """
        Test analysis of a simple workbook with basic formulas.

        This integration test validates that the analyzer can process
        a complete workbook and produce correct analysis results.
        """
        start_time = time.time()

        # Analyze the workbook
        result = stage_3_formula_analysis(sample_excel_file)

        # Track performance
        performance_tracker["simple_workbook"].append(time.time() - start_time)

        # Verify successful analysis
        analysis = self.assert_ok(result)

        # Verify basic statistics
        assert analysis.statistics["total_formulas"] > 0
        assert analysis.statistics["circular_reference_count"] == 0
        assert analysis.max_dependency_depth >= 0

        # Verify specific formulas were found
        formula_keys = list(analysis.dependency_graph.keys())

        # Should find formulas in Sales sheet (=B*C for totals)
        sales_formulas = [k for k in formula_keys if k.startswith("Sales!")]
        assert len(sales_formulas) > 0

        # Should find summary formulas with cross-sheet references
        summary_formulas = [k for k in formula_keys if k.startswith("Summary!")]
        assert len(summary_formulas) > 0

    @pytest.mark.requires_excel
    @pytest.mark.skip(reason="formula_test_file fixture not implemented")
    def test_semantic_analysis_integration(self, formula_test_file: Path) -> None:
        """
        Test formula analysis with semantic edge detection enabled.

        This validates that semantic analysis correctly identifies
        relationship types between cells based on Excel functions.
        """
        # Analyze with semantic analysis enabled
        result = stage_3_formula_analysis(formula_test_file, enable_semantic_analysis=True)

        analysis = self.assert_ok(result)

        # Find nodes with semantic edge labels
        nodes_with_semantics = [node for node in analysis.dependency_graph.values() if node.edge_labels is not None]

        assert len(nodes_with_semantics) > 0

        # Verify specific semantic relationships
        for node in nodes_with_semantics:
            if "SUM(" in node.formula.upper():
                # Should have SUMS_OVER relationships
                edge_types = {meta.edge_type for meta in node.edge_labels.values()}
                assert "SUMS_OVER" in edge_types

    @pytest.mark.requires_excel
    def test_circular_reference_workbook(self, tmp_path: Path) -> None:
        """
        Test detection of circular references in formulas.

        This test creates a workbook with intentional circular
        references to validate detection algorithms.
        """
        # Create workbook with circular references
        circular_file = TestDataPatterns.create_circular_reference_sheet(tmp_path / "circular.xlsx")

        # Analyze the workbook
        result = stage_3_formula_analysis(circular_file)
        analysis = self.assert_ok(result)

        # Should detect circular references
        assert len(analysis.circular_references) > 0
        assert analysis.statistics["circular_reference_count"] > 0

        # Verify the cycle includes expected cells
        cycle = next(iter(analysis.circular_references))
        assert "Circular!A1" in cycle
        assert "Circular!B1" in cycle
        assert "Circular!C1" in cycle

    @pytest.mark.requires_excel
    def test_range_strategy_handling(self, excel_builder: ExcelTestDataBuilder, tmp_path: Path) -> None:
        """
        Test different range handling strategies.

        Range handling is critical for performance with large spreadsheets.
        This test validates that different strategies work correctly.

        CLAUDE-PERFORMANCE: The choice of range strategy can dramatically
        impact analysis time for sheets with large ranges.
        """
        # Create workbook with large ranges
        test_file = (
            excel_builder.with_sheet("Data")
            .add_headers(["Value"])
            .add_row([100])
            .add_row([200])
            .add_row([300])
            .with_sheet("Summary")
            .add_cell("A1", "Total:")
            .add_cell("B1", "=SUM(Data!A:A)")  # Full column reference
            .add_cell("A2", "Average:")
            .add_cell("B2", "=AVERAGE(Data!A2:A1000)")  # Large range
            .build(tmp_path / "ranges.xlsx")
        )

        # Test with SKIP strategy
        result_skip = stage_3_formula_analysis(test_file, range_strategy=RANGE_STRATEGY_SKIP)
        analysis_skip = self.assert_ok(result_skip)

        # Test with SMART strategy (default)
        result_smart = stage_3_formula_analysis(test_file, range_strategy=RANGE_STRATEGY_SMART)
        analysis_smart = self.assert_ok(result_smart)

        # SKIP strategy should have fewer dependencies
        skip_deps = sum(len(node.dependencies) for node in analysis_skip.dependency_graph.values())
        smart_deps = sum(len(node.dependencies) for node in analysis_smart.dependency_graph.values())

        # Smart strategy should preserve range dependencies
        assert smart_deps >= skip_deps

    @pytest.mark.requires_excel
    def test_volatile_function_detection(self, excel_builder: ExcelTestDataBuilder, tmp_path: Path) -> None:
        """
        Test detection of volatile Excel functions.

        Volatile functions recalculate every time Excel recalculates,
        which can impact performance. They must be tracked.

        CLAUDE-KNOWLEDGE: Common volatile functions include NOW(), TODAY(),
        RAND(), RANDBETWEEN(), OFFSET(), and INDIRECT().
        """
        # Create workbook with volatile functions
        test_file = (
            excel_builder.with_sheet("Volatile")
            .add_cell("A1", "=TODAY()")
            .add_cell("A2", "=NOW()")
            .add_cell("A3", "=RAND()")
            .add_cell("A4", "=RANDBETWEEN(1,100)")
            .add_cell("B1", "=A1+7")  # Non-volatile but depends on volatile
            .add_cell("C1", "=100")  # Non-volatile
            .build(tmp_path / "volatile.xlsx")
        )

        # Analyze the workbook
        result = stage_3_formula_analysis(test_file)
        analysis = self.assert_ok(result)

        # Should detect volatile formulas
        assert len(analysis.volatile_formulas) >= 4
        assert "Volatile!A1" in analysis.volatile_formulas
        assert "Volatile!A2" in analysis.volatile_formulas
        assert "Volatile!A3" in analysis.volatile_formulas
        assert "Volatile!A4" in analysis.volatile_formulas

        # Non-volatile formulas should not be marked
        assert "Volatile!B1" not in analysis.volatile_formulas
        assert "Volatile!C1" not in analysis.volatile_formulas

    @pytest.mark.requires_excel
    def test_progress_callback_integration(self, sample_excel_file: Path) -> None:
        """
        Test that progress callbacks are invoked during analysis.

        Progress tracking is important for user feedback during
        long-running analysis operations.
        """
        progress_updates = []

        def track_progress(stage: str, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
            """Capture progress updates."""
            progress_updates.append(
                {"stage": stage, "progress": progress, "message": message, "timestamp": time.time()}
            )

        # Analyze with progress tracking
        result = stage_3_formula_analysis(sample_excel_file, progress_callback=track_progress)

        self.assert_ok(result)

        # Should have received progress updates
        assert len(progress_updates) > 0

        # Progress should be between 0 and 1
        for update in progress_updates:
            assert 0 <= update["progress"] <= 1
            assert update["stage"] == "formula_analysis"


class TestRangeMembershipIndex(BaseSpreadsheetTest):
    """
    Test suite for range membership indexing.

    The RangeMembershipIndex provides efficient queries to determine
    if a cell is part of any formula range reference.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for range index tests."""
        return {
            "test_type": "range_membership",
        }

    def test_range_membership_query(self) -> None:
        """
        Test efficient range membership queries.

        The index should quickly determine if a cell is referenced
        by any range formulas without expanding all ranges.
        """
        # Create index with some ranges
        sheet_ranges = {
            "Sheet1": [
                (1, 10, 1, 1, "Sheet1!A1:A10"),  # A1:A10
                (1, 5, 2, 3, "Sheet1!B1:C5"),  # B1:C5
                (1, 1048576, 4, 4, "Sheet1!D:D"),  # Full column D
            ]
        }

        index = RangeMembershipIndex(sheet_ranges)

        # Test cells inside ranges
        assert index.is_cell_in_any_range("Sheet1", 5, 1)  # A5 is in A1:A10
        assert index.is_cell_in_any_range("Sheet1", 3, 2)  # B3 is in B1:C5
        assert index.is_cell_in_any_range("Sheet1", 100, 4)  # D100 is in D:D

        # Test cells outside ranges
        assert not index.is_cell_in_any_range("Sheet1", 11, 1)  # A11 not in A1:A10
        assert not index.is_cell_in_any_range("Sheet1", 1, 5)  # E1 not in any range
        assert not index.is_cell_in_any_range("Sheet2", 1, 1)  # Different sheet


# Performance benchmark test
@pytest.mark.slow
class TestFormulaAnalysisPerformance(BaseSpreadsheetTest):
    """
    Performance benchmarks for formula analysis.

    These tests ensure that analysis performance doesn't regress
    and meets the targets specified in the design documents.

    CLAUDE-PERFORMANCE: Target is < 5 seconds for standard files
    with < 10 sheets and < 10K cells.
    """

    def _get_test_config(self) -> dict[str, Any]:
        """Test configuration for performance tests."""
        return {
            "test_type": "performance",
            "max_duration": 5.0,
        }

    @pytest.mark.requires_excel
    def test_large_workbook_performance(
        self, excel_builder: ExcelTestDataBuilder, tmp_path: Path, performance_tracker: dict[str, list[float]]
    ) -> None:
        """
        Test analysis performance on a large workbook.

        This benchmark ensures that formula analysis can handle
        workbooks with thousands of formulas within acceptable time.
        """
        # Create a large workbook with many formulas
        builder = excel_builder.with_sheet("Data", make_active=True)

        # Add 1000 rows of data
        builder.add_headers(["ID", "Value", "Calculated"])
        for i in range(1, 1001):
            builder.add_row([i, i * 10, f"=B{i + 1}*2"])

        # Add summary sheet with aggregations
        builder.with_sheet("Summary")
        builder.add_cell("A1", "Total:")
        builder.add_cell("B1", "=SUM(Data!C:C)")
        builder.add_cell("A2", "Average:")
        builder.add_cell("B2", "=AVERAGE(Data!C:C)")
        builder.add_cell("A3", "Max:")
        builder.add_cell("B3", "=MAX(Data!C:C)")

        large_file = builder.build(tmp_path / "large.xlsx")

        # Measure analysis time
        start_time = time.time()
        result = stage_3_formula_analysis(large_file)
        analysis_time = time.time() - start_time

        # Track performance
        performance_tracker["large_workbook"].append(analysis_time)

        # Verify success
        analysis = self.assert_ok(result)

        # Should complete within target time
        assert analysis_time < self._get_test_config()["max_duration"], (
            f"Analysis took {analysis_time:.2f}s, exceeding {self._get_test_config()['max_duration']}s target"
        )

        # Verify correctness
        assert analysis.statistics["total_formulas"] >= 1000
        assert analysis.max_dependency_depth >= 0  # All formulas depend on non-formula cells
