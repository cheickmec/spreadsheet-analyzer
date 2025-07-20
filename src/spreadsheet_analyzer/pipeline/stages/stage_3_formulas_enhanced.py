"""
Enhanced Stage 3: Formula Analysis with Semantic Edge Detection.

This module extends the basic formula analysis to capture semantic
relationships and additional metadata for richer graph representation.
"""

import re
from typing import Any

from spreadsheet_analyzer.pipeline.stages.stage_3_formulas import (
    FormulaAnalyzer,
    FormulaParser,
    format_cell_key,
    parse_range_size,
)
from spreadsheet_analyzer.pipeline.types import CellReference


class SemanticEdgeDetector:
    """Detects semantic edge types from formula patterns."""

    # Function to semantic edge mapping
    SEMANTIC_MAPPINGS = {
        # Aggregation functions
        "SUM": "SUMS_OVER",
        "SUMIF": "SUMS_OVER",
        "SUMIFS": "SUMS_OVER",
        "AVERAGE": "AVERAGES_OVER",
        "AVERAGEIF": "AVERAGES_OVER",
        "AVERAGEIFS": "AVERAGES_OVER",
        "COUNT": "COUNTS_IN",
        "COUNTIF": "COUNTS_IN",
        "COUNTIFS": "COUNTS_IN",
        "COUNTA": "COUNTS_IN",
        # Lookup functions
        "VLOOKUP": "LOOKS_UP_IN",
        "HLOOKUP": "LOOKS_UP_IN",
        "LOOKUP": "LOOKS_UP_IN",
        "INDEX": "LOOKS_UP_IN",
        "MATCH": "MATCHES_IN",
        # Statistical functions
        "MAX": "FINDS_MAX_IN",
        "MIN": "FINDS_MIN_IN",
        "MEDIAN": "CALCULATES_MEDIAN_FROM",
        "STDEV": "CALCULATES_STDEV_FROM",
        # Default
        "DEFAULT": "DEPENDS_ON",
    }

    @classmethod
    def detect_edge_type(cls, formula: str, dep_position: int = 0) -> tuple[str, str, dict[str, Any]]:
        """
        Detect semantic edge type from formula context.

        Returns:
            Tuple of (edge_label, function_name, metadata)
        """
        # Extract function name
        func_match = re.match(r"=\s*([A-Z]+)\s*\(", formula.upper())
        if not func_match:
            return ("DEPENDS_ON", "", {})

        func_name = func_match.group(1)
        edge_label = cls.SEMANTIC_MAPPINGS.get(func_name, "DEPENDS_ON")

        # Extract formula context
        formula_template = cls._create_formula_template(formula, func_name)

        metadata = {
            "function_name": func_name,
            "formula_context": formula_template,
            "argument_position": dep_position,
            "is_nested": cls._is_nested_function(formula, func_name),
        }

        return (edge_label, func_name, metadata)

    @staticmethod
    def _create_formula_template(formula: str, func_name: str) -> str:
        """Create a template showing how the reference is used."""
        # Simplified - in production would parse more carefully
        template = formula[:50]  # First 50 chars
        if len(formula) > 50:
            template += "..."
        return template.replace("\n", " ")

    @staticmethod
    def _is_nested_function(formula: str, func_name: str) -> bool:
        """Check if function is nested within another function."""
        # Count parentheses before function name
        before_func = formula.upper().split(func_name)[0]
        open_parens = before_func.count("(")
        close_parens = before_func.count(")")
        return open_parens > close_parens


class EnhancedFormulaAnalyzer(FormulaAnalyzer):
    """
    Enhanced formula analyzer that captures semantic relationships
    and additional metadata.
    """

    def __init__(self, workbook, *, data_only: bool = False):
        """Initialize enhanced analyzer."""
        super().__init__(workbook, data_only=data_only)
        self.edge_detector = SemanticEdgeDetector()
        self.cell_metadata = {}  # Store additional cell metadata

    def _analyze_formula(self, cell, sheet_name: str, parser: FormulaParser):
        """Enhanced formula analysis with semantic edge detection."""
        formula = str(cell.value) if cell.value else ""
        cell_ref = cell.coordinate
        cell_key = format_cell_key(sheet_name, cell_ref)

        # Detect if formula contains volatile functions
        is_volatile = any(func in formula.upper() for func in self.VOLATILE_FUNCTIONS)

        # Store cell metadata
        self.cell_metadata[cell_key] = {
            "value_type": self._detect_value_type(cell),
            "is_volatile": is_volatile,
            "has_error": cell.data_type == "e" if hasattr(cell, "data_type") else False,
        }

        # Add node to graph with enhanced metadata
        self.graph.add_node(cell_key, sheet_name, cell_ref, formula, node_type="cell")

        # Track volatile formulas
        if is_volatile:
            self.volatile_formulas.append(cell_key)

        # Check for external references
        if "[" in formula and "]" in formula:
            self.external_references.append(cell_key)

        # Parse dependencies with position tracking
        dependencies = parser.parse_formula(formula)

        # Add edges with semantic information
        for idx, dep in enumerate(dependencies):
            dep_key = format_cell_key(dep.sheet, dep.cell)

            # Detect semantic edge type
            edge_label, func_name, edge_metadata = self.edge_detector.detect_edge_type(formula, idx)

            if not dep.is_range:
                # Single cell reference
                self._add_semantic_edge(
                    cell_key,
                    dep_key,
                    edge_label,
                    {
                        **edge_metadata,
                        "ref_type": "single_cell",
                        "is_cross_sheet": sheet_name != dep.sheet,
                        "weight": 1.0,
                    },
                )
            else:
                # Range reference - enhanced handling
                self._handle_range_dependency_enhanced(cell_key, dep, dep_key, edge_label, edge_metadata, sheet_name)

    def _add_semantic_edge(self, from_key: str, to_key: str, edge_label: str, metadata: dict):
        """Add edge with semantic label and metadata."""
        # Store edge metadata for later use in graph export
        edge_key = f"{from_key}->{to_key}"
        if not hasattr(self, "edge_metadata"):
            self.edge_metadata = {}
        self.edge_metadata[edge_key] = {"label": edge_label, **metadata}

        # Add to graph (base class handles the structure)
        self.graph.add_edge(from_key, to_key)

    def _handle_range_dependency_enhanced(
        self,
        formula_key: str,
        dep: CellReference,
        range_key: str,
        edge_label: str,
        edge_metadata: dict,
        source_sheet: str,
    ):
        """Enhanced range handling with metadata."""
        # Calculate range size and metadata
        range_size, range_metadata = parse_range_size(dep.cell)

        # Always create range node with enhanced metadata
        self.graph.add_node(
            range_key,
            sheet=dep.sheet,
            cell_ref=dep.cell,
            formula=None,
            node_type="range",
            range_metadata={
                **range_metadata,
                "pattern_type": "unknown",  # Would be detected in post-processing
                "stats": {},  # Would be calculated from actual data
            },
        )

        # Add semantic edge to range
        self._add_semantic_edge(
            formula_key,
            range_key,
            edge_label,
            {
                **edge_metadata,
                "ref_type": "range",
                "is_cross_sheet": source_sheet != dep.sheet,
                "weight": float(range_size),
            },
        )

    def _detect_value_type(self, cell) -> str:
        """Detect the data type of a cell."""
        if cell.data_type == "n":
            return "number"
        elif cell.data_type == "s":
            return "text"
        elif cell.data_type == "d":
            return "date"
        elif cell.data_type == "b":
            return "boolean"
        elif cell.data_type == "e":
            return "error"
        elif cell.data_type == "f":
            return "formula"
        else:
            return "unknown"

    def finalize(self):
        """Finalize with enhanced metadata."""
        result = super().analyze()

        # Enhance the result with collected metadata
        # This would be integrated into the graph export
        if hasattr(self, "edge_metadata"):
            result.edge_metadata = self.edge_metadata

        result.cell_metadata = self.cell_metadata

        return result
