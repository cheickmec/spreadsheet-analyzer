"""
Proposed semantic classification system for spreadsheet regions.
Based on research from TableSense, SpreadsheetLLM, and Microsoft Research.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StructuralType(str, Enum):
    """
    Tier 1: Structural classification based on layout patterns.
    These are proven categories from academic research.
    """

    DATA_TABLE = "DataTable"  # Regular rows/columns with consistent schema
    HEADER_ZONE = "HeaderZone"  # Column/row headers, possibly hierarchical
    AGGREGATION_ZONE = "AggregationZone"  # Summaries, totals, calculated metrics
    METADATA_ZONE = "MetadataZone"  # Titles, descriptions, parameters
    VISUALIZATION_ANCHOR = "VisualizationAnchor"  # Charts, sparklines, conditional formatting
    FORM_INPUT_ZONE = "FormInputZone"  # Data entry areas, dropdowns, validation rules
    NAVIGATION_ZONE = "NavigationZone"  # Links, buttons, index areas
    UNSTRUCTURED_TEXT = "UnstructuredText"  # Notes, comments, free-form text
    EMPTY_PADDING = "EmptyPadding"  # Intentional whitespace for layout


@dataclass
class DataCharacteristics:
    """Rich metadata about the data in a region."""

    primary_data_type: str | None = None  # "financial", "temporal", "categorical", etc.
    has_headers: bool = False
    header_rows: int = 0
    header_cols: int = 0
    has_formulas: bool = False
    formula_density: float = 0.0
    has_merged_cells: bool = False
    is_time_series: bool = False
    time_granularity: str | None = None  # "daily", "monthly", "yearly"
    aggregation_type: str | None = None  # "sum", "average", "count"
    data_density: float = 0.0
    numeric_ratio: float = 0.0
    has_totals_row: bool = False
    has_totals_col: bool = False
    likely_pivot_table: bool = False
    has_filters: bool = False


@dataclass
class SemanticRegion:
    """
    Complete semantic classification for a spreadsheet region.
    Combines structural type with LLM-generated semantic understanding.
    """

    # Identity
    id: str
    range: str  # A1:F20

    # Two-tier classification
    structural_type: StructuralType
    semantic_description: str  # LLM-generated natural language

    # Confidence and reasoning
    confidence: float
    classification_reasoning: str  # Why this classification was chosen

    # Hierarchy support
    parent_region_id: str | None = None
    child_region_ids: list[str] = field(default_factory=list)
    nesting_level: int = 0  # 0 = top-level, 1+ = nested

    # Rich metadata
    data_characteristics: DataCharacteristics = field(default_factory=DataCharacteristics)

    # Relationships to other regions
    related_regions: dict[str, str] = field(default_factory=dict)
    # e.g., {"data_source": "region_123", "aggregates": "region_456"}

    # Domain-specific attributes (extensible)
    domain_attributes: dict[str, Any] = field(default_factory=dict)
    # e.g., {"account_hierarchy": "revenue/sales/product_a"}

    # Suggestions for downstream processing
    suggested_operations: list[str] = field(default_factory=list)
    # e.g., ["validate_formulas", "extract_time_series", "normalize_headers"]


class ImprovedSheetCartographer:
    """
    Enhanced cartographer with two-tier classification system.
    """

    STRUCTURAL_CLASSIFICATION_PROMPT = """
    Classify this spreadsheet region into ONE of these structural types based on its layout pattern:

    1. DataTable - Regular rows and columns with consistent schema, like a database table
    2. HeaderZone - Column headers, row headers, or hierarchical headers
    3. AggregationZone - Summary rows/columns with totals, averages, or other aggregations
    4. MetadataZone - Titles, descriptions, parameters, or configuration values
    5. VisualizationAnchor - Location of charts, sparklines, or visual elements
    6. FormInputZone - Data entry areas with dropdowns, validation, or input fields
    7. NavigationZone - Links, buttons, or index/menu areas for navigation
    8. UnstructuredText - Free-form text, notes, or comments without tabular structure
    9. EmptyPadding - Intentional whitespace used for visual layout

    Also provide:
    - A natural language description of what this region semantically represents
    - Whether this region appears to contain nested sub-regions
    - Data characteristics (headers, formulas, time series, etc.)
    - Suggested operations for downstream processing
    """

    def classify_with_nested_detection(self, range_text: str, range_str: str) -> SemanticRegion:
        """
        Classify a region with support for nested structure detection.
        """
        # First pass: structural classification and semantic description
        initial_classification = self._llm_classify(range_text, range_str)

        # Check for nested structures
        if self._might_contain_nested_regions(initial_classification):
            # Recursive detection of child regions
            child_regions = self._detect_child_regions(range_text, range_str)

            # Update parent with child relationships
            for child in child_regions:
                child.parent_region_id = initial_classification.id
                child.nesting_level = initial_classification.nesting_level + 1
                initial_classification.child_region_ids.append(child.id)

        return initial_classification

    def _might_contain_nested_regions(self, region: SemanticRegion) -> bool:
        """
        Heuristics to determine if a region might contain nested structures.
        """
        indicators = [
            # Structural indicators
            region.data_characteristics.has_totals_row
            and region.data_characteristics.data_density < 0.8,  # Gaps suggest grouping
            # Semantic indicators
            any(
                keyword in region.semantic_description.lower()
                for keyword in ["grouped", "nested", "hierarchical", "drill-down"]
            ),
            # Size indicators
            region.structural_type == StructuralType.DATA_TABLE
            and self._get_region_size(region.range) > 100,  # Large tables often have structure
            # Pattern indicators
            "subtotal" in region.semantic_description.lower(),
            region.data_characteristics.header_cols > 1,  # Multi-level headers
        ]

        return sum(indicators) >= 2  # Need multiple indicators

    def _detect_child_regions(
        self, parent_text: str, parent_range: str
    ) -> list[SemanticRegion]:
        """
        Detect nested regions within a parent region.
        """
        child_regions = []

        # Pattern 1: Grouped data with subtotals
        if self._has_subtotal_pattern(parent_text):
            groups = self._split_by_subtotals(parent_text, parent_range)
            for group_range, group_text in groups:
                child = self.classify_with_nested_detection(group_text, group_range)
                child_regions.append(child)

        # Pattern 2: Mixed structural types (e.g., header + data + summary)
        elif self._has_mixed_structure(parent_text):
            segments = self._segment_by_structure(parent_text, parent_range)
            for segment_range, segment_text in segments:
                child = self.classify_with_nested_detection(segment_text, segment_range)
                child_regions.append(child)

        # Pattern 3: Hierarchical indentation (parent-child accounts)
        elif self._has_indentation_hierarchy(parent_text):
            levels = self._extract_hierarchy_levels(parent_text, parent_range)
            for level_range, level_text, level_depth in levels:
                child = self.classify_with_nested_detection(level_text, level_range)
                child.domain_attributes["hierarchy_depth"] = level_depth
                child_regions.append(child)

        return child_regions


# Example usage showing the benefits
def demonstrate_classification():
    """
    Show how the new system handles complex real-world cases.
    """

    # Case 1: Financial statement with nested structure
    financial_region = SemanticRegion(
        id="region_001",
        range="A1:F50",
        structural_type=StructuralType.DATA_TABLE,
        semantic_description="Quarterly income statement with revenue breakdown by product line and geographic region",
        confidence=0.92,
        classification_reasoning="Detected account hierarchy with subtotals at multiple levels",
        child_region_ids=["region_002", "region_003", "region_004"],
        data_characteristics=DataCharacteristics(
            primary_data_type="financial",
            has_headers=True,
            header_rows=2,  # Multi-level headers
            has_formulas=True,
            formula_density=0.35,
            has_totals_row=True,
            aggregation_type="sum",
            time_granularity="quarterly",
        ),
        domain_attributes={
            "statement_type": "income_statement",
            "fiscal_period": "Q3_2024",
            "currency": "USD",
            "accounting_standard": "GAAP",
        },
        suggested_operations=[
            "validate_accounting_equations",
            "extract_period_over_period_changes",
            "identify_variance_drivers",
        ],
    )

    # Case 2: Dashboard with multiple KPI zones
    dashboard_region = SemanticRegion(
        id="region_010",
        range="A1:M25",
        structural_type=StructuralType.METADATA_ZONE,
        semantic_description="Executive dashboard with sales KPIs, customer metrics, and operational efficiency indicators",
        confidence=0.88,
        classification_reasoning="Multiple distinct metric zones with different update frequencies",
        child_region_ids=["region_011", "region_012", "region_013"],
        data_characteristics=DataCharacteristics(
            primary_data_type="metrics",
            has_formulas=True,
            formula_density=0.80,  # Heavily calculated
            is_time_series=True,
            time_granularity="daily",
            has_filters=True,
        ),
        related_regions={
            "data_source": "region_020",  # Links to underlying data
            "refreshed_by": "region_021",  # Links to refresh timestamp
        },
        suggested_operations=["monitor_metric_thresholds", "detect_anomalies", "generate_executive_summary"],
    )

    return financial_region, dashboard_region


if __name__ == "__main__":
    # Demonstrate the classification system
    financial, dashboard = demonstrate_classification()

    print(f"Financial Region: {financial.structural_type} - {financial.semantic_description}")
    print(f"  Children: {len(financial.child_region_ids)} nested regions")
    print(f"  Domain: {financial.domain_attributes}")

    print(f"\nDashboard Region: {dashboard.structural_type} - {dashboard.semantic_description}")
    print(f"  Formula density: {dashboard.data_characteristics.formula_density:.0%}")
    print(f"  Related regions: {dashboard.related_regions}")
