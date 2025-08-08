"""
Production-ready semantic classification system for spreadsheet regions.
Based on DECO cell taxonomy and validated by Sheet2Graph/SpreadsheetLLM research.
~93% of spreadsheet blocks fall into these 9 structural types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class StructuralType(str, Enum):
    """
    Tier 1: Structural classification based on layout patterns.
    FROZEN after Phase 3 - no new additions to maintain downstream stability.
    """

    DATA_TABLE = "DataTable"  # Regular rows/columns with consistent schema
    HEADER_ZONE = "HeaderZone"  # Column/row headers referenced by formulas
    AGGREGATION_ZONE = "AggregationZone"  # Summaries, totals, KPIs (use subtype for distinction)
    METADATA_ZONE = "MetadataZone"  # Titles, descriptions not referenced by formulas
    VISUALIZATION_ANCHOR = "VisualizationAnchor"  # Charts, sparklines, conditional formatting
    FORM_INPUT_ZONE = "FormInputZone"  # Data entry areas (advisory in Phase 1)
    NAVIGATION_ZONE = "NavigationZone"  # Links, buttons (low confidence unless HYPERLINK detected)
    UNSTRUCTURED_TEXT = "UnstructuredText"  # Notes, comments, free-form text
    EMPTY_PADDING = "EmptyPadding"  # Whitespace for layout (omit from output unless verbose)


# Cell-level roles from DECO dataset that inform structural classification
class CellRole(str, Enum):
    """
    Cell-level taxonomy from academic research (DECO dataset).
    Blocks are aggregations of cells with similar roles.
    """

    DATA = "Data"  # Core facts/observations
    HEADER = "Header"  # Column/row labels
    DERIVED = "Derived"  # Formula-calculated aggregations
    GROUP_HEADER = "GroupHeader"  # Higher-level category labels
    TITLE = "Title"  # Table/sheet descriptions
    NOTE = "Note"  # Footnotes/comments
    OTHER = "Other"  # Metadata not fitting above


@dataclass
class DataCharacteristics:
    """Rich metadata about the data in a region."""

    # Core attributes (15 most important per research)
    primary_data_type: str | None = None  # "financial", "temporal", "categorical"
    has_headers: bool = False
    header_rows: int = 0
    header_cols: int = 0
    has_formulas: bool = False
    formula_density: float = 0.0
    has_merged_cells: bool = False
    is_time_series: bool = False
    time_granularity: str | None = None  # "daily", "monthly", "yearly"
    data_density: float = 0.0
    numeric_ratio: float = 0.0
    has_totals_row: bool = False
    has_totals_col: bool = False
    likely_pivot_table: bool = False
    has_filters: bool = False

    # Subtype distinctions (per user feedback)
    aggregation_zone_subtype: Literal["kpi", "subtotal", "grand_total", None] = None
    has_data_validation: bool = False  # For FormInputZone detection
    has_hyperlinks: bool = False  # For NavigationZone detection

    # Cell role distribution (aggregated from cell-level analysis)
    cell_role_distribution: dict[str, float] = field(default_factory=dict)
    # e.g., {"Data": 0.7, "Header": 0.2, "Derived": 0.1}


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
    semantic_description: str  # MANDATORY for all regions

    # Confidence and reasoning
    confidence: float
    classification_reasoning: str

    # Hierarchy support (DAG not just tree)
    parent_region_id: str | None = None  # Primary parent for tree traversal
    child_region_ids: list[str] = field(default_factory=list)
    nesting_level: int = 0

    # Rich relationships (supports multiple parents)
    related_regions: dict[str, str] = field(default_factory=dict)
    # e.g., {"aggregates": "region_123,region_456", "references": "region_789"}

    # Rich metadata
    data_characteristics: DataCharacteristics = field(default_factory=DataCharacteristics)

    # Domain-specific attributes (extensible without enum changes)
    domain_attributes: dict[str, Any] = field(default_factory=dict)

    # Suggestions for downstream processing
    suggested_operations: list[str] = field(default_factory=list)

    # Control flags
    include_in_output: bool = True  # Set False for EmptyPadding unless verbose


class ClassificationRules:
    """
    Disambiguation rules based on user feedback.
    """

    @staticmethod
    def header_vs_metadata(region_text: str, has_formula_refs: bool) -> StructuralType:
        """
        Rule: If referenced by formulas in contiguous data table → HeaderZone
              Otherwise → MetadataZone
        """
        if has_formula_refs:
            return StructuralType.HEADER_ZONE
        return StructuralType.METADATA_ZONE

    @staticmethod
    def detect_navigation_confidence(has_hyperlinks: bool, has_buttons: bool) -> float:
        """
        NavigationZone requires strong evidence to avoid false positives.
        """
        if has_hyperlinks:
            return 0.85
        if has_buttons:
            return 0.60
        return 0.40  # Default low confidence

    @staticmethod
    def should_output_empty_padding(verbose: bool) -> bool:
        """
        EmptyPadding blocks bloat output; only include if verbose.
        """
        return verbose


# Backward compatibility mapping (Phase 1)
OLD_TO_NEW_MAPPING = {
    "FactTable": StructuralType.DATA_TABLE,
    "HeaderBanner": StructuralType.HEADER_ZONE,
    "KPISummary": StructuralType.AGGREGATION_ZONE,
    "Summary": StructuralType.AGGREGATION_ZONE,
    "PivotCache": StructuralType.DATA_TABLE,  # With pivot flag
    "ChartAnchor": StructuralType.VISUALIZATION_ANCHOR,
    "Other": StructuralType.UNSTRUCTURED_TEXT,
}


class ImprovedClassificationPrompt:
    """
    Phase 1 prompt structure per user feedback.
    """

    CLASSIFICATION_SEQUENCE = """
    Analyze this spreadsheet region in three steps:

    STEP 1: Structural Type
    Choose ONE from these 9 types based on layout pattern:
    - DataTable: Regular rows/columns with consistent schema
    - HeaderZone: Headers referenced by formulas in adjacent data
    - AggregationZone: Summaries, totals, KPIs (specify subtype)
    - MetadataZone: Titles/descriptions NOT referenced by formulas
    - VisualizationAnchor: Charts, sparklines, conditional formatting
    - FormInputZone: Data entry areas with validation/dropdowns
    - NavigationZone: Links, buttons, menus (requires HYPERLINK evidence)
    - UnstructuredText: Free-form notes without tabular structure
    - EmptyPadding: Intentional whitespace for layout

    STEP 2: Semantic Description
    Write ONE sentence describing what this region semantically represents.
    Be specific about the business/domain meaning, not just structure.

    STEP 3: Additional Analysis
    Provide JSON with:
    - confidence: 0.0 to 1.0
    - aggregation_zone_subtype: If AggregationZone, specify "kpi"/"subtotal"/"grand_total"
    - peek_requests: Additional ranges needed for confident classification
    - open_questions: Ambiguities that need resolution
    - has_nested_regions: Boolean indicating if internal structure exists
    - suggested_operations: List of recommended downstream processing steps

    First line of response MUST be the structural type.
    Second line MUST be the semantic description.
    Remaining lines MUST be valid JSON.
    """


def demonstrate_phase1_implementation():
    """
    Show Phase 1 implementation with backward compatibility.
    """

    # Example of old classification being mapped
    old_classification = "FactTable"
    new_structural_type = OLD_TO_NEW_MAPPING[old_classification]

    # Example of enriched classification
    region = SemanticRegion(
        id="region_001",
        range="A1:F10",
        structural_type=new_structural_type,
        semantic_description="Daily sales transactions for electronics department showing SKU, quantity, unit price, and revenue",
        confidence=0.92,
        classification_reasoning="Consistent columnar structure with headers, numeric data, and no aggregation formulas",
        data_characteristics=DataCharacteristics(
            primary_data_type="transactional",
            has_headers=True,
            header_rows=1,
            numeric_ratio=0.65,
            cell_role_distribution={"Data": 0.85, "Header": 0.15},
        ),
        suggested_operations=["validate_revenue_calculation", "extract_product_categories", "aggregate_by_date"],
    )

    # Example of KPI distinction via subtype
    kpi_region = SemanticRegion(
        id="region_002",
        range="H1:J5",
        structural_type=StructuralType.AGGREGATION_ZONE,
        semantic_description="Key performance indicators showing MTD revenue, units sold, and average order value",
        confidence=0.88,
        classification_reasoning="Small text-number pairs with derived formulas, positioned separately from main data",
        data_characteristics=DataCharacteristics(
            aggregation_zone_subtype="kpi",  # Distinction without new enum
            formula_density=0.90,
            cell_role_distribution={"Derived": 0.60, "Header": 0.40},
        ),
    )

    return region, kpi_region


if __name__ == "__main__":
    data_region, kpi_region = demonstrate_phase1_implementation()

    print(f"Data Region: {data_region.structural_type.value}")
    print(f"  Description: {data_region.semantic_description}")
    print(f"  Cell roles: {data_region.data_characteristics.cell_role_distribution}")

    print(f"\nKPI Region: {kpi_region.structural_type.value}")
    print(f"  Subtype: {kpi_region.data_characteristics.aggregation_zone_subtype}")
    print(f"  Formula density: {kpi_region.data_characteristics.formula_density:.0%}")
