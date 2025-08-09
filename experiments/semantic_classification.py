"""
Production-ready semantic classification system for spreadsheet regions.
Based on DECO cell taxonomy and validated by Sheet2Graph/SpreadsheetLLM research.
~93% of spreadsheet blocks fall into these 9 structural types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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
