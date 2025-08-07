"""Data models for semantic layout comprehension in spreadsheets."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ZoneType(str, Enum):
    """Types of semantic zones in spreadsheets."""

    HEADER = "header"
    DATA = "data"
    FORMULA = "formula"
    SUMMARY = "summary"
    METADATA = "metadata"
    NAVIGATION = "navigation"
    ANNOTATION = "annotation"
    OTHER = "other"  # Catch-all for ambiguous regions


class RegionCoordinates(BaseModel):
    """Coordinates defining a region in the spreadsheet."""

    start_row: int = Field(ge=0, description="Starting row index (0-based)")
    end_row: int = Field(ge=0, description="Ending row index (inclusive)")
    start_col: int = Field(ge=0, description="Starting column index (0-based)")
    end_col: int = Field(ge=0, description="Ending column index (inclusive)")

    def __str__(self) -> str:
        """String representation of coordinates."""
        return f"[{self.start_row}:{self.end_row}, {self.start_col}:{self.end_col}]"

    @property
    def row_count(self) -> int:
        """Number of rows in the region."""
        return self.end_row - self.start_row + 1

    @property
    def col_count(self) -> int:
        """Number of columns in the region."""
        return self.end_col - self.start_col + 1

    @property
    def cell_count(self) -> int:
        """Total number of cells in the region."""
        return self.row_count * self.col_count


class LayoutRegion(BaseModel):
    """Represents a semantic region in a spreadsheet layout."""

    region_id: str = Field(description="Unique identifier for this region")
    zone_type: ZoneType = Field(description="Type of semantic zone")
    semantic_role: str = Field(description="Semantic role/purpose (e.g., 'transaction_records', 'monthly_totals')")
    coordinates: RegionCoordinates = Field(description="Physical location in spreadsheet")

    # Relationships and navigation
    relationships: list[str] = Field(
        default_factory=list, description="IDs of related regions (e.g., header linked to its data)"
    )
    navigation_hints: list[str] = Field(
        default_factory=list, description="Guidance for analyzer on how to process this region"
    )

    # Confidence and metadata
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for this classification")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional region-specific information")

    # Content characteristics
    description: str | None = Field(None, description="Human-readable description of the region's content")
    sample_values: list[str] | None = Field(None, description="Sample values from the region for context")

    def __str__(self) -> str:
        """String representation of the region."""
        return f"Region '{self.region_id}' ({self.zone_type.value}): {self.semantic_role} at {self.coordinates}"


class NavigationGuide(BaseModel):
    """Provides navigation guidance for analyzing the spreadsheet."""

    suggested_flow: list[str] = Field(description="Recommended order of region IDs for analysis")
    key_insights: list[str] = Field(default_factory=list, description="Important observations about the layout")
    analysis_recommendations: list[str] = Field(
        default_factory=list, description="Specific recommendations for analysis approach"
    )
    warnings: list[str] = Field(default_factory=list, description="Potential issues or complexities to be aware of")


class SpreadsheetLayout(BaseModel):
    """Complete semantic layout comprehension of a spreadsheet."""

    layout_type: str = Field(description="Overall layout pattern (e.g., 'financial_report', 'inventory_list')")
    regions: list[LayoutRegion] = Field(description="All identified semantic regions")
    navigation_guide: NavigationGuide = Field(description="Navigation guidance for analysis")

    # Sheet metadata
    total_rows: int = Field(ge=0, description="Total rows in sheet")
    total_cols: int = Field(ge=0, description="Total columns in sheet")

    # Layout characteristics
    complexity_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Estimated complexity of the layout (0=simple, 1=very complex)"
    )
    has_multiple_tables: bool = Field(
        default=False, description="Whether the sheet contains multiple distinct data regions"
    )
    has_formulas: bool = Field(default=False, description="Whether formula zones were detected")
    has_summaries: bool = Field(default=False, description="Whether summary zones were detected")

    def get_regions_by_type(self, zone_type: ZoneType) -> list[LayoutRegion]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.zone_type == zone_type]

    def get_data_regions(self) -> list[LayoutRegion]:
        """Get all data regions."""
        return self.get_regions_by_type(ZoneType.DATA)

    def get_header_regions(self) -> list[LayoutRegion]:
        """Get all header regions."""
        return self.get_regions_by_type(ZoneType.HEADER)

    def get_related_regions(self, region_id: str) -> list[LayoutRegion]:
        """Get all regions related to a specific region."""
        region = next((r for r in self.regions if r.region_id == region_id), None)
        if not region:
            return []

        related = []
        for rel_id in region.relationships:
            rel_region = next((r for r in self.regions if r.region_id == rel_id), None)
            if rel_region:
                related.append(rel_region)
        return related

    def to_detection_summary(self) -> str:
        """Generate a human-readable summary of the layout."""
        lines = [
            f"Layout Type: {self.layout_type}",
            f"Sheet Dimensions: {self.total_rows} rows × {self.total_cols} columns",
            f"Regions Identified: {len(self.regions)}",
            "",
        ]

        # Group regions by type
        by_type = {}
        for region in self.regions:
            if region.zone_type not in by_type:
                by_type[region.zone_type] = []
            by_type[region.zone_type].append(region)

        # Summarize each type
        for zone_type, regions in by_type.items():
            lines.append(f"{zone_type.value.title()} Zones ({len(regions)}):")
            for r in regions:
                conf_str = f" (confidence: {r.confidence:.2f})" if r.confidence < 1.0 else ""
                lines.append(f"  - {r.semantic_role} at {r.coordinates}{conf_str}")

        # Add navigation guide
        if self.navigation_guide.key_insights:
            lines.append("\nKey Insights:")
            for insight in self.navigation_guide.key_insights:
                lines.append(f"  - {insight}")

        if self.navigation_guide.warnings:
            lines.append("\nWarnings:")
            for warning in self.navigation_guide.warnings:
                lines.append(f"  ⚠️ {warning}")

        return "\n".join(lines)


# Zone type definitions for the detector agent
ZONE_TYPE_DEFINITIONS = """
## Zone Type Definitions

### 1. HEADER Zone
**Definition**: Row(s) containing column labels that describe the data below them.
**Examples**:
- ["Date", "Description", "Amount", "Balance"]
- ["Product ID", "Product Name", "Quantity", "Unit Price"]
- Multi-level headers with year/month/day structure
**Characteristics**: Usually in first few rows, text-heavy, no calculations, consistent labels

### 2. DATA Zone
**Definition**: The main body of records/transactions containing the actual data values.
**Examples**:
- Transaction rows with dates and amounts
- Product inventory listings with quantities
- Customer records with contact information
**Characteristics**: Consistent structure, multiple rows, follows header pattern, contains actual values

### 3. FORMULA Zone
**Definition**: Cells containing calculations, references, or derived values using Excel formulas.
**Examples**:
- =SUM(B2:B100)
- =VLOOKUP(A2, Sheet2!A:C, 3, FALSE)
- Percentage calculations, running totals, complex models
**Characteristics**: Contains formulas (not just values), often references other cells, dynamic values

### 4. SUMMARY Zone
**Definition**: Aggregated metrics, totals, or statistical summaries of data zones.
**Examples**:
- "Total Revenue: $45,000"
- Row with SUM/AVERAGE/COUNT results
- Key performance indicators section
**Characteristics**: Usually at bottom or separated from main data, contains aggregations, often has labels like "Total", "Sum", "Average"

### 5. METADATA Zone
**Definition**: Information about the spreadsheet itself or contextual information.
**Examples**:
- "Report Generated: 2024-01-15"
- "Department: Sales"
- "Fiscal Year 2024 Q1"
**Characteristics**: Descriptive text, often at top or corners, not part of data structure, provides context

### 6. NAVIGATION Zone
**Definition**: Empty or separator regions that visually organize the sheet.
**Examples**:
- Empty row between different tables
- Column of dashes "--------"
- Blank columns separating sections
**Characteristics**: Mostly empty cells, used for visual separation, no meaningful data

### 7. ANNOTATION Zone
**Definition**: Explanatory text, notes, or comments about the data.
**Examples**:
- "Note: Excludes returns"
- "*Pending approval"
- Footnotes or legends
**Characteristics**: Explanatory text, often uses special characters (*,†,‡), provides additional context

### 8. OTHER Zone (Catch-all)
**Definition**: Regions that don't clearly fit into above categories or have ambiguous purpose.
**Examples**:
- Mixed content areas
- Unusual formatting blocks
- Graphics or special objects
- Embedded mini-reports
**Use When**: The region doesn't match patterns of other zones, purpose is unclear, or contains mixed content types
"""


@dataclass
class ZoneDetectionExample:
    """Example for helping the agent understand zone types."""

    zone_type: ZoneType
    description: str
    sample_content: list[str]
    identifying_features: list[str]


# Example library for agent training
ZONE_EXAMPLES = [
    ZoneDetectionExample(
        zone_type=ZoneType.HEADER,
        description="Product inventory headers",
        sample_content=["Product Code", "Description", "Qty on Hand", "Unit Cost", "Total Value"],
        identifying_features=["First row", "Text labels", "No numbers", "Describes columns below"],
    ),
    ZoneDetectionExample(
        zone_type=ZoneType.DATA,
        description="Transaction records",
        sample_content=[
            "2024-01-15 | Office Supplies | -$234.50 | $1,456.78",
            "2024-01-16 | Client Payment | $5,000.00 | $6,456.78",
        ],
        identifying_features=["Multiple similar rows", "Consistent format", "Actual data values"],
    ),
    ZoneDetectionExample(
        zone_type=ZoneType.SUMMARY,
        description="Financial totals section",
        sample_content=["Total Revenue: $45,678.90", "Total Expenses: -$23,456.78", "Net Profit: $22,222.12"],
        identifying_features=["Keywords: Total, Sum, Average", "Aggregated values", "Usually at bottom"],
    ),
]
