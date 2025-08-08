# Cartographer Classification System Update

## Executive Summary

This document outlines the migration from our current rigid classification system to a research-backed two-tier semantic classification system. Based on DECO cell taxonomy and validated by Sheet2Graph/SpreadsheetLLM research showing ~93% coverage of real-world spreadsheet patterns.

## Academic Foundation

### Cell-Level Taxonomy (DECO Dataset)

The research literature establishes 7 fundamental cell roles:

| Cell Role       | Description                  | Example                          |
| --------------- | ---------------------------- | -------------------------------- |
| **Data**        | Core facts/observations      | Sales figures, dates, quantities |
| **Header**      | Column/row labels            | "Product Name", "Q1 2024"        |
| **Derived**     | Formula-calculated values    | SUM(), AVERAGE() results         |
| **GroupHeader** | Higher-level category labels | "2024 Totals", "North Region"    |
| **Title**       | Table/sheet descriptions     | "Annual Sales Report"            |
| **Note**        | Footnotes/comments           | "\* Excludes returns"            |
| **Other**       | Metadata not fitting above   | Version numbers, timestamps      |

Blocks are aggregations of cells with similar roles. This bottom-up approach is more principled than top-down classification.

## Migration Path from Current to Proposed System

### Classification Mapping

| Current      | New Structural Type              | Disambiguation Rule                                             | Semantic Description Examples                               |
| ------------ | -------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| FactTable    | DataTable                        | >70% Data cells                                                 | "Transaction log", "Inventory records", "Customer database" |
| HeaderBanner | HeaderZone OR MetadataZone       | Referenced by formulas → HeaderZone<br>Otherwise → MetadataZone | "Column headers with units", "Report title and date"        |
| KPISummary   | AggregationZone                  | Set `aggregation_zone_subtype="kpi"`                            | "Executive KPIs", "Daily metrics dashboard"                 |
| Summary      | AggregationZone                  | Set `aggregation_zone_subtype="subtotal"` or `"grand_total"`    | "Monthly totals", "Statistical summary"                     |
| PivotCache   | DataTable                        | Set `likely_pivot_table=True`                                   | "Sales pivot by region and product"                         |
| ChartAnchor  | VisualizationAnchor              | Check for chart objects                                         | "Revenue trend chart", "Pie chart of market share"          |
| Other        | UnstructuredText OR EmptyPadding | Has content → UnstructuredText<br>Empty → EmptyPadding          | "Meeting notes", "Intentional spacing"                      |

### Three-Phase Implementation

#### Phase 1: Semantic Enrichment (Non-breaking, Q1 2025)

**Code Changes:**

```python
# Backward compatibility shim
OLD_TO_NEW_MAPPING = {
    "FactTable": StructuralType.DATA_TABLE,
    "HeaderBanner": StructuralType.HEADER_ZONE,
    # ... (see semantic_classification.py)
}

# Extend existing Block
@dataclass
class Block:
    id: str
    range: str
    type: str  # Keep old enum for compatibility
    confidence: float
    # NEW FIELDS:
    semantic_description: str = ""  # Mandatory in Phase 1
    structural_type_preview: Optional[StructuralType] = None  # Shadow field
```

**LLM Prompt Update:**

```
STEP 1: Choose structural type (DataTable, HeaderZone, etc.)
STEP 2: Write ONE sentence semantic description
STEP 3: Return JSON with confidence, subtypes, and suggestions
```

**Validation Checklist:**

- [ ] All existing tests pass with compatibility shim
- [ ] Semantic descriptions generated for 100% of regions
- [ ] Dashboard metrics unchanged (map old→new for reporting)
- [ ] Performance impact \<5% (measure token usage)

#### Phase 2: Data Characteristics (Enhancement, Q2 2025)

**New Capabilities:**

```python
# Add rich metadata
data_characteristics = DataCharacteristics(
    # Core 15 attributes
    primary_data_type="financial",
    has_headers=True,
    header_rows=2,
    formula_density=0.35,
    numeric_ratio=0.75,
    
    # Cell role distribution (from DECO)
    cell_role_distribution={
        "Data": 0.70,
        "Header": 0.15,
        "Derived": 0.15
    },
    
    # Disambiguation helpers
    aggregation_zone_subtype="kpi",  # For AggregationZone
    has_data_validation=True,         # For FormInputZone
    has_hyperlinks=True               # For NavigationZone
)
```

**Nested Region Detection:**

```python
# Enable parent-child relationships
if has_subtotal_pattern or has_indentation_hierarchy:
    child_regions = detect_nested_structures()
    parent.child_region_ids = [c.id for c in child_regions]
```

#### Phase 3: Full Migration (Breaking with deprecation, Q3 2025)

**Deprecation Timeline:**

- Month 1-2: Warn on old enum usage
- Month 3-4: Require explicit `use_legacy=True` flag
- Month 5-6: Remove old enum entirely

**Final State:**

- StructuralType enum FROZEN (no additions)
- All regions have semantic_description
- Nested hierarchies fully supported
- DAG relationships via related_regions

## Critical Disambiguation Rules

### HeaderZone vs MetadataZone

```python
def classify_header_or_metadata(region, has_formula_refs):
    """
    Key insight: Headers are REFERENCED, metadata is DESCRIPTIVE
    """
    if has_formula_refs:  # Used in VLOOKUP, INDEX, or direct cell refs
        return StructuralType.HEADER_ZONE
    else:  # Titles, descriptions, parameters
        return StructuralType.METADATA_ZONE
```

### AggregationZone Subtypes

| Subtype         | Characteristics                      | Example              |
| --------------- | ------------------------------------ | -------------------- |
| `"kpi"`         | ≤5 rows, text-number pairs, isolated | "Revenue: $1.2M"     |
| `"subtotal"`    | Row/column totals within data table  | Department subtotals |
| `"grand_total"` | Final summation row/column           | "GRAND TOTAL" row    |

### Confidence Thresholds

| Structural Type     | Default Confidence | High Confidence Requires           |
| ------------------- | ------------------ | ---------------------------------- |
| DataTable           | 0.70               | Consistent schema, >10 rows        |
| HeaderZone          | 0.65               | Adjacent to DataTable, referenced  |
| AggregationZone     | 0.75               | Formulas present, "total" keywords |
| MetadataZone        | 0.60               | Bold/merged, not referenced        |
| VisualizationAnchor | 0.90               | Chart object detected              |
| FormInputZone       | 0.40               | 0.85 if data validation found      |
| NavigationZone      | 0.40               | 0.85 if HYPERLINK() found          |
| UnstructuredText    | 0.50               | No tabular structure               |
| EmptyPadding        | 0.95               | All cells empty                    |

## Benefits for Downstream Agents

### Data Extraction Agents

**Before:** "Is this a FactTable or Summary?"

**After:**

- Structural type indicates parsing strategy
- Cell role distribution shows data vs headers vs derived
- Semantic description provides domain context
- Suggested operations guide processing order

### Analysis Agents

**Before:** Re-analyze every region to understand content

**After:**

- `primary_data_type` selects appropriate statistics
- `time_granularity` informs time-series analysis
- `aggregation_zone_subtype` triggers KPI monitoring
- Parent-child relationships enable drill-down

### Validation Agents

**Before:** Generic validation across all regions

**After:**

```python
if region.structural_type == StructuralType.AGGREGATION_ZONE:
    if region.data_characteristics.aggregation_zone_subtype == "grand_total":
        # Verify this sums all subtotals
        validate_grand_total(region, region.related_regions["aggregates"])
    elif "accounting_standard" in region.domain_attributes:
        # Apply GAAP/IFRS rules
        validate_accounting_rules(region)
```

## Handling Complex Cases

### Nested Financial Statements

```python
income_statement = SemanticRegion(
    structural_type=StructuralType.DATA_TABLE,
    semantic_description="Quarterly P&L with nested account hierarchy",
    child_region_ids=["revenue_detail", "expense_detail", "totals"],
    data_characteristics=DataCharacteristics(
        cell_role_distribution={
            "GroupHeader": 0.20,  # Account categories
            "Data": 0.60,         # Amounts
            "Derived": 0.20       # Subtotals
        }
    )
)
```

### Mixed KPI Dashboard

```python
dashboard = SemanticRegion(
    structural_type=StructuralType.METADATA_ZONE,  # Container
    semantic_description="Operations dashboard with 6 KPI boxes",
    child_region_ids=["kpi_1", "kpi_2", ..., "kpi_6"],
    # Each child has structural_type=AGGREGATION_ZONE, subtype="kpi"
)
```

### Graph Relationships (Not Just Tree)

```python
# Multiple aggregation relationships
total_revenue = SemanticRegion(
    structural_type=StructuralType.AGGREGATION_ZONE,
    related_regions={
        "aggregates": "product_sales,service_revenue,other_income",  # Multiple parents
        "compared_to": "last_year_total",  # Cross-reference
        "feeds_into": "net_income"  # Downstream dependency
    }
)
```

## Validation Strategy

### Test Coverage Matrix

| Spreadsheet Type       | Key Patterns to Validate     | Expected Structural Types                                 |
| ---------------------- | ---------------------------- | --------------------------------------------------------- |
| Financial Statements   | Nested accounts, subtotals   | DataTable + GroupHeaders + AggregationZone                |
| Scientific Data        | Units in headers, statistics | HeaderZone + DataTable + AggregationZone(kpi)             |
| Operational Dashboards | Multiple KPIs, charts        | MetadataZone + AggregationZone(kpi) + VisualizationAnchor |
| Survey Results         | Categories, response counts  | DataTable + AggregationZone(subtotal)                     |
| Project Plans          | Task hierarchy, dates        | DataTable with nested GroupHeaders                        |
| Data Entry Forms       | Validation, dropdowns        | FormInputZone + NavigationZone                            |

### Success Metrics

| Metric                       | Target                              | Measurement                        |
| ---------------------------- | ----------------------------------- | ---------------------------------- |
| Classification Coverage      | >93% regions classified confidently | Count regions with confidence >0.7 |
| Semantic Description Quality | 100% non-empty, \<5% generic        | Manual review sample of 100        |
| Backward Compatibility       | Zero breaks in Phase 1              | All existing tests pass            |
| Token Efficiency             | \<10% increase                      | Compare tokens before/after        |
| Downstream Success           | >20% reduction in re-analysis       | Track extraction agent perf        |

## Implementation Checklist

### Phase 1 Sprint (2 weeks)

- [ ] Add semantic_description to Block dataclass
- [ ] Implement OLD_TO_NEW_MAPPING
- [ ] Update LLM prompt with 3-step structure
- [ ] Add semantic_classification.py to models/
- [ ] Create test suite with 20 diverse spreadsheets
- [ ] Update logging to capture both old and new classifications
- [ ] Documentation for downstream teams

### Phase 2 Sprint (3 weeks)

- [ ] Implement DataCharacteristics calculation
- [ ] Add cell role distribution analysis
- [ ] Enable nested region detection
- [ ] Add aggregation_zone_subtype logic
- [ ] Implement confidence rules table
- [ ] Performance benchmarking

### Phase 3 Sprint (4 weeks)

- [ ] Full StructuralType migration
- [ ] Deprecation warnings
- [ ] DAG relationship support
- [ ] Update all downstream agents
- [ ] Final documentation
- [ ] Freeze enum specification

## FAQ

**Q: Why not let the LLM define its own categories?**\
A: Research shows 93% of regions fit these 9 types. Consistency helps downstream agents. Edge cases handled via semantic_description.

**Q: What about regions that span multiple types?**\
A: Use parent-child relationships. Parent gets most general type, children get specific types.

**Q: How do we handle EmptyPadding in output?**\
A: Set `include_in_output=False` unless `verbose=True` to avoid bloat.

**Q: Can we add new structural types later?**\
A: No. After Phase 3, enum is FROZEN. Use domain_attributes or subtypes for extensions.

**Q: What if classification confidence is low?**\
A: Regions with confidence \<0.5 should trigger deeper splitting or human review.

## References

- DECO Dataset: Cell-level taxonomy for spreadsheet understanding
- TableSense (Microsoft Research): CNN-based table detection achieving 91.3% recall
- SpreadsheetLLM: 25.6% improvement over vanilla GPT-4 with structured encoding
- Sheet2Graph: ~93% of spreadsheet blocks covered by these structural types
