# Cartographer Classification System Update

## Migration Path from Current to Proposed System

### Current Classifications → New Structural Types

| Current      | New Structural Type              | Semantic Description Examples                            |
| ------------ | -------------------------------- | -------------------------------------------------------- |
| FactTable    | DataTable                        | "Transaction records", "Customer list", "Inventory data" |
| HeaderBanner | HeaderZone or MetadataZone       | "Report title", "Column headers with units"              |
| KPISummary   | AggregationZone or MetadataZone  | "Monthly performance metrics", "YTD totals"              |
| Summary      | AggregationZone                  | "Revenue totals by region", "Statistical summary"        |
| PivotCache   | DataTable (with pivot flag)      | "Pivot table of sales by quarter and product"            |
| ChartAnchor  | VisualizationAnchor              | "Revenue trend chart", "Performance dashboard"           |
| Other        | UnstructuredText or EmptyPadding | "Meeting notes", "Layout spacing"                        |

### Implementation Steps

1. **Phase 1: Add Semantic Description** (Non-breaking)

   - Keep existing enum for backward compatibility
   - Add `semantic_description` field to Block dataclass
   - Update LLM prompt to generate descriptions

1. **Phase 2: Introduce Data Characteristics** (Enhancement)

   - Add DataCharacteristics calculation
   - Enrich blocks with metadata
   - Enable nested region detection

1. **Phase 3: Full Migration** (Breaking change with deprecation)

   - Replace enum with StructuralType
   - Implement parent-child relationships
   - Add domain attributes

### Benefits for Downstream Agents

1. **Data Extraction Agents**

   - Know exactly what type of data structure to expect
   - Can apply appropriate parsing strategies
   - Understand relationships between regions

1. **Analysis Agents**

   - Semantic descriptions provide business context
   - Data characteristics inform statistical approaches
   - Hierarchies enable drill-down analysis

1. **Validation Agents**

   - Can apply domain-specific rules (e.g., accounting equations)
   - Understand which regions should sum to totals
   - Detect inconsistencies in nested structures

### Example: How Classification Helps Downstream

```python
# Current approach - limited context
block = Block(
    id="blk_01",
    range="A1:F10", 
    type="FactTable",
    confidence=0.8
)
# Downstream agent: "Is this financial data? Time series? What are the columns?"

# Proposed approach - rich context
region = SemanticRegion(
    id="region_01",
    range="A1:F10",
    structural_type=StructuralType.DATA_TABLE,
    semantic_description="Daily sales transactions for Q3 2024 with product SKU, quantity, unit price, and customer details",
    data_characteristics=DataCharacteristics(
        primary_data_type="transactional",
        is_time_series=True,
        time_granularity="daily",
        has_headers=True,
        header_rows=1
    ),
    suggested_operations=[
        "validate_price_calculations",
        "aggregate_by_customer",
        "detect_seasonal_patterns"
    ]
)
# Downstream agent: Has everything needed to process intelligently
```

### Handling Edge Cases

1. **Ambiguous Regions**

   - Use confidence scores to indicate uncertainty
   - Provide multiple possible interpretations in `suggested_operations`
   - Let downstream agents make final determination

1. **Complex Nested Structures**

   - Build complete hierarchy tree
   - Mark regions that span multiple parents
   - Provide traversal helpers for downstream agents

1. **Dynamic/Formula-Heavy Regions**

   - Flag high formula density
   - Indicate if region depends on others
   - Suggest order of processing

### Validation Approach

Test on diverse spreadsheet types:

- Financial statements (nested accounts)
- Scientific data (measurements with units)
- Operational dashboards (mixed KPIs)
- Survey results (categorical data)
- Project plans (hierarchical tasks)
