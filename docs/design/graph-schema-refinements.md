# Graph Schema Refinements

## Enhanced Node and Edge Attributes

Based on the detailed analysis, here are the key refinements to the graph schema:

### Node Schema Enhancements

#### Cell Nodes

```cypher
(:Cell {
  // Core attributes
  key: "Sheet1!A1",          // Unique identifier
  sheet: "Sheet1",           // Sheet name
  ref: "A1",                 // Cell reference
  formula: "=SUM(B1:B10)",   // Formula string (if present)
  depth: 2,                  // Dependency depth
  
  // Enhanced attributes
  value_type: "number",      // "number" | "text" | "date" | "error" | "boolean"
  is_volatile: false,        // Contains NOW(), RAND(), etc.
  pagerank: 0.0023,         // Pre-computed importance
  
  // Additional metadata
  has_error: false,          // Formula evaluation error
  named_range: null,         // If part of a named range
  last_modified: timestamp   // For change tracking
})
```

#### Range Nodes

```cypher
(:Range {
  // Core attributes
  key: "Sheet1!B41:B47",
  sheet: "Sheet1",
  ref: "B41:B47",
  depth: 0,
  
  // Range-specific attributes
  range_size: 7,
  start_cell: "B41",
  end_cell: "B47",
  range_type: "column",      // "column" | "row" | "block" | "full_column" | "full_row"
  
  // Summary statistics (for numeric ranges)
  stats: {
    min: 10.5,
    max: 150.0,
    sum: 525.0,
    avg: 75.0,
    count: 7,
    non_null_count: 6
  },
  
  // Pattern detection
  pattern_type: "sequential",  // "sequential" | "constant" | "formula_pattern" | "mixed"
  pattern_confidence: 0.95
})
```

### Edge Schema Enhancements

```cypher
// Enhanced dependency edges
(:Cell)-[dep:DEPENDS_ON {
  // Core attributes
  ref_type: "range",         // "single_cell" | "range" | "named_range" | "external"
  is_cross_sheet: false,
  
  // Semantic labels for specialized relationships
  semantic_type: "SUMS_OVER",  // "SUMS_OVER" | "LOOKS_UP" | "MATCHES" | "CALCULATES"
  
  // Weighting for algorithms
  weight: 7.0,               // Range size or 1.0 for single cells
  
  // Context from source formula
  formula_context: "SUM({ref})",  // Template showing usage
  function_name: "SUM",           // Primary function using this ref
  
  // Position in formula (for complex formulas)
  argument_position: 1,      // Which argument of the function
  is_nested: false          // Used within another function
}]->(:Range)
```

### Specialized Edge Types

Instead of just `DEPENDS_ON`, we can use semantic edge labels:

1. **SUMS_OVER**: For SUM, SUMIF, SUMIFS functions
1. **AVERAGES_OVER**: For AVERAGE, AVERAGEIF functions
1. **COUNTS_IN**: For COUNT, COUNTIF functions
1. **LOOKS_UP_IN**: For VLOOKUP, HLOOKUP, INDEX/MATCH
1. **VALIDATES_AGAINST**: For data validation rules
1. **FORMATS_BASED_ON**: For conditional formatting

### Cross-Sheet Dependency Tracking

For edges that span sheets, additional properties:

```cypher
(:Cell {sheet: "Summary"})-[:DEPENDS_ON {
  is_cross_sheet: true,
  source_sheet: "Summary",
  target_sheet: "Data",
  workbook_external: false,  // true if references external workbook
  external_path: null,       // Path if external
  link_status: "valid"       // "valid" | "broken" | "circular"
}]->(:Cell {sheet: "Data"})
```

### Implementation Updates Needed

1. **Update Stage 3 Analysis** to capture:

   - Value types during parsing
   - Volatile function detection
   - Semantic edge labeling based on function context

1. **Enhance Graph Loader** to:

   - Calculate range statistics during loading
   - Detect patterns in ranges
   - Create specialized edge types

1. **Add Post-Processing** for:

   - PageRank computation
   - Pattern detection in ranges
   - Cross-sheet dependency analysis

### Benefits of These Refinements

1. **Richer Queries**: Can query by semantic relationship (e.g., "Find all cells that look up values in Sheet2")
1. **Better Analysis**: Summary stats on ranges enable quick insights without expanding nodes
1. **Performance**: Weighted edges improve PageRank and path algorithms
1. **Debugging**: Formula context helps understand dependencies without loading full formulas
1. **Cross-Sheet Awareness**: Explicitly tracks and can filter/highlight cross-sheet dependencies

These refinements maintain the scalability of the range node approach while providing the detailed information needed for sophisticated analysis and agent queries.
