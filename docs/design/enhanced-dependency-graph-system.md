# Enhanced Dependency Graph System Design

## Document Purpose

This document outlines the design for enhancing the spreadsheet analyzer's dependency graph system to:

1. Handle range references in formulas (addressing current limitations)
1. Integrate with graph databases for scalable querying
1. Provide an agent-friendly query interface
1. Support Pandas-to-Excel coordinate mapping

## Table of Contents

1. [Current Limitations](#current-limitations)
1. [Design Philosophy](#design-philosophy)
1. [Range Reference Handling](#range-reference-handling)
1. [Graph Database Integration](#graph-database-integration)
1. [Agent Query Interface](#agent-query-interface)
1. [Pandas-Excel Coordinate Mapping](#pandas-excel-coordinate-mapping)
1. [Implementation Plan](#implementation-plan)
1. [Performance Considerations](#performance-considerations)

## Current Limitations

The current Stage 3 implementation has a critical limitation:

- **Range references are ignored**: Formulas like `=SUM(B1:B100)` create nodes but no edges
- **Result**: Empty dependency graphs for spreadsheets using aggregation functions
- **Impact**: Incomplete analysis, missed circular references, poor test coverage

This affects ~80% of business spreadsheets that rely heavily on range-based formulas.

## Design Philosophy

### Core Principles

- **Completeness over Simplicity**: Capture all dependencies, including ranges
- **Scalability through Abstraction**: Use range nodes to prevent edge explosion
- **Query Efficiency**: Leverage graph databases for complex traversals
- **Agent Usability**: Provide intuitive interfaces that handle coordinate mapping

### Quality Attributes

| Attribute    | Requirement                           | Measurement         |
| ------------ | ------------------------------------- | ------------------- |
| Completeness | Handle 100% of Excel formula patterns | Dependency coverage |
| Performance  | < 10s for 10K formulas with ranges    | Processing time     |
| Scalability  | Support graphs with 1M+ nodes         | Graph size limit    |
| Query Speed  | < 100ms for typical queries           | Response time       |
| Usability    | Agent-friendly query interface        | User feedback       |

## Range Reference Handling

### Approach: Hybrid Node Strategy

Instead of creating edges to every cell in a range, we introduce **Range Nodes** as first-class citizens in the graph.

#### Node Types

```yaml
CellNode:
  type: "cell"
  properties:
    sheet: string
    ref: string (e.g., "A1")
    formula: string (optional)
    value: any (optional)
    depth: integer

RangeNode:
  type: "range"
  properties:
    sheet: string
    ref: string (e.g., "B1:B100")
    start_cell: string
    end_cell: string
    size: integer
    is_column_range: boolean
    is_row_range: boolean
```

#### Edge Types

```yaml
DEPENDS_ON:
  from: CellNode (formula)
  to: CellNode | RangeNode
  properties:
    dependency_type: "direct" | "range"

CONTAINS:
  from: RangeNode
  to: CellNode
  properties:
    position: integer (optional)
```

### Implementation Strategy

1. **Small Ranges (< 10 cells)**: Expand to individual DEPENDS_ON edges
1. **Medium Ranges (10-100 cells)**: Create RangeNode with selective CONTAINS edges
1. **Large Ranges (> 100 cells)**: Create RangeNode with metadata only

### Configuration Options

```python
range_handling_config = {
    "mode": "smart",  # "skip" | "expand" | "summarize" | "smart"
    "small_range_threshold": 10,
    "medium_range_threshold": 100,
    "expand_column_ranges": False,  # A:A ranges
    "expand_row_ranges": False,     # 1:1 ranges
}
```

## Graph Database Integration

### Technology Choice: Neo4j

**Rationale**:

- Native graph storage and processing
- Built-in graph algorithms (PageRank, centrality, paths)
- Cypher query language familiar to LLMs
- Scales to millions of nodes
- ACID compliance for data integrity

### Schema Design

```cypher
// Node types
(:Cell {
  key: "Sheet1!A1",
  sheet: "Sheet1",
  ref: "A1",
  formula: "=SUM(B1:B10)",
  depth: 2,
  pagerank: 0.85
})

(:Range {
  key: "Sheet1!B1:B10",
  sheet: "Sheet1",
  ref: "B1:B10",
  start_ref: "B1",
  end_ref: "B10",
  size: 10,
  type: "column"
})

// Relationships
(cell:Cell)-[:DEPENDS_ON]->(range:Range)
(range:Range)-[:CONTAINS]->(cell:Cell)
(cell1:Cell)-[:DEPENDS_ON]->(cell2:Cell)
```

### Pre-computation Strategy

1. **Graph Loading**: Post-pipeline batch import
1. **PageRank**: Pre-compute and store as node property
1. **Aggregates**: Cache key metrics (node count, edge count, max depth)
1. **Indexes**: Create on `key`, `sheet`, `ref` for fast lookups

### Loader Implementation

```python
class GraphDatabaseLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def load_pipeline_result(self, result: PipelineResult) -> dict:
        """Load pipeline result into Neo4j and return summary."""
        with self.driver.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create constraints for performance
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS 
                FOR (c:Cell) REQUIRE c.key IS UNIQUE
            """)
            
            # Batch import nodes and edges
            self._import_nodes(session, result.formulas)
            self._import_edges(session, result.formulas)
            
            # Pre-compute PageRank
            self._compute_pagerank(session)
            
            # Generate summary
            return self._generate_summary(session)
```

## Agent Query Interface

### Interface Design

The agent interacts with the graph through a high-level query interface that abstracts Neo4j complexity.

```python
class GraphQueryInterface:
    """Agent-friendly interface for graph queries."""
    
    def get_dependencies(self, cell_ref: str, depth: int = 1) -> list[dict]:
        """Get dependencies of a cell up to specified depth."""
        
    def get_dependents(self, cell_ref: str, depth: int = 1) -> list[dict]:
        """Get cells that depend on this cell."""
        
    def find_circular_references(self, sheet: str = None) -> list[list[str]]:
        """Find all circular reference chains."""
        
    def get_critical_cells(self, limit: int = 10) -> list[dict]:
        """Get cells with highest PageRank scores."""
        
    def trace_calculation_path(self, from_cell: str, to_cell: str) -> list[str]:
        """Find calculation path between two cells."""
        
    def get_range_details(self, range_ref: str) -> dict:
        """Get information about a range node."""
```

### Agent Context Summary

Provide agents with a concise overview (~200 tokens):

```
Graph Analysis Summary:
- Total Nodes: 2,847 (2,000 cells, 847 ranges)
- Total Dependencies: 5,234
- Cross-Sheet Dependencies: 342 (15%)
- Max Calculation Depth: 12
- Critical Cells (PageRank > 0.9): 23
- Circular References: 2 chains detected
- Volatile Formulas: 45 cells

Available tools:
- query_graph(cypher: str) -> results
- get_cell_dependencies(ref: str) -> list
- find_calculation_chains() -> list
```

## Pandas-Excel Coordinate Mapping

### The Challenge

When agents load Excel data into Pandas:

- Pandas uses 0-based indexing
- Excel uses 1-based with letter columns
- Data may have offsets (skiprows, usecols)
- Multiple sheets complicate mapping

### Solution: Enhanced DataFrame Wrapper

```python
class ExcelAwareDataFrame:
    """DataFrame wrapper that maintains Excel coordinate mapping."""
    
    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.df = df
        self.sheet_name = metadata['sheet']
        self.start_row = metadata['start_row']  # Excel row (1-based)
        self.start_col = metadata['start_col']  # Excel col letter
        self.col_mapping = metadata['col_mapping']  # idx -> letter
    
    def to_excel_ref(self, row: int, col: int) -> str:
        """Convert Pandas indices to Excel reference."""
        excel_row = self.start_row + row
        excel_col = self.col_mapping.get(col, self._idx_to_letter(col))
        return f"{self.sheet_name}!{excel_col}{excel_row}"
    
    def from_excel_ref(self, ref: str) -> tuple[int, int]:
        """Convert Excel reference to Pandas indices."""
        # Parse ref and return (row_idx, col_idx)
    
    def query_dependencies(self, row: int, col: int) -> list[str]:
        """Get dependencies for a cell using graph interface."""
        excel_ref = self.to_excel_ref(row, col)
        return graph_interface.get_dependencies(excel_ref)
```

### Agent Tool Enhancement

```python
def load_excel_with_mapping(
    file_path: str,
    sheet_name: str = None,
    skiprows: int = 0,
    usecols: str = None
) -> ExcelAwareDataFrame:
    """
    Load Excel file with coordinate mapping preserved.
    
    Returns ExcelAwareDataFrame with methods:
    - df.to_excel_ref(row, col) -> "Sheet1!B5"
    - df.from_excel_ref("B5") -> (row, col)
    - df.query_dependencies(row, col) -> list
    """
```

## Implementation Plan

### Phase 1: Range Reference Handling (Week 1)

1. ✅ Update `FormulaParser` to identify range types
1. ✅ Modify `DependencyGraph` to support range nodes
1. ✅ Update `_analyze_formula` to create appropriate edges
1. ✅ Add configuration options to pipeline
1. ✅ Test with Business Accounting fixture

### Phase 2: Graph Database Integration (Week 2)

1. ⬜ Set up Neo4j development instance
1. ⬜ Implement `GraphDatabaseLoader`
1. ⬜ Add async loading to pipeline post-processing
1. ⬜ Create indexes and constraints
1. ⬜ Implement PageRank pre-computation

### Phase 3: Agent Interface (Week 3)

1. ⬜ Implement `GraphQueryInterface`
1. ⬜ Create `ExcelAwareDataFrame` wrapper
1. ⬜ Update agent tools for graph queries
1. ⬜ Add context summary generation
1. ⬜ Create usage examples and documentation

### Phase 4: Testing and Optimization (Week 4)

1. ⬜ Performance testing with large workbooks
1. ⬜ Query optimization
1. ⬜ Edge case handling
1. ⬜ Documentation updates
1. ⬜ Production deployment guide

## Performance Considerations

### Memory Management

- Range nodes reduce edges from O(n) to O(1) per range
- Lazy loading of CONTAINS relationships
- Configurable thresholds based on available memory

### Query Optimization

- Indexes on frequently queried properties
- Pre-computed aggregates for common queries
- Query result caching with TTL

### Scalability Targets

- Handle 1M+ nodes (cells + ranges)
- Sub-100ms query response for depth-1 traversals
- Support concurrent agent queries
- Horizontal scaling via Neo4j clustering

## Conclusion

This enhanced dependency graph system addresses current limitations while providing a scalable, agent-friendly interface for spreadsheet analysis. The hybrid approach to range handling balances completeness with performance, while graph database integration enables sophisticated queries that would be impractical with in-memory structures.

Key benefits:

1. **Complete dependency tracking** including range references
1. **Scalable architecture** supporting millions of nodes
1. **Agent-optimized interface** with coordinate mapping
1. **Performance optimization** through smart abstractions
1. **Future-proof design** supporting advanced graph algorithms
