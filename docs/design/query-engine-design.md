# Spreadsheet Query Engine Design

## Overview

The `SpreadsheetQueryEngine` provides a unified, intuitive interface for querying spreadsheet dependencies without needing to understand the distinction between "direct" and "range" dependencies. While designed with LLMs in mind, it's suitable for any client including APIs, CLIs, or web interfaces.

## Key Design Decisions

### 1. Unified Dependency Model

Instead of separating dependencies into "direct" and "range" categories, the interface treats all dependencies uniformly:

- A cell can **use** another cell (dependency)
- A cell can be **used by** another cell (dependent)
- The relationship includes metadata about whether it's via a range

### 2. Flexible Query Types

The interface supports various query types that mirror natural questions:

```python
# What does this cell depend on?
engine.query("dependencies", sheet="Sheet1", cell="B500")

# What would be affected if this cell changes?
engine.query("impact", sheet="Sheet1", cell="A1", depth=3)

# Is there a path between these cells?
engine.query("path", from_sheet="Sheet1", from_cell="A1", 
            to_sheet="Sheet1", to_cell="Z100")

# What are the data sources for this calculation?
engine.query("sources", sheet="Sheet1", cell="D10")
```

### 3. Rich Result Structure

Every query returns a consistent `QueryResult` with:

- `query_type`: What kind of query was executed
- `success`: Whether the query succeeded
- `data`: Structured data specific to the query type
- `cells`: List of `CellInfo` objects with complete cell information
- `relationships`: List of `DependencyInfo` objects describing connections
- `explanation`: Human-readable explanation of the results

### 4. Automatic Range Membership

The interface automatically includes range membership information:

- Empty cells are tracked if they're part of formula ranges
- Each `CellInfo` includes `in_ranges` listing all ranges containing that cell
- No need to explicitly ask for range information

## Example LLM Usage

### Function Definitions for LLM

```python
functions = [
    {
        "name": "query_spreadsheet",
        "description": "Query spreadsheet dependencies and relationships",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["dependencies", "dependents", "impact", "sources", 
                            "path", "circular", "neighbors", "formula", "exists", "stats"],
                    "description": "Type of query to perform"
                },
                "sheet": {
                    "type": "string",
                    "description": "Sheet name"
                },
                "cell": {
                    "type": "string",
                    "description": "Cell reference (e.g., 'B500')"
                },
                "depth": {
                    "type": "integer",
                    "description": "How many levels to traverse (for impact queries)"
                },
                "from_sheet": {
                    "type": "string",
                    "description": "Source sheet (for path queries)"
                },
                "from_cell": {
                    "type": "string",
                    "description": "Source cell (for path queries)"
                },
                "to_sheet": {
                    "type": "string",
                    "description": "Target sheet (for path queries)"
                },
                "to_cell": {
                    "type": "string",
                    "description": "Target cell (for path queries)"
                }
            },
            "required": ["query_type"]
        }
    }
]
```

### Example Conversations

**User**: "What does cell B500 in Sheet1 depend on?"

**LLM Function Call**:

```python
query_spreadsheet(query_type="dependencies", sheet="Sheet1", cell="B500")
```

**Result**:

```python
QueryResult(
    query_type=QueryType.DEPENDENCIES,
    success=True,
    data={"total_dependencies": 0},
    cells=[
        CellInfo(
            ref="B500",
            sheet="Sheet1",
            formula=None,
            is_empty=True,
            in_ranges=["Sheet1!B1:B1000", "Sheet1!A1:C1000"]
        )
    ],
    relationships=[],
    explanation="B500 depends on 0 cells and is part of 2 ranges"
)
```

**LLM Response**: "Cell B500 in Sheet1 is empty (has no formula) but is included in 2 ranges: B1:B1000 and A1:C1000. This means any formulas that reference these ranges will include B500 in their calculations."

______________________________________________________________________

**User**: "If I change cell A1, what other cells would be affected?"

**LLM Function Call**:

```python
query_spreadsheet(query_type="impact", sheet="Sheet1", cell="A1", depth=3)
```

**Result**:

```python
QueryResult(
    query_type=QueryType.IMPACT,
    success=True,
    data={"impacted_cells": 5, "max_depth": 3},
    cells=[...],  # List of affected cells
    relationships=[...],
    explanation="Changes to A1 would impact 5 cells within 3 steps"
)
```

______________________________________________________________________

**User**: "Show me the calculation path from A1 to D10"

**LLM Function Call**:

```python
query_spreadsheet(
    query_type="path",
    from_sheet="Sheet1",
    from_cell="A1",
    to_sheet="Sheet1", 
    to_cell="D10"
)
```

## Advanced Queries

The interface also supports more complex graph-like queries through the raw Cypher interface:

```python
# For advanced users/LLMs that understand Cypher
from spreadsheet_analyzer.graph_db.query_interface import GraphQueryInterface

# Find all cells that depend on more than 10 other cells
result = graph_db.query_graph("""
    MATCH (n)-[:DEPENDS_ON]->(m)
    WITH n, count(m) as dep_count
    WHERE dep_count > 10
    RETURN n.key as cell, dep_count
    ORDER BY dep_count DESC
""")
```

## Implementation Benefits

1. **No Confusing Terminology**: Avoids the "direct" vs "indirect" confusion
1. **Automatic Range Handling**: Range membership is always included
1. **Flexible Queries**: Natural query types that match how users think
1. **Efficient**: Still uses our RangeMembershipIndex for O(log n) lookups
1. **Extensible**: Easy to add new query types

## Migration from Old Interface

Old interface (query_interface.py):

```python
result = interface.get_cell_dependencies(sheet, cell, include_ranges=True)
if result.direct_dependencies:
    # Handle direct deps
if result.range_dependencies:
    # Handle range deps
```

New interface (query_engine.py):

```python
result = engine.query("dependencies", sheet=sheet, cell=cell)
for rel in result.relationships:
    if rel.via_range:
        # This dependency is through a range
    else:
        # This is a cell-to-cell dependency
```

The new interface is more intuitive and doesn't require understanding our internal categorization of dependencies.

## Module Organization

The `graph_db` package now contains:

1. **query_engine.py** - High-level, flexible query interface (no Neo4j required)
1. **query_interface.py** - Lower-level Neo4j-specific queries (requires Neo4j)
1. **range_membership.py** - Efficient range membership index
1. **loader.py** - Neo4j database loader (requires Neo4j)

The query engine is always available and provides the primary interface for analyzing dependencies, while the Neo4j components are optional for users who want graph database persistence.
