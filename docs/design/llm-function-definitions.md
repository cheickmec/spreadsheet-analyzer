# LLM Function Definitions for Spreadsheet Analysis

## Overview

This document shows how to expose the spreadsheet analysis capabilities to an LLM using function calling interfaces like those used by OpenAI, Anthropic, or other LLM providers.

## Single Unified Function Definition

```json
{
  "name": "analyze_spreadsheet_dependencies",
  "description": "Query and analyze dependencies in Excel spreadsheets. This function can answer questions about cell relationships, formula dependencies, and the impact of changes.",
  "parameters": {
    "type": "object",
    "properties": {
      "query_type": {
        "type": "string",
        "enum": [
          "dependencies",
          "dependents", 
          "impact",
          "sources",
          "path",
          "circular",
          "neighbors",
          "formula",
          "exists",
          "stats"
        ],
        "description": "The type of analysis to perform:\n- dependencies: What cells does this cell depend on?\n- dependents: What cells depend on this cell?\n- impact: What cells would be affected by changes to this cell?\n- sources: What are the ultimate data sources for this cell?\n- path: Is there a calculation path between two cells?\n- circular: Is this cell part of a circular reference?\n- neighbors: All direct connections to this cell\n- formula: Get the formula in this cell\n- exists: Does this cell exist in the spreadsheet?\n- stats: Get statistics about this cell's connections"
      },
      "sheet": {
        "type": "string",
        "description": "The name of the worksheet (e.g., 'Balance Sheet', 'Income Statement')"
      },
      "cell": {
        "type": "string", 
        "description": "The cell reference (e.g., 'A1', 'B500', 'D10')"
      },
      "depth": {
        "type": "integer",
        "default": 2,
        "description": "For 'impact' queries: how many levels of dependencies to traverse"
      },
      "from_sheet": {
        "type": "string",
        "description": "For 'path' queries: the source worksheet"
      },
      "from_cell": {
        "type": "string",
        "description": "For 'path' queries: the source cell reference"
      },
      "to_sheet": {
        "type": "string",
        "description": "For 'path' queries: the target worksheet"
      },
      "to_cell": {
        "type": "string",
        "description": "For 'path' queries: the target cell reference"
      }
    },
    "required": ["query_type"],
    "dependencies": {
      "query_type": {
        "oneOf": [
          {
            "properties": {
              "query_type": {"const": "dependencies"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "dependents"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "impact"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "sources"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "path"}
            },
            "required": ["from_sheet", "from_cell", "to_sheet", "to_cell"]
          },
          {
            "properties": {
              "query_type": {"const": "circular"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "neighbors"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "formula"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "exists"}
            },
            "required": ["sheet", "cell"]
          },
          {
            "properties": {
              "query_type": {"const": "stats"}
            },
            "required": ["sheet", "cell"]
          }
        ]
      }
    }
  }
}
```

## Example Function Calls and Responses

### Example 1: Check Dependencies

**User**: "What does cell B10 in the Balance Sheet depend on?"

**LLM Function Call**:

```json
{
  "name": "analyze_spreadsheet_dependencies",
  "parameters": {
    "query_type": "dependencies",
    "sheet": "Balance Sheet",
    "cell": "B10"
  }
}
```

**Response**:

```json
{
  "query_type": "dependencies",
  "success": true,
  "explanation": "B10 depends on 3 cells",
  "data": {
    "total_dependencies": 3
  },
  "cells": [
    {
      "ref": "B10",
      "sheet": "Balance Sheet",
      "formula": "=B8+B9",
      "is_empty": false,
      "in_ranges": []
    },
    {
      "ref": "B8",
      "sheet": "Balance Sheet",
      "formula": "=SUM(B1:B7)",
      "is_empty": false,
      "in_ranges": []
    },
    {
      "ref": "B9",
      "sheet": "Balance Sheet",
      "formula": null,
      "is_empty": false,
      "in_ranges": []
    }
  ],
  "relationships": [
    {
      "cell": {"ref": "B8", "sheet": "Balance Sheet"},
      "relationship": "uses",
      "via_range": null
    },
    {
      "cell": {"ref": "B9", "sheet": "Balance Sheet"},
      "relationship": "uses",
      "via_range": null
    }
  ]
}
```

### Example 2: Empty Cell in Range

**User**: "Is cell B500 used in any formulas?"

**LLM Function Call**:

```json
{
  "name": "analyze_spreadsheet_dependencies",
  "parameters": {
    "query_type": "dependents",
    "sheet": "Balance Sheet", 
    "cell": "B500"
  }
}
```

**Response**:

```json
{
  "query_type": "dependents",
  "success": true,
  "explanation": "2 cells depend on B500",
  "data": {
    "total_dependents": 2
  },
  "cells": [
    {
      "ref": "B500",
      "sheet": "Balance Sheet",
      "formula": null,
      "is_empty": true,
      "in_ranges": ["Balance Sheet!B1:B1000", "Balance Sheet!A1:C1000"]
    }
  ],
  "relationships": [
    {
      "cell": {"ref": "D1", "sheet": "Balance Sheet"},
      "relationship": "used_by",
      "via_range": "Balance Sheet!B1:B1000"
    },
    {
      "cell": {"ref": "E1", "sheet": "Summary"},
      "relationship": "used_by",
      "via_range": "Balance Sheet!A1:C1000"
    }
  ]
}
```

### Example 3: Impact Analysis

**User**: "What would happen if I change the value in A1?"

**LLM Function Call**:

```json
{
  "name": "analyze_spreadsheet_dependencies",
  "parameters": {
    "query_type": "impact",
    "sheet": "Balance Sheet",
    "cell": "A1",
    "depth": 3
  }
}
```

**Response**:

```json
{
  "query_type": "impact",
  "success": true,
  "explanation": "Changes to A1 would impact 12 cells within 3 steps",
  "data": {
    "impacted_cells": 12,
    "max_depth": 3
  },
  "cells": [
    // List of 12 impacted cells with their information
  ]
}
```

## Integration Code Example

```python
def handle_spreadsheet_function_call(params: dict, spreadsheet_engine):
    """Handle function calls from the LLM."""
    query_type = params.get("query_type")
    
    # Remove query_type from params to pass remaining as kwargs
    query_params = {k: v for k, v in params.items() if k != "query_type"}
    
    # Execute query
    result = spreadsheet_engine.query(query_type, **query_params)
    
    # Convert to JSON-serializable format
    return {
        "query_type": result.query_type.value,
        "success": result.success,
        "explanation": result.explanation,
        "data": result.data,
        "cells": [
            {
                "ref": cell.ref,
                "sheet": cell.sheet,
                "formula": cell.formula,
                "is_empty": cell.is_empty,
                "in_ranges": cell.in_ranges
            }
            for cell in result.cells
        ],
        "relationships": [
            {
                "cell": {
                    "ref": rel.cell.ref,
                    "sheet": rel.cell.sheet
                },
                "relationship": rel.relationship,
                "via_range": rel.via_range
            }
            for rel in result.relationships
        ]
    }
```

## Benefits of This Approach

1. **Single Function**: One function handles all query types, reducing complexity
1. **Self-Documenting**: The enum descriptions explain each query type
1. **Flexible Parameters**: Only required parameters for each query type
1. **Rich Responses**: Consistent response format with explanation text
1. **Range Awareness**: Automatically handles empty cells in ranges
1. **No Ambiguity**: Clear relationships without confusing "direct" terminology

## Example LLM System Prompt

```
You have access to a spreadsheet analysis function that can help you understand Excel file dependencies and relationships. 

Key capabilities:
- Find what cells a formula depends on
- Find what formulas use a specific cell
- Track empty cells that are part of formula ranges
- Analyze the impact of changes
- Find calculation paths between cells
- Detect circular references

When users ask about spreadsheet formulas or dependencies, use the analyze_spreadsheet_dependencies function to get accurate information. The function automatically handles both cell-to-cell dependencies and range memberships.

Important: Empty cells can still be important if they're part of a range used in formulas. Always check the "in_ranges" field to see if an empty cell is included in calculations.
```

This design provides a clean, intuitive interface that LLMs can use effectively without needing to understand the complexities of spreadsheet dependency tracking.
