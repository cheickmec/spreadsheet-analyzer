# Notebook-LLM Integration with Kernel Manager

This document demonstrates how the LLM will interact with notebooks using our kernel manager implementation.

## Architecture Overview

Based on the design documents, the integration follows this pattern:

```
LLM → NAP Dispatcher → Notebook Cell → Kernel Manager → Isolated Execution
```

## Example Implementation

Here's a simplified example showing how the components work together:

```python
import asyncio
from typing import Any
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager


class NotebookLLMInterface:
    """Interface between LLM and notebook execution using kernel manager."""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.notebook = nbformat.v4.new_notebook()
        self.kernel_manager = AgentKernelManager(max_kernels=5)
        self.agent_id = "llm-analyst"
        
    async def __aenter__(self):
        await self.kernel_manager.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.kernel_manager.__aexit__(exc_type, exc_val, exc_tb)
        
    async def dispatch(self, command: dict) -> dict:
        """NAP-style unified dispatcher for notebook operations."""
        op = command["op"]
        
        if op == "add_cell":
            return await self._add_and_execute_cell(command)
        elif op == "get_cells":
            return self._get_cells(command)
        else:
            raise ValueError(f"Unknown operation: {op}")
            
    async def _add_and_execute_cell(self, command: dict) -> dict:
        """Add a cell to notebook and execute it in isolated kernel."""
        source = command["source"]
        cell_type = command.get("type", "code")
        
        # Create notebook cell
        if cell_type == "code":
            cell = new_code_cell(source=source)
            # Execute in isolated kernel
            result = await self._execute_in_kernel(source)
            cell.outputs = self._format_outputs(result)
        else:
            cell = new_markdown_cell(source=source)
            result = {"status": "ok", "outputs": []}
            
        # Add to notebook
        position = command.get("position", -1)
        if position == -1:
            self.notebook.cells.append(cell)
        else:
            self.notebook.cells.insert(position, cell)
            
        return {
            "status": result["status"],
            "outputs": result.get("outputs", []),
            "cell_id": cell.metadata.get("id", str(len(self.notebook.cells))),
            "token_estimate": len(source)  # Simplified
        }
        
    async def _execute_in_kernel(self, code: str) -> dict[str, Any]:
        """Execute code in isolated kernel with our kernel manager."""
        async with self.kernel_manager.acquire_kernel(self.agent_id) as (km, session):
            result = await self.kernel_manager.execute_code(session, code)
            return result
            
    def _format_outputs(self, result: dict) -> list:
        """Convert kernel execution result to notebook output format."""
        outputs = []
        
        if result["status"] == "error":
            outputs.append({
                "output_type": "error",
                "ename": result.get("error", {}).get("ename", "Error"),
                "evalue": result.get("error", {}).get("evalue", ""),
                "traceback": result.get("error", {}).get("traceback", [])
            })
        else:
            for output in result.get("outputs", []):
                if output["type"] == "stream":
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": output["text"]
                    })
                elif output["type"] == "execute_result":
                    outputs.append({
                        "output_type": "execute_result",
                        "data": output["data"],
                        "metadata": {},
                        "execution_count": 1
                    })
                    
        return outputs
        
    def _get_cells(self, command: dict) -> dict:
        """Get cells from notebook with optional range."""
        start = command.get("start", 0)
        end = command.get("end", len(self.notebook.cells))
        
        cells = []
        for i, cell in enumerate(self.notebook.cells[start:end], start):
            cells.append({
                "index": i,
                "type": cell.cell_type,
                "source": cell.source,
                "outputs": getattr(cell, "outputs", [])
            })
            
        return {
            "cells": cells,
            "total": len(self.notebook.cells),
            "token_estimate": sum(len(cell["source"]) for cell in cells)
        }


# Example usage showing LLM interaction pattern
async def llm_analysis_example():
    """Example of how an LLM would analyze a spreadsheet."""
    
    async with NotebookLLMInterface("sales_data.xlsx") as interface:
        # LLM generates bootstrap cell
        result = await interface.dispatch({
            "op": "add_cell",
            "source": """
import pandas as pd
from pathlib import Path

# Load the Excel file
excel_path = Path("sales_data.xlsx")
df = pd.read_excel(excel_path, sheet_name="Sales")
print(f"Loaded {len(df)} rows from Sales sheet")
print(f"Columns: {list(df.columns)}")
df.head()
"""
        })
        print(f"Bootstrap result: {result['status']}")
        
        # LLM analyzes structure and generates exploration cell
        result = await interface.dispatch({
            "op": "add_cell", 
            "source": """
# Check for formulas in the Total column
import openpyxl
wb = openpyxl.load_workbook(excel_path, data_only=False)
ws = wb["Sales"]

formula_count = 0
for cell in ws['E']:
    if cell.data_type == 'f':
        formula_count += 1
        if formula_count == 1:  # Show first formula as example
            print(f"Example formula in {cell.coordinate}: {cell.value}")
            
print(f"\\nTotal formulas in column E: {formula_count}")
"""
        })
        
        # LLM validates findings
        result = await interface.dispatch({
            "op": "add_cell",
            "source": """
# Validate that formulas calculate correctly
# Compare Excel formulas with pandas calculations
df['Calculated_Total'] = df['Units'] * df['Price']
df['Formula_Match'] = df['Total'] == df['Calculated_Total']

mismatches = df[~df['Formula_Match']]
print(f"Formula validation: {len(mismatches)} mismatches found")
if len(mismatches) > 0:
    print("\\nFirst few mismatches:")
    print(mismatches[['Units', 'Price', 'Total', 'Calculated_Total']].head())
"""
        })
        
        # Get all cells for review
        all_cells = await interface.dispatch({
            "op": "get_cells"
        })
        
        print(f"\\nNotebook now has {all_cells['total']} cells")
        print(f"Total tokens used: {all_cells['token_estimate']}")


if __name__ == "__main__":
    asyncio.run(llm_analysis_example())
```

## Key Integration Points

### 1. NAP-Style Dispatcher

The `dispatch` method provides a unified entry point matching the NAP protocol design:

- Single entry point for all operations
- Consistent response format with token estimates
- Execution-by-default for code cells

### 2. Kernel Manager Integration

Each code cell executes in an isolated kernel:

- Kernel acquired per agent (LLM session)
- Session persistence across multiple cells
- Resource limits enforced (CPU, memory, time)
- Automatic cleanup on errors

### 3. Excel Context Enhancement

While not shown in this simplified example, the full implementation would:

- Inject Excel metadata into each cell
- Add formula dependency context
- Include graph database insights
- Provide validation patterns

### 4. Progressive Context Management

The interface supports:

- Getting specific cell ranges
- Token estimation for context budgeting
- Semantic grouping of related cells

## Benefits of This Architecture

1. **Security**: Code executes in isolated kernels with resource limits
1. **Persistence**: Agent state maintained across analysis steps
1. **Auditability**: Complete notebook preserves analysis history
1. **Flexibility**: LLM can explore iteratively, building on previous results
1. **Validation**: Each claim can be verified through code execution

## Next Steps

To complete the integration:

1. Implement the full NAP protocol schema
1. Add Excel context enrichment layer
1. Integrate with graph database for dependency insights
1. Create semantic cell grouping algorithms
1. Build token budget management system
1. Add validation pattern engine

This kernel manager provides the secure execution foundation for the complete notebook-LLM interface described in the design documents.
