# Multi-Table Detection Workflow

This document explains how to use the multi-table detection feature in the spreadsheet analyzer.

## Overview

The multi-table detection workflow uses a two-agent architecture:

1. **Table Detector Agent**: Identifies table boundaries in spreadsheets
1. **Table-Aware Analyst Agent**: Analyzes each detected table with boundary awareness

## Quick Start

### Basic Usage

```python
import asyncio
from pathlib import Path
from spreadsheet_analyzer.workflows.multi_table_workflow import run_multi_table_analysis

async def analyze_spreadsheet():
    result = await run_multi_table_analysis(Path("data.xlsx"))
    if result.is_ok():
        analysis = result.unwrap()
        print(f"Found {analysis['tables_found']} tables")

asyncio.run(analyze_spreadsheet())
```

### With Configuration

```python
from spreadsheet_analyzer.cli.notebook_analysis import AnalysisConfig

config = AnalysisConfig(
    excel_path=Path("data.xlsx"),
    sheet_index=0,
    output_dir=Path("outputs"),
    model="gpt-4o-mini",
    max_rounds=5,
    auto_save_rounds=True,
)

result = await run_multi_table_analysis(Path("data.xlsx"), config=config)
```

## How It Works

### 1. Table Detection Phase

The detector agent:

- Loads the Excel file into a pandas DataFrame
- Analyzes the structure for empty rows (mechanical detection)
- Groups consecutive non-empty rows as tables
- Identifies table types (detail, summary, header)
- Returns table boundaries with metadata

### 2. Analysis Phase

The analyst agent:

- Receives detected table boundaries
- Uses a specialized prompt that focuses on the pre-detected tables
- Analyzes each table independently
- Provides insights without re-detecting tables

### 3. Workflow Coordination

A supervisor node:

- Manages the workflow state
- Routes between detector and analyst agents
- Ensures proper sequencing
- Handles errors gracefully

## Table Detection Methods

### Mechanical Detection (Default)

- Identifies tables based on empty rows
- Groups consecutive non-empty rows
- High confidence for clear separations

### Semantic Detection (Future)

- Analyzes content patterns
- Identifies logical groupings
- Handles tables without clear separations

## Output Structure

The workflow returns:

```python
{
    "detection_notebook": "path/to/detection_notebook.ipynb",
    "analysis_notebook": "path/to/analysis_notebook.ipynb", 
    "tables_found": 3,
    "detection_error": None,
    "analysis_error": None
}
```

## Examples

### Example 1: Multi-Table Sales Report

```python
# Create test data with multiple tables
import pandas as pd

# Table 1: Product sales
products = pd.DataFrame({
    "Product": ["A", "B", "C"],
    "Sales": [100, 200, 150]
})

# Empty separator
empty = pd.DataFrame([[None, None]] * 2)

# Table 2: Regional summary
regions = pd.DataFrame({
    "Region": ["North", "South"],
    "Total": [300, 150]
})

# Combine and save
combined = pd.concat([products, empty, regions], ignore_index=True)
combined.to_excel("multi_table.xlsx", index=False)

# Analyze
result = await run_multi_table_analysis(Path("multi_table.xlsx"))
```

### Example 2: Direct Agent Usage

```python
from spreadsheet_analyzer.agents.table_detector_agent import create_table_detector
from spreadsheet_analyzer.agents.types import AgentMessage, AgentId, AgentState

# Create detector
detector = create_table_detector()

# Create message
message = AgentMessage.create(
    sender=AgentId.generate("test"),
    receiver=detector.id,
    content={
        "dataframe": df,
        "sheet_name": "Sheet1",
        "file_path": "data.xlsx"
    }
)

# Run detection
state = AgentState(agent_id=detector.id, status="idle")
result = detector.process(message, state)
```

## Integration with Existing Pipeline

The multi-table workflow integrates seamlessly:

1. **Backward Compatible**: Single-table sheets work as before
1. **Automatic Detection**: Can detect when multi-table analysis is needed
1. **Manual Override**: Force multi-table analysis when needed

```python
# Check if multi-table analysis is needed
df = pd.read_excel("data.xlsx")
empty_rows = df.isnull().all(axis=1).sum()

if empty_rows > 0:
    # Use multi-table workflow
    result = await run_multi_table_analysis(Path("data.xlsx"))
else:
    # Use standard analysis
    result = await run_standard_analysis(Path("data.xlsx"))
```

## Configuration Options

### Analysis Config

- `model`: LLM model to use
- `max_rounds`: Maximum analysis rounds
- `auto_save_rounds`: Save after each round
- `verbose`: Enable detailed logging
- `track_costs`: Track API costs

### Workflow State

- `table_boundaries`: Detected boundaries
- `detection_notebook_path`: Detection results
- `analysis_notebook_path`: Analysis results
- `messages`: Conversation history

## Troubleshooting

### Common Issues

1. **No tables detected**

   - Check for empty rows between tables
   - Verify data structure
   - Use verbose mode for debugging

1. **Detection errors**

   - Ensure Excel file is valid
   - Check sheet index
   - Verify pandas can read the file

1. **Analysis failures**

   - Check API keys
   - Verify model availability
   - Review error logs

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = AnalysisConfig(
    excel_path=Path("data.xlsx"),
    verbose=True,  # Enable verbose output
    output_dir=Path("debug_output"),
)
```

## Best Practices

1. **Data Preparation**

   - Use empty rows to separate tables
   - Keep table headers clear
   - Avoid merged cells at boundaries

1. **Performance**

   - Use appropriate max_rounds
   - Enable auto_save for large analyses
   - Monitor token usage with track_costs

1. **Error Handling**

   - Always check Result.is_ok()
   - Log errors for debugging
   - Provide fallback behavior

## Future Enhancements

- [ ] Semantic table detection
- [ ] Custom boundary detection strategies
- [ ] Multi-sheet parallel processing
- [ ] Table relationship analysis
- [ ] Export to structured formats
