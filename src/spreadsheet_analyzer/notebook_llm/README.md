# LLM-Jupyter Integration Framework

This module implements a three-layer architecture for LLM-powered spreadsheet analysis using Jupyter notebooks as the execution environment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SpreadsheetLLMAnalyzer                    │
│                    (integration.py)                          │
├─────────────────────────────────────────────────────────────┤
│                    Analysis Strategies                       │
│              (analysis_strategies.py)                        │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │  Structure  │   Formula   │Data Quality │ Cell Level  │ │
│  │  Analysis   │  Analysis   │  Analysis   │  Analysis   │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Notebook Executor                          │
│                 (notebook_executor.py)                       │
│  • Jupyter kernel management                                 │
│  • Code execution with state                                 │
│  • Result extraction                                         │
├─────────────────────────────────────────────────────────────┤
│                     Tool Registry                            │
│                     (registry.py)                            │
│  • Spreadsheet tools registration                            │
│  • LLM query interface                                       │
│  • Tool validation and dispatch                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from spreadsheet_analyzer.notebook_llm.integration import SpreadsheetLLMAnalyzer

# Initialize analyzer
analyzer = SpreadsheetLLMAnalyzer()

# Load workbook
analyzer.load_workbook("path/to/workbook.xlsx")

# Run analysis
results = analyzer.analyze("structure")  # or "formulas", "data_quality", "cell_level"

# Get recommendations
recommendations = analyzer.get_recommendations()

# Export results
analyzer.export_results("report.json", format="json")
```

## Components

### 1. Tool Registry (`registry.py`)

Manages the interface between LLM queries and spreadsheet operations:

- **Workbook Tools**: Access sheets, read cells, get formulas
- **Analysis Tools**: Pattern detection, validation, statistics
- **LLM Tools**: Query interface with context management

### 2. Notebook Executor (`notebook_executor.py`)

Handles Jupyter kernel lifecycle and code execution:

- **Kernel Management**: Start, stop, restart kernels
- **State Persistence**: Maintain context across queries
- **Error Handling**: Graceful recovery from execution errors

### 3. Analysis Strategies (`analysis_strategies.py`)

Implements different analysis approaches:

- **Structure Analysis**: Sheet organization, relationships
- **Formula Analysis**: Dependencies, complexity, errors
- **Data Quality**: Validation, completeness, anomalies
- **Cell Level**: Detailed cell-by-cell examination

### 4. Integration (`integration.py`)

High-level API that combines all components:

- **SpreadsheetLLMAnalyzer**: Main class for analysis
- **Result Management**: Caching and aggregation
- **Export Functions**: JSON and Markdown reports

## Usage Examples

### Basic Analysis

```python
# Quick analysis with convenience function
from spreadsheet_analyzer.notebook_llm.integration import analyze_spreadsheet

summary = analyze_spreadsheet(
    "workbook.xlsx",
    strategies=["structure", "formulas"],
    output_path="report.json"
)
```

### Advanced Analysis with Context

```python
analyzer = SpreadsheetLLMAnalyzer()
analyzer.load_workbook("sales_data.xlsx")

# Provide business context
context = {
    "business_domain": "sales",
    "validation_rules": {
        "amount": {"min": 0, "max": 1000000}
    }
}

# Run targeted analysis
result = analyzer.analyze("data_quality", context=context)
```

### Custom Strategy Usage

```python
# Run specific analysis on selected sheets
cell_result = analyzer.analyze(
    "cell_level",
    sheets=["Sheet1", "Sheet2"],
    sample_size=100,
    focus_areas=["formulas", "formatting"]
)
```

## Extending the Framework

### Adding New Tools

```python
# In registry.py
def register_custom_tool(self, name: str, func: Callable):
    """Register a custom analysis tool."""
    self.tools[name] = ToolSpec(
        name=name,
        func=func,
        params=inspect.signature(func).parameters,
        description=func.__doc__
    )
```

### Creating New Strategies

```python
# Extend AnalysisStrategy base class
class CustomStrategy(AnalysisStrategy):
    def analyze(self, workbook: Workbook, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement your analysis logic
        prompt = self._build_prompt(context)
        result = self.executor.execute_analysis(prompt, workbook)
        return self._process_result(result)
```

## Best Practices

1. **Context Management**: Always provide relevant business context for better analysis
1. **Strategy Selection**: Choose appropriate strategies based on analysis goals
1. **Resource Cleanup**: Always call `cleanup()` to release kernel resources
1. **Error Handling**: Wrap analysis calls in try-except blocks
1. **Result Caching**: Reuse analyzer instances for multiple analyses on same workbook

## Performance Considerations

- **Kernel Startup**: First analysis incurs ~2s kernel startup time
- **Memory Usage**: Large workbooks may require read_only mode
- **Execution Time**: Complex analyses may take 10-30 seconds
- **Parallel Analysis**: Strategies can be run in parallel with separate executors

## Troubleshooting

### Common Issues

1. **Kernel Won't Start**: Check Jupyter installation with `jupyter --version`
1. **Import Errors**: Ensure all dependencies installed with `uv sync`
1. **Memory Issues**: Use read_only mode for large files
1. **Timeout Errors**: Increase timeout in NotebookExecutor

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("spreadsheet_analyzer.notebook_llm").setLevel(logging.DEBUG)
```

## Integration with Main Analyzer

This framework integrates with the main BaseAnalyzer through the stage system:

```python
from spreadsheet_analyzer.base_analyzer import BaseAnalyzer
from spreadsheet_analyzer.notebook_llm.integration import SpreadsheetLLMAnalyzer

class HybridAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.llm_analyzer = SpreadsheetLLMAnalyzer()
    
    def analyze(self, file_path: Path):
        # Run deterministic stages
        basic_results = super().analyze(file_path)
        
        # Add LLM-powered insights
        self.llm_analyzer.load_workbook(file_path)
        llm_results = self.llm_analyzer.analyze_all()
        
        # Combine results
        return {**basic_results, "llm_insights": llm_results}
```
