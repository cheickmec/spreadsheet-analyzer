# Iterative Refinement in Spreadsheet Analyzer

## Overview

The spreadsheet analyzer now includes **iterative refinement** capabilities that automatically improve analysis quality through execution feedback loops. Instead of making a single LLM call and hoping for the best, the system now:

1. Generates initial analysis code
1. Executes the code and captures outputs/errors
1. Inspects the results for quality issues
1. Refines the analysis based on execution feedback
1. Repeats until satisfactory results are achieved

This implements the **ReAct pattern** (Reasoning → Action → Observation) where the LLM learns from actual execution results.

## How It Works

### 1. Observation Building

After each execution round, the system builds an "observation" from the notebook outputs:

```python
def build_observation_from_notebook(notebook: NotebookDocument, max_output_chars: int = 2000) -> str:
    """Build an observation string from notebook execution outputs."""
```

This captures:

- **Errors**: Python exceptions, tracebacks, and error messages
- **Outputs**: Print statements, data displays, and visualization results
- **Empty cells**: Code that produced no output

### 2. Quality Inspection

The system automatically inspects notebook quality to determine if refinement is needed:

```python
def inspect_notebook_quality(notebook: NotebookDocument) -> tuple[bool, str]:
    """Inspect notebook to determine if refinement is needed."""
```

Quality checks include:

- Execution errors that need fixing
- Empty outputs (no print/display statements)
- Insufficient analysis depth (too few cells)
- Missing visualizations or statistics

### 3. Iterative Loop

The main analysis function now includes a refinement loop:

```python
async def analyze_sheet_with_llm(
    notebook: NotebookDocument, 
    provider, 
    strategy_name: str = "hierarchical", 
    log_path: Path | None = None,
    max_rounds: int = 3,      # Maximum refinement rounds
    cost_limit: float = 0.50  # Maximum cost in USD
) -> tuple[NotebookDocument, dict]:
```

Key features:

- **Max rounds**: Prevents infinite loops (default: 3)
- **Cost tracking**: Monitors LLM costs per round
- **Cost limit**: Stops if cost exceeds threshold (default: $0.50)
- **Refinement history**: Tracks all rounds and decisions

### 4. Refinement Prompts

When refinement is needed, the system provides the LLM with:

- Execution results (errors and outputs)
- Specific instructions to fix issues
- Request for deeper analysis if needed
- Context from previous attempts

## Example Scenarios

### Scenario 1: Handling Data Type Errors

**Round 1**: LLM generates code assuming numeric columns

```python
df['Total'] = df['Quantity'] * df['Price']  # Fails if columns are strings
```

**Observation**: `TypeError: can't multiply sequence by non-int of type 'str'`

**Round 2**: LLM fixes the issue

```python
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'].str.replace('$', ''), errors='coerce')
df['Total'] = df['Quantity'] * df['Price']
print(f"Calculated totals: {df['Total'].sum()}")
```

### Scenario 2: Incomplete Analysis

**Round 1**: LLM provides basic code without outputs

```python
stats = df.describe()
correlations = df.corr()
```

**Observation**: "All code cells produced no output"

**Round 2**: LLM adds display statements

```python
print("Statistical Summary:")
print(df.describe())
print("\nCorrelation Matrix:")
print(df.corr())

# Add visualization
import matplotlib.pyplot as plt
df.plot(kind='scatter', x='Price', y='Quantity')
plt.title('Price vs Quantity Analysis')
plt.show()
```

## Usage

The iterative refinement is **automatically enabled** when using `analyze_sheet.py`. No special configuration needed:

```bash
uv run src/spreadsheet_analyzer/cli/analyze_sheet.py data.xlsx "Sheet1"
```

### Monitoring Refinement

With verbose mode, you can see the refinement process:

```bash
uv run src/spreadsheet_analyzer/cli/analyze_sheet.py data.xlsx "Sheet1" --verbose
```

Output shows:

```
=== Analysis Round 1/3 ===
Sending request to LLM...
  - Tokens: 1500
  - Round cost: $0.0105
  - Total cost: $0.0105
Executing generated code...
  - Quality check: Found 2 execution errors that need to be fixed
  → Refining analysis...

=== Analysis Round 2/3 ===
...
```

### Configuration Options

In the code, you can adjust:

```python
# Maximum refinement attempts
max_rounds = 5  # Default: 3

# Cost limit in USD
cost_limit = 1.00  # Default: $0.50
```

## Benefits

1. **Self-correcting**: Automatically fixes common errors like:

   - Type conversion issues
   - Missing imports
   - Incorrect column names
   - Empty DataFrames

1. **Comprehensive analysis**: Ensures analyses include:

   - Meaningful outputs
   - Statistical summaries
   - Visualizations when appropriate
   - Clear insights

1. **Cost-aware**: Tracks spending and stops if limits exceeded

1. **Transparent**: All interactions logged for debugging

## Technical Details

### Cost Tracking

The system tracks costs based on token usage:

**Claude models**:

- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens

**GPT models**:

- Input: $0.001 per 1K tokens
- Output: $0.002 per 1K tokens

### Logging

All LLM interactions are logged to:

```
analysis_results/[excel_name]/[sheet_name]_llm_log.json
```

Each log entry includes:

- Request/response content
- Token usage
- Cost per round
- Refinement decisions
- Execution errors

### Integration with LangChain

The iterative refinement is also available in the LangChain integration, which uses a graph-based approach with explicit refinement nodes. See `langchain_integration.py` for the LangGraph implementation.

## Limitations

1. **Maximum rounds**: Limited to prevent infinite loops
1. **Cost limits**: Analysis stops if cost threshold exceeded
1. **Context window**: Very large notebooks may exceed LLM context limits
1. **Execution time**: Each round requires code execution, which takes time

## Future Improvements

Potential enhancements:

1. **Memory across sheets**: Learn from errors in one sheet to improve analysis of others
1. **Error pattern library**: Pre-built solutions for common Excel data issues
1. **Smarter inspection**: More sophisticated quality metrics
1. **Parallel refinement**: Refine multiple issues simultaneously
1. **User feedback loop**: Allow users to guide refinement direction
