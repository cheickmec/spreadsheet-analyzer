# Tools System for LangChain Integration

This package provides a functional wrapper system for LangChain tools used by agents.

## Overview

The `tools` package implements:

- **Tool protocols** defining tool interfaces
- **Functional tool wrappers** for pure function tools
- **Tool composition** for chaining and conditional execution
- **Tool registry** for dynamic tool discovery
- **LangChain adapters** for integration

## Core Concepts

### Tool Definition

Tools are functions that agents can call:

```python
from spreadsheet_analyzer.tools import create_tool, ToolMetadata
from spreadsheet_analyzer.core import Result, ok, err
from pydantic import BaseModel

# Define input schema
class ExcelReadArgs(BaseModel):
    file_path: str
    sheet_name: str = "Sheet1"
    range: str | None = None

# Create tool as pure function
def read_excel_impl(args: ExcelReadArgs) -> Result[pd.DataFrame, ToolError]:
    try:
        df = pd.read_excel(args.file_path, sheet_name=args.sheet_name)
        if args.range:
            df = df.loc[args.range]
        return ok(df)
    except Exception as e:
        return err(ToolError(f"Failed to read Excel: {e}"))

# Create tool with metadata
read_excel_tool = create_tool(
    name="read_excel",
    description="Read data from an Excel file",
    args_schema=ExcelReadArgs,
    execute_fn=read_excel_impl,
    category="excel",
    tags=["io", "data"]
)
```

### Tool Composition

Chain tools together:

```python
from spreadsheet_analyzer.tools import ToolChain

# Create a pipeline of tools
analysis_pipeline = ToolChain(tools=(
    read_excel_tool,
    validate_data_tool,
    analyze_formulas_tool,
    generate_report_tool
))

# Execute pipeline
result = analysis_pipeline.execute(
    ExcelReadArgs(file_path="data.xlsx")
)
```

### Conditional Tools

Execute tools based on conditions:

```python
from spreadsheet_analyzer.tools import ToolCondition

# Different analysis based on file size
conditional_analysis = ToolCondition(
    condition=lambda args: args.file_size < 10_000_000,
    if_true=quick_analysis_tool,
    if_false=chunked_analysis_tool
)
```

## Tool Categories

### Excel Tools

Tools for Excel file manipulation:

```python
# Read Excel data
read_excel = create_tool(...)

# Write Excel data  
write_excel = create_tool(...)

# Analyze formulas
analyze_formulas = create_tool(...)

# Detect tables
detect_tables = create_tool(...)
```

### Notebook Tools

Tools for notebook operations:

```python
# Execute code in notebook
execute_code = create_tool(
    name="execute_code",
    description="Execute Python code in notebook",
    args_schema=CodeExecutionArgs,
    execute_fn=execute_code_impl,
    category="notebook"
)

# Add markdown cell
add_markdown = create_tool(...)

# Save notebook
save_notebook = create_tool(...)
```

### Analysis Tools

Tools for data analysis:

```python
# Statistical analysis
analyze_statistics = create_tool(...)

# Pattern detection
detect_patterns = create_tool(...)

# Anomaly detection
find_anomalies = create_tool(...)
```

## Tool Registry

Register and discover tools dynamically:

```python
from spreadsheet_analyzer.tools import ToolRegistry

# Create registry
registry = ToolRegistry()

# Register tools
registry.register(read_excel_tool)
registry.register(analyze_formulas_tool)

# Discover tools
excel_tools = registry.list_tools(category="excel")
all_tools = registry.list_tools()

# Search tools
formula_tools = registry.search("formula")

# Get specific tool
tool = registry.get("read_excel")
if tool.is_some():
    result = tool.unwrap().execute(args)
```

## LangChain Integration

Convert functional tools to LangChain format:

```python
# Convert to LangChain tool
langchain_tool = read_excel_tool.to_langchain()

# Use with LangChain agent
from langchain.agents import create_structured_chat_agent

tools = [
    read_excel_tool.to_langchain(),
    analyze_formulas_tool.to_langchain(),
    write_report_tool.to_langchain()
]

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
```

## Creating Custom Tools

### Step 1: Define Arguments

```python
from pydantic import BaseModel, Field

class FormulaValidationArgs(BaseModel):
    """Arguments for formula validation tool."""
    
    formula: str = Field(description="Excel formula to validate")
    context: dict = Field(
        default_factory=dict,
        description="Cell context for references"
    )
    strict: bool = Field(
        default=True,
        description="Use strict validation rules"
    )
```

### Step 2: Implement Logic

```python
def validate_formula_impl(args: FormulaValidationArgs) -> Result[ValidationResult, ToolError]:
    """Pure function implementation."""
    
    # Parse formula
    parsed = parse_formula(args.formula)
    if parsed.is_err():
        return err(ToolError(f"Invalid syntax: {parsed.unwrap_err()}"))
    
    # Check references
    ast = parsed.unwrap()
    invalid_refs = check_references(ast, args.context)
    
    if invalid_refs and args.strict:
        return err(ToolError(f"Invalid references: {invalid_refs}"))
    
    # Return validation result
    return ok(ValidationResult(
        valid=True,
        warnings=invalid_refs if not args.strict else [],
        ast=ast
    ))
```

### Step 3: Create Tool

```python
validate_formula = create_tool(
    name="validate_formula",
    description="Validate Excel formula syntax and references",
    args_schema=FormulaValidationArgs,
    execute_fn=validate_formula_impl,
    category="excel",
    tags=["validation", "formulas"],
    requires=["formula_parser"]  # Required capabilities
)
```

## Tool Testing

Test tools in isolation:

```python
def test_validate_formula():
    # Test valid formula
    args = FormulaValidationArgs(
        formula="=SUM(A1:A10)",
        context={"A1": 10, "A10": 20}
    )
    result = validate_formula.execute(args)
    assert result.is_ok()
    
    # Test invalid formula
    args = FormulaValidationArgs(formula="=SUM(")
    result = validate_formula.execute(args)
    assert result.is_err()
```

## Best Practices

1. **Keep tools focused** - Single responsibility per tool
1. **Use pure functions** - No side effects in tool logic
1. **Validate inputs** - Use Pydantic for arg validation
1. **Handle errors gracefully** - Return Result types
1. **Document thoroughly** - Clear descriptions and examples

## Performance Considerations

- **Lazy loading** - Load heavy dependencies only when needed
- **Caching** - Cache expensive computations
- **Streaming** - Support streaming for large data
- **Async support** - Async variants for I/O tools

## Future Enhancements

- **Tool versioning** for compatibility
- **Tool composition DSL** for complex pipelines
- **Automatic tool generation** from function signatures
- **Cross-language tools** via RPC
- **Tool marketplace** for sharing
