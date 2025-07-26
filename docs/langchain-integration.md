# LangChain/LangGraph Integration

This document describes the new LangChain/LangGraph integration that replaces the custom LLM provider interface and enables multi-step analysis workflows.

## Overview

The integration provides:

1. **Unified LLM Interface**: Replace custom provider code with LangChain's `init_chat_model()`
1. **Workflow Orchestration**: Use LangGraph's `StateGraph` for complex analysis workflows
1. **Iterative Refinement**: Support multiple LLM interactions with error recovery
1. **Production Features**: Checkpointing, tracing, and observability via LangSmith

## Architecture

### State Management

All workflow state is managed through a typed `SheetState` dictionary:

```python
class SheetState(TypedDict, total=False):
    # Input
    excel_path: Path
    sheet_name: str
    skip_deterministic: bool
    provider: str
    model: str
    temperature: float

    # Intermediate results
    deterministic: dict
    notebook_json: dict
    llm_response: str
    notebook_final: dict

    # Tracking
    messages: list[BaseMessage]
    execution_errors: list[str]
    token_usage: dict[str, int]

    # Output
    output_path: Path
    metadata_path: Path
    
    # Performance tracking
    file_size_mb: float
    is_large_file: bool
```

### Workflow Nodes

The analysis workflow consists of 7 nodes:

1. **deterministic**: Run deterministic analysis (structure, formulas, security)
1. **create_notebook**: Create initial notebook scaffold
1. **execute_initial**: Execute data loading cells
1. **llm_analysis**: Analyze with LLM using LCEL chains
1. **execute_llm**: Execute LLM-generated code
1. **refine**: Refine analysis on errors (conditional)
1. **save_results**: Save notebook and metadata

### Error Recovery

The workflow includes automatic error recovery:

- If LLM code execution fails, the `refine` node is triggered
- Up to 3 refinement attempts before giving up
- All errors are tracked in state

## Usage

### Command Line

```bash
# Basic usage with LangChain
analyze-sheet-langchain file.xlsx "Sheet1" --use-langchain

# With different provider
analyze-sheet-langchain file.xlsx "Sheet1" --use-langchain --provider openai

# Enable LangSmith tracing
export LANGSMITH_API_KEY="your-key"
analyze-sheet-langchain file.xlsx "Sheet1" --use-langchain --enable-langsmith

# Skip deterministic analysis
analyze-sheet-langchain file.xlsx "Sheet1" --use-langchain --skip-deterministic
```

### Programmatic Usage

```python
from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain
)

# Run analysis
final_state = await analyze_sheet_with_langchain(
    excel_path=Path("data.xlsx"),
    sheet_name="Sheet1",
    skip_deterministic=False,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.1,
    enable_tracing=True,
)

# Access results
print(f"Output: {final_state['output_path']}")
print(f"Errors: {final_state.get('execution_errors', [])}")
print(f"Tokens used: {final_state.get('token_usage', {})}")
```

## Performance Optimizations

### Memory-Efficient Excel Loading

The integration includes several optimizations for handling large Excel files:

1. **Read-Only Mode**: Uses `openpyxl` with `read_only=True` to reduce memory usage
1. **Async I/O**: Wraps blocking operations in `asyncio.to_thread()` to keep the event loop responsive
1. **NA Filter Disabled**: Sets `na_filter=False` in pandas for faster loading
1. **File Size Detection**: Automatically detects large files and applies appropriate optimizations

```python
# Example of optimized loading
df = await asyncio.to_thread(
    pd.read_excel,
    excel_path,
    sheet_name=sheet_name,
    header=None,
    na_filter=False,  # Faster loading
    engine='openpyxl'
)
```

### Large File Handling

Files larger than 50MB are automatically detected and handled with special care:

- Memory usage warnings in the notebook
- Suggestions for chunked processing
- Performance metrics tracking

## Key Benefits

### 1. Provider Abstraction

Before:

```python
provider = get_provider("anthropic", model="claude-3-5-sonnet")
response = provider.complete(messages)
```

After:

```python
llm = init_chat_model("claude-3-5-sonnet", model_provider="anthropic")
response = await llm.ainvoke(messages)
```

### 2. Workflow Visibility

- Each node execution is tracked
- State transitions are explicit
- Easy to add new nodes or modify flow

### 3. Error Handling

- Automatic retries with refinement
- Error accumulation in state
- Graceful degradation

### 4. Observability

With LangSmith enabled:

- Token usage per node
- Execution timings
- Full trace of LLM calls
- State evolution tracking

## Migration from Original

The new implementation is backward compatible:

```bash
# Original (single LLM call)
analyze-sheet file.xlsx "Sheet1"

# New (multi-step with LangChain)
analyze-sheet-langchain file.xlsx "Sheet1" --use-langchain
```

When `--use-langchain` is not specified, the original implementation is used.

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Required for Anthropic models
- `OPENAI_API_KEY`: Required for OpenAI models
- `LANGCHAIN_TRACING_V2`: Set to "true" to enable tracing
- `LANGSMITH_API_KEY`: Required for LangSmith tracing
- `LANGCHAIN_PROJECT`: Project name for LangSmith (default: "default")

### Checkpointing

The workflow uses in-memory checkpointing by default. For production:

```python
# Use SQLite for persistence
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
```

## Extending the Workflow

### Adding New Nodes

```python
async def custom_analysis_node(state: SheetState) -> dict[str, Any]:
    """Custom analysis step."""
    # Access state
    notebook = NotebookDocument(**state["notebook_json"])
    
    # Perform analysis
    result = await custom_analysis(notebook)
    
    # Return state updates
    return {"custom_result": result}

# Add to graph
builder.add_node("custom_analysis", custom_analysis_node)
builder.add_edge("llm_analysis", "custom_analysis")
builder.add_edge("custom_analysis", "execute_llm")
```

### Custom LLM Chains

```python
from langchain_core.prompts import ChatPromptTemplate

# Create custom prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data analysis expert."),
    ("human", "{context}\n\nAnalyze this data: {query}")
])

# Build chain
chain = prompt | llm | StrOutputParser()

# Use in node
async def analysis_node(state: SheetState) -> dict[str, Any]:
    result = await chain.ainvoke({
        "context": state["notebook_json"],
        "query": "Find patterns in the data"
    })
    return {"analysis": result}
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure dependencies are installed: `uv sync`
1. **API Key Error**: Set appropriate environment variables
1. **Tracing Not Working**: Check LANGSMITH_API_KEY is set
1. **Memory Issues**: Use read_only mode for large Excel files

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger("spreadsheet_analyzer").setLevel(logging.DEBUG)
```

### Performance Tips

1. Skip deterministic analysis for simple files: `--skip-deterministic`
1. Use appropriate temperature (0.1 for consistency, 0.7 for creativity)
1. Monitor token usage via LangSmith or state output
1. Consider caching for repeated analyses

## Future Enhancements

1. **Streaming**: Stream LLM responses for better UX
1. **Parallel Execution**: Process multiple sheets concurrently
1. **Custom Tools**: Add tool calling for specialized analysis
1. **Cloud Checkpoints**: S3/GCS checkpoint storage
1. **Web UI**: Gradio/Streamlit interface for the workflow
