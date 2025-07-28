# Notebook Tools Interface

A pragmatic, functional approach to notebook interaction with LLM tool calling capabilities.

## Reading Order

Read these modules in order to understand the architecture:

### 1. `src/spreadsheet_analyzer/notebook_tools.py`

**Core functional interfaces**

- `CellOutput`, `CellExecution`, `NotebookState` - Immutable data structures
- `CellType` - Enum for code/markdown/raw cells
- `NotebookToolkit` - Main toolkit for notebook operations
- `create_toolkit()` - Factory function for creating toolkits
- Uses your existing `core_exec` modules under the hood

### 2. `src/spreadsheet_analyzer/notebook_session.py`

**Session management**

- `NotebookSession` - Manages session lifecycle
- `notebook_session()` - Async context manager
- `SessionManager` - Manages multiple sessions
- Proper async/await patterns with Result types

### 3. `src/spreadsheet_analyzer/notebook_llm_interface.py`

**LangChain integration**

- Tool definitions for LLM function calling
- Input/output models for each tool
- Session management integration
- Tool descriptions for LLM prompting

### 4. `examples/demo_notebook_tools.py`

**Entry point and demonstrations**

- Complete working examples
- Different cell type demonstrations
- LLM tool interface showcase

## Key Features

### ✅ **Cell Type Support**

- **Code cells**: Execute Python code and get outputs
- **Markdown cells**: Render formatted documentation
- **Raw cells**: Store unprocessed content

### ✅ **LLM Tool Set**

- `execute_code` - Run new Python code
- `edit_and_execute` - Fix and run existing code (most common)
- `add_cell` - Add any cell type
- `add_markdown_cell` - Quick markdown addition
- `delete_cell` - Remove unwanted cells
- `read_cell` - Inspect cell content
- `get_notebook_state` - Overall status
- `save_notebook` - Persist work

### ✅ **Session Management**

- Single kernel per session (prevents LLM chaos)
- Proper async context management
- Global session manager for tool access

### ✅ **Functional Design**

- Immutable data structures
- Result types for error handling
- Pure functions where possible
- Composable operations

## Usage Examples

### Basic Session

```python
async with notebook_session(session_id="my_session") as session:
    result = await session.toolkit.execute_code("print('Hello!')")
    if isinstance(result, Ok):
        print(f"Executed: {result.value.cell_id}")
```

### Different Cell Types

```python
# Code cell (executes)
await session.toolkit.execute_code("x = 42")

# Markdown cell (renders)
await session.toolkit.render_markdown("# Title\nContent")

# Raw cell (stores)
await session.toolkit.add_raw_cell("Raw content")
```

### LLM Integration

```python
tools = get_notebook_tools()
# Present these tools to your LLM for function calling
```

## Design Principles

1. **Pragmatic**: Built for real-world usage patterns
1. **Functional**: Immutable data, pure functions, Result types
1. **Incremental**: Built on existing `core_exec` modules
1. **LLM-friendly**: Tools designed for function calling patterns
1. **Safe**: Single kernel per session, proper error handling

## Next Steps

The interface is ready for LLM integration. The tools follow common notebook workflow patterns and provide the LLM with the control it needs while maintaining safety and predictability.
