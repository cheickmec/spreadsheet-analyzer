# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for building an intelligent Excel file analyzer system using a hybrid approach of deterministic parsing and AI-powered insights. The system is designed to analyze complex spreadsheets, revealing hidden structures, relationships, and potential issues.

## Critical uv Usage Requirements

**IMPORTANT**: All Python scripts in this repository must be run using `uv run <path to script>`.

Examples:

- ✅ `uv run src/spreadsheet_analyzer/main.py`
- ✅ `uv run tools/html_to_markdown_converter.py input.html`
- ✅ `uv run pytest tests/`
- ❌ `uv run python src/spreadsheet_analyzer/main.py` (NEVER do this)
- ❌ `python src/spreadsheet_analyzer/main.py` (NEVER do this)

## Critical Git Commit Requirements

**IMPORTANT**: NEVER skip pre-commit hooks when committing. All commits MUST pass pre-commit checks.

- ❌ NEVER use `git commit --no-verify`
- ❌ NEVER use `git commit -n`
- ✅ ALWAYS ensure all pre-commit hooks pass before committing
- ✅ Fix all linting, type checking, and security issues before committing

If pre-commit hooks fail, fix the issues first, then commit.

## Anchor Comments System

Use these standardized comment anchors to provide essential context for future development:

### Excel-Specific Anchors

```python
# CLAUDE-KNOWLEDGE: openpyxl loads entire workbook into memory - use read_only mode for large files
# CLAUDE-GOTCHA: Excel formulas return cached values unless data_only=False
# CLAUDE-COMPLEX: This recursive sheet analysis handles circular references by maintaining visited set
# CLAUDE-IMPORTANT: Never modify original Excel file - always work on copies
# CLAUDE-TEST-WORKAROUND: Mock Excel file objects cause issues with openpyxl internals
# CLAUDE-SECURITY: Validate all macro content before execution - Excel files can contain malicious VBA
# CLAUDE-PERFORMANCE: Sheet iteration is O(n*m) - consider chunking for sheets > 10K cells
```

### General Anchors

- **CLAUDE-KNOWLEDGE**: Domain-specific knowledge that isn't obvious from code
- **CLAUDE-GOTCHA**: Non-obvious behavior or common pitfalls
- **CLAUDE-COMPLEX**: Explanation for necessarily complex logic
- **CLAUDE-IMPORTANT**: Critical business rules or constraints
- **CLAUDE-TEST-WORKAROUND**: Testing limitations and their solutions
- **CLAUDE-SECURITY**: Security considerations and requirements
- **CLAUDE-PERFORMANCE**: Performance implications and optimization notes

## Code Generation Patterns

### F-String Escaping in Code Generation

When generating Python code that contains f-strings, proper brace escaping is critical. This applies particularly to task-based code generation for notebook cells.

#### Key Principle

- **Single braces `{}`**: For variables available during code generation (function parameters)
- **Double braces `{{}}`**: For variables that will exist in the generated code

#### Examples

```python
# CLAUDE-GOTCHA: F-strings within f-strings require careful escaping
def _generate_load_code(self, file_path: str, sheet_name: str) -> str:
    """Generate code to load spreadsheet data."""
    return f"""
# Load Excel data
file_path = r"{file_path}"  # ✅ Single braces - file_path is a parameter
sheet_name = "{sheet_name}"  # ✅ Single braces - sheet_name is a parameter

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # ❌ WRONG: print(f"Loaded {len(df)} rows")  # NameError: 'df' not defined during generation
    # ✅ RIGHT: Escape braces for runtime variables
    print(f"Loaded {{len(df)}} rows and {{df.shape[1]}} columns")  
    print(f"Shape: {{df.shape}}")  # ✅ df will exist when generated code runs
except Exception as e:
    print(f"Error: {{e}}")  # ✅ e will exist in the except block
"""
```

#### Common Patterns

1. **DataFrame operations in generated code**:

   ```python
   # In code generation function:
   return f"""
   print(f"DataFrame shape: {{df.shape}}")
   print(f"Columns: {{', '.join(df.columns)}}")
   print(f"Missing values: {{df.isnull().sum().sum()}}")
   """
   ```

1. **Dictionary literals in generated f-strings**:

   ```python
   # In code generation function:
   return f"""
   summary = pd.DataFrame({{
       'Column': df.columns,
       'Missing': df.isnull().sum()
   }})
   """
   ```

1. **Loop variables and cell references**:

   ```python
   # In code generation function:
   return f"""
   for row in worksheet.iter_rows():
       for cell in row:
           # Use string concatenation or format() for complex expressions
           cell_ref = f"{{get_column_letter(cell.column)}}{{cell.row}}"
           print(f"Cell {{cell_ref}}: {{cell.value}}")
   """
   ```

#### Debugging Tips

1. **NameError during generation**: Variable needs double braces `{{}}`
1. **SyntaxError with f-strings**: Check for mismatched braces or quotes
1. **ValueError: Invalid format specifier**: Complex expressions may need alternative formatting

#### Best Practices

1. Test code generation by running the generation function and inspecting output
1. Use string concatenation or `.format()` for very complex expressions
1. Keep generated f-strings simple - extract complex logic to variables first
1. Add comments in generated code to clarify what executes when
