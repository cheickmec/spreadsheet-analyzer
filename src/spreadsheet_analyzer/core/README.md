# Core Functional Programming Utilities

This package provides the foundational functional programming types and utilities used throughout the spreadsheet analyzer.

## Overview

The `core` package implements:

- **Result types** for safe error handling without exceptions
- **Option types** for nullable values
- **Functional utilities** for composition and transformation
- **Immutable configuration** system
- **Functional logging** wrapper

## Key Concepts

### Result Type

The `Result[T, E]` type represents operations that can fail:

```python
from spreadsheet_analyzer.core import Result, Ok, Err, ok, err

def divide(a: float, b: float) -> Result[float, str]:
    if b == 0:
        return err("Division by zero")
    return ok(a / b)

# Usage
result = divide(10, 2)
if result.is_ok():
    print(f"Result: {result.unwrap()}")  # Result: 5.0
else:
    print(f"Error: {result.unwrap_err()}")
```

### Option Type

The `Option[T]` type represents values that may or may not exist:

```python
from spreadsheet_analyzer.core import Option, Some, Nothing, some, nothing

def find_cell(sheet: dict, ref: str) -> Option[Any]:
    if ref in sheet:
        return some(sheet[ref])
    return nothing()

# Usage
cell = find_cell(sheet_data, "A1")
value = cell.unwrap_or("N/A")
```

### Function Composition

Compose functions for cleaner pipelines:

```python
from spreadsheet_analyzer.core import compose, pipe

# Right-to-left composition (mathematical)
process = compose(validate, transform, parse)
result = process(raw_data)  # parse -> transform -> validate

# Left-to-right pipeline
pipeline = pipe(parse, transform, validate)
result = pipeline(raw_data)  # parse -> transform -> validate
```

### Immutable Configuration

Configuration is immutable to prevent side effects:

```python
from spreadsheet_analyzer.core.config import default_config, with_excel_config

config = default_config()
# Create new config with changes (original unchanged)
new_config = with_excel_config(config, 
    config.excel.replace(max_file_size=50_000_000)
)
```

## Module Structure

- `types.py` - Core functional types (Result, Option, Either)
- `functional.py` - Function utilities (compose, pipe, curry, map/flatmap)
- `errors.py` - Immutable error types with categories
- `config.py` - Immutable configuration system
- `logging.py` - Functional logging wrapper

## Design Principles

1. **Immutability** - All data structures are frozen/immutable
1. **Pure Functions** - No side effects in business logic
1. **Explicit Error Handling** - Use Result types instead of exceptions
1. **Type Safety** - Comprehensive type annotations
1. **Composability** - Small functions that combine well

## Usage Examples

### Error Handling Chain

```python
from spreadsheet_analyzer.core import Result, ok, err
from spreadsheet_analyzer.core.functional import kleisli_result

def parse_reference(ref: str) -> Result[tuple[str, int], str]:
    # Parse "A1" -> ("A", 1)
    ...

def validate_bounds(ref: tuple[str, int]) -> Result[tuple[str, int], str]:
    # Check if reference is in valid range
    ...

# Compose operations
process_reference = kleisli_result(parse_reference, validate_bounds)
result = process_reference("A1")
```

### Functional Logging

```python
from spreadsheet_analyzer.core.logging import create_logger, info, error

logger = create_logger("my_module")

# Create log entries (pure functions)
entry = info("Processing started", file="data.xlsx")
logger.log(entry)  # Only this has side effects

# With context
from spreadsheet_analyzer.core.logging import LogContext, log_with_context

context = LogContext(
    operation="excel_analysis",
    file_path="data.xlsx",
    sheet_name="Sheet1"
)
log_with_context(logger, LogLevel.INFO, "Analysis complete", context)
```

## Best Practices

1. **Prefer Result over exceptions** for expected errors
1. **Use Option for nullable values** instead of None checks
1. **Compose small functions** rather than large procedures
1. **Keep functions pure** - isolate side effects
1. **Use frozen dataclasses** for data structures

## Testing

Pure functions are easy to test:

```python
def test_divide():
    assert divide(10, 2) == ok(5.0)
    assert divide(10, 0) == err("Division by zero")
```

No mocking needed for pure functions!
