# Test Fixtures

This directory contains test fixtures for the Spreadsheet Analyzer project in a language-agnostic JSON format.

## Overview

The fixtures capture outputs from processing Excel test files, enabling:

1. **Regression Testing**: Ensure analyzer behavior remains consistent
1. **Cross-Language Support**: Clean JSON consumable by any language
1. **Visibility**: See what each test file produces
1. **Type Safety**: Python code can reconstruct typed dataclasses

## Structure

```
fixtures/
├── captured_outputs/       # Outputs from test Excel files
│   ├── business-accounting/
│   ├── data-analysis/
│   └── ...
├── schemas/               # JSON schemas for validation
│   └── pipeline_result.schema.json
└── README.md             # This file
```

## Quick Start

### Capture Outputs

```bash
# Capture all test files
uv run scripts/capture_test_outputs.py

# Force update all fixtures
uv run scripts/capture_test_outputs.py --force
```

### Explore Outputs

```bash
# List available fixtures
uv run scripts/explore_test_output.py --list

# Explore specific file output
uv run scripts/explore_test_output.py "business-accounting/Business Accounting.xlsx"

# Visualize all fixtures
uv run scripts/visualize_fixtures.py list --details
```

## Language-Agnostic Design

Fixtures are stored as clean JSON without Python-specific type markers:

```json
{
  "metadata": {
    "test_file": "category/file.xlsx",
    "file_size": 167763,
    "processing_time": 0.448,
    "pipeline_success": true
  },
  "pipeline_result": {
    "success": true,
    "execution_time": 0.445,
    "errors": [],
    "structure": {
      "sheet_count": 10,
      "total_cells": 6959,
      "complexity_score": 32
    }
    // ... other stages
  }
}
```

## Usage in Tests

### Any Language

```javascript
// JavaScript
const fixture = JSON.parse(fs.readFileSync('fixture.json'));
console.log(fixture.pipeline_result.structure.sheet_count);
```

### Python with Type Safety

```python
from spreadsheet_analyzer.testing.loader import FixtureLoader

loader = FixtureLoader()

# Load as raw dict (like any language)
raw = loader.load_raw("business-accounting/Business Accounting.xlsx")

# Or as typed dataclasses
result = loader.load_as_dataclass("business-accounting/Business Accounting.xlsx")
print(result.structure.sheet_count)  # IDE knows this is int
```

## Test Example

See `tests/test_against_captured_fixtures.py` for a complete example of using fixtures in tests.

## Benefits

- **No Lock-in**: Not tied to Python serialization
- **Portable**: Same fixtures work across languages
- **Debuggable**: Human-readable JSON
- **Future-Proof**: Ready for serverless/Lambda deployments
