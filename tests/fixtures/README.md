# Test Fixtures System

This directory contains expected outputs for test cases in JSON format. Each test file has its own subdirectory with JSON files for each test function's expected output.

## Structure

```
fixtures/
├── outputs/                      # Expected output fixtures
│   ├── test_parser/             # Outputs for test_parser.py
│   │   ├── test_basic_parse.json
│   │   ├── test_formula_extraction.json
│   │   └── test_large_file_handling.json
│   ├── test_analyzer/           # Outputs for test_analyzer.py
│   │   ├── test_pattern_detection.json
│   │   └── test_validation_chains.json
│   └── test_integration/        # Integration test outputs
│       └── test_full_pipeline.json
├── inputs/                      # Input test files
│   ├── simple.xlsx
│   ├── complex_formulas.xlsx
│   └── large_dataset.xlsx
└── schemas/                     # JSON schemas for validation
    ├── parser_output.schema.json
    └── analysis_result.schema.json
```

## Usage

### In Tests

```python
from tests.fixtures import load_expected_output, update_expected_output

def test_basic_parse():
    # Run the function
    result = parse_excel("simple.xlsx")
    
    # Load expected output
    expected = load_expected_output("test_parser", "test_basic_parse")
    
    # Compare
    assert result == expected
```

### Updating Fixtures

When output changes are intentional:

```bash
# Update all fixtures
pytest tests/ --update-fixtures

# Update specific test fixtures
pytest tests/test_parser.py::test_basic_parse --update-fixtures

# Review changes
git diff tests/fixtures/outputs/
```

## Benefits

1. **Reviewable Changes**: Output changes appear in PRs as JSON diffs
1. **Regression Detection**: Unintended changes are caught immediately
1. **Documentation**: Expected outputs serve as documentation
1. **Debugging**: Easy to see what the expected output should be
1. **Performance**: No need to recompute expected values in tests
