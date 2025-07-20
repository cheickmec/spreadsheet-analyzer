# Quick Reference Guide

## CLI Usage

### Installation

```bash
# Install dependencies and CLI
uv sync
uv pip install -e .
```

### Basic Commands

```bash
# Analyze a single Excel file
spreadsheet-analyzer analyze financial-model.xlsx

# Verbose output with progress tracking
spreadsheet-analyzer -vv analyze data.xlsx

# Fast mode with JSON output
spreadsheet-analyzer analyze data.xlsx --mode fast --format json

# Save results to file
spreadsheet-analyzer analyze sensitive.xlsx -o results.yaml

# Skip specific analysis stages
spreadsheet-analyzer analyze large-file.xlsx --no-formulas
spreadsheet-analyzer analyze file.xlsx --no-security --no-content

# Get help
spreadsheet-analyzer --help
spreadsheet-analyzer analyze --help
```

### Output Formats

- `--format table` (default) - Rich terminal tables
- `--format json` - JSON output
- `--format yaml` - YAML output
- `--format markdown` - Markdown report

### Analysis Modes

- `--mode fast` - Quick analysis, basic checks only
- `--mode standard` (default) - Comprehensive analysis
- `--mode deep` - Deep analysis with AI insights

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_excel_parser.py

# Run tests matching a pattern
uv run pytest -k "circular_reference"

# Run with verbose output
uv run pytest -v

# Exclude problematic test files
uv run pytest --ignore=tests/test_base_stage.py --ignore=tests/test_strict_typing.py
```

### Test Options

- `-x` - Stop on first failure
- `-s` - Show print statements
- `--tb=short` - Shorter traceback format
- `--no-cov` - Skip coverage check
- `-k "pattern"` - Run tests matching pattern

### Common Test Patterns

```bash
# Run integration tests only
uv run pytest tests/integration/

# Run unit tests for a specific module
uv run pytest tests/test_formula_analysis.py

# Run tests with specific markers (when configured)
uv run pytest -m "not slow"

# Run tests in parallel (if pytest-xdist installed)
uv run pytest -n auto
```

## Development Workflow

### Before Committing

```bash
# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Fix common issues automatically
uv run ruff check --fix src/
uv run ruff format src/

# Type check
uv run mypy src/
```

### Quick Development Cycle

```bash
# 1. Make changes
# 2. Run relevant tests
uv run pytest tests/test_affected_module.py -x

# 3. Run pre-commit
uv run pre-commit run --all-files

# 4. Commit (if all passes)
git add -A && git commit -m "feat: your change description"
```

## Troubleshooting

### Common Issues

1. **Import Errors in Tests**

   ```bash
   # Some test files have import issues, exclude them:
   uv run pytest --ignore=tests/test_base_stage.py --ignore=tests/test_strict_typing.py
   ```

1. **Pre-commit Hook Failures**

   - Unicode characters: Replace × with x, ℹ with i
   - Magic numbers: Define as constants with `Final[type]`
   - Boolean parameters: Make them keyword-only with `*,`
   - Logging: Use `logger.exception()` in except blocks

1. **Coverage Failures**

   ```bash
   # Run without coverage check for quick testing
   uv run pytest --no-cov
   ```

## Example Analysis Session

```bash
# 1. Install and setup
uv sync
uv pip install -e .

# 2. Analyze a file with progress
spreadsheet-analyzer -v analyze test-files/financial-models/tesla_valuation_model.xlsx

# 3. Get detailed JSON output
spreadsheet-analyzer analyze test-files/financial-models/tesla_valuation_model.xlsx \
  --format json -o analysis_results.json

# 4. Quick analysis skipping expensive stages
spreadsheet-analyzer analyze large_workbook.xlsx \
  --mode fast --no-formulas --no-content
```
