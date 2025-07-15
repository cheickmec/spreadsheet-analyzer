# Spreadsheet Analyzer Scripts

This directory contains utility scripts for analyzing Excel files with the deterministic pipeline.

## Available Scripts

### analyze_excel.py

Analyze a single Excel file with detailed or summary output.

```bash
# Basic usage
uv run python scripts/analyze_excel.py test.xlsx

# With strict mode (fail on any security risk)
uv run python scripts/analyze_excel.py test.xlsx --mode strict

# Fast mode (minimal checks)
uv run python scripts/analyze_excel.py test.xlsx --mode fast

# Show detailed results
uv run python scripts/analyze_excel.py test.xlsx --detailed
```

### batch_analyze.py

Batch analyze multiple Excel files in a directory.

```bash
# Analyze all Excel files in test-files/
uv run python scripts/batch_analyze.py

# Analyze specific directory
uv run python scripts/batch_analyze.py /path/to/excel/files

# Include subdirectories
uv run python scripts/batch_analyze.py --recursive

# Summary only (no individual file output)
uv run python scripts/batch_analyze.py --summary-only
```

### run_test_suite.py

Run comprehensive test suite with detailed reporting.

```bash
# Run full test suite on test-files directory
uv run python scripts/run_test_suite.py

# Generates timestamped JSON report with:
# - Detailed results for each file
# - Success/failure statistics
# - Security threats found
# - Data quality insights
# - Performance metrics
```

## Usage Examples

### Quick Analysis

```bash
# Analyze a single file quickly
uv run python scripts/analyze_excel.py "Business Accounting.xlsx" --mode fast
```

### Security Audit

```bash
# Strict security check on all files
uv run python scripts/batch_analyze.py /path/to/files --mode strict
```

### Full Test Report

```bash
# Generate comprehensive test report
uv run python scripts/run_test_suite.py
# Report saved to: test_results_YYYYMMDD_HHMMSS.json
```

## Notes

- All scripts use the lenient pipeline mode by default
- Scripts automatically add the src directory to Python path for development
- Excel file extensions supported: .xlsx, .xls, .xlsm, .xlsb
- Files that are actually HTML with Excel extensions will be properly blocked
