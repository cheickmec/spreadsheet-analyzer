# Table Detection Benchmarking System

## Overview

This document-based benchmarking system tracks and evaluates the performance of different models on table detection tasks. It uses:

- **YAML files** for ground truth definitions (human-readable, version-controlled)
- **JSON files** for detection results (automatically logged)
- **Markdown reports** for performance analysis (human-readable summaries)

## Directory Structure

```
docs/benchmarks/
├── ground_truth/           # YAML ground truth definitions
│   ├── business_accounting.yaml
│   └── multi_table_sample.yaml
├── reports/               # Generated markdown reports
│   └── benchmark_report.md
└── README.md             # This file

outputs/benchmarks/
└── detection_results.json  # Automatically logged detection results
```

## How It Works

### 1. Ground Truth Definition

Ground truth is defined in YAML files under `docs/benchmarks/ground_truth/`. Each file specifies:

- Sheet metadata (name, index, dimensions)
- Expected table boundaries (exact coordinates)
- Table types and entity descriptions
- Notes about special cases

Example structure:

```yaml
sheets:
  - sheet_index: 2
    sheet_name: "Yiriden 2023 Loans"
    total_rows: 121
    total_cols: 14
    expected_tables:
      - table_id: "loan_details"
        start_row: 0
        end_row: 1
        start_col: 0
        end_col: 6
        table_type: "HEADER"
        entity_type: "loan_details"
```

### 2. Automatic Result Logging

Every time the table detector runs, results are automatically appended to `outputs/benchmarks/detection_results.json`:

```json
{
  "timestamp": "2025-01-07T10:21:00",
  "model": "gpt-4.1-nano",
  "excel_file": "multi_table_sample.xlsx",
  "sheet_index": 0,
  "tables_detected": [...],
  "token_usage": {...},
  "cost_usd": 0.0,
  "success": true
}
```

### 3. Report Generation

Run the benchmark report script to analyze results:

```bash
uv run python scripts/generate_benchmark_report.py
```

This generates `docs/benchmarks/reports/benchmark_report.md` with:

- Model performance comparison
- Sheet-by-sheet analysis
- Detection accuracy metrics
- Common failure patterns

## Evaluation Metrics

### Spatial Accuracy

- **IoU (Intersection over Union)**: Measures how well detected boundaries match ground truth
  - Threshold: 0.5 for considering a match
  - Perfect match: 1.0
- **Coordinate Deviation**: Average cell distance from ground truth

### Detection Quality

- **Precision**: Correctly detected tables / Total detected
- **Recall**: Correctly detected tables / Total ground truth tables
- **F1 Score**: Harmonic mean of precision and recall
- **Table Count Accuracy**: Exact match vs over/under detection

### Performance Metrics

- **Token Usage**: Average tokens per detection
- **Detection Time**: Seconds per sheet (when tracked)
- **Cost**: USD per detection
- **Success Rate**: Percentage of successful runs

## Usage

### Adding New Ground Truth

1. Create a YAML file in `docs/benchmarks/ground_truth/`
1. Define sheet structure and expected tables
1. Use 0-indexed coordinates

### Running Benchmarks

1. Run detections as normal - results are logged automatically:

```bash
uv run python -m spreadsheet_analyzer.notebook_cli \
  "path/to/excel.xlsx" \
  --model gpt-4.1-nano \
  --sheet-index 0 \
  --multi-table \
  --detector-only
```

2. Generate report:

```bash
uv run python scripts/generate_benchmark_report.py
```

3. View report in `docs/benchmarks/reports/benchmark_report.md`

### Interpreting Results

- **High Precision, Low Recall**: Model is conservative, missing some tables
- **Low Precision, High Recall**: Model over-detects, finding tables that don't exist
- **IoU < 0.5**: Poor boundary detection accuracy
- **IoU > 0.8**: Excellent boundary detection

## Common Issues and Solutions

### Over-Detection

Model detects more tables than expected:

- Often happens with summary rows
- Check if model is splitting single tables

### Under-Detection

Model misses tables:

- Common with side-by-side tables
- May treat multiple tables as one

### Boundary Accuracy

Low IoU scores indicate:

- Incorrect start/end row detection
- Missing columns in detection
- Including extra empty rows/columns

## Best Practices

1. **Consistent Ground Truth**: Ensure ground truth accurately reflects the actual Excel structure
1. **Multiple Runs**: Test each model multiple times to account for variability
1. **Diverse Test Cases**: Include various table layouts (single, side-by-side, with summaries)
1. **Regular Updates**: Update ground truth when Excel files change
1. **Model Comparison**: Test multiple models on the same sheets for fair comparison

## Future Enhancements

- [ ] Add detection timing tracking
- [ ] Create visualization of detected vs expected boundaries
- [ ] Add confidence score analysis
- [ ] Support for multiple Excel files in reports
- [ ] Automated testing pipeline
