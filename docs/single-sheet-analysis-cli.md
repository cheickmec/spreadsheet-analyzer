# Single Sheet Analysis CLI

The `analyze_single_sheet.py` script provides a command-line interface for analyzing individual Excel sheets using LLM-powered analysis.

## Quick Start

```bash
# Analyze a specific sheet
uv run analyze_single_sheet.py "path/to/file.xlsx" "Sheet Name"

# List all sheets in an Excel file
uv run analyze_single_sheet.py "path/to/file.xlsx" --list-sheets
```

## Features

- **One notebook per sheet**: Each sheet gets its own dedicated Jupyter notebook
- **One agent per notebook**: Focused analysis with a dedicated LLM agent per sheet
- **No inlined data**: Reads Excel files directly with pandas (no data duplication)
- **Organized output**: Results saved to `analysis_results/[excel_file]/[sheet_name].ipynb`

## Command Syntax

```bash
uv run analyze_single_sheet.py <excel_file> [sheet_name] [options]
```

### Positional Arguments

- `excel_file`: Path to the Excel file to analyze (required)
- `sheet_name`: Name of the sheet to analyze (required unless using `--list-sheets`)

### Options

- `--list-sheets`, `-l`: List all available sheets in the Excel file and exit
- `--model MODEL`, `-m MODEL`: LLM model to use (default: claude-sonnet-4-20250514)
- `--strategy {basic,hierarchical,detailed}`, `-s`: Analysis strategy (default: hierarchical)
- `--output-dir OUTPUT_DIR`, `-o`: Output directory for results (default: analysis_results)
- `--skip-deterministic`: Skip the deterministic analysis phase
- `--verbose`, `-v`: Enable verbose output for debugging
- `--help`, `-h`: Show help message

## Examples

### List available sheets

```bash
uv run analyze_single_sheet.py "test-files/business-accounting/Business Accounting.xlsx" -l
```

### Analyze a specific sheet

```bash
uv run analyze_single_sheet.py "test-files/business-accounting/Business Accounting.xlsx" "Truck Revenue Projections"
```

### Use a different model

```bash
uv run analyze_single_sheet.py "data.xlsx" "Sheet1" --model claude-opus-4-20250514
```

### Use detailed analysis strategy

```bash
uv run analyze_single_sheet.py "data.xlsx" "Sheet1" --strategy detailed
```

### Skip deterministic analysis for faster results

```bash
uv run analyze_single_sheet.py "data.xlsx" "Sheet1" --skip-deterministic
```

### Custom output directory

```bash
uv run analyze_single_sheet.py "data.xlsx" "Sheet1" -o "my_analysis"
```

## Output Structure

The analysis creates the following output structure:

```
analysis_results/
└── [excel_file_name]/
    ├── [sheet_name].ipynb          # Jupyter notebook with analysis
    └── [sheet_name]_metadata.json  # Analysis metadata and LLM results
```

For example, analyzing "Truck Revenue Projections" from "Business Accounting.xlsx":

```
analysis_results/
└── Business Accounting/
    ├── truck_revenue_projections.ipynb
    └── truck_revenue_projections_metadata.json
```

## Prerequisites

1. **Environment Variable**: Set the appropriate API key for your chosen model:

   ```bash
   # For Claude models (default)
   export ANTHROPIC_API_KEY='your-api-key'

   # For GPT models
   export OPENAI_API_KEY='your-api-key'

   # For Gemini models
   export GEMINI_API_KEY='your-api-key'
   ```

1. **Dependencies**: Install project dependencies:

   ```bash
   uv sync --dev
   ```

## Analysis Process

The tool performs a multi-step analysis:

1. **Deterministic Analysis** (optional): Analyzes the entire Excel file for structure, formulas, and security
1. **Notebook Creation**: Creates a Jupyter notebook with cells to load and explore the sheet
1. **LLM Analysis**: Uses the selected strategy to compress context and analyze the sheet
1. **Results Saving**: Saves the notebook and metadata to the output directory

## Strategies

- **basic**: Simple analysis with minimal context compression
- **hierarchical**: Hierarchical summarization for better token efficiency (default)
- **detailed**: More comprehensive analysis with additional context

## Troubleshooting

- **"Sheet not found"**: Use `--list-sheets` to see available sheet names (case-sensitive)
- **"API_KEY not set"**: Export the appropriate API key as shown in Prerequisites
- **LLM errors**: Use `--verbose` to see detailed error messages
- **Memory issues**: For large files, consider using `--skip-deterministic`
