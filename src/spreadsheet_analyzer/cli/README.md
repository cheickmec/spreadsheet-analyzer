# Command-Line Interface Package

This package provides CLI interfaces for the spreadsheet analyzer, organized functionally.

## Overview

The `cli` package will contain:

- **Notebook CLI** - LLM-powered analysis interface
- **Pipeline CLI** - Deterministic analysis interface
- **Utility functions** - Pure functions for CLI operations

## Migration Status

**Current State**: This package provides a compatibility layer during migration.

**Phase 2 Goals**:

- Extract `StructuredFileNameGenerator` as pure functions
- Extract `PipelineResultsToMarkdown` as pure functions
- Create functional argument parsing
- Separate I/O from business logic

## Planned Structure

```
cli/
├── __init__.py          # Compatibility layer
├── notebook.py          # Notebook CLI (main entry point)
├── pipeline.py          # Pipeline CLI
├── utils/
│   ├── __init__.py
│   ├── naming.py        # File naming functions
│   ├── markdown.py      # Markdown generation
│   └── parsers.py       # Argument parsing functions
└── README.md
```

## Functional Design

### File Naming (utils/naming.py)

Pure functions for generating structured file names:

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class FileNameConfig:
    """Immutable configuration for file naming."""
    excel_file: Path
    model: str
    sheet_index: int
    sheet_name: str | None = None
    max_rounds: int = 5
    timestamp: datetime | None = None

def generate_notebook_name(config: FileNameConfig) -> str:
    """Generate notebook filename - pure function."""
    parts = [
        config.excel_file.stem,
        f"sheet{config.sheet_index}",
        sanitize_sheet_name(config.sheet_name) if config.sheet_name else None,
        sanitize_model_name(config.model),
        f"r{config.max_rounds}"
    ]
    
    # Filter None values and join
    name_parts = [p for p in parts if p]
    return f"{'_'.join(name_parts)}.ipynb"

def sanitize_model_name(model: str) -> str:
    """Sanitize model name for filenames - pure function."""
    # Logic extracted from StructuredFileNameGenerator
    ...
```

### Markdown Generation (utils/markdown.py)

Pure functions for converting results to markdown:

```python
from spreadsheet_analyzer.pipeline.types import PipelineResult
from spreadsheet_analyzer.core import Result

def pipeline_to_markdown(result: PipelineResult) -> list[str]:
    """Convert pipeline results to markdown cells - pure function."""
    cells = []
    
    cells.append(create_header(result))
    cells.append(integrity_to_markdown(result.integrity))
    
    if should_show_security(result.security):
        cells.append(security_to_markdown(result.security))
    
    cells.append(structure_to_markdown(result.structure))
    
    return [cell for cell in cells if cell]  # Filter empty

def create_header(result: PipelineResult) -> str:
    """Create header markdown - pure function."""
    return f"""# 📊 Excel Analysis Report

**File:** `{result.context.file_path.name}`
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
```

### Argument Parsing (utils/parsers.py)

Functional approach to argument parsing:

```python
from dataclasses import dataclass
from pathlib import Path
from spreadsheet_analyzer.core import Result, ok, err

@dataclass(frozen=True)
class CLIArguments:
    """Immutable CLI arguments."""
    excel_file: Path
    model: str = "claude-3-5-sonnet-20241022"
    max_rounds: int = 5
    sheet_index: int = 0
    output_dir: Path | None = None
    # ... other args

def parse_arguments(args: list[str]) -> Result[CLIArguments, str]:
    """Parse CLI arguments - pure function."""
    parser = create_argument_parser()
    
    try:
        namespace = parser.parse_args(args)
        
        # Validate arguments
        validation = validate_arguments(namespace)
        if validation.is_err():
            return validation
        
        # Convert to immutable dataclass
        return ok(CLIArguments(
            excel_file=namespace.excel_file,
            model=namespace.model,
            max_rounds=namespace.max_rounds,
            # ...
        ))
    except SystemExit:
        return err("Invalid arguments")

def validate_arguments(args) -> Result[None, str]:
    """Validate parsed arguments - pure function."""
    if not args.excel_file.exists():
        return err(f"File not found: {args.excel_file}")
    
    if args.max_rounds < 1:
        return err("Max rounds must be positive")
    
    return ok(None)
```

## Compatibility Layer

During migration, the package provides backwards compatibility:

```python
# cli/__init__.py
def __getattr__(name: str) -> Any:
    """Provide compatibility for old imports."""
    if name == "NotebookCLI":
        warnings.warn(
            "Importing NotebookCLI from cli package is deprecated.",
            DeprecationWarning
        )
        # Import from old location during migration
        from spreadsheet_analyzer.notebook_cli import NotebookCLI
        return NotebookCLI
```

## Future CLI Structure

### Main Entry Point (notebook.py)

```python
from spreadsheet_analyzer.core import Result
from .utils.parsers import parse_arguments
from .utils.naming import FileNameConfig, generate_notebook_name

def main(args: list[str] | None = None) -> int:
    """Main entry point - handles I/O."""
    # Parse arguments (pure)
    result = parse_arguments(args or sys.argv[1:])
    if result.is_err():
        print(f"Error: {result.unwrap_err()}")
        return 1
    
    config = result.unwrap()
    
    # Run analysis (pure until I/O needed)
    analysis_result = run_analysis(config)
    
    # Handle result (I/O)
    if analysis_result.is_ok():
        save_results(analysis_result.unwrap(), config)
        return 0
    else:
        print(f"Analysis failed: {analysis_result.unwrap_err()}")
        return 1
```

## Best Practices

1. **Separate parsing from validation** - Parse first, then validate
1. **Use immutable configs** - Pass configs not individual args
1. **Return Results** - Let main handle exit codes
1. **Pure argument handling** - No I/O during parsing
1. **Descriptive help text** - Clear usage examples

## Testing

Pure functions make testing easy:

```python
def test_generate_notebook_name():
    config = FileNameConfig(
        excel_file=Path("data.xlsx"),
        model="gpt-4",
        sheet_index=0,
        sheet_name="Revenue Analysis"
    )
    
    name = generate_notebook_name(config)
    assert name == "data_sheet0_Revenue_Analysis_gpt4_r5.ipynb"

def test_parse_arguments():
    args = ["data.xlsx", "--model", "gpt-4", "--max-rounds", "3"]
    result = parse_arguments(args)
    
    assert result.is_ok()
    config = result.unwrap()
    assert config.model == "gpt-4"
    assert config.max_rounds == 3
```

## Migration Timeline

1. **Phase 2.1**: Extract file naming functions
1. **Phase 2.2**: Extract markdown generation
1. **Phase 2.3**: Create functional parsers
1. **Phase 2.4**: Build new CLI entry points
1. **Phase 2.5**: Update imports and deprecate old
