# CLI Architecture Design - Terminal-First with API-Ready Foundation

> **Created**: July 20, 2025\
> **Status**: Active Design\
> **Purpose**: Design a CLI interface that provides excellent user experience while laying the groundwork for future API implementation

## Overview

This design document outlines a terminal-first architecture for the Spreadsheet Analyzer that prioritizes immediate usability through a rich CLI experience while ensuring the codebase is structured to easily support API endpoints (FastAPI) in the future.

## Design Goals

1. **Immediate Usability**: Rich terminal interface with progress tracking and comprehensive logging
1. **API-Ready Architecture**: Clean separation of concerns enabling easy API addition
1. **Developer Experience**: Clear, informative output with debugging capabilities
1. **Performance Visibility**: Real-time progress updates and performance metrics
1. **Extensibility**: Plugin-style command structure for easy feature addition

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Layer (Click)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   analyze   │  │    batch    │  │     watch      │   │
│  │   command   │  │   command   │  │    command     │   │
│  └──────┬──────┘  └──────┬──────┘  └───────┬──────────┘   │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                  │
├───────────────────────────┼─────────────────────────────────┤
│                    Service Layer                             │
│  ┌─────────────────────────┴────────────────────────────┐   │
│  │              AnalysisService                          │   │
│  ├───────────────────────────────────────────────────────┤   │
│  │ - analyze_file(path, options) -> AnalysisResult      │   │
│  │ - analyze_batch(paths, options) -> List[Result]      │   │
│  │ - watch_directory(path, callback) -> None            │   │
│  └───────────────────────────────────────────────────────┘   │
│                           │                                  │
├───────────────────────────┼─────────────────────────────────┤
│                  Core Pipeline (Existing)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Stage 0  │→ │ Stage 1  │→ │ Stage 2  │→ │ Stage 3  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. CLI Layer (using Click)

```python
# src/spreadsheet_analyzer/cli/__init__.py
import click
from spreadsheet_analyzer.cli import commands

@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
@click.option('--format', type=click.Choice(['json', 'yaml', 'table', 'markdown']))
@click.pass_context
def cli(ctx, verbose, quiet, format):
    """Spreadsheet Analyzer - Intelligent Excel Analysis"""
    ctx.ensure_object(dict)
    ctx.obj['verbosity'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['format'] = format or 'table'

cli.add_command(commands.analyze)
cli.add_command(commands.batch)
cli.add_command(commands.watch)
cli.add_command(commands.validate)
cli.add_command(commands.compare)
```

### 2. Service Layer Pattern

```python
# src/spreadsheet_analyzer/services/analysis_service.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any
import asyncio

@dataclass
class AnalysisOptions:
    """Options that can come from CLI args or API requests"""
    mode: str = 'standard'  # fast, standard, deep
    include_formulas: bool = True
    include_security: bool = True
    max_depth: Optional[int] = None
    progress_callback: Optional[Callable[[str, float, str], None]] = None

class AnalysisService:
    """
    Service layer that encapsulates business logic.
    Can be called from CLI or API endpoints.
    """
    
    def __init__(self, *, pipeline_factory=None):
        self._pipeline_factory = pipeline_factory or create_default_pipeline
        self._logger = structlog.get_logger(__name__)
    
    async def analyze_file(
        self, 
        file_path: Path, 
        options: AnalysisOptions
    ) -> AnalysisResult:
        """Analyze single file with options."""
        # This same method can be called from:
        # - CLI command
        # - Future FastAPI endpoint
        # - Programmatic API
        
        with self._logger.contextvars.bind(file=str(file_path)):
            self._logger.info("Starting analysis", mode=options.mode)
            
            pipeline = self._pipeline_factory()
            
            # Progress tracking that works for both CLI and API
            if options.progress_callback:
                pipeline.set_progress_callback(options.progress_callback)
            
            result = await pipeline.analyze(file_path, options)
            
            self._logger.info(
                "Analysis complete",
                duration=result.duration,
                issues_found=len(result.issues)
            )
            
            return result
```

### 3. Rich Terminal Output

```python
# src/spreadsheet_analyzer/cli/console.py
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import structlog

class RichConsoleHandler:
    """Rich terminal output with progress tracking."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )
    
    def show_analysis_progress(self, file_path: Path):
        """Show live progress during analysis."""
        with self.progress:
            task = self.progress.add_task(
                f"Analyzing {file_path.name}...", 
                total=5  # 5 stages
            )
            
            def update_progress(stage: str, progress: float, message: str):
                self.progress.update(
                    task, 
                    advance=1,
                    description=f"[cyan]{stage}[/]: {message}"
                )
            
            return update_progress
    
    def display_results(self, result: AnalysisResult, format: str):
        """Display results in requested format."""
        if format == 'table':
            self._display_table(result)
        elif format == 'json':
            self._display_json(result)
        elif format == 'markdown':
            self._display_markdown(result)
```

### 4. Structured Logging

```python
# src/spreadsheet_analyzer/logging.py
import structlog
from pathlib import Path
import sys

def setup_logging(*, verbosity: int = 0, log_file: Optional[Path] = None):
    """
    Configure structured logging for both CLI and future API.
    
    Verbosity levels:
    0 = WARNING and above
    1 = INFO and above  
    2 = DEBUG and above
    """
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Console output
    if sys.stdout.isatty():
        # Rich console output for terminals
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON for pipes/files
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### 5. Command Implementation Example

```python
# src/spreadsheet_analyzer/cli/commands/analyze.py
import click
from pathlib import Path
from spreadsheet_analyzer.services import AnalysisService
from spreadsheet_analyzer.cli.console import RichConsoleHandler

@click.command()
@click.argument('file', type=click.Path(exists=True, path_type=Path))
@click.option('--mode', '-m', type=click.Choice(['fast', 'standard', 'deep']), 
              default='standard', help='Analysis depth')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
@click.option('--watch', '-w', is_flag=True, help='Watch file for changes')
@click.pass_context
def analyze(ctx, file, mode, output, watch):
    """Analyze a single Excel file."""
    console = RichConsoleHandler()
    service = AnalysisService()
    
    # Get format from context
    output_format = ctx.obj.get('format', 'table')
    
    # Setup progress callback for console
    progress_callback = console.show_analysis_progress(file)
    
    # Create options (same structure for future API)
    options = AnalysisOptions(
        mode=mode,
        progress_callback=progress_callback
    )
    
    try:
        # Run analysis (async-ready for future API)
        result = asyncio.run(service.analyze_file(file, options))
        
        # Display results
        console.display_results(result, output_format)
        
        # Save if requested
        if output:
            save_results(result, output, output_format)
            
        # Exit code based on issues found
        ctx.exit(0 if result.is_healthy else 1)
        
    except Exception as e:
        console.console.print(f"[red]Error:[/] {e}")
        ctx.exit(2)
```

## Migration Path to FastAPI

When ready to add API endpoints, the migration is straightforward:

```python
# future: src/spreadsheet_analyzer/api/app.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from spreadsheet_analyzer.services import AnalysisService, AnalysisOptions

app = FastAPI()
service = AnalysisService()  # Same service used by CLI

@app.post("/analyze")
async def analyze_file(
    file: UploadFile,
    mode: str = "standard",
    background_tasks: BackgroundTasks = None
):
    """API endpoint using the same service layer."""
    
    # Save uploaded file
    file_path = save_upload(file)
    
    # Create options (same as CLI)
    options = AnalysisOptions(
        mode=mode,
        progress_callback=None  # Use different progress tracking for API
    )
    
    # Use exact same service method
    result = await service.analyze_file(file_path, options)
    
    return result.to_dict()
```

## Implementation Phases

### Phase 1: Basic CLI Structure (Week 1)

- [ ] Setup Click application structure
- [ ] Implement basic analyze command
- [ ] Add structured logging
- [ ] Create service layer skeleton

### Phase 2: Rich Terminal Experience (Week 2)

- [ ] Add Rich console output
- [ ] Implement progress tracking
- [ ] Add multiple output formats
- [ ] Create interactive mode

### Phase 3: Advanced Commands (Week 3)

- [ ] Batch analysis command
- [ ] Watch mode for continuous monitoring
- [ ] Comparison command for multiple files
- [ ] Validation command with rules

### Phase 4: Performance & Polish (Week 4)

- [ ] Add performance metrics display
- [ ] Implement result caching
- [ ] Add configuration file support
- [ ] Create shell completion

## Key Design Decisions

1. **Service Layer Pattern**: All business logic in services, not in CLI commands
1. **Structured Logging**: Using structlog for both human and machine readable logs
1. **Rich Terminal UI**: Using Rich library for beautiful, informative output
1. **Async-First**: Service methods are async-ready for future API needs
1. **Configuration as Code**: Options objects that work for both CLI and API

## Benefits of This Approach

1. **Immediate Value**: Users get a powerful CLI tool right away
1. **Future-Proof**: Clean architecture makes API addition trivial
1. **Testing**: Service layer is easily testable without CLI/API concerns
1. **Flexibility**: Can run as CLI, API, or embedded library
1. **Operations**: Structured logging works for both development and production

## Example Usage

```bash
# Basic analysis
$ spreadsheet-analyzer analyze financial-model.xlsx

# Verbose mode with progress
$ spreadsheet-analyzer -vv analyze large-file.xlsx --mode deep

# Batch analysis with JSON output
$ spreadsheet-analyzer batch *.xlsx --format json > results.json

# Watch directory for changes
$ spreadsheet-analyzer watch /path/to/excel/files --on-change analyze

# Compare two versions
$ spreadsheet-analyzer compare old-version.xlsx new-version.xlsx
```

## Next Steps

1. Create the CLI package structure
1. Implement the service layer
1. Build the first analyze command
1. Add rich terminal output
1. Document CLI usage

This architecture provides an excellent user experience through the terminal while ensuring that when we're ready to add FastAPI, it will be a simple addition rather than a refactor.
