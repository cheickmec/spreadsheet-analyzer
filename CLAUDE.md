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

## Development Commands

### Setup

```bash
uv sync                    # Install dependencies
uv sync --dev             # Install development dependencies
uv run pre-commit install # Install pre-commit hooks
```

### Running Code

```bash
uv run src/spreadsheet_analyzer/main.py  # Run main application
uv run pytest                            # Run all tests
uv run pytest tests/test_specific.py     # Run specific test file
uv run pytest -k "test_name"             # Run test by name pattern
```

### Code Quality

```bash
uv run pre-commit run --all-files        # Run all pre-commit checks
uv run pre-commit run ruff --all-files   # Run only ruff linting
uv run pre-commit run mypy --all-files   # Run only type checking
```

### Documentation Tools

#### HTML to Markdown Converter

Convert HTML files to well-formatted Markdown using pypandoc:

```bash
# Basic conversion (creates .md file with same name)
uv run tools/html_to_markdown_converter.py input.html

# Specify output file
uv run tools/html_to_markdown_converter.py input.html output.md

# Use inline links instead of reference-style links
uv run tools/html_to_markdown_converter.py input.html --inline-links

# Batch convert multiple files to a directory
uv run tools/html_to_markdown_converter.py *.html -o output_dir/

# Verbose mode for debugging
uv run tools/html_to_markdown_converter.py input.html -v
```

**Note**: Requires pandoc to be installed (`brew install pandoc` on macOS)

This tool automatically detects file encodings using `chardet` and converts non-UTF-8 files to UTF-8 to ensure compatibility with mdformat and other tools.

## Project Architecture

The project follows the **src layout** pattern for better import isolation:

```
spreadsheet-analyzer/
├──  src/spreadsheet_analyzer/  # Main application code
├──  tests/                     # Test files
├──  tools/                     # Documentation utilities (not part of main app)
├──  docs/                      # Comprehensive documentation
│   ├──  design/               # System design documents
│   │   ├──  comprehensive-system-design.md  # Authoritative design doc
│   │   └──  conversations/    # Historical AI design discussions
│   ├──  complete-guide/       # Implementation guides
│   └──  research/             # Technical research organized by topic
```

## Development Philosophy

Adopt these core principles when developing:

1. **"Optimize for maintainability over cleverness"** - Clear, obvious code is better than clever, obscure code
1. **"When in doubt, choose the boring solution"** - Proven patterns over novel approaches
1. **"Never assume business logic"** - Always ask for clarification when requirements are ambiguous
1. **"Tests Over Tools"** - When tools conflict with testing, prioritize having the test
1. **"Documentation-First Development"** - ALWAYS Start with clear requirements before coding

## Code Evolution Practices

When modifying or improving code:

1. **Edit Files In Place** - Never create new files with suffixes like `_refactored`, `_enhanced`, `_improved`, `_v2`, etc. Always edit the existing file directly.

1. **Rely on Git History** - Version control tracks all changes. There's no need to keep old versions as separate files. Ensure changes are committed before making major edits.

1. **Use Descriptive Names** - File names should describe what the code does, not its version state. For example:

   - ✅ `stage_3_formulas.py` (describes functionality)
   - ❌ `stage_3_formulas_refactored.py` (indicates version)
   - ❌ `stage_3_formulas_enhanced.py` (indicates improvement)

1. **Replace, Don't Duplicate** - When improving code, replace the existing implementation rather than creating alternatives that make the original redundant.

## Key Design Principles

Based on the comprehensive system design document:

1. **Validation-First Philosophy**: Never trust assumptions about data structure. Validate every analysis claim through actual operations.

1. **Hybrid Architecture**: Monolithic deployment with multi-agent intelligence for parallel sheet analysis within a single deployable unit.

1. **Notebook-Based Execution**: Uses Jupyter kernels for maintaining audit trails and state persistence throughout analysis.

1. **Tool Bus Governance**: All tool access is mediated through a governed registry for security and resource control.

1. **Performance Targets**:

   - File Upload: < 2 seconds for files up to 10MB
   - Basic Analysis: < 5 seconds for standard files (< 10 sheets, < 10K cells)
   - Deep AI Analysis: < 30 seconds for complex workbooks (< 50 sheets, < 100K cells)

## Code Style Configuration

- Line length: 120 characters
- Python 3.12+ syntax
- Ruff for linting and formatting (replaces Black, isort, flake8)
- Type hints encouraged but not required
- Comprehensive pre-commit hooks for quality assurance

## Development Practices

For detailed development practices, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Important Linting Rules to Follow

To avoid common pre-commit hook failures, follow these patterns:

#### Logging Best Practices

```python
# ✅ DO: Use logger instance, not root logger
logger = logging.getLogger(__name__)
logger.info("Found version: %s", version)

# ❌ DON'T: Use root logger or f-strings in logging
logging.info(f"Found version: {version}")
```

#### Exception Handling

```python
# ✅ DO: Use specific exceptions and logger.exception() without redundant error info
try:
    risky_operation()
except (OSError, IOError):
    logger.exception("Operation failed")

# ❌ DON'T: Catch generic Exception or include exception in message
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

#### Function Arguments

```python
# ✅ DO: Use keyword-only for boolean parameters
def setup_logging(*, verbose: bool = False) -> None:
    pass

# ❌ DON'T: Use positional boolean parameters
def setup_logging(verbose: bool = False) -> None:
    pass
```

#### Control Flow

```python
# ✅ DO: Use else block with try/except for success path
try:
    result = operation()
except ValueError:
    logger.error("Failed")
    return None
else:
    logger.info("Success")
    return result

# ❌ DON'T: Return directly from try block
try:
    result = operation()
    return result  # Ruff wants this in else block
```

#### Simplifications

```python
# ✅ DO: Use ternary operators for simple conditionals
output_file = output_dir / f"{name}.md" if output_dir else None

# ❌ DON'T: Use verbose if/else blocks for simple assignments
if output_dir:
    output_file = output_dir / f"{name}.md"
else:
    output_file = None
```

#### String Formatting

```python
# ✅ DO: Use % formatting for logging, f-strings elsewhere
logger.info("Processing %d files in %s", count, directory)
print(f"Processing {count} files")

# ❌ DON'T: Use f-strings in logging calls
logger.info(f"Processing {count} files")
```

## Commit Message Convention

Uses conventional commits format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

**IMPORTANT**: Never include the "ü§ñ Generated with Claude Code" signature or any Claude-related signatures in commit messages.

## Pull Request Requirements

**IMPORTANT**: All pull requests MUST use the PR template located at `.github/pull_request_template.md`.

When creating a PR:

1. Fill out ALL sections of the template completely
1. Check all applicable boxes in the checklists
1. Provide detailed testing evidence
1. Include performance metrics if applicable
1. List specific areas for reviewer focus

The PR template ensures consistent quality, proper testing, and thorough documentation of changes.

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

## Functional Programming Patterns

Prefer functional approaches for Excel analysis:

### 1. Immutable Data Structures

```python
from dataclasses import dataclass, replace
from typing import FrozenSet, Tuple

@dataclass(frozen=True)
class CellReference:
    sheet: str
    row: int
    column: str

@dataclass(frozen=True)
class FormulaAnalysis:
    formula: str
    dependencies: FrozenSet[CellReference]
    is_circular: bool

# Create modified version without mutation
new_analysis = replace(analysis, is_circular=True)
```

### 2. Pure Functions

```python
# ✅ DO: Pure function with no side effects
def calculate_cell_dependencies(
    formula: str,
    sheet_name: str
) -> FrozenSet[CellReference]:
    """Extract cell references from formula - no side effects."""
    references = set()
    # Parse formula and extract references
    return frozenset(references)

# ❌ DON'T: Function with side effects
def analyze_and_log_formula(formula: str, logger):
    logger.info(f"Analyzing {formula}")  # Side effect!
    return parse_formula(formula)
```

### 3. Function Composition

```python
from functools import partial
from typing import Callable

# Compose analysis functions
analyze_structure = partial(analyze_sheet, analysis_type="structure")
analyze_formulas = partial(analyze_sheet, analysis_type="formulas")

# Pipeline of transformations
def analysis_pipeline(
    workbook_path: Path
) -> WorkbookAnalysis:
    return (
        load_workbook_safe(workbook_path)
        |> validate_workbook
        |> extract_metadata
        |> analyze_all_sheets
        |> generate_report
    )
```

### 4. Result Types for Error Handling

```python
from typing import Union, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')

@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

Result = Union[Ok[T], Err[E]]

# Usage
def parse_excel_file(path: Path) -> Result[Workbook, str]:
    try:
        wb = openpyxl.load_workbook(path, read_only=True)
        return Ok(wb)
    except Exception as e:
        return Err(f"Failed to load Excel file: {e}")
```

## Development Dependencies

All development tools are specified in `pyproject.toml` under `[project.optional-dependencies]` and include:

- pytest for testing
- mypy for type checking
- ruff for linting/formatting
- bandit for security scanning
- pre-commit for automated checks
- pypandoc for HTML to Markdown conversion
