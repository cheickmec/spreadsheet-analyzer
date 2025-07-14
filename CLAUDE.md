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

## Project Architecture

The project follows the **src layout** pattern for better import isolation:

```
spreadsheet-analyzer/
├── src/spreadsheet_analyzer/  # Main application code
├── tests/                     # Test files
├── tools/                     # Documentation utilities (not part of main app)
├── docs/                      # Comprehensive documentation
│   ├── design/               # System design documents
│   │   ├── comprehensive-system-design.md  # Authoritative design doc
│   │   └── conversations/    # Historical AI design discussions
│   ├── complete-guide/       # Implementation guides
│   └── research/             # Technical research organized by topic
```

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

**IMPORTANT**: Never include the "🤖 Generated with Claude Code" signature or any Claude-related signatures in commit messages.

## Development Dependencies

All development tools are specified in `pyproject.toml` under `[project.optional-dependencies]` and include:

- pytest for testing
- mypy for type checking
- ruff for linting/formatting
- bandit for security scanning
- pre-commit for automated checks
- pypandoc for HTML to Markdown conversion
