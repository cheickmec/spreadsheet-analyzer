# Contributing to Spreadsheet Analyzer

Thank you for your interest in contributing to the Spreadsheet Analyzer project! This guide outlines our development practices, testing philosophy, and contribution workflow.

## Table of Contents

- [Development Philosophy](#development-philosophy)
- [Getting Started](#getting-started)
- [Testing Philosophy](#testing-philosophy)
- [Code Style and Standards](#code-style-and-standards)
- [Development Workflow](#development-workflow)
- [Documentation Standards](#documentation-standards)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)

## Development Philosophy

We follow these core principles:

1. **Optimize for maintainability over cleverness** - Clear, obvious code is better than clever, obscure code
1. **When in doubt, choose the boring solution** - Proven patterns over novel approaches
1. **Never assume business logic** - Always ask for clarification when requirements are ambiguous
1. **Tests Over Tools** - When tools conflict with testing, prioritize having the test
1. **Documentation-First Development** - Start with clear requirements before coding

## Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` for Python package management
- `pandoc` for documentation tools (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/spreadsheet-analyzer.git
cd spreadsheet-analyzer

# Install dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests to verify setup
uv run pytest
```

## Testing Philosophy

### Coverage Requirements

- **Minimum coverage: 90%** - Non-negotiable quality standard
- **Test pyramid approach**:
  - 70% Unit tests - Fast, isolated tests for individual functions/classes
  - 20% Integration tests - Test component interactions
  - 10% End-to-end tests - Full workflow validation

### Testing Principles

1. **Test behavior, not implementation** - Tests should survive refactoring
1. **One assertion per test** - Clear failure messages
1. **Descriptive test names** - `test_excel_parser_handles_circular_references`
1. **Arrange-Act-Assert pattern** - Consistent test structure
1. **Test edge cases** - Empty files, huge files, corrupted files

### Excel-Specific Test Fixtures

```python
@pytest.fixture
def simple_excel_file(tmp_path):
    """Create a simple Excel file for testing."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws['A1'] = 'Test'
    ws['B1'] = 42
    file_path = tmp_path / "test.xlsx"
    wb.save(file_path)
    return file_path

@pytest.fixture
def excel_with_formulas(tmp_path):
    """Create Excel file with formulas for testing."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws['A1'] = 10
    ws['A2'] = 20
    ws['A3'] = '=SUM(A1:A2)'
    file_path = tmp_path / "formulas.xlsx"
    wb.save(file_path)
    return file_path
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_excel_parser.py

# Run tests matching pattern
uv run pytest -k "circular_reference"

# Run with verbose output
uv run pytest -v
```

## Code Style and Standards

### Type Annotations

Always use type hints for function signatures:

```python
from pathlib import Path
from typing import List, Optional, Dict, Any

def analyze_workbook(
    file_path: Path,
    *,  # Force keyword-only arguments
    include_formulas: bool = True,
    max_depth: int = 10
) -> Dict[str, Any]:
    """Analyze Excel workbook structure."""
    ...
```

### Error Handling

Use specific exceptions and Result types:

```python
from typing import Union
from dataclasses import dataclass

@dataclass
class AnalysisError:
    message: str
    error_code: str
    details: Optional[Dict[str, Any]] = None

def parse_sheet(sheet_data) -> Union[ParsedSheet, AnalysisError]:
    try:
        # Parsing logic
        return ParsedSheet(...)
    except ValueError as e:
        return AnalysisError(
            message="Invalid sheet structure",
            error_code="PARSE_ERROR",
            details={"error": str(e)}
        )
```

### Functional Programming

Prefer immutable data structures and pure functions:

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class CellAnalysis:
    cell_ref: str
    value: Any
    formula: Optional[str]
    dependencies: FrozenSet[str]

# Create modified version without mutation
updated_analysis = replace(analysis, formula="=SUM(A1:A10)")
```

## Development Workflow

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

### Commit Messages

Follow conventional commits format:

```
feat: add support for pivot table analysis

- Implement pivot table detection algorithm
- Add tests for complex pivot scenarios
- Update documentation with examples

Closes #123
```

### Pull Request Process

1. **Create focused PRs** - One feature/fix per PR
1. **Write descriptive PR descriptions** - Include motivation and approach
1. **Ensure all tests pass** - Including new tests for your changes
1. **Update documentation** - Keep docs in sync with code
1. **Request review** - Tag appropriate reviewers

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Pre-commit hooks pass
- [ ] Coverage remains above 90%
- [ ] No security vulnerabilities introduced

## Documentation Standards

### Code Documentation

Use comprehensive docstrings:

```python
def analyze_cell_dependencies(
    worksheet: Worksheet,
    cell_ref: str
) -> Set[CellReference]:
    """
    Analyze dependencies for a given cell.

    This function parses the formula in the specified cell and extracts
    all cell references it depends on. Handles both simple references
    (A1) and range references (A1:B10).

    Args:
        worksheet: The worksheet containing the cell
        cell_ref: Cell reference in A1 notation (e.g., 'B5')

    Returns:
        Set of CellReference objects that this cell depends on

    Raises:
        InvalidCellReference: If cell_ref is not valid A1 notation

    Example:
        >>> deps = analyze_cell_dependencies(ws, 'C3')
        >>> print(deps)
        {CellReference('Sheet1', 'A1'), CellReference('Sheet1', 'B2')}

    Note:
        CLAUDE-KNOWLEDGE: Excel allows circular references but flags them.
        This function detects but doesn't prevent circular dependencies.
    """
```

### Anchor Comments

Use standardized comment anchors:

- `CLAUDE-KNOWLEDGE`: Domain-specific knowledge
- `CLAUDE-GOTCHA`: Non-obvious behavior
- `CLAUDE-COMPLEX`: Explanation for complex logic
- `CLAUDE-IMPORTANT`: Critical business rules
- `CLAUDE-SECURITY`: Security considerations
- `CLAUDE-PERFORMANCE`: Performance implications

## Performance Considerations

### Large File Handling

```python
# CLAUDE-PERFORMANCE: Use read_only mode for large files
def process_large_excel(file_path: Path) -> AnalysisResult:
    # Stream processing for memory efficiency
    wb = openpyxl.load_workbook(
        file_path,
        read_only=True,
        keep_vba=False,
        data_only=True
    )

    # Process in chunks
    for sheet in wb.worksheets:
        for chunk in sheet_to_chunks(sheet, chunk_size=1000):
            process_chunk(chunk)
```

### Optimization Guidelines

1. **Profile before optimizing** - Use cProfile for bottleneck identification
1. **Cache expensive operations** - Especially formula parsing
1. **Use generators for large datasets** - Avoid loading everything into memory
1. **Parallelize sheet analysis** - Sheets can be analyzed independently

## Security Guidelines

### Input Validation

```python
# CLAUDE-SECURITY: Always validate file inputs
def validate_excel_file(file_path: Path) -> List[ValidationIssue]:
    issues = []

    # Check file size
    if file_path.stat().st_size > MAX_FILE_SIZE:
        issues.append(ValidationIssue("File too large", "error"))

    # Verify file format
    if not is_valid_excel_format(file_path):
        issues.append(ValidationIssue("Invalid Excel format", "error"))

    # Scan for macros if needed
    if contains_vba_macros(file_path):
        issues.append(ValidationIssue("Contains macros", "warning"))

    return issues
```

### Security Checklist

- [ ] Validate all file inputs
- [ ] Sanitize file paths
- [ ] Check for malicious macros
- [ ] Limit resource consumption
- [ ] Use safe parsing modes
- [ ] Log security events

## Questions?

If you have questions or need clarification:

1. Check existing issues and discussions
1. Review the documentation in `/docs`
1. Create a new discussion for design questions
1. Create an issue for bugs or feature requests

Happy contributing!
