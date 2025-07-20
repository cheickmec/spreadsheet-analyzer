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

### Test-Driven Development (TDD) Methodology

We follow Kent Beck's Test-Driven Development principles and Tidy First approach:

#### Core TDD Cycle

Always follow the TDD cycle: **Red → Green → Refactor**

1. **Red**: Write the simplest failing test first
1. **Green**: Implement the minimum code needed to make tests pass
1. **Refactor**: Refactor only after tests are passing

#### TDD Implementation Guidelines

- Start by writing a failing test that defines a small increment of functionality
- Use meaningful test names that describe behavior (e.g., `test_should_sum_two_positive_numbers`)
- Make test failures clear and informative
- Write just enough code to make the test pass - no more
- Once tests pass, consider if refactoring is needed
- Repeat the cycle for new functionality

#### Tidy First Approach

Separate all changes into two distinct types:

1. **STRUCTURAL CHANGES**: Rearranging code without changing behavior (renaming, extracting methods, moving code)
1. **BEHAVIORAL CHANGES**: Adding or modifying actual functionality

Key principles:

- Never mix structural and behavioral changes in the same commit
- Always make structural changes first when both are needed
- Validate structural changes do not alter behavior by running tests before and after

#### Example TDD Workflow

```python
# Step 1: Write a failing test (Red)
def test_excel_cell_parser_extracts_simple_reference():
    """Test that parser can extract a simple cell reference."""
    parser = ExcelFormulaParser()
    result = parser.extract_references("=A1")
    assert result == [CellReference(sheet=None, cell="A1")]

# Step 2: Implement minimum code to pass (Green)
class ExcelFormulaParser:
    def extract_references(self, formula: str) -> List[CellReference]:
        if formula == "=A1":
            return [CellReference(sheet=None, cell="A1")]
        return []

# Step 3: Refactor if needed (still Green)
class ExcelFormulaParser:
    def extract_references(self, formula: str) -> List[CellReference]:
        # Extract pattern for simple references
        if formula.startswith("=") and len(formula) > 1:
            cell_ref = formula[1:]
            if self._is_valid_cell_reference(cell_ref):
                return [CellReference(sheet=None, cell=cell_ref)]
        return []
```

#### Commit Discipline for TDD

Only commit when:

1. ALL tests are passing
1. ALL compiler/linter warnings have been resolved
1. The change represents a single logical unit of work
1. Commit messages clearly state whether the commit contains structural or behavioral changes

Example commit messages:

```
refactor: extract cell reference validation to separate method (structural)
feat: add support for simple cell reference parsing (behavioral)
refactor: rename ExcelParser to ExcelFormulaParser for clarity (structural)
test: add test for range reference extraction (behavioral)
```

#### Refactoring Guidelines

- Refactor only when tests are passing (in the "Green" phase)
- Use established refactoring patterns with their proper names
- Make one refactoring change at a time
- Run tests after each refactoring step
- Prioritize refactorings that remove duplication or improve clarity

#### Python-Specific TDD Patterns

Prefer functional programming style over imperative style:

```python
# ✅ DO: Use functional style with Result types
def parse_formula(formula: str) -> Result[ParsedFormula, str]:
    return (
        validate_formula(formula)
        .and_then(tokenize_formula)
        .and_then(build_ast)
        .map(optimize_ast)
    )

# ❌ DON'T: Use imperative style with exceptions
def parse_formula(formula: str) -> ParsedFormula:
    if not validate_formula(formula):
        raise ValueError("Invalid formula")
    tokens = tokenize_formula(formula)
    ast = build_ast(tokens)
    return optimize_ast(ast)
```

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

### Pre-commit Best Practices

To avoid pre-commit hook failures and maintain code quality, follow these patterns:

#### Magic Numbers (PLR2004)

Always use typed constants instead of magic numbers:

```python
from typing import Final

# ✅ DO: Use typed constants
EXCEL_DATE_MAX: Final[int] = 100000
SMALL_SHEET_COUNT: Final[int] = 3
EMAIL_PATTERN_THRESHOLD: Final[float] = 0.7

def detect_date_patterns(values):
    if date_count > len(values) * EMAIL_PATTERN_THRESHOLD:
        # More readable and maintainable
```

```python
# ❌ DON'T: Use magic numbers
if date_count > len(values) * 0.7:  # What does 0.7 mean?
```

#### Boolean Arguments (FBT001/FBT002)

Make boolean parameters keyword-only to improve clarity:

```python
# ✅ DO: Use keyword-only boolean parameters
def analyze_workbook(file_path: Path, *, read_only: bool = True, include_formulas: bool = False):
    pass

# Called with explicit keywords
analyze_workbook(path, read_only=True, include_formulas=False)
```

```python
# ❌ DON'T: Use positional boolean parameters
def analyze_workbook(file_path: Path, read_only: bool = True, include_formulas: bool = False):
    pass

# Unclear what True/False mean
analyze_workbook(path, True, False)
```

#### Modern Python Syntax (UP038, C408)

Use modern Python 3.12+ syntax patterns:

```python
# ✅ DO: Use union syntax and tuple literals
isinstance(value, int | float)
locations = ()

# ❌ DON'T: Use old syntax
isinstance(value, (int, float))
locations = tuple()
```

#### Exception Handling (BLE001, S110)

Use specific exceptions and proper logging:

```python
# ✅ DO: Catch specific exceptions
try:
    process_excel_file(path)
except (OSError, ValueError, zipfile.BadZipFile) as e:
    logger.exception("Failed to process Excel file")
    return Err(f"Processing failed: {e}")
```

```python
# ❌ DON'T: Catch generic exceptions or ignore silently
try:
    process_excel_file(path)
except Exception:  # Too broad
    pass  # Silent failure
```

#### Import Organization (E402, PLC0415)

Keep imports at the top and avoid function-level imports:

```python
# ✅ DO: Top-level imports
from pathlib import Path
from typing import Final
import openpyxl
from openpyxl.utils import get_column_letter

def calculate_range():
    # Use imported function
    return get_column_letter(col)
```

```python
# ❌ DON'T: Function-level imports
def calculate_range():
    from openpyxl.utils import get_column_letter  # Move to top
    return get_column_letter(col)
```

#### ElementTree Naming (N817)

Use descriptive names for security-focused imports:

```python
# ✅ DO: Use descriptive names
import defusedxml.ElementTree as DefusedElementTree

root = DefusedElementTree.fromstring(xml_content)
```

```python
# ❌ DON'T: Use generic acronyms
import defusedxml.ElementTree as ET  # Not descriptive enough
```

#### Performance Patterns (PERF401)

Use list.extend() instead of loops for building lists:

```python
# ✅ DO: Use list.extend() with comprehensions
threats.extend([
    SecurityThreat(type="MACRO", file=f)
    for f in macro_files
])
```

```python
# ❌ DON'T: Use loops to build lists
for f in macro_files:
    threats.append(SecurityThreat(type="MACRO", file=f))
```

#### Line Length (E501)

Break long lines sensibly, especially with long constant names:

```python
# ✅ DO: Break long lines logically
incomplete_columns = [
    (col, score) for col, score in column_scores.items()
    if score < LOW_COMPLETENESS_THRESHOLD
]
```

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
