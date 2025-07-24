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
