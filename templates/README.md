# Jinja2 Template System for LLM Prompts

This directory contains the Jinja2 template system for generating LLM prompts in the spreadsheet analyzer. The system provides a flexible, maintainable way to manage prompt engineering strategies.

## Directory Structure

```
templates/
├── base/                       # Base templates and common components
│   ├── master.jinja2          # Master template with common structure
│   ├── components/            # Reusable components
│   │   ├── context.jinja2     # Context formatting macros
│   │   ├── instructions.jinja2 # Common instructions
│   │   └── validation.jinja2  # Validation prompts
│   └── partials/              # Small reusable snippets
│       ├── cell_format.jinja2 # Cell presentation
│       └── error_format.jinja2 # Error handling
├── strategies/                # Strategy-specific templates
│   ├── hierarchical/         # Hierarchical exploration strategy
│   ├── graph_based/          # Graph-based analysis (PROMPT-SAW)
│   ├── chain_of_thought/     # Chain-of-thought reasoning
│   └── default/              # Default/fallback strategy
└── custom/                   # User-defined templates
```

## Template Hierarchy

All strategy templates extend from `base/master.jinja2`, which provides:

- Common system context
- Standard task structure
- Default instructions
- Output format guidelines

## Using the Template System

### Basic Usage

```python
from spreadsheet_analyzer.notebook_llm import get_template_manager, render_prompt

# Quick rendering
prompt = render_prompt("strategies/default/analysis.jinja2", {
    "task": {"description": "Analyze sales data"},
    "excel_metadata": {...}
})
```

### Advanced Usage

```python
from spreadsheet_analyzer.notebook_llm import TemplateManager, StrategyTemplateLoader

# Create manager
manager = TemplateManager()

# Load specific strategy
loader = StrategyTemplateLoader(manager)
prompt = loader.render_strategy_prompt(
    "hierarchical",
    "exploration",
    context
)
```

## Available Strategies

### 1. Hierarchical Exploration

- **Template**: `strategies/hierarchical/exploration.jinja2`
- **Purpose**: Progressive analysis from overview to details
- **Levels**: overview → sheet → region
- **Use case**: Complex workbooks with many sheets

### 2. Graph-Based Analysis (PROMPT-SAW)

- **Template**: `strategies/graph_based/analysis.jinja2`
- **Purpose**: Dependency-aware analysis using PageRank
- **Features**: Importance scoring, critical path detection
- **Use case**: Complex formula dependencies

### 3. Chain-of-Thought Reasoning

- **Template**: `strategies/chain_of_thought/reasoning.jinja2`
- **Purpose**: Step-by-step analytical reasoning
- **Features**: Self-validation, progressive building
- **Use case**: Complex problem solving, debugging

### 4. Default Strategy

- **Template**: `strategies/default/analysis.jinja2`
- **Purpose**: General-purpose analysis fallback
- **Features**: Comprehensive but generic approach
- **Use case**: When no specific strategy matches

## Creating Custom Templates

### 1. Extend the Base Template

```jinja2
{% extends "base/master.jinja2" %}

{% block system_context %}
Your custom system context here
{% endblock %}

{% block task_description %}
{{ task.description }}
Your custom task formatting
{% endblock %}
```

### 2. Use Provided Macros

```jinja2
{% from 'base/components/context.jinja2' import format_excel_metadata %}
{{ format_excel_metadata(metadata) }}
```

### 3. Add Custom Sections

```jinja2
{% block additional_sections %}
<custom_section>
Your custom content
</custom_section>
{% endblock %}
```

## Available Macros

### Context Formatting

- `format_notebook_state(notebook)` - Format current notebook variables and cells
- `format_excel_metadata(metadata)` - Display Excel file information
- `format_analysis_focus(focus)` - Show analysis target details
- `format_cell_samples(cells)` - Display sample cells
- `format_dependency_graph(graph_info)` - Show formula dependencies

### Cell Formatting

- `format_cell(cell)` - Compact cell representation
- `format_cell_detailed(cell)` - Detailed cell information
- `format_range(cells)` - Format multiple cells
- `format_formula_info(formula_info)` - Formula analysis details

### Error/Validation

- `format_error(error)` - Error formatting
- `format_warning(warning)` - Warning formatting
- `format_validation_result(result)` - Validation outcome
- `validation_checklist()` - Standard validation requirements

## Custom Filters

The template system includes custom Jinja2 filters:

- `truncate_middle(length)` - Truncate text preserving start/end
- `format_cell_ref(sheet)` - Format Excel cell references
- `format_bytes` - Human-readable file sizes
- `highlight_formulas` - Emphasize Excel formulas in text

## Best Practices

1. **Inherit from Base**: Always extend `base/master.jinja2` for consistency
1. **Use Macros**: Leverage existing macros for common formatting
1. **Keep DRY**: Extract repeated patterns into new macros
1. **Document Variables**: Comment expected context variables
1. **Test Templates**: Use the test suite to verify rendering

## Example Context Structure

```python
context = {
    "task": {
        "description": "Analyze Q4 financial results",
        "focus": "revenue trends"
    },
    "excel_metadata": {
        "filename": "financial_2024.xlsx",
        "size_mb": 3.2,
        "sheet_count": 12,
        "sheet_names": ["Summary", "Q1", "Q2", "Q3", "Q4"],
        "formula_count": 1500
    },
    "analysis_focus": {
        "sheet_name": "Q4",
        "cell_range": "A1:Z100",
        "analysis_type": "trend_analysis"
    },
    "notebook": {
        "variables": {
            "df": {"type": "DataFrame", "description": "Q4 data"},
            "wb": {"type": "Workbook", "description": "Excel file"}
        },
        "executed_cells": [...]
    }
}
```

## Testing Templates

Run the template tests:

```bash
uv run pytest tests/test_template_system.py -v
```

Run the demo script:

```bash
uv run python examples/template_usage_demo.py
```
