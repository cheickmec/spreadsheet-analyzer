# nbformat API Reference

## Overview

`nbformat` is the core library for reading, writing, and validating Jupyter notebook files (`.ipynb`). It provides the data structures and utilities needed to work with the notebook format specification.

## Quick Start

```python
import nbformat

# Read a notebook
with open('notebook.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Write a notebook
with open('output.ipynb', 'w') as f:
    nbformat.write(nb, f)

# Create a new notebook
nb = nbformat.v4.new_notebook()
```

## Package Structure

### Core Modules
- **[nbformat](./modules/nbformat.md)** - Main package module with core functions
- **[nbformat.v4](./modules/nbformat.v4.md)** - Version 4 notebook format utilities
- **[nbformat.v3](./modules/nbformat.v3.md)** - Version 3 notebook format utilities (legacy)
- **[nbformat.validator](./modules/nbformat.validator.md)** - Notebook validation utilities

### Data Structures
- **[NotebookNode](./classes/NotebookNode.md)** - Main notebook container class
- **[Cell](./classes/Cell.md)** - Base cell class
- **[CodeCell](./classes/CodeCell.md)** - Code cell implementation
- **[MarkdownCell](./classes/MarkdownCell.md)** - Markdown cell implementation
- **[RawCell](./classes/RawCell.md)** - Raw cell implementation

## Key Functions

### Reading and Writing
- **[read()](./functions/read.md)** - Read notebook from file or string
- **[write()](./functions/write.md)** - Write notebook to file
- **[parse()](./functions/parse.md)** - Parse notebook JSON string
- **[mimetype2notebook()](./functions/mimetype2notebook.md)** - Convert MIME type to notebook

### Notebook Creation
- **[new_notebook()](./functions/new_notebook.md)** - Create new notebook
- **[new_code_cell()](./functions/new_code_cell.md)** - Create new code cell
- **[new_markdown_cell()](./functions/new_markdown_cell.md)** - Create new markdown cell
- **[new_raw_cell()](./functions/new_raw_cell.md)** - Create new raw cell

### Validation
- **[validate()](./functions/validate.md)** - Validate notebook structure
- **[validate_one()](./functions/validate_one.md)** - Validate single cell
- **[is_type()](./functions/is_type.md)** - Check if object is of specific type

## Version Support

### Version 4 (Current)
- Full support for all modern notebook features
- Recommended for new notebooks
- Supports metadata, attachments, and rich outputs

### Version 3 (Legacy)
- Backward compatibility for older notebooks
- Limited feature set
- Automatic conversion to v4 when reading

## Common Patterns

### Creating a Notebook Programmatically
```python
import nbformat as nbf

# Create notebook
nb = nbf.v4.new_notebook()

# Add cells
nb.cells = [
    nbf.v4.new_markdown_cell("# My Analysis"),
    nbf.v4.new_code_cell("import pandas as pd"),
    nbf.v4.new_code_cell("df = pd.read_csv('data.csv')"),
    nbf.v4.new_code_cell("df.head()")
]

# Write to file
nbf.write(nb, 'analysis.ipynb')
```

### Reading and Modifying
```python
import nbformat as nbf

# Read existing notebook
with open('notebook.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Add new cell
new_cell = nbf.v4.new_code_cell("print('Hello, World!')")
nb.cells.append(new_cell)

# Write back
with open('modified_notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
```

### Validation
```python
import nbformat as nbf

# Validate notebook
try:
    nbf.validate(nb)
    print("Notebook is valid")
except nbf.ValidationError as e:
    print(f"Validation error: {e}")
```

## Integration with Other Packages

### With nbclient
```python
import nbformat as nbf
from nbclient import NotebookClient

# Create and execute notebook
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell("2 + 2")]

client = NotebookClient(nb)
client.execute()
```

### With papermill
```python
import nbformat as nbf
import papermill as pm

# Create parameterized notebook
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_code_cell("input_file = '{{input_file}}'"),
    nbf.v4.new_code_cell("df = pd.read_csv(input_file)")
]

# Execute with parameters
pm.execute_notebook(nb, 'output.ipynb', parameters={'input_file': 'data.csv'})
```

## Error Handling

### Common Exceptions
- **ValidationError**: Invalid notebook structure
- **NotJSONError**: Invalid JSON format
- **VersionError**: Unsupported notebook version

### Best Practices
```python
import nbformat as nbf

try:
    with open('notebook.ipynb', 'r') as f:
        nb = nbf.read(f, as_version=4)
    nbf.validate(nb)
except FileNotFoundError:
    print("Notebook file not found")
except nbf.ValidationError as e:
    print(f"Invalid notebook: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- **Large notebooks**: Use streaming for very large notebooks
- **Memory usage**: Notebooks are loaded entirely into memory
- **Validation**: Disable validation for better performance in production

## External Resources

- [nbformat Documentation](https://nbformat.readthedocs.io/)
- [Jupyter Notebook Format Specification](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [GitHub Repository](https://github.com/jupyter/nbformat)

---

*This documentation covers nbformat version 5.x. For older versions, see the [legacy documentation](./legacy/).* 