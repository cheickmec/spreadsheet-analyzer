# nbformat.read()

## Description

Read a notebook from a file or string and return the notebook object. This is the primary function for loading Jupyter notebooks into Python objects.

## Signature

```python
nbformat.read(fp, as_version=None, as_version_minor=None, **kwargs) -> NotebookNode
```

## Parameters

- **fp** (`Union[str, Path, TextIO, BinaryIO]`): File path, file-like object, or string containing notebook content
- **as_version** (`int`, optional): Version to convert the notebook to (default: auto-detect)
- **as_version_minor** (`int`, optional): Minor version number (default: auto-detect)
- \*\***kwargs**: Additional arguments passed to `json.loads()` or `json.load()`

## Returns

- **NotebookNode**: A notebook object representing the loaded notebook

## Raises

- **FileNotFoundError**: When the specified file path doesn't exist
- **NotJSONError**: When the file content is not valid JSON
- **ValidationError**: When the notebook structure is invalid
- **VersionError**: When the notebook version is unsupported

## Examples

### Basic Usage

```python
import nbformat

# Read from file path
nb = nbformat.read('notebook.ipynb')

# Read from file object
with open('notebook.ipynb', 'r') as f:
    nb = nbformat.read(f)

# Read from string
content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}'
nb = nbformat.read(content, as_version=4)
```

### Version Conversion

```python
import nbformat

# Force conversion to version 4
nb = nbformat.read('old_notebook.ipynb', as_version=4)

# Specify both major and minor version
nb = nbformat.read('notebook.ipynb', as_version=4, as_version_minor=4)
```

### Error Handling

```python
import nbformat

try:
    nb = nbformat.read('notebook.ipynb')
except FileNotFoundError:
    print("Notebook file not found")
except nbformat.NotJSONError:
    print("File is not valid JSON")
except nbformat.ValidationError as e:
    print(f"Invalid notebook structure: {e}")
except nbformat.VersionError as e:
    print(f"Unsupported notebook version: {e}")
```

### Reading with Custom JSON Options

```python
import nbformat

# Read with custom JSON parsing options
nb = nbformat.read('notebook.ipynb', parse_float=Decimal)
```

## Notes

### Version Detection

The function automatically detects the notebook version from the `nbformat` field in the JSON. If `as_version` is not specified, it will:

1. Read the version from the notebook metadata
1. Convert to the latest supported version if needed
1. Validate the notebook structure

### File Format Support

The function supports multiple input formats:

- **File paths**: String or Path objects pointing to `.ipynb` files
- **File objects**: Open file handles (text or binary mode)
- **Strings**: JSON strings containing notebook content

### Memory Usage

For large notebooks, consider using file objects instead of loading the entire content into memory as a string.

### Performance Considerations

- Reading from file paths is generally faster than reading from strings
- Version conversion adds overhead but ensures compatibility
- Validation can be disabled for better performance in production

## See Also

- **[nbformat.write()](./write.md)** - Write notebook to file
- **[nbformat.parse()](./parse.md)** - Parse notebook JSON string
- **[nbformat.validate()](./validate.md)** - Validate notebook structure
- **[NotebookNode](./../classes/NotebookNode.md)** - Notebook container class

## Related Documentation

- [Jupyter Notebook Format Specification](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [Version Migration Guide](https://nbformat.readthedocs.io/en/latest/migrate.html)
