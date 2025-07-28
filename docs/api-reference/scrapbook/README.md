# scrapbook API Reference

## Overview

`scrapbook` is a library for data persistence and cross-notebook communication in the Jupyter ecosystem. It allows you to save data from one notebook execution and retrieve it in subsequent executions, enabling complex workflows and data sharing between notebooks.

## Quick Start

```python
import scrapbook as sb

# Save data to notebook
sb.glue("my_data", df)
sb.glue("my_plot", fig)

# Read data from notebook
nb = sb.read_notebook('analysis.ipynb')
df = nb.scraps.get("my_data")
fig = nb.scraps.get("my_plot")
```

## Package Structure

### Core Modules

- **[scrapbook](./modules/scrapbook.md)** - Main package module with core functions
- **[scrapbook.models](./modules/scrapbook.models.md)** - Data models for scraps and notebooks
- **[scrapbook.encoders](./modules/scrapbook.encoders.md)** - Data encoding and serialization

### Data Structures

- **[Scrapbook](./classes/Scrapbook.md)** - Main scrapbook container class
- **[Scrap](./classes/Scrap.md)** - Individual data scrap with metadata
- **[Notebook](./classes/Notebook.md)** - Enhanced notebook with scrapbook functionality

## Key Functions

### Data Persistence

- **[glue()](./functions/glue.md)** - Save data to the current notebook
- **[read_notebook()](./functions/read_notebook.md)** - Read notebook with scrapbook data
- **[display()](./functions/display.md)** - Display scrapbook data in notebook

### Data Retrieval

- **[get()](./functions/get.md)** - Retrieve data from scrapbook
- **[get_scrap()](./functions/get_scrap.md)** - Get scrap with metadata
- **[scraps](./functions/scraps.md)** - Access all scraps in notebook

### Utility Functions

- **[reglue()](./functions/reglue.md)** - Update existing scrap data
- **[clear()](./functions/clear.md)** - Clear all scraps from notebook
- **[list_scraps()](./functions/list_scraps.md)** - List all available scraps

## Data Types Support

### Supported Types

- **Pandas DataFrames**: Full support with metadata preservation
- **Matplotlib Figures**: Automatic serialization and display
- **NumPy Arrays**: Efficient serialization
- **Python Objects**: Pickle-based serialization
- **JSON-serializable**: Native JSON support
- **Files**: Path references and binary data

### Custom Encoders

```python
import scrapbook as sb
from my_encoder import CustomEncoder

# Register custom encoder
sb.encoders.register('custom', CustomEncoder)

# Use custom encoder
sb.glue("my_data", custom_object, encoder='custom')
```

## Common Patterns

### Basic Data Sharing

```python
import scrapbook as sb
import pandas as pd

# Save data
df = pd.read_csv('data.csv')
sb.glue("raw_data", df, display=False)

# Add metadata
sb.glue("data_info", {
    "rows": len(df),
    "columns": list(df.columns),
    "timestamp": pd.Timestamp.now()
})
```

### Figure and Plot Persistence

```python
import scrapbook as sb
import matplotlib.pyplot as plt

# Create and save plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4])
ax.set_title("My Plot")

# Save with display
sb.glue("my_plot", fig, display=True)
```

### Cross-Notebook Workflows

```python
import scrapbook as sb

# Read data from previous notebook
nb = sb.read_notebook('data_processing.ipynb')
processed_data = nb.scraps.get("processed_data")

# Continue analysis
results = analyze_data(processed_data)
sb.glue("analysis_results", results)
```

### Batch Processing

```python
import scrapbook as sb
import glob

# Process multiple notebooks
for notebook_path in glob.glob("*.ipynb"):
    nb = sb.read_notebook(notebook_path)
    
    # Extract specific data
    if "model_results" in nb.scraps:
        results = nb.scraps.get("model_results")
        # Process results...
```

## Integration with Other Packages

### With papermill

```python
import scrapbook as sb
import papermill as pm

# Execute notebook with parameters
pm.execute_notebook(
    'template.ipynb',
    'output.ipynb',
    parameters={'input_file': 'data.csv'}
)

# Read results
nb = sb.read_notebook('output.ipynb')
results = nb.scraps.get("results")
```

### With nbformat

```python
import scrapbook as sb
import nbformat as nbf

# Create notebook with scrapbook
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_code_cell("import scrapbook as sb"),
    nbf.v4.new_code_cell("sb.glue('test', 'hello world')")
]

# Write and read
nbf.write(nb, 'test.ipynb')
loaded_nb = sb.read_notebook('test.ipynb')
```

### With pandas

```python
import scrapbook as sb
import pandas as pd

# Save DataFrame with metadata
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
sb.glue("dataframe", df, metadata={
    "description": "Sample data",
    "source": "manual input"
})

# Retrieve with type preservation
retrieved_df = sb.get("dataframe")
assert isinstance(retrieved_df, pd.DataFrame)
```

## Error Handling

### Common Exceptions

- **ScrapbookException**: Base exception for scrapbook errors
- **ScrapNotFoundError**: When requested scrap doesn't exist
- **EncoderError**: When data encoding/decoding fails
- **ValidationError**: When scrap data is invalid

### Best Practices

```python
import scrapbook as sb

try:
    # Try to get existing data
    data = sb.get("my_data")
except sb.ScrapNotFoundError:
    # Create new data if not found
    data = create_new_data()
    sb.glue("my_data", data)
except sb.EncoderError as e:
    print(f"Data encoding error: {e}")
    # Handle encoding issues
```

## Performance Considerations

### Memory Usage

- Large DataFrames are stored efficiently
- Figures are serialized to reduce memory footprint
- Consider using `display=False` for large datasets

### Storage Optimization

```python
import scrapbook as sb

# Save only essential data
sb.glue("summary", df.describe(), display=False)

# Use compression for large objects
sb.glue("large_data", large_object, compress=True)
```

### Batch Operations

```python
import scrapbook as sb

# Efficient batch reading
notebooks = [sb.read_notebook(f) for f in notebook_files]
all_data = [nb.scraps.get("data") for nb in notebooks]
```

## Configuration

### Global Settings

```python
import scrapbook as sb

# Configure default encoder
sb.config.default_encoder = 'pandas'

# Set compression threshold
sb.config.compress_threshold = 1024 * 1024  # 1MB

# Enable debug mode
sb.config.debug = True
```

### Environment Variables

- `SCRAPBOOK_DEFAULT_ENCODER`: Set default encoder
- `SCRAPBOOK_COMPRESS_THRESHOLD`: Set compression threshold
- `SCRAPBOOK_DEBUG`: Enable debug mode

## External Resources

- [scrapbook Documentation](https://nteract-scrapbook.readthedocs.io/)
- [GitHub Repository](https://github.com/nteract/scrapbook)
- [nteract Project](https://nteract.io/)

______________________________________________________________________

*This documentation covers scrapbook version 0.5.x. For older versions, see the [legacy documentation](./legacy/).*
