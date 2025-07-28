# nbclient API Reference

## Overview

`nbclient` is the modern, official Jupyter library for programmatically executing notebooks. It provides a robust, asynchronous API for running notebook cells, handling kernel communication, and managing execution state. This is the recommended replacement for the older `nbconvert.preprocessors.execute` approach.

## Quick Start

```python
from nbclient import NotebookClient
import nbformat

# Load notebook
with open('notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook
client = NotebookClient(nb)
client.execute()
```

## Package Structure

### Core Modules

- **[nbclient](./modules/nbclient.md)** - Main package module with core classes
- **[nbclient.client](./modules/nbclient.client.md)** - NotebookClient implementation
- **[nbclient.exceptions](./modules/nbclient.exceptions.md)** - Exception classes
- **[nbclient.util](./modules/nbclient.util.md)** - Utility functions

### Main Classes

- **[NotebookClient](./classes/NotebookClient.md)** - Primary client for notebook execution
- **[KernelManager](./classes/KernelManager.md)** - Kernel lifecycle management
- **[KernelClient](./classes/KernelClient.md)** - Kernel communication interface

## Key Classes

### Execution Control

- **[NotebookClient](./classes/NotebookClient.md)** - Main execution client
- **[AsyncNotebookClient](./classes/AsyncNotebookClient.md)** - Asynchronous execution client
- **[CellExecutor](./classes/CellExecutor.md)** - Individual cell execution

### Kernel Management

- **[KernelManager](./classes/KernelManager.md)** - Kernel startup and shutdown
- **[KernelClient](./classes/KernelClient.md)** - Message communication
- **[KernelSpecManager](./classes/KernelSpecManager.md)** - Kernel specification management

## Core Methods

### Execution Methods

- **[execute()](./methods/execute.md)** - Execute all cells in notebook
- **[execute_cell()](./methods/execute_cell.md)** - Execute single cell
- **[execute_cells()](./methods/execute_cells.md)** - Execute specific cells
- **[reset_execution_trackers()](./methods/reset_execution_trackers.md)** - Reset execution state

### Kernel Management

- **[start_new_kernel()](./methods/start_new_kernel.md)** - Start new kernel instance
- **[shutdown_kernel()](./methods/shutdown_kernel.md)** - Shutdown kernel gracefully
- **[restart_kernel()](./methods/restart_kernel.md)** - Restart kernel with same spec

### Configuration

- **[set_kernel_name()](./methods/set_kernel_name.md)** - Set kernel specification
- **[set_timeout()](./methods/set_timeout.md)** - Set execution timeout
- **[set_allow_errors()](./methods/set_allow_errors.md)** - Configure error handling

## Common Usage Patterns

### Basic Execution

```python
from nbclient import NotebookClient
import nbformat

# Load and execute notebook
with open('analysis.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb)
client.execute()

# Save executed notebook
with open('executed_analysis.ipynb', 'w') as f:
    nbformat.write(nb, f)
```

### Asynchronous Execution

```python
from nbclient import AsyncNotebookClient
import nbformat
import asyncio

async def execute_notebook():
    with open('analysis.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    
    client = AsyncNotebookClient(nb)
    await client.execute()
    
    return nb

# Run async execution
nb = asyncio.run(execute_notebook())
```

### Selective Cell Execution

```python
from nbclient import NotebookClient
import nbformat

with open('notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb)

# Execute only specific cells
client.execute_cells(['cell_1', 'cell_3'])

# Execute cells by index
client.execute_cells([0, 2, 4])

# Execute cells by tag
client.execute_cells(tag='analysis')
```

### Error Handling

```python
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat

with open('notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, allow_errors=True)

try:
    client.execute()
except CellExecutionError as e:
    print(f"Cell {e.cell_index} failed: {e.error}")
    # Continue with remaining cells
```

### Custom Kernel Configuration

```python
from nbclient import NotebookClient
import nbformat

with open('notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    kernel_name='python3',
    timeout=300,  # 5 minutes
    allow_errors=False,
    record_timing=True
)

client.execute()
```

### Integration with papermill

```python
from nbclient import NotebookClient
import nbformat
import papermill as pm

# Use papermill for parameter injection
pm.execute_notebook(
    'template.ipynb',
    'output.ipynb',
    parameters={'input_file': 'data.csv'}
)

# Use nbclient for additional execution
with open('output.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb)
client.execute()

# Save final result
with open('final_output.ipynb', 'w') as f:
    nbformat.write(nb, f)
```

## Configuration Options

### Execution Parameters

- **timeout**: Maximum execution time per cell (default: 300s)
- **allow_errors**: Continue execution on cell errors (default: False)
- **record_timing**: Record execution timing (default: False)
- **kernel_name**: Kernel specification name (default: auto-detect)
- **resources**: Additional kernel resources

### Kernel Parameters

- **kernel_spec**: Kernel specification object
- **kernel_name**: Kernel name string
- **extra_arguments**: Additional kernel arguments
- **env**: Environment variables for kernel

### Output Parameters

- **output_path**: Path for output notebook
- **output_format**: Output format specification
- **metadata**: Additional notebook metadata

## Error Handling

### Exception Types

- **CellExecutionError**: Individual cell execution failure
- **KernelError**: Kernel communication or startup error
- **TimeoutError**: Cell execution timeout
- **NotebookError**: General notebook processing error

### Error Recovery

```python
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, KernelError
import nbformat

def execute_with_recovery(notebook_path, max_retries=3):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    for attempt in range(max_retries):
        try:
            client = NotebookClient(nb)
            client.execute()
            return nb
        except KernelError:
            # Restart kernel and retry
            client.shutdown_kernel()
            continue
        except CellExecutionError as e:
            # Handle specific cell errors
            print(f"Cell {e.cell_index} failed: {e.error}")
            if not client.allow_errors:
                raise
```

## Performance Optimization

### Parallel Execution

```python
from nbclient import NotebookClient
import nbformat
from concurrent.futures import ThreadPoolExecutor

def execute_notebook_parallel(notebook_paths):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for path in notebook_paths:
            with open(path) as f:
                nb = nbformat.read(f, as_version=4)
            client = NotebookClient(nb)
            futures.append(executor.submit(client.execute))
        
        results = [future.result() for future in futures]
        return results
```

### Memory Management

```python
from nbclient import NotebookClient
import nbformat

# Execute with memory limits
client = NotebookClient(
    nb,
    kernel_name='python3',
    resources={'memory_limit': '512m'}
)

client.execute()
```

### Caching Results

```python
from nbclient import NotebookClient
import nbformat
import hashlib
import json

def get_notebook_hash(nb):
    content = json.dumps(nb, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def execute_with_cache(notebook_path, cache_dir='.nbclient_cache'):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    nb_hash = get_notebook_hash(nb)
    cache_path = f"{cache_dir}/{nb_hash}.ipynb"
    
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return nbformat.read(f, as_version=4)
    
    client = NotebookClient(nb)
    client.execute()
    
    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        nbformat.write(nb, f)
    
    return nb
```

## Integration Examples

### With pandas for data analysis

```python
from nbclient import NotebookClient
import nbformat
import pandas as pd

# Create notebook with pandas analysis
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_code_cell("import pandas as pd"),
    nbformat.v4.new_code_cell("df = pd.read_csv('data.csv')"),
    nbformat.v4.new_code_cell("print(df.head())"),
    nbformat.v4.new_code_cell("print(df.describe())")
]

client = NotebookClient(nb)
client.execute()
```

### With matplotlib for visualization

```python
from nbclient import NotebookClient
import nbformat
import matplotlib.pyplot as plt

# Create notebook with plotting
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_code_cell("import matplotlib.pyplot as plt"),
    nbformat.v4.new_code_cell("import numpy as np"),
    nbformat.v4.new_code_cell("x = np.linspace(0, 10, 100)"),
    nbformat.v4.new_code_cell("y = np.sin(x)"),
    nbformat.v4.new_code_cell("plt.plot(x, y)"),
    nbformat.v4.new_code_cell("plt.title('Sine Wave')"),
    nbformat.v4.new_code_cell("plt.show()")
]

client = NotebookClient(nb)
client.execute()
```

### With scrapbook for data persistence

```python
from nbclient import NotebookClient
import nbformat
import scrapbook as sb

# Create notebook with scrapbook
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_code_cell("import scrapbook as sb"),
    nbformat.v4.new_code_cell("import pandas as pd"),
    nbformat.v4.new_code_cell("df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})"),
    nbformat.v4.new_code_cell("sb.glue('my_data', df)")
]

client = NotebookClient(nb)
client.execute()

# Read scrapbook data
loaded_nb = sb.read_notebook(nb)
data = loaded_nb.scraps.get("my_data")
```

## External Resources

- [nbclient Documentation](https://nbclient.readthedocs.io/)
- [GitHub Repository](https://github.com/jupyter/nbclient)
- [Jupyter Project](https://jupyter.org/)

______________________________________________________________________

*This documentation covers nbclient version 0.8.x. For older versions, see the [legacy documentation](./legacy/).*
