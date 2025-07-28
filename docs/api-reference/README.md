# Jupyter Ecosystem API Reference

## Overview

This directory contains comprehensive API documentation for all Jupyter ecosystem packages used in the spreadsheet analyzer project. Each package is documented with its complete API surface, including all classes, methods, functions, and their parameters.

## 📚 API Documentation Structure

### Core Jupyter Packages

#### **Foundation Layer**

- **[jupyter_core](./jupyter-core/)** - Core infrastructure for paths, configuration, and subcommand discovery
- **[jupyter_client](./jupyter-client/)** - Kernel communication protocol and client implementation
- **[jupyter_protocol](./jupyter-protocol/)** - Messaging protocol specification

#### **Notebook Format & Processing**

- **[nbformat](./nbformat/)** - Reading, writing, and validating `.ipynb` files
- **[nbclient](./nbclient/)** - Executing notebooks programmatically
- **[nbconvert](./nbconvert/)** - Converting notebooks to other formats

#### **Kernel Management**

- **[jupyter_kernel_gateway](./jupyter-kernel-gateway/)** - HTTP API for kernel management
- **[jupyter_kernel_spec](./jupyter-kernel-spec/)** - Kernel specification management

### Extended Ecosystem

#### **Notebook Enhancement**

- **[papermill](./papermill/)** - Parameterized notebook execution
- **[scrapbook](./scrapbook/)** - Data persistence and cross-notebook communication
- **[jupytext](./jupytext/)** - Notebook format conversion and synchronization

#### **Documentation & Publishing**

- **[papyri](./papyri/)** - Documentation generation and cross-linking
- **[jupyter_book](./jupyter-book/)** - Creating publication-quality books from notebooks
- **[jupyter_contrib_nbextensions](./jupyter-contrib-nbextensions/)** - Community-contributed extensions

#### **Development & Testing**

- **[jupyter_server](./jupyter-server/)** - Server-side Jupyter functionality
- **[jupyter_lsp](./jupyter-lsp/)** - Language Server Protocol support
- **[jupyter_packaging](./jupyter-packaging/)** - Packaging utilities for Jupyter extensions

## Documentation Standards

Each package documentation follows a standardized format:

### File Structure

```
package-name/
├── README.md              # Package overview and quick start
├── classes/               # Class documentation
│   ├── ClassName.md      # Individual class documentation
│   └── ...
├── functions/             # Function documentation
│   ├── function_name.md  # Individual function documentation
│   └── ...
├── modules/               # Module documentation
│   ├── module_name.md    # Individual module documentation
│   └── ...
└── examples/              # Usage examples
    ├── basic_usage.md    # Basic usage patterns
    ├── advanced_usage.md # Advanced usage patterns
    └── integration.md    # Integration with other packages
```

### Documentation Template

Each API endpoint follows this structure:

````markdown
# Function/Class Name

## Description
Brief description of the function/class purpose.

## Signature
```python
function_name(param1: type, param2: type = default) -> return_type
````

## Parameters

- **param1** (`type`): Description of parameter
- **param2** (`type`, optional): Description of optional parameter

## Returns

- **return_type**: Description of return value

## Raises

- **ExceptionType**: When and why this exception is raised

## Examples

```python
# Basic usage example
result = function_name(value1, value2)
```

## Notes

Additional implementation details, performance considerations, etc.

## See Also

- Related functions or classes
- External documentation links

```

## Integration with Research Documentation

This API reference complements the research documentation in `../research/jupyter-ecosystem/`:

- **Research docs**: High-level architecture, design patterns, and ecosystem relationships
- **API docs**: Detailed implementation reference for all public APIs

## Contributing

When adding new package documentation:

1. Create a new directory for the package
2. Follow the standardized documentation template
3. Include comprehensive examples
4. Link to official documentation where available
5. Update this README with the new package entry

## Quick Navigation

### By Category
- [Core Infrastructure](./core-infrastructure/)
- [Notebook Processing](./notebook-processing/)
- [Kernel Management](./kernel-management/)
- [Documentation Tools](./documentation-tools/)
- [Development Tools](./development-tools/)

### By Package Name
- [jupyter_core](./jupyter-core/)
- [jupyter_client](./jupyter-client/)
- [nbformat](./nbformat/)
- [nbclient](./nbclient/)
- [nbconvert](./nbconvert/)
- [papermill](./papermill/)
- [scrapbook](./scrapbook/)
- [papyri](./papyri/)

## External Resources

- [Jupyter Official Documentation](https://jupyter.org/documentation)
- [Jupyter GitHub Organization](https://github.com/jupyter)
- [Jupyter Discourse Community](https://discourse.jupyter.org/)
- [Jupyter Enhancement Proposals (JEPs)](https://jupyter.org/enhancement-proposals/)

---

*This API reference is maintained as part of the spreadsheet analyzer project to ensure comprehensive documentation of all Jupyter ecosystem dependencies.* 
```
