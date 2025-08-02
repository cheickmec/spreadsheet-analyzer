"""Tools system for spreadsheet analysis.

This package provides a functional tool system for LangChain integration,
with support for Excel operations, notebook management, and analysis tasks.
"""

from .composition import (
    MappedTool,
    ParallelTools,
    RetryTool,
    ToolWorkflow,
    cached_tool,
    chain_tools,
    conditional_tool,
    create_workflow,
    fallback_tool,
    filter_tool,
    map_tool,
    parallel_tools,
    retry_tool,
    transform_input,
    transform_output,
)

# Import concrete tools
from .impl import (
    create_cell_executor_tool,
    create_cell_reader_tool,
    create_formula_analyzer_tool,
    create_markdown_generator_tool,
    create_notebook_builder_tool,
    create_notebook_saver_tool,
    create_range_reader_tool,
    create_sheet_reader_tool,
    create_workbook_reader_tool,
)
from .registry import (
    ImmutableToolRegistry,
    RegistryOperations,
    create_default_registry,
    create_registry,
    get_global_registry,
    set_global_registry,
)
from .types import (
    FunctionalTool,
    Tool,
    ToolCall,
    ToolChain,
    ToolCondition,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
    create_tool,
)

__all__ = [
    # Composition
    "MappedTool",
    "ParallelTools",
    "RetryTool",
    "ToolWorkflow",
    "cached_tool",
    "chain_tools",
    "conditional_tool",
    # Excel tools
    "create_cell_executor_tool",
    "create_cell_reader_tool",
    "create_default_registry",
    "create_formula_analyzer_tool",
    "create_markdown_generator_tool",
    "create_notebook_builder_tool",
    "create_notebook_saver_tool",
    "create_range_reader_tool",
    "create_registry",
    "create_sheet_reader_tool",
    "create_tool",
    "create_workbook_reader_tool",
    "create_workflow",
    "fallback_tool",
    "filter_tool",
    # Types
    "FunctionalTool",
    "get_global_registry",
    # Registry
    "ImmutableToolRegistry",
    "map_tool",
    "parallel_tools",
    "RegistryOperations",
    "retry_tool",
    "set_global_registry",
    "Tool",
    "ToolCall",
    "ToolChain",
    "ToolCondition",
    "ToolMetadata",
    "ToolRegistry",
    "ToolResult",
    "transform_input",
    "transform_output",
]
