"""Tools system for LangChain integration.

This package provides functional wrappers around tools that agents
can use, including Excel manipulation, notebook operations, and
analysis utilities.
"""

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
    "FunctionalTool",
    # Core types
    "Tool",
    "ToolCall",
    # Composition
    "ToolChain",
    "ToolCondition",
    "ToolMetadata",
    "ToolRegistry",
    "ToolResult",
    # Helpers
    "create_tool",
]
