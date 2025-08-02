"""Tool registry implementation.

This module provides a functional registry for discovering and
managing tools in the system.

CLAUDE-KNOWLEDGE: The registry pattern allows dynamic tool discovery
and composition while maintaining functional principles.
"""

from dataclasses import dataclass, field
from typing import Any

from ..core.types import Option, Result, err, none, ok, some
from ..core.errors import ToolError
from .types import FunctionalTool, Tool, ToolMetadata, ToolRegistry


@dataclass(frozen=True)
class ImmutableToolRegistry:
    """Immutable tool registry implementation."""
    
    tools: dict[str, Tool] = field(default_factory=dict)
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    
    def register(self, tool: Tool) -> Result["ImmutableToolRegistry", ToolError]:
        """Register a new tool, returning a new registry."""
        if tool.metadata.name in self.tools:
            return err(ToolError(
                f"Tool already registered: {tool.metadata.name}",
                tool_name=tool.metadata.name
            ))
        
        # Update tools dict
        new_tools = {**self.tools, tool.metadata.name: tool}
        
        # Update categories
        category = tool.metadata.category
        current_tools = self.categories.get(category, ())
        new_categories = {
            **self.categories,
            category: current_tools + (tool.metadata.name,)
        }
        
        return ok(ImmutableToolRegistry(
            tools=new_tools,
            categories=new_categories
        ))
    
    def get(self, name: str) -> Option[Tool]:
        """Get a tool by name."""
        return some(self.tools[name]) if name in self.tools else none()
    
    def list_tools(self, category: str | None = None) -> list[ToolMetadata]:
        """List available tools."""
        if category:
            tool_names = self.categories.get(category, ())
            return [
                self.tools[name].metadata
                for name in tool_names
                if name in self.tools
            ]
        else:
            return [tool.metadata for tool in self.tools.values()]
    
    def search(self, query: str) -> list[ToolMetadata]:
        """Search for tools by description or tags."""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            metadata = tool.metadata
            
            # Check name
            if query_lower in metadata.name.lower():
                results.append(metadata)
                continue
            
            # Check description
            if query_lower in metadata.description.lower():
                results.append(metadata)
                continue
            
            # Check tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                results.append(metadata)
                continue
            
            # Check category
            if query_lower in metadata.category.lower():
                results.append(metadata)
        
        return results
    
    def get_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        tool_names = self.categories.get(category, ())
        return [
            self.tools[name]
            for name in tool_names
            if name in self.tools
        ]
    
    def get_categories(self) -> list[str]:
        """Get all registered categories."""
        return list(self.categories.keys())


# Factory functions
def create_registry(tools: list[Tool] | None = None) -> ImmutableToolRegistry:
    """Create a new tool registry."""
    registry = ImmutableToolRegistry()
    
    if tools:
        for tool in tools:
            result = registry.register(tool)
            if result.is_ok():
                registry = result.unwrap()
            else:
                # Log error but continue
                print(f"Failed to register tool: {result.unwrap_err()}")
    
    return registry


def create_default_registry() -> ImmutableToolRegistry:
    """Create registry with default tools."""
    from .impl.excel_tools import (
        create_cell_reader_tool,
        create_formula_analyzer_tool,
        create_range_reader_tool,
        create_sheet_reader_tool,
        create_workbook_reader_tool,
    )
    from .impl.notebook_tools import (
        create_cell_executor_tool,
        create_markdown_generator_tool,
        create_notebook_builder_tool,
        create_notebook_saver_tool,
    )
    
    tools = [
        # Excel tools
        create_cell_reader_tool(),
        create_range_reader_tool(),
        create_sheet_reader_tool(),
        create_workbook_reader_tool(),
        create_formula_analyzer_tool(),
        # Notebook tools
        create_cell_executor_tool(),
        create_notebook_builder_tool(),
        create_notebook_saver_tool(),
        create_markdown_generator_tool(),
    ]
    
    return create_registry(tools)


# Registry operations
@dataclass(frozen=True)
class RegistryOperations:
    """Functional operations on tool registries."""
    
    @staticmethod
    def merge(registry1: ImmutableToolRegistry, registry2: ImmutableToolRegistry) -> ImmutableToolRegistry:
        """Merge two registries, with registry2 taking precedence."""
        # Merge tools
        merged_tools = {**registry1.tools, **registry2.tools}
        
        # Merge categories
        merged_categories = {}
        all_categories = set(registry1.categories.keys()) | set(registry2.categories.keys())
        
        for category in all_categories:
            tools1 = set(registry1.categories.get(category, ()))
            tools2 = set(registry2.categories.get(category, ()))
            merged_categories[category] = tuple(tools1 | tools2)
        
        return ImmutableToolRegistry(
            tools=merged_tools,
            categories=merged_categories
        )
    
    @staticmethod
    def filter_by_tags(registry: ImmutableToolRegistry, tags: list[str]) -> ImmutableToolRegistry:
        """Create a new registry with only tools matching given tags."""
        filtered_tools = {}
        filtered_categories = {}
        
        for name, tool in registry.tools.items():
            if any(tag in tool.metadata.tags for tag in tags):
                filtered_tools[name] = tool
                
                category = tool.metadata.category
                if category not in filtered_categories:
                    filtered_categories[category] = ()
                filtered_categories[category] = filtered_categories[category] + (name,)
        
        return ImmutableToolRegistry(
            tools=filtered_tools,
            categories=filtered_categories
        )
    
    @staticmethod
    def filter_by_requirements(registry: ImmutableToolRegistry, available_capabilities: set[str]) -> ImmutableToolRegistry:
        """Filter tools by available capabilities."""
        filtered_tools = {}
        filtered_categories = {}
        
        for name, tool in registry.tools.items():
            required = set(tool.metadata.requires)
            if required.issubset(available_capabilities):
                filtered_tools[name] = tool
                
                category = tool.metadata.category
                if category not in filtered_categories:
                    filtered_categories[category] = ()
                filtered_categories[category] = filtered_categories[category] + (name,)
        
        return ImmutableToolRegistry(
            tools=filtered_tools,
            categories=filtered_categories
        )


# Singleton pattern for global registry (optional)
_global_registry: ImmutableToolRegistry | None = None


def get_global_registry() -> ImmutableToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = create_default_registry()
    return _global_registry


def set_global_registry(registry: ImmutableToolRegistry) -> None:
    """Set the global tool registry."""
    global _global_registry
    _global_registry = registry