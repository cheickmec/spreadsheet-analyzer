"""Tool composition utilities.

This module provides functional utilities for composing tools
into more complex workflows.

CLAUDE-KNOWLEDGE: Tool composition allows building complex workflows
from simple tools while maintaining functional principles.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from ..core.errors import ToolError
from ..core.functional import compose, pipe
from ..core.types import Result, err, ok
from .types import FunctionalTool, Tool, ToolChain, ToolCondition, ToolMetadata


T = TypeVar("T")
R = TypeVar("R")


# Tool combinators
def chain_tools(*tools: Tool) -> ToolChain:
    """Chain tools to execute in sequence."""
    return ToolChain(tools=tools)


def conditional_tool(
    condition: Callable[[Any], bool],
    if_true: Tool,
    if_false: Tool | None = None
) -> ToolCondition:
    """Create a conditional tool execution."""
    return ToolCondition(
        condition=condition,
        if_true=if_true,
        if_false=if_false
    )


@dataclass(frozen=True)
class ParallelTools:
    """Execute multiple tools in parallel and combine results."""
    
    tools: tuple[Tool, ...]
    combiner: Callable[[list[Any]], Any]
    
    def execute(self, args: Any) -> Result[Any, ToolError]:
        """Execute all tools and combine results."""
        results = []
        errors = []
        
        for tool in self.tools:
            try:
                # Validate args for each tool
                validated_args = tool.metadata.args_schema(**args) if isinstance(args, dict) else args
                result = tool.execute(validated_args)
                
                if result.is_ok():
                    results.append(result.unwrap())
                else:
                    errors.append(result.unwrap_err())
            except Exception as e:
                errors.append(ToolError(
                    f"Failed to execute {tool.metadata.name}",
                    tool_name=tool.metadata.name,
                    cause=e
                ))
        
        if errors:
            return err(ToolError(
                "Some tools failed in parallel execution",
                details={"errors": errors, "successful": len(results)}
            ))
        
        try:
            combined = self.combiner(results)
            return ok(combined)
        except Exception as e:
            return err(ToolError(
                "Failed to combine parallel results",
                cause=e
            ))


def parallel_tools(tools: list[Tool], combiner: Callable[[list[Any]], Any]) -> ParallelTools:
    """Create parallel tool execution."""
    return ParallelTools(tools=tuple(tools), combiner=combiner)


@dataclass(frozen=True)
class MappedTool:
    """Apply a tool to each item in a list."""
    
    tool: Tool
    
    def execute(self, items: list[Any]) -> Result[list[Any], ToolError]:
        """Execute tool on each item."""
        results = []
        
        for i, item in enumerate(items):
            try:
                result = self.tool.execute(item)
                if result.is_err():
                    return err(ToolError(
                        f"Failed on item {i}",
                        details={"index": i, "error": result.unwrap_err()}
                    ))
                results.append(result.unwrap())
            except Exception as e:
                return err(ToolError(
                    f"Failed on item {i}",
                    details={"index": i},
                    cause=e
                ))
        
        return ok(results)


def map_tool(tool: Tool) -> MappedTool:
    """Create a mapped tool for list processing."""
    return MappedTool(tool=tool)


@dataclass(frozen=True)
class FilteredTool:
    """Apply tool only to items matching a predicate."""
    
    tool: Tool
    predicate: Callable[[Any], bool]
    
    def execute(self, items: list[Any]) -> Result[list[tuple[int, Any]], ToolError]:
        """Execute tool on filtered items."""
        results = []
        
        for i, item in enumerate(items):
            if self.predicate(item):
                try:
                    result = self.tool.execute(item)
                    if result.is_err():
                        return err(ToolError(
                            f"Failed on item {i}",
                            details={"index": i, "error": result.unwrap_err()}
                        ))
                    results.append((i, result.unwrap()))
                except Exception as e:
                    return err(ToolError(
                        f"Failed on item {i}",
                        details={"index": i},
                        cause=e
                    ))
        
        return ok(results)


def filter_tool(tool: Tool, predicate: Callable[[Any], bool]) -> FilteredTool:
    """Create a filtered tool."""
    return FilteredTool(tool=tool, predicate=predicate)


@dataclass(frozen=True)
class RetryTool:
    """Tool with retry logic."""
    
    tool: Tool
    max_retries: int = 3
    retry_predicate: Callable[[ToolError], bool] | None = None
    
    def execute(self, args: Any) -> Result[Any, ToolError]:
        """Execute with retries."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            result = self.tool.execute(args)
            
            if result.is_ok():
                return result
            
            last_error = result.unwrap_err()
            
            # Check if we should retry
            if self.retry_predicate and not self.retry_predicate(last_error):
                return result
            
            # Don't retry on last attempt
            if attempt == self.max_retries:
                break
        
        return err(ToolError(
            f"Failed after {self.max_retries} retries",
            tool_name=self.tool.metadata.name,
            details={"attempts": self.max_retries + 1},
            cause=last_error
        ))


def retry_tool(tool: Tool, max_retries: int = 3, retry_predicate: Callable[[ToolError], bool] | None = None) -> RetryTool:
    """Create a tool with retry logic."""
    return RetryTool(
        tool=tool,
        max_retries=max_retries,
        retry_predicate=retry_predicate
    )


# Tool transformation functions
def transform_input(tool: Tool, transformer: Callable[[Any], Any]) -> FunctionalTool:
    """Transform tool input before execution."""
    def execute(args: Any) -> Result[Any, ToolError]:
        try:
            transformed = transformer(args)
            return tool.execute(transformed)
        except Exception as e:
            return err(ToolError(
                "Failed to transform input",
                tool_name=tool.metadata.name,
                cause=e
            ))
    
    metadata = ToolMetadata(
        name=f"{tool.metadata.name}_transformed_input",
        description=f"{tool.metadata.description} (with transformed input)",
        args_schema=tool.metadata.args_schema,
        return_type=tool.metadata.return_type,
        category=tool.metadata.category,
        requires=tool.metadata.requires,
        tags=tool.metadata.tags + ("transformed",)
    )
    
    return FunctionalTool(metadata=metadata, execute_fn=execute)


def transform_output(tool: Tool, transformer: Callable[[Any], Any]) -> FunctionalTool:
    """Transform tool output after execution."""
    def execute(args: Any) -> Result[Any, ToolError]:
        result = tool.execute(args)
        if result.is_err():
            return result
        
        try:
            transformed = transformer(result.unwrap())
            return ok(transformed)
        except Exception as e:
            return err(ToolError(
                "Failed to transform output",
                tool_name=tool.metadata.name,
                cause=e
            ))
    
    metadata = ToolMetadata(
        name=f"{tool.metadata.name}_transformed_output",
        description=f"{tool.metadata.description} (with transformed output)",
        args_schema=tool.metadata.args_schema,
        return_type=type(None),  # Unknown after transformation
        category=tool.metadata.category,
        requires=tool.metadata.requires,
        tags=tool.metadata.tags + ("transformed",)
    )
    
    return FunctionalTool(metadata=metadata, execute_fn=execute)


# Higher-order tool functions
def fallback_tool(primary: Tool, fallback: Tool) -> FunctionalTool:
    """Use fallback tool if primary fails."""
    def execute(args: Any) -> Result[Any, ToolError]:
        result = primary.execute(args)
        if result.is_ok():
            return result
        
        # Try fallback
        fallback_result = fallback.execute(args)
        if fallback_result.is_ok():
            return fallback_result
        
        # Both failed
        return err(ToolError(
            "Both primary and fallback tools failed",
            details={
                "primary_error": result.unwrap_err(),
                "fallback_error": fallback_result.unwrap_err()
            }
        ))
    
    metadata = ToolMetadata(
        name=f"{primary.metadata.name}_with_fallback",
        description=f"{primary.metadata.description} (with fallback to {fallback.metadata.name})",
        args_schema=primary.metadata.args_schema,
        return_type=primary.metadata.return_type,
        category=primary.metadata.category,
        requires=tuple(set(primary.metadata.requires) | set(fallback.metadata.requires)),
        tags=primary.metadata.tags + ("fallback",)
    )
    
    return FunctionalTool(metadata=metadata, execute_fn=execute)


def cached_tool(tool: Tool, cache_key_fn: Callable[[Any], str]) -> FunctionalTool:
    """Create a cached version of a tool."""
    cache: dict[str, Any] = {}
    
    def execute(args: Any) -> Result[Any, ToolError]:
        key = cache_key_fn(args)
        
        if key in cache:
            return ok(cache[key])
        
        result = tool.execute(args)
        if result.is_ok():
            cache[key] = result.unwrap()
        
        return result
    
    metadata = ToolMetadata(
        name=f"{tool.metadata.name}_cached",
        description=f"{tool.metadata.description} (with caching)",
        args_schema=tool.metadata.args_schema,
        return_type=tool.metadata.return_type,
        category=tool.metadata.category,
        requires=tool.metadata.requires,
        tags=tool.metadata.tags + ("cached",)
    )
    
    return FunctionalTool(metadata=metadata, execute_fn=execute)


# Tool workflow builder
@dataclass(frozen=True)
class ToolWorkflow:
    """Build complex tool workflows."""
    
    steps: list[tuple[str, Tool | ToolChain | ParallelTools | ToolCondition]]
    
    def execute(self, initial_args: dict[str, Any]) -> Result[dict[str, Any], ToolError]:
        """Execute workflow steps."""
        context = dict(initial_args)
        
        for step_name, step_tool in self.steps:
            # Execute step
            if isinstance(step_tool, (Tool, FunctionalTool)):
                result = step_tool.execute(context)
            elif isinstance(step_tool, (ToolChain, ParallelTools, ToolCondition)):
                result = step_tool.execute(context)
            else:
                return err(ToolError(f"Unknown step type: {type(step_tool)}"))
            
            if result.is_err():
                return err(ToolError(
                    f"Workflow failed at step '{step_name}'",
                    details={"step": step_name, "error": result.unwrap_err()}
                ))
            
            # Update context with result
            context[step_name] = result.unwrap()
        
        return ok(context)
    
    def add_step(self, name: str, tool: Tool) -> "ToolWorkflow":
        """Add a step to the workflow."""
        new_steps = self.steps + [(name, tool)]
        return ToolWorkflow(steps=new_steps)


def create_workflow() -> ToolWorkflow:
    """Create an empty workflow."""
    return ToolWorkflow(steps=[])