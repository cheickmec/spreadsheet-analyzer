"""Type definitions for the tools system.

This module defines protocols and types for LangChain tools,
following functional programming principles.

CLAUDE-KNOWLEDGE: Tools in LangChain are functions that agents can
call. This module provides a functional wrapper around tool concepts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ..core.errors import ToolError
from ..core.types import Err, Ok, Option, Result, err, ok

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


@dataclass(frozen=True)
class ToolMetadata:
    """Immutable tool metadata."""

    name: str
    description: str
    args_schema: type[BaseModel]
    return_type: type
    category: str  # "excel", "notebook", "analysis", etc.
    requires: tuple[str, ...] = ()  # Required capabilities
    tags: tuple[str, ...] = ()

    @property
    def full_name(self) -> str:
        """Get full tool name with category."""
        return f"{self.category}.{self.name}"


@dataclass(frozen=True)
class ToolCall:
    """Immutable representation of a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = None

    def validate_args(self, schema: type[BaseModel]) -> Result[BaseModel, ToolError]:
        """Validate arguments against schema."""
        try:
            validated = schema(**self.arguments)
            return ok(validated)
        except Exception as e:
            return err(
                ToolError(
                    f"Invalid arguments for tool {self.tool_name}",
                    tool_name=self.tool_name,
                    details={"arguments": self.arguments},
                    cause=e,
                )
            )


@dataclass(frozen=True)
class ToolResult:
    """Result from tool execution."""

    tool_name: str
    success: bool
    output: Any
    error: ToolError | None = None
    metadata: dict[str, Any] = None

    @classmethod
    def success_result(cls, tool_name: str, output: Any, metadata: dict[str, Any] | None = None) -> "ToolResult":
        """Create a successful result."""
        return cls(tool_name=tool_name, success=True, output=output, error=None, metadata=metadata)

    @classmethod
    def error_result(cls, tool_name: str, error: ToolError, metadata: dict[str, Any] | None = None) -> "ToolResult":
        """Create an error result."""
        return cls(tool_name=tool_name, success=False, output=None, error=error, metadata=metadata)


class Tool(Protocol[T, R]):
    """Protocol for tool implementations.

    Tools should be implemented as pure functions that return
    Results for proper error handling.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        ...

    def execute(self, args: T) -> Result[R, ToolError]:
        """Execute the tool with validated arguments.

        This should be a pure function when possible, with
        side effects isolated and documented.
        """
        ...


# Functional tool wrapper
@dataclass(frozen=True)
class FunctionalTool:
    """Functional wrapper for tools."""

    metadata: ToolMetadata
    execute_fn: Callable[[Any], Result[Any, ToolError]]

    def execute(self, args: BaseModel) -> Result[Any, ToolError]:
        """Execute the tool."""
        return self.execute_fn(args)

    def to_langchain(self):
        """Convert to LangChain tool format.

        This is an adapter for LangChain compatibility.
        """
        from langchain.tools import StructuredTool

        def wrapper(**kwargs):
            # Convert kwargs to Pydantic model
            try:
                args = self.metadata.args_schema(**kwargs)
                result = self.execute(args)

                if isinstance(result, Ok):
                    return result.value
                else:
                    raise Exception(str(result.error))
            except Exception as e:
                raise Exception(f"Tool execution failed: {e}")

        return StructuredTool(
            name=self.metadata.name,
            description=self.metadata.description,
            func=wrapper,
            args_schema=self.metadata.args_schema,
        )


class ToolRegistry(Protocol):
    """Protocol for tool registries."""

    def register(self, tool: Tool) -> Result[None, ToolError]:
        """Register a tool."""
        ...

    def get(self, name: str) -> Option[Tool]:
        """Get a tool by name."""
        ...

    def list_tools(self, category: str | None = None) -> list[ToolMetadata]:
        """List available tools."""
        ...

    def search(self, query: str) -> list[ToolMetadata]:
        """Search for tools by description or tags."""
        ...


# Tool creation helpers
def create_tool(
    name: str,
    description: str,
    args_schema: type[T],
    execute_fn: Callable[[T], Result[R, ToolError]],
    category: str = "general",
    return_type: type | None = None,
    requires: list[str] | None = None,
    tags: list[str] | None = None,
) -> FunctionalTool:
    """Create a functional tool from components."""
    metadata = ToolMetadata(
        name=name,
        description=description,
        args_schema=args_schema,
        return_type=return_type or type(None),
        category=category,
        requires=tuple(requires or []),
        tags=tuple(tags or []),
    )

    return FunctionalTool(metadata=metadata, execute_fn=execute_fn)


# Composite tool types
@dataclass(frozen=True)
class ToolChain:
    """Chain of tools to execute in sequence."""

    tools: tuple[Tool, ...]

    def execute(self, initial_args: Any) -> Result[list[ToolResult], ToolError]:
        """Execute all tools in sequence."""
        results = []
        current_input = initial_args

        for tool in self.tools:
            # Validate and execute
            if hasattr(current_input, "dict"):
                args_dict = current_input.dict()
            else:
                args_dict = {"input": current_input}

            try:
                args = tool.metadata.args_schema(**args_dict)
                result = tool.execute(args)

                if isinstance(result, Err):
                    return err(result.error)

                tool_result = ToolResult.success_result(tool_name=tool.metadata.name, output=result.value)
                results.append(tool_result)
                current_input = result.value

            except Exception as e:
                return err(
                    ToolError(
                        f"Failed to execute tool {tool.metadata.name} in chain", tool_name=tool.metadata.name, cause=e
                    )
                )

        return ok(results)


@dataclass(frozen=True)
class ToolCondition:
    """Conditional tool execution."""

    condition: Callable[[Any], bool]
    if_true: Tool
    if_false: Tool | None = None

    def execute(self, args: Any) -> Result[Any, ToolError]:
        """Execute based on condition."""
        if self.condition(args):
            return self.if_true.execute(args)
        elif self.if_false:
            return self.if_false.execute(args)
        else:
            return ok(None)
