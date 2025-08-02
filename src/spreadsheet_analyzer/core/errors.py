"""Error types for the spreadsheet analyzer.

This module defines error types used throughout the application,
following a functional approach with immutable error structures.

CLAUDE-KNOWLEDGE: Error types are designed to be informative and
composable, allowing for detailed error tracking without exceptions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ErrorCategory(Enum):
    """Categories of errors in the system."""

    VALIDATION = auto()
    IO = auto()
    PARSING = auto()
    ANALYSIS = auto()
    CONFIGURATION = auto()
    AGENT = auto()
    CONTEXT = auto()
    TOOL = auto()
    LLM = auto()


@dataclass(frozen=True)
class BaseError:
    """Base error type with common fields."""

    message: str
    category: ErrorCategory
    details: dict[str, Any] | None = None
    cause: Exception | None = None

    def __str__(self) -> str:
        """Human-readable error message."""
        base = f"[{self.category.name}] {self.message}"
        if self.details:
            base += f" - Details: {self.details}"
        if self.cause:
            base += f" - Caused by: {type(self.cause).__name__}: {self.cause!s}"
        return base


# Specific error types
@dataclass(frozen=True)
class ValidationError(BaseError):
    """Error during validation of inputs or data."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, cause: Exception | None = None):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.VALIDATION)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class IOError(BaseError):
    """Error during I/O operations."""

    path: str | None = None

    def __init__(
        self,
        message: str,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.IO)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class ParsingError(BaseError):
    """Error during parsing operations."""

    location: str | None = None

    def __init__(
        self,
        message: str,
        location: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.PARSING)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class AnalysisError(BaseError):
    """Error during analysis operations."""

    sheet_name: str | None = None
    cell_reference: str | None = None

    def __init__(
        self,
        message: str,
        sheet_name: str | None = None,
        cell_reference: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.ANALYSIS)
        object.__setattr__(self, "sheet_name", sheet_name)
        object.__setattr__(self, "cell_reference", cell_reference)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class ConfigurationError(BaseError):
    """Error in configuration."""

    config_key: str | None = None

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.CONFIGURATION)
        object.__setattr__(self, "config_key", config_key)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class AgentError(BaseError):
    """Error in agent operations."""

    agent_id: str | None = None

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.AGENT)
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class ContextError(BaseError):
    """Error in context management."""

    strategy_name: str | None = None
    token_count: int | None = None

    def __init__(
        self,
        message: str,
        strategy_name: str | None = None,
        token_count: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.CONTEXT)
        object.__setattr__(self, "strategy_name", strategy_name)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class ToolError(BaseError):
    """Error in tool execution."""

    tool_name: str | None = None

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.TOOL)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


@dataclass(frozen=True)
class LLMError(BaseError):
    """Error in LLM operations."""

    provider: str | None = None
    model: str | None = None

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "category", ErrorCategory.LLM)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "cause", cause)


# Error composition utilities
def chain_errors(errors: list[BaseError]) -> BaseError:
    """Combine multiple errors into a single error."""
    if not errors:
        return ValidationError("No errors to chain")

    if len(errors) == 1:
        return errors[0]

    # Determine primary category (most common)
    categories = [e.category for e in errors]
    primary_category = max(set(categories), key=categories.count)

    # Combine messages
    combined_message = "; ".join(e.message for e in errors)

    # Combine details
    combined_details = {
        "error_count": len(errors),
        "errors": [{"message": e.message, "category": e.category.name} for e in errors],
    }

    return BaseError(
        message=f"Multiple errors occurred: {combined_message}", category=primary_category, details=combined_details
    )


def error_to_dict(error: BaseError) -> dict[str, Any]:
    """Convert an error to a dictionary for serialization."""
    result = {
        "message": error.message,
        "category": error.category.name,
    }

    if error.details:
        result["details"] = error.details

    if error.cause:
        result["cause"] = {"type": type(error.cause).__name__, "message": str(error.cause)}

    # Add specific fields based on error type
    if isinstance(error, IOError) and error.path:
        result["path"] = error.path
    elif isinstance(error, ParsingError) and error.location:
        result["location"] = error.location
    elif isinstance(error, AnalysisError):
        if error.sheet_name:
            result["sheet_name"] = error.sheet_name
        if error.cell_reference:
            result["cell_reference"] = error.cell_reference
    elif isinstance(error, ConfigurationError) and error.config_key:
        result["config_key"] = error.config_key
    elif isinstance(error, AgentError) and error.agent_id:
        result["agent_id"] = error.agent_id
    elif isinstance(error, ContextError):
        if error.strategy_name:
            result["strategy_name"] = error.strategy_name
        if error.token_count is not None:
            result["token_count"] = error.token_count
    elif isinstance(error, ToolError) and error.tool_name:
        result["tool_name"] = error.tool_name
    elif isinstance(error, LLMError):
        if error.provider:
            result["provider"] = error.provider
        if error.model:
            result["model"] = error.model

    return result
