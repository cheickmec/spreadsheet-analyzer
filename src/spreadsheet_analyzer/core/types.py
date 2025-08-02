"""Core functional types for the spreadsheet analyzer.

This module implements Result, Option, and Either types following functional
programming patterns. These types enable safe error handling without exceptions
and explicit handling of nullable values.

CLAUDE-KNOWLEDGE: These types are inspired by Rust's Result and Option types,
and Haskell's Either type. They enforce explicit error handling at the type level.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, NoReturn, TypeVar, Union, cast

# Type variables
T = TypeVar("T")  # Success/Some type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Transform target type
L = TypeVar("L")  # Left type
R = TypeVar("R")  # Right type


# Result Type
@dataclass(frozen=True)
class Ok(Generic[T]):
    """Represents a successful result."""

    value: T

    def is_ok(self) -> bool:
        """Check if this is an Ok result."""
        return True

    def is_err(self) -> bool:
        """Check if this is an Err result."""
        return False

    def unwrap(self) -> T:
        """Extract the value. Safe for Ok."""
        return self.value

    def unwrap_err(self) -> NoReturn:
        """Attempt to extract error. Raises for Ok."""
        raise ValueError("Called unwrap_err on Ok value")

    def map(self, func: Callable[[T], U]) -> Result[U, Any]:
        """Transform the success value."""
        return Ok(func(self.value))

    def map_err(self, func: Callable[[Any], Any]) -> Result[T, Any]:
        """Transform the error value (no-op for Ok)."""
        return self

    def and_then(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that return Results."""
        return func(self.value)

    def or_else(self, func: Callable[[Any], Result[T, E]]) -> Result[T, E]:
        """Provide alternative for error (no-op for Ok)."""
        return cast("Result[T, E]", self)

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return self.value

    def unwrap_or_else(self, func: Callable[[Any], T]) -> T:
        """Extract value or compute default."""
        return self.value


@dataclass(frozen=True)
class Err(Generic[E]):
    """Represents an error result."""

    error: E

    def is_ok(self) -> bool:
        """Check if this is an Ok result."""
        return False

    def is_err(self) -> bool:
        """Check if this is an Err result."""
        return True

    def unwrap(self) -> NoReturn:
        """Attempt to extract value. Raises for Err."""
        raise ValueError(f"Called unwrap on Err value: {self.error}")

    def unwrap_err(self) -> E:
        """Extract the error. Safe for Err."""
        return self.error

    def map(self, func: Callable[[Any], Any]) -> Result[Any, E]:
        """Transform the success value (no-op for Err)."""
        return cast("Result[Any, E]", self)

    def map_err(self, func: Callable[[E], U]) -> Result[Any, U]:
        """Transform the error value."""
        return Err(func(self.error))

    def and_then(self, func: Callable[[Any], Result[Any, E]]) -> Result[Any, E]:
        """Chain operations that return Results (no-op for Err)."""
        return cast("Result[Any, E]", self)

    def or_else(self, func: Callable[[E], Result[T, U]]) -> Result[T, U]:
        """Provide alternative for error."""
        return func(self.error)

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return default

    def unwrap_or_else(self, func: Callable[[E], T]) -> T:
        """Extract value or compute default."""
        return func(self.error)


# Result is a union of Ok and Err
Result = Union[Ok[T], Err[E]]


# Option Type
@dataclass(frozen=True)
class Some(Generic[T]):
    """Represents a value that exists."""

    value: T

    def is_some(self) -> bool:
        """Check if this is Some."""
        return True

    def is_nothing(self) -> bool:
        """Check if this is Nothing."""
        return False

    def unwrap(self) -> T:
        """Extract the value. Safe for Some."""
        return self.value

    def map(self, func: Callable[[T], U]) -> Option[U]:
        """Transform the value."""
        return Some(func(self.value))

    def and_then(self, func: Callable[[T], Option[U]]) -> Option[U]:
        """Chain operations that return Options."""
        return func(self.value)

    def or_else(self, func: Callable[[], Option[T]]) -> Option[T]:
        """Provide alternative (no-op for Some)."""
        return self

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return self.value

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """Extract value or compute default."""
        return self.value


class _Nothing:
    """Represents absence of a value."""

    def is_some(self) -> bool:
        """Check if this is Some."""
        return False

    def is_nothing(self) -> bool:
        """Check if this is Nothing."""
        return True

    def unwrap(self) -> NoReturn:
        """Attempt to extract value. Raises for Nothing."""
        raise ValueError("Called unwrap on Nothing")

    def map(self, func: Callable[[Any], Any]) -> Option[Any]:
        """Transform the value (no-op for Nothing)."""
        return cast("Option[Any]", self)

    def and_then(self, func: Callable[[Any], Option[Any]]) -> Option[Any]:
        """Chain operations (no-op for Nothing)."""
        return cast("Option[Any]", self)

    def or_else(self, func: Callable[[], Option[T]]) -> Option[T]:
        """Provide alternative."""
        return func()

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return default

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """Extract value or compute default."""
        return func()

    def __repr__(self) -> str:
        return "Nothing"


# Singleton instance of Nothing
Nothing: _Nothing = _Nothing()

# Option is a union of Some and Nothing
Option = Union[Some[T], _Nothing]


# Either Type (more general than Result)
@dataclass(frozen=True)
class Left(Generic[L]):
    """Left side of Either (conventionally error/failure)."""

    value: L

    def is_left(self) -> bool:
        """Check if this is Left."""
        return True

    def is_right(self) -> bool:
        """Check if this is Right."""
        return False

    def map_left(self, func: Callable[[L], U]) -> Either[U, Any]:
        """Transform the left value."""
        return Left(func(self.value))

    def map_right(self, func: Callable[[Any], Any]) -> Either[L, Any]:
        """Transform the right value (no-op for Left)."""
        return cast("Either[L, Any]", self)


@dataclass(frozen=True)
class Right(Generic[R]):
    """Right side of Either (conventionally success/value)."""

    value: R

    def is_left(self) -> bool:
        """Check if this is Left."""
        return False

    def is_right(self) -> bool:
        """Check if this is Right."""
        return True

    def map_left(self, func: Callable[[Any], Any]) -> Either[Any, R]:
        """Transform the left value (no-op for Right)."""
        return cast("Either[Any, R]", self)

    def map_right(self, func: Callable[[R], U]) -> Either[Any, U]:
        """Transform the right value."""
        return Right(func(self.value))


# Either is a union of Left and Right
Either = Union[Left[L], Right[R]]


# Helper functions for creating Result values
def ok(value: T) -> Ok[T]:
    """Create a successful Result."""
    return Ok(value)


def err(error: E) -> Err[E]:
    """Create an error Result."""
    return Err(error)


# Helper functions for creating Option values
def some(value: T) -> Some[T]:
    """Create a Some Option."""
    return Some(value)


def nothing() -> _Nothing:
    """Return the Nothing singleton."""
    return Nothing


# Helper functions for creating Either values
def left(value: L) -> Left[L]:
    """Create a Left Either."""
    return Left(value)


def right(value: R) -> Right[R]:
    """Create a Right Either."""
    return Right(value)


# Utility function to convert exceptions to Results
def try_result(func: Callable[[], T], exception_types: tuple = (Exception,)) -> Result[T, Exception]:
    """Execute a function and wrap the result in Result type.

    Args:
        func: Function to execute
        exception_types: Tuple of exception types to catch

    Returns:
        Ok with the function result or Err with the caught exception
    """
    try:
        return ok(func())
    except exception_types as e:
        return err(e)


# Utility function to convert None to Option
def from_optional(value: T | None) -> Option[T]:
    """Convert an optional value to Option type."""
    if value is None:
        return nothing()
    return some(value)
