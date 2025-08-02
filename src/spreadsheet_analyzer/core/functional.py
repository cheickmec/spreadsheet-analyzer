"""Functional programming utilities and combinators.

This module provides core functional programming utilities including
function composition, currying, and specialized map/flatmap operations
for Result and Option types.

CLAUDE-KNOWLEDGE: These utilities follow standard FP conventions from
languages like Haskell and Scala, adapted for Python's syntax.
"""

from collections.abc import Callable
from functools import reduce, wraps
from typing import Any, TypeVar, cast

from .types import Err, Ok, Option, Result, Some, nothing, ok, some

# Type variables
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")


# Function composition
def compose(*functions: Callable) -> Callable:
    """Compose functions right-to-left (mathematical order).

    compose(f, g, h)(x) = f(g(h(x)))

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> composed = compose(add_one, double)
        >>> composed(5)  # (5 * 2) + 1 = 11
        11
    """

    def composed(x: Any) -> Any:
        return reduce(lambda val, func: func(val), reversed(functions), x)

    return composed


def pipe(*functions: Callable) -> Callable:
    """Compose functions left-to-right (pipeline order).

    pipe(f, g, h)(x) = h(g(f(x)))

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> piped = pipe(add_one, double)
        >>> piped(5)  # (5 + 1) * 2 = 12
        12
    """

    def piped(x: Any) -> Any:
        return reduce(lambda val, func: func(val), functions, x)

    return piped


# Currying
def curry(func: Callable) -> Callable:
    """Convert a function to its curried form.

    A curried function takes arguments one at a time, returning
    a new function for each argument until all are provided.

    Example:
        >>> @curry
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> add_five = add(5)
        >>> add_five(3)
        8
    """

    @wraps(func)
    def curried(*args: Any, **kwargs: Any) -> Any:
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})

    return curried


# Partial application (more Pythonic than curry for some cases)
def partial(func: Callable, *args: Any, **kwargs: Any) -> Callable:
    """Partially apply a function with some arguments.

    Example:
        >>> def greet(greeting: str, name: str) -> str:
        ...     return f"{greeting}, {name}!"
        >>> say_hello = partial(greet, "Hello")
        >>> say_hello("Alice")
        'Hello, Alice!'
    """

    @wraps(func)
    def partially_applied(*more_args: Any, **more_kwargs: Any) -> Any:
        return func(*(args + more_args), **{**kwargs, **more_kwargs})

    return partially_applied


# Basic combinators
def identity(x: A) -> A:
    """Identity function - returns its argument unchanged."""
    return x


def const(x: A) -> Callable[[Any], A]:
    """Constant function - ignores its argument and returns x."""
    return lambda _: x


def flip(func: Callable[[A, B], C]) -> Callable[[B, A], C]:
    """Flip the arguments of a binary function.

    Example:
        >>> subtract = lambda x, y: x - y
        >>> flipped_subtract = flip(subtract)
        >>> subtract(10, 3)  # 10 - 3 = 7
        7
        >>> flipped_subtract(10, 3)  # 3 - 10 = -7
        -7
    """
    return lambda b, a: func(a, b)


# Result type utilities
def map_result(func: Callable[[A], B]) -> Callable[[Result[A, E]], Result[B, E]]:
    """Create a function that maps over a Result's success value.

    Example:
        >>> double = lambda x: x * 2
        >>> map_double = map_result(double)
        >>> map_double(ok(5))
        Ok(value=10)
        >>> map_double(err("error"))
        Err(error='error')
    """

    def mapper(result: Result[A, E]) -> Result[B, E]:
        if isinstance(result, Ok):
            return ok(func(result.value))
        return cast("Result[B, E]", result)

    return mapper


def flatmap_result(func: Callable[[A], Result[B, E]]) -> Callable[[Result[A, E]], Result[B, E]]:
    """Create a function that flatmaps over a Result.

    Also known as 'bind' or 'and_then' in other languages.

    Example:
        >>> def safe_divide(x: int, y: int) -> Result[float, str]:
        ...     if y == 0:
        ...         return err("Division by zero")
        ...     return ok(x / y)
        >>> divide_by_two = lambda x: safe_divide(x, 2)
        >>> flatmap_divide = flatmap_result(divide_by_two)
        >>> flatmap_divide(ok(10))
        Ok(value=5.0)
    """

    def flatmapper(result: Result[A, E]) -> Result[B, E]:
        if isinstance(result, Ok):
            return func(result.value)
        return cast("Result[B, E]", result)

    return flatmapper


# Option type utilities
def map_option(func: Callable[[A], B]) -> Callable[[Option[A]], Option[B]]:
    """Create a function that maps over an Option's value.

    Example:
        >>> double = lambda x: x * 2
        >>> map_double = map_option(double)
        >>> map_double(some(5))
        Some(value=10)
        >>> map_double(nothing())
        Nothing
    """

    def mapper(option: Option[A]) -> Option[B]:
        if isinstance(option, Some):
            return some(func(option.value))
        return cast("Option[B]", nothing())

    return mapper


def flatmap_option(func: Callable[[A], Option[B]]) -> Callable[[Option[A]], Option[B]]:
    """Create a function that flatmaps over an Option.

    Example:
        >>> def safe_reciprocal(x: float) -> Option[float]:
        ...     if x == 0:
        ...         return nothing()
        ...     return some(1 / x)
        >>> flatmap_reciprocal = flatmap_option(safe_reciprocal)
        >>> flatmap_reciprocal(some(2.0))
        Some(value=0.5)
        >>> flatmap_reciprocal(some(0.0))
        Nothing
    """

    def flatmapper(option: Option[A]) -> Option[B]:
        if isinstance(option, Some):
            return func(option.value)
        return cast("Option[B]", nothing())

    return flatmapper


# Utility functions for working with collections of Results/Options
def sequence_results(results: list[Result[A, E]]) -> Result[list[A], E]:
    """Convert a list of Results to a Result of list.

    If all Results are Ok, returns Ok with list of values.
    If any Result is Err, returns the first Err.

    Example:
        >>> sequence_results([ok(1), ok(2), ok(3)])
        Ok(value=[1, 2, 3])
        >>> sequence_results([ok(1), err("error"), ok(3)])
        Err(error='error')
    """
    values = []
    for result in results:
        if isinstance(result, Err):
            return cast("Result[list[A], E]", result)
        values.append(result.value)
    return ok(values)


def sequence_options(options: list[Option[A]]) -> Option[list[A]]:
    """Convert a list of Options to an Option of list.

    If all Options are Some, returns Some with list of values.
    If any Option is Nothing, returns Nothing.

    Example:
        >>> sequence_options([some(1), some(2), some(3)])
        Some(value=[1, 2, 3])
        >>> sequence_options([some(1), nothing(), some(3)])
        Nothing
    """
    values = []
    for option in options:
        if option.is_nothing():
            return nothing()
        values.append(cast("Some[A]", option).value)
    return some(values)


# Kleisli composition for Result and Option
def kleisli_result(f: Callable[[A], Result[B, E]], g: Callable[[B], Result[C, E]]) -> Callable[[A], Result[C, E]]:
    """Compose two functions that return Results.

    Also known as Kleisli composition for the Result monad.

    Example:
        >>> def parse_int(s: str) -> Result[int, str]:
        ...     try:
        ...         return ok(int(s))
        ...     except ValueError:
        ...         return err(f"Cannot parse '{s}' as int")
        >>> def safe_divide_10(x: int) -> Result[float, str]:
        ...     if x == 0:
        ...         return err("Division by zero")
        ...     return ok(10 / x)
        >>> composed = kleisli_result(parse_int, safe_divide_10)
        >>> composed("5")
        Ok(value=2.0)
        >>> composed("0")
        Err(error='Division by zero')
    """
    return lambda a: f(a).and_then(g)


def kleisli_option(f: Callable[[A], Option[B]], g: Callable[[B], Option[C]]) -> Callable[[A], Option[C]]:
    """Compose two functions that return Options.

    Example:
        >>> def get_first(lst: list) -> Option[Any]:
        ...     return some(lst[0]) if lst else nothing()
        >>> def parse_positive_int(x: Any) -> Option[int]:
        ...     if isinstance(x, int) and x > 0:
        ...         return some(x)
        ...     return nothing()
        >>> composed = kleisli_option(get_first, parse_positive_int)
        >>> composed([5, 2, 3])
        Some(value=5)
        >>> composed([-1, 2, 3])
        Nothing
    """
    return lambda a: f(a).and_then(g)
