"""Core functional programming utilities and types.

This module provides the foundation for functional programming patterns
used throughout the spreadsheet analyzer, including Result types, Option types,
and functional combinators.
"""

from .functional import (
    compose,
    const,
    curry,
    flatmap_option,
    flatmap_result,
    flip,
    identity,
    map_option,
    map_result,
    partial,
    pipe,
)
from .types import (
    Either,
    Err,
    Left,
    Nothing,
    Ok,
    Option,
    Result,
    Right,
    Some,
)

__all__ = [
    # Either type
    "Either",
    "Err",
    "Left",
    "Nothing",
    "Ok",
    # Option type
    "Option",
    # Result type
    "Result",
    "Right",
    "Some",
    # Functional utilities
    "compose",
    "const",
    "curry",
    "flatmap_option",
    "flatmap_result",
    "flip",
    "identity",
    "map_option",
    "map_result",
    "partial",
    "pipe",
]
