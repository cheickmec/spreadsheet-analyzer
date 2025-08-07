"""Prompt management with hash-based version enforcement.

This package provides a secure prompt loading system that enforces
version tracking through content hashing. Any change to a prompt
requires updating its hash in the registry.

Example:
    >>> from spreadsheet_analyzer.prompts import load_prompt
    >>>
    >>> result = load_prompt("data_analyst_system")
    >>> if result.is_err():
    ...     raise RuntimeError(result.err_value)
    >>>
    >>> prompt_data = result.ok_value
    >>> template = prompt_data["template"]
    >>> variables = prompt_data["input_variables"]
"""

from .loader import compute_file_hash, load_prompt, validate_all_prompts
from .registry import (
    PROMPT_REGISTRY,
    PromptDefinition,
    get_prompt_definition,
    list_prompts,
)

__all__ = [
    # Main API
    "load_prompt",
    "validate_all_prompts",
    # Registry access
    "PROMPT_REGISTRY",
    "PromptDefinition",
    "get_prompt_definition",
    "list_prompts",
    # Utilities
    "compute_file_hash",
]
