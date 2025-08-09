"""Secure prompt loader with content hash validation.

This module provides the loading mechanism for prompts with automatic
hash validation to ensure prompt integrity and version tracking.

CLAUDE-KNOWLEDGE: The loader computes SHA-256 hash of the prompt file
and compares it with the registry. If they don't match, it fails with
a clear error message showing both hashes.
"""

import hashlib
from pathlib import Path
from typing import Any

import yaml

from ..core.types import Result, err, ok
from .registry import PROMPT_REGISTRY


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hash string in format "sha256:hexdigest"
    """
    content = file_path.read_bytes()
    hash_digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{hash_digest}"


def load_prompt(prompt_name: str) -> Result[dict[str, Any], str]:
    """Load and validate a prompt with hash checking.

    This function enforces prompt versioning by validating that the
    content hash in the registry matches the actual file hash.
    If they don't match, the prompt has been modified and the registry
    must be updated before the prompt can be used.

    Args:
        prompt_name: Name of the prompt to load from registry

    Returns:
        Result containing prompt data (with template and input_variables)
        or error message with hash mismatch details

    Example:
        >>> result = load_prompt("data_analyst_system")
        >>> if result.is_err():
        ...     print(f"Error: {result.err_value}")
        ... else:
        ...     prompt_data = result.ok_value
        ...     template = prompt_data["template"]
    """
    # Check if prompt exists in registry
    if prompt_name not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        return err(f"Unknown prompt: '{prompt_name}'\nAvailable prompts: {available}")

    definition = PROMPT_REGISTRY[prompt_name]
    prompt_path = Path(__file__).parent / definition.file_name

    # Check if file exists
    if not prompt_path.exists():
        return err(f"Prompt file not found: {definition.file_name}\nExpected at: {prompt_path}")

    # Compute current hash
    try:
        current_hash = compute_file_hash(prompt_path)
    except Exception as e:
        return err(f"Failed to compute hash for {definition.file_name}: {e}")

    # Validate hash matches
    if current_hash != definition.content_hash:
        # CLAUDE-IMPORTANT: This error message is designed to be copy-pasteable
        # for easy registry updates when intentional changes are made
        return err(
            f"\n"
            f"❌ Prompt '{prompt_name}' has been modified without updating its hash!\n"
            f"\n"
            f"Expected hash: {definition.content_hash}\n"
            f"Current hash:  {current_hash}\n"
            f"\n"
            f"To fix this, update the hash in registry.py:\n"
            f"  PROMPT_REGISTRY['{prompt_name}'].content_hash = \"{current_hash}\"\n"
            f"\n"
            f"This enforcement ensures all prompt changes are tracked."
        )

    # Load and return prompt data
    try:
        with prompt_path.open() as f:
            prompt_data = yaml.safe_load(f)

        # Validate prompt structure
        if not isinstance(prompt_data, dict):
            return err(f"Invalid prompt format in {definition.file_name}: expected dict")

        if "template" not in prompt_data:
            return err(f"Missing 'template' field in {definition.file_name}")

        if "input_variables" not in prompt_data:
            return err(f"Missing 'input_variables' field in {definition.file_name}")

        return ok(prompt_data)

    except yaml.YAMLError as e:
        return err(f"Failed to parse YAML in {definition.file_name}: {e}")
    except Exception as e:
        return err(f"Failed to load prompt {prompt_name}: {e}")


def validate_all_prompts() -> Result[dict[str, bool], str]:
    """Validate all registered prompts have matching hashes.

    Useful for testing and CI/CD validation.

    Returns:
        Result containing dict of prompt_name -> validation_status
        or error if validation process fails
    """
    results = {}

    for prompt_name in PROMPT_REGISTRY:
        result = load_prompt(prompt_name)
        results[prompt_name] = result.is_ok()

    # Check if any failed
    failed = [name for name, success in results.items() if not success]
    if failed:
        return err(
            f"Prompt validation failed for: {', '.join(failed)}\nRun each individually for detailed error messages."
        )

    return ok(results)
