"""Immutable registry of prompt definitions with content hashes.

This module provides a centralized registry of all prompts with their
SHA-256 content hashes to enforce version tracking.

CLAUDE-KNOWLEDGE: When a prompt file is modified, its hash will change
and the system will refuse to load it until the registry is updated.
This forces conscious versioning of prompt changes.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PromptDefinition:
    """Immutable prompt definition with content hash for version tracking.

    Attributes:
        name: Unique identifier for the prompt
        file_name: Prompt file name (*.prompt.yml) in the prompts directory
        content_hash: SHA-256 hash of the file content (acts as version)
        description: Brief description of prompt purpose
    """

    name: str
    file_name: str
    content_hash: str  # Format: "sha256:hexdigest" - this IS the version
    description: str


# CLAUDE-IMPORTANT: When modifying any prompt file (*.prompt.yml), you MUST update
# the corresponding content_hash here. The system will fail at runtime
# if the hash doesn't match the actual file content.
PROMPT_REGISTRY: Final[dict[str, PromptDefinition]] = {
    "data_analyst_system": PromptDefinition(
        name="data_analyst_system",
        file_name="data_analyst_system.prompt.yml",
        content_hash="sha256:531bcb715f64c7367f8361a1b129c080771684a06b0bdaef32871cd7e0e26280",
        description="System prompt for general data analysis agent",
    ),
    "data_analyst_initial": PromptDefinition(
        name="data_analyst_initial",
        file_name="data_analyst_initial.prompt.yml",
        content_hash="sha256:61250e4d29ba0491d7cb8e591222212b26c308381a36edb5c0cc12e8d66e4fdf",
        description="Initial human prompt for data analysis",
    ),
    "table_aware_analyst_system": PromptDefinition(
        name="table_aware_analyst_system",
        file_name="table_aware_analyst_system.prompt.yml",
        content_hash="sha256:db5236da91f7f998d514b4d5c402463cb713eb4fb037b0f6b36ace9b0670b22e",
        description="System prompt for analysis with pre-detected table boundaries",
    ),
    "table_detector_system": PromptDefinition(
        name="table_detector_system",
        file_name="table_detector_system.prompt.yml",
        content_hash="sha256:988988e6c8b4421bc6f7ae15bf1c17ca9296acb2e6ff4e3a8dd1a5b6376f2675",
        description="System prompt for multi-table detection agent",
    ),
    "layout_comprehension_system": PromptDefinition(
        name="layout_comprehension_system",
        file_name="layout_comprehension_system.prompt.yml",
        content_hash="sha256:68f3ba81c72c55c41eceec8cdca137fad04cac77009c39512e1cc5870264e0de",
        description="System prompt for semantic layout comprehension in spreadsheets",
    ),
}


def get_prompt_definition(name: str) -> PromptDefinition | None:
    """Get prompt definition by name.

    Args:
        name: Prompt name to look up

    Returns:
        PromptDefinition if found, None otherwise
    """
    return PROMPT_REGISTRY.get(name)


def list_prompts() -> list[str]:
    """List all registered prompt names.

    Returns:
        List of prompt names in the registry
    """
    return list(PROMPT_REGISTRY.keys())
