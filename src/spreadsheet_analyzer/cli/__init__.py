"""Command-line interfaces for the spreadsheet analyzer.

This package provides CLI interfaces for both the deterministic
pipeline and the LLM-powered notebook analysis.
"""

# Temporary compatibility imports
# These will be removed after full migration
import warnings
from typing import Any


def _deprecated_import(old_module: str, new_location: str) -> None:
    """Warn about deprecated imports."""
    warnings.warn(
        f"Importing from '{old_module}' is deprecated. Please import from '{new_location}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Compatibility for notebook_cli imports
def __getattr__(name: str) -> Any:
    """Provide compatibility for old imports."""
    if name == "NotebookCLI":
        _deprecated_import("spreadsheet_analyzer.notebook_cli", "spreadsheet_analyzer.cli.notebook")
        # Import will be available after Phase 2
        raise ImportError(
            "NotebookCLI has not been migrated yet. Please use 'spreadsheet_analyzer.notebook_cli' for now."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = []
