"""
Base plugin interfaces and registry system.

This module defines the core plugin protocols and registry system
for extending notebook functionality with domain-specific features.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..core_exec import NotebookBuilder, NotebookCell
from ..core_exec import QualityMetrics as CoreQualityMetrics


class Task(Protocol):
    """
    Protocol for notebook tasks that generate cells.

    Tasks are the primary way to extend notebook functionality with
    domain-specific analysis. They generate initial cells based on
    context and can post-process notebooks after execution.
    """

    name: str

    def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
        """
        Generate initial cells for the notebook.

        Args:
            context: Dictionary containing file_path, sheet_name, etc.

        Returns:
            List of NotebookCell objects to add to the notebook
        """
        ...

    def postprocess(self, notebook: NotebookBuilder, context: dict[str, Any]) -> list[NotebookCell]:
        """
        Post-process notebook after execution (optional).

        Args:
            notebook: The notebook after execution
            context: Execution context including stats

        Returns:
            Additional cells to append (empty list if none)
        """
        ...

    def validate_context(self, context: dict[str, Any]) -> list[str]:
        """
        Validate that the context contains required information.

        Args:
            context: The context dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        ...


class QualityInspector(Protocol):
    """
    Protocol for notebook quality assessment.

    Quality inspectors analyze notebooks and provide metrics
    and suggestions for improvement.
    """

    name: str

    def inspect(self, notebook: NotebookBuilder, context: dict[str, Any]) -> CoreQualityMetrics:
        """
        Inspect notebook quality.

        Args:
            notebook: The notebook to inspect
            context: Context including file info, execution stats

        Returns:
            QualityMetrics with scores, issues, and suggestions
        """
        ...


class BaseTask(ABC):
    """
    Base class for implementing tasks.

    Provides common functionality and default implementations
    for optional methods.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.category = "generic"

    @abstractmethod
    def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
        """Generate initial cells - must be implemented by subclasses."""
        pass

    def postprocess(self, notebook: NotebookBuilder, context: dict[str, Any]) -> list[NotebookCell]:
        """Default: no post-processing."""
        return []

    def validate_context(self, context: dict[str, Any]) -> list[str]:
        """Default: no validation."""
        return []


class BaseQualityInspector(ABC):
    """
    Base class for implementing quality inspectors.

    Provides common functionality for quality assessment.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def inspect(self, notebook: NotebookBuilder, context: dict[str, Any]) -> CoreQualityMetrics:
        """Inspect notebook quality - must be implemented by subclasses."""
        pass


class PluginRegistry:
    """
    Registry for managing tasks and quality inspectors.

    Provides centralized registration and discovery of plugins.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._quality_inspectors: dict[str, QualityInspector] = {}

    def register_task(self, task: Task) -> None:
        """Register a task."""
        self._tasks[task.name] = task

    def register_quality_inspector(self, inspector: QualityInspector) -> None:
        """Register a quality inspector."""
        self._quality_inspectors[inspector.name] = inspector

    def get_task(self, name: str) -> Task | None:
        """Get a task by name."""
        return self._tasks.get(name)

    def get_quality_inspector(self, name: str) -> QualityInspector | None:
        """Get a quality inspector by name."""
        return self._quality_inspectors.get(name)

    def list_tasks(self) -> list[Task]:
        """List all registered tasks."""
        return list(self._tasks.values())

    def list_quality_inspectors(self) -> list[QualityInspector]:
        """List all registered quality inspectors."""
        return list(self._quality_inspectors.values())

    def clear(self) -> None:
        """Clear all registrations."""
        self._tasks.clear()
        self._quality_inspectors.clear()


# Global registry instance
registry = PluginRegistry()
