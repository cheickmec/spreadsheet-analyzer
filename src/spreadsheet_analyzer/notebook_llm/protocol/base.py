"""Base protocol interface for Notebook-Augmented Processing (NAP).

This module defines the foundational protocol for the NAP system, which ensures
validation-first philosophy and maintains audit trails throughout analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from spreadsheet_analyzer.pipeline.types import SheetStructure


class FileType(Enum):
    """Supported file types for analysis."""

    EXCEL = "excel"
    CSV = "csv"
    UNKNOWN = "unknown"


class AnalysisState(Enum):
    """Current state of the analysis process."""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class NotebookCellType(Enum):
    """Types of cells in the analysis notebook."""

    MARKDOWN = "markdown"
    CODE = "code"
    OUTPUT = "output"
    VALIDATION = "validation"


class AnalysisPhase(Enum):
    """Phases of the notebook-based analysis process."""

    INITIALIZATION = "initialization"
    STRUCTURE_DISCOVERY = "structure_discovery"
    FORMULA_ANALYSIS = "formula_analysis"
    DATA_VALIDATION = "data_validation"
    INSIGHT_GENERATION = "insight_generation"
    REPORT_COMPILATION = "report_compilation"


@dataclass
class NotebookCell:
    """Represents a single cell in the analysis notebook."""

    cell_type: NotebookCellType
    content: str | dict[str, Any]
    metadata: dict[str, Any]
    execution_order: int | None = None
    outputs: list[dict[str, Any]] | None = None
    validation_status: bool | None = None


@dataclass
class AnalysisContext:
    """Context for the current analysis operation."""

    file_path: Path
    file_type: FileType
    current_phase: AnalysisPhase
    state: AnalysisState
    metadata: dict[str, Any]
    validation_requirements: list[str]


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    message: str
    evidence: dict[str, Any]
    suggested_actions: list[str] | None = None


class NotebookProtocol(Protocol):
    """Protocol for notebook-based analysis operations."""

    def create_cell(
        self,
        cell_type: NotebookCellType,
        content: str | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCell:
        """Create a new notebook cell."""
        ...

    def execute_cell(self, cell: NotebookCell, context: AnalysisContext) -> dict[str, Any]:
        """Execute a notebook cell and return results."""
        ...

    def validate_output(self, cell: NotebookCell, output: dict[str, Any]) -> ValidationResult:
        """Validate the output of a cell execution."""
        ...

    def persist_state(self, context: AnalysisContext, cells: list[NotebookCell]) -> None:
        """Persist the current notebook state."""
        ...


class BaseNAPProcessor(ABC):
    """Base class for NAP processors implementing validation-first philosophy."""

    def __init__(self, notebook_protocol: NotebookProtocol) -> None:
        """Initialize the NAP processor.

        Args:
            notebook_protocol: Protocol implementation for notebook operations
        """
        self.notebook = notebook_protocol
        self._current_context: AnalysisContext | None = None
        self._cells: list[NotebookCell] = []

    @abstractmethod
    def initialize_analysis(self, file_path: Path) -> AnalysisContext:
        """Initialize the analysis context for a file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            AnalysisContext: Initialized context for analysis
        """
        pass

    @abstractmethod
    def discover_structure(self, context: AnalysisContext) -> dict[str, SheetStructure]:
        """Discover and validate the structure of the spreadsheet.

        Args:
            context: Current analysis context

        Returns:
            Dict mapping sheet names to their structures
        """
        pass

    @abstractmethod
    def analyze_formulas(self, context: AnalysisContext, sheet_structures: dict[str, SheetStructure]) -> dict[str, Any]:
        """Analyze formulas within the spreadsheet.

        Args:
            context: Current analysis context
            sheet_structures: Discovered sheet structures

        Returns:
            Formula analysis results
        """
        pass

    @abstractmethod
    def validate_data(self, context: AnalysisContext, analysis_results: dict[str, Any]) -> list[ValidationResult]:
        """Validate data integrity and consistency.

        Args:
            context: Current analysis context
            analysis_results: Results from previous analysis phases

        Returns:
            List of validation results
        """
        pass

    @abstractmethod
    def generate_insights(
        self,
        context: AnalysisContext,
        analysis_results: dict[str, Any],
        validation_results: list[ValidationResult],
    ) -> dict[str, Any]:
        """Generate insights from analysis and validation results.

        Args:
            context: Current analysis context
            analysis_results: Complete analysis results
            validation_results: Data validation results

        Returns:
            Generated insights
        """
        pass

    def create_validation_cell(self, validation_code: str, description: str) -> NotebookCell:
        """Create a validation cell with proper metadata.

        Args:
            validation_code: Code to perform validation
            description: Human-readable description of validation

        Returns:
            NotebookCell configured for validation
        """
        return self.notebook.create_cell(
            cell_type=NotebookCellType.VALIDATION,
            content=validation_code,
            metadata={
                "description": description,
                "validation": True,
                "timestamp": self._get_timestamp(),
            },
        )

    def execute_with_validation(self, cell: NotebookCell, context: AnalysisContext) -> dict[str, Any]:
        """Execute a cell and validate its output.

        Args:
            cell: Cell to execute
            context: Current analysis context

        Returns:
            Validated execution results

        Raises:
            ValidationError: If validation fails
        """
        output = self.notebook.execute_cell(cell, context)
        validation = self.notebook.validate_output(cell, output)

        if not validation.is_valid:
            # Create validation failure cell
            failure_cell = self.notebook.create_cell(
                cell_type=NotebookCellType.MARKDOWN,
                content=f"## Validation Failed\n\n{validation.message}",
                metadata={"validation_failure": True, "evidence": validation.evidence},
            )
            self._cells.append(failure_cell)
            raise ValidationError(validation.message, validation.evidence)

        return output

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()


class ValidationError(Exception):
    """Raised when validation fails during analysis."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            evidence: Evidence of validation failure
        """
        super().__init__(message)
        self.evidence = evidence


class AnalysisStrategy(ABC):
    """Base class for pluggable analysis strategies."""

    @abstractmethod
    def apply(self, processor: BaseNAPProcessor, context: AnalysisContext) -> dict[str, Any]:
        """Apply the analysis strategy.

        Args:
            processor: NAP processor instance
            context: Analysis context

        Returns:
            Strategy-specific results
        """
        pass

    @abstractmethod
    def get_required_capabilities(self) -> list[str]:
        """Get list of required capabilities for this strategy.

        Returns:
            List of capability identifiers
        """
        pass
