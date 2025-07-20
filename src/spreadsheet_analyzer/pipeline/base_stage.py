"""
Base abstractions for pipeline stages.

This module provides the abstract base class and common functionality
for all pipeline stages, following functional programming principles
with immutable data structures.

CLAUDE-KNOWLEDGE: This abstraction layer ensures consistency across
all pipeline stages while maintaining flexibility for stage-specific
requirements.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Generic, Protocol, TypeVar

from spreadsheet_analyzer.pipeline.types import (
    Err,
    Ok,
    PipelineContext,
    Result,
)

logger = logging.getLogger(__name__)

# Type variables for generic stage results
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TConfig = TypeVar("TConfig")

# Stage identifiers
STAGE_0_INTEGRITY: Final[str] = "integrity"
STAGE_1_SECURITY: Final[str] = "security"
STAGE_2_STRUCTURE: Final[str] = "structure"
STAGE_3_FORMULAS: Final[str] = "formulas"
STAGE_4_CONTENT: Final[str] = "content"


# Progress callback protocol
class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, stage: str, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
        """Report progress update."""
        ...


@dataclass(frozen=True)
class StageConfig:
    """Base configuration for all stages."""

    enable_validation: bool = True
    enable_progress: bool = True
    timeout_seconds: float | None = None
    max_retries: int = 0
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageMetrics:
    """Performance metrics for stage execution."""

    stage_name: str
    start_time: float
    end_time: float
    items_processed: int
    items_skipped: int
    warnings: list[str] = field(default_factory=list)

    @property
    def execution_time(self) -> float:
        """Total execution time in seconds."""
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Items processed per second."""
        if self.execution_time == 0:
            return 0.0
        return self.items_processed / self.execution_time


@dataclass(frozen=True)
class StageResult(Generic[TOutput]):
    """Wrapper for stage execution results with metrics."""

    output: TOutput
    metrics: StageMetrics
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)


class BaseStage(ABC, Generic[TInput, TOutput, TConfig]):
    """
    Abstract base class for all pipeline stages.

    This class provides common functionality for:
    - Input validation
    - Progress tracking
    - Error handling
    - Performance metrics
    - Result validation

    CLAUDE-COMPLEX: The generic type parameters allow each stage to
    define its own input/output types while maintaining type safety.
    """

    def __init__(self, stage_name: str, config: TConfig | None = None):
        """
        Initialize stage with configuration.

        Args:
            stage_name: Unique identifier for the stage
            config: Stage-specific configuration
        """
        self.stage_name = stage_name
        self.config = config or self._get_default_config()
        self.metrics = self._initialize_metrics()
        self._progress_callback: ProgressCallback | None = None

    @abstractmethod
    def _get_default_config(self) -> TConfig:
        """Get default configuration for the stage."""
        pass

    @abstractmethod
    def _validate_input(self, input_data: TInput) -> Result:
        """
        Validate input data before processing.

        Args:
            input_data: Input to validate

        Returns:
            Ok(None) if valid, Err(reason) if invalid
        """
        pass

    @abstractmethod
    def _process(self, input_data: TInput, context: PipelineContext | None = None) -> Result:
        """
        Core processing logic for the stage.

        Args:
            input_data: Validated input data
            context: Optional pipeline context

        Returns:
            Ok(output) on success, Err(error) on failure
        """
        pass

    @abstractmethod
    def _validate_output(self, output: TOutput) -> Result:
        """
        Validate output data after processing.

        Args:
            output: Output to validate

        Returns:
            Ok(None) if valid, Err(reason) if invalid
        """
        pass

    def execute(
        self,
        input_data: TInput,
        *,
        context: PipelineContext | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Result:
        """
        Execute the stage with full validation and metrics.

        This is the main entry point for stage execution. It handles:
        1. Input validation
        2. Processing with error handling
        3. Output validation
        4. Metrics collection
        5. Progress reporting

        Args:
            input_data: Input data for the stage
            context: Optional pipeline context
            progress_callback: Optional progress callback

        Returns:
            Ok(StageResult) on success, Err(error) on failure

        CLAUDE-KNOWLEDGE: This method implements the Template Method pattern,
        defining the skeleton of the algorithm while allowing subclasses to
        override specific steps.
        """
        # Set progress callback
        self._progress_callback = progress_callback

        # Start metrics
        start_time = time.time()
        self._report_progress(0.0, f"Starting {self.stage_name} stage")

        try:
            # Validate input
            if hasattr(self.config, "enable_validation") and self.config.enable_validation:
                validation_result = self._validate_input(input_data)
                if isinstance(validation_result, Err):
                    return Err(f"Input validation failed: {validation_result.error}")

            self._report_progress(0.1, "Input validated")

            # Process data
            process_result = self._process(input_data, context)
            if isinstance(process_result, Err):
                return Err(f"Processing failed: {process_result.error}")

            output = process_result.value
            self._report_progress(0.9, "Processing complete")

            # Validate output
            validation_errors = []
            if hasattr(self.config, "enable_validation") and self.config.enable_validation:
                output_validation = self._validate_output(output)
                if isinstance(output_validation, Err):
                    validation_errors.append(output_validation.error)

            # Complete metrics
            end_time = time.time()
            metrics = StageMetrics(
                stage_name=self.stage_name,
                start_time=start_time,
                end_time=end_time,
                items_processed=self.metrics.get("items_processed", 0),
                items_skipped=self.metrics.get("items_skipped", 0),
                warnings=self.metrics.get("warnings", []),
            )

            # Create result
            result = StageResult(
                output=output,
                metrics=metrics,
                validation_passed=len(validation_errors) == 0,
                validation_errors=validation_errors,
            )

            self._report_progress(1.0, f"{self.stage_name} stage complete")

            return Ok(result)

        except Exception as e:
            error_msg = f"Stage {self.stage_name} failed with unexpected error: {e!s}"
            logger.exception(error_msg)
            return Err(error_msg)

    def _initialize_metrics(self) -> dict[str, Any]:
        """Initialize metrics tracking dictionary."""
        return {"items_processed": 0, "items_skipped": 0, "warnings": []}

    def _report_progress(self, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
        """Report progress if callback is available."""
        if self._progress_callback and hasattr(self.config, "enable_progress") and self.config.enable_progress:
            self._progress_callback(self.stage_name, progress, message, details)


class FileProcessingStage(BaseStage[Path, TOutput, TConfig]):
    """
    Base class for stages that process files.

    Adds file-specific validation and utilities to the base stage.
    """

    def _validate_input(self, input_data: Path) -> Result:
        """Validate that the input file exists and is readable."""
        if not isinstance(input_data, Path):
            return Err(f"Expected Path, got {type(input_data).__name__}")

        if not input_data.exists():
            return Err(f"File not found: {input_data}")

        if not input_data.is_file():
            return Err(f"Not a file: {input_data}")

        # Check if file is readable
        try:
            with input_data.open("rb") as f:
                f.read(1)  # Try to read one byte
        except PermissionError:
            return Err(f"Permission denied: {input_data}")
        except Exception as e:
            return Err(f"Cannot read file {input_data}: {e!s}")

        return Ok(None)


class ComposableStage(BaseStage[TInput, TOutput, TConfig]):
    """
    Base class for stages that can be composed with other stages.

    Supports functional composition patterns for building pipelines.

    CLAUDE-KNOWLEDGE: This enables building complex pipelines by
    composing simple stages, following functional programming principles.
    """

    def __init__(
        self,
        stage_name: str,
        config: TConfig | None = None,
        pre_processors: list[Callable[[TInput], TInput]] | None = None,
        post_processors: list[Callable[[TOutput], TOutput]] | None = None,
    ):
        """
        Initialize composable stage.

        Args:
            stage_name: Unique identifier
            config: Stage configuration
            pre_processors: Functions to apply before processing
            post_processors: Functions to apply after processing
        """
        super().__init__(stage_name, config)
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []

    def with_pre_processor(self, processor: Callable[[TInput], TInput]) -> "ComposableStage[TInput, TOutput, TConfig]":
        """Add a pre-processor to the stage."""
        return ComposableStage(self.stage_name, self.config, [*self.pre_processors, processor], self.post_processors)

    def with_post_processor(
        self, processor: Callable[[TOutput], TOutput]
    ) -> "ComposableStage[TInput, TOutput, TConfig]":
        """Add a post-processor to the stage."""
        return ComposableStage(self.stage_name, self.config, self.pre_processors, [*self.post_processors, processor])

    def _apply_pre_processors(self, input_data: TInput) -> TInput:
        """Apply all pre-processors in order."""
        result = input_data
        for processor in self.pre_processors:
            result = processor(result)
        return result

    def _apply_post_processors(self, output: TOutput) -> TOutput:
        """Apply all post-processors in order."""
        result = output
        for processor in self.post_processors:
            result = processor(result)
        return result


def create_stage_chain(stages: list[BaseStage]) -> Callable[[Any, PipelineContext | None], Result]:
    """
    Create a function that chains multiple stages together.

    Args:
        stages: List of stages to chain

    Returns:
        Function that executes all stages in sequence

    Example:
        chain = create_stage_chain([
            IntegrityStage(),
            SecurityStage(),
            StructureStage()
        ])
        result = chain(file_path, context)
    """

    def execute_chain(initial_input: Any, context: PipelineContext | None = None) -> Result:
        """Execute all stages in sequence."""
        current_input = initial_input
        current_context = context

        for stage in stages:
            result = stage.execute(current_input, context=current_context)

            if isinstance(result, Err):
                return result

            # Extract output for next stage
            stage_result = result.value
            current_input = stage_result.output

            # Update context if provided
            if current_context:
                current_context = current_context.with_stage_result(stage.stage_name, stage_result)

        return Ok(current_input)

    return execute_chain


# ============================================================================
# STAGE VALIDATION MIXINS
# ============================================================================


class ExcelValidationMixin:
    """Mixin for Excel-specific validation logic."""

    @staticmethod
    def validate_excel_file(file_path: Path) -> Result:
        """Validate that file is a valid Excel file."""
        valid_extensions = {".xlsx", ".xlsm", ".xls", ".xlsb"}

        if file_path.suffix.lower() not in valid_extensions:
            return Err(f"Not an Excel file: {file_path}")

        # Could add more validation here (file signature, etc.)
        return Ok(None)

    @staticmethod
    def validate_workbook_size(file_path: Path, max_size_mb: float = 100.0) -> Result:
        """Validate workbook size is within limits."""
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            return Err(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")

        return Ok(None)


class ThresholdValidationMixin:
    """Mixin for threshold-based validation."""

    @staticmethod
    def validate_threshold(
        value: float, min_value: float | None = None, max_value: float | None = None, name: str = "value"
    ) -> Result:
        """Validate that a value is within thresholds."""
        if min_value is not None and value < min_value:
            return Err(f"{name} too low: {value} (min: {min_value})")

        if max_value is not None and value > max_value:
            return Err(f"{name} too high: {value} (max: {max_value})")

        return Ok(None)
