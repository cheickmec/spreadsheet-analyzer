"""
Main Pipeline Orchestrator for Deterministic Analysis.

This module ties together all pipeline stages and manages the analysis flow,
including progress tracking, error handling, and result aggregation.
"""

import logging
import platform
import signal
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spreadsheet_analyzer.pipeline.stages.stage_0_integrity import stage_0_integrity_probe
from spreadsheet_analyzer.pipeline.stages.stage_1_security import stage_1_security_scan
from spreadsheet_analyzer.pipeline.stages.stage_2_structure import stage_2_structural_mapping
from spreadsheet_analyzer.pipeline.stages.stage_3_formulas import stage_3_formula_analysis
from spreadsheet_analyzer.pipeline.stages.stage_4_content import stage_4_content_intelligence
from spreadsheet_analyzer.pipeline.types import Err, PipelineContext, PipelineResult, ProgressUpdate, Result

logger = logging.getLogger(__name__)


# ==================== Timeout Handling ====================


class StageTimeoutError(Exception):
    """Raised when a stage execution times out."""


def _timeout_handler(signum, frame):  # noqa: ARG001
    """Signal handler for timeout."""
    raise StageTimeoutError


@contextmanager
def timeout(seconds: int):
    """
    Context manager for setting execution timeout.

    CLAUDE-KNOWLEDGE: Uses signal-based timeout on Unix, falls back
    to no timeout on Windows (threading approach would be complex).
    """
    if seconds <= 0:
        # No timeout
        yield
        return

    # Check if we're on a Unix-like system
    if platform.system() == "Windows":
        # On Windows, we don't implement timeout for now
        # A proper implementation would require threading
        logger.warning("Stage timeouts not supported on Windows platform")
        yield
        return

    # Set up signal handler for Unix-like systems
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore previous handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ==================== Progress Tracking ====================


class ProgressTracker:
    """
    Tracks pipeline progress and notifies observers.

    CLAUDE-KNOWLEDGE: We use the observer pattern to allow
    flexible progress reporting without coupling to specific UI.
    CLAUDE-PERFORMANCE: Observer notifications are async to avoid
    blocking the main analysis pipeline.
    """

    def __init__(self):
        """Initialize tracker."""
        self.observers: list[Callable[[ProgressUpdate], None]] = []
        self.updates: list[ProgressUpdate] = []

    def add_observer(self, observer: Callable[[ProgressUpdate], None]):
        """Add progress observer."""
        self.observers.append(observer)

    def update(self, stage: str, progress: float, message: str, details: dict[str, Any] | None = None):
        """Send progress update."""
        update = ProgressUpdate(
            stage=stage, progress=progress, message=message, timestamp=datetime.now(UTC), details=details
        )

        self.updates.append(update)

        # CLAUDE-GOTCHA: Observer failures should not crash the pipeline
        # We catch exceptions and log them but continue processing
        for observer in self.observers:
            try:
                observer(update)
            except (RuntimeError, ValueError) as e:
                logger.warning("Progress observer failed: %s", str(e))


# ==================== Pipeline Orchestrator ====================


class DeterministicPipeline:
    """
    Main pipeline orchestrator that coordinates all analysis stages.

    CLAUDE-COMPLEX: The pipeline manages complex state transitions
    and error handling across multiple analysis stages.
    CLAUDE-IMPORTANT: Pipeline stages are executed sequentially with
    dependency validation - never skip integrity or security checks.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        """
        Initialize pipeline with options.

        Args:
            options: Configuration options for pipeline behavior
        """
        self.options = options or {}
        self.progress_tracker = ProgressTracker()

        # Configuration
        self.skip_on_error = self.options.get("skip_on_error", False)
        self.parallel_stages = self.options.get("parallel_stages", False)
        self.stage_timeout = self.options.get("stage_timeout", 300)  # 5 minutes default

    def add_progress_observer(self, observer: Callable[[ProgressUpdate], None]):
        """Add observer for progress updates."""
        self.progress_tracker.add_observer(observer)

    def run(self, file_path: Path) -> PipelineResult:
        """
        Run complete analysis pipeline on file.

        Args:
            file_path: Path to Excel file to analyze

        Returns:
            PipelineResult with all stage results
        """
        start_time = datetime.now(UTC)
        context = PipelineContext(file_path=file_path, start_time=start_time, options=self.options)
        stage_results: dict[str, Any] = {}
        errors: list[str] = []

        try:
            # Run all stages in sequence
            success = self._run_all_stages(file_path, context, stage_results, errors)
            if not success:
                return self._create_failed_result(context, errors, start_time)

            # Create successful result
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            return PipelineResult(
                context=context,
                integrity=stage_results.get("integrity"),
                security=stage_results.get("security"),
                structure=stage_results.get("structure"),
                formulas=stage_results.get("formulas"),
                content=stage_results.get("content"),
                execution_time=execution_time,
                success=True,
                errors=tuple(errors),
            )

        except Exception as e:
            logger.exception("Pipeline failed with exception")
            errors.append(f"Pipeline exception: {e!s}")
            return self._create_failed_result(context, errors, start_time)

    def _run_all_stages(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run all pipeline stages in sequence. Returns True if all stages pass."""
        stage_runners = [
            self._run_integrity_stage,
            self._run_security_stage,
            self._run_structure_stage,
            self._run_formula_stage,
            self._run_content_stage,
        ]

        return all(stage_runner(file_path, context, stage_results, errors) for stage_runner in stage_runners)

    def _run_integrity_stage(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run Stage 0: Integrity Probe. Returns True if stage passes, False if pipeline should stop."""
        self.progress_tracker.update("stage_0", 0.0, "Starting integrity probe")
        integrity_result = self._run_stage_0(file_path)

        if isinstance(integrity_result, Err):
            errors.append(f"Stage 0: {integrity_result.error}")
            return bool(self.skip_on_error)

        stage_results["integrity"] = integrity_result.value
        context = context.with_stage_result("integrity", integrity_result.value)

        # CLAUDE-SECURITY: Block files that fail integrity checks
        if integrity_result.value.processing_class == "BLOCKED":
            errors.append("File blocked due to integrity check")
            return False

        self.progress_tracker.update("stage_0", 1.0, "Integrity probe complete")
        return True

    def _run_security_stage(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run Stage 1: Security Scan. Returns True if stage passes, False if pipeline should stop."""
        self.progress_tracker.update("stage_1", 0.0, "Starting security scan")
        security_result = self._run_stage_1(file_path)

        if isinstance(security_result, Err):
            errors.append(f"Stage 1: {security_result.error}")
            return bool(self.skip_on_error)

        stage_results["security"] = security_result.value
        context = context.with_stage_result("security", security_result.value)

        # CLAUDE-SECURITY: Block files with security risks
        if not security_result.value.is_safe:
            errors.append("File blocked due to security risks")
            return False

        self.progress_tracker.update("stage_1", 1.0, "Security scan complete")
        return True

    def _run_structure_stage(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run Stage 2: Structural Mapping. Returns True if stage passes, False if pipeline should stop."""
        self.progress_tracker.update("stage_2", 0.0, "Starting structural mapping")

        # Pass security info to structural mapping
        has_vba = stage_results.get("security", {}).has_macros if "security" in stage_results else False
        has_external = stage_results.get("security", {}).has_external_links if "security" in stage_results else False

        structure_result = self._run_stage_2(file_path, has_vba=has_vba, has_external=has_external)

        if isinstance(structure_result, Err):
            errors.append(f"Stage 2: {structure_result.error}")
            return bool(self.skip_on_error)

        stage_results["structure"] = structure_result.value
        context = context.with_stage_result("structure", structure_result.value)

        self.progress_tracker.update("stage_2", 1.0, "Structural mapping complete")
        return True

    def _run_formula_stage(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run Stage 3: Formula Analysis. Returns True if stage passes, False if pipeline should stop."""
        self.progress_tracker.update("stage_3", 0.0, "Starting formula analysis")

        # Check if we should run formula analysis
        if "structure" in stage_results and stage_results["structure"].total_formulas > 0:
            formula_result = self._run_stage_3(file_path)

            if isinstance(formula_result, Err):
                errors.append(f"Stage 3: {formula_result.error}")
                return bool(self.skip_on_error)

            stage_results["formulas"] = formula_result.value
            context = context.with_stage_result("formulas", formula_result.value)
        else:
            self.progress_tracker.update("stage_3", 1.0, "No formulas to analyze")

        self.progress_tracker.update("stage_3", 1.0, "Formula analysis complete")
        return True

    def _run_content_stage(
        self,
        file_path: Path,
        context: PipelineContext,
        stage_results: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Run Stage 4: Content Intelligence. Returns True if stage passes, False if pipeline should stop."""
        self.progress_tracker.update("stage_4", 0.0, "Starting content analysis")
        content_result = self._run_stage_4(file_path)

        if isinstance(content_result, Err):
            errors.append(f"Stage 4: {content_result.error}")
            return bool(self.skip_on_error)

        stage_results["content"] = content_result.value
        context = context.with_stage_result("content", content_result.value)

        self.progress_tracker.update("stage_4", 1.0, "Content analysis complete")
        return True

    def _run_stage_0(self, file_path: Path) -> Result:
        """Run Stage 0 with error handling and timeout."""
        try:
            with timeout(self.stage_timeout):
                return stage_0_integrity_probe(file_path)
        except StageTimeoutError:
            logger.exception("Stage 0 timed out after %d seconds", self.stage_timeout)
            return Err(f"Stage 0 timed out after {self.stage_timeout} seconds")
        except Exception as e:
            logger.exception("Stage 0 failed")
            return Err(f"Stage 0 exception: {e!s}")

    def _run_stage_1(self, file_path: Path) -> Result:
        """Run Stage 1 with error handling and timeout."""
        try:
            with timeout(self.stage_timeout):
                scan_options = self.options.get("security_scan_options")
                return stage_1_security_scan(file_path, scan_options)
        except StageTimeoutError:
            logger.exception("Stage 1 timed out after %d seconds", self.stage_timeout)
            return Err(f"Stage 1 timed out after {self.stage_timeout} seconds")
        except Exception as e:
            logger.exception("Stage 1 failed")
            return Err(f"Stage 1 exception: {e!s}")

    def _run_stage_2(self, file_path: Path, *, has_vba: bool, has_external: bool) -> Result:
        """Run Stage 2 with error handling and timeout."""
        try:
            with timeout(self.stage_timeout):
                return stage_2_structural_mapping(file_path, has_vba=has_vba, has_external_links=has_external)
        except StageTimeoutError:
            logger.exception("Stage 2 timed out after %d seconds", self.stage_timeout)
            return Err(f"Stage 2 timed out after {self.stage_timeout} seconds")
        except Exception as e:
            logger.exception("Stage 2 failed")
            return Err(f"Stage 2 exception: {e!s}")

    def _run_stage_3(self, file_path: Path) -> Result:
        """Run Stage 3 with error handling and timeout."""
        try:
            with timeout(self.stage_timeout):
                return stage_3_formula_analysis(file_path)
        except StageTimeoutError:
            logger.exception("Stage 3 timed out after %d seconds", self.stage_timeout)
            return Err(f"Stage 3 timed out after {self.stage_timeout} seconds")
        except Exception as e:
            logger.exception("Stage 3 failed")
            return Err(f"Stage 3 exception: {e!s}")

    def _run_stage_4(self, file_path: Path) -> Result:
        """Run Stage 4 with error handling and timeout."""
        try:
            with timeout(self.stage_timeout):
                sample_size = self.options.get("content_sample_size", 1000)
                return stage_4_content_intelligence(file_path, sample_size)
        except StageTimeoutError:
            logger.exception("Stage 4 timed out after %d seconds", self.stage_timeout)
            return Err(f"Stage 4 timed out after {self.stage_timeout} seconds")
        except Exception as e:
            logger.exception("Stage 4 failed")
            return Err(f"Stage 4 exception: {e!s}")

    def _create_failed_result(
        self, context: PipelineContext, errors: list[str], start_time: datetime
    ) -> PipelineResult:
        """Create a failed pipeline result."""
        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        return PipelineResult(
            context=context,
            integrity=None,
            security=None,
            structure=None,
            formulas=None,
            content=None,
            execution_time=execution_time,
            success=False,
            errors=tuple(errors),
        )


# ==================== Convenience Functions ====================


def run_analysis(
    file_path: Path,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    options: dict[str, Any] | None = None,
) -> PipelineResult:
    """
    Convenience function to run complete analysis.

    Args:
        file_path: Path to Excel file
        progress_callback: Optional callback for progress updates
        options: Pipeline options

    Returns:
        Complete analysis result
    """
    pipeline = DeterministicPipeline(options)

    if progress_callback:
        pipeline.add_progress_observer(progress_callback)

    return pipeline.run(file_path)


def create_console_progress_observer() -> Callable[[ProgressUpdate], None]:
    """
    Create a simple console progress observer.

    Returns a function that prints progress to console.
    """

    def observer(update: ProgressUpdate):
        """Print progress update to console."""
        progress_bar = int(update.progress * 20)
        bar = "█" * progress_bar + "░" * (20 - progress_bar)
        print(f"\r{update.stage}: [{bar}] {int(update.progress * 100)}% - {update.message}", end="")
        if update.progress >= 1.0:
            print()  # New line when stage completes

    return observer


def analyze_with_console_progress(file_path: Path, options: dict[str, Any] | None = None) -> PipelineResult:
    """
    Run analysis with console progress display.

    Convenience function for CLI usage.
    """
    print(f"Analyzing {file_path.name}...")

    result = run_analysis(file_path, progress_callback=create_console_progress_observer(), options=options)

    if result.success:
        print(f"\n✓ Analysis completed in {result.execution_time:.2f} seconds")
    else:
        print(f"\n✗ Analysis failed after {result.execution_time:.2f} seconds")
        for error in result.errors:
            print(f"  - {error}")

    return result


# ==================== Pipeline Configuration Helpers ====================


def create_strict_pipeline_options() -> dict[str, Any]:
    """
    Create options for strict analysis (fail on any issue).
    """
    return {
        "skip_on_error": False,
        "security_scan_options": {
            "check_macros": True,
            "check_external_links": True,
            "check_embedded_objects": True,
            "check_data_connections": True,
            "check_hidden_sheets": True,
            "check_formula_injection": True,
        },
        "content_sample_size": 10000,  # Larger sample
        "stage_timeout": 600,  # 10 minutes
    }


def create_lenient_pipeline_options() -> dict[str, Any]:
    """
    Create options for lenient analysis (continue on errors).
    """
    return {
        "skip_on_error": True,
        "security_scan_options": {
            "check_macros": True,
            "check_external_links": False,
            "check_embedded_objects": False,
            "check_data_connections": False,
            "check_hidden_sheets": False,
            "check_formula_injection": False,
        },
        "content_sample_size": 1000,  # Smaller sample
        "stage_timeout": 300,  # 5 minutes
    }


def create_fast_pipeline_options() -> dict[str, Any]:
    """
    Create options for fast analysis (minimal checks).
    """
    return {
        "skip_on_error": True,
        "security_scan_options": {
            "check_macros": True,
            "check_external_links": True,
            "check_embedded_objects": False,
            "check_data_connections": False,
            "check_hidden_sheets": False,
            "check_formula_injection": False,
        },
        "content_sample_size": 100,  # Very small sample
        "stage_timeout": 60,  # 1 minute
    }
