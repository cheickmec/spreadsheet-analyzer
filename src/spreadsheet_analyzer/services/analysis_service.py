"""Analysis service for spreadsheet processing.

This service encapsulates all business logic for analyzing Excel files and can be
used from both CLI commands and future API endpoints.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from spreadsheet_analyzer.logging_config import get_logger
from spreadsheet_analyzer.pipeline import DeterministicPipeline
from spreadsheet_analyzer.pipeline.types import (
    AnalysisOptions,
    AnalysisResult,
    PipelineResult,
)

# Type alias for progress callback
ProgressCallback = Callable[[str, float, str], None]


class AnalysisService:
    """Service layer that encapsulates business logic for spreadsheet analysis.

    This service can be called from CLI commands, API endpoints, or used
    programmatically. It handles the orchestration of the analysis pipeline
    and provides a clean interface for different front-ends.
    """

    def __init__(self, *, pipeline_factory=None):
        """Initialize the analysis service.

        Args:
            pipeline_factory: Optional factory function for creating pipeline instances.
                             Defaults to creating standard AnalysisPipeline.
        """
        self._pipeline_factory = pipeline_factory or self._create_default_pipeline
        self._logger = get_logger(__name__)

    def _create_default_pipeline(self) -> DeterministicPipeline:
        """Create default analysis pipeline."""
        return DeterministicPipeline()

    async def analyze_file(self, file_path: Path, options: AnalysisOptions | None = None) -> AnalysisResult:
        """Analyze a single Excel file with given options.

        This method is async-ready for future API compatibility and can handle
        progress callbacks for real-time updates.

        Args:
            file_path: Path to the Excel file to analyze
            options: Analysis options, defaults to standard settings

        Returns:
            AnalysisResult with complete analysis information
        """
        options = options or AnalysisOptions()

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create result object
        result = AnalysisResult(
            file_path=file_path,
            file_size=file_path.stat().st_size,
            analysis_mode=options.mode,
        )

        self._logger.info("Starting analysis", file=str(file_path), mode=options.mode, size=result.file_size)

        try:
            # Create and configure pipeline
            pipeline = self._pipeline_factory()

            # Set progress callback if provided
            if options.progress_callback:
                # Wrap callback to work with pipeline's expected signature
                from spreadsheet_analyzer.pipeline.types import ProgressUpdate

                def pipeline_observer(update: ProgressUpdate) -> None:
                    options.progress_callback(update.stage, update.progress, update.message)

                pipeline.add_progress_observer(pipeline_observer)

            # Run pipeline analysis
            pipeline_result = await self._run_pipeline_async(pipeline, file_path, options)

            # Extract results from pipeline
            self._populate_result_from_pipeline(result, pipeline_result, options)

            # Mark completion
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

            self._logger.info(
                "Analysis complete",
                duration=result.duration_seconds,
                issues=len(result.issues),
                warnings=len(result.warnings),
            )

        except Exception as e:
            self._logger.exception("Analysis failed")
            result.issues.append(
                {"type": "analysis_error", "severity": "critical", "message": str(e), "stage": "pipeline"}
            )
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
            return result

        return result

    async def _run_pipeline_async(
        self,
        pipeline: DeterministicPipeline,
        file_path: Path,
        options: AnalysisOptions,  # noqa: ARG002
    ) -> PipelineResult:
        """Run pipeline analysis asynchronously.

        This wrapper allows us to run the synchronous pipeline in an async context,
        preparing for future async pipeline implementations.
        """
        loop = asyncio.get_event_loop()

        # Run synchronous pipeline in thread pool to avoid blocking
        return await loop.run_in_executor(None, pipeline.run, file_path)

    def _populate_result_from_pipeline(
        self, result: AnalysisResult, pipeline_result: PipelineResult, options: AnalysisOptions
    ) -> None:
        """Populate AnalysisResult from PipelineResult."""
        # Copy stage results
        result.integrity = pipeline_result.integrity
        result.security = pipeline_result.security
        result.structure = pipeline_result.structure
        result.formulas = pipeline_result.formulas
        result.content = pipeline_result.content

        # Extract issues and warnings
        self._extract_issues_and_warnings(result, pipeline_result)

        # Generate statistics if requested
        if options.include_statistics:
            result.statistics = self._generate_statistics(pipeline_result)

    def _extract_issues_and_warnings(self, result: AnalysisResult, pipeline_result: PipelineResult) -> None:
        """Extract issues and warnings from pipeline results."""
        # Security issues
        if pipeline_result.security:
            for threat in pipeline_result.security.threats:
                if threat.risk_level in ["CRITICAL", "HIGH"]:
                    result.issues.append(
                        {
                            "type": "security",
                            "severity": threat.risk_level.lower(),
                            "message": threat.description,
                            "stage": "security",
                            "location": threat.location,
                        }
                    )
                else:
                    result.warnings.append(
                        {
                            "type": "security",
                            "severity": threat.risk_level.lower(),
                            "message": threat.description,
                            "stage": "security",
                            "location": threat.location,
                        }
                    )

        # Formula issues
        if pipeline_result.formulas:
            if pipeline_result.formulas.circular_references:
                result.issues.append(
                    {
                        "type": "circular_reference",
                        "severity": "high",
                        "message": f"Found {len(pipeline_result.formulas.circular_references)} circular references",
                        "stage": "formulas",
                        "details": [list(ref) for ref in pipeline_result.formulas.circular_references],
                    }
                )

            # Check for volatile formulas (can be performance issues)
            if pipeline_result.formulas.volatile_formulas:
                result.warnings.append(
                    {
                        "type": "volatile_formulas",
                        "severity": "medium",
                        "message": f"Found {len(pipeline_result.formulas.volatile_formulas)} volatile formulas that recalculate on every change",
                        "stage": "formulas",
                        "count": len(pipeline_result.formulas.volatile_formulas),
                    }
                )

        # Content insights and issues
        if pipeline_result.content:
            for insight in pipeline_result.content.insights:
                if insight.severity in ["HIGH", "CRITICAL"]:
                    result.issues.append(
                        {
                            "type": "content_insight",
                            "severity": insight.severity.lower(),
                            "message": insight.description,
                            "stage": "content",
                            "title": insight.title,
                            "affected_areas": list(insight.affected_areas),
                        }
                    )
                elif insight.severity in ["MEDIUM", "LOW"]:
                    result.warnings.append(
                        {
                            "type": "content_insight",
                            "severity": insight.severity.lower(),
                            "message": insight.description,
                            "stage": "content",
                            "title": insight.title,
                        }
                    )

    def _generate_statistics(self, pipeline_result: PipelineResult) -> dict[str, Any]:
        """Generate analysis statistics."""
        stats: dict[str, Any] = {"file_metrics": {}, "analysis_coverage": {}, "complexity_metrics": {}}

        # File metrics
        if pipeline_result.structure:
            stats["file_metrics"] = {
                "sheet_count": pipeline_result.structure.sheet_count,
                "total_cells": pipeline_result.structure.total_cells,
                "total_formulas": pipeline_result.structure.total_formulas,
                "complexity_score": pipeline_result.structure.complexity_score,
            }

        # Formula metrics
        if pipeline_result.formulas:
            stats["complexity_metrics"] = {
                "formula_count": len(pipeline_result.formulas.dependency_graph),
                "max_dependency_depth": pipeline_result.formulas.max_dependency_depth,
                "formula_complexity_score": pipeline_result.formulas.formula_complexity_score,
                "circular_references": len(pipeline_result.formulas.circular_references),
                "volatile_formulas": len(pipeline_result.formulas.volatile_formulas),
                "external_references": len(pipeline_result.formulas.external_references),
            }

        # Analysis coverage
        stats["analysis_coverage"] = {
            "integrity": pipeline_result.integrity is not None,
            "security": pipeline_result.security is not None,
            "structure": pipeline_result.structure is not None,
            "formulas": pipeline_result.formulas is not None,
            "content": pipeline_result.content is not None,
        }

        return stats

    async def analyze_batch(
        self, file_paths: list[Path], options: AnalysisOptions | None = None, *, max_concurrent: int = 4
    ) -> list[AnalysisResult]:
        """Analyze multiple Excel files concurrently.

        Args:
            file_paths: List of paths to analyze
            options: Analysis options to apply to all files
            max_concurrent: Maximum number of concurrent analyses

        Returns:
            List of AnalysisResult objects
        """
        options = options or AnalysisOptions()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(file_path: Path) -> AnalysisResult:
            async with semaphore:
                return await self.analyze_file(file_path, options)

        # Run analyses concurrently
        tasks = [analyze_with_semaphore(file_path) for file_path in file_paths]

        return await asyncio.gather(*tasks)

    def analyze_file_sync(self, file_path: Path, options: AnalysisOptions | None = None) -> AnalysisResult:
        """Synchronous wrapper for analyze_file.

        This is provided for backwards compatibility and simple use cases.
        """
        return asyncio.run(self.analyze_file(file_path, options))
