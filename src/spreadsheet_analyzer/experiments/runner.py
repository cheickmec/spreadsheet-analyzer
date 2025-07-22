"""Experiment runner for testing LLM performance on Excel analysis tasks.

This module provides a framework for running experiments with different LLMs,
strategies, and Excel files to compare their effectiveness at spreadsheet analysis.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from spreadsheet_analyzer.excel_to_notebook import ExcelToNotebookConverter
from spreadsheet_analyzer.jupyter_kernel import KernelManager
from spreadsheet_analyzer.notebook_llm.llm_providers import get_provider
from spreadsheet_analyzer.notebook_llm.orchestration.python_orchestrator import (
    AnalysisConfig,
    PythonWorkflowOrchestrator,
)
from spreadsheet_analyzer.notebook_llm.strategies.base import AnalysisFocus

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    excel_files: list[Path]
    llm_providers: list[str]  # e.g., ["openai", "anthropic"]
    llm_models: dict[str, str]  # e.g., {"openai": "gpt-4", "anthropic": "claude-3-sonnet"}
    strategies: list[str]  # e.g., ["hierarchical", "graph_based"]
    analysis_focus: AnalysisFocus = AnalysisFocus.GENERAL
    output_dir: Path = Path("experiments/results")
    max_parallel_kernels: int = 2
    timeout_seconds: int = 300


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    provider: str
    model: str
    strategy: str
    excel_file: Path
    success: bool
    duration_seconds: float
    tokens_used: dict[str, int] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Runs experiments comparing different LLMs on Excel analysis tasks."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.kernel_manager = KernelManager(max_kernels=config.max_parallel_kernels)
        self.excel_converter = ExcelToNotebookConverter()
        self.results: list[ExperimentResult] = []

    async def run(self) -> list[ExperimentResult]:
        """Run the configured experiments.

        Returns:
            List of experiment results
        """
        logger.info("Starting experiment: %s", self.config.name)

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Start kernel manager
        await self.kernel_manager.start()

        try:
            # Run experiments for each combination
            tasks = []
            for excel_file in self.config.excel_files:
                for provider_name in self.config.llm_providers:
                    model = self.config.llm_models.get(provider_name)
                    if not model:
                        logger.warning("No model specified for provider %s", provider_name)
                        continue

                    for strategy in self.config.strategies:
                        task = self._run_single_experiment(excel_file, provider_name, model, strategy)
                        tasks.append(task)

            # Run experiments with concurrency limit
            self.results = await self._run_with_concurrency(tasks)

            # Save results
            self._save_results()

            return self.results

        finally:
            # Cleanup
            await self.kernel_manager.stop()

    async def _run_single_experiment(
        self,
        excel_file: Path,
        provider_name: str,
        model: str,
        strategy: str,
    ) -> ExperimentResult:
        """Run a single experiment configuration.

        Args:
            excel_file: Excel file to analyze
            provider_name: LLM provider name
            model: Model name
            strategy: Analysis strategy

        Returns:
            Experiment result
        """
        start_time = datetime.now()
        result = ExperimentResult(
            provider=provider_name,
            model=model,
            strategy=strategy,
            excel_file=excel_file,
            success=False,
            duration_seconds=0,
        )

        try:
            logger.info(
                "Running experiment: %s with %s/%s on %s",
                strategy,
                provider_name,
                model,
                excel_file.name,
            )

            # Convert Excel to notebook
            notebook_doc = self.excel_converter.convert(excel_file)

            # Create a kernel for this experiment
            kernel_id = await self.kernel_manager.create_kernel()

            try:
                # Get LLM provider
                llm = get_provider(provider_name, model=model)

                # Create orchestrator
                orchestrator = PythonWorkflowOrchestrator(
                    kernel_manager=self.kernel_manager,
                    llm=llm,
                    strategy=strategy,
                )

                # Run analysis
                analysis_config = AnalysisConfig(
                    focus=self.config.analysis_focus,
                    max_iterations=3,
                    enable_validation=True,
                    temperature=0.7,
                )

                analysis_result = await asyncio.wait_for(
                    orchestrator.analyze_spreadsheet(
                        kernel_id,
                        excel_file,
                        analysis_config,
                    ),
                    timeout=self.config.timeout_seconds,
                )

                # Extract results
                result.success = analysis_result.success
                result.insights = analysis_result.insights
                result.tokens_used = {
                    "total": analysis_result.total_tokens_used,
                    "prompt": analysis_result.metadata.get("prompt_tokens", 0),
                    "completion": analysis_result.metadata.get("completion_tokens", 0),
                }
                result.metadata = {
                    "confidence": analysis_result.confidence,
                    "iterations": analysis_result.metadata.get("iterations", 1),
                    "patterns_found": len(analysis_result.metadata.get("patterns", [])),
                }

            finally:
                # Cleanup kernel
                await self.kernel_manager.shutdown_kernel(kernel_id)

        except TimeoutError:
            result.errors.append(f"Experiment timed out after {self.config.timeout_seconds}s")
            logger.error("Experiment timed out: %s", result)
        except Exception as e:
            result.errors.append(str(e))
            logger.exception("Experiment failed")

        # Calculate duration
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        return result

    async def _run_with_concurrency(self, tasks: list[asyncio.Task]) -> list[ExperimentResult]:
        """Run tasks with concurrency limit.

        Args:
            tasks: List of experiment tasks

        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(self.config.max_parallel_kernels)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        # Wrap tasks with semaphore
        limited_tasks = [run_with_semaphore(task) for task in tasks]

        # Run all tasks
        return await asyncio.gather(*limited_tasks, return_exceptions=False)

    def _save_results(self) -> None:
        """Save experiment results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_dir / f"{self.config.name}_{timestamp}.json"

        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "provider": result.provider,
                    "model": result.model,
                    "strategy": result.strategy,
                    "excel_file": str(result.excel_file),
                    "success": result.success,
                    "duration_seconds": result.duration_seconds,
                    "tokens_used": result.tokens_used,
                    "insights": result.insights,
                    "errors": result.errors,
                    "metadata": result.metadata,
                }
            )

        # Save to JSON
        with output_file.open("w") as f:
            json.dump(
                {
                    "experiment": self.config.name,
                    "timestamp": timestamp,
                    "config": {
                        "providers": self.config.llm_providers,
                        "models": self.config.llm_models,
                        "strategies": self.config.strategies,
                        "focus": self.config.analysis_focus.value,
                    },
                    "results": results_data,
                },
                f,
                indent=2,
            )

        logger.info("Results saved to: %s", output_file)

        # Generate summary report
        self._generate_summary_report(output_file.parent / f"{self.config.name}_{timestamp}_summary.md")

    def _generate_summary_report(self, output_file: Path) -> None:
        """Generate a markdown summary report."""
        lines = [
            f"# Experiment Summary: {self.config.name}",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Configuration",
            f"- **Providers**: {', '.join(self.config.llm_providers)}",
            f"- **Strategies**: {', '.join(self.config.strategies)}",
            f"- **Excel Files**: {len(self.config.excel_files)}",
            f"- **Total Runs**: {len(self.results)}",
            "\n## Results Summary",
        ]

        # Calculate success rate by provider/strategy
        success_rates = {}
        for result in self.results:
            key = f"{result.provider}/{result.model} - {result.strategy}"
            if key not in success_rates:
                success_rates[key] = {"success": 0, "total": 0}
            success_rates[key]["total"] += 1
            if result.success:
                success_rates[key]["success"] += 1

        lines.append("\n### Success Rates")
        for key, stats in success_rates.items():
            rate = stats["success"] / stats["total"] * 100
            lines.append(f"- **{key}**: {rate:.1f}% ({stats['success']}/{stats['total']})")

        # Average performance metrics
        lines.append("\n### Performance Metrics")
        for provider in self.config.llm_providers:
            provider_results = [r for r in self.results if r.provider == provider]
            if provider_results:
                avg_duration = sum(r.duration_seconds for r in provider_results) / len(provider_results)
                avg_tokens = sum(r.tokens_used.get("total", 0) for r in provider_results) / len(provider_results)
                lines.append(f"\n#### {provider}")
                lines.append(f"- Average Duration: {avg_duration:.2f}s")
                lines.append(f"- Average Tokens: {avg_tokens:.0f}")

        # Sample insights
        lines.append("\n## Sample Insights")
        for i, result in enumerate(self.results[:5]):
            if result.insights:
                lines.append(f"\n### Run {i + 1}: {result.provider}/{result.strategy}")
                for insight in result.insights[:3]:
                    lines.append(f"- {insight}")

        # Write report
        output_file.write_text("\n".join(lines))
        logger.info("Summary report saved to: %s", output_file)
