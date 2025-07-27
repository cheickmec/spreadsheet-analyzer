"""
Main notebook workflow orchestrator.

This module provides the high-level workflow that combines all components:
- Task selection based on file type and user preferences
- Notebook building with plugin-generated cells
- Optional execution with kernel management
- Quality assessment and validation
- File I/O with proper error handling

The workflow serves as the main API for generating analysis notebooks.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core_exec import (
    ExecutionBridge,
    ExecutionStats,
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookIO,
    QualityMetrics,
)
from ..plugins.base import QualityInspector, Task, registry


class WorkflowMode(Enum):
    """Workflow execution modes."""

    BUILD_ONLY = "build_only"  # Generate notebook without execution
    BUILD_AND_EXECUTE = "build_and_execute"  # Generate and execute
    EXECUTE_EXISTING = "execute_existing"  # Execute existing notebook


@dataclass
class WorkflowConfig:
    """
    Configuration for notebook workflow execution.

    Args:
        file_path: Path to the data file to analyze
        output_path: Where to save the generated notebook
        sheet_name: Excel sheet name (ignored for CSV files)
        mode: Execution mode (build only, build+execute, execute existing)
        tasks: Specific tasks to include (auto-detected if empty)
        kernel_profile: Kernel configuration for execution
        execute_timeout: Per-cell execution timeout in seconds
        quality_checks: Whether to run quality inspection
        auto_register_plugins: Whether to auto-discover and register plugins
    """

    file_path: str | None = None
    output_path: str | None = None
    sheet_name: str = "Sheet1"
    mode: WorkflowMode = WorkflowMode.BUILD_ONLY
    tasks: list[str] = field(default_factory=list)
    kernel_profile: KernelProfile | None = None
    execute_timeout: int = 300
    quality_checks: bool = True
    auto_register_plugins: bool = True

    def __post_init__(self):
        """Set defaults and validate configuration."""
        if self.kernel_profile is None:
            self.kernel_profile = KernelProfile()


@dataclass
class WorkflowResult:
    """
    Result from workflow execution.

    Args:
        notebook: The generated/executed notebook
        execution_stats: Statistics from execution (if executed)
        quality_metrics: Quality assessment results
        output_path: Path where notebook was saved
        errors: Any errors encountered during workflow
        warnings: Non-fatal warnings from workflow
    """

    notebook: NotebookBuilder
    execution_stats: ExecutionStats | None = None
    quality_metrics: QualityMetrics | None = None
    output_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether the workflow completed successfully."""
        return len(self.errors) == 0


class NotebookWorkflow:
    """
    Main workflow orchestrator for notebook generation and execution.

    This class provides the high-level API for creating analysis notebooks
    by coordinating between core execution primitives and domain-specific plugins.
    """

    def __init__(self):
        self.notebook_io = NotebookIO()
        self._kernel_service: KernelService | None = None
        self._execution_bridge: ExecutionBridge | None = None

    async def run(self, config: WorkflowConfig) -> WorkflowResult:
        """
        Run the complete workflow.

        Args:
            config: Workflow configuration

        Returns:
            WorkflowResult with notebook, execution stats, and quality metrics
        """
        result = WorkflowResult(notebook=NotebookBuilder())

        try:
            # Auto-register plugins if requested
            if config.auto_register_plugins:
                self._register_plugins()

            # Determine workflow mode and execute accordingly
            if config.mode == WorkflowMode.BUILD_ONLY:
                await self._build_notebook(config, result)
            elif config.mode == WorkflowMode.BUILD_AND_EXECUTE:
                await self._build_notebook(config, result)
                await self._execute_notebook(config, result)
            elif config.mode == WorkflowMode.EXECUTE_EXISTING:
                await self._load_and_execute_notebook(config, result)

            # Run quality checks if requested
            if config.quality_checks:
                await self._assess_quality(config, result)

            # Save notebook if output path specified
            if config.output_path:
                await self._save_notebook(config, result)

        except Exception as e:
            result.errors.append(f"Workflow error: {e!s}")

        return result

    async def _build_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Build notebook with task-generated cells."""
        try:
            # Create context for tasks
            context = {
                "file_path": config.file_path,
                "sheet_name": config.sheet_name,
                "workflow_mode": config.mode.value,
            }

            # Select tasks based on config and file type
            selected_tasks = self._select_tasks(config, context)

            if not selected_tasks:
                result.warnings.append("No tasks selected for notebook generation")
                return

            # Generate cells from tasks
            for task in selected_tasks:
                try:
                    # Validate task context
                    task_issues = task.validate_context(context)
                    if task_issues:
                        for issue in task_issues:
                            result.warnings.append(f"Task {task.name}: {issue}")
                        continue

                    # Generate cells
                    cells = task.build_initial_cells(context)
                    for cell in cells:
                        result.notebook.add_cell(cell)

                    # Add separator comment as markdown cell
                    result.notebook.add_markdown_cell(f"---\n\n*End of {task.name}*\n\n---")

                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    result.warnings.append(f"Task {task.name} failed: {e!s}")
                    print(f"\n❌ Task {task.name} failed with traceback:\n{tb}")

        except Exception as e:
            result.errors.append(f"Notebook building failed: {e!s}")

    async def _execute_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Execute notebook cells using kernel service."""
        try:
            # Initialize kernel service if needed
            if self._kernel_service is None:
                self._kernel_service = KernelService(config.kernel_profile)

            if self._execution_bridge is None:
                self._execution_bridge = ExecutionBridge(self._kernel_service)

            # Execute notebook
            session_id = await self._kernel_service.create_session("default")
            try:
                # ExecutionBridge returns the updated notebook, not stats
                executed_notebook = await self._execution_bridge.execute_notebook(session_id, result.notebook)
                result.notebook = executed_notebook

                # Extract execution stats from notebook metadata if available
                notebook_dict = result.notebook.to_dict()
                if "metadata" in notebook_dict and "execution_stats" in notebook_dict["metadata"]:
                    stats_data = notebook_dict["metadata"]["execution_stats"]
                    result.execution_stats = ExecutionStats(
                        total_cells=stats_data.get("total_cells", 0),
                        executed_cells=stats_data.get("executed_cells", 0),
                        skipped_cells=stats_data.get("skipped_cells", 0),
                        error_cells=stats_data.get("error_cells", 0),
                        total_duration_seconds=stats_data.get("total_duration_seconds", 0.0),
                    )

                # Post-process with tasks if they support it
                context = {
                    "file_path": config.file_path,
                    "sheet_name": config.sheet_name,
                    "execution_stats": result.execution_stats,
                }

                selected_tasks = self._select_tasks(config, context)
                for task in selected_tasks:
                    try:
                        additional_cells = task.postprocess(result.notebook, context)
                        for cell in additional_cells:
                            result.notebook.add_cell(cell)
                    except Exception as e:
                        result.warnings.append(f"Task {task.name} postprocess failed: {e!s}")
            finally:
                # Clean up session if the service has a close_session method
                if hasattr(self._kernel_service, "close_session"):
                    await self._kernel_service.close_session(session_id)

        except Exception as e:
            result.errors.append(f"Notebook execution failed: {e!s}")

    async def _load_and_execute_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Load existing notebook and execute it."""
        try:
            if not config.file_path or not Path(config.file_path).exists():
                result.errors.append("No valid notebook file path provided for execution")
                return

            # Load existing notebook
            result.notebook = self.notebook_io.read_notebook(config.file_path)

            # Execute it
            await self._execute_notebook(config, result)

        except Exception as e:
            result.errors.append(f"Loading and executing notebook failed: {e!s}")

    async def _assess_quality(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Run quality assessment on the notebook."""
        try:
            # Create context for quality assessment
            context = {
                "file_path": config.file_path,
                "sheet_name": config.sheet_name,
                "execution_stats": result.execution_stats,
            }

            # Find appropriate quality inspector
            inspector = self._select_quality_inspector(config, context)

            if inspector:
                result.quality_metrics = inspector.inspect(result.notebook, context)
            else:
                result.warnings.append("No quality inspector available")

        except Exception as e:
            result.warnings.append(f"Quality assessment failed: {e!s}")

    async def _save_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Save notebook to file."""
        try:
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.notebook_io.write_notebook(result.notebook, str(output_path), overwrite=True)
            result.output_path = str(output_path)

        except Exception as e:
            result.errors.append(f"Failed to save notebook: {e!s}")

    def _register_plugins(self) -> None:
        """Auto-register available plugins."""
        try:
            # Register spreadsheet plugins
            from ..plugins.spreadsheet import register_all_plugins

            register_all_plugins()
        except ImportError:
            pass  # Plugins not available

    def _select_tasks(self, config: WorkflowConfig, context: dict[str, Any]) -> list[Task]:
        """Select appropriate tasks based on configuration and context."""

        # If specific tasks requested, use those
        if config.tasks:
            selected = []
            for task_name in config.tasks:
                task = registry.get_task(task_name)
                if task:
                    selected.append(task)
                else:
                    # This will add to warnings in the calling method
                    pass
            return selected

        # Auto-select based on file type
        file_path = context.get("file_path", "")
        if not file_path:
            return []

        file_ext = Path(file_path).suffix.lower()

        # Spreadsheet file types
        if file_ext in [".xlsx", ".xls", ".xlsm", ".csv"]:
            tasks = [
                registry.get_task("data_profiling"),
                registry.get_task("formula_analysis") if file_ext != ".csv" else None,
                registry.get_task("outlier_detection"),
            ]
            # Filter out None values
            return [task for task in tasks if task is not None]

        # Default: return all available tasks
        return registry.list_tasks()

    def _select_quality_inspector(self, config: WorkflowConfig, context: dict[str, Any]) -> QualityInspector | None:
        """Select appropriate quality inspector."""

        file_path = context.get("file_path", "")
        if not file_path:
            return None

        file_ext = Path(file_path).suffix.lower()

        # Use spreadsheet inspector for spreadsheet files
        if file_ext in [".xlsx", ".xls", ".xlsm", ".csv"]:
            return registry.get_quality_inspector("spreadsheet_quality")

        # Default inspector
        return registry.get_quality_inspector("core")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._kernel_service:
            await self._kernel_service.shutdown()


# Convenience functions for common use cases


async def create_analysis_notebook(
    file_path: str, output_path: str, sheet_name: str = "Sheet1", execute: bool = False
) -> WorkflowResult:
    """
    Convenience function to create an analysis notebook.

    Args:
        file_path: Path to the data file
        output_path: Where to save the notebook
        sheet_name: Excel sheet name
        execute: Whether to execute the notebook

    Returns:
        WorkflowResult with the generated notebook
    """

    config = WorkflowConfig(
        file_path=file_path,
        output_path=output_path,
        sheet_name=sheet_name,
        mode=WorkflowMode.BUILD_AND_EXECUTE if execute else WorkflowMode.BUILD_ONLY,
    )

    workflow = NotebookWorkflow()
    try:
        return await workflow.run(config)
    finally:
        await workflow.cleanup()


async def execute_notebook(notebook_path: str, output_path: str | None = None) -> WorkflowResult:
    """
    Convenience function to execute an existing notebook.

    Args:
        notebook_path: Path to the notebook file
        output_path: Where to save the executed notebook (optional)

    Returns:
        WorkflowResult with execution statistics
    """

    config = WorkflowConfig(
        file_path=notebook_path, output_path=output_path or notebook_path, mode=WorkflowMode.EXECUTE_EXISTING
    )

    workflow = NotebookWorkflow()
    try:
        return await workflow.run(config)
    finally:
        await workflow.cleanup()
