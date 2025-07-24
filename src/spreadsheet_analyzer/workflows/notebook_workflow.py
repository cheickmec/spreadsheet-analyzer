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

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from ..core_exec import (
    KernelService, KernelProfile, ExecutionBridge, ExecutionStats,
    NotebookBuilder, NotebookIO, QualityMetrics
)
from ..plugins.base import registry, Task, QualityInspector


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
    file_path: Optional[str] = None
    output_path: Optional[str] = None
    sheet_name: str = "Sheet1"
    mode: WorkflowMode = WorkflowMode.BUILD_ONLY
    tasks: List[str] = field(default_factory=list)
    kernel_profile: Optional[KernelProfile] = None
    execute_timeout: int = 300
    quality_checks: bool = True
    auto_register_plugins: bool = True
    
    def __post_init__(self):
        """Set defaults and validate configuration."""
        if self.kernel_profile is None:
            self.kernel_profile = KernelProfile.create_default()


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
    execution_stats: Optional[ExecutionStats] = None
    quality_metrics: Optional[QualityMetrics] = None
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
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
        self._kernel_service: Optional[KernelService] = None
        self._execution_bridge: Optional[ExecutionBridge] = None
    
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
            result.errors.append(f"Workflow error: {str(e)}")
        
        return result
    
    async def _build_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Build notebook with task-generated cells."""
        try:
            # Create context for tasks
            context = {
                'file_path': config.file_path,
                'sheet_name': config.sheet_name,
                'workflow_mode': config.mode.value
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
                    
                    result.notebook.add_cell_separator(f"End of {task.name}")
                    
                except Exception as e:
                    result.warnings.append(f"Task {task.name} failed: {str(e)}")
                    
        except Exception as e:
            result.errors.append(f"Notebook building failed: {str(e)}")
    
    async def _execute_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Execute notebook cells using kernel service."""
        try:
            # Initialize kernel service if needed
            if self._kernel_service is None:
                self._kernel_service = KernelService()
            
            if self._execution_bridge is None:
                self._execution_bridge = ExecutionBridge(self._kernel_service)
            
            # Execute notebook
            async with self._kernel_service.create_session(config.kernel_profile) as session_id:
                result.execution_stats = await self._execution_bridge.execute_notebook(
                    result.notebook,
                    session_id,
                    timeout=config.execute_timeout
                )
                
                # Post-process with tasks if they support it
                context = {
                    'file_path': config.file_path,
                    'sheet_name': config.sheet_name,
                    'execution_stats': result.execution_stats
                }
                
                selected_tasks = self._select_tasks(config, context)
                for task in selected_tasks:
                    try:
                        additional_cells = task.postprocess(result.notebook, context)
                        for cell in additional_cells:
                            result.notebook.add_cell(cell)
                    except Exception as e:
                        result.warnings.append(f"Task {task.name} postprocess failed: {str(e)}")
                
        except Exception as e:
            result.errors.append(f"Notebook execution failed: {str(e)}")
    
    async def _load_and_execute_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Load existing notebook and execute it."""
        try:
            if not config.file_path or not Path(config.file_path).exists():
                result.errors.append("No valid notebook file path provided for execution")
                return
            
            # Load existing notebook
            result.notebook = self.notebook_io.load_notebook(config.file_path)
            
            # Execute it
            await self._execute_notebook(config, result)
            
        except Exception as e:
            result.errors.append(f"Loading and executing notebook failed: {str(e)}")
    
    async def _assess_quality(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Run quality assessment on the notebook."""
        try:
            # Create context for quality assessment
            context = {
                'file_path': config.file_path,
                'sheet_name': config.sheet_name,
                'execution_stats': result.execution_stats
            }
            
            # Find appropriate quality inspector
            inspector = self._select_quality_inspector(config, context)
            
            if inspector:
                result.quality_metrics = inspector.inspect(result.notebook, context)
            else:
                result.warnings.append("No quality inspector available")
                
        except Exception as e:
            result.warnings.append(f"Quality assessment failed: {str(e)}")
    
    async def _save_notebook(self, config: WorkflowConfig, result: WorkflowResult) -> None:
        """Save notebook to file."""
        try:
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.notebook_io.save_notebook(result.notebook, str(output_path))
            result.output_path = str(output_path)
            
        except Exception as e:
            result.errors.append(f"Failed to save notebook: {str(e)}")
    
    def _register_plugins(self) -> None:
        """Auto-register available plugins."""
        try:
            # Register spreadsheet plugins
            from ..plugins.spreadsheet import register_all_plugins
            register_all_plugins()
        except ImportError:
            pass  # Plugins not available
    
    def _select_tasks(self, config: WorkflowConfig, context: Dict[str, Any]) -> List[Task]:
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
        file_path = context.get('file_path', '')
        if not file_path:
            return []
        
        file_ext = Path(file_path).suffix.lower()
        
        # Spreadsheet file types
        if file_ext in ['.xlsx', '.xls', '.xlsm', '.csv']:
            return [
                registry.get_task('data_profiling'),
                registry.get_task('formula_analysis') if file_ext != '.csv' else None,
                registry.get_task('outlier_detection')
            ]
        
        # Default: return all available tasks
        return registry.list_tasks()
    
    def _select_quality_inspector(self, config: WorkflowConfig, context: Dict[str, Any]) -> Optional[QualityInspector]:
        """Select appropriate quality inspector."""
        
        file_path = context.get('file_path', '')
        if not file_path:
            return None
        
        file_ext = Path(file_path).suffix.lower()
        
        # Use spreadsheet inspector for spreadsheet files
        if file_ext in ['.xlsx', '.xls', '.xlsm', '.csv']:
            return registry.get_quality_inspector('spreadsheet_quality')
        
        # Default inspector
        return registry.get_quality_inspector('core')
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._kernel_service:
            await self._kernel_service.cleanup()


# Convenience functions for common use cases

async def create_analysis_notebook(
    file_path: str,
    output_path: str,
    sheet_name: str = "Sheet1",
    execute: bool = False
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
        mode=WorkflowMode.BUILD_AND_EXECUTE if execute else WorkflowMode.BUILD_ONLY
    )
    
    workflow = NotebookWorkflow()
    try:
        return await workflow.run(config)
    finally:
        await workflow.cleanup()


async def execute_notebook(
    notebook_path: str,
    output_path: Optional[str] = None
) -> WorkflowResult:
    """
    Convenience function to execute an existing notebook.
    
    Args:
        notebook_path: Path to the notebook file
        output_path: Where to save the executed notebook (optional)
        
    Returns:
        WorkflowResult with execution statistics
    """
    
    config = WorkflowConfig(
        file_path=notebook_path,
        output_path=output_path or notebook_path,
        mode=WorkflowMode.EXECUTE_EXISTING
    )
    
    workflow = NotebookWorkflow()
    try:
        return await workflow.run(config)
    finally:
        await workflow.cleanup() 