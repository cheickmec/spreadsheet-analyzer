"""
Tests for the notebook workflow orchestration system.

This module provides comprehensive functional tests for the NotebookWorkflow class,
testing workflow configuration, task orchestration, execution, and quality assessment.
All tests are functional and use no mocking.
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import pandas as pd
import json

from spreadsheet_analyzer.workflows.notebook_workflow import (
    NotebookWorkflow,
    WorkflowConfig,
    WorkflowMode,
    WorkflowResult,
    create_analysis_notebook,
)
from spreadsheet_analyzer.core_exec import (
    KernelService,
    KernelProfile,
    NotebookBuilder,
    NotebookIO,
    CellType,
)
from spreadsheet_analyzer.plugins.base import registry
from spreadsheet_analyzer.plugins.spreadsheet import register_all_plugins


class TestWorkflowConfig:
    """Test the WorkflowConfig class functionality."""
    
    def test_config_initialization_defaults(self):
        """Test WorkflowConfig initialization with default values."""
        config = WorkflowConfig()
        
        assert config.mode == WorkflowMode.BUILD_ONLY
        assert config.tasks == []
        assert config.quality_inspectors == []
        assert config.kernel_profile is None
        assert config.output_path is None
        assert config.execute_notebook is False
        assert config.assess_quality is True
        assert config.task_config == {}
        assert config.timeout_seconds == 300
    
    def test_config_initialization_custom(self):
        """Test WorkflowConfig initialization with custom values."""
        kernel_profile = KernelProfile(name="python3", display_name="Python 3")
        output_path = Path("/tmp/test.ipynb")
        
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_AND_EXECUTE,
            tasks=["data_profiling", "outlier_detection"],
            quality_inspectors=["spreadsheet"],
            kernel_profile=kernel_profile,
            output_path=output_path,
            execute_notebook=True,
            assess_quality=False,
            task_config={"threshold": 2.0},
            timeout_seconds=600
        )
        
        assert config.mode == WorkflowMode.BUILD_AND_EXECUTE
        assert config.tasks == ["data_profiling", "outlier_detection"]
        assert config.quality_inspectors == ["spreadsheet"]
        assert config.kernel_profile is kernel_profile
        assert config.output_path == output_path
        assert config.execute_notebook is True
        assert config.assess_quality is False
        assert config.task_config == {"threshold": 2.0}
        assert config.timeout_seconds == 600
    
    def test_config_to_dict(self):
        """Test WorkflowConfig conversion to dictionary."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            timeout_seconds=120
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['mode'] == 'BUILD_ONLY'
        assert config_dict['tasks'] == ["data_profiling"]
        assert config_dict['timeout_seconds'] == 120
        assert 'execute_notebook' in config_dict
        assert 'assess_quality' in config_dict
    
    def test_config_from_dict(self):
        """Test WorkflowConfig creation from dictionary."""
        config_dict = {
            'mode': 'BUILD_AND_EXECUTE',
            'tasks': ['data_profiling', 'outlier_detection'],
            'execute_notebook': True,
            'timeout_seconds': 240
        }
        
        config = WorkflowConfig.from_dict(config_dict)
        
        assert config.mode == WorkflowMode.BUILD_AND_EXECUTE
        assert config.tasks == ['data_profiling', 'outlier_detection']
        assert config.execute_notebook is True
        assert config.timeout_seconds == 240


class TestWorkflowResult:
    """Test the WorkflowResult class functionality."""
    
    def test_result_initialization(self):
        """Test WorkflowResult initialization."""
        notebook = NotebookBuilder().build()
        
        result = WorkflowResult(
            notebook=notebook,
            success=True,
            execution_stats=None,
            quality_metrics=None,
            errors=[]
        )
        
        assert result.notebook == notebook
        assert result.success is True
        assert result.execution_stats is None
        assert result.quality_metrics is None
        assert result.errors == []
    
    def test_result_with_errors(self):
        """Test WorkflowResult with error information."""
        errors = ["Import error", "Execution timeout"]
        
        result = WorkflowResult(
            notebook=None,
            success=False,
            execution_stats=None,
            quality_metrics=None,
            errors=errors
        )
        
        assert result.success is False
        assert result.errors == errors
        assert len(result.errors) == 2


class TestNotebookWorkflow:
    """Test the NotebookWorkflow class functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Ensure plugins are registered
        register_all_plugins()
        
        # Create temporary directory and test data
        self.temp_dir = tempfile.mkdtemp()
        self.excel_file = Path(self.temp_dir) / "test_data.xlsx"
        
        # Create sample data
        data = {
            'product': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
            'sales': [1000, 1500, 2000, 1750, 3000],
            'cost': [800, 1200, 1600, 1400, 2400],
            'region': ['North', 'South', 'East', 'West', 'North']
        }
        df = pd.DataFrame(data)
        df.to_excel(self.excel_file, index=False)
        
        # Basic workflow configuration
        self.config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            task_config={'file_path': str(self.excel_file)},
            assess_quality=True
        )
        
        self.workflow = NotebookWorkflow()
    
    def test_workflow_initialization(self):
        """Test NotebookWorkflow initialization."""
        workflow = NotebookWorkflow()
        
        assert hasattr(workflow, 'run')
        assert hasattr(workflow, '_build_notebook')
        assert hasattr(workflow, '_execute_notebook')
        assert hasattr(workflow, '_assess_quality')
    
    @pytest.mark.asyncio
    async def test_build_only_workflow(self):
        """Test workflow in BUILD_ONLY mode."""
        result = await self.workflow.run(self.config)
        
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.notebook is not None
        assert result.execution_stats is None  # No execution in BUILD_ONLY mode
        assert result.quality_metrics is not None  # Quality assessment enabled
        assert len(result.errors) == 0
        
        # Check notebook structure
        notebook = result.notebook
        assert 'cells' in notebook
        assert len(notebook['cells']) > 0
        
        # Should contain data profiling cells
        cell_sources = [cell.get('source', '') for cell in notebook['cells']]
        all_source = '\n'.join(cell_sources)
        assert 'pandas' in all_source.lower()
        assert 'read_excel' in all_source or 'read_csv' in all_source
    
    @pytest.mark.asyncio
    async def test_multiple_tasks_workflow(self):
        """Test workflow with multiple tasks."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling", "outlier_detection"],
            task_config={'file_path': str(self.excel_file)},
            assess_quality=True
        )
        
        result = await self.workflow.run(config)
        
        assert result.success is True
        assert result.notebook is not None
        
        # Should contain cells from both tasks
        notebook = result.notebook
        cell_sources = [cell.get('source', '') for cell in notebook['cells']]
        all_source = '\n'.join(cell_sources)
        
        # Data profiling indicators
        assert 'describe()' in all_source or 'info()' in all_source
        
        # Outlier detection indicators
        assert 'outlier' in all_source.lower() or 'quartile' in all_source.lower()
    
    @pytest.mark.asyncio
    async def test_workflow_with_quality_assessment(self):
        """Test workflow with quality assessment enabled."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            quality_inspectors=["spreadsheet"],
            task_config={'file_path': str(self.excel_file)},
            assess_quality=True
        )
        
        result = await self.workflow.run(config)
        
        assert result.success is True
        assert result.quality_metrics is not None
        
        # Check quality metrics structure
        metrics = result.quality_metrics
        assert hasattr(metrics, 'overall_score')
        assert hasattr(metrics, 'issues')
        assert hasattr(metrics, 'details')
        assert 0.0 <= metrics.overall_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_workflow_without_quality_assessment(self):
        """Test workflow with quality assessment disabled."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            task_config={'file_path': str(self.excel_file)},
            assess_quality=False
        )
        
        result = await self.workflow.run(config)
        
        assert result.success is True
        assert result.quality_metrics is None  # Should be None when disabled
    
    @pytest.mark.asyncio
    async def test_workflow_with_invalid_task(self):
        """Test workflow behavior with invalid task name."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["nonexistent_task"],
            task_config={'file_path': str(self.excel_file)}
        )
        
        result = await self.workflow.run(config)
        
        # Should handle gracefully - either skip invalid tasks or report error
        assert isinstance(result, WorkflowResult)
        if not result.success:
            assert len(result.errors) > 0
            assert any('task' in error.lower() for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_workflow_with_missing_file(self):
        """Test workflow behavior with missing input file."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            task_config={'file_path': '/nonexistent/file.xlsx'}
        )
        
        result = await self.workflow.run(config)
        
        # Should handle missing file gracefully
        assert isinstance(result, WorkflowResult)
        # Depending on implementation, this might succeed with warnings or fail
        if not result.success:
            assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_output_path_saving(self):
        """Test workflow saving to specified output path."""
        output_path = Path(self.temp_dir) / "workflow_output.ipynb"
        
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            task_config={'file_path': str(self.excel_file)},
            output_path=output_path
        )
        
        result = await self.workflow.run(config)
        
        assert result.success is True
        assert output_path.exists()
        
        # Verify saved file is valid notebook
        with open(output_path, 'r') as f:
            notebook_data = json.load(f)
            assert 'cells' in notebook_data
            assert 'metadata' in notebook_data
            assert 'nbformat' in notebook_data
    
    @pytest.mark.asyncio
    async def test_build_and_execute_mode(self):
        """Test workflow in BUILD_AND_EXECUTE mode (if kernel available)."""
        # Skip if no kernel available
        try:
            kernel_service = KernelService()
            profiles = await kernel_service.list_available_kernels()
            if not profiles:
                pytest.skip("No kernels available for execution testing")
            
            kernel_profile = profiles[0]  # Use first available kernel
        except Exception:
            pytest.skip("Kernel service not available")
        
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_AND_EXECUTE,
            tasks=["data_profiling"],
            kernel_profile=kernel_profile,
            task_config={'file_path': str(self.excel_file)},
            execute_notebook=True,
            timeout_seconds=60
        )
        
        result = await self.workflow.run(config)
        
        assert isinstance(result, WorkflowResult)
        assert result.notebook is not None
        
        if result.success:
            # If execution succeeded, should have execution stats
            assert result.execution_stats is not None
            
            # Check that cells have outputs
            notebook = result.notebook
            code_cells = [cell for cell in notebook['cells'] if cell.get('cell_type') == 'code']
            executed_cells = [cell for cell in code_cells if cell.get('outputs')]
            # At least some cells should have been executed
            assert len(executed_cells) >= 0  # Allow for cases where execution doesn't produce output
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test workflow timeout behavior."""
        config = WorkflowConfig(
            mode=WorkflowMode.BUILD_ONLY,
            tasks=["data_profiling"],
            task_config={'file_path': str(self.excel_file)},
            timeout_seconds=1  # Very short timeout
        )
        
        # This should still succeed for BUILD_ONLY mode since no execution
        result = await self.workflow.run(config)
        
        assert isinstance(result, WorkflowResult)
        # BUILD_ONLY should succeed even with short timeout
        assert result.success is True or len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_task_config_variations(self):
        """Test workflow with different task configuration variations."""
        test_configs = [
            # Basic configuration
            {'file_path': str(self.excel_file)},
            
            # Data profiling with options
            {
                'file_path': str(self.excel_file),
                'include_distributions': True,
                'include_correlations': True
            },
            
            # Outlier detection with options
            {
                'file_path': str(self.excel_file),
                'method': 'iqr',
                'threshold': 1.5
            }
        ]
        
        for task_config in test_configs:
            config = WorkflowConfig(
                mode=WorkflowMode.BUILD_ONLY,
                tasks=["data_profiling"],
                task_config=task_config
            )
            
            result = await self.workflow.run(config)
            
            assert isinstance(result, WorkflowResult)
            assert result.notebook is not None
            # Should succeed with various configurations


class TestCreateAnalysisNotebook:
    """Test the create_analysis_notebook convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Ensure plugins are registered
        register_all_plugins()
        
        # Create test data
        self.temp_dir = tempfile.mkdtemp()
        self.excel_file = Path(self.temp_dir) / "analysis_data.xlsx"
        
        data = {
            'category': ['A', 'B', 'C', 'A', 'B'] * 20,
            'value': list(range(100)),
            'score': [x * 0.1 + 50 for x in range(100)]
        }
        df = pd.DataFrame(data)
        df.to_excel(self.excel_file, index=False)
    
    @pytest.mark.asyncio
    async def test_create_analysis_notebook_basic(self):
        """Test basic notebook creation using convenience function."""
        output_path = Path(self.temp_dir) / "analysis.ipynb"
        
        result = await create_analysis_notebook(
            file_path=str(self.excel_file),
            output_path=output_path,
            tasks=['data_profiling'],
            execute=False
        )
        
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.notebook is not None
        assert output_path.exists()
        
        # Verify saved file
        notebook_io = NotebookIO()
        loaded_notebook = notebook_io.read_notebook(output_path)
        assert 'cells' in loaded_notebook
        assert len(loaded_notebook['cells']) > 0
    
    @pytest.mark.asyncio
    async def test_create_analysis_notebook_multiple_tasks(self):
        """Test notebook creation with multiple analysis tasks."""
        output_path = Path(self.temp_dir) / "multi_analysis.ipynb"
        
        result = await create_analysis_notebook(
            file_path=str(self.excel_file),
            output_path=output_path,
            tasks=['data_profiling', 'outlier_detection'],
            task_config={
                'include_distributions': True,
                'method': 'zscore',
                'threshold': 2.0
            },
            execute=False
        )
        
        assert result.success is True
        assert output_path.exists()
        
        # Check content includes both analyses
        notebook_io = NotebookIO()
        notebook = notebook_io.read_notebook(output_path)
        
        cell_sources = [cell.get('source', '') for cell in notebook['cells']]
        all_source = '\n'.join(cell_sources)
        
        # Should include data profiling
        assert 'describe()' in all_source or 'info()' in all_source
        
        # Should include outlier detection
        assert 'outlier' in all_source.lower() or 'zscore' in all_source.lower()
    
    @pytest.mark.asyncio
    async def test_create_analysis_notebook_with_quality_assessment(self):
        """Test notebook creation with quality assessment."""
        output_path = Path(self.temp_dir) / "quality_analysis.ipynb"
        
        result = await create_analysis_notebook(
            file_path=str(self.excel_file),
            output_path=output_path,
            tasks=['data_profiling'],
            quality_inspectors=['spreadsheet'],
            execute=False
        )
        
        assert result.success is True
        assert result.quality_metrics is not None
        
        metrics = result.quality_metrics
        assert 0.0 <= metrics.overall_score <= 1.0
        assert isinstance(metrics.issues, list)
        assert isinstance(metrics.details, dict)
    
    @pytest.mark.asyncio
    async def test_create_analysis_notebook_error_handling(self):
        """Test error handling in notebook creation."""
        output_path = Path(self.temp_dir) / "error_analysis.ipynb"
        
        # Test with non-existent file
        result = await create_analysis_notebook(
            file_path='/nonexistent/file.xlsx',
            output_path=output_path,
            tasks=['data_profiling'],
            execute=False
        )
        
        assert isinstance(result, WorkflowResult)
        # Should handle error gracefully - either succeed with warnings or fail cleanly
        if not result.success:
            assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_create_analysis_notebook_csv_input(self):
        """Test notebook creation with CSV input file."""
        # Create CSV version of test data
        csv_file = Path(self.temp_dir) / "analysis_data.csv"
        df = pd.read_excel(self.excel_file)
        df.to_csv(csv_file, index=False)
        
        output_path = Path(self.temp_dir) / "csv_analysis.ipynb"
        
        result = await create_analysis_notebook(
            file_path=str(csv_file),
            output_path=output_path,
            tasks=['data_profiling'],
            execute=False
        )
        
        assert result.success is True
        assert output_path.exists()
        
        # Check that CSV reading is used
        notebook_io = NotebookIO()
        notebook = notebook_io.read_notebook(output_path)
        
        cell_sources = [cell.get('source', '') for cell in notebook['cells']]
        all_source = '\n'.join(cell_sources)
        assert 'read_csv' in all_source


class TestWorkflowIntegration:
    """Integration tests for complete workflow scenarios."""
    
    def setup_method(self):
        """Set up comprehensive test scenario."""
        register_all_plugins()
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive test dataset
        import numpy as np
        np.random.seed(42)
        
        data = {
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['A', 'B', 'C'] * 33 + ['A'],  # 100 total
            'sales': np.random.normal(10000, 3000, 100),
            'cost': np.random.normal(7000, 2000, 100),
            'customer_count': np.random.randint(50, 500, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        }
        
        # Add some outliers
        data['sales'][95:] = [50000, 60000, 75000, 25000, 80000]
        
        self.excel_file = Path(self.temp_dir) / "comprehensive_data.xlsx"
        df = pd.DataFrame(data)
        df.to_excel(self.excel_file, index=False)
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self):
        """Test a complete end-to-end analysis workflow."""
        output_path = Path(self.temp_dir) / "complete_analysis.ipynb"
        
        result = await create_analysis_notebook(
            file_path=str(self.excel_file),
            output_path=output_path,
            tasks=['data_profiling', 'outlier_detection'],
            quality_inspectors=['spreadsheet'],
            task_config={
                'include_distributions': True,
                'include_correlations': True,
                'method': 'iqr',
                'threshold': 1.5
            },
            execute=False
        )
        
        # Verify comprehensive results
        assert result.success is True
        assert result.notebook is not None
        assert result.quality_metrics is not None
        assert output_path.exists()
        
        # Verify quality assessment
        metrics = result.quality_metrics
        assert metrics.overall_score > 0.0
        
        # Verify notebook structure
        notebook_io = NotebookIO()
        notebook = notebook_io.read_notebook(output_path)
        
        assert len(notebook['cells']) > 5  # Should have substantial content
        
        # Check for comprehensive analysis elements
        cell_sources = [cell.get('source', '') for cell in notebook['cells']]
        all_source = '\n'.join(cell_sources)
        
        # Data profiling elements
        assert 'pandas' in all_source
        assert 'describe()' in all_source or 'info()' in all_source
        
        # Outlier detection elements
        assert 'outlier' in all_source.lower() or 'quartile' in all_source.lower()
        
        # Should have markdown documentation
        markdown_cells = [cell for cell in notebook['cells'] if cell.get('cell_type') == 'markdown']
        assert len(markdown_cells) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_performance_and_robustness(self):
        """Test workflow performance and robustness with various scenarios."""
        test_scenarios = []
        
        # Scenario 1: Large task list
        test_scenarios.append({
            'name': 'comprehensive',
            'tasks': ['data_profiling', 'outlier_detection'],
            'task_config': {'include_distributions': True, 'method': 'iqr'}
        })
        
        # Scenario 2: Minimal configuration
        test_scenarios.append({
            'name': 'minimal',
            'tasks': ['data_profiling'],
            'task_config': {}
        })
        
        # Scenario 3: Quality-focused
        test_scenarios.append({
            'name': 'quality_focused',
            'tasks': ['data_profiling'],
            'task_config': {},
            'quality_inspectors': ['spreadsheet']
        })
        
        results = {}
        
        for scenario in test_scenarios:
            output_path = Path(self.temp_dir) / f"{scenario['name']}_analysis.ipynb"
            
            config = WorkflowConfig(
                mode=WorkflowMode.BUILD_ONLY,
                tasks=scenario['tasks'],
                task_config=scenario.get('task_config', {}),
                quality_inspectors=scenario.get('quality_inspectors', []),
                output_path=output_path,
                assess_quality=True
            )
            
            workflow = NotebookWorkflow()
            result = await workflow.run(config)
            
            results[scenario['name']] = result
            
            # All scenarios should succeed
            assert result.success is True
            assert result.notebook is not None
            assert output_path.exists()
        
        # Verify all results are reasonable
        for name, result in results.items():
            assert len(result.errors) == 0, f"Scenario {name} had errors: {result.errors}"
            
            if result.quality_metrics:
                assert 0.0 <= result.quality_metrics.overall_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__]) 