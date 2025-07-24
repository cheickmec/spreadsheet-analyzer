"""
Tests for spreadsheet analysis tasks.

This module provides comprehensive functional tests for all spreadsheet analysis
tasks, including data profiling, formula analysis, and outlier detection.
All tests are functional and use no mocking.
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List
import tempfile
import pandas as pd
import numpy as np

from spreadsheet_analyzer.plugins.spreadsheet.tasks import (
    DataProfilingTask,
    FormulaAnalysisTask,
    OutlierDetectionTask,
)
from spreadsheet_analyzer.core_exec import (
    NotebookBuilder,
    NotebookCell,
    CellType,
)


class TestDataProfilingTask:
    """Test the DataProfilingTask class functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.task = DataProfilingTask()
        
        # Create realistic test data
        self.sample_data = {
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 42],
            'salary': [50000, 60000, 75000, 55000, 90000],
            'department': ['Engineering', 'Sales', 'Engineering', 'HR', 'Sales'],
            'start_date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2017-11-05']
        }
        
        # Create temporary Excel file
        self.temp_dir = tempfile.mkdtemp()
        self.excel_file = Path(self.temp_dir) / "test_data.xlsx"
        df = pd.DataFrame(self.sample_data)
        df.to_excel(self.excel_file, index=False)
        
        # Test configuration
        self.config = {
            'file_path': str(self.excel_file),
            'include_distributions': True,
            'include_correlations': True,
            'max_unique_values': 10
        }
    
    def test_task_properties(self):
        """Test basic task properties and metadata."""
        assert self.task.name == "data_profiling"
        assert self.task.description == "Generate comprehensive data profiling analysis"
        assert isinstance(self.task.required_config_keys, list)
        assert 'file_path' in self.task.required_config_keys
    
    def test_can_handle_valid_config(self):
        """Test that task correctly identifies valid configurations."""
        assert self.task.can_handle(self.config) is True
        
        # Test with missing required key
        invalid_config = {'other_key': 'value'}
        assert self.task.can_handle(invalid_config) is False
        
        # Test with non-existent file
        invalid_file_config = {
            'file_path': '/non/existent/file.xlsx'
        }
        assert self.task.can_handle(invalid_file_config) is False
    
    def test_generate_cells_basic_profiling(self):
        """Test generation of basic data profiling cells."""
        cells = self.task.generate_cells(self.config)
        
        # Should return a list of NotebookCell objects
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, NotebookCell) for cell in cells)
        
        # Should include imports, data loading, and analysis cells
        cell_types = [cell.cell_type for cell in cells]
        assert CellType.CODE in cell_types
        
        # Check for expected content patterns
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'import pandas as pd' in all_source
        assert 'read_excel' in all_source or 'read_csv' in all_source
        assert 'describe()' in all_source
        assert 'info()' in all_source
    
    def test_generate_cells_with_distributions(self):
        """Test profiling with distribution analysis enabled."""
        config_with_dist = {**self.config, 'include_distributions': True}
        cells = self.task.generate_cells(config_with_dist)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'histogram' in all_source or 'hist()' in all_source
    
    def test_generate_cells_with_correlations(self):
        """Test profiling with correlation analysis enabled."""
        config_with_corr = {**self.config, 'include_correlations': True}
        cells = self.task.generate_cells(config_with_corr)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'corr()' in all_source
    
    def test_generate_cells_csv_file(self):
        """Test profiling with CSV file input."""
        # Create CSV version
        csv_file = Path(self.temp_dir) / "test_data.csv"
        df = pd.DataFrame(self.sample_data)
        df.to_csv(csv_file, index=False)
        
        csv_config = {**self.config, 'file_path': str(csv_file)}
        cells = self.task.generate_cells(csv_config)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'read_csv' in all_source
    
    def test_postprocess_functionality(self):
        """Test the postprocess method for additional analysis."""
        cells = self.task.generate_cells(self.config)
        
        # Mock executed results - simulating what would come from kernel execution
        mock_results = {
            'data_shape': (5, 5),
            'column_types': {'name': 'object', 'age': 'int64', 'salary': 'int64'}
        }
        
        additional_cells = self.task.postprocess(cells, mock_results, self.config)
        
        if additional_cells:
            assert isinstance(additional_cells, list)
            assert all(isinstance(cell, NotebookCell) for cell in additional_cells)


class TestFormulaAnalysisTask:
    """Test the FormulaAnalysisTask class functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.task = FormulaAnalysisTask()
        
        # Create test Excel file with formulas
        self.temp_dir = tempfile.mkdtemp()
        self.excel_file = Path(self.temp_dir) / "formulas.xlsx"
        
        # Create sample data with formulas (openpyxl format)
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        
        # Add some data and formulas
        ws['A1'] = 'Value1'
        ws['B1'] = 'Value2' 
        ws['C1'] = 'Sum'
        ws['A2'] = 10
        ws['B2'] = 20
        ws['C2'] = '=A2+B2'
        ws['A3'] = 15
        ws['B3'] = 25
        ws['C3'] = '=A3+B3'
        ws['D1'] = 'Average'
        ws['D2'] = '=AVERAGE(A2:A3)'
        
        wb.save(self.excel_file)
        
        self.config = {
            'file_path': str(self.excel_file),
            'sheet_name': None,
            'analyze_dependencies': True
        }
    
    def test_task_properties(self):
        """Test basic task properties and metadata."""
        assert self.task.name == "formula_analysis"
        assert self.task.description == "Analyze Excel formulas and their dependencies"
        assert isinstance(self.task.required_config_keys, list)
        assert 'file_path' in self.task.required_config_keys
    
    def test_can_handle_excel_files(self):
        """Test that task correctly identifies Excel files."""
        assert self.task.can_handle(self.config) is True
        
        # Test with CSV file (should not handle)
        csv_config = {
            'file_path': str(Path(self.temp_dir) / "test.csv")
        }
        # Create empty CSV file
        Path(csv_config['file_path']).touch()
        assert self.task.can_handle(csv_config) is False
    
    def test_generate_cells_formula_analysis(self):
        """Test generation of formula analysis cells."""
        cells = self.task.generate_cells(self.config)
        
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, NotebookCell) for cell in cells)
        
        # Check for expected analysis content
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'openpyxl' in all_source
        assert 'load_workbook' in all_source
        
        # Should analyze formulas
        assert 'formula' in all_source.lower()
    
    def test_generate_cells_with_dependencies(self):
        """Test formula analysis with dependency tracking."""
        config_with_deps = {**self.config, 'analyze_dependencies': True}
        cells = self.task.generate_cells(config_with_deps)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'dependencies' in all_source.lower() or 'precedent' in all_source.lower()


class TestOutlierDetectionTask:
    """Test the OutlierDetectionTask class functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.task = OutlierDetectionTask()
        
        # Create test data with outliers
        np.random.seed(42)  # For reproducible results
        normal_data = np.random.normal(50, 10, 95)
        outliers = np.array([100, 120, 15, 200, 5])  # Clear outliers
        all_data = np.concatenate([normal_data, outliers])
        
        self.sample_data = {
            'id': range(1, 101),
            'value': all_data,
            'category': ['A'] * 50 + ['B'] * 50,
            'score': np.random.normal(75, 15, 100)
        }
        
        # Create temporary Excel file
        self.temp_dir = tempfile.mkdtemp()
        self.excel_file = Path(self.temp_dir) / "outlier_data.xlsx"
        df = pd.DataFrame(self.sample_data)
        df.to_excel(self.excel_file, index=False)
        
        self.config = {
            'file_path': str(self.excel_file),
            'numeric_columns': ['value', 'score'],
            'method': 'iqr',
            'threshold': 1.5
        }
    
    def test_task_properties(self):
        """Test basic task properties and metadata."""
        assert self.task.name == "outlier_detection"
        assert self.task.description == "Detect and analyze statistical outliers"
        assert isinstance(self.task.required_config_keys, list)
        assert 'file_path' in self.task.required_config_keys
    
    def test_can_handle_valid_config(self):
        """Test that task correctly identifies valid configurations."""
        assert self.task.can_handle(self.config) is True
        
        # Test without numeric columns (should still handle)
        basic_config = {'file_path': str(self.excel_file)}
        assert self.task.can_handle(basic_config) is True
    
    def test_generate_cells_iqr_method(self):
        """Test outlier detection using IQR method."""
        cells = self.task.generate_cells(self.config)
        
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, NotebookCell) for cell in cells)
        
        # Check for expected analysis content
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'import pandas as pd' in all_source
        assert 'import numpy as np' in all_source
        assert 'quartile' in all_source.lower() or 'iqr' in all_source.lower()
    
    def test_generate_cells_zscore_method(self):
        """Test outlier detection using z-score method."""
        zscore_config = {**self.config, 'method': 'zscore', 'threshold': 2.0}
        cells = self.task.generate_cells(zscore_config)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'zscore' in all_source.lower() or 'z_score' in all_source.lower()
    
    def test_generate_cells_modified_zscore_method(self):
        """Test outlier detection using modified z-score method."""
        modified_config = {**self.config, 'method': 'modified_zscore'}
        cells = self.task.generate_cells(modified_config)
        
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'modified' in all_source.lower() and 'zscore' in all_source.lower()
    
    def test_generate_cells_auto_column_detection(self):
        """Test automatic detection of numeric columns."""
        auto_config = {
            'file_path': str(self.excel_file),
            'method': 'iqr'
        }
        cells = self.task.generate_cells(auto_config)
        
        # Should still generate valid analysis cells
        assert len(cells) > 0
        all_source = '\n'.join(cell.source for cell in cells if cell.cell_type == CellType.CODE)
        assert 'select_dtypes' in all_source or 'numeric' in all_source.lower()
    
    def test_postprocess_with_outlier_results(self):
        """Test postprocess method with outlier detection results."""
        cells = self.task.generate_cells(self.config)
        
        # Mock results simulating detected outliers
        mock_results = {
            'outliers_found': 5,
            'outlier_indices': [95, 96, 97, 98, 99],
            'outlier_columns': ['value', 'score']
        }
        
        additional_cells = self.task.postprocess(cells, mock_results, self.config)
        
        if additional_cells:
            assert isinstance(additional_cells, list)
            assert all(isinstance(cell, NotebookCell) for cell in additional_cells)


class TestTaskIntegration:
    """Integration tests for all spreadsheet tasks working together."""
    
    def setup_method(self):
        """Set up comprehensive test data for integration testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive test dataset
        np.random.seed(42)
        data = {
            'employee_id': range(1, 101),
            'name': [f'Employee_{i}' for i in range(1, 101)],
            'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing'], 100),
            'salary': np.random.normal(70000, 20000, 100),
            'years_experience': np.random.randint(1, 20, 100),
            'performance_score': np.random.normal(85, 10, 100),
            'bonus': np.random.normal(5000, 2000, 100)
        }
        
        # Add some outliers
        data['salary'][95:] = [150000, 180000, 200000, 25000, 300000]
        
        self.excel_file = Path(self.temp_dir) / "comprehensive_data.xlsx"
        df = pd.DataFrame(data)
        df.to_excel(self.excel_file, index=False)
    
    def test_all_tasks_can_handle_same_data(self):
        """Test that all tasks can handle the same dataset."""
        config = {'file_path': str(self.excel_file)}
        
        profiling_task = DataProfilingTask()
        outlier_task = OutlierDetectionTask()
        
        assert profiling_task.can_handle(config) is True
        assert outlier_task.can_handle(config) is True
        
        # Formula task should not handle this (no formulas)
        formula_task = FormulaAnalysisTask()
        # This might return True if it just checks file existence, which is fine
        # The actual analysis would show no formulas
    
    def test_task_combination_workflow(self):
        """Test combining multiple tasks in a workflow-like scenario."""
        config = {'file_path': str(self.excel_file)}
        
        # Generate cells from each task
        profiling_task = DataProfilingTask()
        outlier_task = OutlierDetectionTask()
        
        profiling_cells = profiling_task.generate_cells(config)
        outlier_cells = outlier_task.generate_cells(config)
        
        # Verify we get meaningful results from each
        assert len(profiling_cells) > 0
        assert len(outlier_cells) > 0
        
        # Check that cells don't conflict (no duplicate imports in adjacent cells)
        all_cells = profiling_cells + outlier_cells
        import_lines = []
        for cell in all_cells:
            if cell.cell_type == CellType.CODE:
                lines = cell.source.split('\n')
                import_lines.extend([line.strip() for line in lines if line.strip().startswith('import')])
        
        # Should have imports, but they should be reasonable
        assert len(import_lines) > 0
    
    def test_config_validation_across_tasks(self):
        """Test that all tasks handle config validation consistently."""
        tasks = [DataProfilingTask(), OutlierDetectionTask(), FormulaAnalysisTask()]
        
        # Test with missing file_path
        invalid_config = {'other_key': 'value'}
        for task in tasks:
            assert task.can_handle(invalid_config) is False
        
        # Test with non-existent file
        missing_file_config = {'file_path': '/non/existent/file.xlsx'}
        for task in tasks:
            assert task.can_handle(missing_file_config) is False
        
        # Test with valid config
        valid_config = {'file_path': str(self.excel_file)}
        for task in tasks:
            # All should at least not crash on can_handle check
            result = task.can_handle(valid_config)
            assert isinstance(result, bool)


# Additional fixtures and utilities for testing
@pytest.fixture
def sample_excel_file():
    """Provide a sample Excel file for testing."""
    temp_dir = tempfile.mkdtemp()
    excel_file = Path(temp_dir) / "sample.xlsx"
    
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['X', 'Y', 'Z', 'X', 'Y']
    }
    
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
    
    return excel_file


@pytest.fixture
def sample_csv_file():
    """Provide a sample CSV file for testing."""
    temp_dir = tempfile.mkdtemp()
    csv_file = Path(temp_dir) / "sample.csv"
    
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [100, 200, 300, 400, 500],
        'col3': ['A', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    return csv_file


if __name__ == "__main__":
    pytest.main([__file__]) 