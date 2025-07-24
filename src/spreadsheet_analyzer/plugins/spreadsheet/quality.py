"""
Spreadsheet-specific quality inspection.

This module provides quality assessment tailored to spreadsheet analysis notebooks:
- Excel/CSV specific quality checks  
- Formula validation completeness
- Data profiling depth assessment
- Spreadsheet analysis best practices
- Domain-specific quality metrics
"""

from typing import Dict, List, Any
import re
from pathlib import Path

from ..base import BaseQualityInspector
from ...core_exec import NotebookBuilder, CellType
from ...core_exec.quality import QualityMetrics, QualityIssue, QualityLevel


class SpreadsheetQualityInspector(BaseQualityInspector):
    """
    Quality inspector specialized for spreadsheet analysis notebooks.
    
    Extends core quality assessment with spreadsheet-specific checks:
    - Presence of data profiling analysis
    - Formula validation completeness (for Excel files)
    - Outlier detection coverage
    - Statistical analysis depth
    - Visualization appropriateness
    """
    
    def __init__(self):
        super().__init__(
            name="spreadsheet_quality",
            description="Spreadsheet analysis quality assessment"
        )
    
    def inspect(
        self, 
        notebook: NotebookBuilder, 
        context: Dict[str, Any]
    ) -> QualityMetrics:
        """Perform spreadsheet-specific quality inspection."""
        
        # Start with core quality metrics
        from ...core_exec.quality import QualityInspector as CoreInspector
        core_inspector = CoreInspector()
        core_metrics = core_inspector.inspect(notebook)
        
        # Add spreadsheet-specific analysis
        spreadsheet_issues = []
        spreadsheet_metrics = {}
        
        # Analyze cells for spreadsheet-specific content
        analysis_coverage = self._analyze_analysis_coverage(notebook, context, spreadsheet_issues)
        code_quality = self._analyze_code_quality(notebook, spreadsheet_issues)
        file_handling = self._analyze_file_handling(notebook, context, spreadsheet_issues)
        
        # Combine metrics
        spreadsheet_metrics.update(analysis_coverage)
        spreadsheet_metrics.update(code_quality)
        spreadsheet_metrics.update(file_handling)
        
        # Adjust overall score based on spreadsheet-specific criteria
        adjusted_score = self._calculate_spreadsheet_score(
            core_metrics.overall_score,
            spreadsheet_metrics,
            len(spreadsheet_issues)
        )
        
        # Determine adjusted quality level
        if adjusted_score >= 85:
            level = QualityLevel.EXCELLENT
        elif adjusted_score >= 70:
            level = QualityLevel.GOOD
        elif adjusted_score >= 50:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR
        
        # Combine issues
        all_issues = core_metrics.issues + spreadsheet_issues
        
        # Combine metrics
        combined_metrics = {**core_metrics.metrics, **spreadsheet_metrics}
        
        return QualityMetrics(
            overall_score=adjusted_score,
            overall_level=level,
            total_cells=core_metrics.total_cells,
            code_cells=core_metrics.code_cells,
            markdown_cells=core_metrics.markdown_cells,
            empty_cells=core_metrics.empty_cells,
            cells_with_outputs=core_metrics.cells_with_outputs,
            cells_with_errors=core_metrics.cells_with_errors,
            avg_cell_length=core_metrics.avg_cell_length,
            issues=all_issues,
            metrics=combined_metrics
        )
    
    def _analyze_analysis_coverage(
        self,
        notebook: NotebookBuilder,
        context: Dict[str, Any],
        issues: List[QualityIssue]
    ) -> Dict[str, Any]:
        """Analyze coverage of spreadsheet analysis tasks."""
        
        metrics = {
            "has_data_profiling": False,
            "has_formula_analysis": False,
            "has_outlier_detection": False,
            "has_statistical_analysis": False,
            "has_visualization": False,
            "profiling_depth": "none"
        }
        
        # Analyze code cells for specific analysis patterns
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != CellType.CODE:
                continue
                
            cell_content = ''.join(cell.source).lower()
            
            # Data profiling indicators
            profiling_patterns = [
                'describe()', 'info()', 'profile', 'dtype', 'isnull()', 'nunique()',
                'missing', 'data_profiling', 'statistical_summary'
            ]
            if any(pattern in cell_content for pattern in profiling_patterns):
                metrics["has_data_profiling"] = True
                
                # Assess profiling depth
                if any(deep in cell_content for deep in ['profilereport', 'ydata_profiling', 'comprehensive']):
                    metrics["profiling_depth"] = "comprehensive"
                elif any(med in cell_content for med in ['describe', 'summary', 'info']):
                    metrics["profiling_depth"] = "basic"
            
            # Formula analysis indicators
            formula_patterns = [
                'openpyxl', 'formula', 'cell.value', '.formula', 'xlwings', 'formula_analysis'
            ]
            if any(pattern in cell_content for pattern in formula_patterns):
                metrics["has_formula_analysis"] = True
            
            # Outlier detection indicators
            outlier_patterns = [
                'outlier', 'anomaly', 'zscore', 'iqr', 'quantile', 'std()', 'boxplot'
            ]
            if any(pattern in cell_content for pattern in outlier_patterns):
                metrics["has_outlier_detection"] = True
            
            # Statistical analysis indicators
            stats_patterns = [
                'correlation', 'mean()', 'median()', 'std()', 'var()', 'skew', 'kurt'
            ]
            if any(pattern in cell_content for pattern in stats_patterns):
                metrics["has_statistical_analysis"] = True
            
            # Visualization indicators
            viz_patterns = [
                'plt.', 'matplotlib', 'seaborn', 'plot()', 'hist()', 'scatter', 'sns.'
            ]
            if any(pattern in cell_content for pattern in viz_patterns):
                metrics["has_visualization"] = True
        
        # Check for missing essential analysis
        file_path = context.get('file_path', '')
        is_excel = Path(file_path).suffix.lower() in ['.xlsx', '.xls', '.xlsm'] if file_path else False
        
        if not metrics["has_data_profiling"]:
            issues.append(QualityIssue(
                category="analysis_coverage",
                severity="warning",
                message="No data profiling analysis detected",
                suggestion="Add basic data profiling (df.describe(), df.info(), missing values analysis)"
            ))
        
        if is_excel and not metrics["has_formula_analysis"]:
            issues.append(QualityIssue(
                category="analysis_coverage",
                severity="warning",
                message="No Excel formula analysis for .xlsx/.xls file",
                suggestion="Add formula validation and error detection for Excel files"
            ))
        
        if not metrics["has_outlier_detection"]:
            issues.append(QualityIssue(
                category="analysis_coverage",
                severity="info",
                message="No outlier detection analysis",
                suggestion="Consider adding outlier detection (IQR method, Z-score analysis)"
            ))
        
        if not metrics["has_visualization"]:
            issues.append(QualityIssue(
                category="analysis_coverage",
                severity="info",
                message="No data visualizations detected",
                suggestion="Add charts and plots to better understand data patterns"
            ))
        
        return metrics
    
    def _analyze_code_quality(
        self,
        notebook: NotebookBuilder,
        issues: List[QualityIssue]
    ) -> Dict[str, Any]:
        """Analyze spreadsheet-specific code quality."""
        
        metrics = {
            "has_error_handling": False,
            "has_data_validation": False,
            "uses_best_practices": False,
            "has_efficient_loading": False
        }
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != CellType.CODE:
                continue
                
            cell_content = ''.join(cell.source)
            
            # Error handling patterns
            if any(pattern in cell_content for pattern in ['try:', 'except:', 'finally:']):
                metrics["has_error_handling"] = True
            
            # Data validation patterns
            validation_patterns = [
                'assert ', 'validate', 'check', 'empty', 'isnull', 'dtype'
            ]
            if any(pattern in cell_content for pattern in validation_patterns):
                metrics["has_data_validation"] = True
            
            # Best practices
            best_practices = [
                'warnings.filterwarnings', 'pd.set_option', 'plt.style.use'
            ]
            if any(pattern in cell_content for pattern in best_practices):
                metrics["uses_best_practices"] = True
            
            # Efficient loading patterns
            efficient_patterns = [
                'usecols=', 'nrows=', 'chunksize=', 'dtype='
            ]
            if any(pattern in cell_content for pattern in efficient_patterns):
                metrics["has_efficient_loading"] = True
            
            # Check for problematic patterns
            if 'pd.read_excel' in cell_content and 'sheet_name=None' in cell_content:
                issues.append(QualityIssue(
                    category="code_quality",
                    severity="warning",
                    message="Loading all Excel sheets may be inefficient",
                    cell_index=i,
                    suggestion="Consider loading specific sheets only"
                ))
            
            # Check for hardcoded paths
            hardcoded_path_patterns = [r'[C-Z]:\\', r'/Users/', r'/home/']
            if any(re.search(pattern, cell_content) for pattern in hardcoded_path_patterns):
                issues.append(QualityIssue(
                    category="code_quality",
                    severity="warning",
                    message="Hardcoded file paths detected",
                    cell_index=i,
                    suggestion="Use relative paths or path variables for better portability"
                ))
        
        # Add suggestions for missing elements
        if not metrics["has_error_handling"]:
            issues.append(QualityIssue(
                category="code_quality",
                severity="info",
                message="No error handling detected",
                suggestion="Add try-except blocks around file loading and data operations"
            ))
        
        return metrics
    
    def _analyze_file_handling(
        self,
        notebook: NotebookBuilder,
        context: Dict[str, Any],
        issues: List[QualityIssue]
    ) -> Dict[str, Any]:
        """Analyze file handling patterns."""
        
        metrics = {
            "handles_file_paths": False,
            "validates_file_existence": False,
            "handles_multiple_sheets": False,
            "uses_appropriate_loader": False
        }
        
        file_path = context.get('file_path', '')
        file_ext = Path(file_path).suffix.lower() if file_path else ''
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != CellType.CODE:
                continue
                
            cell_content = ''.join(cell.source)
            
            # File path handling
            if any(pattern in cell_content for pattern in ['Path(', 'os.path', 'pathlib']):
                metrics["handles_file_paths"] = True
            
            # File existence validation
            if any(pattern in cell_content for pattern in ['.exists()', 'os.path.exists', 'isfile']):
                metrics["validates_file_existence"] = True
            
            # Multiple sheets handling
            if any(pattern in cell_content for pattern in ['sheet_name', 'sheet_names', 'ExcelFile']):
                metrics["handles_multiple_sheets"] = True
            
            # Appropriate loader usage
            if file_ext in ['.xlsx', '.xls', '.xlsm'] and 'pd.read_excel' in cell_content:
                metrics["uses_appropriate_loader"] = True
            elif file_ext == '.csv' and 'pd.read_csv' in cell_content:
                metrics["uses_appropriate_loader"] = True
        
        # File-specific suggestions
        if file_ext in ['.xlsx', '.xls', '.xlsm'] and not metrics["handles_multiple_sheets"]:
            issues.append(QualityIssue(
                category="file_handling",
                severity="info",
                message="Excel file may have multiple sheets - consider sheet handling",
                suggestion="Check for multiple sheets and handle them appropriately"
            ))
        
        return metrics
    
    def _calculate_spreadsheet_score(
        self,
        base_score: float,
        metrics: Dict[str, Any],
        num_issues: int
    ) -> float:
        """Calculate adjusted score based on spreadsheet-specific criteria."""
        
        score = base_score
        
        # Bonus for comprehensive analysis
        analysis_bonus = 0
        if metrics.get("has_data_profiling"):
            analysis_bonus += 5
        if metrics.get("has_formula_analysis"):
            analysis_bonus += 5
        if metrics.get("has_outlier_detection"):
            analysis_bonus += 3
        if metrics.get("has_statistical_analysis"):
            analysis_bonus += 3
        if metrics.get("has_visualization"):
            analysis_bonus += 4
        
        # Bonus for code quality
        quality_bonus = 0
        if metrics.get("has_error_handling"):
            quality_bonus += 3
        if metrics.get("has_data_validation"):
            quality_bonus += 2
        if metrics.get("uses_best_practices"):
            quality_bonus += 2
        
        # Profiling depth bonus
        profiling_depth = metrics.get("profiling_depth", "none")
        if profiling_depth == "comprehensive":
            analysis_bonus += 5
        elif profiling_depth == "basic":
            analysis_bonus += 2
        
        # Apply bonuses
        score += analysis_bonus + quality_bonus
        
        # Cap at 100
        return min(100.0, max(0.0, score)) 