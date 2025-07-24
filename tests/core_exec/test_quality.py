"""
Tests for the generic quality inspection module.

This test suite validates the quality assessment functionality including:
- QualityLevel enumeration
- QualityIssue data structure
- QualityMetrics aggregation
- QualityInspector base functionality
- Generic notebook quality assessment
- Quality report generation

Following TDD principles with functional tests - no mocking used.
All tests use real NotebookBuilder instances for authentic quality assessment.
"""

from typing import Dict, List

import pytest

from spreadsheet_analyzer.core_exec.quality import (
    QualityLevel,
    QualityIssue,
    QualityMetrics,
    QualityInspector,
)
from spreadsheet_analyzer.core_exec.notebook_builder import (
    NotebookBuilder,
    CellType,
)


class TestQualityLevel:
    """Test QualityLevel enumeration."""

    def test_quality_level_values(self) -> None:
        """Test that QualityLevel has correct enumeration values."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.FAIR.value == "fair"
        assert QualityLevel.POOR.value == "poor"

    def test_quality_level_iteration(self) -> None:
        """Test that all quality levels can be iterated."""
        levels = list(QualityLevel)
        assert len(levels) == 4
        assert QualityLevel.EXCELLENT in levels
        assert QualityLevel.GOOD in levels
        assert QualityLevel.FAIR in levels
        assert QualityLevel.POOR in levels

    def test_quality_level_ordering(self) -> None:
        """Test that quality levels can be compared."""
        # Define ordering from best to worst
        ordered_levels = [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
            QualityLevel.FAIR,
            QualityLevel.POOR
        ]
        
        # Test that we can compare them semantically
        assert QualityLevel.EXCELLENT.value != QualityLevel.POOR.value
        assert len(set(level.value for level in ordered_levels)) == 4


class TestQualityIssue:
    """Test QualityIssue data structure."""

    def test_quality_issue_creation(self) -> None:
        """Test creating a QualityIssue with all fields."""
        issue = QualityIssue(
            category="structure",
            severity=QualityLevel.POOR,
            message="Missing markdown cells for documentation",
            cell_index=None,
            details={"expected_markdown_ratio": 0.3, "actual_ratio": 0.1}
        )
        
        assert issue.category == "structure"
        assert issue.severity == QualityLevel.POOR
        assert issue.message == "Missing markdown cells for documentation"
        assert issue.cell_index is None
        assert issue.details["expected_markdown_ratio"] == 0.3

    def test_quality_issue_with_cell_index(self) -> None:
        """Test creating a QualityIssue with specific cell reference."""
        issue = QualityIssue(
            category="code_quality",
            severity=QualityLevel.FAIR,
            message="Code cell lacks proper documentation",
            cell_index=5,
            details={"code_length": 150, "comment_ratio": 0.05}
        )
        
        assert issue.category == "code_quality"
        assert issue.severity == QualityLevel.FAIR
        assert issue.cell_index == 5
        assert issue.details["code_length"] == 150

    def test_quality_issue_minimal(self) -> None:
        """Test creating a QualityIssue with minimal required fields."""
        issue = QualityIssue(
            category="content",
            severity=QualityLevel.GOOD,
            message="Content is well structured"
        )
        
        assert issue.category == "content"
        assert issue.severity == QualityLevel.GOOD
        assert issue.message == "Content is well structured"
        assert issue.cell_index is None
        assert issue.details == {}


class TestQualityMetrics:
    """Test QualityMetrics aggregation."""

    def test_quality_metrics_creation(self) -> None:
        """Test creating QualityMetrics with all fields."""
        issues = [
            QualityIssue("structure", QualityLevel.POOR, "Missing documentation"),
            QualityIssue("code_quality", QualityLevel.FAIR, "Long function")
        ]
        
        metrics = QualityMetrics(
            overall_score=72.5,
            overall_level=QualityLevel.GOOD,
            total_issues=2,
            issues_by_severity={
                QualityLevel.POOR: 1,
                QualityLevel.FAIR: 1,
                QualityLevel.GOOD: 0,
                QualityLevel.EXCELLENT: 0
            },
            category_scores={
                "structure": 60.0,
                "code_quality": 75.0,
                "content": 85.0
            },
            issues=issues,
            recommendations=[
                "Add more markdown cells for documentation",
                "Break down long code cells into smaller units"
            ]
        )
        
        assert metrics.overall_score == 72.5
        assert metrics.overall_level == QualityLevel.GOOD
        assert metrics.total_issues == 2
        assert metrics.issues_by_severity[QualityLevel.POOR] == 1
        assert len(metrics.issues) == 2
        assert len(metrics.recommendations) == 2

    def test_quality_metrics_defaults(self) -> None:
        """Test QualityMetrics with default values."""
        metrics = QualityMetrics()
        
        assert metrics.overall_score == 0.0
        assert metrics.overall_level == QualityLevel.POOR
        assert metrics.total_issues == 0
        assert metrics.issues_by_severity == {}
        assert metrics.category_scores == {}
        assert metrics.issues == []
        assert metrics.recommendations == []

    def test_quality_metrics_calculation_helpers(self) -> None:
        """Test helper methods for quality metrics calculation."""
        issues = [
            QualityIssue("structure", QualityLevel.POOR, "Issue 1"),
            QualityIssue("structure", QualityLevel.FAIR, "Issue 2"),
            QualityIssue("code_quality", QualityLevel.GOOD, "Issue 3"),
            QualityIssue("content", QualityLevel.POOR, "Issue 4")
        ]
        
        metrics = QualityMetrics(
            issues=issues,
            total_issues=len(issues)
        )
        
        # Test severity counting
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        assert severity_counts[QualityLevel.POOR] == 2
        assert severity_counts[QualityLevel.FAIR] == 1
        assert severity_counts[QualityLevel.GOOD] == 1
        
        # Test category grouping
        category_issues = {}
        for issue in issues:
            if issue.category not in category_issues:
                category_issues[issue.category] = []
            category_issues[issue.category].append(issue)
        
        assert len(category_issues["structure"]) == 2
        assert len(category_issues["code_quality"]) == 1
        assert len(category_issues["content"]) == 1


class TestQualityInspector:
    """Test QualityInspector base functionality."""

    def test_inspector_empty_notebook(self) -> None:
        """Test quality inspection of an empty notebook."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        metrics = inspector.inspect(notebook)
        
        # Empty notebook should have issues
        assert metrics.overall_level in [QualityLevel.POOR, QualityLevel.FAIR]
        assert metrics.overall_score < 50.0
        assert metrics.total_issues > 0
        
        # Should identify lack of content
        issue_messages = [issue.message for issue in metrics.issues]
        assert any("empty" in msg.lower() or "no cells" in msg.lower() for msg in issue_messages)

    def test_inspector_markdown_only_notebook(self) -> None:
        """Test quality inspection of notebook with only markdown cells."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Add markdown cells
        notebook.add_markdown_cell("# Introduction")
        notebook.add_markdown_cell("This notebook contains documentation.")
        notebook.add_markdown_cell("## Conclusion")
        
        metrics = inspector.inspect(notebook)
        
        # Should identify lack of code
        assert metrics.total_issues > 0
        issue_messages = [issue.message for issue in metrics.issues]
        assert any("code" in msg.lower() for msg in issue_messages)

    def test_inspector_code_only_notebook(self) -> None:
        """Test quality inspection of notebook with only code cells."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Add code cells
        notebook.add_code_cell("import pandas as pd")
        notebook.add_code_cell("df = pd.DataFrame({'x': [1, 2, 3]})")
        notebook.add_code_cell("print(df.head())")
        
        metrics = inspector.inspect(notebook)
        
        # Should identify lack of documentation
        assert metrics.total_issues > 0
        issue_categories = [issue.category for issue in metrics.issues]
        assert "documentation" in issue_categories or "structure" in issue_categories

    def test_inspector_balanced_notebook(self) -> None:
        """Test quality inspection of well-balanced notebook."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Create balanced notebook
        notebook.add_markdown_cell("# Data Analysis Notebook")
        notebook.add_markdown_cell("This notebook analyzes sample data.")
        notebook.add_code_cell("import pandas as pd\nimport numpy as np")
        notebook.add_markdown_cell("## Load Data")
        notebook.add_code_cell("# Load the dataset\ndf = pd.read_csv('data.csv')")
        notebook.add_markdown_cell("## Analysis")
        notebook.add_code_cell("# Calculate summary statistics\nsummary = df.describe()")
        notebook.add_code_cell("print(summary)")
        notebook.add_markdown_cell("## Conclusion")
        notebook.add_markdown_cell("The analysis shows interesting patterns.")
        
        metrics = inspector.inspect(notebook)
        
        # Should have better quality score
        assert metrics.overall_score > 50.0
        assert metrics.overall_level in [QualityLevel.GOOD, QualityLevel.FAIR]

    def test_inspector_large_notebook(self) -> None:
        """Test quality inspection of large notebook."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Create large notebook
        notebook.add_markdown_cell("# Large Analysis Notebook")
        
        for i in range(50):
            if i % 3 == 0:
                notebook.add_markdown_cell(f"## Section {i//3 + 1}")
            else:
                notebook.add_code_cell(f"# Analysis step {i}\nresult_{i} = {i} * 2")
        
        metrics = inspector.inspect(notebook)
        
        # Should handle large notebooks
        assert metrics.total_issues >= 0
        assert metrics.overall_score >= 0.0
        assert isinstance(metrics.overall_level, QualityLevel)

    def test_inspector_notebook_with_outputs(self) -> None:
        """Test quality inspection of notebook with cell outputs."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Add cells with outputs
        notebook.add_markdown_cell("# Analysis with Results")
        
        outputs = [{"output_type": "stream", "text": "Hello, World!\n"}]
        notebook.add_code_cell("print('Hello, World!')", outputs=outputs)
        
        execute_result = {"output_type": "execute_result", "data": {"text/plain": "42"}}
        notebook.add_code_cell("2 + 2", outputs=[execute_result])
        
        metrics = inspector.inspect(notebook)
        
        # Should recognize executed cells as positive
        assert metrics.overall_score > 30.0  # Better than empty notebook

    def test_inspector_notebook_with_metadata(self) -> None:
        """Test quality inspection considering cell metadata."""
        inspector = QualityInspector()
        notebook = NotebookBuilder()
        
        # Add cells with rich metadata
        header_meta = {"tags": ["header", "introduction"]}
        notebook.add_markdown_cell("# Well-Documented Analysis", header_meta)
        
        code_meta = {"tags": ["data-loading"], "collapsed": False}
        notebook.add_code_cell("import pandas as pd", metadata=code_meta)
        
        analysis_meta = {"tags": ["analysis", "statistics"]}
        notebook.add_code_cell("df.describe()", metadata=analysis_meta)
        
        metrics = inspector.inspect(notebook)
        
        # Well-tagged notebook should score better
        assert metrics.overall_score > 40.0

    def test_inspector_quality_categories(self) -> None:
        """Test that inspector identifies different quality categories."""
        inspector = QualityInspector()
        
        # Create notebook with various quality issues
        notebook = NotebookBuilder()
        notebook.add_code_cell("x=1")  # Poor formatting
        notebook.add_code_cell("y=2")  # No documentation
        notebook.add_code_cell("print(x+y)")  # Minimal content
        
        metrics = inspector.inspect(notebook)
        
        # Should have multiple categories of issues
        categories = set(issue.category for issue in metrics.issues)
        assert len(categories) > 1
        
        # Common categories should be present
        expected_categories = {"documentation", "structure", "code_quality", "content"}
        assert len(categories.intersection(expected_categories)) > 0

    def test_inspector_recommendations(self) -> None:
        """Test that inspector provides actionable recommendations."""
        inspector = QualityInspector()
        
        # Create problematic notebook
        notebook = NotebookBuilder()
        notebook.add_code_cell("import pandas")
        notebook.add_code_cell("df = pandas.DataFrame()")
        notebook.add_code_cell("print(df)")
        
        metrics = inspector.inspect(notebook)
        
        # Should provide recommendations
        assert len(metrics.recommendations) > 0
        
        # Recommendations should be strings
        for recommendation in metrics.recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 10  # Should be meaningful

    def test_inspector_severity_distribution(self) -> None:
        """Test that inspector properly distributes issue severities."""
        inspector = QualityInspector()
        
        # Create notebook with mixed quality aspects
        notebook = NotebookBuilder()
        notebook.add_markdown_cell("# Good Documentation")  # Good aspect
        notebook.add_code_cell("x = 1")  # Poor: no documentation
        notebook.add_code_cell("# Better documented\ny = x * 2")  # Fair: some docs
        notebook.add_markdown_cell("## Results")  # Good aspect
        
        metrics = inspector.inspect(notebook)
        
        # Should have varied severity levels
        severity_levels = set(issue.severity for issue in metrics.issues)
        assert len(severity_levels) > 1

    def test_inspector_score_calculation(self) -> None:
        """Test that quality scores are calculated consistently."""
        inspector = QualityInspector()
        
        # Test multiple notebooks with different quality levels
        notebooks = []
        
        # Poor quality notebook
        poor_notebook = NotebookBuilder()
        poor_notebook.add_code_cell("x=1")
        notebooks.append(poor_notebook)
        
        # Better quality notebook
        good_notebook = NotebookBuilder()
        good_notebook.add_markdown_cell("# Analysis")
        good_notebook.add_code_cell("# Load data\nimport pandas as pd")
        good_notebook.add_markdown_cell("## Results")
        good_notebook.add_code_cell("# Analyze data\nresult = pd.DataFrame()")
        notebooks.append(good_notebook)
        
        # Compare scores
        metrics_list = [inspector.inspect(nb) for nb in notebooks]
        
        # Better notebook should have higher score
        assert metrics_list[1].overall_score > metrics_list[0].overall_score

    def test_inspector_edge_cases(self) -> None:
        """Test inspector behavior with edge cases."""
        inspector = QualityInspector()
        
        # Test with very long cell content
        long_notebook = NotebookBuilder()
        long_code = "x = 1\n" * 1000  # Very long cell
        long_notebook.add_code_cell(long_code)
        
        metrics = inspector.inspect(long_notebook)
        assert metrics.total_issues >= 0  # Should handle gracefully
        
        # Test with special characters
        special_notebook = NotebookBuilder()
        special_notebook.add_markdown_cell("# Spécial Çharacters: éñ中文🚀")
        special_notebook.add_code_cell("print('Unicode: αβγδε')")
        
        metrics = inspector.inspect(special_notebook)
        assert metrics.total_issues >= 0  # Should handle gracefully

    def test_inspector_consistency(self) -> None:
        """Test that inspector produces consistent results."""
        inspector = QualityInspector()
        
        # Create notebook
        notebook = NotebookBuilder()
        notebook.add_markdown_cell("# Test Notebook")
        notebook.add_code_cell("import pandas as pd")
        notebook.add_code_cell("df = pd.DataFrame({'x': [1, 2, 3]})")
        
        # Inspect multiple times
        metrics1 = inspector.inspect(notebook)
        metrics2 = inspector.inspect(notebook)
        metrics3 = inspector.inspect(notebook)
        
        # Results should be consistent
        assert metrics1.overall_score == metrics2.overall_score == metrics3.overall_score
        assert metrics1.total_issues == metrics2.total_issues == metrics3.total_issues
        assert metrics1.overall_level == metrics2.overall_level == metrics3.overall_level

    def test_inspector_comprehensive_analysis(self) -> None:
        """Test comprehensive quality analysis of complex notebook."""
        inspector = QualityInspector()
        
        # Create comprehensive notebook
        notebook = NotebookBuilder()
        
        # Title and introduction
        notebook.add_markdown_cell("# Comprehensive Data Analysis")
        notebook.add_markdown_cell("""
This notebook performs a comprehensive analysis of sales data.

## Objectives
- Load and clean the data
- Perform exploratory data analysis
- Generate insights and recommendations
        """)
        
        # Setup and imports
        notebook.add_markdown_cell("## Setup")
        notebook.add_code_cell("""
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display options
pd.set_option('display.max_columns', None)
        """)
        
        # Data loading
        notebook.add_markdown_cell("## Data Loading")
        notebook.add_code_cell("""
# Load the sales data
df = pd.read_csv('sales_data.csv')
print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
        """)
        
        # Data exploration
        notebook.add_markdown_cell("## Data Exploration")
        notebook.add_code_cell("""
# Check basic information
print('Data Info:')
print(df.info())
print('\\nSummary Statistics:')
print(df.describe())
        """)
        
        # Analysis
        notebook.add_markdown_cell("## Analysis")
        notebook.add_code_cell("""
# Calculate key metrics
total_sales = df['sales'].sum()
avg_sales = df['sales'].mean()
top_products = df.groupby('product')['sales'].sum().sort_values(ascending=False).head(10)

print(f'Total Sales: ${total_sales:,.2f}')
print(f'Average Sales: ${avg_sales:,.2f}')
print('\\nTop 10 Products:')
print(top_products)
        """)
        
        # Conclusions
        notebook.add_markdown_cell("""
## Conclusions

The analysis reveals several key insights:

1. Total sales performance meets expectations
2. Top products drive majority of revenue
3. Seasonal patterns are evident in the data

## Recommendations

Based on the analysis, we recommend:
- Focus marketing on top-performing products
- Develop strategies for seasonal optimization
- Continue monitoring key performance metrics
        """)
        
        # Inspect this comprehensive notebook
        metrics = inspector.inspect(notebook)
        
        # Should receive high quality score
        assert metrics.overall_score > 60.0
        assert metrics.overall_level in [QualityLevel.GOOD, QualityLevel.EXCELLENT]
        
        # Should have balanced content
        markdown_cells = len([cell for cell in notebook.cells if cell.cell_type == CellType.MARKDOWN])
        code_cells = len([cell for cell in notebook.cells if cell.cell_type == CellType.CODE])
        
        assert markdown_cells >= 5  # Good documentation
        assert code_cells >= 4      # Substantial analysis
        
        # Should have fewer issues than poor notebooks
        assert metrics.total_issues < 10 