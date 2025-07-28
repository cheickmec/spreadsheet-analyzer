"""
Generic notebook quality inspection.

This module provides domain-agnostic quality metrics for notebooks:
- Structure validation and completeness checks
- Code quality metrics (basic)
- Output validation and format checking
- Generic best practice checks

No domain-specific logic - pure notebook quality primitives.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .notebook_builder import NotebookBuilder


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityIssue:
    """
    Represents a quality issue found in a notebook.

    Args:
        category: Category of the issue (structure, content, output, etc.)
        severity: How severe the issue is (error, warning, info)
        message: Human-readable description of the issue
        cell_index: Index of the cell with the issue (if applicable)
        suggestion: Optional suggestion for fixing the issue
    """

    category: str
    severity: str
    message: str
    cell_index: int | None = None
    suggestion: str | None = None


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for a notebook.

    Args:
        overall_score: Overall quality score (0-100)
        overall_level: Overall quality level
        total_cells: Total number of cells
        code_cells: Number of code cells
        markdown_cells: Number of markdown cells
        empty_cells: Number of empty cells
        cells_with_outputs: Number of code cells with outputs
        cells_with_errors: Number of code cells with error outputs
        avg_cell_length: Average number of lines per cell
        issues: List of quality issues found
        metrics: Dictionary of additional metrics
    """

    overall_score: float
    overall_level: QualityLevel
    total_cells: int
    code_cells: int
    markdown_cells: int
    empty_cells: int
    cells_with_outputs: int
    cells_with_errors: int
    avg_cell_length: float
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class QualityInspector:
    """
    Generic notebook quality inspector.

    Provides comprehensive quality assessment for notebooks without
    any domain-specific assumptions. Focuses on structure, completeness,
    and general best practices.

    Usage:
        inspector = QualityInspector()
        metrics = inspector.inspect(notebook_builder)
        if metrics.overall_level == QualityLevel.POOR:
            print("Notebook needs improvement")
    """

    def __init__(
        self,
        min_cells: int = 1,
        max_empty_cells_ratio: float = 0.3,
        min_documentation_ratio: float = 0.1,
        max_error_ratio: float = 0.1,
    ):
        """
        Initialize quality inspector with thresholds.

        Args:
            min_cells: Minimum number of cells for a good notebook
            max_empty_cells_ratio: Maximum ratio of empty cells (0-1)
            min_documentation_ratio: Minimum ratio of markdown cells (0-1)
            max_error_ratio: Maximum ratio of cells with errors (0-1)
        """
        self.min_cells = min_cells
        self.max_empty_cells_ratio = max_empty_cells_ratio
        self.min_documentation_ratio = min_documentation_ratio
        self.max_error_ratio = max_error_ratio

    def inspect(self, notebook: NotebookBuilder) -> QualityMetrics:
        """
        Perform comprehensive quality inspection of a notebook.

        Args:
            notebook: NotebookBuilder to inspect

        Returns:
            QualityMetrics with detailed assessment
        """
        issues: list[QualityIssue] = []

        # Basic counts
        total_cells = notebook.cell_count()
        code_cells = notebook.code_cell_count()
        markdown_cells = notebook.markdown_cell_count()

        # Analyze cells
        empty_cells = 0
        cells_with_outputs = 0
        cells_with_errors = 0
        total_lines = 0

        for i, cell in enumerate(notebook.to_notebook().cells):
            cell_content = "".join(cell.source).strip()
            cell_lines = len([line for line in cell.source if line.strip()])
            total_lines += cell_lines

            # Check for empty cells
            if not cell_content:
                empty_cells += 1
                issues.append(
                    QualityIssue(
                        category="structure",
                        severity="warning",
                        message="Empty cell found",
                        cell_index=i,
                        suggestion="Consider removing empty cells or adding content",
                    )
                )

            # Analyze code cells
            if cell.cell_type == "code":
                # Check for outputs
                if cell.outputs:
                    cells_with_outputs += 1

                    # Check for errors in outputs
                    for output in cell.outputs:
                        if isinstance(output, dict) and output.get("output_type") == "error":
                            cells_with_errors += 1
                            issues.append(
                                QualityIssue(
                                    category="execution",
                                    severity="error",
                                    message=f"Cell has error output: {output.get('ename', 'Unknown error')}",
                                    cell_index=i,
                                    suggestion="Fix the code error before finalizing",
                                )
                            )
                            break

                # Check for very long cells
                if cell_lines > 50:
                    issues.append(
                        QualityIssue(
                            category="structure",
                            severity="warning",
                            message=f"Very long code cell ({cell_lines} lines)",
                            cell_index=i,
                            suggestion="Consider breaking long cells into smaller, focused cells",
                        )
                    )

                # Check for basic code quality issues
                self._check_code_quality(cell_content, i, issues)

        # Calculate ratios
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
        doc_ratio = markdown_cells / total_cells if total_cells > 0 else 0
        error_ratio = cells_with_errors / code_cells if code_cells > 0 else 0
        output_ratio = cells_with_outputs / code_cells if code_cells > 0 else 0
        avg_cell_length = total_lines / total_cells if total_cells > 0 else 0

        # Structure checks
        self._check_structure(notebook, total_cells, empty_ratio, doc_ratio, issues)

        # Execution checks
        self._check_execution(code_cells, error_ratio, output_ratio, issues)

        # Calculate overall score
        score = self._calculate_score(total_cells, empty_ratio, doc_ratio, error_ratio, output_ratio, len(issues))

        # Determine quality level
        if score >= 85:
            level = QualityLevel.EXCELLENT
        elif score >= 70:
            level = QualityLevel.GOOD
        elif score >= 50:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR

        return QualityMetrics(
            overall_score=score,
            overall_level=level,
            total_cells=total_cells,
            code_cells=code_cells,
            markdown_cells=markdown_cells,
            empty_cells=empty_cells,
            cells_with_outputs=cells_with_outputs,
            cells_with_errors=cells_with_errors,
            avg_cell_length=avg_cell_length,
            issues=issues,
            metrics={
                "empty_ratio": empty_ratio,
                "documentation_ratio": doc_ratio,
                "error_ratio": error_ratio,
                "output_ratio": output_ratio,
            },
        )

    def _check_structure(
        self,
        notebook: NotebookBuilder,
        total_cells: int,
        empty_ratio: float,
        doc_ratio: float,
        issues: list[QualityIssue],
    ) -> None:
        """Check notebook structure quality."""

        # Check minimum cells
        if total_cells < self.min_cells:
            issues.append(
                QualityIssue(
                    category="structure",
                    severity="warning",
                    message=f"Notebook has only {total_cells} cells (minimum recommended: {self.min_cells})",
                    suggestion="Add more cells to provide comprehensive analysis",
                )
            )

        # Check empty cells ratio
        if empty_ratio > self.max_empty_cells_ratio:
            issues.append(
                QualityIssue(
                    category="structure",
                    severity="warning",
                    message=f"High ratio of empty cells ({empty_ratio:.1%})",
                    suggestion="Remove unnecessary empty cells",
                )
            )

        # Check documentation ratio
        if doc_ratio < self.min_documentation_ratio:
            issues.append(
                QualityIssue(
                    category="documentation",
                    severity="warning",
                    message=f"Low documentation ratio ({doc_ratio:.1%})",
                    suggestion="Add more markdown cells to explain the analysis",
                )
            )

        # Check for notebook structure patterns
        if notebook.cells:
            first_cell = notebook.cells[0]
            if first_cell.cell_type != "markdown":
                issues.append(
                    QualityIssue(
                        category="structure",
                        severity="info",
                        message="First cell is not markdown",
                        suggestion="Consider starting with a markdown cell explaining the notebook's purpose",
                    )
                )

    def _check_execution(
        self, code_cells: int, error_ratio: float, output_ratio: float, issues: list[QualityIssue]
    ) -> None:
        """Check execution quality."""

        # Check error ratio
        if error_ratio > self.max_error_ratio:
            issues.append(
                QualityIssue(
                    category="execution",
                    severity="error",
                    message=f"High error ratio ({error_ratio:.1%})",
                    suggestion="Fix errors in code cells before finalizing",
                )
            )

        # Check if code cells have outputs
        if code_cells > 0 and output_ratio < 0.5:
            issues.append(
                QualityIssue(
                    category="execution",
                    severity="warning",
                    message=f"Many code cells lack outputs ({output_ratio:.1%} have outputs)",
                    suggestion="Execute all code cells to show results",
                )
            )

    def _check_code_quality(self, code: str, cell_index: int, issues: list[QualityIssue]) -> None:
        """Check basic code quality (language-agnostic checks)."""

        lines = code.split("\n")

        # Check for very long lines
        for line_num, line in enumerate(lines):
            if len(line) > 120:
                issues.append(
                    QualityIssue(
                        category="code_quality",
                        severity="info",
                        message=f"Very long line in cell {cell_index} (line {line_num + 1})",
                        cell_index=cell_index,
                        suggestion="Consider breaking long lines for better readability",
                    )
                )

        # Check for TODO/FIXME comments
        todo_patterns = ["TODO", "FIXME", "HACK", "XXX"]
        for pattern in todo_patterns:
            if pattern in code.upper():
                issues.append(
                    QualityIssue(
                        category="code_quality",
                        severity="warning",
                        message=f"Found {pattern} comment in cell {cell_index}",
                        cell_index=cell_index,
                        suggestion=f"Address the {pattern} comment before finalizing",
                    )
                )

    def _calculate_score(
        self,
        total_cells: int,
        empty_ratio: float,
        doc_ratio: float,
        error_ratio: float,
        output_ratio: float,
        num_issues: int,
    ) -> float:
        """Calculate overall quality score (0-100)."""

        score = 100.0

        # Deduct for structural issues
        if total_cells < self.min_cells:
            score -= 10

        # Deduct for empty cells
        if empty_ratio > self.max_empty_cells_ratio:
            score -= (empty_ratio - self.max_empty_cells_ratio) * 50

        # Deduct for lack of documentation
        if doc_ratio < self.min_documentation_ratio:
            score -= (self.min_documentation_ratio - doc_ratio) * 30

        # Heavily penalize errors
        score -= error_ratio * 40

        # Deduct for lack of outputs
        if output_ratio < 0.5:
            score -= (0.5 - output_ratio) * 20

        # Deduct for issues
        error_issues = len([i for i in range(num_issues) if i < len([1]) and "error" in str(i)])
        warning_issues = num_issues - error_issues

        score -= error_issues * 5
        score -= warning_issues * 2

        return max(0.0, min(100.0, score))

    def create_summary_report(self, metrics: QualityMetrics) -> str:
        """
        Create a human-readable summary report.

        Args:
            metrics: QualityMetrics to summarize

        Returns:
            Formatted summary report string
        """
        lines = [
            "Notebook Quality Report",
            "=" * 23,
            "",
            f"Overall Score: {metrics.overall_score:.1f}/100 ({metrics.overall_level.value.title()})",
            "",
            "Structure:",
            f"  Total Cells: {metrics.total_cells}",
            f"  Code Cells: {metrics.code_cells}",
            f"  Markdown Cells: {metrics.markdown_cells}",
            f"  Empty Cells: {metrics.empty_cells}",
            "",
            "Execution:",
            f"  Cells with Outputs: {metrics.cells_with_outputs}/{metrics.code_cells}",
            f"  Cells with Errors: {metrics.cells_with_errors}",
            "",
            "Quality Metrics:",
            f"  Documentation Ratio: {metrics.metrics.get('documentation_ratio', 0):.1%}",
            f"  Error Ratio: {metrics.metrics.get('error_ratio', 0):.1%}",
            f"  Output Ratio: {metrics.metrics.get('output_ratio', 0):.1%}",
            "",
        ]

        if metrics.issues:
            lines.extend([f"Issues Found ({len(metrics.issues)}):", ""])

            # Group issues by severity
            errors = [i for i in metrics.issues if i.severity == "error"]
            warnings = [i for i in metrics.issues if i.severity == "warning"]
            infos = [i for i in metrics.issues if i.severity == "info"]

            for severity, issue_list in [("Errors", errors), ("Warnings", warnings), ("Info", infos)]:
                if issue_list:
                    lines.append(f"  {severity}:")
                    for issue in issue_list:
                        cell_ref = f" (Cell {issue.cell_index})" if issue.cell_index is not None else ""
                        lines.append(f"    • {issue.message}{cell_ref}")
                    lines.append("")
        else:
            lines.append("No issues found!")

        return "\n".join(lines)
