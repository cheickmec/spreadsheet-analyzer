"""Rich console output handler for CLI.

This module provides beautiful terminal output with progress tracking,
tables, and colored formatting using the Rich library.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.table import Table

from spreadsheet_analyzer.pipeline.types import AnalysisResult, ContentInsight, SecurityThreat

# Console instance for all output
console = Console()

# Progress bar configuration
PROGRESS_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
]


@dataclass
class ConsoleTheme:
    """Color theme for console output."""

    # Status colors
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "blue"

    # Element colors
    header: str = "bold cyan"
    subheader: str = "bold white"
    key: str = "bright_blue"
    value: str = "white"

    # Severity colors
    critical: str = "bold red"
    high: str = "red"
    medium: str = "yellow"
    low: str = "blue"


# Default theme
theme = ConsoleTheme()


class RichConsoleHandler:
    """Rich terminal output handler with progress tracking and formatted display."""

    def __init__(self, *, no_color: bool = False):
        """Initialize console handler.

        Args:
            no_color: Disable colored output
        """
        self.console = Console(no_color=no_color)
        self.progress = None
        self._current_task = None

    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        self.console.print(f"\n[{theme.header}]{text}[/{theme.header}]\n")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[{theme.error}]✗ Error:[/{theme.error}] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[{theme.warning}]⚠ Warning:[/{theme.warning}] {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[{theme.success}]✓ Success:[/{theme.success}] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[{theme.info}]i Info:[/{theme.info}] {message}")

    def create_progress_callback(self, total_stages: int = 5) -> Callable[[str, float, str], None]:
        """Create a progress callback for analysis.

        Args:
            total_stages: Total number of pipeline stages

        Returns:
            Callback function for progress updates
        """
        self.progress = Progress(*PROGRESS_COLUMNS, console=self.console)
        self.progress.start()

        self._current_task = self.progress.add_task("[cyan]Analyzing file...", total=total_stages)

        def update_progress(stage: str, progress: float, message: str) -> None:
            """Update progress display."""
            # Update task description with current stage
            self.progress.update(
                self._current_task, description=f"[cyan]{stage}:[/cyan] {message}", advance=1 if progress >= 1.0 else 0
            )

        return update_progress

    def stop_progress(self) -> None:
        """Stop progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self._current_task = None

    def display_results(self, result: AnalysisResult, format: str) -> None:
        """Display analysis results in requested format.

        Args:
            result: Analysis results to display
            format: Output format (table, json, yaml, markdown)
        """
        if format == "table":
            self._display_table(result)
        elif format == "json":
            self._display_json(result)
        elif format == "yaml":
            self._display_yaml(result)
        elif format == "markdown":
            self._display_markdown(result)

    def _display_table(self, result: AnalysisResult) -> None:
        """Display results as rich tables."""
        # Summary panel
        summary = self._create_summary_panel(result)
        self.console.print(summary)

        # Issues and warnings
        if result.issues:
            self._display_issues_table(result.issues)

        if result.warnings:
            self._display_warnings_table(result.warnings)

        # File structure
        if result.structure:
            self._display_structure_table(result)

        # Security report
        if result.security and result.security.threats:
            self._display_security_table(list(result.security.threats))

        # Formula analysis
        if result.formulas:
            self._display_formula_table(result)

        # Content insights
        if result.content and result.content.insights:
            self._display_insights_table(list(result.content.insights))

    def _create_summary_panel(self, result: AnalysisResult) -> Panel:
        """Create summary panel for results."""
        status_color = theme.success if result.is_healthy else theme.error
        status_icon = "✓" if result.is_healthy else "✗"

        content = f"""[{theme.key}]File:[/{theme.key}] {result.file_path.name}
[{theme.key}]Size:[/{theme.key}] {result.file_size:,} bytes
[{theme.key}]Analysis Mode:[/{theme.key}] {result.analysis_mode}
[{theme.key}]Duration:[/{theme.key}] {result.duration_seconds:.2f}s
[{theme.key}]Status:[/{theme.key}] [{status_color}]{status_icon} {"Healthy" if result.is_healthy else "Issues Found"}[/{status_color}]
[{theme.key}]Issues:[/{theme.key}] {len(result.issues)}
[{theme.key}]Warnings:[/{theme.key}] {len(result.warnings)}"""

        return Panel(content, title="[bold]Analysis Summary[/bold]", border_style=status_color, box=box.ROUNDED)

    def _display_issues_table(self, issues: list[dict[str, Any]]) -> None:
        """Display issues in a table."""
        self.console.print(f"\n[{theme.error}]Issues Found:[/{theme.error}]\n")

        table = Table(box=box.SIMPLE_HEAD, show_header=True)
        table.add_column("Type", style="bright_red")
        table.add_column("Severity", style="red")
        table.add_column("Stage", style="cyan")
        table.add_column("Description", style="white")

        for issue in issues:
            table.add_row(
                issue.get("type", "unknown"),
                issue.get("severity", "unknown"),
                issue.get("stage", "unknown"),
                issue.get("message", "No description"),
            )

        self.console.print(table)

    def _display_warnings_table(self, warnings: list[dict[str, Any]]) -> None:
        """Display warnings in a table."""
        self.console.print(f"\n[{theme.warning}]Warnings:[/{theme.warning}]\n")

        table = Table(box=box.SIMPLE_HEAD, show_header=True)
        table.add_column("Type", style="yellow")
        table.add_column("Severity", style="yellow")
        table.add_column("Stage", style="cyan")
        table.add_column("Description", style="white")

        for warning in warnings:
            table.add_row(
                warning.get("type", "unknown"),
                warning.get("severity", "unknown"),
                warning.get("stage", "unknown"),
                warning.get("message", "No description"),
            )

        self.console.print(table)

    def _display_structure_table(self, result: AnalysisResult) -> None:
        """Display file structure information."""
        self.console.print(f"\n[{theme.header}]File Structure:[/{theme.header}]\n")

        structure = result.structure
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Property", style=theme.key)
        table.add_column("Value", style=theme.value)

        table.add_row("Sheets", str(structure.sheet_count))
        table.add_row("Total Cells", f"{structure.total_cells:,}")
        table.add_row("Total Formulas", f"{structure.total_formulas:,}")
        table.add_row("Complexity Score", f"{structure.complexity_score:.2f}")

        self.console.print(table)

        # Sheet details
        if structure.sheets:
            sheet_table = Table(box=box.SIMPLE_HEAD, title="Sheet Details")
            sheet_table.add_column("Sheet Name", style="cyan")
            sheet_table.add_column("Rows", justify="right")
            sheet_table.add_column("Columns", justify="right")
            sheet_table.add_column("Formulas", justify="right")

            for sheet in structure.sheets:
                sheet_table.add_row(sheet.name, str(sheet.row_count), str(sheet.column_count), str(sheet.formula_count))

            self.console.print(sheet_table)

    def _display_security_table(self, threats: list[SecurityThreat]) -> None:
        """Display security threats."""
        self.console.print(f"\n[{theme.error}]Security Threats:[/{theme.error}]\n")

        table = Table(box=box.SIMPLE_HEAD, show_header=True)
        table.add_column("Type", style="red")
        table.add_column("Risk Level", style="red")
        table.add_column("Location", style="cyan")
        table.add_column("Description", style="white")

        for threat in threats:
            risk_color = getattr(theme, threat.risk_level.lower(), "white")
            table.add_row(
                threat.threat_type,
                f"[{risk_color}]{threat.risk_level}[/{risk_color}]",
                threat.location or "N/A",
                threat.description,
            )

        self.console.print(table)

    def _display_formula_table(self, result: AnalysisResult) -> None:
        """Display formula analysis."""
        self.console.print(f"\n[{theme.header}]Formula Analysis:[/{theme.header}]\n")

        formulas = result.formulas
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style=theme.key)
        table.add_column("Value", style=theme.value)

        table.add_row("Total Formulas", str(len(formulas.dependency_graph)))
        table.add_row("Max Dependency Depth", str(formulas.max_dependency_depth))
        table.add_row("Complexity Score", f"{formulas.formula_complexity_score:.2f}")
        table.add_row("Circular References", str(len(formulas.circular_references)))
        table.add_row("Volatile Formulas", str(len(formulas.volatile_formulas)))
        table.add_row("External References", str(len(formulas.external_references)))

        self.console.print(table)

    def _display_insights_table(self, insights: list[ContentInsight]) -> None:
        """Display content insights."""
        self.console.print(f"\n[{theme.header}]Content Insights:[/{theme.header}]\n")

        for insight in insights:
            severity_color = getattr(theme, insight.severity.lower(), "white")

            panel = Panel(
                f"{insight.description}\n\n[dim]Affected areas: {', '.join(insight.affected_areas)}[/dim]",
                title=f"[{severity_color}]{insight.title}[/{severity_color}]",
                border_style=severity_color,
                box=box.ROUNDED,
            )
            self.console.print(panel)

    def _display_json(self, result: AnalysisResult) -> None:
        """Display results as JSON."""
        json_str = json.dumps(result.to_dict(), indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        self.console.print(syntax)

    def _display_yaml(self, result: AnalysisResult) -> None:
        """Display results as YAML."""
        yaml_str = yaml.dump(result.to_dict(), default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
        self.console.print(syntax)

    def _display_markdown(self, result: AnalysisResult) -> None:
        """Display results as Markdown."""
        md_lines = [
            f"# Analysis Report: {result.file_path.name}",
            "",
            "## Summary",
            "",
            f"- **File Size**: {result.file_size:,} bytes",
            f"- **Analysis Mode**: {result.analysis_mode}",
            f"- **Duration**: {result.duration_seconds:.2f}s",
            f"- **Status**: {'✓ Healthy' if result.is_healthy else '✗ Issues Found'}",
            f"- **Issues**: {len(result.issues)}",
            f"- **Warnings**: {len(result.warnings)}",
            "",
        ]

        if result.issues:
            md_lines.extend(
                [
                    "## Issues",
                    "",
                    "| Type | Severity | Stage | Description |",
                    "|------|----------|-------|-------------|",
                ]
            )
            for issue in result.issues:
                md_lines.append(
                    f"| {issue.get('type', 'unknown')} | "
                    f"{issue.get('severity', 'unknown')} | "
                    f"{issue.get('stage', 'unknown')} | "
                    f"{issue.get('message', 'No description')} |"
                )
            md_lines.append("")

        if result.structure:
            md_lines.extend(
                [
                    "## File Structure",
                    "",
                    f"- **Sheets**: {result.structure.sheet_count}",
                    f"- **Total Cells**: {result.structure.total_cells:,}",
                    f"- **Total Formulas**: {result.structure.total_formulas:,}",
                    f"- **Complexity Score**: {result.structure.complexity_score:.2f}",
                    "",
                ]
            )

        markdown = "\n".join(md_lines)
        syntax = Syntax(markdown, "markdown", theme="monokai", line_numbers=False)
        self.console.print(syntax)
