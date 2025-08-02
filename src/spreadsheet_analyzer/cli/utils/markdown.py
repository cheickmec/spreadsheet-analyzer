"""Pure functions for converting pipeline results to markdown.

This module extracts the markdown generation logic from PipelineResultsToMarkdown
into pure functions that operate on immutable data.

CLAUDE-KNOWLEDGE: Markdown generation creates readable reports from analysis
results, using emojis and formatting for better visual presentation.
"""

from datetime import datetime

from ...pipeline.types import (
    ContentAnalysis,
    FormulaAnalysis,
    IntegrityResult,
    PipelineResult,
    SecurityReport,
    WorkbookStructure,
)


def pipeline_to_markdown(result: PipelineResult) -> list[str]:
    """Convert complete pipeline results to markdown cells.

    Creates a structured report with all analysis sections.

    Args:
        result: Complete pipeline analysis result

    Returns:
        List of markdown strings, one per notebook cell
    """
    cells = []

    # Always include header
    cells.append(create_header(result))

    # Always include integrity section
    cells.append(integrity_to_markdown(result.integrity))

    # Include security only if relevant
    security_md = security_to_markdown(result.security)
    if security_md:
        cells.append(security_md)

    # Always include structure
    cells.append(structure_to_markdown(result.structure, result.security))

    # Include formulas if present
    if result.formulas:
        formula_md = formulas_to_markdown(result.formulas)
        if formula_md:
            cells.append(formula_md)

    # Include content analysis if present
    if result.content:
        content_md = content_to_markdown(result.content)
        if content_md:
            cells.append(content_md)

    return [cell for cell in cells if cell]  # Filter out empty cells


def create_header(result: PipelineResult) -> str:
    """Create the main header markdown.

    Args:
        result: Pipeline result containing context

    Returns:
        Header markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# 📊 Excel Analysis Report

**File:** `{result.context.file_path.name}`
**Analysis Date:** {timestamp}
**Processing Time:** {result.context.total_duration:.2f}s"""


def integrity_to_markdown(integrity: IntegrityResult) -> str:
    """Convert integrity results to markdown.

    Args:
        integrity: File integrity analysis results

    Returns:
        Markdown string for integrity section
    """
    trust_stars = "⭐" * integrity.trust_tier

    md = f"""## 🔒 File Integrity Analysis

**File Hash:** `{integrity.file_hash[:16]}...`
**File Size:** {integrity.metadata.size_mb:.2f} MB
**MIME Type:** {integrity.metadata.mime_type}
**Excel Format:** {"✅ Valid Excel" if integrity.is_excel else "❌ Not Excel"}
**Trust Level:** {trust_stars} ({integrity.trust_tier}/5)
**Processing Class:** {integrity.processing_class}"""

    if not integrity.validation_passed:
        md += "\n\n### ⚠️ Validation Status: Failed"
        if integrity.validation_errors:
            md += "\n**Errors:**\n"
            for error in integrity.validation_errors[:5]:
                md += f"- {error}\n"
            if len(integrity.validation_errors) > 5:
                md += f"- ...and {len(integrity.validation_errors) - 5} more errors\n"

    return md


def security_to_markdown(security: SecurityReport) -> str:
    """Convert security results to markdown.

    Only shows security section if risk is MEDIUM or higher.

    Args:
        security: Security analysis results

    Returns:
        Markdown string for security section, or empty if low risk
    """
    # Only show security section if risk is MEDIUM or higher
    if security.risk_level in ("LOW", "NONE"):
        return ""

    risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}

    md = f"""## 🛡️ Security Analysis

**Risk Level:** {risk_emoji.get(security.risk_level, "❓")} {security.risk_level}"""

    if security.has_macros:
        md += "\n⚠️ **Contains VBA Macros**"
    if security.has_external_links:
        md += "\n⚠️ **Contains External Links**"
    if security.has_data_connections:
        md += "\n⚠️ **Contains Data Connections**"

    # Only show non-hidden sheet threats
    relevant_threats = [t for t in security.threats if t.threat_type not in ("HIDDEN_SHEET", "VERY_HIDDEN_SHEET")]

    if relevant_threats:
        md += "\n\n### Detected Threats:"
        for threat in relevant_threats[:5]:
            md += f"\n- **{threat.threat_type}**: {threat.description}"
            if threat.locations:
                md += f" (Found in: {', '.join(threat.locations[:3])})"

        if len(relevant_threats) > 5:
            md += f"\n- ...and {len(relevant_threats) - 5} more threats"

    return md


def structure_to_markdown(structure: WorkbookStructure, security: SecurityReport | None = None) -> str:
    """Convert structure results to markdown.

    Args:
        structure: Workbook structure analysis
        security: Optional security report for sheet visibility info

    Returns:
        Markdown string for structure section
    """
    # Count hidden sheets if security info available
    hidden_count = 0
    if security:
        hidden_count = sum(
            1 for threat in security.threats if threat.threat_type in ("HIDDEN_SHEET", "VERY_HIDDEN_SHEET")
        )

    total_with_hidden = structure.sheet_count + hidden_count

    md = f"""## 📋 Workbook Structure

**Total Sheets:** {structure.sheet_count}"""

    if hidden_count > 0:
        md += f" (+ {hidden_count} hidden)"

    md += f"""
**Total Cells:** {structure.total_cells:,}
**Used Range:** {structure.used_range or "N/A"}"""

    # Show sheet details
    if structure.sheets:
        md += "\n\n### Sheet Details:"
        for sheet in structure.sheets[:10]:  # Limit to first 10 sheets
            md += f"\n\n**{sheet.name}**"
            md += f"\n- Cells: {sheet.total_cells:,} ({sheet.used_cells:,} non-empty)"
            if sheet.dimensions:
                md += f"\n- Dimensions: {sheet.dimensions}"
            if sheet.has_tables:
                md += f"\n- Tables: {sheet.table_count}"
            if sheet.has_pivot_tables:
                md += f"\n- Pivot Tables: {sheet.pivot_table_count}"

        if len(structure.sheets) > 10:
            md += f"\n\n*...and {len(structure.sheets) - 10} more sheets*"

    return md


def formulas_to_markdown(formulas: FormulaAnalysis) -> str:
    """Convert formula analysis to markdown.

    Args:
        formulas: Formula analysis results

    Returns:
        Markdown string for formulas section
    """
    if formulas.total_formulas == 0:
        return ""

    complexity_emoji = {range(0, 20): "🟢", range(20, 50): "🟡", range(50, 80): "🟠", range(80, 101): "🔴"}

    # Find appropriate emoji for complexity score
    emoji = "❓"
    for range_obj, emoji_val in complexity_emoji.items():
        if formulas.formula_complexity_score in range_obj:
            emoji = emoji_val
            break

    md = f"""## 🧮 Formula Analysis

**Total Formulas:** {formulas.total_formulas:,}
**Unique Formulas:** {formulas.unique_formulas:,}
**Complexity Score:** {emoji} {formulas.formula_complexity_score}/100"""

    if formulas.has_circular_references:
        md += "\n\n⚠️ **WARNING: Circular References Detected!**"

    if formulas.has_errors:
        md += "\n\n❌ **Formula Errors Found**"

    # Show most complex formulas
    if formulas.most_complex_formulas:
        md += "\n\n### Most Complex Formulas:"
        for i, (location, formula) in enumerate(formulas.most_complex_formulas[:5], 1):
            # Truncate long formulas
            if len(formula) > 100:
                formula = formula[:97] + "..."
            md += f"\n{i}. `{location}`: `{formula}`"

    # Show common functions
    if formulas.function_usage:
        top_functions = sorted(formulas.function_usage.items(), key=lambda x: x[1], reverse=True)[:10]

        md += "\n\n### Top Functions Used:"
        for func, count in top_functions:
            md += f"\n- {func}: {count:,} times"

    return md


def content_to_markdown(content: ContentAnalysis) -> str:
    """Convert content analysis to markdown.

    Args:
        content: Content analysis results

    Returns:
        Markdown string for content section
    """
    md = """## 📝 Content Analysis"""

    # Data type distribution
    if content.data_type_distribution:
        total = sum(content.data_type_distribution.values())
        md += "\n\n### Data Types:"
        for dtype, count in sorted(content.data_type_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            md += f"\n- {dtype}: {count:,} ({percentage:.1f}%)"

    # Pattern insights
    if content.pattern_insights:
        md += "\n\n### Detected Patterns:"
        for insight in content.pattern_insights[:10]:
            md += f"\n- {insight}"

        if len(content.pattern_insights) > 10:
            md += f"\n- ...and {len(content.pattern_insights) - 10} more patterns"

    # Data quality metrics
    if content.completeness_score is not None:
        completeness_emoji = (
            "🟢" if content.completeness_score > 0.8 else "🟡" if content.completeness_score > 0.5 else "🔴"
        )
        md += "\n\n### Data Quality:"
        md += f"\n- Completeness: {completeness_emoji} {content.completeness_score:.1%}"

    if content.duplicate_row_count > 0:
        md += f"\n- Duplicate Rows: {content.duplicate_row_count:,}"

    # Potential issues
    if content.potential_issues:
        md += "\n\n### ⚠️ Potential Issues:"
        for issue in content.potential_issues[:5]:
            md += f"\n- {issue}"

        if len(content.potential_issues) > 5:
            md += f"\n- ...and {len(content.potential_issues) - 5} more issues"

    return md


def should_show_security(security: SecurityReport) -> bool:
    """Determine if security section should be shown.

    Security is only shown for MEDIUM risk or higher.

    Args:
        security: Security report to check

    Returns:
        True if security section should be displayed
    """
    return security.risk_level not in ("LOW", "NONE")
