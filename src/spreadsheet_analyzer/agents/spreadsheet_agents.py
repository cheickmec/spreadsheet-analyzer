"""Specialized agents for spreadsheet analysis.

This module provides functional implementations of agents
specifically designed for analyzing spreadsheet data.

CLAUDE-KNOWLEDGE: These agents handle specific aspects of spreadsheet
analysis including formula parsing, pattern detection, and data validation.
"""

from ..context import (
    ContextQuery,
    TokenBudget,
    build_context_from_cells,
)
from ..core.errors import AgentError
from ..core.types import Result, err, ok
from .core import FunctionalAgent, create_simple_agent
from .types import (
    AgentCapability,
    AgentMessage,
    AgentState,
)

# Formula Analysis Agent


def create_formula_analyzer() -> FunctionalAgent:
    """Create an agent specialized in formula analysis."""

    def analyze_formulas(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Analyze formulas in spreadsheet data."""
        try:
            data = message.content

            # Extract cells with formulas
            formula_cells = [cell for cell in data.get("cells", []) if cell.get("type") == "formula"]

            # Analyze formula patterns
            formula_patterns = {}
            function_usage = {}
            complexity_scores = []

            for cell in formula_cells:
                formula = cell.get("content", "")

                # Extract functions used
                import re

                functions = re.findall(r"([A-Z]+)\(", formula)
                for func in functions:
                    function_usage[func] = function_usage.get(func, 0) + 1

                # Simple complexity scoring
                complexity = len(functions) + formula.count("(") + formula.count(",")
                complexity_scores.append(complexity)

                # Pattern detection (simplified)
                pattern = re.sub(r"[A-Z]+\d+", "CELL", formula)
                formula_patterns[pattern] = formula_patterns.get(pattern, 0) + 1

            # Prepare analysis results
            analysis = {
                "total_formulas": len(formula_cells),
                "unique_patterns": len(formula_patterns),
                "function_usage": function_usage,
                "average_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
                "most_common_pattern": max(formula_patterns.items(), key=lambda x: x[1])[0]
                if formula_patterns
                else None,
                "insights": [],
            }

            # Generate insights
            if len(formula_patterns) < len(formula_cells) * 0.3:
                analysis["insights"].append("High formula reuse detected - consider template optimization")

            if analysis["average_complexity"] > 10:
                analysis["insights"].append("Complex formulas detected - consider breaking down for clarity")

            # Create response
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=analysis,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Formula analysis failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="analyze_formulas",
            description="Analyze formula patterns, complexity, and usage",
            input_type=dict,
            output_type=dict,
        )
    ]

    return create_simple_agent("formula_analyzer", capabilities, analyze_formulas)


# Pattern Detection Agent


def create_pattern_detector() -> FunctionalAgent:
    """Create an agent specialized in detecting data patterns."""

    def detect_patterns(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Detect patterns in spreadsheet data."""
        try:
            data = message.content
            cells = data.get("cells", [])

            patterns = {"sequential": [], "repeating": [], "formatting": [], "structural": []}

            # Detect sequential patterns (simplified)
            numeric_sequences = []
            for i in range(len(cells) - 1):
                if isinstance(cells[i].get("content"), int | float) and isinstance(
                    cells[i + 1].get("content"), int | float
                ):
                    diff = cells[i + 1]["content"] - cells[i]["content"]
                    numeric_sequences.append(diff)

            if numeric_sequences and all(d == numeric_sequences[0] for d in numeric_sequences):
                patterns["sequential"].append(
                    {
                        "type": "arithmetic_sequence",
                        "difference": numeric_sequences[0],
                        "locations": [cells[0]["location"], cells[-1]["location"]],
                    }
                )

            # Detect repeating values
            value_counts = {}
            for cell in cells:
                content = str(cell.get("content", ""))
                if content:
                    value_counts[content] = value_counts.get(content, 0) + 1

            for value, count in value_counts.items():
                if count > 5 and count > len(cells) * 0.1:
                    patterns["repeating"].append(
                        {"value": value, "frequency": count, "percentage": count / len(cells) * 100}
                    )

            # Detect structural patterns
            sheets = {}
            for cell in cells:
                location = cell.get("location", "")
                sheet = location.split("!")[0] if "!" in location else "Sheet1"
                sheets[sheet] = sheets.get(sheet, 0) + 1

            patterns["structural"].append({"sheet_distribution": sheets, "multi_sheet": len(sheets) > 1})

            # Create response
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content={
                    "patterns": patterns,
                    "summary": {
                        "total_patterns": sum(len(p) for p in patterns.values()),
                        "pattern_types": [k for k, v in patterns.items() if v],
                    },
                },
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Pattern detection failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="detect_patterns",
            description="Detect data patterns including sequences, repetitions, and structure",
            input_type=dict,
            output_type=dict,
        )
    ]

    return create_simple_agent("pattern_detector", capabilities, detect_patterns)


# Data Validation Agent


def create_data_validator() -> FunctionalAgent:
    """Create an agent specialized in data validation."""

    def validate_data(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Validate spreadsheet data for quality issues."""
        try:
            data = message.content
            cells = data.get("cells", [])
            validation_rules = data.get("rules", {})

            issues = []
            warnings = []

            # Check for empty cells
            empty_count = sum(1 for cell in cells if not cell.get("content"))
            if empty_count > len(cells) * 0.5:
                warnings.append(f"High percentage of empty cells: {empty_count / len(cells) * 100:.1f}%")

            # Check for formula errors
            error_patterns = ["#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A", "#NULL!"]
            formula_errors = []

            for cell in cells:
                content = str(cell.get("content", ""))
                for error in error_patterns:
                    if error in content:
                        formula_errors.append({"location": cell.get("location"), "error": error})

            if formula_errors:
                issues.append(
                    {
                        "type": "formula_errors",
                        "count": len(formula_errors),
                        "locations": [e["location"] for e in formula_errors[:5]],  # First 5
                    }
                )

            # Check data types consistency
            column_types = {}
            for cell in cells:
                location = cell.get("location", "")
                # Extract column (simplified)
                import re

                col_match = re.search(r"([A-Z]+)\d+", location)
                if col_match:
                    col = col_match.group(1)
                    content_type = type(cell.get("content")).__name__

                    if col not in column_types:
                        column_types[col] = {}
                    column_types[col][content_type] = column_types[col].get(content_type, 0) + 1

            # Check for mixed types in columns
            for col, types in column_types.items():
                if len(types) > 2:  # More than 2 types (including NoneType)
                    warnings.append(f"Column {col} has mixed data types: {list(types.keys())}")

            # Apply custom validation rules
            if validation_rules:
                for rule_name, rule_func in validation_rules.items():
                    try:
                        rule_issues = rule_func(cells)
                        if rule_issues:
                            issues.extend(rule_issues)
                    except Exception:
                        warnings.append(f"Custom rule '{rule_name}' failed to execute")

            # Create validation report
            validation_report = {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "statistics": {
                    "total_cells": len(cells),
                    "empty_cells": empty_count,
                    "formula_errors": len(formula_errors),
                    "data_quality_score": max(0, 100 - len(issues) * 10 - len(warnings) * 5),
                },
            }

            # Create response
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=validation_report,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Data validation failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="validate_data",
            description="Validate spreadsheet data for quality issues and errors",
            input_type=dict,
            output_type=dict,
        )
    ]

    return create_simple_agent("data_validator", capabilities, validate_data)


# Context Optimization Agent


def create_context_optimizer() -> FunctionalAgent:
    """Create an agent specialized in optimizing context for LLM analysis."""

    def optimize_context(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Optimize context for LLM token limits."""
        try:
            data = message.content
            cells = data.get("cells", [])
            query = data.get("query", "general analysis")
            token_budget = data.get("token_budget", 4000)
            model = data.get("model", "gpt-4")

            # Build context query
            context_query = ContextQuery(query_text=query, include_formulas=True, include_values=True)

            # Create token budget
            budget = TokenBudget(
                total=token_budget,
                context=int(token_budget * 0.8),  # 80% for context
            )

            # Build optimized context
            result = build_context_from_cells(cells, query, budget.context, model)

            if result.is_err():
                return err(AgentError(f"Context optimization failed: {result.unwrap_err()}"))

            package = result.unwrap()

            # Create response with optimization info
            response_content = {
                "optimized_cells": [
                    {
                        "location": cell.location,
                        "content": cell.content,
                        "type": cell.cell_type,
                        "importance": cell.importance,
                    }
                    for cell in package.cells
                ],
                "optimization_info": {
                    "original_cell_count": len(cells),
                    "optimized_cell_count": len(package.cells),
                    "token_count": package.token_count,
                    "token_budget": budget.context,
                    "compression_method": package.compression_method,
                    "reduction_percentage": (1 - len(package.cells) / len(cells)) * 100 if cells else 0,
                },
                "metadata": package.metadata,
                "focus_hints": list(package.focus_hints),
            }

            # Create response
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=response_content,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Context optimization failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="optimize_context",
            description="Optimize spreadsheet data for LLM context windows",
            input_type=dict,
            output_type=dict,
        )
    ]

    return create_simple_agent("context_optimizer", capabilities, optimize_context)


# Summary Generator Agent


def create_summary_generator() -> FunctionalAgent:
    """Create an agent that generates summaries from analysis results."""

    def generate_summary(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Generate summary from multiple analysis results."""
        try:
            data = message.content
            analyses = data.get("analyses", {})

            summary = {"overview": {}, "key_findings": [], "recommendations": [], "metrics": {}}

            # Process formula analysis
            if "formula_analysis" in analyses:
                fa = analyses["formula_analysis"]
                summary["overview"]["formulas"] = {
                    "total": fa.get("total_formulas", 0),
                    "complexity": fa.get("average_complexity", 0),
                }
                if fa.get("insights"):
                    summary["key_findings"].extend(fa["insights"])

            # Process pattern detection
            if "pattern_detection" in analyses:
                pd = analyses["pattern_detection"]
                summary["overview"]["patterns"] = pd.get("summary", {})

                patterns = pd.get("patterns", {})
                if patterns.get("sequential"):
                    summary["key_findings"].append("Sequential patterns detected in data")
                if patterns.get("repeating"):
                    summary["key_findings"].append(f"{len(patterns['repeating'])} repeating value patterns found")

            # Process data validation
            if "data_validation" in analyses:
                dv = analyses["data_validation"]
                summary["overview"]["data_quality"] = {
                    "score": dv.get("statistics", {}).get("data_quality_score", 0),
                    "issues": len(dv.get("issues", [])),
                }

                if not dv.get("valid"):
                    summary["recommendations"].append("Address data quality issues before analysis")

                summary["metrics"]["empty_cells"] = dv.get("statistics", {}).get("empty_cells", 0)
                summary["metrics"]["formula_errors"] = dv.get("statistics", {}).get("formula_errors", 0)

            # Generate overall assessment
            quality_score = summary["overview"].get("data_quality", {}).get("score", 100)
            if quality_score >= 90:
                assessment = "Excellent"
            elif quality_score >= 70:
                assessment = "Good"
            elif quality_score >= 50:
                assessment = "Fair"
            else:
                assessment = "Poor"

            summary["overall_assessment"] = (
                f"{assessment} data quality with {len(summary['key_findings'])} key findings"
            )

            # Create response
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=summary,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Summary generation failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="generate_summary",
            description="Generate comprehensive summary from multiple analyses",
            input_type=dict,
            output_type=dict,
        )
    ]

    return create_simple_agent("summary_generator", capabilities, generate_summary)


# Factory function for creating all spreadsheet agents


def create_spreadsheet_analysis_team() -> dict[str, FunctionalAgent]:
    """Create a team of agents for spreadsheet analysis."""
    return {
        "formula_analyzer": create_formula_analyzer(),
        "pattern_detector": create_pattern_detector(),
        "data_validator": create_data_validator(),
        "context_optimizer": create_context_optimizer(),
        "summary_generator": create_summary_generator(),
    }
