#!/usr/bin/env python3
"""
Notebook Tools CLI

Automated Excel analysis using LLM function calling with the notebook tools interface.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from structlog import get_logger

from spreadsheet_analyzer.graph_db.query_interface import (
    create_enhanced_query_interface,
)
from spreadsheet_analyzer.notebook_session import notebook_session
from spreadsheet_analyzer.pipeline import DeterministicPipeline
from spreadsheet_analyzer.pipeline.types import (
    ContentAnalysis,
    FormulaAnalysis,
    IntegrityResult,
    PipelineResult,
    SecurityReport,
    WorkbookStructure,
)

logger = get_logger(__name__)


class PipelineResultsToMarkdown:
    """Convert pipeline results to well-formatted markdown cells."""

    @staticmethod
    def create_header(pipeline_result: PipelineResult) -> str:
        """Create the main header markdown."""
        return f"""# 📊 Excel Analysis Report

**File:** `{pipeline_result.context.file_path.name}`
**Analysis Time:** `{pipeline_result.context.start_time.strftime("%Y-%m-%d %H:%M:%S")}`
**Execution Time:** `{pipeline_result.execution_time:.2f} seconds`
**Status:** {"✅ Success" if pipeline_result.success else "❌ Failed"}

---
"""

    @staticmethod
    def integrity_to_markdown(integrity: IntegrityResult) -> str:
        """Convert integrity results to markdown."""
        trust_stars = "⭐" * integrity.trust_tier

        md = f"""## 🔒 File Integrity Analysis

**File Hash:** `{integrity.file_hash[:16]}...`
**File Size:** {integrity.metadata.size_mb:.2f} MB
**MIME Type:** {integrity.metadata.mime_type}
**Excel Format:** {"✅ Valid Excel" if integrity.is_excel else "❌ Not Excel"}
**Trust Level:** {trust_stars} ({integrity.trust_tier}/5)
**Processing Class:** {integrity.processing_class}
"""

        if not integrity.validation_passed:
            md += "\n### ⚠️ Validation Status: Failed\n"

        return md

    @staticmethod
    def security_to_markdown(security: SecurityReport) -> str:
        """Convert security results to markdown."""
        risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "✅"}

        md = f"""## 🛡️ Security Analysis

**Risk Level:** {risk_emoji.get(security.risk_level, "❓")} {security.risk_level}
**Safe to Process:** {"✅ Yes" if security.is_safe else "❌ No"}
**Threats Found:** {security.threat_count}
"""

        if security.has_macros:
            md += "\n⚠️ **Contains VBA Macros**\n"
        if security.has_external_links:
            md += "\n⚠️ **Contains External Links**\n"

        if security.threats:
            md += "\n### Detected Threats:\n"
            for threat in security.threats[:5]:  # Show first 5
                md += f"- **{threat.threat_type}** ({threat.risk_level}): {threat.description}\n"
            if len(security.threats) > 5:
                md += f"- ...and {len(security.threats) - 5} more threats\n"

        return md

    @staticmethod
    def structure_to_markdown(structure: WorkbookStructure) -> str:
        """Convert structure results to markdown."""
        md = f"""## 📋 Structural Analysis

**Total Sheets:** {structure.sheet_count}
**Total Cells with Data:** {structure.total_cells:,}
**Total Formulas:** {structure.total_formulas:,}
**Named Ranges:** {len(structure.named_ranges)}
**Complexity Score:** {structure.complexity_score}/100

### Sheet Details:
"""

        for sheet in structure.sheets[:10]:  # Show first 10 sheets
            md += f"- **{sheet.name}**: {sheet.row_count:,} rows × {sheet.column_count:,} cols, {sheet.formula_count:,} formulas\n"

        if len(structure.sheets) > 10:
            md += f"- ...and {len(structure.sheets) - 10} more sheets\n"

        return md

    @staticmethod
    def formulas_to_markdown(formulas: FormulaAnalysis) -> str:
        """Convert formula analysis to markdown."""
        md = f"""## 🔗 Formula Analysis

**Total Formulas:** {len(formulas.dependency_graph):,}
**Max Dependency Depth:** {formulas.max_dependency_depth} levels
**Formula Complexity Score:** {formulas.formula_complexity_score}/100
**Circular References:** {"⚠️ Yes" if formulas.has_circular_references else "✅ No"}
**Volatile Formulas:** {len(formulas.volatile_formulas)}
**External References:** {len(formulas.external_references)}
"""

        if formulas.circular_references:
            md += "\n### ⚠️ Circular References Found:\n"
            for i, chain in enumerate(list(formulas.circular_references)[:3], 1):
                chain_str = " → ".join(f"`{cell}`" for cell in list(chain)[:5])
                if len(chain) > 5:
                    chain_str += " → ..."
                md += f"{i}. {chain_str}\n"
            if len(formulas.circular_references) > 3:
                md += f"...and {len(formulas.circular_references) - 3} more circular reference chains\n"

        if formulas.volatile_formulas:
            md += "\n### 🔄 Volatile Formulas (recalculate on every change):\n"
            for cell in list(formulas.volatile_formulas)[:5]:
                md += f"- `{cell}`\n"
            if len(formulas.volatile_formulas) > 5:
                md += f"- ...and {len(formulas.volatile_formulas) - 5} more volatile formulas\n"

        return md

    @staticmethod
    def content_to_markdown(content: ContentAnalysis) -> str:
        """Convert content analysis to markdown."""
        quality_emoji = "🟢" if content.data_quality_score >= 80 else "🟡" if content.data_quality_score >= 60 else "🔴"

        md = f"""## 📊 Content Analysis

**Data Quality Score:** {quality_emoji} {content.data_quality_score}/100
**Patterns Found:** {len(content.data_patterns)}
**Insights Generated:** {len(content.insights)}

### Summary:
{content.summary}
"""

        if content.insights:
            md += "\n### 💡 Key Insights:\n"
            for insight in content.insights[:5]:
                severity_emoji = "🔴" if insight.severity == "HIGH" else "🟡" if insight.severity == "MEDIUM" else "🟢"
                md += f"\n**{insight.title}** {severity_emoji}\n"
                md += f"{insight.description}\n"
                if insight.recommendation:
                    md += f"- **Recommendation:** {insight.recommendation}\n"
            if len(content.insights) > 5:
                md += f"\n...and {len(content.insights) - 5} more insights\n"

        if content.data_patterns:
            md += "\n### 🔍 Data Patterns:\n"
            for pattern in content.data_patterns[:3]:
                confidence = int(pattern.confidence * 100)
                md += f"- **{pattern.pattern_type}** (Confidence: {confidence}%): {pattern.description}\n"

        return md

    @staticmethod
    def create_summary(pipeline_result: PipelineResult) -> str:
        """Create a summary markdown section."""
        md = "## 📌 Analysis Summary\n\n"

        if pipeline_result.errors:
            md += "### ❌ Errors Encountered:\n"
            for error in pipeline_result.errors:
                md += f"- {error}\n"
            md += "\n"

        md += "### ✅ Completed Analysis Stages:\n"
        if pipeline_result.integrity:
            md += "- ✓ File Integrity Check\n"
        if pipeline_result.security:
            md += "- ✓ Security Scan\n"
        if pipeline_result.structure:
            md += "- ✓ Structural Analysis\n"
        if pipeline_result.formulas:
            md += "- ✓ Formula Analysis\n"
        if pipeline_result.content:
            md += "- ✓ Content Intelligence\n"

        md += "\n---\n\n*This analysis was generated using the deterministic pipeline. Additional interactive analysis can be performed using the cells below.*"

        return md


class NotebookCLI:
    """CLI interface for automated Excel analysis with LLM integration."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Automated Excel analysis using LLM function calling.")
        parser.add_argument("excel_file", type=Path, help="Path to the Excel file to analyze.")
        parser.add_argument(
            "--model",
            type=str,
            default="claude-3-sonnet-20240229",
            help="LLM model to use (e.g., 'claude-3-sonnet-20240229', 'gpt-4').",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key for the LLM. Defaults to environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY).",
        )
        parser.add_argument(
            "--session-id",
            type=str,
            default=None,
            help="Unique ID for the notebook session. Defaults to a name derived from the Excel file.",
        )
        parser.add_argument(
            "--notebook-path",
            type=Path,
            default=None,
            help="Path to save/load the notebook. Defaults to notebook_{session_id}.ipynb.",
        )
        parser.add_argument(
            "--max-rounds",
            type=int,
            default=5,
            help="Maximum number of analysis rounds (i.e., LLM calls).",
        )
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
        return parser

    async def run_analysis(self, args):
        """Run the automated analysis loop."""
        # Resolve the excel file path to be absolute
        excel_path = args.excel_file.resolve()
        notebook_path = args.notebook_path or excel_path.parent / f"{excel_path.stem}_analysis.ipynb"
        session_id = args.session_id or f"{args.excel_file.stem}_analysis_session"

        # Check if file exists
        if not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
            return

        logger.info(f"Starting analysis of: {excel_path}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Notebook Path: {notebook_path}")

        # Step 1: Run deterministic pipeline
        logger.info("Running deterministic pipeline analysis...")
        pipeline_result = None
        query_interface = None

        try:
            # Run the pipeline with progress tracking
            pipeline = DeterministicPipeline()
            pipeline_result = await asyncio.to_thread(pipeline.run, excel_path)

            if pipeline_result.success:
                logger.info(f"✅ Pipeline analysis completed in {pipeline_result.execution_time:.2f} seconds")

                # Create query interface if formula analysis succeeded
                if pipeline_result.formulas:
                    query_interface = create_enhanced_query_interface(pipeline_result.formulas)
                    logger.info("Created enhanced query interface for formula analysis")
            else:
                logger.error("Pipeline analysis failed:")
                for error in pipeline_result.errors:
                    logger.error(f"  - {error}")

        except Exception as e:
            logger.exception(f"Error running deterministic pipeline: {e}")
            # Continue anyway - we can still create a notebook

        # Step 2: Create notebook and add pipeline results
        async with notebook_session(session_id, notebook_path) as session:
            # Add query interface to session if available
            session.query_interface = query_interface

            # Register the session with the global session manager so tools can access it
            from spreadsheet_analyzer.notebook_llm_interface import get_session_manager

            session_manager = get_session_manager()
            session_manager._sessions["default_session"] = session

            # Add pipeline results as markdown cells if available
            if pipeline_result:
                logger.info("Adding pipeline results to notebook...")
                toolkit = session.toolkit

                # Add header
                header_md = PipelineResultsToMarkdown.create_header(pipeline_result)
                result = await toolkit.render_markdown(header_md)
                if result.is_err():
                    logger.warning(f"Failed to add header: {result.err_value}")

                # Add integrity analysis
                if pipeline_result.integrity:
                    integrity_md = PipelineResultsToMarkdown.integrity_to_markdown(pipeline_result.integrity)
                    result = await toolkit.render_markdown(integrity_md)
                    if result.is_err():
                        logger.warning(f"Failed to add integrity analysis: {result.err_value}")

                # Add security analysis
                if pipeline_result.security:
                    security_md = PipelineResultsToMarkdown.security_to_markdown(pipeline_result.security)
                    result = await toolkit.render_markdown(security_md)
                    if result.is_err():
                        logger.warning(f"Failed to add security analysis: {result.err_value}")

                # Add structure analysis
                if pipeline_result.structure:
                    structure_md = PipelineResultsToMarkdown.structure_to_markdown(pipeline_result.structure)
                    result = await toolkit.render_markdown(structure_md)
                    if result.is_err():
                        logger.warning(f"Failed to add structure analysis: {result.err_value}")

                # Add formula analysis
                if pipeline_result.formulas:
                    formulas_md = PipelineResultsToMarkdown.formulas_to_markdown(pipeline_result.formulas)
                    result = await toolkit.render_markdown(formulas_md)
                    if result.is_err():
                        logger.warning(f"Failed to add formula analysis: {result.err_value}")

                # Add content analysis
                if pipeline_result.content:
                    content_md = PipelineResultsToMarkdown.content_to_markdown(pipeline_result.content)
                    result = await toolkit.render_markdown(content_md)
                    if result.is_err():
                        logger.warning(f"Failed to add content analysis: {result.err_value}")

                # Add summary
                summary_md = PipelineResultsToMarkdown.create_summary(pipeline_result)
                result = await toolkit.render_markdown(summary_md)
                if result.is_err():
                    logger.warning(f"Failed to add summary: {result.err_value}")

                logger.info("Pipeline results added to notebook")

            # Add a basic data loading cell
            logger.info("Adding data loading cell...")

            # Calculate relative path from repo root
            try:
                repo_root = Path.cwd()
                relative_path = excel_path.relative_to(repo_root)
                path_str = f'"{relative_path}"'
            except ValueError:
                # If file is outside repo, use absolute path
                path_str = f'r"{excel_path}"'

            load_data_code = f"""import pandas as pd
from pathlib import Path

# Load the Excel file (path relative to repo root)
excel_path = Path({path_str})
print(f"Loading data from: {{excel_path}}")

try:
    # Try to read the first sheet
    df = pd.read_excel(excel_path)
    print(f"\\nLoaded data with shape: {{df.shape}}")
    print(f"Columns: {{list(df.columns)}}")

    # Display first few rows
    print("\\nFirst 5 rows:")
    display(df.head())

    # Basic info
    print("\\nData types:")
    print(df.dtypes)

except Exception as e:
    print(f"Error loading Excel file: {{e}}")
    df = None
"""

            result = await toolkit.execute_code(load_data_code)
            if result.is_ok():
                logger.info(f"✅ Data loading cell executed: {result.ok_value.cell_id}")
            else:
                logger.error(f"❌ Failed to execute data loading cell: {result.err_value}")

            # Add graph query interface tools if formula analysis succeeded
            if query_interface and pipeline_result and pipeline_result.formulas:
                logger.info("Adding graph query interface tools...")

                # Add markdown documentation for graph queries
                graph_tools_doc = """## 🔍 Formula Dependency Query Tools

The deterministic pipeline has analyzed all formulas and created a dependency graph. You can query this graph using the following tools:

### Available Tools:

1. **get_cell_dependencies** - Analyze what a cell depends on and what depends on it
   - Parameters: `sheet` (e.g., "Summary"), `cell_ref` (e.g., "D2")

2. **find_cells_affecting_range** - Find all cells that affect a specific range
   - Parameters: `sheet`, `start_cell`, `end_cell`

3. **find_empty_cells_in_formula_ranges** - Find gaps in data that formulas reference
   - Parameters: `sheet`

4. **get_formula_statistics** - Get overall statistics about formulas
   - No parameters needed

5. **find_circular_references** - Find all circular reference chains
   - No parameters needed

### Usage:
These tools are available through the tool-calling interface. Each query will be documented in a markdown cell showing both the query and its results.
"""

                result = await toolkit.render_markdown(graph_tools_doc)
                if result.is_err():
                    logger.warning(f"Failed to add graph tools documentation: {result.err_value}")

                logger.info("Graph query tools are available via tool-calling interface")

            # Skip LLM interaction for now
            logger.info("Skipping LLM interaction (commented out)")

            # initial_prompt = (
            #     "You are an expert data analyst. Your goal is to conduct a thorough analysis of the provided Excel file. "
            #     f"The file is located at: '{args.excel_file.resolve()}'.\n\n"
            #     "Follow these steps:\n"
            #     "1. Load the data from the Excel file into a pandas DataFrame. Use the tools provided.\n"
            #     "2. Explore the data to understand its structure, columns, and data types.\n"
            #     "3. Identify patterns, trends, and insights in the data.\n"
            #     "4. Create visualizations if appropriate.\n"
            #     "5. Summarize your findings and provide actionable insights.\n\n"
            #     "Start by loading the data and showing its basic information."
            # )

            # Commented out LLM interaction for testing
            # messages = [
            #     SystemMessage(
            #         "You are an AI assistant that can interact with a Jupyter notebook to analyze data. "
            #         "Use the provided tools to execute code, manage cells, and explore the data thoroughly. "
            #         "Always use the tools to perform actions rather than just describing what you would do."
            #     ),
            #     HumanMessage(content=initial_prompt),
            # ]

            # for round_num in range(1, args.max_rounds + 1):
            #     logger.info(f"Starting analysis round {round_num}/{args.max_rounds}")

            #     if args.verbose:
            #         logger.info("Sending messages to LLM:", messages=messages)

            #     try:
            #         response = await llm_with_tools.ainvoke(messages)
            #     except Exception as api_error:
            #         logger.warning(f"Primary model API call failed: {api_error}")

            #         # Try fallback if we haven't already
            #         if not isinstance(llm, ChatOpenAI):
            #             try:
            #                 logger.info("Switching to fallback model: gpt-4")
            #                 llm = ChatOpenAI(model_name="gpt-4")
            #                 llm_with_tools = llm.bind_tools(tools)
            #                 response = await llm_with_tools.ainvoke(messages)
            #             except Exception as fallback_error:
            #                 logger.error(f"Fallback model also failed: {fallback_error}")
            #                 break
            #         else:
            #             logger.error(f"API call failed: {api_error}")
            #             break

            #     if args.verbose:
            #         logger.info("Received response from LLM:", response=response)

            #     # Process tool calls
            #     tool_output_messages = []
            #     if response.tool_calls:
            #         # Add the AI response with tool calls to the conversation first
            #         messages.append(response)

            #         for tool_call in response.tool_calls:
            #             tool_name = tool_call.get("name")
            #             tool_args = tool_call.get("args")
            #             logger.info(f"LLM called tool: {tool_name}", args=tool_args)

            #             # Dynamically call the tool function
            #             tool_func = next((t for t in tools if t.name == tool_name), None)
            #             if tool_func:
            #                 tool_output = await tool_func.ainvoke(tool_args)
            #                 logger.info(f"Tool output: {tool_output}")
            #                 tool_output_messages.append(
            #                     ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            #                 )
            #             else:
            #                 logger.warning(f"LLM tried to call unknown tool: {tool_name}")

            #         # Add tool results to conversation
            #         messages.extend(tool_output_messages)

            #     elif response.content:
            #         logger.info(f"LLM response: {response.content}")
            #         # If no tool calls, the LLM is done
            #         break
            #     else:
            #         logger.warning("LLM response was empty.")
            #         break

            # Ensure notebook is saved at the end
            logger.info("Analysis complete. Saving notebook...")
            save_result = session.toolkit.save_notebook(notebook_path, overwrite=True)
            if save_result.is_ok():
                logger.info(f"✅ Notebook saved successfully to: {save_result.ok_value}")
            else:
                logger.error(f"❌ Failed to save notebook: {save_result.err_value}")

            logger.info("Analysis session completed.")

    def run(self):
        """Parse arguments and run the analysis."""
        args = self.parser.parse_args()
        # Setup basic logging
        log_level = logging.INFO if args.verbose else logging.WARNING
        logging.basicConfig(level=log_level, stream=sys.stdout)

        # Give a more specific logger name
        global logger
        logger = get_logger(f"notebook_cli.{args.model}")

        asyncio.run(self.run_analysis(args))


if __name__ == "__main__":
    cli = NotebookCLI()
    cli.run()
