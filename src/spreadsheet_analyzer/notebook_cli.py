#!/usr/bin/env python3
"""
Notebook Tools CLI

Automated Excel analysis using LLM function calling with the notebook tools interface.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from structlog import get_logger

from spreadsheet_analyzer.notebook_llm_interface import (
    get_notebook_tools,
)
from spreadsheet_analyzer.notebook_session import notebook_session

logger = get_logger(__name__)


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
        llm = None

        # Try primary model first
        try:
            if "claude" in args.model.lower():
                if args.api_key:
                    os.environ["ANTHROPIC_API_KEY"] = args.api_key
                llm = ChatAnthropic(model_name=args.model)
            elif "gpt" in args.model.lower():
                if args.api_key:
                    os.environ["OPENAI_API_KEY"] = args.api_key
                llm = ChatOpenAI(model_name=args.model)
            else:
                logger.error(f"Unsupported primary model: {args.model}")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize primary model {args.model}: {e}")
            llm = None

        # If primary model failed, try fallback
        if llm is None:
            try:
                logger.info("Trying fallback model: gpt-4")
                llm = ChatOpenAI(model_name="gpt-4")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback model gpt-4: {fallback_e}")
                return

        tools = get_notebook_tools()
        llm_with_tools = llm.bind_tools(tools)

        session_id = args.session_id or f"{args.excel_file.stem}_analysis_session"

        # Resolve the excel file path to be absolute
        excel_path = args.excel_file.resolve()
        notebook_path = args.notebook_path or excel_path.parent / f"{excel_path.stem}_analysis.ipynb"

        logger.info(f"Starting notebook session with model: {type(llm).__name__}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Notebook Path: {notebook_path}")
        logger.info(f"Max rounds: {args.max_rounds}")

        async with notebook_session(session_id, notebook_path) as session:
            # Register the session with the global session manager so tools can access it
            from spreadsheet_analyzer.notebook_llm_interface import get_session_manager

            session_manager = get_session_manager()
            session_manager._sessions["default_session"] = session

            initial_prompt = (
                "You are an expert data analyst. Your goal is to conduct a thorough analysis of the provided Excel file. "
                f"The file is located at: '{args.excel_file.resolve()}'.\n\n"
                "Follow these steps:\n"
                "1. Load the data from the Excel file into a pandas DataFrame. Use the tools provided.\n"
                "2. Explore the data to understand its structure, columns, and data types.\n"
                "3. Identify patterns, trends, and insights in the data.\n"
                "4. Create visualizations if appropriate.\n"
                "5. Summarize your findings and provide actionable insights.\n\n"
                "Start by loading the data and showing its basic information."
            )

            messages = [
                SystemMessage(
                    "You are an AI assistant that can interact with a Jupyter notebook to analyze data. "
                    "Use the provided tools to execute code, manage cells, and explore the data thoroughly. "
                    "Always use the tools to perform actions rather than just describing what you would do."
                ),
                HumanMessage(content=initial_prompt),
            ]

            for round_num in range(1, args.max_rounds + 1):
                logger.info(f"Starting analysis round {round_num}/{args.max_rounds}")

                if args.verbose:
                    logger.info("Sending messages to LLM:", messages=messages)

                try:
                    response = await llm_with_tools.ainvoke(messages)
                except Exception as api_error:
                    logger.warning(f"Primary model API call failed: {api_error}")

                    # Try fallback if we haven't already
                    if not isinstance(llm, ChatOpenAI):
                        try:
                            logger.info("Switching to fallback model: gpt-4")
                            llm = ChatOpenAI(model_name="gpt-4")
                            llm_with_tools = llm.bind_tools(tools)
                            response = await llm_with_tools.ainvoke(messages)
                        except Exception as fallback_error:
                            logger.error(f"Fallback model also failed: {fallback_error}")
                            break
                    else:
                        logger.error(f"API call failed: {api_error}")
                        break

                if args.verbose:
                    logger.info("Received response from LLM:", response=response)

                # Process tool calls
                tool_output_messages = []
                if response.tool_calls:
                    # Add the AI response with tool calls to the conversation first
                    messages.append(response)

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args")
                        logger.info(f"LLM called tool: {tool_name}", args=tool_args)

                        # Dynamically call the tool function
                        tool_func = next((t for t in tools if t.name == tool_name), None)
                        if tool_func:
                            tool_output = await tool_func.ainvoke(tool_args)
                            logger.info(f"Tool output: {tool_output}")
                            tool_output_messages.append(
                                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                            )
                        else:
                            logger.warning(f"LLM tried to call unknown tool: {tool_name}")

                    # Add tool results to conversation
                    messages.extend(tool_output_messages)

                elif response.content:
                    logger.info(f"LLM response: {response.content}")
                    # If no tool calls, the LLM is done
                    break
                else:
                    logger.warning("LLM response was empty.")
                    break

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
