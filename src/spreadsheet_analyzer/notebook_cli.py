#!/usr/bin/env python3
"""
Notebook Tools CLI

Interactive notebook analysis using LLM function calling with the notebook tools interface.
"""

import argparse
import asyncio
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from structlog import get_logger

from spreadsheet_analyzer.notebook_llm_interface import create_notebook_tool_descriptions, get_notebook_tools
from spreadsheet_analyzer.notebook_session import notebook_session

logger = get_logger(__name__)


class NotebookCLI:
    """CLI interface for notebook tools with LLM integration."""

    def __init__(self, model_name: str, api_key: str | None = None):
        """Initialize the CLI with LLM configuration."""
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._create_llm()
        self.tools = get_notebook_tools()
        self.tool_descriptions = create_notebook_tool_descriptions()

    def _create_llm(self):
        """Create the LLM instance based on model name."""
        if self.model_name.startswith("claude"):
            if not self.api_key:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required for Claude models")
            return ChatAnthropic(model=self.model_name, anthropic_api_key=self.api_key, temperature=0.1)
        elif self.model_name.startswith("gpt"):
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for GPT models")
            return ChatOpenAI(model=self.model_name, openai_api_key=self.api_key, temperature=0.1)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    async def start_session(self):
        """Start a notebook session."""
        async with notebook_session("default_session") as session:
            logger.info("Notebook session started", session_id=session.session_id)
            return session

    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return f"""You are an AI assistant that can interact with Jupyter notebooks programmatically.

You have access to the following tools for notebook manipulation:

{self.tool_descriptions}

Guidelines:
1. Always use the appropriate tool for the task
2. For code execution, use `execute_code` for new code or `edit_and_execute` to modify existing cells
3. Use `add_markdown_cell` for documentation and explanations
4. Check the notebook state with `get_notebook_state` when needed
5. Save your work with `save_notebook` when appropriate
6. Be conversational and explain what you're doing
7. If code fails, try to fix it and re-execute

You can analyze spreadsheets, create visualizations, perform data analysis, and document your findings.
"""

    async def interactive_loop(self):
        """Run the interactive CLI loop."""
        print("🤖 Notebook Tools CLI")
        print(f"📊 Model: {self.model_name}")
        print("💡 Type 'quit' or 'exit' to end the session")
        print("=" * 50)

        # Start the notebook session
        await self.start_session()

        # Create the system message
        system_message = SystemMessage(content=self.create_system_prompt())

        while True:
            try:
                # Get user input
                user_input = input("\n🔍 What would you like to do? ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Create the user message
                user_message = HumanMessage(content=user_input)

                # Get LLM response with tool calling
                response = await self.llm.ainvoke([system_message, user_message], tools=self.tools)

                # Handle tool calls if any
                if response.tool_calls:
                    print(f"\n🔧 Executing {len(response.tool_calls)} tool(s)...")

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        print(f"  📝 {tool_name}: {tool_args}")

                        # Find and execute the tool
                        tool_func = next((t for t in self.tools if t.name == tool_name), None)
                        if tool_func:
                            try:
                                result = await tool_func.ainvoke(tool_args)
                                print(f"  ✅ Result: {result}")
                            except Exception as e:
                                print(f"  ❌ Error: {e!s}")
                        else:
                            print(f"  ❌ Tool '{tool_name}' not found")

                # Show the LLM's response
                if response.content:
                    print(f"\n🤖 {response.content}")

            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e!s}")
                logger.error("CLI error", error=str(e), exc_info=True)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive notebook analysis with LLM function calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m spreadsheet_analyzer.notebook_cli --model claude-3-sonnet-20240229
  python -m spreadsheet_analyzer.notebook_cli --model gpt-4 --api-key sk-...
        """,
    )

    parser.add_argument(
        "--model", default="claude-3-sonnet-20240229", help="LLM model to use (default: claude-3-sonnet-20240229)"
    )

    parser.add_argument("--api-key", help="API key for the LLM (will use environment variables if not provided)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    try:
        # Create and run the CLI
        cli = NotebookCLI(args.model, args.api_key)
        await cli.interactive_loop()

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e!s}")
        logger.error("Fatal CLI error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
