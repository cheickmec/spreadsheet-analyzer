"""CLI interface for Spreadsheet Analyzer.

This module provides the unified CLI entry point for spreadsheet analysis
using the modern three-tier architecture (core_exec, plugins, workflows).

The CLI uses LLM analysis by default with --no-llm flag for deterministic operation.
"""

from typing import NoReturn

from .analyze import main as analyze_main

__version__ = "0.4.0"


def main() -> NoReturn:
    """Main CLI entry point - delegates to the unified analyze command."""
    analyze_main()


if __name__ == "__main__":
    main()
