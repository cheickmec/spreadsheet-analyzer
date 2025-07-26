#!/usr/bin/env python3
"""Debug script to test notebook outputs preservation."""

import asyncio
import logging
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def main():
    """Test analyze_sheet_with_langchain with focus on outputs."""
    excel_path = Path("test-files/business-accounting/Business Accounting.xlsx")
    sheet_name = "TEST outputs debug"

    print(f"Testing analysis of {sheet_name} from {excel_path}")
    print("=" * 80)

    # Run analysis with debug logging enabled
    result = await analyze_sheet_with_langchain(
        excel_path=excel_path,
        sheet_name=sheet_name,
        skip_deterministic=True,  # Skip to speed up testing
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
        enable_tracing=False,
    )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Output path: {result.get('output_path')}")
    print(f"Metadata path: {result.get('metadata_path')}")

    # Check the final notebook
    if "notebook_final" in result:
        notebook_data = result["notebook_final"]
        print(f"\nFinal notebook has {len(notebook_data.get('cells', []))} cells")
        for i, cell in enumerate(notebook_data.get("cells", [])):
            outputs = cell.get("outputs", [])
            print(f"Cell {i}: {cell.get('cell_type')} - {len(outputs)} outputs")


if __name__ == "__main__":
    asyncio.run(main())
