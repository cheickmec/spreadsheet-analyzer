#!/usr/bin/env python3
"""
Test Phoenix integration with the Spreadsheet Analyzer.

This script verifies that Phoenix observability is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structlog import get_logger

from spreadsheet_analyzer.observability import (
    PhoenixConfig,
    initialize_cost_tracker,
    initialize_phoenix,
    instrument_all,
)

logger = get_logger(__name__)


async def test_phoenix_integration():
    """Test Phoenix integration components."""

    print("🧪 Testing Phoenix Integration...\n")

    # 1. Test Phoenix initialization
    print("1️⃣ Testing Phoenix initialization...")
    config = PhoenixConfig(mode="local")
    tracer = initialize_phoenix(config)

    if tracer:
        print("✅ Phoenix initialized successfully")
    else:
        print("❌ Phoenix initialization failed")
        return False

    # 2. Test instrumentation
    print("\n2️⃣ Testing instrumentation...")
    results = instrument_all(tracer)

    for provider, success in results.items():
        status = "✅" if success else "⚠️"
        print(f"{status} {provider}: {'instrumented' if success else 'not available'}")

    # 3. Test cost tracking
    print("\n3️⃣ Testing cost tracking...")

    # Initialize with a limit
    tracker = initialize_cost_tracker(cost_limit=1.0)

    # Track some usage
    test_models = [
        ("gpt-4", 100, 50),
        ("claude-3-5-sonnet-20241022", 200, 100),
        ("gpt-3.5-turbo", 1000, 500),
    ]

    for model, input_tokens, output_tokens in test_models:
        usage = tracker.track_usage(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens, metadata={"test": True}
        )
        print(f"✅ Tracked {model}: ${usage.total_cost:.4f}")

    # Get summary
    summary = tracker.get_summary()
    print("\n📊 Cost Summary:")
    print(f"   Total Cost: ${summary['total_cost_usd']:.4f}")
    print(f"   Total Tokens: {summary['total_tokens']['total']:,}")
    print(f"   Within Budget: {'✅' if summary['within_budget'] else '❌'}")

    # 4. Test LangChain integration
    print("\n4️⃣ Testing LangChain integration...")
    try:
        from langchain.schema import LLMResult
        from langchain_core.messages import HumanMessage

        # Create a mock LLM response
        mock_response = LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
        )
        print("✅ LangChain types available")
    except ImportError:
        print("⚠️ LangChain not fully configured")

    print("\n✅ All tests completed!")
    print("\n📝 Next steps:")
    print("1. Start Phoenix with: ./scripts/start_phoenix.sh")
    print("2. Run an analysis: python -m spreadsheet_analyzer.notebook_cli data.xlsx")
    print("3. View traces at: http://localhost:6006")

    return True


def main():
    """Run the test."""
    success = asyncio.run(test_phoenix_integration())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
