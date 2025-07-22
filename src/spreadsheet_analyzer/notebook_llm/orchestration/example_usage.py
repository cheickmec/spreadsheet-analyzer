"""Example usage of the orchestration layer.

This module demonstrates how to use the orchestration layer for spreadsheet analysis.
"""

import asyncio
import logging
from pathlib import Path

from spreadsheet_analyzer.agents.kernel_manager import KernelManager
from spreadsheet_analyzer.notebook_llm.orchestration import (
    CostController,
    ModelRouter,
    PythonWorkflowOrchestrator,
)
from spreadsheet_analyzer.notebook_llm.strategies.registry import StrategyRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def analyze_spreadsheet_example():
    """Example of analyzing a spreadsheet with the orchestration layer."""

    # Initialize components
    strategy_registry = StrategyRegistry()
    model_router = ModelRouter()
    kernel_manager = KernelManager()

    # Create orchestrator
    orchestrator = PythonWorkflowOrchestrator(
        strategy_registry=strategy_registry,
        model_router=model_router,
        kernel_manager=kernel_manager,
        max_concurrent_agents=5,
    )

    # Path to Excel file
    excel_file = Path("example_data/financial_model.xlsx")

    # Set token budget (e.g., $10 worth at ~$0.01 per 1K tokens)
    token_budget = 100000  # 100K tokens

    # Run analysis
    print(f"Starting analysis of {excel_file}")
    print(f"Token budget: {token_budget:,} tokens")
    print("-" * 60)

    result = await orchestrator.analyze_spreadsheet(
        file_path=excel_file,
        token_budget=token_budget,
    )

    if result.success:
        print("\nAnalysis completed successfully!")
        print("\nKey Results:")
        print(f"- Total execution time: {result.value['total_execution_time']:.2f} seconds")
        print(f"- Total tokens used: {result.value['total_tokens_used']:,}")
        print(f"- Sheets analyzed: {len(result.value.get('agent_analyses', {}))}")

        # Print summary if available
        if "analysis_summary" in result.value:
            summary = result.value["analysis_summary"]
            if summary:
                print(f"\nExecutive Summary: {summary.get('executive_summary', 'N/A')}")

        # Get cost report
        cost_report = model_router.get_usage_report()
        print("\nCost Report:")
        print(f"- Total cost: ${cost_report['total_cost']:.4f}")
        print(f"- Total requests: {cost_report['total_requests']}")

    else:
        print(f"\nAnalysis failed: {result.error}")
        if result.details:
            print("\nError details:")
            for key, value in result.details.items():
                print(f"  {key}: {value}")


async def demonstrate_model_routing():
    """Demonstrate the model routing capabilities."""

    print("\n" + "=" * 60)
    print("Model Routing Examples")
    print("=" * 60)

    router = ModelRouter()

    # Example 1: Simple extraction task
    print("\n1. Simple extraction task:")
    estimates = router.estimate_task_cost(
        task_type="extraction",
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
    )
    for model, cost in sorted(estimates.items(), key=lambda x: x[1]):
        print(f"   {model}: ${cost:.4f}")

    # Example 2: Complex reasoning task
    print("\n2. Complex reasoning task:")
    estimates = router.estimate_task_cost(
        task_type="reasoning",
        estimated_input_tokens=5000,
        estimated_output_tokens=2000,
    )
    for model, cost in sorted(estimates.items(), key=lambda x: x[1]):
        print(f"   {model}: ${cost:.4f}")

    # Example 3: Cost control
    print("\n3. Cost control example:")
    controller = CostController(budget_limit=1.0)  # $1 budget

    # Reserve budget for tasks
    if controller.reserve_budget("task1", 0.25):
        print("   Reserved $0.25 for task1")

    if controller.reserve_budget("task2", 0.50):
        print("   Reserved $0.50 for task2")

    print(f"   Remaining budget: ${controller.get_remaining_budget():.2f}")

    # Commit actual costs
    controller.commit_cost("task1", 0.20)  # Actual cost less than reserved
    print("   Task1 completed, cost: $0.20")
    print(f"   Budget summary: {controller.get_budget_summary()}")


async def demonstrate_workflow_steps():
    """Demonstrate the workflow step structure."""

    print("\n" + "=" * 60)
    print("Workflow Steps")
    print("=" * 60)

    # Create a simple orchestrator to show workflow
    strategy_registry = StrategyRegistry()
    model_router = ModelRouter()

    orchestrator = PythonWorkflowOrchestrator(
        strategy_registry=strategy_registry,
        model_router=model_router,
    )

    print("\nConfigured workflow steps:")
    for i, step in enumerate(orchestrator.workflow_steps, 1):
        print(f"\n{i}. {step.name}")
        print(f"   Type: {step.step_type.value}")
        print(f"   Description: {step.description}")
        print(f"   Dependencies: {step.dependencies or 'None'}")
        print(f"   Token budget: {step.token_budget_percentage * 100:.0f}%")
        print(f"   Timeout: {step.timeout_seconds}s")


async def main():
    """Run all examples."""

    # Demonstrate model routing
    await demonstrate_model_routing()

    # Demonstrate workflow steps
    await demonstrate_workflow_steps()

    # Run full analysis (if file exists)
    excel_file = Path("example_data/financial_model.xlsx")
    if excel_file.exists():
        await analyze_spreadsheet_example()
    else:
        print(f"\n\nSkipping full analysis - file not found: {excel_file}")
        print("To run a full analysis, create an Excel file at that location.")


if __name__ == "__main__":
    asyncio.run(main())
