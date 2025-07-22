# Orchestration Layer

The orchestration layer manages the complete workflow for spreadsheet analysis, coordinating multiple agents, managing token budgets, and routing tasks to appropriate models.

## Overview

The orchestration layer consists of three main components:

1. **Base Orchestrator** (`base.py`) - Abstract base class and workflow step abstractions
1. **Python Orchestrator** (`python_orchestrator.py`) - Immediate Python implementation
1. **Model Management** (`models.py`) - Model routing, cost tracking, and optimization

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestration Layer                    │
│  • Workflow Management  • Multi-Agent Coordination       │
│  • Token Budget Control • Error Recovery                 │
├─────────────────────────────────────────────────────────┤
│                    Model Router                          │
│  • Model Selection      • Cost Optimization             │
│  • Usage Tracking       • Multi-Tier Support            │
├─────────────────────────────────────────────────────────┤
│                    Agent Manager                         │
│  • Agent Creation       • Task Distribution             │
│  • Result Synthesis     • Progress Monitoring           │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Workflow Management

The orchestrator manages a multi-phase analysis workflow:

1. **Deterministic Analysis** - Fast, zero-cost structural analysis
1. **Complexity Assessment** - Evaluate workbook complexity
1. **Agent Creation** - Create specialized agents for each sheet
1. **Parallel Analysis** - Analyze sheets concurrently with LLMs
1. **Relationship Analysis** - Discover cross-sheet dependencies
1. **Validation** - Verify findings and calculations
1. **Synthesis** - Generate final report

### 2. Model Routing

Intelligent routing based on task complexity:

- **Small Models** (e.g., GPT-3.5) - Simple extraction, counting
- **Medium Models** (e.g., Claude Haiku) - Pattern recognition, analysis
- **Large Models** (e.g., GPT-4, Claude Opus) - Complex reasoning, synthesis

### 3. Cost Management

- Token budget allocation across workflow steps
- Cost estimation before execution
- Real-time cost tracking
- Budget enforcement with graceful degradation

### 4. Error Recovery

- Automatic retry with exponential backoff
- Graceful handling of API failures
- Partial result preservation
- Configurable recovery strategies

## Usage

### Basic Example

```python
from spreadsheet_analyzer.notebook_llm.orchestration import (
    PythonWorkflowOrchestrator,
    ModelRouter,
)
from spreadsheet_analyzer.notebook_llm.strategies.registry import StrategyRegistry

# Initialize components
strategy_registry = StrategyRegistry()
model_router = ModelRouter()

# Create orchestrator
orchestrator = PythonWorkflowOrchestrator(
    strategy_registry=strategy_registry,
    model_router=model_router,
    max_concurrent_agents=5,
)

# Analyze spreadsheet
result = await orchestrator.analyze_spreadsheet(
    file_path=Path("financial_model.xlsx"),
    token_budget=100000,  # 100K tokens
)

if result.success:
    print(f"Analysis complete: {result.value['analysis_summary']}")
else:
    print(f"Analysis failed: {result.error}")
```

### Custom Workflow

```python
from spreadsheet_analyzer.notebook_llm.orchestration import (
    BaseOrchestrator,
    WorkflowStep,
    StepType,
)

class CustomOrchestrator(BaseOrchestrator):
    def _setup_workflow(self):
        self.workflow_steps = [
            WorkflowStep(
                name="custom_analysis",
                step_type=StepType.LLM_ANALYSIS,
                description="Custom analysis step",
                token_budget_percentage=0.5,
            ),
        ]
    
    async def execute_step(self, step, context):
        # Custom step implementation
        pass
```

### Model Configuration

```python
from spreadsheet_analyzer.notebook_llm.orchestration import (
    ModelRouter,
    ModelConfig,
    ModelProvider,
    ModelTier,
)

router = ModelRouter()

# Register custom model
custom_model = ModelConfig(
    name="custom-llm",
    provider=ModelProvider.HUGGINGFACE,
    tier=ModelTier.MEDIUM,
    max_tokens=4096,
    cost_per_1k_tokens=0.001,
    cost_per_1k_completion_tokens=0.002,
)

# Estimate costs
estimates = router.estimate_task_cost(
    task_type="sheet_analysis",
    estimated_input_tokens=5000,
    estimated_output_tokens=2000,
)
```

## Workflow Steps

### Step Configuration

Each workflow step has:

- **name**: Unique identifier
- **step_type**: Category of step (deterministic, LLM, validation, etc.)
- **dependencies**: Steps that must complete first
- **token_budget_percentage**: Portion of total budget
- **timeout_seconds**: Maximum execution time
- **retry_count**: Number of retry attempts

### Token Budget Allocation

Default allocation:

- Deterministic Analysis: 0% (no LLM)
- Complexity Assessment: 5%
- Sheet Analysis: 60%
- Relationship Analysis: 20%
- Validation: 10%
- Synthesis: 5%

## Model Selection Strategy

The model router selects models based on:

1. **Task Type** - Routing rules for specific tasks
1. **Complexity** - Higher complexity → larger models
1. **Cost Constraints** - Respect budget limits
1. **Availability** - Fallback options

## Error Handling

The orchestrator handles:

- **Transient Errors** - Automatic retry with backoff
- **Budget Exhaustion** - Graceful degradation
- **Model Failures** - Fallback to alternative models
- **Timeout** - Step cancellation and partial results

## Future Enhancements

1. **YAML Workflow Engine** - Declarative workflow configuration
1. **Dynamic Step Generation** - Create steps based on workbook structure
1. **Adaptive Budget Allocation** - Adjust budgets based on complexity
1. **Multi-Model Ensembles** - Combine results from multiple models
1. **Streaming Results** - Real-time progress updates

## Testing

Run the example script to see the orchestration in action:

```bash
uv run python -m spreadsheet_analyzer.notebook_llm.orchestration.example_usage
```

This will demonstrate:

- Model routing and cost estimation
- Workflow step configuration
- Full analysis pipeline (if test file exists)
