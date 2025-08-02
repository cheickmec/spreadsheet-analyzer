# Multi-Agent System Architecture

This package implements a functional multi-agent architecture for intelligent spreadsheet analysis.

## Overview

The `agents` package provides:

- **Agent protocols** defining agent interfaces
- **Message passing** system for agent communication
- **Task management** for coordinating work
- **Memory systems** for agent state
- **Orchestration** for multi-agent collaboration

## Architecture

### Agent Protocol

Agents are defined as protocols (interfaces) rather than classes:

```python
from spreadsheet_analyzer.agents import Agent, AgentMessage, AgentState
from spreadsheet_analyzer.core import Result

class AnalystAgent:
    """Example agent implementation."""
    
    @property
    def id(self) -> AgentId:
        return AgentId.generate("analyst")
    
    @property
    def capabilities(self) -> tuple[AgentCapability, ...]:
        return (
            AgentCapability(
                name="analyze_formulas",
                description="Analyze Excel formulas",
                input_type=FormulaRequest,
                output_type=FormulaAnalysis
            ),
        )
    
    def process(self, message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Process message - pure function."""
        # Extract content, perform analysis, return response
        ...
```

### Message Passing

Agents communicate through immutable messages:

```python
from spreadsheet_analyzer.agents import AgentMessage, AgentId

# Create a message
message = AgentMessage.create(
    sender=agent1.id,
    receiver=agent2.id,
    content={"action": "analyze", "data": spreadsheet_data}
)

# Process through agent (pure function)
response = agent2.process(message, agent2_state)
```

### Task Coordination

Complex tasks are broken down and coordinated:

```python
from spreadsheet_analyzer.agents import Task, CoordinationPlan, CoordinationStep

# Define a task
task = Task.create(
    name="analyze_workbook",
    description="Complete workbook analysis",
    input_data={"file_path": "data.xlsx"}
)

# Create coordination plan
steps = [
    CoordinationStep.create(
        agent_id=structure_agent.id,
        action="analyze_structure",
        input_data=task.input_data
    ),
    CoordinationStep.create(
        agent_id=formula_agent.id,
        action="analyze_formulas",
        input_data=task.input_data,
        depends_on=[steps[0].id]  # Depends on structure
    )
]

plan = CoordinationPlan.create(task, steps)
```

## Agent Types (Future)

Planned specialized agents:

### 1. Structure Agent

- Analyzes workbook structure
- Identifies sheets, ranges, and tables
- Maps relationships

### 2. Formula Expert Agent

- Parses and validates formulas
- Detects circular references
- Suggests optimizations

### 3. Table Detector Agent

- Identifies multiple tables in sheets
- Determines table boundaries
- Extracts table metadata

### 4. Pattern Finder Agent

- Detects data patterns
- Identifies anomalies
- Suggests data types

### 5. Coordinator Agent

- Orchestrates other agents
- Manages task decomposition
- Aggregates results

## Memory Systems

Agents can have different memory types:

```python
from spreadsheet_analyzer.agents import AgentMemory

class EpisodicMemory:
    """Stores specific interactions."""
    
    def store(self, agent_id: AgentId, key: str, value: Any) -> Result[None, AgentError]:
        # Store in vector database
        ...
    
    def retrieve(self, agent_id: AgentId, key: str) -> Option[Any]:
        # Retrieve from storage
        ...
```

## Design Principles

1. **Protocol-Based** - Use protocols not base classes
1. **Pure Functions** - Agent logic is side-effect free
1. **Immutable Messages** - All communication is immutable
1. **Explicit Dependencies** - Clear task dependencies
1. **Composable** - Agents can be composed into teams

## Example: Multi-Agent Analysis

```python
from spreadsheet_analyzer.agents import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(agents=[
    structure_agent,
    formula_agent,
    pattern_agent
])

# Run analysis
result = orchestrator.analyze_workbook(
    file_path="complex_spreadsheet.xlsx",
    context={"focus": "formula_validation"}
)

# Result contains aggregated findings from all agents
```

## Communication Patterns

### Direct Messaging

Agent-to-agent communication for specific requests.

### Broadcast

One agent sends to all interested agents.

### Request-Response

Synchronous communication pattern.

### Publish-Subscribe

Asynchronous event-based communication.

## Best Practices

1. **Keep agents focused** - Single responsibility
1. **Use immutable state** - No shared mutable state
1. **Design for failure** - Handle communication failures
1. **Test in isolation** - Agents should be independently testable
1. **Document capabilities** - Clear agent interfaces

## Future Enhancements

- **Learning agents** that improve over time
- **Negotiation protocols** for resource allocation
- **Hierarchical teams** with supervisor agents
- **Dynamic agent creation** based on workload
- **Cross-language agents** via standard protocols
