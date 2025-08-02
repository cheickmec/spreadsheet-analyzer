"""Example usage of the multi-agent system.

This file demonstrates how to use the agent architecture
for collaborative spreadsheet analysis.

CLAUDE-KNOWLEDGE: This example shows common patterns for
multi-agent coordination in spreadsheet analysis.
"""

from typing import Any
from uuid import uuid4

from ..core.types import Result
from .types import Task, AgentState
from .core import create_echo_agent
from .communication import create_message_bus, scatter_gather
from .coordination import (
    create_coordinator,
    SequentialStrategy,
    ParallelStrategy,
    MapReduceStrategy,
)
from .spreadsheet_agents import create_spreadsheet_analysis_team


def example_basic_agent_communication():
    """Basic example of agent communication."""
    print("=== Basic Agent Communication ===\n")
    
    # Create agents
    agent1 = create_echo_agent("agent1")
    agent2 = create_echo_agent("agent2")
    
    # Create message bus
    bus = create_message_bus([agent1, agent2])
    
    # Send a message
    from .types import AgentMessage
    message = AgentMessage.create(
        sender=agent1.id,
        receiver=agent2.id,
        content="Hello from agent1!"
    )
    
    # Queue message
    bus = bus.send(message)
    
    # Process messages
    agent_states = {
        agent1.id: AgentState(agent_id=agent1.id, status="idle"),
        agent2.id: AgentState(agent_id=agent2.id, status="idle")
    }
    
    results = bus.process_all(agent_states)
    
    for result in results:
        if result.is_ok():
            response = result.unwrap()
            print(f"Response: {response.content}")
        else:
            print(f"Error: {result.unwrap_err()}")


def example_spreadsheet_analysis_team():
    """Example of specialized spreadsheet analysis agents."""
    print("\n=== Spreadsheet Analysis Team ===\n")
    
    # Create team of specialized agents
    team = create_spreadsheet_analysis_team()
    
    # Sample spreadsheet data
    sample_data = {
        "cells": [
            {"location": "Sheet1!A1", "content": "Product", "type": "value"},
            {"location": "Sheet1!B1", "content": "Price", "type": "value"},
            {"location": "Sheet1!C1", "content": "Tax", "type": "value"},
            {"location": "Sheet1!D1", "content": "Total", "type": "value"},
            {"location": "Sheet1!A2", "content": "Widget", "type": "value"},
            {"location": "Sheet1!B2", "content": 100, "type": "value"},
            {"location": "Sheet1!C2", "content": "=B2*0.08", "type": "formula"},
            {"location": "Sheet1!D2", "content": "=B2+C2", "type": "formula"},
            {"location": "Sheet1!A3", "content": "Gadget", "type": "value"},
            {"location": "Sheet1!B3", "content": 150, "type": "value"},
            {"location": "Sheet1!C3", "content": "=B3*0.08", "type": "formula"},
            {"location": "Sheet1!D3", "content": "=B3+C3", "type": "formula"},
        ]
    }
    
    # Test each agent
    for agent_name, agent in team.items():
        print(f"\n--- Testing {agent_name} ---")
        
        from .types import AgentMessage
        message = AgentMessage.create(
            sender=AgentMessage.create(
                sender=team["summary_generator"].id,
                receiver=agent.id,
                content=sample_data
            ).sender,
            receiver=agent.id,
            content=sample_data
        )
        
        state = AgentState(agent_id=agent.id, status="processing")
        result = agent.process(message, state)
        
        if result.is_ok():
            response = result.unwrap()
            print(f"Analysis result: {response.content}")
        else:
            print(f"Error: {result.unwrap_err()}")


def example_sequential_coordination():
    """Example of sequential task coordination."""
    print("\n=== Sequential Coordination ===\n")
    
    # Create agents
    team = create_spreadsheet_analysis_team()
    agents = list(team.values())
    
    # Create coordinator with sequential strategy
    strategy = SequentialStrategy(
        agent_sequence=tuple(agent.id for agent in [
            team["data_validator"],
            team["formula_analyzer"],
            team["pattern_detector"],
            team["summary_generator"]
        ])
    )
    
    coordinator = create_coordinator(agents, {"sequential": strategy})
    
    # Create task
    task = Task.create(
        name="analyze_spreadsheet",
        description="Complete spreadsheet analysis",
        input_data={
            "cells": [
                {"location": f"Sheet1!A{i}", "content": i*10, "type": "value"}
                for i in range(1, 11)
            ] + [
                {"location": f"Sheet1!B{i}", "content": f"=A{i}*2", "type": "formula"}
                for i in range(1, 11)
            ]
        }
    )
    
    # Execute coordination
    agent_states = {agent.id: AgentState(agent_id=agent.id, status="idle") for agent in agents}
    result = coordinator.coordinate(task, "sequential", agent_states)
    
    if result.is_ok():
        task_result = result.unwrap()
        print(f"Task completed: {task_result.status}")
        print(f"Result: {task_result.result}")
    else:
        print(f"Coordination failed: {result.unwrap_err()}")


def example_parallel_coordination():
    """Example of parallel task coordination."""
    print("\n=== Parallel Coordination ===\n")
    
    # Create agents
    team = create_spreadsheet_analysis_team()
    
    # Create coordinator with parallel strategy
    strategy = ParallelStrategy(
        agent_ids=(
            team["formula_analyzer"].id,
            team["pattern_detector"].id,
            team["data_validator"].id
        ),
        aggregator_id=team["summary_generator"].id
    )
    
    agents = list(team.values())
    coordinator = create_coordinator(agents, {"parallel": strategy})
    
    # Create task with more complex data
    task = Task.create(
        name="parallel_analysis",
        description="Analyze multiple aspects in parallel",
        input_data={
            "cells": [
                {"location": f"Sales!A{i}", "content": f"Product{i}", "type": "value"}
                for i in range(1, 21)
            ] + [
                {"location": f"Sales!B{i}", "content": i * 100, "type": "value"}
                for i in range(1, 21)
            ] + [
                {"location": f"Sales!C{i}", "content": f"=B{i}*0.15", "type": "formula"}
                for i in range(1, 21)
            ]
        }
    )
    
    # Execute coordination
    agent_states = {agent.id: AgentState(agent_id=agent.id, status="idle") for agent in agents}
    result = coordinator.coordinate(task, "parallel", agent_states)
    
    if result.is_ok():
        task_result = result.unwrap()
        print(f"Parallel task completed: {task_result.status}")
    else:
        print(f"Parallel coordination failed: {result.unwrap_err()}")


def example_scatter_gather_pattern():
    """Example of scatter-gather communication pattern."""
    print("\n=== Scatter-Gather Pattern ===\n")
    
    # Create analysis agents
    team = create_spreadsheet_analysis_team()
    
    # Sender is the coordinator
    sender = team["summary_generator"]
    
    # Receivers are analysis agents
    receivers = [
        team["formula_analyzer"],
        team["pattern_detector"],
        team["data_validator"]
    ]
    
    # Request content
    request_data = {
        "cells": [
            {"location": "Data!A1", "content": "Total", "type": "value"},
            {"location": "Data!B1", "content": "=SUM(B2:B10)", "type": "formula"},
            {"location": "Data!A2", "content": "Item1", "type": "value"},
            {"location": "Data!B2", "content": 100, "type": "value"},
        ]
    }
    
    # Define aggregator function
    def aggregate_analyses(responses):
        """Aggregate multiple analysis results."""
        aggregated = {
            "combined_analysis": {},
            "all_findings": []
        }
        
        for response in responses:
            content = response.content
            # Combine results based on agent type
            if "total_formulas" in content:
                aggregated["combined_analysis"]["formulas"] = content
            elif "patterns" in content:
                aggregated["combined_analysis"]["patterns"] = content
            elif "valid" in content:
                aggregated["combined_analysis"]["validation"] = content
        
        return aggregated
    
    # Execute scatter-gather
    sender_state = AgentState(agent_id=sender.id, status="coordinating")
    receiver_states = {
        agent.id: AgentState(agent_id=agent.id, status="idle")
        for agent in receivers
    }
    
    result = scatter_gather(
        sender=sender,
        receivers=receivers,
        request_content=request_data,
        sender_state=sender_state,
        receiver_states=receiver_states,
        aggregator=aggregate_analyses
    )
    
    if result.is_ok():
        aggregated = result.unwrap()
        print(f"Scatter-gather completed successfully")
        print(f"Combined analysis: {aggregated}")
    else:
        print(f"Scatter-gather failed: {result.unwrap_err()}")


def example_context_optimization_agent():
    """Example of using context optimization agent."""
    print("\n=== Context Optimization Agent ===\n")
    
    # Create context optimizer
    optimizer = create_spreadsheet_analysis_team()["context_optimizer"]
    
    # Large dataset that needs optimization
    large_data = {
        "cells": [
            {
                "location": f"Sheet1!{chr(65 + col)}{row}",
                "content": f"Data_{col}_{row}" if row > 1 else f"Column_{col}",
                "type": "value"
            }
            for row in range(1, 101)
            for col in range(0, 10)
        ],
        "query": "summarize sales trends",
        "token_budget": 2000,
        "model": "gpt-3.5-turbo"
    }
    
    # Request optimization
    from .types import AgentMessage
    message = AgentMessage.create(
        sender=AgentMessage.create(
            sender=optimizer.id,
            receiver=optimizer.id,
            content=large_data
        ).sender,
        receiver=optimizer.id,
        content=large_data
    )
    
    state = AgentState(agent_id=optimizer.id, status="optimizing")
    result = optimizer.process(message, state)
    
    if result.is_ok():
        response = result.unwrap()
        optimization_info = response.content["optimization_info"]
        print(f"Original cells: {optimization_info['original_cell_count']}")
        print(f"Optimized cells: {optimization_info['optimized_cell_count']}")
        print(f"Token count: {optimization_info['token_count']}/{optimization_info['token_budget']}")
        print(f"Reduction: {optimization_info['reduction_percentage']:.1f}%")
    else:
        print(f"Optimization failed: {result.unwrap_err()}")


if __name__ == "__main__":
    print("=== Multi-Agent System Examples ===\n")
    
    example_basic_agent_communication()
    example_spreadsheet_analysis_team()
    example_sequential_coordination()
    example_parallel_coordination()
    example_scatter_gather_pattern()
    example_context_optimization_agent()
    
    print("\n=== Examples Complete ===")