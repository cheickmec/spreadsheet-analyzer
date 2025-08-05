#!/usr/bin/env python3
"""Test LangGraph state management behavior."""

import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class TestState(TypedDict, total=False):
    counter: int
    data: str
    preserved: str


async def node_a(state: TestState) -> dict[str, Any]:
    """Node A increments counter and sets data."""
    print(f"Node A - Input state: {state}")
    result = {
        "counter": state.get("counter", 0) + 1,
        "data": "from_node_a",
    }
    print(f"Node A - Returning: {result}")
    return result


async def node_b(state: TestState) -> dict[str, Any]:
    """Node B sets data but should preserve counter."""
    print(f"Node B - Input state: {state}")
    # If we don't return counter, it gets lost!
    result = {
        "data": "from_node_b",
        # "counter": state.get("counter", 0),  # Uncommenting this preserves it
    }
    print(f"Node B - Returning: {result}")
    return result


async def test_state_preservation():
    """Test how LangGraph handles state updates."""

    # Build simple graph
    builder = StateGraph(TestState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)

    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)

    graph = builder.compile()

    # Run with initial state
    initial_state = TestState(counter=0, data="initial", preserved="should_remain")
    final_state = await graph.ainvoke(initial_state)

    print(f"\nInitial state: {initial_state}")
    print(f"Final state: {final_state}")
    print(f"Counter preserved? {final_state.get('counter') == 1}")
    print(f"Preserved field? {final_state.get('preserved') == 'should_remain'}")


if __name__ == "__main__":
    asyncio.run(test_state_preservation())
