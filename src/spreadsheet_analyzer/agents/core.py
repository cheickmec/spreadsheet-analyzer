"""Core agent implementations and factories.

This module provides the foundational agent implementations
following functional programming principles.

CLAUDE-KNOWLEDGE: Agents are designed as composable units that
process messages purely functionally, enabling easy testing
and predictable behavior.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.errors import AgentError
from ..core.types import Result, err, ok
from .types import (
    Agent,
    AgentCapability,
    AgentId,
    AgentMessage,
    AgentState,
)


@dataclass(frozen=True)
class FunctionalAgent:
    """A functional implementation of the Agent protocol.

    This agent wraps a pure processing function and manages
    its capabilities and state functionally.
    """

    id: AgentId
    capabilities: tuple[AgentCapability, ...]
    process_fn: Callable[[AgentMessage, AgentState], Result[AgentMessage, AgentError]]

    def process(self, message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Process a message using the wrapped function."""
        return self.process_fn(message, state)


@dataclass(frozen=True)
class CompositeAgent:
    """An agent composed of multiple sub-agents.

    Routes messages to sub-agents based on capabilities.
    """

    id: AgentId
    sub_agents: tuple[Agent, ...]
    router: Callable[[AgentMessage, tuple[Agent, ...]], Agent | None]

    @property
    def capabilities(self) -> tuple[AgentCapability, ...]:
        """Aggregate capabilities from all sub-agents."""
        all_capabilities = []
        for agent in self.sub_agents:
            all_capabilities.extend(agent.capabilities)
        return tuple(all_capabilities)

    def process(self, message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Route message to appropriate sub-agent."""
        target_agent = self.router(message, self.sub_agents)

        if target_agent is None:
            return err(AgentError(f"No suitable agent found for message: {message.id}"))

        return target_agent.process(message, state)


@dataclass(frozen=True)
class StatefulAgent:
    """An agent that maintains internal state between messages.

    State is managed functionally through state transformations.
    """

    id: AgentId
    capabilities: tuple[AgentCapability, ...]
    initial_state: dict[str, Any]
    process_fn: Callable[
        [AgentMessage, AgentState, dict[str, Any]], tuple[Result[AgentMessage, AgentError], dict[str, Any]]
    ]

    def process(self, message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Process with state transformation."""
        # In a real implementation, state would be retrieved from memory
        # For now, we use initial state
        internal_state = self.initial_state

        result, new_internal_state = self.process_fn(message, state, internal_state)

        # In a real implementation, we would store new_internal_state
        return result


# Agent factory functions


def create_simple_agent(
    agent_type: str,
    capabilities: list[AgentCapability],
    process_fn: Callable[[AgentMessage, AgentState], Result[AgentMessage, AgentError]],
) -> FunctionalAgent:
    """Create a simple functional agent.

    Args:
        agent_type: Type identifier for the agent
        capabilities: List of agent capabilities
        process_fn: Pure function to process messages

    Returns:
        A new functional agent
    """
    return FunctionalAgent(id=AgentId.generate(agent_type), capabilities=tuple(capabilities), process_fn=process_fn)


def create_echo_agent(agent_type: str = "echo") -> FunctionalAgent:
    """Create an agent that echoes messages back.

    Useful for testing and debugging.
    """

    def echo_process(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        response = AgentMessage.create(
            sender=state.agent_id,
            receiver=message.sender,
            content=f"Echo: {message.content}",
            reply_to=message.id,
            correlation_id=message.correlation_id,
        )
        return ok(response)

    capabilities = [
        AgentCapability(name="echo", description="Echoes any message back to sender", input_type=Any, output_type=str)
    ]

    return create_simple_agent(agent_type, capabilities, echo_process)


def create_transform_agent(
    agent_type: str, transform_fn: Callable[[Any], Any], capability_name: str, description: str
) -> FunctionalAgent:
    """Create an agent that transforms message content.

    Args:
        agent_type: Type identifier for the agent
        transform_fn: Pure function to transform content
        capability_name: Name of the transformation capability
        description: Description of what the agent does

    Returns:
        A new transform agent
    """

    def transform_process(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        try:
            transformed = transform_fn(message.content)
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=transformed,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )
            return ok(response)
        except Exception as e:
            return err(AgentError(f"Transform failed: {e!s}"))

    capabilities = [AgentCapability(name=capability_name, description=description, input_type=Any, output_type=Any)]

    return create_simple_agent(agent_type, capabilities, transform_process)


def create_filter_agent(
    agent_type: str, filter_fn: Callable[[Any], bool], capability_name: str, description: str
) -> FunctionalAgent:
    """Create an agent that filters messages.

    Only forwards messages that pass the filter function.

    Args:
        agent_type: Type identifier for the agent
        filter_fn: Pure function to test if content should pass
        capability_name: Name of the filter capability
        description: Description of what the agent filters

    Returns:
        A new filter agent
    """

    def filter_process(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        if filter_fn(message.content):
            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=message.content,
                reply_to=message.id,
                correlation_id=message.correlation_id,
                metadata={"passed_filter": True},
            )
            return ok(response)
        else:
            return err(AgentError(f"Message filtered out: {message.id}"))

    capabilities = [AgentCapability(name=capability_name, description=description, input_type=Any, output_type=Any)]

    return create_simple_agent(agent_type, capabilities, filter_process)


def create_composite_agent(
    agent_type: str,
    sub_agents: list[Agent],
    router: Callable[[AgentMessage, tuple[Agent, ...]], Agent | None] | None = None,
) -> CompositeAgent:
    """Create a composite agent from sub-agents.

    Args:
        agent_type: Type identifier for the composite agent
        sub_agents: List of sub-agents to compose
        router: Optional custom routing function

    Returns:
        A new composite agent
    """
    if router is None:
        # Default router based on capability matching
        def default_router(message: AgentMessage, agents: tuple[Agent, ...]) -> Agent | None:
            # Simple routing based on metadata hint
            if "target_capability" in message.metadata:
                target_cap = message.metadata["target_capability"]
                for agent in agents:
                    for cap in agent.capabilities:
                        if cap.name == target_cap:
                            return agent

            # Return first agent as fallback
            return agents[0] if agents else None

        router = default_router

    return CompositeAgent(id=AgentId.generate(agent_type), sub_agents=tuple(sub_agents), router=router)


def create_pipeline_agent(agent_type: str, agents: list[Agent]) -> CompositeAgent:
    """Create an agent that processes messages through a pipeline.

    Each agent processes the output of the previous agent.

    Args:
        agent_type: Type identifier for the pipeline
        agents: Ordered list of agents in the pipeline

    Returns:
        A new pipeline agent
    """

    def pipeline_router(message: AgentMessage, agents: tuple[Agent, ...]) -> Agent | None:
        # Route to next agent in pipeline based on metadata
        current_index = message.metadata.get("pipeline_index", 0)
        if 0 <= current_index < len(agents):
            return agents[current_index]
        return None

    # Create wrapper agents that update pipeline index
    wrapped_agents = []
    for i, agent in enumerate(agents):
        original_process = agent.process

        def make_wrapper(index: int, original: Callable):
            def wrapped_process(msg: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
                result = original(msg, state)
                if result.is_ok():
                    response = result.unwrap()
                    # Update metadata for next stage
                    new_metadata = {**response.metadata, "pipeline_index": index + 1}
                    from dataclasses import replace

                    return ok(replace(response, metadata=new_metadata))
                return result

            return wrapped_process

        wrapped = FunctionalAgent(
            id=agent.id, capabilities=agent.capabilities, process_fn=make_wrapper(i, original_process)
        )
        wrapped_agents.append(wrapped)

    return create_composite_agent(agent_type, wrapped_agents, pipeline_router)
