"""Agent communication protocols and message passing.

This module provides pure functional implementations for agent
communication patterns including request-response, publish-subscribe,
and broadcast messaging.

CLAUDE-KNOWLEDGE: Communication patterns are critical for multi-agent
systems. This module implements common patterns functionally.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.errors import AgentError
from ..core.types import Result, err, ok
from .types import (
    Agent,
    AgentId,
    AgentMessage,
    AgentState,
)


@dataclass(frozen=True)
class MessageBus:
    """Immutable message bus for agent communication.

    Manages message routing and delivery between agents.
    """

    agents: dict[AgentId, Agent]
    subscriptions: dict[str, tuple[AgentId, ...]]
    pending_messages: tuple[AgentMessage, ...]

    def send(self, message: AgentMessage) -> "MessageBus":
        """Add a message to the pending queue.

        Returns a new MessageBus with the message queued.
        """
        from dataclasses import replace

        return replace(self, pending_messages=(*self.pending_messages, message))

    def broadcast(self, sender: AgentId, content: Any, topic: str) -> "MessageBus":
        """Broadcast a message to all subscribers of a topic.

        Returns a new MessageBus with messages queued for all subscribers.
        """
        subscribers = self.subscriptions.get(topic, ())
        messages = []

        for subscriber_id in subscribers:
            if subscriber_id != sender:  # Don't send to self
                msg = AgentMessage.create(
                    sender=sender, receiver=subscriber_id, content=content, metadata={"topic": topic, "broadcast": True}
                )
                messages.append(msg)

        from dataclasses import replace

        return replace(self, pending_messages=self.pending_messages + tuple(messages))

    def subscribe(self, agent_id: AgentId, topic: str) -> "MessageBus":
        """Subscribe an agent to a topic.

        Returns a new MessageBus with updated subscriptions.
        """
        current_subs = self.subscriptions.get(topic, ())
        if agent_id not in current_subs:
            new_subs = (*current_subs, agent_id)
            from dataclasses import replace

            return replace(self, subscriptions={**self.subscriptions, topic: new_subs})
        return self

    def unsubscribe(self, agent_id: AgentId, topic: str) -> "MessageBus":
        """Unsubscribe an agent from a topic.

        Returns a new MessageBus with updated subscriptions.
        """
        current_subs = self.subscriptions.get(topic, ())
        new_subs = tuple(s for s in current_subs if s != agent_id)

        if new_subs:
            from dataclasses import replace

            return replace(self, subscriptions={**self.subscriptions, topic: new_subs})
        else:
            # Remove topic if no subscribers
            new_subscriptions = {k: v for k, v in self.subscriptions.items() if k != topic}
            from dataclasses import replace

            return replace(self, subscriptions=new_subscriptions)

    def process_next(
        self, agent_states: dict[AgentId, AgentState]
    ) -> tuple["MessageBus", list[Result[AgentMessage, AgentError]]]:
        """Process the next pending message.

        Returns a new MessageBus and the processing results.
        """
        if not self.pending_messages:
            return self, []

        # Take first message
        message, *rest = self.pending_messages

        # Find target agent
        target_agent = self.agents.get(message.receiver)
        if not target_agent:
            error_result = err(AgentError(f"Agent not found: {message.receiver}"))
            from dataclasses import replace

            return replace(self, pending_messages=tuple(rest)), [error_result]

        # Get agent state
        agent_state = agent_states.get(message.receiver, AgentState(agent_id=message.receiver, status="idle"))

        # Process message
        result = target_agent.process(message, agent_state)

        # Update message bus
        new_bus = self
        if result.is_ok():
            response = result.unwrap()
            # Queue response if it's not a terminal message
            if response.receiver != response.sender:
                new_bus = new_bus.send(response)

        from dataclasses import replace

        return replace(new_bus, pending_messages=tuple(rest)), [result]

    def process_all(
        self, agent_states: dict[AgentId, AgentState], max_iterations: int = 100
    ) -> list[Result[AgentMessage, AgentError]]:
        """Process all pending messages.

        Returns list of all processing results.
        """
        results = []
        current_bus = self
        iterations = 0

        while current_bus.pending_messages and iterations < max_iterations:
            current_bus, batch_results = current_bus.process_next(agent_states)
            results.extend(batch_results)
            iterations += 1

        return results


@dataclass(frozen=True)
class DirectRouter:
    """Direct message router - routes to specific agent by ID."""

    def route(self, message: AgentMessage, agents: dict[AgentId, Agent]) -> Result[Agent, AgentError]:
        """Route message directly to receiver."""
        agent = agents.get(message.receiver)
        if agent:
            return ok(agent)
        return err(AgentError(f"Agent not found: {message.receiver}"))


@dataclass(frozen=True)
class CapabilityRouter:
    """Routes messages based on agent capabilities."""

    capability_map: dict[str, AgentId]

    def route(self, message: AgentMessage, agents: dict[AgentId, Agent]) -> Result[Agent, AgentError]:
        """Route based on required capability in message metadata."""
        required_capability = message.metadata.get("required_capability")
        if not required_capability:
            return err(AgentError("No required_capability in message metadata"))

        agent_id = self.capability_map.get(required_capability)
        if not agent_id:
            return err(AgentError(f"No agent with capability: {required_capability}"))

        agent = agents.get(agent_id)
        if agent:
            return ok(agent)
        return err(AgentError(f"Agent not found: {agent_id}"))


@dataclass(frozen=True)
class RoundRobinRouter:
    """Routes messages in round-robin fashion among agents."""

    agent_ids: tuple[AgentId, ...]
    current_index: int = 0

    def route(self, _message: AgentMessage, agents: dict[AgentId, Agent]) -> Result[Agent, AgentError]:
        """Route to next agent in round-robin order."""
        if not self.agent_ids:
            return err(AgentError("No agents available for routing"))

        agent_id = self.agent_ids[self.current_index]
        agent = agents.get(agent_id)

        if agent:
            # Update index for next routing (in practice, this would be stored)
            return ok(agent)
        return err(AgentError(f"Agent not found: {agent_id}"))

    def next_router(self) -> "RoundRobinRouter":
        """Get router with updated index."""
        from dataclasses import replace

        next_index = (self.current_index + 1) % len(self.agent_ids)
        return replace(self, current_index=next_index)


# Communication patterns


def request_response(
    sender: Agent,
    receiver: Agent,
    request_content: Any,
    sender_state: AgentState,
    receiver_state: AgentState,
    timeout_ms: int = 5000,
) -> Result[AgentMessage, AgentError]:
    """Execute a request-response pattern between two agents.

    Args:
        sender: Agent sending the request
        receiver: Agent receiving the request
        request_content: Content of the request
        sender_state: Current state of sender
        receiver_state: Current state of receiver
        timeout_ms: Timeout in milliseconds (not enforced in pure function)

    Returns:
        Response message or error
    """
    # Create request message
    request = AgentMessage.create(
        sender=sender.id,
        receiver=receiver.id,
        content=request_content,
        metadata={"pattern": "request_response", "timeout_ms": timeout_ms},
    )

    # Process request
    response_result = receiver.process(request, receiver_state)

    return response_result


def scatter_gather(
    sender: Agent,
    receivers: list[Agent],
    request_content: Any,
    sender_state: AgentState,
    receiver_states: dict[AgentId, AgentState],
    aggregator: Callable[[list[AgentMessage]], Any],
) -> Result[Any, AgentError]:
    """Execute scatter-gather pattern.

    Sends request to multiple agents and aggregates responses.

    Args:
        sender: Agent sending requests
        receivers: Agents to receive requests
        request_content: Content to send
        sender_state: Sender's state
        receiver_states: States of all receivers
        aggregator: Function to aggregate responses

    Returns:
        Aggregated result or error
    """
    responses = []
    errors = []

    for receiver in receivers:
        request = AgentMessage.create(
            sender=sender.id, receiver=receiver.id, content=request_content, metadata={"pattern": "scatter_gather"}
        )

        receiver_state = receiver_states.get(receiver.id, AgentState(agent_id=receiver.id, status="idle"))

        result = receiver.process(request, receiver_state)

        if result.is_ok():
            responses.append(result.unwrap())
        else:
            errors.append(result.unwrap_err())

    if not responses:
        return err(AgentError(f"All receivers failed: {errors}"))

    try:
        aggregated = aggregator(responses)
        return ok(aggregated)
    except Exception as e:
        return err(AgentError(f"Aggregation failed: {e!s}"))


def chain_of_responsibility(
    initiator: Agent,
    chain: list[Agent],
    request_content: Any,
    agent_states: dict[AgentId, AgentState],
    can_handle: Callable[[Agent, Any], bool],
) -> Result[AgentMessage, AgentError]:
    """Execute chain of responsibility pattern.

    Passes request through chain until an agent handles it.

    Args:
        initiator: Agent initiating the chain
        chain: Ordered list of agents in chain
        request_content: Content to process
        agent_states: States of all agents
        can_handle: Function to check if agent can handle request

    Returns:
        Response from handling agent or error
    """
    for i, agent in enumerate(chain):
        if can_handle(agent, request_content):
            # This agent can handle it
            sender_id = chain[i - 1].id if i > 0 else initiator.id

            request = AgentMessage.create(
                sender=sender_id,
                receiver=agent.id,
                content=request_content,
                metadata={"pattern": "chain_of_responsibility", "chain_position": i},
            )

            agent_state = agent_states.get(agent.id, AgentState(agent_id=agent.id, status="idle"))

            return agent.process(request, agent_state)

    return err(AgentError("No agent in chain could handle request"))


# Helper functions


def create_message_bus(agents: list[Agent]) -> MessageBus:
    """Create a new message bus with agents."""
    agent_dict = {agent.id: agent for agent in agents}
    return MessageBus(agents=agent_dict, subscriptions={}, pending_messages=())


def create_capability_router(agents: list[Agent]) -> CapabilityRouter:
    """Create router based on agent capabilities."""
    capability_map = {}

    for agent in agents:
        for capability in agent.capabilities:
            # Map first agent for each capability
            if capability.name not in capability_map:
                capability_map[capability.name] = agent.id

    return CapabilityRouter(capability_map=capability_map)
