"""Type definitions for the agent system.

This module defines protocols and types used throughout the agent
architecture, following functional programming principles.

CLAUDE-KNOWLEDGE: Agents are defined as protocols (interfaces) rather
than base classes to allow for more flexible implementations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from ..core.errors import AgentError
from ..core.types import Option, Result


@dataclass(frozen=True)
class AgentId:
    """Immutable agent identifier."""

    value: str
    agent_type: str

    @classmethod
    def generate(cls, agent_type: str) -> "AgentId":
        """Generate a new unique agent ID."""
        return cls(value=f"{agent_type}_{uuid4().hex[:8]}", agent_type=agent_type)


@dataclass(frozen=True)
class AgentMessage:
    """Immutable message passed between agents."""

    id: UUID
    sender: AgentId
    receiver: AgentId
    content: Any
    timestamp: datetime
    correlation_id: UUID | None = None
    reply_to: UUID | None = None
    metadata: dict[str, Any] = None

    @classmethod
    def create(
        cls,
        sender: AgentId,
        receiver: AgentId,
        content: Any,
        correlation_id: UUID | None = None,
        reply_to: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AgentMessage":
        """Create a new message with generated ID and timestamp."""
        return cls(
            id=uuid4(),
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id or uuid4(),
            reply_to=reply_to,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class AgentCapability:
    """Description of what an agent can do."""

    name: str
    description: str
    input_type: type
    output_type: type
    metadata: dict[str, Any] = None


@dataclass(frozen=True)
class AgentState:
    """Immutable agent state."""

    agent_id: AgentId
    status: str  # "idle", "processing", "waiting", "error"
    current_task: UUID | None = None
    last_activity: datetime | None = None
    metadata: dict[str, Any] = None

    def with_status(self, status: str) -> "AgentState":
        """Create new state with updated status."""
        from dataclasses import replace

        return replace(self, status=status, last_activity=datetime.now())

    def with_task(self, task_id: UUID | None) -> "AgentState":
        """Create new state with updated task."""
        from dataclasses import replace

        return replace(self, current_task=task_id, last_activity=datetime.now())


class Agent(Protocol):
    """Protocol defining the agent interface.

    Agents process messages and return results. They should be
    implemented as pure functions or modules of pure functions.
    """

    @property
    def id(self) -> AgentId:
        """Get the agent's unique identifier."""
        ...

    @property
    def capabilities(self) -> tuple[AgentCapability, ...]:
        """Get the agent's capabilities."""
        ...

    def process(self, message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Process a message and return a response.

        This should be a pure function that takes the current state
        and returns a new message without side effects.
        """
        ...


class AgentMemory(Protocol):
    """Protocol for agent memory systems."""

    def store(self, agent_id: AgentId, key: str, value: Any) -> Result[None, AgentError]:
        """Store a value in memory."""
        ...

    def retrieve(self, agent_id: AgentId, key: str) -> Option[Any]:
        """Retrieve a value from memory."""
        ...

    def search(self, agent_id: AgentId, query: str) -> list[tuple[str, Any]]:
        """Search memory for relevant entries."""
        ...

    def clear(self, agent_id: AgentId) -> Result[None, AgentError]:
        """Clear all memory for an agent."""
        ...


class MessageRouter(Protocol):
    """Protocol for message routing between agents."""

    def route(self, message: AgentMessage, agents: dict[AgentId, Agent]) -> Result[Agent, AgentError]:
        """Route a message to the appropriate agent."""
        ...


@dataclass(frozen=True)
class AgentResponse:
    """Response from agent processing."""

    message: AgentMessage
    new_state: AgentState
    side_effects: list[Callable[[], Result[None, AgentError]]] = None

    def execute_effects(self) -> list[Result[None, AgentError]]:
        """Execute all side effects."""
        if not self.side_effects:
            return []
        return [effect() for effect in self.side_effects]


# Task-related types
@dataclass(frozen=True)
class Task:
    """Immutable task definition."""

    id: UUID
    name: str
    description: str
    input_data: Any
    created_at: datetime
    priority: int = 0
    metadata: dict[str, Any] = None

    @classmethod
    def create(
        cls, name: str, description: str, input_data: Any, priority: int = 0, metadata: dict[str, Any] | None = None
    ) -> "Task":
        """Create a new task with generated ID and timestamp."""
        return cls(
            id=uuid4(),
            name=name,
            description=description,
            input_data=input_data,
            created_at=datetime.now(),
            priority=priority,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class TaskResult:
    """Result of task execution."""

    task_id: UUID
    agent_id: AgentId
    status: str  # "success", "failure", "partial"
    result: Any
    completed_at: datetime
    error: AgentError | None = None
    metadata: dict[str, Any] = None


# Coordination types
@dataclass(frozen=True)
class CoordinationPlan:
    """Plan for coordinating multiple agents."""

    id: UUID
    task: Task
    steps: tuple["CoordinationStep", ...]
    created_at: datetime

    @classmethod
    def create(cls, task: Task, steps: list["CoordinationStep"]) -> "CoordinationPlan":
        """Create a new coordination plan."""
        return cls(id=uuid4(), task=task, steps=tuple(steps), created_at=datetime.now())


@dataclass(frozen=True)
class CoordinationStep:
    """Single step in a coordination plan."""

    id: UUID
    agent_id: AgentId
    action: str
    input_data: Any
    depends_on: tuple[UUID, ...] = ()
    timeout_seconds: int = 60

    @classmethod
    def create(
        cls,
        agent_id: AgentId,
        action: str,
        input_data: Any,
        depends_on: list[UUID] | None = None,
        timeout_seconds: int = 60,
    ) -> "CoordinationStep":
        """Create a new coordination step."""
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            action=action,
            input_data=input_data,
            depends_on=tuple(depends_on or []),
            timeout_seconds=timeout_seconds,
        )
