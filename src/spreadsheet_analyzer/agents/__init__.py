"""Agent system for multi-agent spreadsheet analysis.

This package provides a functional multi-agent architecture for
analyzing spreadsheets, with specialized agents for different tasks.
"""

from .types import (
    Agent,
    AgentCapability,
    AgentId,
    AgentMemory,
    AgentMessage,
    AgentResponse,
    AgentState,
    CoordinationPlan,
    CoordinationStep,
    MessageRouter,
    Task,
    TaskResult,
)

__all__ = [
    # Core types
    "Agent",
    "AgentCapability",
    "AgentId",
    "AgentMemory",
    "AgentMessage",
    "AgentResponse",
    "AgentState",
    # Coordination types
    "CoordinationPlan",
    "CoordinationStep",
    "MessageRouter",
    # Task types
    "Task",
    "TaskResult",
]
