"""Agent coordination and orchestration.

This module provides functional implementations for coordinating
multiple agents to accomplish complex tasks.

CLAUDE-KNOWLEDGE: Coordination is essential for multi-agent systems
to work together effectively. This module implements common
coordination patterns functionally.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from ..core.errors import AgentError
from ..core.types import Result, err, ok
from .types import (
    Agent,
    AgentId,
    AgentMessage,
    AgentState,
    CoordinationPlan,
    CoordinationStep,
    Task,
    TaskResult,
)


@dataclass(frozen=True)
class Coordinator:
    """Functional coordinator for multi-agent task execution.

    Manages task distribution and result aggregation.
    """

    id: AgentId
    agents: dict[AgentId, Agent]
    strategies: dict[str, "CoordinationStrategy"]

    def coordinate(
        self, task: Task, strategy_name: str, agent_states: dict[AgentId, AgentState]
    ) -> Result[TaskResult, AgentError]:
        """Coordinate task execution using specified strategy.

        Args:
            task: Task to execute
            strategy_name: Name of coordination strategy
            agent_states: Current states of all agents

        Returns:
            Task result or error
        """
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return err(AgentError(f"Unknown strategy: {strategy_name}"))

        # Create coordination plan
        plan_result = strategy.create_plan(task, self.agents)
        if plan_result.is_err():
            return err(plan_result.unwrap_err())

        plan = plan_result.unwrap()

        # Execute plan
        return self._execute_plan(plan, agent_states)

    def _execute_plan(
        self, plan: CoordinationPlan, agent_states: dict[AgentId, AgentState]
    ) -> Result[TaskResult, AgentError]:
        """Execute a coordination plan.

        Args:
            plan: Plan to execute
            agent_states: Current agent states

        Returns:
            Final task result or error
        """
        step_results: dict[UUID, Result[Any, AgentError]] = {}

        # Execute steps in dependency order
        for step in self._order_steps(plan.steps):
            # Check dependencies
            if not self._dependencies_met(step, step_results):
                return err(AgentError(f"Dependencies not met for step: {step.id}"))

            # Execute step
            result = self._execute_step(step, agent_states, step_results)
            step_results[step.id] = result

            if result.is_err():
                # Fail fast on errors
                return err(AgentError(f"Step {step.id} failed: {result.unwrap_err()}"))

        # Aggregate results
        final_result = self._aggregate_results(plan, step_results)

        return ok(
            TaskResult(
                task_id=plan.task.id,
                agent_id=self.id,
                status="success" if final_result.is_ok() else "failure",
                result=final_result.unwrap() if final_result.is_ok() else None,
                completed_at=datetime.now(),
                error=final_result.unwrap_err() if final_result.is_err() else None,
            )
        )

    def _order_steps(self, steps: tuple[CoordinationStep, ...]) -> list[CoordinationStep]:
        """Order steps based on dependencies (simple topological sort)."""
        ordered = []
        remaining = list(steps)
        completed_ids = set()

        while remaining:
            # Find steps with satisfied dependencies
            ready = [step for step in remaining if all(dep in completed_ids for dep in step.depends_on)]

            if not ready:
                # Circular dependency or missing steps
                break

            # Add ready steps to ordered list
            ordered.extend(ready)
            for step in ready:
                completed_ids.add(step.id)
                remaining.remove(step)

        return ordered

    def _dependencies_met(self, step: CoordinationStep, results: dict[UUID, Result[Any, AgentError]]) -> bool:
        """Check if all dependencies are successfully completed."""
        for dep_id in step.depends_on:
            if dep_id not in results:
                return False
            if results[dep_id].is_err():
                return False
        return True

    def _execute_step(
        self,
        step: CoordinationStep,
        agent_states: dict[AgentId, AgentState],
        previous_results: dict[UUID, Result[Any, AgentError]],
    ) -> Result[Any, AgentError]:
        """Execute a single coordination step."""
        agent = self.agents.get(step.agent_id)
        if not agent:
            return err(AgentError(f"Agent not found: {step.agent_id}"))

        # Prepare input with dependency results
        input_data = step.input_data
        if step.depends_on:
            dependency_data = {}
            for dep_id in step.depends_on:
                if dep_id in previous_results and previous_results[dep_id].is_ok():
                    dependency_data[str(dep_id)] = previous_results[dep_id].unwrap()
            input_data = {"original": input_data, "dependencies": dependency_data}

        # Create message for agent
        message = AgentMessage.create(
            sender=self.id,
            receiver=step.agent_id,
            content={"action": step.action, "data": input_data},
            metadata={"step_id": str(step.id), "timeout": step.timeout_seconds},
        )

        # Get agent state
        agent_state = agent_states.get(step.agent_id, AgentState(agent_id=step.agent_id, status="idle"))

        # Process message
        response = agent.process(message, agent_state)

        if response.is_ok():
            return ok(response.unwrap().content)
        else:
            return response

    def _aggregate_results(
        self, plan: CoordinationPlan, results: dict[UUID, Result[Any, AgentError]]
    ) -> Result[Any, AgentError]:
        """Aggregate step results into final result."""
        # Simple aggregation - return last step's result
        if plan.steps:
            last_step_id = plan.steps[-1].id
            if last_step_id in results:
                return results[last_step_id]

        # No steps or missing result
        return ok({"status": "completed", "steps": len(plan.steps)})


# Coordination strategies


class CoordinationStrategy:
    """Base protocol for coordination strategies."""

    def create_plan(self, task: Task, agents: dict[AgentId, Agent]) -> Result[CoordinationPlan, AgentError]:
        """Create a coordination plan for the task."""
        raise NotImplementedError


@dataclass(frozen=True)
class SequentialStrategy:
    """Execute task through agents sequentially."""

    agent_sequence: tuple[AgentId, ...]

    def create_plan(self, task: Task, agents: dict[AgentId, Agent]) -> Result[CoordinationPlan, AgentError]:
        """Create sequential execution plan."""
        steps = []
        previous_id = None

        for agent_id in self.agent_sequence:
            if agent_id not in agents:
                return err(AgentError(f"Agent not found in sequence: {agent_id}"))

            step = CoordinationStep.create(
                agent_id=agent_id,
                action="process",
                input_data=task.input_data,
                depends_on=[previous_id] if previous_id else [],
            )
            steps.append(step)
            previous_id = step.id

        return ok(CoordinationPlan.create(task, steps))


@dataclass(frozen=True)
class ParallelStrategy:
    """Execute task in parallel across multiple agents."""

    agent_ids: tuple[AgentId, ...]
    aggregator_id: AgentId

    def create_plan(self, task: Task, agents: dict[AgentId, Agent]) -> Result[CoordinationPlan, AgentError]:
        """Create parallel execution plan."""
        steps = []

        # Create parallel processing steps
        parallel_step_ids = []
        for agent_id in self.agent_ids:
            if agent_id not in agents:
                return err(AgentError(f"Agent not found: {agent_id}"))

            step = CoordinationStep.create(
                agent_id=agent_id,
                action="process",
                input_data=task.input_data,
                depends_on=[],  # No dependencies - can run in parallel
            )
            steps.append(step)
            parallel_step_ids.append(step.id)

        # Create aggregation step
        if self.aggregator_id not in agents:
            return err(AgentError(f"Aggregator not found: {self.aggregator_id}"))

        aggregation_step = CoordinationStep.create(
            agent_id=self.aggregator_id,
            action="aggregate",
            input_data={"task": task.name},
            depends_on=parallel_step_ids,  # Depends on all parallel steps
        )
        steps.append(aggregation_step)

        return ok(CoordinationPlan.create(task, steps))


@dataclass(frozen=True)
class MapReduceStrategy:
    """Map-reduce coordination pattern."""

    mapper_ids: tuple[AgentId, ...]
    reducer_id: AgentId
    partitioner: Callable[[Any], list[Any]]

    def create_plan(self, task: Task, agents: dict[AgentId, Agent]) -> Result[CoordinationPlan, AgentError]:
        """Create map-reduce execution plan."""
        steps = []

        # Partition input data
        try:
            partitions = self.partitioner(task.input_data)
        except Exception as e:
            return err(AgentError(f"Partitioning failed: {e!s}"))

        if len(partitions) > len(self.mapper_ids):
            return err(AgentError("More partitions than mappers"))

        # Create mapping steps
        map_step_ids = []
        for _, (partition, mapper_id) in enumerate(zip(partitions, self.mapper_ids, strict=False)):
            if mapper_id not in agents:
                return err(AgentError(f"Mapper not found: {mapper_id}"))

            step = CoordinationStep.create(agent_id=mapper_id, action="map", input_data=partition, depends_on=[])
            steps.append(step)
            map_step_ids.append(step.id)

        # Create reduce step
        if self.reducer_id not in agents:
            return err(AgentError(f"Reducer not found: {self.reducer_id}"))

        reduce_step = CoordinationStep.create(
            agent_id=self.reducer_id, action="reduce", input_data={"task": task.name}, depends_on=map_step_ids
        )
        steps.append(reduce_step)

        return ok(CoordinationPlan.create(task, steps))


@dataclass(frozen=True)
class HierarchicalStrategy:
    """Hierarchical decomposition strategy."""

    supervisor_id: AgentId
    worker_ids: tuple[AgentId, ...]
    decomposer: Callable[[Task], list[Task]]

    def create_plan(self, task: Task, agents: dict[AgentId, Agent]) -> Result[CoordinationPlan, AgentError]:
        """Create hierarchical execution plan."""
        steps = []

        # Supervisor decomposes task
        if self.supervisor_id not in agents:
            return err(AgentError(f"Supervisor not found: {self.supervisor_id}"))

        decompose_step = CoordinationStep.create(
            agent_id=self.supervisor_id, action="decompose", input_data=task.input_data, depends_on=[]
        )
        steps.append(decompose_step)

        # Workers process subtasks
        worker_step_ids = []
        for worker_id in self.worker_ids:
            if worker_id not in agents:
                return err(AgentError(f"Worker not found: {worker_id}"))

            step = CoordinationStep.create(
                agent_id=worker_id,
                action="process_subtask",
                input_data={"parent_task": task.id},
                depends_on=[decompose_step.id],
            )
            steps.append(step)
            worker_step_ids.append(step.id)

        # Supervisor aggregates results
        aggregate_step = CoordinationStep.create(
            agent_id=self.supervisor_id,
            action="aggregate_results",
            input_data={"task": task.name},
            depends_on=worker_step_ids,
        )
        steps.append(aggregate_step)

        return ok(CoordinationPlan.create(task, steps))


# Factory functions


def create_coordinator(agents: list[Agent], strategies: dict[str, CoordinationStrategy] | None = None) -> Coordinator:
    """Create a coordinator with default strategies."""
    if strategies is None:
        # Create default strategies
        agent_ids = tuple(agent.id for agent in agents)

        strategies = {
            "sequential": SequentialStrategy(agent_sequence=agent_ids),
            "parallel": ParallelStrategy(
                agent_ids=agent_ids[:-1] if len(agent_ids) > 1 else agent_ids,
                aggregator_id=agent_ids[-1] if agent_ids else AgentId("none", "none"),
            ),
        }

    return Coordinator(
        id=AgentId.generate("coordinator"), agents={agent.id: agent for agent in agents}, strategies=strategies
    )


def create_map_reduce_coordinator(
    mappers: list[Agent], reducer: Agent, partitioner: Callable[[Any], list[Any]]
) -> Coordinator:
    """Create a coordinator for map-reduce pattern."""
    all_agents = [*mappers, reducer]

    strategy = MapReduceStrategy(
        mapper_ids=tuple(m.id for m in mappers), reducer_id=reducer.id, partitioner=partitioner
    )

    return Coordinator(
        id=AgentId.generate("mapreduce_coordinator"),
        agents={agent.id: agent for agent in all_agents},
        strategies={"mapreduce": strategy},
    )
