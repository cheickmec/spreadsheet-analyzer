"""Base classes and protocols for the orchestration layer.

This module provides the foundational abstractions for workflow orchestration,
including base orchestrator classes, workflow step abstractions, and error
recovery mechanisms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from spreadsheet_analyzer.typing import Failure, Result, Success


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class StepType(Enum):
    """Types of workflow steps."""

    DETERMINISTIC_ANALYSIS = "deterministic_analysis"
    AGENT_CREATION = "agent_creation"
    LLM_ANALYSIS = "llm_analysis"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    RESULT_GENERATION = "result_generation"


@dataclass
class WorkflowStep:
    """Represents a single step in the analysis workflow."""

    name: str
    step_type: StepType
    description: str
    required: bool = True
    dependencies: list[str] = field(default_factory=list)
    token_budget_percentage: float = 0.0  # Percentage of total budget
    timeout_seconds: int = 300  # Default 5 minutes
    retry_count: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from a workflow step execution."""

    step_name: str
    status: WorkflowStatus
    data: Any = None
    error: str | None = None
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Context maintained throughout workflow execution."""

    workbook_path: Path
    deterministic_results: dict[str, Any] = field(default_factory=dict)
    agent_results: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, StepResult] = field(default_factory=dict)
    total_token_budget: int = 100000
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step_result(self, result: StepResult) -> None:
        """Add a step result to the context."""
        self.step_results[result.step_name] = result
        if result.tokens_used:
            self.tokens_used += result.tokens_used

    def get_remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.total_token_budget - self.tokens_used)

    def get_step_token_budget(self, step: WorkflowStep) -> int:
        """Calculate token budget for a specific step."""
        if step.token_budget_percentage > 0:
            allocated = int(self.total_token_budget * step.token_budget_percentage)
            return min(allocated, self.get_remaining_tokens())
        return self.get_remaining_tokens()


class WorkflowRecoveryStrategy(Protocol):
    """Protocol for workflow recovery strategies."""

    def can_recover(self, step: WorkflowStep, error: Exception, attempt: int) -> bool:
        """Check if recovery is possible for the given error."""
        ...

    def recover(self, context: WorkflowContext, step: WorkflowStep, error: Exception) -> Any:
        """Attempt to recover from the error."""
        ...


class DefaultRecoveryStrategy:
    """Default recovery strategy with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def can_recover(self, step: WorkflowStep, error: Exception, attempt: int) -> bool:
        """Check if recovery is possible."""
        # Don't retry if we've exceeded max attempts
        if attempt >= min(step.retry_count, self.max_retries):
            return False

        # Retry on transient errors
        transient_errors = (
            TimeoutError,
            ConnectionError,
            # Add more transient error types as needed
        )
        return isinstance(error, transient_errors)

    def recover(self, context: WorkflowContext, step: WorkflowStep, error: Exception) -> Any:
        """Simple recovery - just wait and retry."""
        import time

        # Exponential backoff
        delay = self.base_delay * (
            2
            ** (
                len(
                    [
                        r
                        for r in context.step_results.values()
                        if r.step_name == step.name and r.status == WorkflowStatus.FAILED
                    ]
                )
            )
        )
        time.sleep(min(delay, 30))  # Cap at 30 seconds
        return None


class BaseOrchestrator(ABC):
    """Abstract base class for workflow orchestrators."""

    def __init__(
        self,
        recovery_strategy: WorkflowRecoveryStrategy | None = None,
        max_concurrent_agents: int = 10,
    ):
        """Initialize the orchestrator.

        Args:
            recovery_strategy: Strategy for recovering from errors
            max_concurrent_agents: Maximum number of concurrent agents
        """
        self.recovery_strategy = recovery_strategy or DefaultRecoveryStrategy()
        self.max_concurrent_agents = max_concurrent_agents
        self.workflow_steps: list[WorkflowStep] = []
        self._setup_workflow()

    @abstractmethod
    def _setup_workflow(self) -> None:
        """Set up the workflow steps.

        Subclasses should populate self.workflow_steps.
        """
        pass

    @abstractmethod
    async def execute_step(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
    ) -> StepResult:
        """Execute a single workflow step.

        Args:
            step: The step to execute
            context: Current workflow context

        Returns:
            Result of the step execution
        """
        pass

    async def execute_workflow(
        self,
        workbook_path: Path,
        total_token_budget: int = 100000,
    ) -> Result[dict[str, Any]]:
        """Execute the complete workflow.

        Args:
            workbook_path: Path to the Excel file
            total_token_budget: Total token budget for analysis

        Returns:
            Success with final results or Failure with error details
        """
        context = WorkflowContext(
            workbook_path=workbook_path,
            total_token_budget=total_token_budget,
        )

        try:
            # Execute steps in order, respecting dependencies
            for step in self._get_execution_order():
                if not self._can_execute_step(step, context):
                    continue

                result = await self._execute_step_with_recovery(step, context)
                context.add_step_result(result)

                if result.status == WorkflowStatus.FAILED:
                    return Failure(
                        error=f"Step '{step.name}' failed: {result.error}",
                        details={"step_results": context.step_results},
                    )

            # Synthesize final results
            final_results = self._synthesize_results(context)
            return Success(final_results)

        except Exception as e:
            return Failure(error=f"Workflow execution failed: {e!s}", details={"step_results": context.step_results})

    async def _execute_step_with_recovery(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
    ) -> StepResult:
        """Execute a step with error recovery."""
        attempt = 0
        last_error = None

        while attempt < step.retry_count:
            try:
                return await self.execute_step(step, context)
            except Exception as e:
                last_error = e
                attempt += 1

                if self.recovery_strategy.can_recover(step, e, attempt):
                    await self.recovery_strategy.recover(context, step, e)
                else:
                    break

        # All attempts failed
        return StepResult(
            step_name=step.name,
            status=WorkflowStatus.FAILED,
            error=str(last_error) if last_error else "Unknown error",
        )

    def _get_execution_order(self) -> list[WorkflowStep]:
        """Get steps in execution order based on dependencies.

        Uses topological sort to respect dependencies.
        """
        # For now, return steps as-is (assumes they're already ordered)
        # TODO: Implement proper topological sort
        return self.workflow_steps

    def _can_execute_step(self, step: WorkflowStep, context: WorkflowContext) -> bool:
        """Check if a step can be executed based on dependencies."""
        for dep in step.dependencies:
            dep_result = context.step_results.get(dep)
            if not dep_result or dep_result.status != WorkflowStatus.COMPLETED:
                if step.required:
                    return False
        return True

    @abstractmethod
    def _synthesize_results(self, context: WorkflowContext) -> dict[str, Any]:
        """Synthesize final results from the workflow context.

        Args:
            context: Complete workflow context

        Returns:
            Final analysis results
        """
        pass


class ProgressMonitor(Protocol):
    """Protocol for monitoring workflow progress."""

    def on_step_start(self, step: WorkflowStep) -> None:
        """Called when a step starts."""
        ...

    def on_step_complete(self, step: WorkflowStep, result: StepResult) -> None:
        """Called when a step completes."""
        ...

    def on_workflow_complete(self, success: bool, results: Any) -> None:
        """Called when the workflow completes."""
        ...
