"""Tests for notebook_llm orchestration components."""

from pathlib import Path

import pytest

from spreadsheet_analyzer.notebook_llm.orchestration.base import (
    BaseOrchestrator,
    DefaultRecoveryStrategy,
    StepResult,
    StepType,
    WorkflowContext,
    WorkflowStatus,
    WorkflowStep,
)
from spreadsheet_analyzer.type_definitions import Success


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_workflow_status_values(self):
        """Test workflow status values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"
        assert WorkflowStatus.RECOVERING.value == "recovering"


class TestStepType:
    """Tests for StepType enum."""

    def test_step_type_values(self):
        """Test step type values."""
        assert StepType.DETERMINISTIC_ANALYSIS.value == "deterministic_analysis"
        assert StepType.AGENT_CREATION.value == "agent_creation"
        assert StepType.LLM_ANALYSIS.value == "llm_analysis"
        assert StepType.VALIDATION.value == "validation"
        assert StepType.SYNTHESIS.value == "synthesis"
        assert StepType.RESULT_GENERATION.value == "result_generation"


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_workflow_step_creation(self):
        """Test creating workflow step."""
        step = WorkflowStep(
            name="test_step",
            step_type=StepType.LLM_ANALYSIS,
            description="Test step",
            required=True,
            dependencies=["previous_step"],
            token_budget_percentage=0.2,
        )
        assert step.name == "test_step"
        assert step.step_type == StepType.LLM_ANALYSIS
        assert step.required
        assert "previous_step" in step.dependencies
        assert step.token_budget_percentage == 0.2

    def test_workflow_step_defaults(self):
        """Test workflow step default values."""
        step = WorkflowStep(
            name="minimal",
            step_type=StepType.VALIDATION,
            description="Minimal step",
        )
        assert step.required
        assert step.dependencies == []
        assert step.token_budget_percentage == 0.0
        assert step.retry_count == 3
        assert step.timeout_seconds == 300


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_success(self):
        """Test successful step result."""
        result = StepResult(
            step_name="test_step",
            status=WorkflowStatus.COMPLETED,
            data={"result": "success"},
            tokens_used=100,
            execution_time_seconds=1.5,
        )
        assert result.step_name == "test_step"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.error is None
        assert result.tokens_used == 100

    def test_step_result_failure(self):
        """Test failed step result."""
        result = StepResult(
            step_name="failed_step",
            status=WorkflowStatus.FAILED,
            error="Test error",
            data=None,
        )
        assert result.status == WorkflowStatus.FAILED
        assert result.error == "Test error"


class TestWorkflowContext:
    """Tests for WorkflowContext dataclass."""

    def test_workflow_context_creation(self):
        """Test creating workflow context."""
        context = WorkflowContext(
            workbook_path=Path("test.xlsx"),
            total_token_budget=100000,
            metadata={"user": "test"},
        )
        assert context.workbook_path == Path("test.xlsx")
        assert context.total_token_budget == 100000
        assert context.tokens_used == 0

    def test_add_step_result(self):
        """Test adding step results."""
        context = WorkflowContext(workbook_path=Path("test.xlsx"))
        result = StepResult(
            step_name="step1",
            status=WorkflowStatus.COMPLETED,
            tokens_used=500,
        )

        context.add_step_result(result)
        assert "step1" in context.step_results
        assert context.tokens_used == 500

    def test_token_budget_management(self):
        """Test token budget calculations."""
        context = WorkflowContext(
            workbook_path=Path("test.xlsx"),
            total_token_budget=10000,
        )

        # Add some token usage
        result = StepResult(step_name="step1", status=WorkflowStatus.COMPLETED, tokens_used=3000)
        context.add_step_result(result)

        assert context.get_remaining_tokens() == 7000

        # Test step token budget
        step = WorkflowStep(
            name="step2",
            step_type=StepType.LLM_ANALYSIS,
            description="Test",
            token_budget_percentage=0.2,
        )
        assert context.get_step_token_budget(step) == 2000  # 20% of 10000


class TestDefaultRecoveryStrategy:
    """Tests for DefaultRecoveryStrategy."""

    def test_recovery_strategy_creation(self):
        """Test creating recovery strategy."""
        strategy = DefaultRecoveryStrategy(max_retries=5, base_delay=2.0)
        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0

    def test_can_recover_transient_errors(self):
        """Test recovery for transient errors."""
        strategy = DefaultRecoveryStrategy()
        step = WorkflowStep(name="test", step_type=StepType.LLM_ANALYSIS, description="Test")

        # Should recover from TimeoutError
        assert strategy.can_recover(step, TimeoutError(), 1)
        assert strategy.can_recover(step, ConnectionError(), 1)

        # Should not recover from other errors
        assert not strategy.can_recover(step, ValueError(), 1)

        # Should not recover after max attempts
        assert not strategy.can_recover(step, TimeoutError(), 3)


class TestBaseOrchestrator:
    """Tests for BaseOrchestrator abstract class."""

    def test_base_orchestrator_interface(self):
        """Test base orchestrator interface."""

        class TestOrchestrator(BaseOrchestrator):
            def _setup_workflow(self):
                self.workflow_steps = [
                    WorkflowStep(
                        name="step1",
                        step_type=StepType.DETERMINISTIC_ANALYSIS,
                        description="First step",
                    ),
                    WorkflowStep(
                        name="step2",
                        step_type=StepType.LLM_ANALYSIS,
                        description="Second step",
                        dependencies=["step1"],
                    ),
                ]

            async def execute_step(self, step: WorkflowStep, context: WorkflowContext) -> StepResult:
                return StepResult(
                    step_name=step.name,
                    status=WorkflowStatus.COMPLETED,
                    data={"test": True},
                )

            def _synthesize_results(self, context: WorkflowContext) -> dict:
                return {"final": "results"}

        orchestrator = TestOrchestrator()
        assert len(orchestrator.workflow_steps) == 2
        assert hasattr(orchestrator, "execute_step")
        assert hasattr(orchestrator, "_synthesize_results")

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test workflow execution."""

        class SimpleOrchestrator(BaseOrchestrator):
            def _setup_workflow(self):
                self.workflow_steps = [
                    WorkflowStep(
                        name="analyze",
                        step_type=StepType.DETERMINISTIC_ANALYSIS,
                        description="Analyze",
                    )
                ]

            async def execute_step(self, step: WorkflowStep, context: WorkflowContext) -> StepResult:
                return StepResult(
                    step_name=step.name,
                    status=WorkflowStatus.COMPLETED,
                    data={"analyzed": True},
                    tokens_used=100,
                )

            def _synthesize_results(self, context: WorkflowContext) -> dict:
                return {"success": True, "steps": len(context.step_results)}

        orchestrator = SimpleOrchestrator()
        result = await orchestrator.execute_workflow(Path("test.xlsx"), 10000)

        assert isinstance(result, Success)
        assert result.value["success"] is True
        assert result.value["steps"] == 1

    def test_can_execute_step(self):
        """Test step execution logic."""

        class TestOrchestrator(BaseOrchestrator):
            def _setup_workflow(self):
                pass

            async def execute_step(self, step, context):
                pass

            def _synthesize_results(self, context):
                pass

        orchestrator = TestOrchestrator()
        context = WorkflowContext(workbook_path=Path("test.xlsx"))

        # Step with no dependencies should be executable
        step1 = WorkflowStep(name="step1", step_type=StepType.LLM_ANALYSIS, description="Test")
        assert orchestrator._can_execute_step(step1, context)

        # Step with completed dependency should be executable
        context.add_step_result(StepResult("step1", WorkflowStatus.COMPLETED))
        step2 = WorkflowStep(
            name="step2",
            step_type=StepType.LLM_ANALYSIS,
            description="Test",
            dependencies=["step1"],
        )
        assert orchestrator._can_execute_step(step2, context)

        # Step with failed dependency should not be executable
        context.add_step_result(StepResult("step3", WorkflowStatus.FAILED))
        step4 = WorkflowStep(
            name="step4",
            step_type=StepType.LLM_ANALYSIS,
            description="Test",
            dependencies=["step3"],
        )
        assert not orchestrator._can_execute_step(step4, context)


class TestProgressMonitor:
    """Tests for ProgressMonitor protocol."""

    def test_progress_monitor_protocol(self):
        """Test implementing progress monitor."""

        class TestMonitor:
            def __init__(self):
                self.events = []

            def on_step_start(self, step: WorkflowStep):
                self.events.append(("start", step.name))

            def on_step_complete(self, step: WorkflowStep, result: StepResult):
                self.events.append(("complete", step.name, result.status))

            def on_workflow_complete(self, success: bool, results):
                self.events.append(("workflow_complete", success))

        monitor = TestMonitor()
        step = WorkflowStep(name="test", step_type=StepType.LLM_ANALYSIS, description="Test")
        result = StepResult(step_name="test", status=WorkflowStatus.COMPLETED)

        monitor.on_step_start(step)
        monitor.on_step_complete(step, result)
        monitor.on_workflow_complete(True, {})

        assert len(monitor.events) == 3
        assert monitor.events[0] == ("start", "test")
        assert monitor.events[1] == ("complete", "test", WorkflowStatus.COMPLETED)
        assert monitor.events[2] == ("workflow_complete", True)
