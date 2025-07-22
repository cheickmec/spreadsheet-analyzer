"""Python-based workflow orchestrator for spreadsheet analysis.

This module implements the immediate Python-based orchestration approach,
providing full programmatic control over the analysis workflow.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager
from spreadsheet_analyzer.notebook_llm.orchestration.base import (
    BaseOrchestrator,
    StepResult,
    StepType,
    WorkflowContext,
    WorkflowStatus,
    WorkflowStep,
)
from spreadsheet_analyzer.notebook_llm.orchestration.models import (
    AnalysisComplexity,
    ModelRouter,
    ModelTier,
)
from spreadsheet_analyzer.notebook_llm.strategies.base import (
    AnalysisFocus,
    AnalysisTask,
    ResponseFormat,
)
from spreadsheet_analyzer.notebook_llm.strategies.registry import StrategyRegistry
from spreadsheet_analyzer.typing import Failure, Result, Success

logger = logging.getLogger(__name__)


class PythonWorkflowOrchestrator(BaseOrchestrator):
    """Pure Python workflow orchestration - immediate implementation.

    This orchestrator implements the complete spreadsheet analysis workflow
    using direct Python code, providing maximum flexibility and debuggability.
    """

    def __init__(
        self,
        strategy_registry: StrategyRegistry,
        model_router: ModelRouter,
        kernel_manager: AgentKernelManager | None = None,
        **kwargs,
    ):
        """Initialize the Python orchestrator.

        Args:
            strategy_registry: Registry of available analysis strategies
            model_router: Router for selecting appropriate models
            kernel_manager: Manager for Jupyter kernels (optional)
            **kwargs: Additional arguments passed to base class
        """
        self.strategy_registry = strategy_registry
        self.model_router = model_router
        self.kernel_manager = kernel_manager
        self.deterministic_pipeline = AnalysisPipeline()
        super().__init__(**kwargs)

    def _setup_workflow(self) -> None:
        """Set up the workflow steps for spreadsheet analysis."""
        self.workflow_steps = [
            # Phase 1: Deterministic Analysis
            WorkflowStep(
                name="deterministic_analysis",
                step_type=StepType.DETERMINISTIC_ANALYSIS,
                description="Run deterministic analysis pipeline",
                required=True,
                dependencies=[],
                token_budget_percentage=0.0,  # No LLM tokens
                timeout_seconds=60,
            ),
            # Phase 2: Complexity Assessment
            WorkflowStep(
                name="complexity_assessment",
                step_type=StepType.DETERMINISTIC_ANALYSIS,
                description="Assess workbook complexity and plan agent allocation",
                required=True,
                dependencies=["deterministic_analysis"],
                token_budget_percentage=0.05,  # 5% for assessment
                timeout_seconds=30,
            ),
            # Phase 3: Agent Creation
            WorkflowStep(
                name="agent_creation",
                step_type=StepType.AGENT_CREATION,
                description="Create and initialize analysis agents",
                required=True,
                dependencies=["complexity_assessment"],
                token_budget_percentage=0.0,
                timeout_seconds=120,
            ),
            # Phase 4: Parallel Analysis
            WorkflowStep(
                name="sheet_analysis",
                step_type=StepType.LLM_ANALYSIS,
                description="Analyze individual sheets with agents",
                required=True,
                dependencies=["agent_creation"],
                token_budget_percentage=0.60,  # 60% for main analysis
                timeout_seconds=600,  # 10 minutes
            ),
            # Phase 5: Cross-Sheet Analysis
            WorkflowStep(
                name="relationship_analysis",
                step_type=StepType.LLM_ANALYSIS,
                description="Analyze cross-sheet relationships",
                required=False,
                dependencies=["sheet_analysis"],
                token_budget_percentage=0.20,  # 20% for relationships
                timeout_seconds=300,
            ),
            # Phase 6: Validation
            WorkflowStep(
                name="validation",
                step_type=StepType.VALIDATION,
                description="Validate findings and calculations",
                required=True,
                dependencies=["sheet_analysis"],
                token_budget_percentage=0.10,  # 10% for validation
                timeout_seconds=300,
            ),
            # Phase 7: Synthesis
            WorkflowStep(
                name="synthesis",
                step_type=StepType.SYNTHESIS,
                description="Synthesize findings into final report",
                required=True,
                dependencies=["validation", "relationship_analysis"],
                token_budget_percentage=0.05,  # 5% for synthesis
                timeout_seconds=120,
            ),
        ]

    async def execute_step(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
    ) -> StepResult:
        """Execute a single workflow step.

        Routes to appropriate handler based on step type.
        """
        start_time = datetime.now()

        try:
            if step.step_type == StepType.DETERMINISTIC_ANALYSIS:
                if step.name == "deterministic_analysis":
                    result = await self._run_deterministic_analysis(context)
                else:
                    result = await self._assess_complexity(context)

            elif step.step_type == StepType.AGENT_CREATION:
                result = await self._create_agents(context)

            elif step.step_type == StepType.LLM_ANALYSIS:
                if step.name == "sheet_analysis":
                    result = await self._run_sheet_analysis(context, step)
                else:
                    result = await self._run_relationship_analysis(context, step)

            elif step.step_type == StepType.VALIDATION:
                result = await self._run_validation(context, step)

            elif step.step_type == StepType.SYNTHESIS:
                result = await self._synthesize_report(context, step)

            else:
                raise ValueError(f"Unknown step type: {step.step_type}")

            execution_time = (datetime.now() - start_time).total_seconds()

            return StepResult(
                step_name=step.name,
                status=WorkflowStatus.COMPLETED,
                data=result.get("data"),
                execution_time_seconds=execution_time,
                tokens_used=result.get("tokens_used", 0),
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            logger.exception(f"Error executing step {step.name}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return StepResult(
                step_name=step.name,
                status=WorkflowStatus.FAILED,
                error=str(e),
                execution_time_seconds=execution_time,
            )

    async def _run_deterministic_analysis(
        self,
        context: WorkflowContext,
    ) -> dict[str, Any]:
        """Run the deterministic analysis pipeline."""
        logger.info(f"Running deterministic analysis on {context.workbook_path}")

        # Run the pipeline
        result = await asyncio.to_thread(
            self.deterministic_pipeline.run,
            context.workbook_path,
        )

        if isinstance(result, Success):
            context.deterministic_results = result.value
            return {
                "data": result.value,
                "tokens_used": 0,
                "metadata": {"stages_completed": len(result.value.get("stages", []))},
            }
        else:
            raise RuntimeError(f"Deterministic analysis failed: {result.error}")

    async def _assess_complexity(
        self,
        context: WorkflowContext,
    ) -> dict[str, Any]:
        """Assess workbook complexity and plan resource allocation."""
        det_results = context.deterministic_results

        # Extract complexity metrics
        total_sheets = len(det_results.get("sheets", {}))
        total_formulas = sum(sheet.get("formula_count", 0) for sheet in det_results.get("sheets", {}).values())
        has_vba = det_results.get("security", {}).get("has_vba_macros", False)
        has_external_refs = det_results.get("security", {}).get("has_external_references", False)

        # Determine complexity level
        if total_sheets > 20 or total_formulas > 10000 or has_vba:
            complexity = AnalysisComplexity.HIGH
        elif total_sheets > 5 or total_formulas > 1000 or has_external_refs:
            complexity = AnalysisComplexity.MEDIUM
        else:
            complexity = AnalysisComplexity.LOW

        # Plan agent allocation
        agents_per_sheet = 1  # Default
        if complexity == AnalysisComplexity.HIGH:
            # May need specialized agents for complex sheets
            agents_per_sheet = 1.5

        estimated_agents = min(int(total_sheets * agents_per_sheet), self.max_concurrent_agents)

        return {
            "data": {
                "complexity": complexity,
                "total_sheets": total_sheets,
                "total_formulas": total_formulas,
                "estimated_agents": estimated_agents,
                "resource_plan": {
                    "model_tier": ModelTier.SMALL if complexity == AnalysisComplexity.LOW else ModelTier.LARGE,
                    "parallel_agents": min(estimated_agents, 5),  # Start conservatively
                    "token_allocation_strategy": "proportional_to_complexity",
                },
            },
            "tokens_used": 0,
            "metadata": {"assessment_method": "rule_based"},
        }

    async def _create_agents(
        self,
        context: WorkflowContext,
    ) -> dict[str, Any]:
        """Create and initialize analysis agents."""
        complexity_data = context.step_results["complexity_assessment"].data
        sheets_info = context.deterministic_results.get("sheets", {})

        agents = []
        agent_configs = []

        # Create agent configurations based on sheet characteristics
        for sheet_name, sheet_info in sheets_info.items():
            # Determine agent strategy based on sheet characteristics
            if sheet_info.get("has_pivot_tables"):
                strategy = "pivot_table_specialist"
            elif sheet_info.get("formula_count", 0) > 100:
                strategy = "formula_analyst"
            elif sheet_info.get("has_charts"):
                strategy = "visualization_analyst"
            else:
                strategy = "general_analyst"

            agent_config = {
                "name": f"agent_{sheet_name}",
                "sheet": sheet_name,
                "strategy": strategy,
                "priority": self._calculate_sheet_priority(sheet_info),
                "token_budget": 0,  # Will be allocated based on priority
            }
            agent_configs.append(agent_config)

        # Sort by priority and allocate token budgets
        agent_configs.sort(key=lambda x: x["priority"], reverse=True)
        total_priority = sum(cfg["priority"] for cfg in agent_configs)

        for cfg in agent_configs:
            cfg["token_budget"] = int(
                context.get_step_token_budget(context.step_results["sheet_analysis"]) * cfg["priority"] / total_priority
            )

        # Initialize kernel manager if needed
        if self.kernel_manager:
            await self.kernel_manager.initialize()

        return {
            "data": {
                "agent_configs": agent_configs,
                "total_agents": len(agent_configs),
            },
            "tokens_used": 0,
            "metadata": {"initialization_method": "priority_based"},
        }

    async def _run_sheet_analysis(
        self,
        context: WorkflowContext,
        step: WorkflowStep,
    ) -> dict[str, Any]:
        """Run parallel sheet analysis with agents."""
        agent_configs = context.step_results["agent_creation"].data["agent_configs"]
        complexity_data = context.step_results["complexity_assessment"].data

        # Determine model tier based on complexity
        model_tier = complexity_data["resource_plan"]["model_tier"]

        # Create analysis tasks
        tasks = []
        for agent_config in agent_configs:
            task = self._analyze_sheet(
                context=context,
                agent_config=agent_config,
                model_tier=model_tier,
            )
            tasks.append(task)

        # Run analyses in parallel (with concurrency limit)
        max_concurrent = complexity_data["resource_plan"]["parallel_agents"]
        results = await self._run_concurrent_tasks(tasks, max_concurrent)

        # Aggregate results
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        sheet_analyses = {r["sheet"]: r["analysis"] for r in results}

        context.agent_results.update(sheet_analyses)

        return {
            "data": sheet_analyses,
            "tokens_used": total_tokens,
            "metadata": {
                "sheets_analyzed": len(results),
                "model_tier": model_tier.value,
            },
        }

    async def _analyze_sheet(
        self,
        context: WorkflowContext,
        agent_config: dict[str, Any],
        model_tier: ModelTier,
    ) -> dict[str, Any]:
        """Analyze a single sheet with an agent."""
        sheet_name = agent_config["sheet"]
        strategy_name = agent_config["strategy"]
        token_budget = agent_config["token_budget"]

        # Get appropriate strategy
        strategy = self.strategy_registry.get_strategy(strategy_name)

        # Get appropriate model
        model = self.model_router.select_model(
            task_type="sheet_analysis",
            complexity=AnalysisComplexity.MEDIUM,  # Per-sheet complexity
            tier_preference=model_tier,
        )

        # Create analysis task
        task = AnalysisTask(
            name=f"analyze_{sheet_name}",
            description=f"Analyze sheet {sheet_name}",
            focus=AnalysisFocus.GENERAL,
            expected_format=ResponseFormat.STRUCTURED,
            focus_area=sheet_name,
        )

        # Execute analysis (placeholder - would use actual LLM)
        # In real implementation, this would:
        # 1. Create/get kernel for agent
        # 2. Load sheet data into notebook
        # 3. Execute strategy with LLM
        # 4. Return results

        # For now, return mock results
        return {
            "sheet": sheet_name,
            "analysis": {
                "summary": f"Analysis of {sheet_name}",
                "findings": [],
                "recommendations": [],
            },
            "tokens_used": int(token_budget * 0.8),  # Simulate 80% usage
        }

    async def _run_relationship_analysis(
        self,
        context: WorkflowContext,
        step: WorkflowStep,
    ) -> dict[str, Any]:
        """Analyze cross-sheet relationships."""
        # Placeholder implementation
        return {
            "data": {
                "relationships": [],
                "dependency_graph": {},
            },
            "tokens_used": 0,
            "metadata": {"method": "graph_analysis"},
        }

    async def _run_validation(
        self,
        context: WorkflowContext,
        step: WorkflowStep,
    ) -> dict[str, Any]:
        """Validate findings and calculations."""
        # Placeholder implementation
        return {
            "data": {
                "validation_results": [],
                "errors_found": 0,
            },
            "tokens_used": 0,
            "metadata": {"validation_method": "cross_check"},
        }

    async def _synthesize_report(
        self,
        context: WorkflowContext,
        step: WorkflowStep,
    ) -> dict[str, Any]:
        """Synthesize findings into final report."""
        # Placeholder implementation
        return {
            "data": {
                "executive_summary": "Spreadsheet analysis complete",
                "key_findings": [],
                "recommendations": [],
            },
            "tokens_used": 0,
            "metadata": {"synthesis_method": "llm_summary"},
        }

    def _synthesize_results(self, context: WorkflowContext) -> dict[str, Any]:
        """Synthesize final results from the workflow context."""
        return {
            "workbook_path": str(context.workbook_path),
            "analysis_summary": context.step_results.get("synthesis", {}).data,
            "deterministic_results": context.deterministic_results,
            "agent_analyses": context.agent_results,
            "total_tokens_used": context.tokens_used,
            "total_execution_time": sum(r.execution_time_seconds for r in context.step_results.values()),
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "orchestrator": "PythonWorkflowOrchestrator",
                "version": "1.0.0",
            },
        }

    def _calculate_sheet_priority(self, sheet_info: dict[str, Any]) -> float:
        """Calculate priority score for a sheet based on its characteristics."""
        priority = 1.0

        # Increase priority for sheets with more formulas
        priority += min(sheet_info.get("formula_count", 0) / 100, 5.0)

        # Increase priority for sheets with external references
        if sheet_info.get("has_external_references"):
            priority += 2.0

        # Increase priority for sheets with pivot tables
        if sheet_info.get("has_pivot_tables"):
            priority += 3.0

        # Increase priority for sheets with charts
        if sheet_info.get("has_charts"):
            priority += 1.0

        return priority

    async def _run_concurrent_tasks(
        self,
        tasks: list[Any],
        max_concurrent: int,
    ) -> list[Any]:
        """Run tasks concurrently with a limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        return await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=False,
        )

    async def analyze_spreadsheet(
        self,
        file_path: Path,
        token_budget: int = 100000,
        progress_callback: Any | None = None,
    ) -> Result[dict[str, Any]]:
        """Main entry point for spreadsheet analysis.

        Args:
            file_path: Path to the Excel file
            token_budget: Total token budget for analysis
            progress_callback: Optional callback for progress updates

        Returns:
            Success with analysis results or Failure with error details
        """
        logger.info(f"Starting analysis of {file_path} with budget {token_budget}")

        # Validate file exists
        if not file_path.exists():
            return Failure(error=f"File not found: {file_path}")

        # Execute workflow
        result = await self.execute_workflow(
            workbook_path=file_path,
            total_token_budget=token_budget,
        )

        return result
