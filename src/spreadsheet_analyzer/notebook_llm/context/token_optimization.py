"""Token optimization and adaptive pipeline selection for context management.

This module implements intelligent token budget management and dynamic pipeline
selection based on the available token budget and analysis requirements.

CLAUDE-KNOWLEDGE: Token limits vary significantly between models:
- GPT-4: 8k-32k tokens
- Claude-3: 100k-200k tokens
- GPT-3.5: 4k-16k tokens
Optimization strategies must adapt to these constraints.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from spreadsheet_analyzer.notebook_llm.context.compressors import (
    BaseCompressor,
    CellObservation,
    CompressionMetrics,
    SpreadsheetLLMCompressor,
    TokenCounter,
)
from spreadsheet_analyzer.notebook_llm.strategies.base import (
    AnalysisFocus,
    AnalysisTask,
    ContextPackage,
)


class CompressionLevel(Enum):
    """Compression levels for different token budget constraints."""

    NONE = auto()  # No compression needed
    LIGHT = auto()  # Basic deduplication
    MODERATE = auto()  # Pattern detection + range aggregation
    AGGRESSIVE = auto()  # Full compression pipeline
    EXTREME = auto()  # Maximum compression, may lose detail


@dataclass
class TokenBudget:
    """Token budget allocation for different components."""

    total: int
    system_prompt: int = 0
    task_prompt: int = 0
    context: int = 0
    response_reserve: int = 0
    safety_margin: int = 0

    @property
    def available_for_context(self) -> int:
        """Calculate tokens available for context after allocations."""
        used = self.system_prompt + self.task_prompt + self.response_reserve + self.safety_margin
        return max(0, self.total - used)

    def allocate_standard(self) -> None:
        """Apply standard allocation ratios."""
        # Standard allocations as percentage of total
        self.system_prompt = int(self.total * 0.10)  # 10% for system
        self.task_prompt = int(self.total * 0.05)  # 5% for task
        self.response_reserve = int(self.total * 0.30)  # 30% for response
        self.safety_margin = int(self.total * 0.05)  # 5% safety
        self.context = self.available_for_context  # 50% for context


@dataclass
class CompressionPipeline:
    """Represents a compression pipeline configuration."""

    name: str
    level: CompressionLevel
    compressor: BaseCompressor
    min_token_budget: int
    max_token_budget: int
    strategies: list[str] = field(default_factory=list)
    suitable_for: list[AnalysisFocus] = field(default_factory=list)

    def is_suitable(self, token_budget: int, focus: AnalysisFocus) -> bool:
        """Check if pipeline is suitable for given constraints."""
        budget_ok = self.min_token_budget <= token_budget <= self.max_token_budget
        focus_ok = not self.suitable_for or focus in self.suitable_for
        return budget_ok and focus_ok


@dataclass
class OptimizationResult:
    """Result of token optimization process."""

    success: bool
    compressed_package: ContextPackage | None
    pipeline_used: str
    compression_level: CompressionLevel
    metrics: CompressionMetrics | None
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CompressorFactory(Protocol):
    """Protocol for creating compressor instances."""

    def __call__(self, level: CompressionLevel) -> BaseCompressor:
        """Create compressor for given level."""
        ...


class TokenOptimizer:
    """Intelligent token budget management and pipeline selection.

    This class manages the allocation of tokens across different components
    and selects appropriate compression strategies based on available budget.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        token_counter: TokenCounter | None = None,
        compressor_factory: CompressorFactory | None = None,
    ):
        """Initialize token optimizer.

        Args:
            model: Model name for token limits
            token_counter: Token counter instance
            compressor_factory: Factory for creating compressors
        """
        self.model = model
        self.token_counter = token_counter or TokenCounter(model)
        self.compressor_factory = compressor_factory or self._default_factory
        self._pipelines = self._initialize_pipelines()

    def _default_factory(self, level: CompressionLevel) -> BaseCompressor:
        """Default compressor factory."""
        if level == CompressionLevel.NONE:
            # Return a pass-through compressor
            return PassThroughCompressor(self.token_counter)
        else:
            # Configure SpreadsheetLLM compressor based on level
            return SpreadsheetLLMCompressor(
                token_counter=self.token_counter,
                enable_pattern_detection=level.value >= CompressionLevel.MODERATE.value,
                enable_range_aggregation=level.value >= CompressionLevel.MODERATE.value,
                enable_semantic_clustering=level.value >= CompressionLevel.AGGRESSIVE.value,
            )

    def _initialize_pipelines(self) -> list[CompressionPipeline]:
        """Initialize compression pipelines for different scenarios."""
        return [
            # No compression for large budgets
            CompressionPipeline(
                name="no_compression",
                level=CompressionLevel.NONE,
                compressor=self.compressor_factory(CompressionLevel.NONE),
                min_token_budget=50000,
                max_token_budget=200000,
                strategies=["pass_through"],
                suitable_for=list(AnalysisFocus),
            ),
            # Light compression for medium budgets
            CompressionPipeline(
                name="light_compression",
                level=CompressionLevel.LIGHT,
                compressor=self.compressor_factory(CompressionLevel.LIGHT),
                min_token_budget=20000,
                max_token_budget=50000,
                strategies=["deduplication", "basic_aggregation"],
                suitable_for=list(AnalysisFocus),
            ),
            # Moderate compression for standard analysis
            CompressionPipeline(
                name="moderate_compression",
                level=CompressionLevel.MODERATE,
                compressor=self.compressor_factory(CompressionLevel.MODERATE),
                min_token_budget=8000,
                max_token_budget=20000,
                strategies=["pattern_detection", "range_aggregation"],
                suitable_for=[AnalysisFocus.STRUCTURE, AnalysisFocus.FORMULAS, AnalysisFocus.GENERAL],
            ),
            # Aggressive compression for tight budgets
            CompressionPipeline(
                name="aggressive_compression",
                level=CompressionLevel.AGGRESSIVE,
                compressor=self.compressor_factory(CompressionLevel.AGGRESSIVE),
                min_token_budget=4000,
                max_token_budget=8000,
                strategies=["pattern_detection", "range_aggregation", "semantic_clustering", "hierarchical_summary"],
                suitable_for=[AnalysisFocus.STRUCTURE, AnalysisFocus.RELATIONSHIPS, AnalysisFocus.GENERAL],
            ),
            # Extreme compression for very limited budgets
            CompressionPipeline(
                name="extreme_compression",
                level=CompressionLevel.EXTREME,
                compressor=self.compressor_factory(CompressionLevel.EXTREME),
                min_token_budget=1000,
                max_token_budget=4000,
                strategies=["aggressive_summarization", "structural_only", "top_k_selection"],
                suitable_for=[AnalysisFocus.STRUCTURE, AnalysisFocus.GENERAL],
            ),
        ]

    def get_model_token_limit(self) -> int:
        """Get token limit for current model."""
        # Model-specific limits
        limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-2.1": 100000,
            "claude-2": 100000,
        }

        # Check for exact match
        if self.model in limits:
            return limits[self.model]

        # Check for partial matches
        for model_key, limit in limits.items():
            if model_key in self.model.lower():
                return limit

        # Default conservative limit
        return 4096

    def allocate_budget(self, task: AnalysisTask, total_tokens: int | None = None) -> TokenBudget:
        """Allocate token budget for analysis task.

        Args:
            task: Analysis task to perform
            total_tokens: Override total token budget

        Returns:
            Allocated token budget
        """
        if total_tokens is None:
            total_tokens = self.get_model_token_limit()

        budget = TokenBudget(total=total_tokens)

        # Adjust allocations based on task requirements
        if task.focus in [AnalysisFocus.FORMULAS, AnalysisFocus.RELATIONSHIPS]:
            # Complex analysis needs more response space
            budget.system_prompt = int(total_tokens * 0.08)
            budget.task_prompt = int(total_tokens * 0.07)
            budget.response_reserve = int(total_tokens * 0.40)
            budget.safety_margin = int(total_tokens * 0.05)
        elif task.focus == AnalysisFocus.DATA_VALIDATION:
            # Validation needs balanced allocation
            budget.allocate_standard()
        else:
            # Default allocation
            budget.allocate_standard()

        budget.context = budget.available_for_context
        return budget

    def select_pipeline(
        self, observations: list[CellObservation], task: AnalysisTask, token_budget: TokenBudget
    ) -> CompressionPipeline | None:
        """Select appropriate compression pipeline.

        Args:
            observations: Observations to compress
            task: Analysis task
            token_budget: Available token budget

        Returns:
            Selected pipeline or None if no suitable pipeline
        """
        # Estimate uncompressed size
        uncompressed_tokens = sum(self.token_counter.count_tokens(str(obs)) for obs in observations)

        # Check if compression is needed
        if uncompressed_tokens <= token_budget.context:
            # No compression needed
            for pipeline in self._pipelines:
                if pipeline.level == CompressionLevel.NONE:
                    return pipeline

        # Find suitable pipeline
        suitable_pipelines = [p for p in self._pipelines if p.is_suitable(token_budget.context, task.focus)]

        if not suitable_pipelines:
            return None

        # Select pipeline based on compression requirements
        compression_ratio_needed = 1 - (token_budget.context / uncompressed_tokens)

        # Sort by compression level
        suitable_pipelines.sort(key=lambda p: p.level.value)

        # Select first pipeline that can achieve needed compression
        for pipeline in suitable_pipelines:
            # Estimate if pipeline can achieve needed compression
            if pipeline.level == CompressionLevel.NONE:
                continue
            elif (
                (pipeline.level == CompressionLevel.LIGHT and compression_ratio_needed < 0.2)
                or (pipeline.level == CompressionLevel.MODERATE and compression_ratio_needed < 0.5)
                or (pipeline.level == CompressionLevel.AGGRESSIVE and compression_ratio_needed < 0.7)
                or pipeline.level == CompressionLevel.EXTREME
            ):
                return pipeline

        # Return most aggressive pipeline if none selected
        return suitable_pipelines[-1] if suitable_pipelines else None

    def optimize(
        self,
        observations: list[CellObservation],
        task: AnalysisTask,
        total_tokens: int | None = None,
        preserve_structure: bool = True,
    ) -> OptimizationResult:
        """Optimize context for given constraints.

        This is the main entry point that:
        1. Allocates token budget
        2. Selects appropriate pipeline
        3. Compresses observations
        4. Validates results

        Args:
            observations: Observations to optimize
            task: Analysis task
            total_tokens: Override total token budget
            preserve_structure: Whether to preserve structural relationships

        Returns:
            Optimization result with compressed package
        """
        # Allocate budget
        budget = self.allocate_budget(task, total_tokens)

        # Select pipeline
        pipeline = self.select_pipeline(observations, task, budget)

        if not pipeline:
            return OptimizationResult(
                success=False,
                compressed_package=None,
                pipeline_used="none",
                compression_level=CompressionLevel.NONE,
                metrics=None,
                warnings=["No suitable compression pipeline found"],
            )

        # Apply compression
        try:
            compressed = pipeline.compressor.compress(observations, budget.context, preserve_structure)

            # Validate result
            if compressed.token_count > budget.context:
                warnings = [f"Compressed size ({compressed.token_count}) exceeds budget ({budget.context})"]
            else:
                warnings = []

            # Generate recommendations
            recommendations = self._generate_recommendations(pipeline, compressed, budget, task)

            return OptimizationResult(
                success=True,
                compressed_package=compressed,
                pipeline_used=pipeline.name,
                compression_level=pipeline.level,
                metrics=pipeline.compressor.metrics,
                recommendations=recommendations,
                warnings=warnings,
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                compressed_package=None,
                pipeline_used=pipeline.name,
                compression_level=pipeline.level,
                metrics=None,
                warnings=[f"Compression failed: {e!s}"],
            )

    def _generate_recommendations(
        self, pipeline: CompressionPipeline, compressed: ContextPackage, budget: TokenBudget, task: AnalysisTask
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check compression efficiency
        if pipeline.compressor.metrics:
            ratio = pipeline.compressor.metrics.compression_ratio
            if ratio < 0.3:
                recommendations.append("Low compression ratio. Consider pre-filtering observations.")
            elif ratio > 0.8:
                recommendations.append("High compression ratio. Some detail may be lost.")

        # Check token utilization
        utilization = compressed.token_count / budget.context
        if utilization < 0.7:
            recommendations.append(
                f"Only using {utilization:.0%} of available context. Consider including more observations."
            )
        elif utilization > 0.95:
            recommendations.append("Near token limit. Consider more aggressive compression.")

        # Task-specific recommendations
        if task.focus == AnalysisFocus.FORMULAS and pipeline.level.value > CompressionLevel.MODERATE.value:
            recommendations.append(
                "Formula analysis with high compression may miss details. Consider increasing token budget."
            )

        return recommendations


class PassThroughCompressor(BaseCompressor):
    """Simple pass-through compressor when no compression is needed."""

    def compress(
        self,
        observations: list[CellObservation],
        token_budget: int,
        preserve_structure: bool = True,  # noqa: ARG002
    ) -> ContextPackage:
        """Pass through observations without compression."""
        import time

        start_time = time.time()

        cells = []
        total_tokens = 0

        for obs in observations:
            if obs.tokens == 0:
                obs.tokens = self.token_counter.count_tokens(str(obs))

            if total_tokens + obs.tokens <= token_budget:
                cells.append(
                    {
                        "location": obs.location,
                        "type": obs.observation_type,
                        "content": obs.content,
                        "metadata": obs.metadata,
                    }
                )
                total_tokens += obs.tokens

        package = ContextPackage(
            cells=cells,
            metadata={
                "compression_method": "pass_through",
                "total_observations": len(observations),
                "included_observations": len(cells),
            },
            focus_hints=[],
            token_count=total_tokens,
            compression_method="PassThrough",
        )

        self._calculate_metrics(observations, package, time.time() - start_time)
        return package
