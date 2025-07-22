"""Tests for token optimization components."""

from unittest.mock import Mock

from spreadsheet_analyzer.notebook_llm.context.token_optimization import (
    CompressionLevel,
    CompressionPipeline,
    OptimizationResult,
    TokenBudget,
    TokenOptimizer,
)


class TestCompressionLevel:
    """Tests for CompressionLevel enum."""

    def test_compression_levels(self):
        """Test compression level values."""
        assert CompressionLevel.NONE.value == "none"
        assert CompressionLevel.MINIMAL.value == "minimal"
        assert CompressionLevel.MODERATE.value == "moderate"
        assert CompressionLevel.AGGRESSIVE.value == "aggressive"


class TestTokenBudget:
    """Tests for TokenBudget class."""

    def test_token_budget_creation(self):
        """Test creating token budget."""
        budget = TokenBudget(
            total=1000,
            used=200,
            reserved=100,
        )
        assert budget.total == 1000
        assert budget.used == 200
        assert budget.reserved == 100

    def test_available_tokens(self):
        """Test calculating available tokens."""
        budget = TokenBudget(total=1000, used=200, reserved=100)
        assert budget.available == 700  # 1000 - 200 - 100

    def test_remaining_tokens(self):
        """Test calculating remaining tokens."""
        budget = TokenBudget(total=1000, used=200, reserved=100)
        assert budget.remaining == 800  # 1000 - 200

    def test_utilization_percentage(self):
        """Test calculating utilization percentage."""
        budget = TokenBudget(total=1000, used=250, reserved=0)
        assert budget.utilization == 0.25  # 250 / 1000


class TestCompressionPipeline:
    """Tests for CompressionPipeline class."""

    def test_compression_pipeline_creation(self):
        """Test creating compression pipeline."""
        pipeline = CompressionPipeline(
            name="test_pipeline",
            level=CompressionLevel.MODERATE,
            steps=["deduplicate", "aggregate", "summarize"],
        )
        assert pipeline.name == "test_pipeline"
        assert pipeline.level == CompressionLevel.MODERATE
        assert len(pipeline.steps) == 3

    def test_compression_pipeline_empty_steps(self):
        """Test pipeline with no steps."""
        pipeline = CompressionPipeline(
            name="empty",
            level=CompressionLevel.NONE,
            steps=[],
        )
        assert len(pipeline.steps) == 0


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            original_tokens=1000,
            optimized_tokens=300,
            compression_ratio=0.7,
            compression_level=CompressionLevel.MODERATE,
            data={"compressed": True},
        )
        assert result.original_tokens == 1000
        assert result.optimized_tokens == 300
        assert result.compression_ratio == 0.7
        assert result.data["compressed"] is True

    def test_optimization_result_metadata(self):
        """Test optimization result with metadata."""
        result = OptimizationResult(
            original_tokens=500,
            optimized_tokens=200,
            compression_ratio=0.6,
            compression_level=CompressionLevel.AGGRESSIVE,
            data={},
            metadata={"method": "pattern_detection", "time": 0.5},
        )
        assert result.metadata["method"] == "pattern_detection"
        assert result.metadata["time"] == 0.5


class TestTokenOptimizer:
    """Tests for TokenOptimizer class."""

    def test_optimizer_initialization(self):
        """Test token optimizer initialization."""
        mock_compressor = Mock()
        optimizer = TokenOptimizer(compressor=mock_compressor)
        assert optimizer._compressor == mock_compressor

    def test_optimize_no_compression_needed(self):
        """Test optimization when no compression needed."""
        mock_compressor = Mock()
        mock_compressor.estimate_tokens.return_value = 500

        optimizer = TokenOptimizer(compressor=mock_compressor)

        data = {"sheet": "data"}
        result = optimizer.optimize(data, token_budget=1000)

        # Should not compress if under budget
        assert result.compression_level == CompressionLevel.NONE
        assert result.optimized_tokens == 500

    def test_optimize_with_compression(self):
        """Test optimization with compression needed."""
        mock_compressor = Mock()
        # Initial estimate exceeds budget
        mock_compressor.estimate_tokens.side_effect = [1500, 800]
        mock_compressor.compress.return_value = {"compressed": "data"}

        optimizer = TokenOptimizer(compressor=mock_compressor)

        data = {"sheet": "large_data"}
        result = optimizer.optimize(data, token_budget=1000)

        # Should apply compression
        assert mock_compressor.compress.called
        assert result.optimized_tokens == 800
        assert result.compression_ratio == (1500 - 800) / 1500

    def test_progressive_compression(self):
        """Test progressive compression levels."""
        mock_compressor = Mock()
        # Simulate progressive compression
        mock_compressor.estimate_tokens.side_effect = [2000, 1500, 1000, 500]
        mock_compressor.compress.side_effect = [
            {"level1": "data"},
            {"level2": "data"},
            {"level3": "data"},
        ]

        optimizer = TokenOptimizer(compressor=mock_compressor)

        data = {"sheet": "very_large_data"}
        result = optimizer.optimize(data, token_budget=600)

        # Should try multiple compression levels
        assert mock_compressor.compress.call_count >= 2
        assert result.optimized_tokens <= 600

    def test_allocate_budget(self):
        """Test budget allocation across components."""
        mock_compressor = Mock()
        optimizer = TokenOptimizer(compressor=mock_compressor)

        context = {
            "sheets": [{"name": "Sheet1"}, {"name": "Sheet2"}],
            "metadata": {"total": 100},
            "formulas": [{"cell": "A1"}],
        }

        allocation = optimizer.allocate_budget(context, total_budget=1000)

        assert "sheets" in allocation
        assert "metadata" in allocation
        assert "formulas" in allocation
        assert sum(allocation.values()) <= 1000

    def test_select_compression_pipeline(self):
        """Test selecting appropriate compression pipeline."""
        mock_compressor = Mock()
        optimizer = TokenOptimizer(compressor=mock_compressor)

        context = {"data": "test"}

        # Large budget - minimal compression
        pipeline = optimizer.select_compression_pipeline(context, token_budget=10000)
        assert pipeline.level in [CompressionLevel.NONE, CompressionLevel.MINIMAL]

        # Small budget - aggressive compression
        pipeline = optimizer.select_compression_pipeline(context, token_budget=100)
        assert pipeline.level in [CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE]

    def test_optimization_with_priority(self):
        """Test optimization with priority settings."""
        mock_compressor = Mock()
        mock_compressor.estimate_tokens.return_value = 800
        mock_compressor.compress.return_value = {"compressed": "data"}

        optimizer = TokenOptimizer(compressor=mock_compressor)

        data = {"critical": "data", "optional": "info"}
        result = optimizer.optimize(
            data,
            token_budget=1000,
            priority_components=["critical"],
        )

        # Should preserve critical components
        assert "critical" in str(result.data)
