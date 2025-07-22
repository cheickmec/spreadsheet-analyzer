"""Tests for notebook_llm model management components."""

from unittest.mock import Mock

from spreadsheet_analyzer.notebook_llm.orchestration.models import (
    AnalysisComplexity,
    BaseModel,
    CostController,
    ModelConfig,
    ModelProvider,
    ModelRouter,
    ModelTier,
    ModelUsage,
)


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_model_tier_values(self):
        """Test model tier values."""
        assert ModelTier.SMALL.value == "small"
        assert ModelTier.MEDIUM.value == "medium"
        assert ModelTier.LARGE.value == "large"
        assert ModelTier.SPECIALIZED.value == "specialized"


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_provider_values(self):
        """Test provider values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.LOCAL.value == "local"
        assert ModelProvider.HUGGINGFACE.value == "huggingface"


class TestAnalysisComplexity:
    """Tests for AnalysisComplexity enum."""

    def test_complexity_values(self):
        """Test complexity values."""
        assert AnalysisComplexity.LOW.value == "low"
        assert AnalysisComplexity.MEDIUM.value == "medium"
        assert AnalysisComplexity.HIGH.value == "high"
        assert AnalysisComplexity.VERY_HIGH.value == "very_high"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            name="gpt-4",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.LARGE,
            max_tokens=4096,
            cost_per_1k_tokens=0.01,
            cost_per_1k_completion_tokens=0.03,
            supports_functions=True,
            supports_vision=True,
            context_window=128000,
        )
        assert config.name == "gpt-4"
        assert config.provider == ModelProvider.OPENAI
        assert config.tier == ModelTier.LARGE
        assert config.supports_functions
        assert config.supports_vision

    def test_estimate_cost(self):
        """Test cost estimation."""
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.SMALL,
            max_tokens=1000,
            cost_per_1k_tokens=0.001,
            cost_per_1k_completion_tokens=0.002,
        )

        # Test with 1000 input and 500 output tokens
        cost = config.estimate_cost(1000, 500)
        expected = (1000 / 1000) * 0.001 + (500 / 1000) * 0.002
        assert cost == expected


class TestModelUsage:
    """Tests for ModelUsage dataclass."""

    def test_model_usage_creation(self):
        """Test creating model usage."""
        usage = ModelUsage(
            model_name="gpt-4",
            task_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            cost=0.025,
            duration_seconds=2.5,
            success=True,
            metadata={"complexity": "high"},
        )
        assert usage.model_name == "gpt-4"
        assert usage.task_type == "analysis"
        assert usage.cost == 0.025
        assert usage.success
        assert usage.metadata["complexity"] == "high"


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_supports_task_by_tier(self):
        """Test task support based on tier."""

        class TestModel(BaseModel):
            def generate(self, prompt, max_tokens=None, temperature=0.1, **kwargs):
                return "response"

            def count_tokens(self, text):
                return len(text.split())

        # Small model
        small_config = ModelConfig(
            name="small",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.SMALL,
            max_tokens=1000,
            cost_per_1k_tokens=0.001,
            cost_per_1k_completion_tokens=0.002,
        )
        small_model = TestModel(small_config)
        assert small_model.supports_task("extraction")
        assert small_model.supports_task("counting")
        assert not small_model.supports_task("reasoning")

        # Large model
        large_config = ModelConfig(
            name="large",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.LARGE,
            max_tokens=4000,
            cost_per_1k_tokens=0.01,
            cost_per_1k_completion_tokens=0.03,
        )
        large_model = TestModel(large_config)
        assert large_model.supports_task("extraction")
        assert large_model.supports_task("reasoning")
        assert large_model.supports_task("synthesis")

    def test_track_usage(self):
        """Test usage tracking."""

        class TestModel(BaseModel):
            def generate(self, prompt, max_tokens=None, temperature=0.1, **kwargs):
                return "response"

            def count_tokens(self, text):
                return len(text.split())

        config = ModelConfig(
            name="test",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.MEDIUM,
            max_tokens=2000,
            cost_per_1k_tokens=0.005,
            cost_per_1k_completion_tokens=0.01,
        )
        model = TestModel(config)

        # Track some usage
        model.track_usage(
            task_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            duration_seconds=2.0,
            success=True,
        )

        assert len(model._usage_history) == 1
        usage = model._usage_history[0]
        assert usage.task_type == "analysis"
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500

    def test_get_usage_summary(self):
        """Test usage summary generation."""

        class TestModel(BaseModel):
            def generate(self, prompt, max_tokens=None, temperature=0.1, **kwargs):
                return "response"

            def count_tokens(self, text):
                return len(text.split())

        config = ModelConfig(
            name="test",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.MEDIUM,
            max_tokens=2000,
            cost_per_1k_tokens=0.005,
            cost_per_1k_completion_tokens=0.01,
        )
        model = TestModel(config)

        # Empty summary
        summary = model.get_usage_summary()
        assert summary["total_requests"] == 0
        assert summary["total_cost"] == 0.0

        # Add usage
        model.track_usage("task1", 1000, 500, 2.0, True)
        model.track_usage("task2", 2000, 1000, 3.0, False)

        summary = model.get_usage_summary()
        assert summary["total_requests"] == 2
        assert summary["total_tokens"] == 4500
        assert summary["success_rate"] == 0.5
        assert "task1" in summary["cost_by_task"]
        assert "task2" in summary["cost_by_task"]


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_router_initialization(self):
        """Test router initialization."""
        router = ModelRouter()
        assert hasattr(router, "models")
        assert hasattr(router, "routing_rules")
        assert hasattr(router, "model_configs")

    def test_register_model(self):
        """Test model registration."""
        router = ModelRouter()

        mock_model = Mock(spec=BaseModel)
        mock_model.config = ModelConfig(
            name="test-model",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.SMALL,
            max_tokens=1000,
            cost_per_1k_tokens=0.001,
            cost_per_1k_completion_tokens=0.002,
        )

        router.register_model(mock_model)
        assert "test-model" in router.models
        assert router.models["test-model"] == mock_model

    def test_select_model_by_task(self):
        """Test model selection by task type."""
        router = ModelRouter()

        # Register models
        small_model = Mock(spec=BaseModel)
        small_model.config = ModelConfig(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.SMALL,
            max_tokens=4096,
            cost_per_1k_tokens=0.0005,
            cost_per_1k_completion_tokens=0.0015,
        )

        large_model = Mock(spec=BaseModel)
        large_model.config = ModelConfig(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.LARGE,
            max_tokens=4096,
            cost_per_1k_tokens=0.01,
            cost_per_1k_completion_tokens=0.03,
        )

        router.register_model(small_model)
        router.register_model(large_model)

        # Test selection for simple task
        selected = router.select_model("extraction", AnalysisComplexity.LOW)
        assert selected.config.name == "gpt-3.5-turbo"

        # Test selection for complex task
        selected = router.select_model("reasoning", AnalysisComplexity.HIGH)
        assert selected.config.name == "gpt-4-turbo"

    def test_estimate_task_cost(self):
        """Test task cost estimation."""
        router = ModelRouter()

        estimates = router.estimate_task_cost(
            task_type="analysis",
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            complexity=AnalysisComplexity.MEDIUM,
        )

        # Should have estimates for configured models
        assert isinstance(estimates, dict)
        assert len(estimates) > 0

    def test_get_usage_report(self):
        """Test usage report generation."""
        router = ModelRouter()

        # Register a model with usage
        model = Mock(spec=BaseModel)
        # Create a proper config mock
        config = Mock()
        config.name = "test-model"
        model.config = config
        model.get_usage_summary.return_value = {
            "total_cost": 10.0,
            "total_requests": 5,
            "cost_by_task": {"analysis": 6.0, "extraction": 4.0},
        }

        router.register_model(model)

        report = router.get_usage_report()
        assert report["total_cost"] == 10.0
        assert report["total_requests"] == 5
        assert "test-model" in report["by_model"]
        assert "analysis" in report["by_task_type"]


class TestCostController:
    """Tests for CostController class."""

    def test_cost_controller_initialization(self):
        """Test cost controller initialization."""
        controller = CostController(budget_limit=100.0)
        assert controller.budget_limit == 100.0
        assert controller.spent == 0.0
        assert controller.reservations == {}

    def test_check_budget(self):
        """Test budget checking."""
        controller = CostController(budget_limit=10.0)

        assert controller.check_budget(5.0)
        assert controller.check_budget(10.0)
        assert not controller.check_budget(11.0)

        # After spending
        controller.spent = 7.0
        assert controller.check_budget(3.0)
        assert not controller.check_budget(4.0)

    def test_reserve_budget(self):
        """Test budget reservation."""
        controller = CostController(budget_limit=10.0)

        # Successful reservation
        assert controller.reserve_budget("task1", 5.0)
        assert "task1" in controller.reservations
        assert controller.reservations["task1"] == 5.0

        # Failed reservation (exceeds remaining budget, not total)
        # After reserving 5.0, only 5.0 remains, so 6.0 should fail
        assert not controller.reserve_budget("task2", 6.0)
        assert "task2" not in controller.reservations

        # But 4.0 should succeed
        assert controller.reserve_budget("task3", 4.0)
        assert "task3" in controller.reservations

    def test_commit_cost(self):
        """Test cost commitment."""
        controller = CostController(budget_limit=10.0)

        # Reserve and commit
        controller.reserve_budget("task1", 5.0)
        controller.commit_cost("task1", 4.5)

        assert controller.spent == 4.5
        assert "task1" not in controller.reservations

    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        controller = CostController(budget_limit=100.0)

        assert controller.get_remaining_budget() == 100.0

        # After spending
        controller.spent = 30.0
        assert controller.get_remaining_budget() == 70.0

        # With reservations
        controller.reserve_budget("task1", 20.0)
        assert controller.get_remaining_budget() == 50.0

    def test_get_budget_summary(self):
        """Test budget summary."""
        controller = CostController(budget_limit=100.0)
        controller.spent = 30.0
        controller.reserve_budget("task1", 20.0)

        summary = controller.get_budget_summary()
        assert summary["limit"] == 100.0
        assert summary["spent"] == 30.0
        assert summary["reserved"] == 20.0
        assert summary["available"] == 50.0
        assert summary["utilization"] == 0.3
