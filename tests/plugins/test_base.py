"""
Tests for the plugin base interfaces and registry system.

This test suite validates the plugin system functionality including:
- Task protocol implementation
- QualityInspector protocol implementation
- PluginRegistry functionality
- Plugin registration and discovery
- Task execution and cell generation
- Quality inspection extension

Following TDD principles with functional tests - no mocking used.
All tests use real implementations and integration testing.
"""

from typing import Any

import pytest

from spreadsheet_analyzer.core_exec import (
    CellType,
    NotebookBuilder,
    NotebookCell,
    QualityIssue,
    QualityLevel,
    QualityMetrics,
)
from spreadsheet_analyzer.plugins.base import (
    BaseQualityInspector,
    BaseTask,
    PluginRegistry,
)


# Test implementations for the protocol interfaces
class SimpleTestTask(BaseTask):
    """Simple test task implementation."""

    def __init__(self, name: str = "simple_test_task"):
        super().__init__(name, "Simple test task for validation")

    def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
        """Generate simple test cells."""
        cells = []

        # Add header
        cells.append(
            NotebookCell(
                cell_type=CellType.MARKDOWN,
                source=[f"# {self.name.title()} Analysis"],
                metadata={"tags": ["test", "header"]},
            )
        )

        # Add setup code
        cells.append(
            NotebookCell(
                cell_type=CellType.CODE,
                source=["# Simple test setup\ntest_data = [1, 2, 3, 4, 5]"],
                metadata={"tags": ["setup"]},
            )
        )

        return cells

    def postprocess(self, notebook: NotebookBuilder, context: dict[str, Any]) -> list[NotebookCell]:
        """Add summary cells after execution."""
        return [
            NotebookCell(
                cell_type=CellType.MARKDOWN,
                source=["## Test Summary\nSimple test task completed successfully."],
                metadata={"tags": ["summary"]},
            )
        ]


class AdvancedTestTask(BaseTask):
    """Advanced test task with context handling."""

    def __init__(self):
        super().__init__("advanced_test_task", "Advanced test task with context handling")

    def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
        """Generate cells based on context."""
        cells = []

        # Check context for configuration
        data_source = context.get("data_source", "default")
        analysis_type = context.get("analysis_type", "basic")

        # Header with context info
        cells.append(
            NotebookCell(
                cell_type=CellType.MARKDOWN,
                source=[f"# Advanced Analysis: {analysis_type.title()}"],
                metadata={"context": context},
            )
        )

        # Conditional code generation based on context
        if data_source == "file":
            cells.append(
                NotebookCell(
                    cell_type=CellType.CODE,
                    source=["# Load data from file\nwith open('data.txt') as f:\n    data = f.read()"],
                    metadata={"data_source": "file"},
                )
            )
        else:
            cells.append(
                NotebookCell(
                    cell_type=CellType.CODE,
                    source=["# Generate synthetic data\ndata = list(range(100))"],
                    metadata={"data_source": "synthetic"},
                )
            )

        # Analysis code based on type
        if analysis_type == "statistical":
            cells.append(
                NotebookCell(
                    cell_type=CellType.CODE,
                    source=["# Statistical analysis\nimport statistics\nmean = statistics.mean(data)"],
                    metadata={"analysis": "statistical"},
                )
            )
        elif analysis_type == "visualization":
            cells.append(
                NotebookCell(
                    cell_type=CellType.CODE,
                    source=["# Create visualization\nimport matplotlib.pyplot as plt\nplt.plot(data)"],
                    metadata={"analysis": "visualization"},
                )
            )

        return cells


class TestQualityInspector(BaseQualityInspector):
    """Test implementation of quality inspector."""

    def __init__(self, name: str = "test_inspector", description: str = "Test quality inspector"):
        super().__init__(name, description)

    def inspect(self, notebook: NotebookBuilder, context: dict[str, Any]) -> QualityMetrics:
        """Perform test-specific quality inspection."""
        issues = []

        # Check for test-specific patterns
        test_cells = 0
        for i, cell in enumerate(notebook.cells):
            if "test" in str(cell.source).lower():
                test_cells += 1

            # Check for missing test documentation
            if cell.cell_type == CellType.CODE and len(cell.source) > 0:
                if not any("#" in line for line in cell.source):
                    issues.append(
                        QualityIssue(
                            category="test_quality",
                            severity="warning",
                            message="Code cell lacks comments for test clarity",
                            cell_index=i,
                            suggestion="Add comments for better test documentation",
                        )
                    )

        # Calculate test-specific score
        base_score = 50.0
        if test_cells > 0:
            base_score += min(test_cells * 10, 30)

        if issues:
            base_score -= len(issues) * 5

        base_score = max(0.0, min(100.0, base_score))

        # Determine overall level
        if base_score >= 80:
            level = QualityLevel.EXCELLENT
        elif base_score >= 60:
            level = QualityLevel.GOOD
        elif base_score >= 40:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR

        return QualityMetrics(
            overall_score=base_score,
            overall_level=level,
            total_cells=notebook.cell_count(),
            code_cells=notebook.code_cell_count(),
            markdown_cells=notebook.markdown_cell_count(),
            empty_cells=0,
            cells_with_outputs=0,
            cells_with_errors=0,
            avg_cell_length=10.0,
            issues=issues,
            metrics={"test_quality": base_score},
        )


class TestPluginRegistry:
    """Test PluginRegistry functionality."""

    def test_registry_creation(self) -> None:
        """Test creating a plugin registry."""
        registry = PluginRegistry()

        assert len(registry._tasks) == 0
        assert len(registry._quality_inspectors) == 0

    def test_register_task(self) -> None:
        """Test registering a task."""
        registry = PluginRegistry()
        task = SimpleTestTask()

        registry.register_task(task)

        assert len(registry._tasks) == 1
        assert "simple_test_task" in registry._tasks
        assert registry._tasks["simple_test_task"] is task

    def test_register_multiple_tasks(self) -> None:
        """Test registering multiple tasks."""
        registry = PluginRegistry()
        task1 = SimpleTestTask("task_one")
        task2 = AdvancedTestTask()

        registry.register_task(task1)
        registry.register_task(task2)

        assert len(registry._tasks) == 2
        assert "task_one" in registry._tasks
        assert "advanced_test_task" in registry._tasks

    def test_register_duplicate_task(self) -> None:
        """Test registering a task with duplicate name."""
        registry = PluginRegistry()
        task1 = SimpleTestTask("duplicate_name")
        task2 = SimpleTestTask("duplicate_name")

        registry.register_task(task1)

        # Should overwrite the first task
        registry.register_task(task2)
        assert registry._tasks["duplicate_name"] is task2

    def test_register_quality_inspector(self) -> None:
        """Test registering a quality inspector."""
        registry = PluginRegistry()
        inspector = TestQualityInspector("test_inspector")

        registry.register_quality_inspector(inspector)

        assert len(registry._quality_inspectors) == 1
        assert "test_inspector" in registry._quality_inspectors
        assert registry._quality_inspectors["test_inspector"] is inspector

    def test_register_duplicate_quality_inspector(self) -> None:
        """Test registering quality inspector with duplicate name."""
        registry = PluginRegistry()
        inspector1 = TestQualityInspector("duplicate_inspector")
        inspector2 = TestQualityInspector("duplicate_inspector")

        registry.register_quality_inspector(inspector1)

        # Should overwrite the first inspector
        registry.register_quality_inspector(inspector2)
        assert registry._quality_inspectors["duplicate_inspector"] is inspector2

    def test_get_task(self) -> None:
        """Test retrieving a registered task."""
        registry = PluginRegistry()
        task = SimpleTestTask()

        registry.register_task(task)

        retrieved_task = registry.get_task("simple_test_task")
        assert retrieved_task is task

    def test_get_nonexistent_task(self) -> None:
        """Test retrieving a task that doesn't exist."""
        registry = PluginRegistry()

        task = registry.get_task("nonexistent_task")
        assert task is None

    def test_get_quality_inspector(self) -> None:
        """Test retrieving a registered quality inspector."""
        registry = PluginRegistry()
        inspector = TestQualityInspector("test_inspector")

        registry.register_quality_inspector(inspector)

        retrieved_inspector = registry.get_quality_inspector("test_inspector")
        assert retrieved_inspector is inspector

    def test_get_nonexistent_quality_inspector(self) -> None:
        """Test retrieving a quality inspector that doesn't exist."""
        registry = PluginRegistry()

        inspector = registry.get_quality_inspector("nonexistent_inspector")
        assert inspector is None

    def test_list_tasks(self) -> None:
        """Test listing all registered tasks."""
        registry = PluginRegistry()
        task1 = SimpleTestTask("task_one")
        task2 = AdvancedTestTask()

        registry.register_task(task1)
        registry.register_task(task2)

        tasks = registry.list_tasks()
        assert len(tasks) == 2
        task_names = [t.name for t in tasks]
        assert "task_one" in task_names
        assert "advanced_test_task" in task_names

    def test_list_quality_inspectors(self) -> None:
        """Test listing all registered quality inspectors."""
        registry = PluginRegistry()
        inspector1 = TestQualityInspector("inspector_one")
        inspector2 = TestQualityInspector("inspector_two")

        registry.register_quality_inspector(inspector1)
        registry.register_quality_inspector(inspector2)

        inspectors = registry.list_quality_inspectors()
        assert len(inspectors) == 2
        inspector_names = [i.name for i in inspectors]
        assert "inspector_one" in inspector_names
        assert "inspector_two" in inspector_names

    def test_clear_registry(self) -> None:
        """Test clearing all registered plugins."""
        registry = PluginRegistry()

        # Register some plugins
        registry.register_task(SimpleTestTask())
        registry.register_quality_inspector(TestQualityInspector("test"))

        assert len(registry._tasks) == 1
        assert len(registry._quality_inspectors) == 1

        # Clear registry
        registry.clear()

        assert len(registry._tasks) == 0
        assert len(registry._quality_inspectors) == 0


class TestTaskProtocol:
    """Test Task protocol implementation."""

    def test_simple_task_cell_generation(self) -> None:
        """Test simple task generates expected cells."""
        task = SimpleTestTask()
        context = {}

        cells = task.build_initial_cells(context)

        assert len(cells) == 2

        # Check first cell (markdown header)
        assert cells[0].cell_type == CellType.MARKDOWN
        assert "Simple_Test_Task Analysis" in cells[0].source[0]
        assert "test" in cells[0].metadata["tags"]

        # Check second cell (code setup)
        assert cells[1].cell_type == CellType.CODE
        assert "test_data = [1, 2, 3, 4, 5]" in cells[1].source[0]

    def test_simple_task_postprocessing(self) -> None:
        """Test simple task postprocessing."""
        task = SimpleTestTask()
        notebook = NotebookBuilder()

        # Add some initial cells
        notebook.add_markdown_cell("# Initial")
        notebook.add_code_cell("x = 1")

        # Run postprocessing
        additional_cells = task.postprocess(notebook, {})

        assert len(additional_cells) == 1
        assert additional_cells[0].cell_type == CellType.MARKDOWN
        assert "Test Summary" in additional_cells[0].source[0]

    def test_advanced_task_context_handling(self) -> None:
        """Test advanced task handles context appropriately."""
        task = AdvancedTestTask()

        # Test with file data source
        context = {"data_source": "file", "analysis_type": "statistical"}
        cells = task.build_initial_cells(context)

        assert len(cells) == 3
        assert "Advanced Analysis: Statistical" in cells[0].source[0]
        assert "Load data from file" in cells[1].source[0]
        assert "Statistical analysis" in cells[2].source[0]

        # Test with synthetic data source and visualization
        context = {"data_source": "synthetic", "analysis_type": "visualization"}
        cells = task.build_initial_cells(context)

        assert len(cells) == 3
        assert "Advanced Analysis: Visualization" in cells[0].source[0]
        assert "Generate synthetic data" in cells[1].source[0]
        assert "Create visualization" in cells[2].source[0]

    def test_advanced_task_default_context(self) -> None:
        """Test advanced task with default context values."""
        task = AdvancedTestTask()
        context = {}  # Empty context, should use defaults

        cells = task.build_initial_cells(context)

        assert len(cells) >= 2
        assert "Advanced Analysis: Basic" in cells[0].source[0]
        assert "Generate synthetic data" in cells[1].source[0]  # Default data source

    def test_task_integration_with_notebook_builder(self) -> None:
        """Test task integration with NotebookBuilder."""
        task = SimpleTestTask()
        notebook = NotebookBuilder()

        # Generate cells from task
        cells = task.build_initial_cells({})

        # Add cells to notebook
        for cell in cells:
            if cell.cell_type == CellType.MARKDOWN:
                notebook.add_markdown_cell("".join(cell.source), cell.metadata)
            elif cell.cell_type == CellType.CODE:
                notebook.add_code_cell("".join(cell.source), metadata=cell.metadata)

        # Check notebook structure
        assert len(notebook.cells) == 2
        assert notebook.markdown_cell_count() == 1
        assert notebook.code_cell_count() == 1

        # Check metadata preservation
        assert "test" in notebook.cells[0].metadata["tags"]
        assert "setup" in notebook.cells[1].metadata["tags"]


class TestQualityInspectorProtocol:
    """Test QualityInspector protocol implementation."""

    def test_test_quality_inspector_basic(self) -> None:
        """Test basic functionality of test quality inspector."""
        inspector = TestQualityInspector("test_inspector")
        notebook = NotebookBuilder()

        # Create notebook with some test content
        notebook.add_markdown_cell("# Test Analysis")
        notebook.add_code_cell("# Test code with comment\ntest_result = True")

        metrics = inspector.inspect(notebook, {})

        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_score >= 0.0
        assert isinstance(metrics.overall_level, QualityLevel)

    def test_quality_inspector_detects_issues(self) -> None:
        """Test that quality inspector detects specific issues."""
        inspector = TestQualityInspector("test_inspector")
        notebook = NotebookBuilder()

        # Create problematic notebook
        notebook.add_code_cell("x = 1")  # No comments
        notebook.add_code_cell("y = 2")  # No comments

        metrics = inspector.inspect(notebook, {})

        # Should detect lack of comments
        assert len(metrics.issues) > 0
        assert any(issue.category == "test_quality" for issue in metrics.issues)
        assert any("lacks comments" in issue.message for issue in metrics.issues)

    def test_quality_inspector_scoring(self) -> None:
        """Test quality inspector scoring logic."""
        inspector = TestQualityInspector("test_inspector")

        # Good notebook with test content and comments
        good_notebook = NotebookBuilder()
        good_notebook.add_markdown_cell("# Test Suite")
        good_notebook.add_code_cell("# Test function\ndef test_function():\n    assert True")

        # Poor notebook without test content or comments
        poor_notebook = NotebookBuilder()
        poor_notebook.add_code_cell("x = 1")
        poor_notebook.add_code_cell("y = 2")

        good_metrics = inspector.inspect(good_notebook, {})
        poor_metrics = inspector.inspect(poor_notebook, {})

        # Good notebook should score higher
        assert good_metrics.overall_score > poor_metrics.overall_score

    def test_quality_inspector_recommendations(self) -> None:
        """Test that quality inspector provides recommendations."""
        inspector = TestQualityInspector("test_inspector")
        notebook = NotebookBuilder()

        # Create notebook that needs improvement
        notebook.add_code_cell("x = 1")
        notebook.add_code_cell("y = 2")

        metrics = inspector.inspect(notebook, {})

        # Should provide recommendations
        assert len(metrics.issues) > 0
        assert any("test documentation" in issue.suggestion.lower() for issue in metrics.issues if issue.suggestion)


class TestPluginIntegration:
    """Test integration between plugins and core components."""

    def test_end_to_end_task_workflow(self) -> None:
        """Test complete workflow from plugin registration to notebook generation."""
        # Setup registry and register plugins
        registry = PluginRegistry()
        task = SimpleTestTask("integration_test")
        registry.register_task(task)

        # Retrieve task and generate cells
        retrieved_task = registry.get_task("integration_test")
        context = {"project": "test_project"}
        cells = retrieved_task.build_initial_cells(context)

        # Build notebook with generated cells
        notebook = NotebookBuilder()
        for cell in cells:
            if cell.cell_type == CellType.MARKDOWN:
                notebook.add_markdown_cell("".join(cell.source), cell.metadata)
            elif cell.cell_type == CellType.CODE:
                notebook.add_code_cell("".join(cell.source), metadata=cell.metadata)

        # Add postprocessing cells
        additional_cells = retrieved_task.postprocess(notebook, context)
        for cell in additional_cells:
            if cell.cell_type == CellType.MARKDOWN:
                notebook.add_markdown_cell("".join(cell.source), cell.metadata)

        # Verify final notebook
        assert len(notebook.cells) == 3  # 2 initial + 1 postprocess
        assert notebook.markdown_cell_count() == 2
        assert notebook.code_cell_count() == 1

    def test_multiple_plugins_workflow(self) -> None:
        """Test workflow with multiple plugins working together."""
        registry = PluginRegistry()

        # Register multiple tasks
        task1 = SimpleTestTask("task_one")
        task2 = AdvancedTestTask()
        inspector = TestQualityInspector("quality_check")

        registry.register_task(task1)
        registry.register_task(task2)
        registry.register_quality_inspector(inspector)

        # Use first task to build notebook
        notebook = NotebookBuilder()
        cells1 = task1.build_initial_cells({"phase": "setup"})

        for cell in cells1:
            if cell.cell_type == CellType.MARKDOWN:
                notebook.add_markdown_cell("".join(cell.source), cell.metadata)
            elif cell.cell_type == CellType.CODE:
                notebook.add_code_cell("".join(cell.source), metadata=cell.metadata)

        # Use second task to add more cells
        cells2 = task2.build_initial_cells({"analysis_type": "statistical"})

        for cell in cells2:
            if cell.cell_type == CellType.MARKDOWN:
                notebook.add_markdown_cell("".join(cell.source), cell.metadata)
            elif cell.cell_type == CellType.CODE:
                notebook.add_code_cell("".join(cell.source), metadata=cell.metadata)

        # Run quality inspection
        quality_metrics = inspector.inspect(notebook, {})

        # Verify integration worked
        assert len(notebook.cells) > 3  # Should have cells from both tasks
        assert quality_metrics.overall_score > 0
        assert isinstance(quality_metrics.overall_level, QualityLevel)

    def test_plugin_error_handling(self) -> None:
        """Test error handling in plugin operations."""
        registry = PluginRegistry()

        # Test graceful handling of task errors
        class FaultyTask(BaseTask):
            def __init__(self):
                super().__init__("faulty_task", "Task that raises errors")

            def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
                raise ValueError("Simulated task error")

        faulty_task = FaultyTask()
        registry.register_task(faulty_task)

        # Should be able to retrieve the task
        retrieved_task = registry.get_task("faulty_task")
        assert retrieved_task is faulty_task

        # But calling it should raise the error
        with pytest.raises(ValueError, match="Simulated task error"):
            retrieved_task.build_initial_cells({})

    def test_plugin_extensibility(self) -> None:
        """Test that plugin system is extensible with custom implementations."""

        # Create custom task with unique functionality
        class CustomAnalysisTask(BaseTask):
            def __init__(self):
                super().__init__("custom_analysis", "Custom analysis with configurable steps")

            def build_initial_cells(self, context: dict[str, Any]) -> list[NotebookCell]:
                cells = []

                # Custom cell generation logic
                analysis_steps = context.get("steps", ["load", "analyze", "report"])

                cells.append(
                    NotebookCell(
                        cell_type=CellType.MARKDOWN,
                        source=[f"# Custom Analysis with {len(analysis_steps)} steps"],
                        metadata={"custom": True},
                    )
                )

                for i, step in enumerate(analysis_steps):
                    cells.append(
                        NotebookCell(
                            cell_type=CellType.CODE,
                            source=[f"# Step {i + 1}: {step.title()}\nstep_{i + 1} = '{step}'"],
                            metadata={"step": step, "order": i + 1},
                        )
                    )

                return cells

        # Register and use custom task
        registry = PluginRegistry()
        custom_task = CustomAnalysisTask()
        registry.register_task(custom_task)

        context = {"steps": ["prepare", "transform", "validate", "output"]}
        cells = custom_task.build_initial_cells(context)

        # Should generate expected structure
        assert len(cells) == 5  # 1 header + 4 steps
        assert cells[0].cell_type == CellType.MARKDOWN
        assert "4 steps" in cells[0].source[0]

        # Check step cells
        for i in range(1, 5):
            assert cells[i].cell_type == CellType.CODE
            assert f"Step {i}" in cells[i].source[0]
            assert cells[i].metadata["order"] == i
