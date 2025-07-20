"""
Tests for base stage abstractions.

This test module validates that the base stage abstractions work correctly
and provide the expected functionality for all pipeline stages.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from spreadsheet_analyzer.pipeline.base_stage import (
    BaseStage,
    ComposableStage,
    FileProcessingStage,
    StageConfig,
    create_stage_chain,
)
from spreadsheet_analyzer.pipeline.stages.stage_3_formulas_base import (
    FormulaAnalysisStage,
    FormulaStageConfig,
    create_formula_stage,
)
from spreadsheet_analyzer.pipeline.types import Err, Ok, PipelineContext, Result
from tests.base_test import BaseSpreadsheetTest, ExcelTestDataBuilder

# ============================================================================
# TEST STAGE IMPLEMENTATIONS
# ============================================================================


@dataclass(frozen=True)
class TestConfig(StageConfig):
    """Test configuration."""

    multiply_by: int = 2
    add_value: int = 0


class SimpleTestStage(BaseStage[int, int, TestConfig]):
    """Simple test stage that transforms integers."""

    def __init__(self, config: TestConfig | None = None):
        super().__init__("test_stage", config)

    def _get_default_config(self) -> TestConfig:
        return TestConfig()

    def _validate_input(self, input_data: int) -> Result:
        if not isinstance(input_data, int):
            return Err(f"Expected int, got {type(input_data).__name__}")
        if input_data < 0:
            return Err("Input must be non-negative")
        return Ok(None)

    def _process(self, input_data: int, context: PipelineContext | None = None) -> Result:
        # Simulate some processing
        time.sleep(0.01)
        result = (input_data * self.config.multiply_by) + self.config.add_value
        self.metrics["items_processed"] = 1
        return Ok(result)

    def _validate_output(self, output: int) -> Result:
        if output > 1000:
            return Err("Output too large")
        return Ok(None)


class FailingTestStage(BaseStage[int, int, TestConfig]):
    """Test stage that always fails."""

    def __init__(self):
        super().__init__("failing_stage")

    def _get_default_config(self) -> TestConfig:
        return TestConfig()

    def _validate_input(self, input_data: int) -> Result:
        return Ok(None)

    def _process(self, input_data: int, context: PipelineContext | None = None) -> Result:
        return Err("Intentional failure for testing")

    def _validate_output(self, output: int) -> Result:
        return Ok(None)


# ============================================================================
# BASE STAGE TESTS
# ============================================================================


class TestBaseStage(BaseSpreadsheetTest):
    """Test suite for base stage functionality."""

    def _get_test_config(self) -> dict[str, Any]:
        return {
            "test_type": "base_stage",
            "timeout": 10.0,
        }

    def test_simple_stage_execution(self) -> None:
        """Test basic stage execution with valid input."""
        stage = SimpleTestStage()
        result = stage.execute(5)

        # Should succeed
        assert isinstance(result, Ok)
        stage_result = result.value

        # Check output
        assert stage_result.output == 10  # 5 * 2
        assert stage_result.validation_passed
        assert len(stage_result.validation_errors) == 0

        # Check metrics
        assert stage_result.metrics.stage_name == "test_stage"
        assert stage_result.metrics.items_processed == 1
        assert stage_result.metrics.execution_time > 0

    def test_stage_with_custom_config(self) -> None:
        """Test stage execution with custom configuration."""
        config = TestConfig(multiply_by=3, add_value=7)
        stage = SimpleTestStage(config)

        result = stage.execute(4)

        assert isinstance(result, Ok)
        assert result.value.output == 19  # (4 * 3) + 7

    def test_input_validation_failure(self) -> None:
        """Test that input validation catches invalid inputs."""
        stage = SimpleTestStage()

        # Test with negative input
        result = stage.execute(-5)

        assert isinstance(result, Err)
        assert "Input validation failed" in result.error
        assert "Input must be non-negative" in result.error

    def test_output_validation_failure(self) -> None:
        """Test that output validation catches invalid outputs."""
        config = TestConfig(multiply_by=100)  # Will create large output
        stage = SimpleTestStage(config)

        result = stage.execute(20)  # 20 * 100 = 2000 > 1000

        assert isinstance(result, Ok)  # Processing succeeds
        assert not result.value.validation_passed
        assert "Output too large" in result.value.validation_errors

    def test_processing_failure(self) -> None:
        """Test handling of processing failures."""
        stage = FailingTestStage()

        result = stage.execute(5)

        assert isinstance(result, Err)
        assert "Processing failed" in result.error
        assert "Intentional failure" in result.error

    def test_progress_tracking(self) -> None:
        """Test that progress callbacks are invoked correctly."""
        progress_updates = []

        def track_progress(stage: str, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
            progress_updates.append({"stage": stage, "progress": progress, "message": message})

        stage = SimpleTestStage()
        result = stage.execute(5, progress_callback=track_progress)

        assert isinstance(result, Ok)

        # Should have progress updates
        assert len(progress_updates) >= 3

        # Check progress values
        assert progress_updates[0]["progress"] == 0.0
        assert progress_updates[-1]["progress"] == 1.0

        # All updates should be for our stage
        assert all(u["stage"] == "test_stage" for u in progress_updates)

    def test_metrics_collection(self) -> None:
        """Test that metrics are collected correctly."""
        stage = SimpleTestStage()

        start_time = time.time()
        result = stage.execute(5)
        end_time = time.time()

        assert isinstance(result, Ok)
        metrics = result.value.metrics

        # Check metric values
        assert metrics.stage_name == "test_stage"
        assert metrics.items_processed == 1
        assert metrics.items_skipped == 0
        assert len(metrics.warnings) == 0

        # Execution time should be reasonable
        assert 0 < metrics.execution_time < (end_time - start_time + 0.1)

        # Throughput should be calculated
        assert metrics.throughput > 0


# ============================================================================
# FILE PROCESSING STAGE TESTS
# ============================================================================


class TestFileProcessingStage(BaseSpreadsheetTest):
    """Test suite for file processing stage functionality."""

    def _get_test_config(self) -> dict[str, Any]:
        return {
            "test_type": "file_processing_stage",
        }

    def test_file_validation(self, tmp_path: Path) -> None:
        """Test file validation in FileProcessingStage."""

        # Create a simple file processing stage
        class TestFileStage(FileProcessingStage[Path, str, StageConfig]):
            def _get_default_config(self) -> StageConfig:
                return StageConfig()

            def _process(self, input_data: Path, context: PipelineContext | None = None) -> Result:
                return Ok(f"Processed {input_data.name}")

            def _validate_output(self, output: str) -> Result:
                return Ok(None)

        stage = TestFileStage("test_file_stage")

        # Test with non-existent file
        result = stage.execute(Path("/does/not/exist.txt"))
        assert isinstance(result, Err)
        assert "File not found" in result.error

        # Test with existing file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = stage.execute(test_file)
        assert isinstance(result, Ok)
        assert result.value.output == "Processed test.txt"

        # Test with directory instead of file
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        result = stage.execute(test_dir)
        assert isinstance(result, Err)
        assert "Not a file" in result.error


# ============================================================================
# COMPOSABLE STAGE TESTS
# ============================================================================


class TestComposableStage(BaseSpreadsheetTest):
    """Test suite for composable stage functionality."""

    def _get_test_config(self) -> dict[str, Any]:
        return {
            "test_type": "composable_stage",
        }

    def test_pre_and_post_processors(self) -> None:
        """Test pre and post processor functionality."""

        class ComposableTestStage(ComposableStage[int, int, TestConfig]):
            def _get_default_config(self) -> TestConfig:
                return TestConfig()

            def _validate_input(self, input_data: int) -> Result:
                return Ok(None)

            def _process(self, input_data: int, context: PipelineContext | None = None) -> Result:
                # Apply pre-processors
                processed_input = self._apply_pre_processors(input_data)

                # Core processing
                result = processed_input * 2

                # Apply post-processors
                final_result = self._apply_post_processors(result)

                return Ok(final_result)

            def _validate_output(self, output: int) -> Result:
                return Ok(None)

        # Create stage with processors
        stage = (
            ComposableTestStage("composable_test")
            .with_pre_processor(lambda x: x + 10)  # Add 10 before
            .with_pre_processor(lambda x: x * 2)  # Then multiply by 2
            .with_post_processor(lambda x: x - 5)  # Subtract 5 after
        )

        result = stage.execute(5)

        assert isinstance(result, Ok)
        # (5 + 10) * 2 * 2 - 5 = 15 * 2 * 2 - 5 = 60 - 5 = 55
        assert result.value.output == 55


# ============================================================================
# STAGE CHAINING TESTS
# ============================================================================


class TestStageChaining(BaseSpreadsheetTest):
    """Test suite for stage chaining functionality."""

    def _get_test_config(self) -> dict[str, Any]:
        return {
            "test_type": "stage_chaining",
        }

    def test_successful_chain(self) -> None:
        """Test chaining multiple stages successfully."""
        stage1 = SimpleTestStage(TestConfig(multiply_by=2))
        stage2 = SimpleTestStage(TestConfig(multiply_by=3))
        stage3 = SimpleTestStage(TestConfig(add_value=10))

        chain = create_stage_chain([stage1, stage2, stage3])

        result = chain(5, None)

        assert isinstance(result, Ok)
        # 5 * 2 = 10, 10 * 3 = 30, 30 * 2 + 10 = 70
        assert result.value == 70

    def test_chain_with_failure(self) -> None:
        """Test that chain stops on first failure."""
        stage1 = SimpleTestStage()
        stage2 = FailingTestStage()
        stage3 = SimpleTestStage()

        chain = create_stage_chain([stage1, stage2, stage3])

        result = chain(5, None)

        assert isinstance(result, Err)
        assert "Intentional failure" in result.error


# ============================================================================
# FORMULA STAGE WITH BASE TESTS
# ============================================================================


class TestFormulaStageWithBase(BaseSpreadsheetTest):
    """Test formula analysis stage using base abstractions."""

    def _get_test_config(self) -> dict[str, Any]:
        return {
            "test_type": "formula_stage_base",
        }

    @pytest.mark.requires_excel
    def test_formula_stage_execution(self, excel_builder: ExcelTestDataBuilder, tmp_path: Path) -> None:
        """Test formula analysis using base stage abstraction."""
        # Create test workbook
        test_file = (
            excel_builder.with_sheet("Data")
            .add_headers(["Value", "Double", "Total"])
            .add_row([10, "=A2*2", "=SUM(A2:B2)"])
            .add_row([20, "=A3*2", "=SUM(A3:B3)"])
            .add_row([30, "=A4*2", "=SUM(A4:B4)"])
            .build(tmp_path / "test.xlsx")
        )

        # Create and execute stage
        stage = create_formula_stage(enable_validation=True)
        result = stage.execute(test_file)

        # Check success
        assert isinstance(result, Ok)
        stage_result = result.value

        # Check that we got formula analysis
        analysis = stage_result.output
        assert len(analysis.dependency_graph) == 6  # 6 formulas
        assert analysis.max_dependency_depth >= 1

        # Check metrics
        assert stage_result.metrics.items_processed == 6
        assert stage_result.metrics.stage_name == "formula_analysis"

        # Check validation passed
        assert stage_result.validation_passed

    @pytest.mark.requires_excel
    def test_formula_stage_with_progress(self, sample_excel_file: Path) -> None:
        """Test formula stage with progress tracking."""
        progress_updates = []

        def track_progress(stage: str, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
            progress_updates.append(progress)

        config = FormulaStageConfig(enable_progress=True)
        stage = FormulaAnalysisStage(config)

        result = stage.execute(sample_excel_file, progress_callback=track_progress)

        assert isinstance(result, Ok)

        # Should have multiple progress updates
        assert len(progress_updates) > 2

        # Progress should increase
        assert progress_updates[0] < progress_updates[-1]

        # Should reach 100%
        assert progress_updates[-1] == 1.0

    def test_formula_stage_validation(self, tmp_path: Path) -> None:
        """Test formula stage input validation."""
        stage = create_formula_stage()

        # Test with non-Excel file
        text_file = tmp_path / "not_excel.txt"
        text_file.write_text("not an excel file")

        result = stage.execute(text_file)

        assert isinstance(result, Err)
        assert "Not an Excel file" in result.error

    @pytest.mark.requires_excel
    def test_formula_stage_summary(self, sample_excel_file: Path) -> None:
        """Test getting summary from formula analysis."""
        stage = create_formula_stage()
        result = stage.execute(sample_excel_file)

        assert isinstance(result, Ok)
        analysis = result.value.output

        # Get summary
        summary = stage.get_summary(analysis)

        # Check summary contents
        assert "total_formulas" in summary
        assert "circular_references" in summary
        assert "recommendations" in summary
        assert isinstance(summary["recommendations"], list)
