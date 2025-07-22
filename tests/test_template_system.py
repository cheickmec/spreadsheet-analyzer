"""Tests for the Jinja2 template system."""

import pytest
from jinja2 import TemplateNotFound

from spreadsheet_analyzer.notebook_llm.templates import (
    StrategyTemplateLoader,
    TemplateManager,
    get_template_manager,
    render_prompt,
)


@pytest.fixture
def template_manager():
    """Create a template manager instance."""
    return TemplateManager()


@pytest.fixture
def strategy_loader(template_manager):
    """Create a strategy template loader."""
    return StrategyTemplateLoader(template_manager)


class TestTemplateManager:
    """Test the TemplateManager class."""

    def test_initialization(self, template_manager):
        """Test template manager initialization."""
        assert template_manager is not None
        assert len(template_manager.template_dirs) > 0
        assert template_manager.env is not None

    def test_get_template(self, template_manager):
        """Test loading templates."""
        # Test loading base template
        template = template_manager.get_template("base/master.jinja2")
        assert template is not None

        # Test loading strategy template
        template = template_manager.get_template("strategies/hierarchical/exploration.jinja2")
        assert template is not None

    def test_template_not_found(self, template_manager):
        """Test handling of missing templates."""
        with pytest.raises(TemplateNotFound):
            template_manager.get_template("nonexistent/template.jinja2")

    def test_render_basic(self, template_manager):
        """Test basic template rendering."""
        context = {
            "task": {"description": "Analyze sales data"},
            "excel_metadata": {
                "filename": "sales_2024.xlsx",
                "size_mb": 2.5,
                "sheet_count": 5,
                "sheet_names": ["Summary", "Q1", "Q2", "Q3", "Q4"],
                "total_cells": 10000,
                "formula_count": 500,
                "last_modified": "2024-01-15",
            },
        }

        result = template_manager.render("strategies/default/analysis.jinja2", context)

        assert "Analyze sales data" in result
        assert "sales_2024.xlsx" in result
        assert "5" in result  # sheet count

    def test_render_with_inheritance(self, template_manager):
        """Test template inheritance."""
        context = {
            "task": {"description": "Explore workbook structure"},
            "exploration_level": "overview",
            "target_sheet": "Summary",
        }

        result = template_manager.render("strategies/hierarchical/exploration.jinja2", context)

        # Check that base template content is included
        assert "You are analyzing an Excel spreadsheet" in result

        # Check strategy-specific content
        assert "hierarchical exploration" in result
        assert "Level 1: Overview Analysis" in result

    def test_custom_filters(self, template_manager):
        """Test custom Jinja2 filters."""
        template_str = """
        {{ long_text | truncate_middle(20) }}
        {{ 'Sheet1' | format_cell_ref('A1') }}
        {{ 1048576 | format_bytes }}
        """

        context = {"long_text": "This is a very long text that should be truncated in the middle"}

        result = template_manager.render_string(template_str, context)

        # Check for truncate_middle output
        assert "..." in result
        assert "This is" in result
        assert "middle" in result

        # Check for format_cell_ref output
        assert "A1" in result
        assert "Sheet1" in result

        # Check for format_bytes output
        assert "1.0 MB" in result

    def test_list_templates(self, template_manager):
        """Test listing available templates."""
        templates = template_manager.list_templates()

        assert len(templates) > 0
        assert "base/master.jinja2" in templates
        assert "strategies/hierarchical/exploration.jinja2" in templates

        # Test with prefix
        strategy_templates = template_manager.list_templates("strategies/")
        assert all(t.startswith("strategies/") for t in strategy_templates)


class TestStrategyTemplateLoader:
    """Test the StrategyTemplateLoader class."""

    def test_load_strategy_template(self, strategy_loader):
        """Test loading strategy-specific templates."""
        template = strategy_loader.load_strategy_template("hierarchical", "exploration")
        assert template is not None

    def test_render_strategy_prompt(self, strategy_loader):
        """Test rendering strategy prompts."""
        context = {
            "task": {"description": "Find calculation errors"},
            "exploration_level": "sheet",
            "target_sheet": "Calculations",
        }

        result = strategy_loader.render_strategy_prompt("hierarchical", "exploration", context)

        assert "Find calculation errors" in result
        assert "Level 2: Sheet-Level Analysis" in result
        assert "Calculations" in result

    def test_fallback_to_default(self, strategy_loader):
        """Test fallback to default template."""
        # This should fall back to default since the template doesn't exist
        template = strategy_loader.load_strategy_template("nonexistent_strategy", "analysis")
        assert template is not None


class TestTemplateMacros:
    """Test Jinja2 macros in templates."""

    def test_context_macros(self, template_manager):
        """Test context formatting macros."""
        template_str = """
        {% from 'base/components/context.jinja2' import format_excel_metadata %}
        {{ format_excel_metadata(metadata) }}
        """

        context = {
            "metadata": {
                "filename": "test.xlsx",
                "size_mb": 1.5,
                "sheet_count": 3,
                "sheet_names": ["Data", "Analysis", "Charts"],
                "total_cells": 5000,
                "formula_count": 200,
                "last_modified": "2024-01-10",
            }
        }

        result = template_manager.render_string(template_str, context)

        assert "test.xlsx" in result
        assert "1.5 MB" in result
        assert "3" in result
        assert "Data, Analysis, Charts" in result

    def test_validation_macros(self, template_manager):
        """Test validation prompt macros."""
        template_str = """
        {% from 'base/components/validation.jinja2' import validation_checklist %}
        {{ validation_checklist() }}
        """

        result = template_manager.render_string(template_str, {})

        assert "Validation Requirements" in result
        assert "Verified all numeric calculations" in result
        assert "Checked formula consistency" in result


def test_singleton_manager():
    """Test singleton template manager."""
    manager1 = get_template_manager()
    manager2 = get_template_manager()

    assert manager1 is manager2


def test_render_prompt_convenience():
    """Test convenience render function."""
    context = {"task": {"description": "Quick analysis"}}

    result = render_prompt("strategies/default/analysis.jinja2", context)

    assert "Quick analysis" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
