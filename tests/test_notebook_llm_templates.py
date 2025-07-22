"""Tests for notebook_llm templates module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from spreadsheet_analyzer.notebook_llm.templates import (
    StrategyTemplateLoader,
    TemplateManager,
    get_template_manager,
    render_prompt,
)


class TestTemplateManager:
    """Tests for TemplateManager class."""

    @pytest.fixture
    def manager(self):
        """Create template manager instance."""
        return TemplateManager()

    def test_initialization(self, manager):
        """Test template manager initialization."""
        assert hasattr(manager, "templates")
        assert hasattr(manager, "template_dir")
        assert hasattr(manager, "jinja_env")
        assert manager.template_dir.name == "templates"

    def test_load_templates(self, manager):
        """Test loading templates."""
        # Mock the template directory structure
        with patch.object(Path, "glob") as mock_glob:
            mock_glob.return_value = []
            manager._load_templates()
            # Should handle empty template directory
            assert isinstance(manager.templates, dict)

    def test_get_template_existing(self, manager):
        """Test getting an existing template."""
        # Add a mock template
        manager.templates["test_template"] = {"name": "test", "content": "Test content"}

        template = manager.get_template("test_template")
        assert template is not None
        assert template["name"] == "test"

    def test_get_template_nonexistent(self, manager):
        """Test getting a non-existent template."""
        template = manager.get_template("nonexistent")
        assert template is None

    def test_render_template_jinja(self, manager):
        """Test rendering a Jinja2 template."""
        # Mock Jinja2 environment
        mock_template = MagicMock()
        mock_template.render.return_value = "Rendered content"
        manager.jinja_env.get_template = MagicMock(return_value=mock_template)

        rendered = manager.render_template("test.j2", context={"var": "value"})
        assert rendered == "Rendered content"
        mock_template.render.assert_called_once_with({"var": "value"})

    def test_render_template_dict(self, manager):
        """Test rendering a dictionary template."""
        manager.templates["test_template"] = {
            "system": "You are {role}",
            "task": "Analyze {target}",
        }

        rendered = manager.render_template("test_template", role="an expert", target="formulas")
        assert "You are an expert" in rendered
        assert "Analyze formulas" in rendered

    def test_list_templates(self, manager):
        """Test listing available templates."""
        manager.templates = {
            "template1": {"name": "Template 1"},
            "template2": {"name": "Template 2"},
        }

        templates = manager.list_templates()
        assert len(templates) == 2
        assert "template1" in templates
        assert "template2" in templates

    def test_validate_template(self, manager):
        """Test template validation."""
        # Valid template
        valid_template = {"system": "System prompt", "task": "Task prompt"}
        assert manager.validate_template(valid_template)

        # Invalid template (missing required fields)
        invalid_template = {"system": "Only system"}
        assert not manager.validate_template(invalid_template)

    def test_get_template_for_strategy(self, manager):
        """Test getting template for a specific strategy."""
        manager.templates["graph_based_formulas"] = {
            "system": "Graph analysis system",
            "task": "Analyze formula graph",
        }

        template = manager.get_template_for_strategy("graph_based", "formulas")
        assert template is not None

        # Test fallback to default
        template = manager.get_template_for_strategy("unknown", "formulas")
        # Should return None or default template


class TestStrategyTemplateLoader:
    """Tests for StrategyTemplateLoader class."""

    @pytest.fixture
    def loader(self):
        """Create template loader instance."""
        return StrategyTemplateLoader()

    def test_initialization(self, loader):
        """Test template loader initialization."""
        assert hasattr(loader, "strategy_templates")
        assert isinstance(loader.strategy_templates, dict)

    def test_load_strategy_template(self, loader):
        """Test loading a strategy-specific template."""
        # Mock template content
        template_content = {
            "base": {
                "system": "You are analyzing spreadsheets",
                "constraints": "Be concise and accurate",
            },
            "focus_specific": {
                "formulas": {"task": "Analyze formulas"},
                "data": {"task": "Analyze data"},
            },
        }

        with patch("builtins.open", mock_open(read_data=str(template_content))):
            with patch("yaml.safe_load", return_value=template_content):
                template = loader.load_strategy_template("test_strategy")
                assert template is not None

    def test_get_prompt_for_focus(self, loader):
        """Test getting prompt for specific focus."""
        # Set up mock templates
        loader.strategy_templates["test_strategy"] = {
            "base": {"system": "Base system"},
            "focus_specific": {
                "formulas": {"task": "Formula task"},
                "data": {"task": "Data task"},
            },
        }

        prompt = loader.get_prompt_for_focus("test_strategy", "formulas")
        assert "Base system" in str(prompt)
        assert "Formula task" in str(prompt)

    def test_merge_templates(self, loader):
        """Test merging base and focus templates."""
        base = {"system": "Base system", "constraints": "Base constraints"}
        focus = {"task": "Focus task", "constraints": "Focus constraints"}

        merged = loader._merge_templates(base, focus)
        assert merged["system"] == "Base system"
        assert merged["task"] == "Focus task"
        assert merged["constraints"] == "Focus constraints"  # Focus overrides base


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_template_manager(self):
        """Test getting singleton template manager."""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        assert manager1 is manager2  # Should be same instance

    def test_render_prompt(self):
        """Test render_prompt convenience function."""
        with patch("spreadsheet_analyzer.notebook_llm.templates.get_template_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.render_template.return_value = "Rendered prompt"
            mock_get.return_value = mock_manager

            result = render_prompt("test_template", var1="value1")
            assert result == "Rendered prompt"
            mock_manager.render_template.assert_called_once_with("test_template", var1="value1")

    def test_render_prompt_with_strategy(self):
        """Test render_prompt with strategy context."""
        context = {
            "strategy": "graph_based",
            "focus": "formulas",
            "data": {"cells": []},
        }

        with patch("spreadsheet_analyzer.notebook_llm.templates.get_template_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.get_template_for_strategy.return_value = {"system": "Strategy template"}
            mock_manager.render_template.return_value = "Strategy rendered"
            mock_get.return_value = mock_manager

            result = render_prompt("strategy_template", **context)
            assert result == "Strategy rendered"
