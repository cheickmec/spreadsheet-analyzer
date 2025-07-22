"""Template management system for LLM prompts using Jinja2."""

import logging
from pathlib import Path
from typing import Any

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateNotFound,
    select_autoescape,
)

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages Jinja2 templates for LLM prompt generation.

    This class provides a centralized system for loading, caching, and
    rendering prompt templates. It supports template inheritance, macros,
    and custom filters for flexible prompt engineering.
    """

    def __init__(
        self,
        template_dirs: list[Path] | None = None,
        cache_size: int = 50,
        auto_reload: bool = True,
    ):
        """Initialize template manager.

        Args:
            template_dirs: List of directories to search for templates.
                         Defaults to project templates directory.
            cache_size: Maximum number of compiled templates to cache.
            auto_reload: Whether to automatically reload changed templates.
        """
        if template_dirs is None:
            # Default to project templates directory
            project_root = Path(__file__).parent.parent.parent.parent
            template_dirs = [project_root / "templates"]

        # Ensure all paths are Path objects and exist
        self.template_dirs = []
        for path in template_dirs:
            path = Path(path)
            if path.exists():
                self.template_dirs.append(path)
            else:
                logger.warning("Template directory not found: %s", path)

        if not self.template_dirs:
            raise ValueError("No valid template directories found")

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dirs),
            autoescape=select_autoescape(["html", "xml"]),
            cache_size=cache_size,
            auto_reload=auto_reload,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self._register_filters()

        # Add custom globals
        self._register_globals()

        logger.info("Template manager initialized with %d directories", len(self.template_dirs))

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        # Add utility filters
        self.env.filters["truncate_middle"] = self._truncate_middle
        self.env.filters["format_cell_ref"] = self._format_cell_ref
        self.env.filters["format_bytes"] = self._format_bytes
        self.env.filters["highlight_formulas"] = self._highlight_formulas

    def _register_globals(self) -> None:
        """Register global functions/variables available in all templates."""
        self.env.globals["max_tokens"] = 4096  # Default max tokens
        self.env.globals["now"] = self._get_timestamp

    def get_template(self, template_name: str) -> Template:
        """Load and return a template by name.

        Args:
            template_name: Name of template file (e.g., 'strategies/hierarchical/exploration.jinja2')

        Returns:
            Compiled Jinja2 template

        Raises:
            TemplateNotFound: If template doesn't exist
        """
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            logger.error("Template not found: %s", template_name)
            logger.debug("Search paths: %s", self.template_dirs)
            raise

    def render(self, template_name: str, context: dict[str, Any], **kwargs: Any) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of template file
            context: Dictionary of variables to pass to template
            **kwargs: Additional variables to merge into context

        Returns:
            Rendered template string
        """
        template = self.get_template(template_name)

        # Merge kwargs into context
        full_context = {**context, **kwargs}

        try:
            return template.render(full_context)
        except Exception as e:
            logger.error("Error rendering template %s: %s", template_name, str(e))
            raise

    def render_string(self, template_string: str, context: dict[str, Any], **kwargs: Any) -> str:
        """Render a template from a string.

        Args:
            template_string: Template content as string
            context: Dictionary of variables
            **kwargs: Additional variables

        Returns:
            Rendered template string
        """
        template = self.env.from_string(template_string)
        full_context = {**context, **kwargs}
        return template.render(full_context)

    def list_templates(self, prefix: str = "") -> list[str]:
        """List all available templates.

        Args:
            prefix: Optional prefix to filter templates

        Returns:
            List of template names
        """
        templates = []
        for template_dir in self.template_dirs:
            for path in Path(template_dir).rglob("*.jinja2"):
                rel_path = path.relative_to(template_dir)
                template_name = str(rel_path)
                if not prefix or template_name.startswith(prefix):
                    templates.append(template_name)

        return sorted(set(templates))

    def add_template_directory(self, directory: Path) -> None:
        """Add a new template directory to the search path.

        Args:
            directory: Path to template directory
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if directory not in self.template_dirs:
            self.template_dirs.append(directory)
            # Recreate loader with new directories
            self.env.loader = FileSystemLoader(self.template_dirs)
            logger.info("Added template directory: %s", directory)

    # Custom filter implementations
    @staticmethod
    def _truncate_middle(text: str, length: int = 80) -> str:
        """Truncate text in the middle, preserving start and end."""
        if len(text) <= length:
            return text

        half = (length - 3) // 2
        return f"{text[:half]}...{text[-half:]}"

    @staticmethod
    def _format_cell_ref(cell: str, sheet: str) -> str:
        """Format a cell reference with sheet name."""
        if "!" in cell:  # Already includes sheet
            return cell
        return f"{sheet}!{cell}" if sheet else cell

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} PB"

    @staticmethod
    def _highlight_formulas(text: str) -> str:
        """Add markers around Excel formulas for visibility."""
        import re

        # Simple pattern for Excel formulas
        formula_pattern = r"=[A-Z]+\([^)]*\)"
        return re.sub(formula_pattern, r"**\g<0>**", text)

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp for templates."""
        from datetime import datetime

        return datetime.now().isoformat()


class StrategyTemplateLoader:
    """Specialized loader for strategy templates."""

    def __init__(self, template_manager: TemplateManager):
        """Initialize with a template manager instance."""
        self.template_manager = template_manager
        self._strategy_cache: dict[str, Template] = {}

    def load_strategy_template(self, strategy_name: str, template_type: str = "main") -> Template:
        """Load a template for a specific strategy.

        Args:
            strategy_name: Name of the strategy (e.g., 'hierarchical')
            template_type: Type of template (e.g., 'exploration', 'refinement')

        Returns:
            Compiled template
        """
        cache_key = f"{strategy_name}:{template_type}"

        if cache_key not in self._strategy_cache:
            # Construct template path
            template_path = f"strategies/{strategy_name}/{template_type}.jinja2"

            try:
                template = self.template_manager.get_template(template_path)
                self._strategy_cache[cache_key] = template
            except TemplateNotFound:
                # Try fallback to default template
                logger.warning("Strategy template not found: %s, using default", template_path)
                template = self.template_manager.get_template("strategies/default/analysis.jinja2")
                self._strategy_cache[cache_key] = template

        return self._strategy_cache[cache_key]

    def render_strategy_prompt(self, strategy_name: str, template_type: str, context: dict[str, Any]) -> str:
        """Render a strategy-specific prompt.

        Args:
            strategy_name: Name of the strategy
            template_type: Type of template
            context: Variables for rendering

        Returns:
            Rendered prompt string
        """
        template = self.load_strategy_template(strategy_name, template_type)

        # Add strategy metadata to context
        context["strategy_name"] = strategy_name
        context["template_type"] = template_type

        return template.render(context)


# Singleton instance for easy access
_default_manager: TemplateManager | None = None


def get_template_manager() -> TemplateManager:
    """Get the default template manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TemplateManager()
    return _default_manager


def render_prompt(template_name: str, context: dict[str, Any], **kwargs: Any) -> str:
    """Convenience function to render a template.

    Args:
        template_name: Name of template file
        context: Template variables
        **kwargs: Additional variables

    Returns:
        Rendered template string
    """
    manager = get_template_manager()
    return manager.render(template_name, context, **kwargs)
