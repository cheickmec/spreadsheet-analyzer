"""Strategy registry for dynamic discovery and loading of analysis strategies.

This module implements a plugin-based system for registering and discovering
analysis strategies using Python entry points.
"""

import logging
from typing import Any

try:
    from importlib.metadata import entry_points
except ImportError:
    # Fallback for Python < 3.8
    from pkg_resources import iter_entry_points as _iter_entry_points

    def entry_points(group: str) -> list[Any]:  # type: ignore[misc]
        """Compatibility wrapper for older Python versions."""
        return list(_iter_entry_points(group))


from .base import AnalysisStrategy, BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Dynamic strategy discovery and loading system.

    This registry automatically discovers strategies registered via Python
    entry points and provides a unified interface for accessing them.

    Entry points should be registered in the package's setup.py/pyproject.toml:

    ```python
    # In setup.py
    entry_points={
        "spreadsheet_analyzer.strategies": [
            "hierarchical = mypackage.strategies:HierarchicalStrategy",
            "graph_based = mypackage.strategies:GraphBasedStrategy",
        ]
    }
    ```

    Or in pyproject.toml:

    ```toml
    [project.entry-points."spreadsheet_analyzer.strategies"]
    hierarchical = "mypackage.strategies:HierarchicalStrategy"
    graph_based = "mypackage.strategies:GraphBasedStrategy"
    ```
    """

    # Default entry point group for strategies
    ENTRY_POINT_GROUP = "spreadsheet_analyzer.strategies"

    def __init__(self, *, entry_point_group: str | None = None):
        """Initialize the strategy registry.

        Args:
            entry_point_group: Override the default entry point group name
        """
        self._strategies: dict[str, type[AnalysisStrategy]] = {}
        self._instances: dict[str, AnalysisStrategy] = {}
        self.entry_point_group = entry_point_group or self.ENTRY_POINT_GROUP
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Discover and load all registered strategies from entry points."""
        logger.info(f"Loading strategies from entry point group: {self.entry_point_group}")

        # Load from entry points
        try:
            eps = entry_points(group=self.entry_point_group)

            for ep in eps:
                try:
                    # Load the strategy class
                    strategy_class = ep.load()

                    # Validate it's a proper strategy
                    if not self._is_valid_strategy(strategy_class):
                        logger.warning(f"Entry point '{ep.name}' does not provide a valid strategy class")
                        continue

                    # Register the strategy
                    self._strategies[ep.name] = strategy_class
                    logger.info(f"Loaded strategy: {ep.name} -> {strategy_class.__name__}")

                except Exception:
                    logger.exception(f"Failed to load strategy '{ep.name}'")

        except Exception:
            logger.exception("Failed to load entry points")

        logger.info(f"Loaded {len(self._strategies)} strategies")

    def _is_valid_strategy(self, strategy_class: type) -> bool:
        """Check if a class is a valid strategy implementation.

        Args:
            strategy_class: The class to validate

        Returns:
            True if the class is a valid strategy
        """
        # Check if it implements the protocol or extends BaseStrategy
        try:
            if issubclass(strategy_class, BaseStrategy):
                return True
        except TypeError:
            # Not a class or invalid for issubclass
            return False

        # Check if it has the required methods (duck typing)
        required_methods = ["prepare_context", "format_prompt", "parse_response"]
        return all(hasattr(strategy_class, method) for method in required_methods)

    def register(self, name: str, strategy_class: type[AnalysisStrategy]) -> None:
        """Manually register a strategy class.

        This method allows runtime registration of strategies without
        using entry points.

        Args:
            name: Name to register the strategy under
            strategy_class: The strategy class to register

        Raises:
            ValueError: If the strategy class is invalid
        """
        if not self._is_valid_strategy(strategy_class):
            raise ValueError(f"Invalid strategy class: {strategy_class}")

        self._strategies[name] = strategy_class
        # Clear any cached instance
        self._instances.pop(name, None)
        logger.info(f"Registered strategy: {name} -> {strategy_class.__name__}")

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry.

        Args:
            name: Name of the strategy to remove
        """
        self._strategies.pop(name, None)
        self._instances.pop(name, None)
        logger.info(f"Unregistered strategy: {name}")

    def get_strategy(self, name: str, config: dict[str, Any] | None = None) -> AnalysisStrategy:
        """Retrieve a strategy instance by name.

        This method returns a singleton instance of the strategy unless
        config is provided, in which case a new instance is created.

        Args:
            name: Name of the strategy to retrieve
            config: Optional configuration for the strategy

        Returns:
            Strategy instance

        Raises:
            ValueError: If the strategy name is not found
        """
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise ValueError(f"Unknown strategy: '{name}'. Available strategies: {available}")

        # If config is provided, always create a new instance
        if config is not None:
            strategy_class = self._strategies[name]
            return self._create_instance(strategy_class, config)

        # Otherwise, return cached singleton instance
        if name not in self._instances:
            strategy_class = self._strategies[name]
            self._instances[name] = self._create_instance(strategy_class, {})

        return self._instances[name]

    def _create_instance(self, strategy_class: type[AnalysisStrategy], config: dict[str, Any]) -> AnalysisStrategy:
        """Create a new instance of a strategy class.

        Args:
            strategy_class: The strategy class to instantiate
            config: Configuration for the strategy

        Returns:
            Strategy instance
        """
        try:
            # Try to instantiate with config
            return strategy_class(config)  # type: ignore[call-arg]
        except TypeError:
            # If that fails, try without config (for strategies that don't take config)
            try:
                instance = strategy_class()
                # If it has a config attribute, set it
                if hasattr(instance, "config"):
                    instance.config = config
            except Exception:
                logger.exception(f"Failed to instantiate strategy {strategy_class.__name__}")
                raise
            else:
                return instance

    def list_strategies(self) -> list[str]:
        """Get a list of all available strategy names.

        Returns:
            List of strategy names
        """
        return sorted(self._strategies.keys())

    def get_strategy_info(self, name: str) -> dict[str, str]:
        """Get information about a specific strategy.

        Args:
            name: Name of the strategy

        Returns:
            Dictionary with strategy information

        Raises:
            ValueError: If the strategy name is not found
        """
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: '{name}'")

        strategy_class = self._strategies[name]
        return {
            "name": name,
            "class": strategy_class.__name__,
            "module": strategy_class.__module__,
            "doc": strategy_class.__doc__ or "No documentation available",
        }

    def reload_plugins(self) -> None:
        """Reload all plugins from entry points.

        This is useful for development when strategies might be added
        or modified.
        """
        logger.info("Reloading strategy plugins...")
        self._strategies.clear()
        self._instances.clear()
        self._load_plugins()


# Global registry instance
_global_registry: StrategyRegistry | None = None


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance.

    Returns:
        The global StrategyRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def register_strategy(name: str, strategy_class: type[AnalysisStrategy]) -> None:
    """Convenience function to register a strategy with the global registry.

    Args:
        name: Name to register the strategy under
        strategy_class: The strategy class to register
    """
    get_registry().register(name, strategy_class)
