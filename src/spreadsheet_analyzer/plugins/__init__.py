"""
Plugin system for domain-specific notebook functionality.

This module provides the plugin architecture for extending the core notebook
system with domain-specific features:
- Task-based cell generation
- Domain-specific quality inspection
- Custom analysis workflows
- Extensible plugin registry

The plugin system follows a simple protocol-based approach for maximum flexibility.
"""

from .base import PluginRegistry, QualityInspector, Task

__all__ = ["PluginRegistry", "QualityInspector", "Task"]
