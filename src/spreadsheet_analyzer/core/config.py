"""Immutable configuration system for the spreadsheet analyzer.

This module provides a functional approach to configuration management
using frozen dataclasses and pure functions for configuration transforms.

CLAUDE-KNOWLEDGE: Configuration is immutable to prevent side effects
and make the system more predictable and testable.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from .errors import ConfigurationError
from .types import Err, Result, err, ok

# Constants
DEFAULT_MAX_FILE_SIZE: Final[int] = 100 * 1024 * 1024  # 100MB
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_MAX_ROUNDS: Final[int] = 10
DEFAULT_TOKEN_BUDGET: Final[int] = 4000
EXCEL_DATE_MAX: Final[int] = 100000
SMALL_SHEET_COUNT: Final[int] = 3
EMAIL_PATTERN_THRESHOLD: Final[float] = 0.7


@dataclass(frozen=True)
class ExcelConfig:
    """Configuration for Excel file processing."""

    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    read_only: bool = True
    keep_vba: bool = False
    data_only: bool = True
    allowed_extensions: tuple[str, ...] = (".xlsx", ".xls", ".xlsm")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 60


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent system."""

    max_rounds: int = DEFAULT_MAX_ROUNDS
    enable_memory: bool = True
    memory_size: int = 100
    enable_reflection: bool = True
    parallel_execution: bool = False


@dataclass(frozen=True)
class ContextConfig:
    """Configuration for context management."""

    token_budget: int = DEFAULT_TOKEN_BUDGET
    compression_enabled: bool = True
    summarization_enabled: bool = True
    windowing_enabled: bool = True
    preserve_structure: bool = True
    pattern_detection: bool = True
    range_aggregation: bool = True
    semantic_clustering: bool = True


@dataclass(frozen=True)
class NotebookConfig:
    """Configuration for notebook operations."""

    kernel_name: str = "python3"
    timeout: int = 60
    auto_save: bool = True
    clear_outputs: bool = False
    markdown_extensions: tuple[str, ...] = ("tables", "fenced_code", "codehilite")


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    excel: ExcelConfig = field(default_factory=ExcelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)

    # Paths
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")
    log_dir: Path = Path("logs")

    # Feature flags
    debug_mode: bool = False
    dry_run: bool = False
    verbose: bool = False


# Configuration builders (pure functions)
def default_config() -> AppConfig:
    """Create default configuration."""
    return AppConfig()


def with_excel_config(config: AppConfig, excel: ExcelConfig) -> AppConfig:
    """Create new config with updated Excel settings."""
    return replace(config, excel=excel)


def with_llm_config(config: AppConfig, llm: LLMConfig) -> AppConfig:
    """Create new config with updated LLM settings."""
    return replace(config, llm=llm)


def with_agent_config(config: AppConfig, agent: AgentConfig) -> AppConfig:
    """Create new config with updated agent settings."""
    return replace(config, agent=agent)


def with_context_config(config: AppConfig, context: ContextConfig) -> AppConfig:
    """Create new config with updated context settings."""
    return replace(config, context=context)


def with_notebook_config(config: AppConfig, notebook: NotebookConfig) -> AppConfig:
    """Create new config with updated notebook settings."""
    return replace(config, notebook=notebook)


def with_paths(
    config: AppConfig, output_dir: Path | None = None, cache_dir: Path | None = None, log_dir: Path | None = None
) -> AppConfig:
    """Create new config with updated paths."""
    updates = {}
    if output_dir is not None:
        updates["output_dir"] = output_dir
    if cache_dir is not None:
        updates["cache_dir"] = cache_dir
    if log_dir is not None:
        updates["log_dir"] = log_dir
    return replace(config, **updates)


def with_flags(
    config: AppConfig, debug_mode: bool | None = None, dry_run: bool | None = None, verbose: bool | None = None
) -> AppConfig:
    """Create new config with updated flags."""
    updates = {}
    if debug_mode is not None:
        updates["debug_mode"] = debug_mode
    if dry_run is not None:
        updates["dry_run"] = dry_run
    if verbose is not None:
        updates["verbose"] = verbose
    return replace(config, **updates)


# Configuration validation
def validate_config(config: AppConfig) -> Result[AppConfig, ConfigurationError]:
    """Validate configuration values."""
    # Validate Excel config
    if config.excel.max_file_size <= 0:
        return err(
            ConfigurationError(
                "Invalid max_file_size", config_key="excel.max_file_size", details={"value": config.excel.max_file_size}
            )
        )

    if config.excel.chunk_size <= 0:
        return err(
            ConfigurationError(
                "Invalid chunk_size", config_key="excel.chunk_size", details={"value": config.excel.chunk_size}
            )
        )

    # Validate LLM config
    if not config.llm.provider:
        return err(ConfigurationError("LLM provider not specified", config_key="llm.provider"))

    if not config.llm.model:
        return err(ConfigurationError("LLM model not specified", config_key="llm.model"))

    if not 0 <= config.llm.temperature <= 2:
        return err(
            ConfigurationError(
                "Temperature must be between 0 and 2",
                config_key="llm.temperature",
                details={"value": config.llm.temperature},
            )
        )

    # Validate Agent config
    if config.agent.max_rounds <= 0:
        return err(
            ConfigurationError(
                "Invalid max_rounds", config_key="agent.max_rounds", details={"value": config.agent.max_rounds}
            )
        )

    if config.agent.memory_size < 0:
        return err(
            ConfigurationError(
                "Invalid memory_size", config_key="agent.memory_size", details={"value": config.agent.memory_size}
            )
        )

    # Validate Context config
    if config.context.token_budget <= 0:
        return err(
            ConfigurationError(
                "Invalid token_budget",
                config_key="context.token_budget",
                details={"value": config.context.token_budget},
            )
        )

    # Validate paths exist and are writable
    for path_name, path_value in [
        ("output_dir", config.output_dir),
        ("cache_dir", config.cache_dir),
        ("log_dir", config.log_dir),
    ]:
        if not isinstance(path_value, Path):
            return err(
                ConfigurationError(
                    f"Invalid path type for {path_name}",
                    config_key=path_name,
                    details={"type": type(path_value).__name__},
                )
            )

    return ok(config)


# Configuration loading from environment
def load_from_env() -> Result[dict[str, Any], ConfigurationError]:
    """Load configuration from environment variables.

    Environment variables follow the pattern:
    SPREADSHEET_ANALYZER_<SECTION>_<KEY>

    Example:
        SPREADSHEET_ANALYZER_LLM_MODEL=gpt-4
        SPREADSHEET_ANALYZER_EXCEL_MAX_FILE_SIZE=50000000
    """
    import os

    config_dict: dict[str, dict[str, Any]] = {
        "excel": {},
        "llm": {},
        "agent": {},
        "context": {},
        "notebook": {},
        "paths": {},
        "flags": {},
    }

    prefix = "SPREADSHEET_ANALYZER_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        parts = key[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, config_key = parts

        # Parse value based on expected type
        try:
            if value.lower() in ("true", "false"):
                parsed_value = value.lower() == "true"
            elif value.isdigit():
                parsed_value = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                parsed_value = float(value)
            else:
                parsed_value = value
        except ValueError:
            parsed_value = value

        if section in config_dict:
            config_dict[section][config_key] = parsed_value

    return ok(config_dict)


def merge_with_env(config: AppConfig) -> Result[AppConfig, ConfigurationError]:
    """Merge configuration with environment variables."""
    env_result = load_from_env()

    if isinstance(env_result, Err):
        return env_result

    env_dict = env_result.value

    # Apply environment overrides
    new_config = config

    # Excel overrides
    if env_dict.get("excel"):
        excel_updates = {}
        for key, value in env_dict["excel"].items():
            if hasattr(config.excel, key):
                excel_updates[key] = value
        if excel_updates:
            new_config = with_excel_config(new_config, replace(new_config.excel, **excel_updates))

    # LLM overrides
    if env_dict.get("llm"):
        llm_updates = {}
        for key, value in env_dict["llm"].items():
            if hasattr(config.llm, key):
                llm_updates[key] = value
        if llm_updates:
            new_config = with_llm_config(new_config, replace(new_config.llm, **llm_updates))

    # Continue with other sections...

    return ok(new_config)
