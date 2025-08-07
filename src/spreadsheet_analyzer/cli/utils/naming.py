"""Pure functions for generating structured file names.

This module extracts the file naming logic from StructuredFileNameGenerator
into pure functions that operate on immutable data.

CLAUDE-KNOWLEDGE: File naming follows a consistent pattern to make files
easily identifiable and sortable by their parameters.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FileNameConfig:
    """Immutable configuration for file naming."""

    excel_file: Path
    model: str
    sheet_index: int
    sheet_name: str | None = None
    max_rounds: int = 5
    session_id: str | None = None
    timestamp: datetime | None = None


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file names.

    Preserves the full model identifier while making it filesystem-safe.

    Args:
        model: Raw model name (e.g., "claude-3-5-sonnet-20241022")

    Returns:
        Sanitized model name (e.g., "claude_3_5_sonnet_20241022")

    Examples:
        >>> sanitize_model_name("claude-3-5-sonnet-20241022")
        'claude_3_5_sonnet_20241022'
        >>> sanitize_model_name("claude-sonnet-4-20250514")
        'claude_sonnet_4_20250514'
        >>> sanitize_model_name("gpt-4-turbo")
        'gpt_4_turbo'
        >>> sanitize_model_name("ollama/mistral:latest")
        'ollama_mistral_latest'
    """
    # Replace filesystem-unsafe characters with underscores
    model_clean = model.replace("-", "_").replace(".", "_").replace("/", "_").replace(":", "_")

    # Remove multiple consecutive underscores and trim
    model_clean = re.sub(r"_+", "_", model_clean).strip("_")

    # Return the full sanitized model name to preserve version information
    return model_clean.lower()


def sanitize_sheet_name(sheet_name: str) -> str:
    """Sanitize sheet name for use in file names.

    Removes or replaces characters that are problematic in file names.

    Args:
        sheet_name: Raw sheet name from Excel

    Returns:
        Sanitized sheet name safe for file systems

    Examples:
        >>> sanitize_sheet_name("Revenue & Costs")
        'Revenue_Costs'
        >>> sanitize_sheet_name("Q1/2024 - Financial Summary")
        'Q1_2024_Financial_Summary'
    """
    # Replace spaces, special characters with underscores
    sanitized = re.sub(r"[^\w\-_.]", "_", sheet_name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Limit length to prevent filesystem issues
    return sanitized[:50] if len(sanitized) > 50 else sanitized


def generate_notebook_name(config: FileNameConfig, include_timestamp: bool = False) -> str:
    """Generate a structured notebook file name.

    Creates a consistent naming pattern for notebooks that includes all
    relevant analysis parameters.

    Args:
        config: File naming configuration
        include_timestamp: Whether to include timestamp in name

    Returns:
        Structured notebook filename ending in .ipynb

    Examples:
        >>> config = FileNameConfig(
        ...     excel_file=Path("sales_data.xlsx"),
        ...     model="gpt-4",
        ...     sheet_index=0,
        ...     sheet_name="Revenue"
        ... )
        >>> generate_notebook_name(config)
        'sales_data_sheet0_Revenue_gpt4_r5.ipynb'
    """
    # Build the base name with all parameters
    parts = [
        config.excel_file.stem,  # Excel file name without extension
        f"sheet{config.sheet_index}",  # Sheet index
    ]

    # Add sheet name if available
    if config.sheet_name:
        sanitized_sheet = sanitize_sheet_name(config.sheet_name)
        if sanitized_sheet:  # Only add if not empty after sanitization
            parts.append(sanitized_sheet)

    # Add model name
    parts.append(sanitize_model_name(config.model))

    # Add max rounds
    parts.append(f"r{config.max_rounds}")

    # Add timestamp if requested
    if include_timestamp:
        timestamp = config.timestamp or datetime.now()
        parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))

    # Join parts and add extension
    return f"{'_'.join(parts)}.ipynb"


def generate_log_name(config: FileNameConfig, include_timestamp: bool = True) -> str:
    """Generate a structured log file name.

    Creates a consistent naming pattern for log files.

    Args:
        config: File naming configuration
        include_timestamp: Whether to include timestamp (default True)

    Returns:
        Structured log filename ending in .log
    """
    # Build the base name with all parameters
    parts = [
        config.excel_file.stem,
        f"sheet{config.sheet_index}",
    ]

    # Add sheet name if available
    if config.sheet_name:
        sanitized_sheet = sanitize_sheet_name(config.sheet_name)
        if sanitized_sheet:
            parts.append(sanitized_sheet)

    # Add model name
    parts.append(sanitize_model_name(config.model))

    # Add max rounds
    parts.append(f"r{config.max_rounds}")

    # Add log suffix
    parts.append("analysis")

    # Add timestamp if requested (usually true for logs)
    if include_timestamp:
        timestamp = config.timestamp or datetime.now()
        parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))

    # Join parts and add extension
    return f"{'_'.join(parts)}.log"


def generate_session_id(config: FileNameConfig) -> str:
    """Generate a structured session ID with timestamp.

    Creates a unique session identifier for tracking analysis runs.

    Args:
        config: File naming configuration

    Returns:
        Structured session ID string
    """
    if config.session_id:
        return config.session_id

    parts = [
        config.excel_file.stem,
        f"sheet{config.sheet_index}",
    ]

    # Add sheet name if available
    if config.sheet_name:
        sanitized_sheet = sanitize_sheet_name(config.sheet_name)
        if sanitized_sheet:
            parts.append(sanitized_sheet)

    # Add model name
    parts.append(sanitize_model_name(config.model))

    # Add max rounds
    parts.append(f"r{config.max_rounds}")

    # Add session suffix
    parts.append("analysis_session")

    # Add timestamp
    timestamp = config.timestamp or datetime.now()
    parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))

    return "_".join(parts)


def get_cost_tracking_path(config: FileNameConfig, output_dir: Path | None = None) -> Path:
    """Get the path for the cost tracking file.

    Determines where to save cost tracking data with date-based organization.

    Args:
        config: File naming configuration
        output_dir: Optional output directory override

    Returns:
        Full path for cost tracking JSON file
    """
    parts = [
        config.excel_file.stem,
        f"sheet{config.sheet_index}",
    ]

    if config.sheet_name:
        sanitized_sheet = sanitize_sheet_name(config.sheet_name)
        if sanitized_sheet:
            parts.append(sanitized_sheet)

    parts.append(sanitize_model_name(config.model))
    parts.append(f"r{config.max_rounds}")
    parts.append("cost_tracking")

    timestamp = config.timestamp or datetime.now()
    parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))

    cost_name = f"{'_'.join(parts)}.json"

    if output_dir:
        # Use provided output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / cost_name
    else:
        # Use logs directory with date organization
        date_str = timestamp.strftime("%Y%m%d")
        logs_dir = Path("logs") / date_str
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / cost_name
