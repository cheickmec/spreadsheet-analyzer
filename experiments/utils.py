"""
Shared utilities for experiments directory.

This module provides common functionality for all experiments, particularly
logging setup with consistent file naming and comprehensive tracing.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """
    Comprehensive logging setup for experiments with consistent file naming.

    Creates multiple loggers for different purposes:
    - main: General experiment logging
    - llm_trace: Detailed LLM interaction logs
    - error: Error-specific logging with full traces

    File naming pattern: {module_name}_{hash[:8]}_{timestamp}_{type}.{ext}
    """

    def __init__(
        self,
        module_path: str,
        model_name: str | None = None,
        excel_path: str | Path | None = None,
        sheet_index: int | None = None,
    ):
        """
        Initialize experiment logger with hierarchical output support.

        Args:
            module_path: __file__ of the calling module
            model_name: Model name (will be updated to full version later)
            excel_path: Path to Excel file being analyzed
            sheet_index: Sheet index being analyzed
        """
        self.module_path = Path(module_path)
        self.module_name = self.module_path.stem
        self.module_hash = self._calculate_module_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store run configuration
        self.model_requested = model_name
        self.model_actual = None  # Will be set when we get first response
        self.excel_path = Path(excel_path) if excel_path else None
        self.sheet_index = sheet_index

        # Create hierarchical output directory
        self.outputs_dir = self._create_hierarchical_output_dir()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all loggers
        self.main = self._setup_main_logger()
        self.llm_trace = self._setup_llm_logger()
        self.error = self._setup_error_logger()

        # Metrics storage
        self.metrics: dict[str, Any] = {}

        # Log initialization
        self.main.info(f"🔬 EXPERIMENT INITIALIZED: {self.module_name}")
        self.main.info(f"📁 Module: {self.module_path}")
        self.main.info(f"🔑 Hash: {self.module_hash}")
        self.main.info(f"⏰ Timestamp: {self.timestamp}")
        self.main.info(f"📂 Outputs: {self.outputs_dir}")
        self.main.info("=" * 80)

    def _create_hierarchical_output_dir(self) -> Path:
        """Create hierarchical output directory structure."""
        base_dir = self.module_path.parent / "outputs" / self.module_name

        # Use actual model name if available, otherwise requested, otherwise 'unknown_model'
        model_dir = self.model_actual or self.model_requested or "unknown_model"

        # Sanitize Excel filename and create input identifier
        if self.excel_path:
            excel_name = self.excel_path.stem
            # Sanitize: remove spaces, special chars, lowercase
            excel_sanitized = excel_name.lower().replace(" ", "-").replace(".", "")
            # Remove any other special characters
            excel_sanitized = "".join(c if c.isalnum() or c == "-" else "" for c in excel_sanitized)
            input_dir = f"{excel_sanitized}_sh{self.sheet_index if self.sheet_index is not None else 0}"
        else:
            input_dir = "unknown_input"

        # Create full path: outputs/module/model/input/timestamp_hash/
        run_dir = base_dir / model_dir / input_dir / f"{self.timestamp}_{self.module_hash}"
        return run_dir

    def update_model_actual(self, model_actual: str):
        """Update the actual model name from API response."""
        if model_actual and model_actual != self.model_requested:
            self.model_actual = model_actual
            # Update directory to use actual model name
            old_dir = self.outputs_dir
            new_dir = self._create_hierarchical_output_dir()
            if new_dir != old_dir:
                # Move existing files to new directory
                new_dir.mkdir(parents=True, exist_ok=True)
                if old_dir.exists():
                    for file in old_dir.glob("*"):
                        file.rename(new_dir / file.name)
                    # Remove old directory if empty
                    try:
                        old_dir.rmdir()
                        # Try to remove parent directories if empty
                        parent = old_dir.parent
                        while parent != self.module_path.parent / "outputs":
                            if parent.exists() and not any(parent.iterdir()):
                                parent.rmdir()
                                parent = parent.parent
                            else:
                                break
                    except:
                        pass
                self.outputs_dir = new_dir

    def _calculate_module_hash(self) -> str:
        """Calculate SHA-256 hash of the calling module."""
        try:
            content = self.module_path.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        except Exception:
            # Fallback to a timestamp-based hash if file can't be read
            fallback = f"fallback_{datetime.now().isoformat()}"
            return hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:8]

    def _get_filename(self, log_type: str, extension: str = "log") -> Path:
        """Generate consistent filename for outputs."""
        # In hierarchical structure, just use type.ext since directory contains all context
        filename = f"{log_type}.{extension}"
        return self.outputs_dir / filename

    def _setup_main_logger(self) -> logging.Logger:
        """Setup main experiment logger with comprehensive formatting."""
        logger = logging.getLogger(f"experiment.{self.module_name}.main")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # Clear any existing handlers

        # File handler
        file_handler = logging.FileHandler(self._get_filename("main"), encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(funcName)-20s:%(lineno)-4d | %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler with simpler format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _setup_llm_logger(self) -> logging.Logger:
        """Setup specialized logger for LLM interactions."""
        logger = logging.getLogger(f"experiment.{self.module_name}.llm")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        file_handler = logging.FileHandler(self._get_filename("llm_trace"), encoding="utf-8")
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def _setup_error_logger(self) -> logging.Logger:
        """Setup specialized logger for errors with full traces."""
        logger = logging.getLogger(f"experiment.{self.module_name}.error")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        file_handler = logging.FileHandler(self._get_filename("errors"), encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(funcName)-20s:%(lineno)-4d | %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def log_llm_interaction(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: dict[str, int],
        request_id: str | None = None,
        actual_model: str | None = None,
        **kwargs,
    ):
        """
        Log detailed LLM interaction.

        Args:
            model: Model name requested (e.g., "gpt-4o", "claude-3")
            prompt: Full prompt sent to model
            response: Full response from model
            tokens: Token usage dict with keys: input, output, total
            request_id: Optional request identifier
            actual_model: Actual model that responded (e.g., "gpt-4o-2024-08-06")
            **kwargs: Additional metadata to log
        """
        # Update actual model if provided and different
        if actual_model and actual_model != self.model_requested:
            self.update_model_actual(actual_model)

        timestamp = datetime.now().isoformat()
        separator = "=" * 100

        log_entry = f"""
{separator}
TIMESTAMP: {timestamp}
REQUEST_ID: {request_id or "N/A"}
MODEL_REQUESTED: {model}
MODEL_ACTUAL: {actual_model or model}
TOKENS: Input={tokens.get("input", 0)} | Output={tokens.get("output", 0)} | Total={tokens.get("total", 0)}
COST_EST: ${self._estimate_cost(actual_model or model, tokens)}

ADDITIONAL_METADATA:
{json.dumps(kwargs, indent=2) if kwargs else "None"}

PROMPT:
{"-" * 50}
{prompt}
{"-" * 50}

RESPONSE:
{"-" * 50}
{response}
{"-" * 50}
{separator}
"""

        self.llm_trace.info(log_entry)

        # Also log summary to main logger
        display_model = actual_model or model
        self.main.info(
            f"🤖 LLM Call: {display_model} | Tokens: {tokens.get('total', 0)} | Request: {request_id or 'N/A'}"
        )

    def log_metrics(self, metrics: dict[str, Any]):
        """
        Log and store experiment metrics.

        Args:
            metrics: Dictionary of metric name -> value pairs
        """
        self.metrics.update(metrics)

        # Log to main logger
        self.main.info("📊 METRICS UPDATE:")
        for key, value in metrics.items():
            self.main.info(f"   {key}: {value}")

        # Save to JSON file
        self._save_metrics()

    def _save_metrics(self):
        """Save current metrics to JSON file."""
        metrics_file = self._get_filename("metrics", "json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "module": self.module_name,
                    "hash": self.module_hash,
                    "timestamp": self.timestamp,
                    "metrics": self.metrics,
                },
                f,
                indent=2,
                default=str,
            )

    def save_results(self, results: dict[str, Any]):
        """
        Save experiment results to JSON file.

        Args:
            results: Dictionary of results to save
        """
        results_file = self._get_filename("results", "json")

        # Enhanced metadata for hierarchical structure
        full_results = {
            "module": self.module_name,
            "hash": self.module_hash,
            "timestamp": self.timestamp,
            "model_requested": self.model_requested,
            "model_actual": self.model_actual,
            "excel_file": str(self.excel_path) if self.excel_path else None,
            "sheet_index": self.sheet_index,
            "results": results,
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, default=str)

        self.main.info(f"💾 Results saved to: {results_file}")

        # Also save run configuration
        self.save_run_config()

    def save_run_config(self):
        """Save run configuration to a separate file for easy reference."""
        config_file = self._get_filename("run_config", "json")

        config = {
            "module": self.module_name,
            "module_hash": self.module_hash,
            "timestamp": self.timestamp,
            "model_requested": self.model_requested,
            "model_actual": self.model_actual,
            "excel_file": str(self.excel_path) if self.excel_path else None,
            "sheet_index": self.sheet_index,
            "output_directory": str(self.outputs_dir),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    def _estimate_cost(self, model: str, tokens: dict[str, int]) -> float:
        """
        Rough cost estimation for different models.

        Args:
            model: Model name (can be full version like gpt-4o-2024-08-06)
            tokens: Token usage dictionary

        Returns:
            Estimated cost in USD
        """
        # Rough pricing per 1K tokens (as of 2024/2025)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
        }

        # Handle full model names by extracting base name
        base_model = model
        for key in pricing:
            if model.startswith(key):
                base_model = key
                break

        # Default to GPT-4 pricing if model not found
        rates = pricing.get(base_model, pricing["gpt-4"])

        input_cost = (tokens.get("input", 0) / 1000) * rates["input"]
        output_cost = (tokens.get("output", 0) / 1000) * rates["output"]

        return round(input_cost + output_cost, 6)

    def finalize(self):
        """Call this at the end of experiment to log summary."""
        self.main.info("=" * 80)
        self.main.info("🏁 EXPERIMENT COMPLETED")
        self.main.info(f"📊 Final metrics count: {len(self.metrics)}")
        if self.metrics:
            self.main.info("📈 Final metrics summary:")
            for key, value in self.metrics.items():
                self.main.info(f"   {key}: {value}")

        # Save final metrics
        self._save_metrics()

        self.main.info(f"📂 All outputs saved to: {self.outputs_dir}")
        self.main.info("=" * 80)


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO):
    """
    Log a decorative section header.

    Args:
        logger: Logger instance to use
        title: Section title
        level: Logging level to use
    """
    decorator = "🔍" * 20
    logger.log(level, f"\n{decorator}")
    logger.log(level, f"🔍{' ' * 8}{title.upper()}{' ' * 8}🔍")
    logger.log(level, f"{decorator}\n")


def log_data_info(logger: logging.Logger, data: Any, name: str = "data"):
    """
    Log comprehensive information about a data structure.

    Args:
        logger: Logger instance to use
        data: Data structure to analyze
        name: Name to use in logs
    """
    logger.debug(f"📊 {name.upper()} ANALYSIS:")
    logger.debug(f"   Type: {type(data).__name__}")

    if hasattr(data, "shape"):
        logger.debug(f"   Shape: {data.shape}")
    elif hasattr(data, "__len__"):
        logger.debug(f"   Length: {len(data)}")

    if hasattr(data, "dtypes"):
        logger.debug(f"   Data types: {dict(data.dtypes)}")
    elif hasattr(data, "columns"):
        logger.debug(f"   Columns: {list(data.columns)}")

    # Sample data if possible
    if hasattr(data, "head"):
        logger.debug(f"   First 3 rows:\n{data.head(3)}")
    elif hasattr(data, "__getitem__") and len(data) > 0:
        sample_size = min(3, len(data))
        logger.debug(f"   First {sample_size} items: {data[:sample_size]}")
