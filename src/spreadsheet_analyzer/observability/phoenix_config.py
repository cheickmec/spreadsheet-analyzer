"""
Phoenix observability configuration and initialization.

This module provides centralized configuration for Arize Phoenix,
including tracing setup, instrumentation, and deployment options.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass

import phoenix as px
from opentelemetry.trace import TracerProvider
from phoenix.otel import register
from structlog import get_logger

try:
    from openinference.instrumentation import using_session
except ImportError:
    # Fallback for older versions
    using_session = None

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
except ImportError:
    LangChainInstrumentor = None

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
except ImportError:
    OpenAIInstrumentor = None

try:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
except ImportError:
    AnthropicInstrumentor = None

try:
    from opentelemetry.trace import get_tracer_provider as otel_get_tracer_provider
except ImportError:
    otel_get_tracer_provider = None

try:
    from opentelemetry import trace
except ImportError:
    trace = None

logger = get_logger(__name__)


@dataclass
class PhoenixConfig:
    """Configuration for Phoenix deployment and tracing."""

    # Deployment mode: "local", "cloud", or "docker"
    mode: str = "local"

    # Phoenix API settings (for cloud mode)
    api_key: str | None = None
    collector_endpoint: str | None = None

    # Local settings
    host: str = "localhost"
    port: int = 6006

    # Docker settings
    grpc_port: int = 4317

    # Tracing settings
    project_name: str = "spreadsheet-analyzer"
    enable_auto_instrumentation: bool = True

    @classmethod
    def from_env(cls) -> "PhoenixConfig":
        """Create config from environment variables."""
        return cls(
            mode=os.getenv("PHOENIX_MODE", "local"),
            api_key=os.getenv("PHOENIX_API_KEY"),
            collector_endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
            host=os.getenv("PHOENIX_HOST", "localhost"),
            port=int(os.getenv("PHOENIX_PORT", "6006")),
            grpc_port=int(os.getenv("PHOENIX_GRPC_PORT", "4317")),
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "spreadsheet-analyzer"),
        )


def initialize_phoenix(config: PhoenixConfig | None = None) -> TracerProvider | None:
    """
    Initialize Phoenix with the given configuration.

    Args:
        config: Phoenix configuration. If None, loads from environment.

    Returns:
        TracerProvider if initialization successful, None otherwise.
    """
    if config is None:
        config = PhoenixConfig.from_env()

    try:
        if config.mode == "cloud":
            # Configure for Phoenix cloud
            if not config.api_key:
                logger.warning("Phoenix cloud mode requires PHOENIX_API_KEY")
                return None

            os.environ["PHOENIX_API_KEY"] = config.api_key
            if config.collector_endpoint:
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = config.collector_endpoint
            else:
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

            logger.info("Configured Phoenix for cloud mode", endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"])

        elif config.mode == "docker":
            # Configure for Docker deployment
            endpoint = f"http://{config.host}:{config.grpc_port}"
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
            logger.info("Configured Phoenix for Docker mode", endpoint=endpoint)

        else:  # local mode
            # Launch local Phoenix app
            try:
                session = px.launch_app(
                    host=config.host,
                    port=config.port,
                )
                logger.info("Launched local Phoenix app", url=f"http://{config.host}:{config.port}")
            except Exception as e:
                logger.warning("Could not launch local Phoenix app", error=str(e))
                # Continue anyway - Phoenix might already be running

        # Register Phoenix with OpenTelemetry
        tracer_provider = register(
            project_name=config.project_name,
        )

        logger.info("Phoenix initialized successfully", mode=config.mode, project=config.project_name)

        return tracer_provider

    except Exception as e:
        logger.error("Failed to initialize Phoenix", error=str(e))
        return None


def instrument_langchain(tracer_provider: TracerProvider | None = None) -> bool:
    """
    Instrument LangChain for tracing.

    Args:
        tracer_provider: Optional tracer provider. If None, uses default.

    Returns:
        True if instrumentation successful, False otherwise.
    """
    if LangChainInstrumentor is None:
        logger.warning("LangChain instrumentation not available")
        return False

    try:
        instrumentor = LangChainInstrumentor()
        if tracer_provider:
            instrumentor.instrument(tracer_provider=tracer_provider)
        else:
            instrumentor.instrument()

        logger.info("LangChain instrumented successfully")
        return True

    except Exception as e:
        logger.error("Failed to instrument LangChain", error=str(e))
        return False


def instrument_openai(tracer_provider: TracerProvider | None = None) -> bool:
    """
    Instrument OpenAI for tracing.

    Args:
        tracer_provider: Optional tracer provider. If None, uses default.

    Returns:
        True if instrumentation successful, False otherwise.
    """
    if OpenAIInstrumentor is None:
        logger.warning("OpenAI instrumentation not available")
        return False

    try:
        instrumentor = OpenAIInstrumentor()
        if tracer_provider:
            instrumentor.instrument(tracer_provider=tracer_provider)
        else:
            instrumentor.instrument()

        logger.info("OpenAI instrumented successfully")
        return True

    except Exception as e:
        logger.error("Failed to instrument OpenAI", error=str(e))
        return False


def instrument_anthropic(tracer_provider: TracerProvider | None = None) -> bool:
    """
    Instrument Anthropic for tracing.

    Args:
        tracer_provider: Optional tracer provider. If None, uses default.

    Returns:
        True if instrumentation successful, False otherwise.
    """
    if AnthropicInstrumentor is None:
        logger.warning("Anthropic instrumentation not available")
        return False

    try:
        instrumentor = AnthropicInstrumentor()
        if tracer_provider:
            instrumentor.instrument(tracer_provider=tracer_provider)
        else:
            instrumentor.instrument()

        logger.info("Anthropic instrumented successfully")
        return True

    except Exception as e:
        logger.error("Failed to instrument Anthropic", error=str(e))
        return False


def get_tracer_provider() -> TracerProvider | None:
    """
    Get the current tracer provider.

    Returns:
        Current tracer provider or None if not initialized.
    """
    if otel_get_tracer_provider is None:
        return None

    try:
        provider = otel_get_tracer_provider()
        # Check if it's the default no-op provider
        if provider.__class__.__name__ == "ProxyTracerProvider":
            return None
        return provider
    except Exception:
        return None


def instrument_all(tracer_provider: TracerProvider | None = None) -> dict[str, bool]:
    """
    Instrument all available providers.

    Args:
        tracer_provider: Optional tracer provider. If None, uses default.

    Returns:
        Dictionary mapping provider names to instrumentation success.
    """
    results = {
        "langchain": instrument_langchain(tracer_provider),
        "openai": instrument_openai(tracer_provider),
        "anthropic": instrument_anthropic(tracer_provider),
    }

    logger.info("Instrumentation complete", results=results)
    return results


@contextmanager
def phoenix_session(session_id: str, user_id: str | None = None):
    """
    Context manager for Phoenix session tracking.

    All traces created within this context will be associated with the session.

    Args:
        session_id: Unique identifier for the session
        user_id: Optional user identifier for additional tracking

    Example:
        with phoenix_session("excel-analysis-123", user_id="user@example.com"):
            # All LLM calls here will be tracked under this session
            llm.invoke("Analyze this data")
    """
    if using_session is not None:
        # Use the official openinference session tracking
        with using_session(session_id=session_id):
            logger.debug("Phoenix session started", session_id=session_id, user_id=user_id)
            yield
    else:
        # Fallback: manually set session attributes
        if trace is None:
            logger.debug("OpenTelemetry trace not available, skipping session tracking")
            yield
            return

        tracer = trace.get_tracer(__name__)

        # Create a root span for the session
        with tracer.start_as_current_span(f"session.{session_id}") as span:
            span.set_attribute("session.id", session_id)
            if user_id:
                span.set_attribute("user.id", user_id)

            logger.debug("Phoenix session started (fallback)", session_id=session_id, user_id=user_id)
            yield


def add_session_metadata(session_id: str, metadata: dict[str, any]) -> None:
    """
    Add metadata to the current session.

    Args:
        session_id: Session identifier
        metadata: Dictionary of metadata to add
    """
    if trace is None:
        logger.debug("OpenTelemetry trace not available, skipping metadata")
        return

    span = trace.get_current_span()
    if span:
        span.set_attribute("session.id", session_id)
        for key, value in metadata.items():
            span.set_attribute(f"session.{key}", value)
