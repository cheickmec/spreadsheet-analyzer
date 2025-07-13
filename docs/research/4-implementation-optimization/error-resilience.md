# Error Handling and Resilience for LLM Systems

## Table of Contents

1. [Introduction](#introduction)
1. [Core Resilience Patterns](#core-resilience-patterns)
1. [Excel-Specific Error Scenarios](#excel-specific-error-scenarios)
1. [Self-Healing Mechanisms](#self-healing-mechanisms)
1. [Distributed System Resilience](#distributed-system-resilience)
1. [Error Logging and Monitoring](#error-logging-and-monitoring)
1. [Testing Methods](#testing-methods)
1. [Implementation Examples](#implementation-examples)
1. [Latest Research (2023-2024)](#latest-research-2023-2024)
1. [Best Practices](#best-practices)

## Introduction

Building resilient LLM systems requires comprehensive error handling strategies that account for the unique challenges of working with language models, including API failures, rate limiting, validation errors, and unpredictable outputs. This document provides a comprehensive guide to implementing robust error handling and resilience patterns for LLM-based spreadsheet analysis systems.

## Core Resilience Patterns

### 1. Retry Pattern with Exponential Backoff

The retry pattern is fundamental for handling transient failures in LLM APIs. Key implementation strategies include:

```python
import time
from functools import wraps
from typing import Callable, Any, Tuple, Optional
import random

def exponential_backoff_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2,
    jitter: bool = True
) -> Callable:
    """
    Decorator implementing exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add randomization to delays
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise

                    # Calculate next delay with exponential backoff
                    delay = min(delay * exponential_base, max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator
```

### 2. Circuit Breaker Pattern

The Circuit Breaker pattern prevents cascading failures by monitoring service health and temporarily blocking requests to failing services:

```python
import time
from enum import Enum
from typing import Callable, Any, Optional
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker implementation for LLM API calls.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
```

### 3. Fallback Pattern

The fallback pattern enables graceful degradation when primary services fail:

```python
from typing import List, Callable, Any, Optional

class FallbackHandler:
    """
    Implements fallback mechanism for LLM services.
    """
    def __init__(self, fallback_models: List[str]):
        self.fallback_models = fallback_models
        self.current_model_index = 0

    def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_funcs: List[Callable],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with fallback options.
        """
        exceptions = []

        # Try primary function
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            exceptions.append(f"Primary failed: {str(e)}")

        # Try fallback functions
        for i, fallback_func in enumerate(fallback_funcs):
            try:
                return fallback_func(*args, **kwargs)
            except Exception as e:
                exceptions.append(f"Fallback {i+1} failed: {str(e)}")

        # All attempts failed
        raise Exception(f"All attempts failed: {'; '.join(exceptions)}")
```

### 4. Timeout Pattern

Implement timeouts to prevent indefinite waiting:

```python
import signal
from contextlib import contextmanager
from typing import Optional

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """
    Context manager for setting execution timeout.
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm
```

## Excel-Specific Error Scenarios

### 1. Circular Reference Detection

```python
from typing import Dict, Set, List, Tuple
import networkx as nx

class CircularReferenceDetector:
    """
    Detects circular references in Excel formulas.
    """
    def __init__(self):
        self.dependency_graph = nx.DiGraph()

    def add_formula(self, cell: str, dependencies: List[str]):
        """
        Add a formula and its dependencies to the graph.
        """
        for dep in dependencies:
            self.dependency_graph.add_edge(dep, cell)

    def detect_circular_references(self) -> List[List[str]]:
        """
        Detect all circular references in the spreadsheet.
        """
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []

    def validate_formula(self, cell: str, dependencies: List[str]) -> bool:
        """
        Check if adding a formula would create a circular reference.
        """
        temp_graph = self.dependency_graph.copy()
        for dep in dependencies:
            temp_graph.add_edge(dep, cell)

        try:
            nx.find_cycle(temp_graph)
            return False  # Circular reference detected
        except nx.NetworkXNoCycle:
            return True  # No circular reference
```

### 2. Corrupt File Handling

```python
import zipfile
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any

class ExcelFileValidator:
    """
    Validates and recovers from corrupt Excel files.
    """
    def __init__(self):
        self.required_files = [
            '[Content_Types].xml',
            'xl/workbook.xml',
            'xl/worksheets/sheet1.xml'
        ]

    def validate_excel_structure(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Excel file structure.
        """
        try:
            with zipfile.ZipFile(file_path, 'r') as excel_zip:
                file_list = excel_zip.namelist()

                # Check for required files
                for required_file in self.required_files:
                    if not any(required_file in f for f in file_list):
                        return False, f"Missing required file: {required_file}"

                # Validate XML structure
                try:
                    workbook_xml = excel_zip.read('xl/workbook.xml')
                    ET.fromstring(workbook_xml)
                except ET.ParseError as e:
                    return False, f"Invalid XML in workbook: {str(e)}"

                return True, None

        except zipfile.BadZipFile:
            return False, "File is not a valid Excel file (corrupt ZIP structure)"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def attempt_recovery(self, file_path: str) -> Optional[str]:
        """
        Attempt to recover data from corrupt Excel file.
        """
        recovered_data = {}

        try:
            with zipfile.ZipFile(file_path, 'r') as excel_zip:
                # Try to extract whatever data we can
                for file_name in excel_zip.namelist():
                    if file_name.startswith('xl/worksheets/') and file_name.endswith('.xml'):
                        try:
                            sheet_data = excel_zip.read(file_name)
                            # Parse and extract cell values
                            recovered_data[file_name] = self._extract_cell_values(sheet_data)
                        except Exception:
                            continue

            return recovered_data if recovered_data else None

        except Exception:
            return None

    def _extract_cell_values(self, sheet_xml: bytes) -> List[Dict[str, Any]]:
        """
        Extract cell values from sheet XML.
        """
        cells = []
        try:
            root = ET.fromstring(sheet_xml)
            # Simplified extraction logic
            for cell in root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                cell_ref = cell.get('r')
                value = cell.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                if value is not None:
                    cells.append({
                        'reference': cell_ref,
                        'value': value.text
                    })
        except Exception:
            pass

        return cells
```

### 3. Large File Handling

```python
import pandas as pd
from typing import Iterator, Dict, Any
import gc

class LargeExcelProcessor:
    """
    Handles large Excel files with memory-efficient processing.
    """
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size

    def process_large_excel(
        self,
        file_path: str,
        sheet_name: str = None
    ) -> Iterator[pd.DataFrame]:
        """
        Process large Excel file in chunks.
        """
        try:
            # Use openpyxl for memory-efficient reading
            with pd.ExcelFile(file_path, engine='openpyxl') as excel_file:
                # Get sheet names if not specified
                if sheet_name is None:
                    sheet_names = excel_file.sheet_names
                else:
                    sheet_names = [sheet_name]

                for sheet in sheet_names:
                    # Read in chunks
                    for chunk_start in range(0, self._get_sheet_size(excel_file, sheet), self.chunk_size):
                        chunk = pd.read_excel(
                            excel_file,
                            sheet_name=sheet,
                            skiprows=chunk_start,
                            nrows=self.chunk_size
                        )

                        yield chunk

                        # Force garbage collection after each chunk
                        gc.collect()

        except MemoryError:
            raise Exception("File too large for available memory")
        except Exception as e:
            raise Exception(f"Error processing large Excel file: {str(e)}")

    def _get_sheet_size(self, excel_file: pd.ExcelFile, sheet_name: str) -> int:
        """
        Get the number of rows in a sheet without loading entire file.
        """
        # This is a simplified version - actual implementation would
        # use openpyxl directly to count rows efficiently
        return 1000000  # Placeholder
```

## Self-Healing Mechanisms

### 1. Automatic Error Recovery

```python
from typing import Dict, Any, Callable, Optional
import logging
from datetime import datetime, timedelta

class SelfHealingSystem:
    """
    Implements self-healing capabilities for LLM systems.
    """
    def __init__(self):
        self.error_history: Dict[str, List[Dict]] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)

    def register_recovery_strategy(
        self,
        error_type: str,
        strategy: Callable
    ):
        """
        Register a recovery strategy for specific error type.
        """
        self.recovery_strategies[error_type] = strategy

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Automatically handle and recover from errors.
        """
        error_type = type(error).__name__

        # Record error
        self._record_error(error_type, error, context)

        # Analyze error pattern
        if self._should_trigger_self_healing(error_type):
            return self._attempt_recovery(error_type, error, context)

        return None

    def _record_error(
        self,
        error_type: str,
        error: Exception,
        context: Dict[str, Any]
    ):
        """
        Record error for pattern analysis.
        """
        if error_type not in self.error_history:
            self.error_history[error_type] = []

        self.error_history[error_type].append({
            'timestamp': datetime.now(),
            'error': str(error),
            'context': context
        })

    def _should_trigger_self_healing(self, error_type: str) -> bool:
        """
        Determine if self-healing should be triggered.
        """
        if error_type not in self.error_history:
            return False

        recent_errors = [
            e for e in self.error_history[error_type]
            if e['timestamp'] > datetime.now() - timedelta(minutes=5)
        ]

        # Trigger if more than 3 errors in 5 minutes
        return len(recent_errors) > 3

    def _attempt_recovery(
        self,
        error_type: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Attempt automatic recovery.
        """
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting self-healing for {error_type}")
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Self-healing failed: {recovery_error}")

        return None
```

### 2. Adaptive Configuration

```python
class AdaptiveConfiguration:
    """
    Dynamically adjusts system configuration based on performance.
    """
    def __init__(self):
        self.performance_metrics = {
            'latency': [],
            'error_rate': [],
            'throughput': []
        }
        self.current_config = {
            'timeout': 30,
            'retry_count': 3,
            'batch_size': 100
        }

    def update_metrics(
        self,
        latency: float,
        success: bool,
        items_processed: int
    ):
        """
        Update performance metrics.
        """
        self.performance_metrics['latency'].append(latency)
        self.performance_metrics['error_rate'].append(0 if success else 1)
        self.performance_metrics['throughput'].append(items_processed / latency)

        # Adapt configuration if needed
        self._adapt_configuration()

    def _adapt_configuration(self):
        """
        Adapt configuration based on recent performance.
        """
        if len(self.performance_metrics['latency']) < 10:
            return

        # Calculate recent averages
        recent_latency = sum(self.performance_metrics['latency'][-10:]) / 10
        recent_error_rate = sum(self.performance_metrics['error_rate'][-10:]) / 10

        # Adjust timeout if latency is high
        if recent_latency > self.current_config['timeout'] * 0.8:
            self.current_config['timeout'] = int(self.current_config['timeout'] * 1.2)

        # Adjust retry count if error rate is high
        if recent_error_rate > 0.2:
            self.current_config['retry_count'] = min(5, self.current_config['retry_count'] + 1)

        # Adjust batch size based on throughput
        avg_throughput = sum(self.performance_metrics['throughput'][-10:]) / 10
        if avg_throughput < 10:
            self.current_config['batch_size'] = max(10, self.current_config['batch_size'] // 2)
```

## Distributed System Resilience

### 1. Distributed Circuit Breaker

```python
import redis
from typing import Optional
import json

class DistributedCircuitBreaker:
    """
    Circuit breaker that works across distributed systems.
    """
    def __init__(
        self,
        redis_client: redis.Redis,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.redis = redis_client
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    def is_open(self) -> bool:
        """
        Check if circuit is open across all instances.
        """
        state = self.redis.get(f"circuit:{self.service_name}:state")
        return state == b"open" if state else False

    def record_success(self):
        """
        Record successful call.
        """
        pipe = self.redis.pipeline()
        pipe.delete(f"circuit:{self.service_name}:failures")
        pipe.set(f"circuit:{self.service_name}:state", "closed")
        pipe.execute()

    def record_failure(self):
        """
        Record failed call and potentially open circuit.
        """
        failure_count = self.redis.incr(f"circuit:{self.service_name}:failures")

        if failure_count >= self.failure_threshold:
            pipe = self.redis.pipeline()
            pipe.set(f"circuit:{self.service_name}:state", "open")
            pipe.expire(f"circuit:{self.service_name}:state", self.recovery_timeout)
            pipe.execute()
```

### 2. Service Mesh Integration

```python
from typing import Dict, Any, Optional
import requests
from dataclasses import dataclass

@dataclass
class ServiceEndpoint:
    host: str
    port: int
    weight: int = 1
    healthy: bool = True

class ServiceMeshClient:
    """
    Client for service mesh integration with built-in resilience.
    """
    def __init__(self, service_registry: Dict[str, List[ServiceEndpoint]]):
        self.service_registry = service_registry
        self.health_check_interval = 30

    def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        **kwargs
    ) -> requests.Response:
        """
        Call a service with automatic failover and load balancing.
        """
        endpoints = self._get_healthy_endpoints(service_name)

        if not endpoints:
            raise Exception(f"No healthy endpoints for service {service_name}")

        # Try each endpoint until success
        last_error = None
        for endpoint in endpoints:
            try:
                url = f"http://{endpoint.host}:{endpoint.port}{endpoint}"
                response = requests.request(method, url, **kwargs)

                if response.status_code < 500:
                    return response

            except Exception as e:
                last_error = e
                self._mark_unhealthy(service_name, endpoint)

        raise Exception(f"All endpoints failed. Last error: {last_error}")

    def _get_healthy_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        """
        Get list of healthy endpoints for a service.
        """
        if service_name not in self.service_registry:
            return []

        return [ep for ep in self.service_registry[service_name] if ep.healthy]

    def _mark_unhealthy(self, service_name: str, endpoint: ServiceEndpoint):
        """
        Mark an endpoint as unhealthy.
        """
        endpoint.healthy = False
        # Schedule health check
        self._schedule_health_check(service_name, endpoint)
```

## Error Logging and Monitoring

### 1. Structured Logging

```python
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

class StructuredLogger:
    """
    Structured logging for LLM applications.
    """
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

        # Configure JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'service': record.name,
                'message': record.getMessage(),
                'trace_id': getattr(record, 'trace_id', None),
                'span_id': getattr(record, 'span_id', None),
            }

            if record.exc_info:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }

            return json.dumps(log_data)

    def log_llm_request(
        self,
        model: str,
        prompt: str,
        response: Optional[str] = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log LLM request with full context.
        """
        log_data = {
            'event_type': 'llm_request',
            'model': model,
            'prompt_length': len(prompt),
            'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'metadata': metadata or {}
        }

        if response:
            log_data['response_length'] = len(response)
            log_data['success'] = True

        if error:
            log_data['success'] = False
            log_data['error_type'] = type(error).__name__
            log_data['error_message'] = str(error)

        self.logger.info("LLM Request", extra=log_data)
```

### 2. Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from contextlib import contextmanager
from typing import Dict, Any, Optional

class DistributedTracer:
    """
    Distributed tracing for LLM applications using OpenTelemetry.
    """
    def __init__(self, service_name: str, endpoint: str):
        # Set up tracer provider
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        self.tracer = trace.get_tracer(service_name)

    @contextmanager
    def trace_llm_call(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Trace an LLM call with automatic error handling.
        """
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                span.record_exception(e)
                raise

    def trace_excel_processing(
        self,
        file_name: str,
        operation: str
    ):
        """
        Specialized tracing for Excel operations.
        """
        attributes = {
            'excel.file_name': file_name,
            'excel.operation': operation,
            'excel.timestamp': datetime.utcnow().isoformat()
        }

        return self.trace_llm_call(
            f"excel.{operation}",
            attributes=attributes
        )
```

### 3. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time
from typing import Dict, Any

class LLMMetricsCollector:
    """
    Collects and exposes metrics for LLM operations.
    """
    def __init__(self):
        # Define metrics
        self.request_count = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['model', 'status']
        )

        self.request_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration',
            ['model']
        )

        self.token_usage = Counter(
            'llm_tokens_total',
            'Total tokens used',
            ['model', 'token_type']
        )

        self.error_rate = Gauge(
            'llm_error_rate',
            'Current error rate',
            ['model']
        )

        self.active_requests = Gauge(
            'llm_active_requests',
            'Number of active requests',
            ['model']
        )

    def record_request(
        self,
        model: str,
        duration: float,
        success: bool,
        tokens_used: Optional[Dict[str, int]] = None
    ):
        """
        Record metrics for an LLM request.
        """
        # Increment request count
        status = 'success' if success else 'error'
        self.request_count.labels(model=model, status=status).inc()

        # Record duration
        self.request_duration.labels(model=model).observe(duration)

        # Record token usage
        if tokens_used:
            for token_type, count in tokens_used.items():
                self.token_usage.labels(
                    model=model,
                    token_type=token_type
                ).inc(count)

    @contextmanager
    def track_request(self, model: str):
        """
        Context manager to track request metrics.
        """
        self.active_requests.labels(model=model).inc()
        start_time = time.time()

        try:
            yield
            duration = time.time() - start_time
            self.record_request(model, duration, success=True)
        except Exception as e:
            duration = time.time() - start_time
            self.record_request(model, duration, success=False)
            raise
        finally:
            self.active_requests.labels(model=model).dec()
```

## Testing Methods

### 1. Chaos Engineering for LLMs

```python
import random
from typing import Callable, Any, Dict, List
from functools import wraps

class LLMChaosEngineer:
    """
    Chaos engineering for testing LLM system resilience.
    """
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.fault_injectors = {
            'timeout': self._inject_timeout,
            'rate_limit': self._inject_rate_limit,
            'invalid_response': self._inject_invalid_response,
            'partial_response': self._inject_partial_response
        }

    def chaos_wrapper(self, fault_types: List[str] = None):
        """
        Decorator to inject faults into LLM calls.
        """
        if fault_types is None:
            fault_types = list(self.fault_injectors.keys())

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if random.random() < self.failure_rate:
                    fault_type = random.choice(fault_types)
                    return self.fault_injectors[fault_type](func, *args, **kwargs)

                return func(*args, **kwargs)

            return wrapper
        return decorator

    def _inject_timeout(self, func: Callable, *args, **kwargs):
        """
        Simulate timeout by sleeping and raising exception.
        """
        import time
        time.sleep(35)  # Simulate long delay
        raise TimeoutError("LLM request timed out")

    def _inject_rate_limit(self, func: Callable, *args, **kwargs):
        """
        Simulate rate limiting error.
        """
        raise Exception("Rate limit exceeded. Please retry after 60 seconds.")

    def _inject_invalid_response(self, func: Callable, *args, **kwargs):
        """
        Return invalid response format.
        """
        return {"error": "Invalid response format", "data": None}

    def _inject_partial_response(self, func: Callable, *args, **kwargs):
        """
        Return truncated response.
        """
        result = func(*args, **kwargs)
        if isinstance(result, str):
            return result[:len(result)//2] + "..."
        return result
```

### 2. Property-Based Testing

```python
from hypothesis import strategies as st, given, settings
import pandas as pd
from typing import Any

class ExcelPropertyTester:
    """
    Property-based testing for Excel processing functions.
    """

    @staticmethod
    @given(
        data=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False),
                    st.text()
                )
            ),
            min_size=1,
            max_size=1000
        )
    )
    @settings(max_examples=100)
    def test_excel_processing_properties(data: List[Dict[str, Any]]):
        """
        Test that Excel processing maintains data integrity.
        """
        # Create DataFrame
        df = pd.DataFrame(data)

        # Process (placeholder for actual processing)
        processed_df = process_excel_data(df)

        # Properties to test
        assert len(processed_df) == len(df), "Row count should be preserved"
        assert set(processed_df.columns) == set(df.columns), "Columns should be preserved"
        assert processed_df.dtypes.equals(df.dtypes), "Data types should be preserved"
```

### 3. Fault Injection Testing

```python
class FaultInjectionTester:
    """
    Systematic fault injection for testing error handling.
    """
    def __init__(self):
        self.test_scenarios = []

    def add_scenario(
        self,
        name: str,
        fault_function: Callable,
        expected_behavior: str
    ):
        """
        Add a fault injection scenario.
        """
        self.test_scenarios.append({
            'name': name,
            'fault': fault_function,
            'expected': expected_behavior
        })

    def run_tests(self, system_under_test: Any) -> Dict[str, bool]:
        """
        Run all fault injection tests.
        """
        results = {}

        for scenario in self.test_scenarios:
            try:
                # Inject fault
                scenario['fault'](system_under_test)

                # Check if system behaved as expected
                # This is simplified - actual implementation would be more complex
                results[scenario['name']] = True

            except Exception as e:
                results[scenario['name']] = False
                print(f"Test {scenario['name']} failed: {str(e)}")

        return results

# Example fault scenarios
def create_excel_fault_scenarios():
    tester = FaultInjectionTester()

    # Corrupt file scenario
    tester.add_scenario(
        "corrupt_excel_file",
        lambda system: system.process_file("corrupt.xlsx"),
        "Should return error message and attempt recovery"
    )

    # Circular reference scenario
    tester.add_scenario(
        "circular_reference",
        lambda system: system.add_formula("A1", "=A1+1"),
        "Should detect and report circular reference"
    )

    # Memory exhaustion scenario
    tester.add_scenario(
        "large_file_processing",
        lambda system: system.process_file("10gb_file.xlsx"),
        "Should process in chunks without memory error"
    )

    return tester
```

## Implementation Examples

### 1. Complete Resilient LLM Client

```python
from typing import Optional, Dict, Any, List
import asyncio
from dataclasses import dataclass
import aiohttp

@dataclass
class LLMConfig:
    primary_model: str
    fallback_models: List[str]
    timeout: float = 30.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5

class ResilientLLMClient:
    """
    Production-ready LLM client with comprehensive error handling.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold
        )
        self.metrics = LLMMetricsCollector()
        self.logger = StructuredLogger("llm_client")
        self.self_healing = SelfHealingSystem()

        # Register recovery strategies
        self._register_recovery_strategies()

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Get completion with full resilience features.
        """
        # Try primary model first
        models_to_try = [self.config.primary_model] + self.config.fallback_models

        for model in models_to_try:
            try:
                # Check circuit breaker
                if self.circuit_breaker.state == CircuitState.OPEN:
                    continue

                # Make request with metrics tracking
                with self.metrics.track_request(model):
                    response = await self._make_request(
                        model, prompt, temperature, max_tokens
                    )

                # Log success
                self.logger.log_llm_request(
                    model=model,
                    prompt=prompt,
                    response=response['text']
                )

                return response

            except Exception as e:
                # Handle error with self-healing
                recovery_result = self.self_healing.handle_error(
                    e,
                    {
                        'model': model,
                        'prompt': prompt,
                        'temperature': temperature,
                        'max_tokens': max_tokens
                    }
                )

                if recovery_result:
                    return recovery_result

                # Log failure
                self.logger.log_llm_request(
                    model=model,
                    prompt=prompt,
                    error=e
                )

                # Continue to next model
                continue

        # All models failed
        raise Exception("All LLM models failed")

    @exponential_backoff_retry(max_retries=3)
    async def _make_request(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Make actual API request with retry logic.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.llm.com/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")

                response.raise_for_status()
                return await response.json()

    def _register_recovery_strategies(self):
        """
        Register self-healing recovery strategies.
        """
        # Rate limit recovery
        self.self_healing.register_recovery_strategy(
            'RateLimitError',
            lambda e, ctx: self._handle_rate_limit(ctx)
        )

        # Timeout recovery
        self.self_healing.register_recovery_strategy(
            'TimeoutError',
            lambda e, ctx: self._handle_timeout(ctx)
        )
```

### 2. Excel Processing with Error Resilience

```python
class ResilientExcelProcessor:
    """
    Excel processor with comprehensive error handling.
    """
    def __init__(self, llm_client: ResilientLLMClient):
        self.llm_client = llm_client
        self.validator = ExcelFileValidator()
        self.circular_detector = CircularReferenceDetector()
        self.logger = StructuredLogger("excel_processor")

    async def process_spreadsheet(
        self,
        file_path: str,
        analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Process spreadsheet with full error handling.
        """
        # Validate file first
        is_valid, error_msg = self.validator.validate_excel_structure(file_path)

        if not is_valid:
            self.logger.logger.warning(f"Invalid Excel file: {error_msg}")

            # Attempt recovery
            recovered_data = self.validator.attempt_recovery(file_path)
            if recovered_data:
                return await self._process_recovered_data(recovered_data)
            else:
                raise ValueError(f"Cannot process file: {error_msg}")

        # Process valid file
        try:
            # Check for circular references
            df = pd.read_excel(file_path)
            formulas = self._extract_formulas(df)

            for cell, formula in formulas.items():
                deps = self._parse_dependencies(formula)
                if not self.circular_detector.validate_formula(cell, deps):
                    self.logger.logger.warning(
                        f"Circular reference detected in {cell}"
                    )

            # Perform LLM analysis
            analysis_prompt = self._create_analysis_prompt(df, analysis_type)

            result = await self.llm_client.complete(
                prompt=analysis_prompt,
                temperature=0.3,
                max_tokens=2000
            )

            return {
                'status': 'success',
                'analysis': result['text'],
                'metadata': {
                    'file': file_path,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'circular_refs': self.circular_detector.detect_circular_references()
                }
            }

        except Exception as e:
            self.logger.logger.error(
                f"Error processing spreadsheet: {str(e)}",
                exc_info=True
            )
            raise
```

## Latest Research (2023-2024)

### 1. Self-Healing Machine Learning Framework (2024)

The Self-Healing Machine Learning (SHML) framework represents a significant advancement in autonomous system adaptation. Key features include:

- **H-LLM Agent**: Uses LLMs to perform self-diagnosis by reasoning about data generation processes
- **Autonomous Adaptation**: Proposes and evaluates corrective actions without human intervention
- **Performance**: 85% MTTR reduction, 95%+ recovery reliability, 98%+ system uptime

### 2. SpreadsheetLLM (Microsoft, 2024)

Microsoft's SpreadsheetLLM addresses unique challenges in processing spreadsheet data:

- **SheetCompressor**: Achieves up to 96% compression for large datasets
- **Performance**: 12.3% improvement over existing methods in table detection
- **Limitations**: Currently ignores cell formatting to reduce token usage

### 3. Distributed Resilience Patterns (2024)

Recent implementations show impressive metrics:

- Recovery time reduced from 90 to 13.5 minutes
- Fault tolerance above 91%
- Resource overhead during recovery under 10%

### 4. LLM-Specific Observability Tools (2024)

Major developments in monitoring:

- **OpenTelemetry Integration**: Standardized tracing for LLM applications
- **Specialized Tools**: Langfuse, Helicone for open-source; Datadog, Dynatrace for enterprise
- **Metrics Focus**: Token usage, latency, cost optimization, hallucination detection

## Best Practices

### 1. Error Handling Strategy

1. **Layer Your Defenses**

   - Implement multiple resilience patterns (retry, circuit breaker, fallback)
   - Don't rely on a single error handling mechanism
   - Combine patterns intelligently (e.g., retry with circuit breaker)

1. **Fail Fast, Recover Faster**

   - Set appropriate timeouts to avoid hanging requests
   - Implement health checks and proactive monitoring
   - Use circuit breakers to prevent cascade failures

1. **Graceful Degradation**

   - Always have fallback options (alternative models, cached responses)
   - Provide partial functionality rather than complete failure
   - Communicate degraded state to users

### 2. Monitoring and Observability

1. **Structured Logging**

   - Use consistent log formats (JSON)
   - Include trace IDs for request correlation
   - Log both successes and failures with context

1. **Comprehensive Metrics**

   - Track latency, error rates, token usage
   - Monitor business-specific metrics
   - Set up alerting based on anomaly detection

1. **Distributed Tracing**

   - Implement end-to-end tracing for complex workflows
   - Use OpenTelemetry for vendor-neutral instrumentation
   - Trace both LLM calls and supporting infrastructure

### 3. Testing and Validation

1. **Chaos Engineering**

   - Regularly inject failures in controlled environments
   - Test all failure modes (timeouts, rate limits, invalid responses)
   - Automate chaos experiments in CI/CD pipelines

1. **Property-Based Testing**

   - Test invariants that should always hold
   - Generate diverse test cases automatically
   - Focus on edge cases and boundary conditions

1. **Load Testing**

   - Test system behavior under stress
   - Identify breaking points and bottlenecks
   - Validate auto-scaling and circuit breaker thresholds

### 4. Excel-Specific Considerations

1. **File Validation**

   - Always validate file structure before processing
   - Implement recovery mechanisms for corrupt files
   - Handle large files with streaming/chunking

1. **Formula Handling**

   - Detect and report circular references
   - Validate formula syntax before execution
   - Implement sandboxing for formula evaluation

1. **Memory Management**

   - Process large files in chunks
   - Implement proper garbage collection
   - Monitor memory usage and set limits

### 5. Cost Optimization

1. **Smart Retries**

   - Use exponential backoff to reduce API costs
   - Implement request deduplication
   - Cache responses when appropriate

1. **Model Selection**

   - Use cheaper models for simple tasks
   - Implement dynamic model selection based on complexity
   - Monitor cost per request and optimize

1. **Token Optimization**

   - Compress prompts without losing context
   - Implement prompt caching for repeated queries
   - Monitor and optimize token usage patterns

## Conclusion

Building resilient LLM systems requires a comprehensive approach that combines traditional distributed systems patterns with LLM-specific considerations. By implementing proper error handling, monitoring, and testing strategies, you can build systems that gracefully handle failures and provide reliable service even in the face of adversity.

The key is to assume failures will happen and design your system to handle them gracefully. With the patterns and practices outlined in this document, you can build robust LLM-based spreadsheet analysis systems that deliver consistent value to users while minimizing downtime and operational overhead.
