# Progress Tracking System Design (Functional Programming Approach)

## Overview

The Progress Tracking System provides real-time updates and monitoring for long-running Excel analysis operations using functional programming principles. The system treats progress as an immutable stream of events, ensuring thread safety and predictable behavior.

## Design Philosophy

This implementation follows functional programming principles:

- **Immutable Data**: Progress updates are immutable records
- **Pure Functions**: All calculations are deterministic and side-effect free
- **Event Streams**: Progress is modeled as a stream of immutable events
- **Minimal State**: State management isolated to system boundaries

## Design Principles

1. **Immutability First**: Progress updates are immutable events in time
1. **Pure Calculations**: Time estimations and progress calculations are pure functions
1. **Stream Processing**: Progress events form a functional stream
1. **Composable Operations**: Filter, map, and reduce operations on progress streams
1. **Effect Isolation**: Side effects (broadcasting, I/O) isolated at boundaries

## Architecture

### Core Immutable Types

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple, Iterator, FrozenSet
from enum import Enum
import statistics
from collections import defaultdict

class ProgressLevel(Enum):
    WORKBOOK = "workbook"
    SHEET = "sheet"
    OPERATION = "operation"
    DETAIL = "detail"

class OperationType(Enum):
    LOADING = "loading"
    PARSING = "parsing"
    STRUCTURE_ANALYSIS = "structure_analysis"
    FORMULA_ANALYSIS = "formula_analysis"
    VALIDATION = "validation"
    AI_ANALYSIS = "ai_analysis"

@dataclass(frozen=True)
class ProgressUpdate:
    """Immutable progress update event."""
    timestamp: datetime
    level: ProgressLevel
    operation: OperationType
    current: int
    total: int
    message: str
    metadata: FrozenSet[Tuple[str, Any]] = field(default_factory=frozenset)

    @property
    def percentage(self) -> float:
        """Calculate completion percentage - pure function."""
        return (self.current / self.total * 100) if self.total > 0 else 0.0

    def with_metadata(self, **kwargs) -> 'ProgressUpdate':
        """Return new update with additional metadata."""
        new_metadata = set(self.metadata)
        new_metadata.update(kwargs.items())
        return dataclass.replace(self, metadata=frozenset(new_metadata))

@dataclass(frozen=True)
class ResourceSnapshot:
    """Immutable system resource state."""
    timestamp: datetime
    memory_mb: float
    cpu_percent: float

@dataclass(frozen=True)
class ProgressState:
    """Immutable progress state at a point in time."""
    updates: Tuple[ProgressUpdate, ...]
    start_time: datetime
    resource_snapshots: Tuple[ResourceSnapshot, ...]

    def with_update(self, update: ProgressUpdate) -> 'ProgressState':
        """Return new state with added update - pure function."""
        return ProgressState(
            updates=(*self.updates, update),
            start_time=self.start_time,
            resource_snapshots=self.resource_snapshots
        )

    def with_resource_snapshot(self, snapshot: ResourceSnapshot) -> 'ProgressState':
        """Return new state with added resource snapshot."""
        return ProgressState(
            updates=self.updates,
            start_time=self.start_time,
            resource_snapshots=(*self.resource_snapshots, snapshot)
        )
```

### Pure Functions for Progress Calculations

```python
def calculate_operation_stats(
    updates: Tuple[ProgressUpdate, ...]
) -> Dict[OperationType, Dict[str, float]]:
    """
    Calculate statistics for each operation type - pure function.

    CLAUDE-KNOWLEDGE: This function has no side effects and always
    returns the same output for the same input.
    """
    # Group updates by operation
    operation_durations: Dict[OperationType, List[float]] = defaultdict(list)

    # Track operation start times
    operation_starts: Dict[OperationType, datetime] = {}

    for update in updates:
        if update.operation not in operation_starts:
            operation_starts[update.operation] = update.timestamp

        # If operation completed
        if update.current == update.total and update.operation in operation_starts:
            duration = (update.timestamp - operation_starts[update.operation]).total_seconds()
            operation_durations[update.operation].append(duration)
            del operation_starts[update.operation]

    # Calculate statistics
    stats = {}
    for op_type, durations in operation_durations.items():
        if durations:
            stats[op_type] = {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'stdev': statistics.stdev(durations) if len(durations) > 1 else 0.0,
                'count': len(durations)
            }

    return stats

def estimate_remaining_time(
    state: ProgressState,
    remaining_operations: List[OperationType]
) -> timedelta:
    """
    Estimate time remaining based on historical data - pure function.

    CLAUDE-COMPLEX: Uses statistical analysis of past operations
    to predict future duration without any side effects.
    """
    stats = calculate_operation_stats(state.updates)

    total_seconds = 0.0
    for op in remaining_operations:
        if op in stats:
            # Use median as it's more robust to outliers
            total_seconds += stats[op]['median']
        else:
            # Conservative estimate for unknown operations
            total_seconds += 30.0

    return timedelta(seconds=total_seconds)

def calculate_overall_progress(updates: Tuple[ProgressUpdate, ...]) -> float:
    """Calculate overall progress percentage - pure function."""
    if not updates:
        return 0.0

    # Get latest update for each operation type
    latest_by_op: Dict[OperationType, ProgressUpdate] = {}
    for update in updates:
        if update.level == ProgressLevel.OPERATION:
            latest_by_op[update.operation] = update

    if not latest_by_op:
        return 0.0

    # Calculate average completion across all operations
    total_percentage = sum(u.percentage for u in latest_by_op.values())
    return total_percentage / len(latest_by_op)

def filter_updates_by_level(
    updates: Iterator[ProgressUpdate],
    level: ProgressLevel
) -> Iterator[ProgressUpdate]:
    """Filter updates by progress level - pure function."""
    return filter(lambda u: u.level == level, updates)

def get_recent_updates(
    updates: Tuple[ProgressUpdate, ...],
    since: datetime
) -> Tuple[ProgressUpdate, ...]:
    """Get updates since a specific time - pure function."""
    return tuple(u for u in updates if u.timestamp >= since)
```

### Functional Progress Stream Processing

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class ProgressStream(Generic[T]):
    """
    Functional stream for progress updates.

    CLAUDE-KNOWLEDGE: This implements a lazy, functional stream
    similar to Java 8 Streams or Scala collections.
    """

    def __init__(self, source: Iterator[T]):
        self._source = source

    def filter(self, predicate: Callable[[T], bool]) -> 'ProgressStream[T]':
        """Filter stream elements - returns new stream."""
        return ProgressStream(filter(predicate, self._source))

    def map(self, mapper: Callable[[T], Any]) -> 'ProgressStream[Any]':
        """Transform stream elements - returns new stream."""
        return ProgressStream(map(mapper, self._source))

    def take(self, n: int) -> 'ProgressStream[T]':
        """Take first n elements - returns new stream."""
        def take_n():
            count = 0
            for item in self._source:
                if count >= n:
                    break
                yield item
                count += 1
        return ProgressStream(take_n())

    def collect(self) -> List[T]:
        """Terminal operation - collect stream to list."""
        return list(self._source)

    def reduce(self, reducer: Callable[[T, T], T], initial: T) -> T:
        """Terminal operation - reduce stream to single value."""
        result = initial
        for item in self._source:
            result = reducer(result, item)
        return result

# Usage examples
def analyze_progress_stream(updates: Tuple[ProgressUpdate, ...]) -> Dict[str, Any]:
    """Analyze progress using functional stream operations."""
    stream = ProgressStream(iter(updates))

    # Get sheet-level updates with high completion
    high_completion_sheets = (
        stream
        .filter(lambda u: u.level == ProgressLevel.SHEET)
        .filter(lambda u: u.percentage > 80)
        .map(lambda u: u.message)
        .collect()
    )

    return {
        'high_completion_sheets': high_completion_sheets,
        'total_updates': len(updates),
        'overall_progress': calculate_overall_progress(updates)
    }
```

### Progress Tracker with Minimal State

```python
import asyncio
from typing import Set

class FunctionalProgressTracker:
    """
    Progress tracker using functional principles with minimal state.

    CLAUDE-COMPLEX: State is managed immutably, with all calculations
    performed by pure functions. Only broadcasting is stateful.
    """

    def __init__(self):
        # Immutable state
        self._state = ProgressState(
            updates=(),
            start_time=datetime.now(),
            resource_snapshots=()
        )
        # Minimal mutable state for observer management
        self._observers: Set[Callable[[ProgressUpdate], None]] = set()
        self._state_lock = asyncio.Lock()

    def subscribe(self, observer: Callable[[ProgressUpdate], None]) -> Callable[[], None]:
        """
        Subscribe to updates - returns unsubscribe function.

        CLAUDE-KNOWLEDGE: Returning the unsubscribe function is a
        functional pattern that avoids needing to track observer identity.
        """
        self._observers.add(observer)
        return lambda: self._observers.discard(observer)

    async def emit_update(
        self,
        level: ProgressLevel,
        operation: OperationType,
        current: int,
        total: int,
        message: str,
        **metadata
    ) -> ProgressState:
        """
        Emit progress update and return new state.

        Pure function except for broadcasting side effect.
        """
        update = ProgressUpdate(
            timestamp=datetime.now(),
            level=level,
            operation=operation,
            current=current,
            total=total,
            message=message,
            metadata=frozenset(metadata.items())
        )

        # Update state immutably
        async with self._state_lock:
            self._state = self._state.with_update(update)
            new_state = self._state

        # Broadcasting is the only side effect
        await self._broadcast(update)

        return new_state

    async def _broadcast(self, update: ProgressUpdate) -> None:
        """Broadcast update to observers - isolated side effect."""
        await asyncio.gather(*[
            self._notify_observer(obs, update)
            for obs in self._observers
        ], return_exceptions=True)

    async def _notify_observer(
        self,
        observer: Callable[[ProgressUpdate], None],
        update: ProgressUpdate
    ) -> None:
        """Notify single observer, handling sync/async."""
        if asyncio.iscoroutinefunction(observer):
            await observer(update)
        else:
            # Run sync observer in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, observer, update
            )

    def get_state(self) -> ProgressState:
        """Get current immutable state."""
        return self._state

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze current progress - pure function on immutable state.

        CLAUDE-KNOWLEDGE: All analysis is performed by pure functions
        on the immutable state snapshot.
        """
        state = self._state
        elapsed = datetime.now() - state.start_time

        # Identify remaining work
        completed_ops = {
            u.operation for u in state.updates
            if u.current == u.total and u.level == ProgressLevel.OPERATION
        }
        all_ops = {u.operation for u in state.updates if u.level == ProgressLevel.OPERATION}
        remaining_ops = list(all_ops - completed_ops)

        # Pure functional analysis
        return {
            'elapsed': elapsed,
            'overall_progress': calculate_overall_progress(state.updates),
            'operation_stats': calculate_operation_stats(state.updates),
            'estimated_remaining': estimate_remaining_time(state, remaining_ops),
            'recent_updates': get_recent_updates(
                state.updates,
                datetime.now() - timedelta(minutes=1)
            ),
            'resource_usage': self._analyze_resources(state.resource_snapshots)
        }

    def _analyze_resources(
        self,
        snapshots: Tuple[ResourceSnapshot, ...]
    ) -> Dict[str, float]:
        """Analyze resource usage - pure function."""
        if not snapshots:
            return {'avg_memory_mb': 0.0, 'avg_cpu_percent': 0.0}

        memory_values = [s.memory_mb for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]

        return {
            'avg_memory_mb': statistics.mean(memory_values),
            'peak_memory_mb': max(memory_values),
            'avg_cpu_percent': statistics.mean(cpu_values),
            'peak_cpu_percent': max(cpu_values)
        }
```

### Resource Monitoring as Pure Functions

```python
import psutil

def create_resource_monitor() -> Callable[[], ResourceSnapshot]:
    """
    Factory for resource monitoring function.

    CLAUDE-KNOWLEDGE: The process handle is captured in closure,
    making the returned function referentially transparent.
    """
    process = psutil.Process()

    def get_snapshot() -> ResourceSnapshot:
        """Get current resource snapshot - deterministic for given system state."""
        return ResourceSnapshot(
            timestamp=datetime.now(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            cpu_percent=process.cpu_percent(interval=0.1)
        )

    return get_snapshot

class ResourceMonitor:
    """Resource monitor with functional design."""

    def __init__(self, tracker: FunctionalProgressTracker):
        self.tracker = tracker
        self.get_snapshot = create_resource_monitor()
        self._monitoring = False

    async def monitor_periodically(self, interval_seconds: float = 5.0) -> None:
        """Monitor resources periodically."""
        self._monitoring = True

        while self._monitoring:
            snapshot = self.get_snapshot()

            # Update tracker state with resource snapshot
            async with self.tracker._state_lock:
                self.tracker._state = self.tracker._state.with_resource_snapshot(snapshot)

            # Alert if threshold exceeded (pure function)
            if self._should_alert(snapshot):
                await self.tracker.emit_update(
                    ProgressLevel.WORKBOOK,
                    OperationType.PARSING,
                    0, 0,
                    f"High resource usage: {snapshot.memory_mb:.1f}MB RAM, {snapshot.cpu_percent:.1f}% CPU",
                    warning=True,
                    memory_mb=snapshot.memory_mb,
                    cpu_percent=snapshot.cpu_percent
                )

            await asyncio.sleep(interval_seconds)

    def _should_alert(self, snapshot: ResourceSnapshot) -> bool:
        """Determine if resource usage warrants alert - pure function."""
        return snapshot.memory_mb > 500 or snapshot.cpu_percent > 80

    def stop(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
```

### Progress Analysis Pipeline

```python
def create_analysis_pipeline(
    tracker: FunctionalProgressTracker
) -> Callable[[Path], AnalysisResult]:
    """
    Create analysis pipeline with progress tracking.

    CLAUDE-KNOWLEDGE: Returns a function that performs analysis
    with progress tracking as a series of composed operations.
    """

    async def analyze_with_progress(file_path: Path) -> AnalysisResult:
        """Analyze Excel file with functional progress tracking."""

        # Compose analysis steps with progress updates
        async def load_step():
            await tracker.emit_update(
                ProgressLevel.WORKBOOK,
                OperationType.LOADING,
                0, 1,
                f"Loading {file_path.name}"
            )
            workbook = await load_workbook(file_path)
            await tracker.emit_update(
                ProgressLevel.WORKBOOK,
                OperationType.LOADING,
                1, 1,
                f"Loaded {len(workbook.sheets)} sheets"
            )
            return workbook

        async def analyze_sheet_step(sheet, index: int, total: int):
            await tracker.emit_update(
                ProgressLevel.SHEET,
                OperationType.PARSING,
                index, total,
                f"Analyzing sheet: {sheet.name}"
            )

            # Structure analysis
            structure = await analyze_structure(sheet)

            # Formula analysis if needed
            formulas = None
            if structure.has_formulas:
                formulas = await analyze_formulas_with_progress(
                    sheet, structure.formula_count, tracker
                )

            return SheetAnalysis(sheet, structure, formulas)

        # Execute pipeline
        workbook = await load_step()

        # Analyze sheets in parallel with progress
        sheet_analyses = await asyncio.gather(*[
            analyze_sheet_step(sheet, idx, len(workbook.sheets))
            for idx, sheet in enumerate(workbook.sheets)
        ])

        return AnalysisResult(workbook, sheet_analyses)

    return analyze_with_progress

async def analyze_formulas_with_progress(
    sheet: Worksheet,
    formula_count: int,
    tracker: FunctionalProgressTracker
) -> FormulaAnalysis:
    """Analyze formulas with progress updates."""

    # Create progress callback
    async def progress_callback(current: int):
        await tracker.emit_update(
            ProgressLevel.DETAIL,
            OperationType.FORMULA_ANALYSIS,
            current, formula_count,
            f"Formula {current}/{formula_count}"
        )

    # Analyze with progress
    return await analyze_formulas(sheet, progress_callback)
```

### Functional Progress Observers

```python
def create_console_observer(
    min_level: ProgressLevel = ProgressLevel.SHEET
) -> Callable[[ProgressUpdate], None]:
    """
    Create console observer with filtering.

    CLAUDE-KNOWLEDGE: Returns a pure function that performs I/O
    as its only side effect.
    """
    def observe(update: ProgressUpdate) -> None:
        if update.level.value <= min_level.value:
            print(f"[{update.timestamp.strftime('%H:%M:%S')}] "
                  f"{update.operation.value}: {update.message} "
                  f"({update.percentage:.1f}%)")

    return observe

def create_metrics_observer() -> Callable[[ProgressUpdate], Dict[str, Any]]:
    """
    Create observer that collects metrics.

    Returns a function that accumulates metrics in a functional way.
    """
    metrics: List[ProgressUpdate] = []

    def observe(update: ProgressUpdate) -> Dict[str, Any]:
        metrics.append(update)

        # Return current metrics summary (pure calculation)
        return {
            'total_updates': len(metrics),
            'operations': list({u.operation for u in metrics}),
            'average_completion': statistics.mean(
                u.percentage for u in metrics
            ) if metrics else 0.0
        }

    return observe
```

## Usage Example

```python
async def analyze_excel_functional(file_path: Path) -> AnalysisResult:
    """Example of functional progress tracking in action."""

    # Create tracker and observers
    tracker = FunctionalProgressTracker()

    # Subscribe observers (returns unsubscribe functions)
    unsubscribe_console = tracker.subscribe(
        create_console_observer(ProgressLevel.SHEET)
    )

    # Create analysis pipeline
    analyze = create_analysis_pipeline(tracker)

    # Start resource monitoring
    monitor = ResourceMonitor(tracker)
    monitor_task = asyncio.create_task(
        monitor.monitor_periodically(interval_seconds=5.0)
    )

    try:
        # Run analysis
        result = await analyze(file_path)

        # Get final analysis
        final_analysis = tracker.analyze()
        print(f"\nCompleted in {final_analysis['elapsed']}")
        print(f"Overall progress: {final_analysis['overall_progress']:.1f}%")
        print(f"Peak memory: {final_analysis['resource_usage']['peak_memory_mb']:.1f}MB")

        return result

    finally:
        # Cleanup
        monitor.stop()
        await monitor_task
        unsubscribe_console()
```

## Testing with Pure Functions

```python
import pytest
from datetime import datetime, timedelta

def test_progress_calculations():
    """Test pure progress calculation functions."""
    # Create test updates
    now = datetime.now()
    updates = (
        ProgressUpdate(now, ProgressLevel.OPERATION, OperationType.LOADING, 0, 1, "Starting"),
        ProgressUpdate(now + timedelta(seconds=1), ProgressLevel.OPERATION, OperationType.LOADING, 1, 1, "Complete"),
        ProgressUpdate(now + timedelta(seconds=2), ProgressLevel.OPERATION, OperationType.PARSING, 0, 10, "Starting"),
        ProgressUpdate(now + timedelta(seconds=5), ProgressLevel.OPERATION, OperationType.PARSING, 10, 10, "Complete"),
    )

    # Test operation stats (pure function)
    stats = calculate_operation_stats(updates)
    assert OperationType.LOADING in stats
    assert stats[OperationType.LOADING]['mean'] == 1.0
    assert stats[OperationType.PARSING]['mean'] == 3.0

    # Test overall progress (pure function)
    progress = calculate_overall_progress(updates)
    assert progress == 100.0  # Both operations complete

def test_progress_stream():
    """Test functional stream processing."""
    updates = tuple(
        ProgressUpdate(
            datetime.now(),
            ProgressLevel.SHEET if i % 2 == 0 else ProgressLevel.DETAIL,
            OperationType.PARSING,
            i, 10,
            f"Update {i}"
        )
        for i in range(10)
    )

    # Test stream operations
    stream = ProgressStream(iter(updates))
    sheet_updates = (
        stream
        .filter(lambda u: u.level == ProgressLevel.SHEET)
        .map(lambda u: u.current)
        .collect()
    )

    assert sheet_updates == [0, 2, 4, 6, 8]

@pytest.mark.asyncio
async def test_functional_tracker():
    """Test functional progress tracker."""
    tracker = FunctionalProgressTracker()

    # Test immutable state updates
    initial_state = tracker.get_state()
    assert len(initial_state.updates) == 0

    # Emit update
    new_state = await tracker.emit_update(
        ProgressLevel.WORKBOOK,
        OperationType.LOADING,
        1, 1,
        "Loading complete"
    )

    # Verify immutability
    assert len(initial_state.updates) == 0  # Original unchanged
    assert len(new_state.updates) == 1      # New state has update

    # Test analysis (pure function on state)
    analysis = tracker.analyze()
    assert analysis['overall_progress'] > 0
```

## Benefits of Functional Approach

1. **Thread Safety**: Immutable data eliminates race conditions
1. **Testability**: Pure functions are trivial to test in isolation
1. **Debugging**: Can replay event streams and recreate any state
1. **Composability**: Stream operations compose naturally
1. **Predictability**: Pure functions guarantee consistent behavior
1. **Performance**: Immutability enables safe parallelization

## Performance Considerations

1. **Structural Sharing**: Use persistent data structures for large update collections
1. **Lazy Evaluation**: Stream operations are lazy until terminal operation
1. **Batch Updates**: Group rapid updates to reduce overhead
1. **Async I/O**: Broadcasting is async to prevent blocking

This functional approach to progress tracking provides a robust, composable, and testable system for monitoring long-running Excel analysis operations.
