# Jupyter Kernel Manager

The `AgentKernelManager` provides isolated Python execution environments for agents in the spreadsheet analyzer system. It manages a pool of Jupyter kernels with resource limits, session persistence, and secure execution.

## Features

- **Kernel Pooling**: Manages a configurable pool of kernels (default: 10) for efficient resource usage
- **Session Persistence**: Maintains execution history and state for each agent across multiple interactions
- **Resource Limits**: Enforces CPU, memory, and execution time limits
- **Async Context Manager**: Safe resource acquisition and release with automatic cleanup
- **Graceful Shutdown**: Properly terminates all kernels on shutdown

## Basic Usage

```python
from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager

# Initialize the kernel manager
manager = AgentKernelManager(
    max_kernels=5,
    kernel_name="python3"
)

# Use as async context manager
async with manager:
    # Acquire a kernel for an agent
    async with manager.acquire_kernel("agent-1") as (kernel_manager, session):
        # Execute code
        result = await manager.execute_code(
            session,
            "x = 10 + 20\nprint(f'Result: {x}')"
        )
        
        # Result includes status, outputs, and any errors
        print(result["status"])  # "ok"
        print(result["outputs"])  # [{"type": "stream", "text": "Result: 30\n"}]
```

## Resource Limits

Configure resource limits to prevent runaway code:

```python
from spreadsheet_analyzer.agents.kernel_manager import KernelResourceLimits

limits = KernelResourceLimits(
    max_cpu_percent=50.0,      # Max 50% CPU usage
    max_memory_mb=2048,        # Max 2GB memory
    max_execution_time=60.0,   # Max 60 seconds execution
    max_output_size_mb=20      # Max 20MB output
)

manager = AgentKernelManager(
    max_kernels=5,
    resource_limits=limits
)
```

## Session Persistence

Each agent maintains its own session with execution history:

```python
# First execution
async with manager.acquire_kernel("agent-1") as (km, session):
    await manager.execute_code(session, "data = [1, 2, 3, 4, 5]")
    await manager.execute_code(session, "total = sum(data)")

# Later execution - state is preserved
async with manager.acquire_kernel("agent-1") as (km, session):
    result = await manager.execute_code(session, "print(total)")
    # Output: "15"
```

## Checkpointing

Save and restore session state:

```python
# Save checkpoint
checkpoint = manager.save_checkpoint(session)

# Later, restore to a new session
new_session = KernelSession(
    session_id="new-session",
    kernel_id="kernel-1", 
    agent_id="agent-1"
)
manager.restore_checkpoint(new_session, checkpoint)
```

## Error Handling

The manager handles various error conditions:

```python
from spreadsheet_analyzer.agents.kernel_manager import (
    KernelTimeoutError,
    KernelPoolExhaustedError
)

try:
    async with manager.acquire_kernel("agent-1", timeout=5.0) as (km, session):
        # Execute code with timeout
        result = await manager.execute_code(
            session,
            "import time; time.sleep(100)"  # Will timeout
        )
except KernelTimeoutError:
    print("Execution exceeded time limit")
except KernelPoolExhaustedError:
    print("No kernels available")
```

## Architecture

The kernel manager consists of several components:

1. **KernelResource**: Individual kernel with availability tracking
1. **KernelPool**: Manages pool of kernels with async acquisition/release
1. **KernelSession**: Tracks agent session state and execution history
1. **AgentKernelManager**: Main orchestrator providing the high-level API

## Security Considerations

> **Note**: The current implementation provides basic isolation through separate Jupyter kernels. For production use, additional security measures should be implemented:

- Use gVisor or similar container runtime for stronger isolation
- Implement network isolation to prevent external access
- Add filesystem sandboxing to restrict file access
- Monitor resource usage in real-time
- Implement audit logging for all executions

## Future Enhancements

Planned improvements include:

- [ ] Real-time resource monitoring with psutil
- [ ] Security sandboxing with gVisor integration
- [ ] Distributed kernel management across multiple machines
- [ ] Kernel warm-up and pre-initialization
- [ ] Advanced scheduling algorithms for kernel allocation
- [ ] Integration with Notebook Agent Protocol (NAP)

## Example: Multi-Agent Analysis

```python
async def analyze_spreadsheet_with_agents():
    """Example of multiple agents analyzing different aspects."""
    
    async with AgentKernelManager(max_kernels=3) as manager:
        # Formula analyzer agent
        async with manager.acquire_kernel("formula-agent") as (km, session):
            await manager.execute_code(session, """
                import openpyxl
                # Analyze formulas...
            """)
        
        # Data pattern agent
        async with manager.acquire_kernel("pattern-agent") as (km, session):
            await manager.execute_code(session, """
                import pandas as pd
                # Detect patterns...
            """)
        
        # Visualization agent
        async with manager.acquire_kernel("viz-agent") as (km, session):
            await manager.execute_code(session, """
                import matplotlib.pyplot as plt
                # Create visualizations...
            """)
```

## Testing

The kernel manager includes comprehensive tests demonstrating usage patterns:

```bash
# Run all kernel manager tests
uv run pytest tests/test_kernel_manager.py -v

# Run integration tests with real kernels
uv run pytest tests/test_kernel_manager.py -v -m integration
```

## Performance Considerations

- Kernel startup time: ~1-2 seconds per kernel
- Memory overhead: ~50-100MB per kernel
- Recommended pool size: 1-2x number of concurrent agents
- Use kernel reuse to minimize startup overhead

______________________________________________________________________

For more details, see the [comprehensive system design](../design/comprehensive-system-design.md) and [agent framework documentation](./README.md).
