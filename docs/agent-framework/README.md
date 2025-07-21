# Agent Framework Documentation

This directory contains documentation for the multi-agent framework used in the Spreadsheet Analyzer system.

## Overview

The agent framework provides intelligent analysis capabilities through specialized agents that work together to understand complex spreadsheets. Each agent runs in an isolated Jupyter kernel with its own execution environment and state.

## Architecture Components

### 1. [Kernel Manager](./kernel-manager.md)

Manages isolated Jupyter kernels for agent code execution with resource limits and session persistence.

### 2. LangGraph Orchestrator (Coming Soon)

Coordinates multiple agents using LangGraph's state machine capabilities.

### 3. Base Agent Class (Coming Soon)

Provides common functionality for all analysis agents.

### 4. Specialized Agents (Coming Soon)

- **Formula Analyzer Agent**: Analyzes Excel formulas and dependencies
- **Pattern Detection Agent**: Identifies data patterns and anomalies
- **Chart Reader Agent**: Interprets visualizations and charts
- **Summary Agent**: Synthesizes findings from other agents

## Design Principles

1. **Isolation**: Each agent runs in its own kernel for security and stability
1. **Persistence**: Agent state is maintained across interactions
1. **Resource Control**: Strict limits on CPU, memory, and execution time
1. **Asynchronous**: All operations are async for better concurrency
1. **Extensible**: Easy to add new agent types and capabilities

## Getting Started

See the [kernel manager documentation](./kernel-manager.md) for the foundation of the agent framework.

## Related Documentation

- [Comprehensive System Design](../design/comprehensive-system-design.md)
- [Notebook-LLM Interface](../design/notebook-llm-interface.md)
- [Implementation Status](../implementation-status.md)
