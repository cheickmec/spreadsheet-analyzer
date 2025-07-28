# Test Reorganization Plan

## Overview

This document outlines a plan for reorganizing the test files currently located at the root of the repository. The goal is to properly organize these files as unit tests or integration tests in the appropriate locations within the project structure.

## Current State

The repository currently has numerous test and debug files at the root level. These files test various aspects of the project but are not organized in a systematic way. This makes it difficult to understand what's being tested and to maintain the tests over time.

## Categorization of Test Files

Based on analysis of the test files, they can be categorized as follows:

1. **Core Execution Tests**

   - Files: `test_kernel_*.py`, `test_bridge_*.py`, `test_execution_*.py`, etc.
   - Purpose: Test the core execution functionality, including kernel services, execution bridges, and notebook builders.
   - Target Location: `tests/core_exec/`

1. **Workflow Tests**

   - Files: `test_workflow_*.py`, `test_simple_workflow.py`, etc.
   - Purpose: Test the workflow functionality, including notebook workflows and execution modes.
   - Target Location: `tests/workflows/`

1. **Pipeline Tests**

   - Files: `test_pipeline_*.py`, `test_stage_*.py`, etc.
   - Purpose: Test the pipeline functionality, including stages and deterministic pipelines.
   - Target Location: `tests/pipeline/`

1. **CLI Tests**

   - Files: `test_cli_*.py`, etc.
   - Purpose: Test the command-line interface functionality.
   - Target Location: `tests/cli/`

1. **Integration Tests**

   - Files: `test_full_*.py`, `test_langchain_integration.py`, etc.
   - Purpose: Test the integration of multiple components.
   - Target Location: `tests/integration/`

1. **Plugin Tests**

   - Files: `test_formula_error_detection.py`, etc.
   - Purpose: Test plugin functionality, such as spreadsheet analysis.
   - Target Location: `tests/plugins/spreadsheet/`

1. **Miscellaneous Tests**

   - Files: `test_shared_utils.py`, etc.
   - Purpose: Test utility functions and other miscellaneous functionality.
   - Target Location: `tests/`

## Reorganization Pattern

For each test file, follow this pattern to reorganize it:

1. **Analyze the Test File**

   - Determine what the file is testing by examining its imports and functionality.
   - Identify the appropriate category and target location.

1. **Create a Proper Test File**

   - Create a new file in the target location with a name that reflects what it's testing.
   - Refactor the test to follow proper testing conventions:
     - Use pytest fixtures and markers.
     - Organize tests into classes.
     - Add clear docstrings.
     - Follow the setup, execution, assertion pattern.

1. **Verify the Test Works**

   - Run the test to ensure it functions correctly.
   - Fix any issues that arise.

1. **Remove the Original File**

   - Once the test is working in its new location, remove the original file from the root directory.

## Example Refactorings

### Example 1: Kernel Profile Tests

Original file: `test_kernel_direct.py`
New file: `tests/core_exec/test_kernel_profiles.py`

The original file tested kernel execution with different profiles. The refactored file organizes these tests into a class with proper fixtures and test methods.

### Example 2: Bridge Output Tests

Original file: `test_bridge_outputs.py`
New file: `tests/core_exec/test_bridge_outputs.py`

This file tests the ExecutionBridge's ability to attach outputs to notebook cells. It should be refactored to follow the same pattern as the kernel profile tests.

### Example 3: Workflow Execution Tests

Original file: `test_workflow_execution.py`
New file: `tests/workflows/test_workflow_execution.py`

This file tests the NotebookWorkflow's execution functionality. It should be refactored to follow the same pattern as the other tests.

### Example 4: Pipeline Tests

Original file: `test_pipeline_direct.py`
New file: `tests/pipeline/test_pipeline_direct.py`

This file tests the DeterministicPipeline's functionality. It should be refactored to follow the same pattern as the other tests.

## Implementation Plan

1. **Start with Core Execution Tests**

   - These are the most numerous and critical tests.
   - Begin with kernel-related tests, then move to bridge-related tests.

1. **Move to Workflow Tests**

   - These tests build on the core execution functionality.

1. **Handle Pipeline Tests**

   - These tests are relatively independent of the other tests.

1. **Address CLI and Integration Tests**

   - These tests often depend on multiple components.

1. **Finish with Plugin and Miscellaneous Tests**

   - These tests are often more specialized and may require more careful refactoring.

## Conclusion

By following this plan, the test files will be properly organized in the appropriate locations within the project structure. This will make it easier to understand what's being tested, maintain the tests over time, and ensure that the project has good test coverage.
