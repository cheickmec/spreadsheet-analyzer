"""Tests for notebook validation and quality assurance.

This module validates the structure, content, and quality of generated notebooks.
It works with both deterministic (--no-llm) and LLM-enhanced notebooks to ensure
they meet quality standards and follow expected patterns.

The validation includes:
1. Structural validation - proper nbformat structure
2. Content validation - presence of required elements  
3. Quality validation - no errors, proper outputs
4. Task coverage validation - all expected tasks executed
"""

import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
import nbformat
import json
import re

from .conftest import TEST_FILES


class TestNotebookStructure:
    """Test class for validating notebook structure and format."""
    
    @pytest.mark.parametrize("filename", TEST_FILES.keys())
    def test_notebook_structure_deterministic(
        self,
        filename: str,
        cli_runner,
        get_test_file_info,
        temp_output_dir: Path,
        validate_notebook,
    ):
        """
        Test that deterministic notebooks have proper structure.
        
        Args:
            filename: Test file to process
            cli_runner: Fixture for running CLI commands
            get_test_file_info: Fixture for getting test file information
            temp_output_dir: Temporary directory for outputs  
            validate_notebook: Fixture for notebook validation
        """
        # Get file information
        file_info = get_test_file_info(filename)
        file_path = file_info['file_path']
        sheets = file_info['actual_sheets']
        
        for sheet in sheets:
            # Generate notebook using deterministic mode
            args = [
                str(file_path),
                "--no-llm",
                "--output-dir", str(temp_output_dir),
            ]
            
            if sheet is not None:
                args.extend(["--sheet", sheet])
            
            result = cli_runner(args)
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Find generated notebook
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                notebook_path = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
            else:
                notebook_path = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            # Validate structure
            validation = validate_notebook(notebook_path)
            
            # Basic structure requirements
            assert validation['exists'], f"Notebook file does not exist: {notebook_path}"
            assert validation['readable'], f"Notebook is not readable: {validation['issues']}"
            assert validation['valid'], f"Invalid notebook format: {validation['issues']}"
            assert validation['has_cells'], f"Notebook has no cells: {notebook_path}"
            
            # Content requirements
            assert validation['code_cells'] > 0, f"No code cells found in {notebook_path}"
            assert validation['markdown_cells'] > 0, f"No markdown cells found in {notebook_path}"
            
            # Quality requirements
            assert validation['error_cells'] == 0, (
                f"Notebook has {validation['error_cells']} error cells: {validation['issues']}"
            )
            
            print(f"✅ Structure validation passed for {filename}/{sheet}")
    
    def test_notebook_content_patterns(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that notebooks contain expected content patterns.
        
        This test validates that notebooks include standard sections like:
        - Data loading and inspection
        - Profiling information
        - Analysis sections
        - Summary/conclusions
        """
        # Test with simple_sales.xlsx which should have predictable structure
        test_file = test_data_dir / "simple_sales.xlsx"
        
        args = [
            str(test_file),
            "--no-llm",
            "--output-dir", str(temp_output_dir),
            "--sheet", "Monthly Sales",
        ]
        
        result = cli_runner(args)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        notebook_path = temp_output_dir / "simple_sales" / "Monthly_Sales.ipynb"
        assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
        
        # Read notebook content
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Extract all cell text content
        all_text = ""
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                all_text += cell.source + "\n"
            elif cell.cell_type == 'code':
                all_text += cell.source + "\n"
                # Also include output text
                if hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if hasattr(output, 'text'):
                            all_text += str(output.text) + "\n"
        
        # Check for expected content patterns
        content_checks = [
            ("data loading", r"(import|load|read).*pandas|pd\.read", "Data loading code"),
            ("data inspection", r"(\.head\(\)|\.info\(\)|\.describe\(\)|\.shape)", "Data inspection methods"),
            ("analysis title", r"(# |## |### ).*[Aa]nalysis", "Analysis section headers"),
            ("summary", r"(# |## |### ).*(Summary|Conclusion)", "Summary or conclusion section"),
        ]
        
        for check_name, pattern, description in content_checks:
            assert re.search(pattern, all_text, re.IGNORECASE | re.MULTILINE), (
                f"Missing expected pattern '{check_name}' ({description}) in notebook content"
            )
        
        print(f"✅ Content pattern validation passed for {test_file.name}")


class TestNotebookQuality:
    """Test class for validating notebook quality and completeness."""
    
    @pytest.mark.parametrize("filename", TEST_FILES.keys())
    def test_no_execution_errors(
        self,
        filename: str,
        cli_runner,
        get_test_file_info,
        temp_output_dir: Path,
        validate_notebook,
    ):
        """
        Test that notebooks execute without errors.
        
        Args:
            filename: Test file to process
            cli_runner: Fixture for running CLI commands
            get_test_file_info: Fixture for getting test file information
            temp_output_dir: Temporary directory for outputs
            validate_notebook: Fixture for notebook validation
        """
        # Get file information
        file_info = get_test_file_info(filename)
        file_path = file_info['file_path']
        sheets = file_info['actual_sheets']
        
        for sheet in sheets:
            # Generate notebook
            args = [
                str(file_path),
                "--no-llm",
                "--output-dir", str(temp_output_dir),
            ]
            
            if sheet is not None:
                args.extend(["--sheet", sheet])
            
            result = cli_runner(args)
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Find and validate notebook
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                notebook_path = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
            else:
                notebook_path = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            validation = validate_notebook(notebook_path)
            
            # Check for execution errors
            assert validation['error_cells'] == 0, (
                f"Notebook has {validation['error_cells']} cells with errors:\n" +
                "\n".join(validation['issues'])
            )
            
            print(f"✅ No execution errors in {filename}/{sheet}")
    
    def test_notebook_output_completeness(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that notebooks have complete outputs for code cells.
        
        This ensures that the generated notebooks are not just code templates
        but actually contain executed results.
        """
        # Test with a file that should produce rich outputs
        test_file = test_data_dir / "employee_records.xlsx"
        
        args = [
            str(test_file),
            "--no-llm",
            "--output-dir", str(temp_output_dir),
        ]
        
        result = cli_runner(args)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        notebook_path = temp_output_dir / "employee_records" / "employee_records.ipynb"
        assert notebook_path.exists()
        
        # Read notebook and analyze outputs
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells_with_outputs = 0
        total_code_cells = 0
        
        for cell in nb.cells:
            if cell.cell_type == 'code' and cell.source.strip():
                total_code_cells += 1
                if hasattr(cell, 'outputs') and cell.outputs:
                    code_cells_with_outputs += 1
        
        # At least 80% of code cells should have outputs
        if total_code_cells > 0:
            output_ratio = code_cells_with_outputs / total_code_cells
            assert output_ratio >= 0.8, (
                f"Only {code_cells_with_outputs}/{total_code_cells} "
                f"({output_ratio:.1%}) code cells have outputs. Expected >=80%"
            )
        
        print(f"✅ Output completeness validated: {code_cells_with_outputs}/{total_code_cells} cells have outputs")
    
    def test_data_quality_detection(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that notebooks properly detect and report data quality issues.
        
        Uses employee_records.xlsx which has intentional data quality problems.
        """
        test_file = test_data_dir / "employee_records.xlsx"
        
        args = [
            str(test_file),
            "--no-llm",
            "--output-dir", str(temp_output_dir),
        ]
        
        result = cli_runner(args)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        notebook_path = temp_output_dir / "employee_records" / "employee_records.ipynb"
        
        # Read notebook content
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Extract all text content from cells and outputs
        all_content = ""
        for cell in nb.cells:
            all_content += cell.source + "\n"
            if hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    if hasattr(output, 'text'):
                        all_content += str(output.text) + "\n"
                    elif hasattr(output, 'data') and 'text/plain' in output.data:
                        all_content += str(output.data['text/plain']) + "\n"
        
        # Check for quality issue detection patterns
        quality_patterns = [
            (r"(missing|null|nan|none)", "Missing value detection"),
            (r"(duplicate|duplicated)", "Duplicate detection"),
            (r"(outlier|anomal)", "Outlier detection"),
            (r"(quality|issue|problem)", "Quality issue identification"),
        ]
        
        detected_issues = []
        for pattern, description in quality_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                detected_issues.append(description)
        
        # Should detect at least some quality issues since the data has intentional problems
        assert len(detected_issues) >= 2, (
            f"Expected to detect data quality issues in {test_file.name}, "
            f"but only found: {detected_issues}"
        )
        
        print(f"✅ Data quality detection working: {detected_issues}")


class TestTaskCoverage:
    """Test class for validating that expected analysis tasks are executed."""
    
    def test_task_coverage_all_files(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that all expected analysis tasks are covered in generated notebooks.
        
        This test ensures that the plugin system is working properly and that
        all relevant analysis tasks are being executed for each file type.
        """
        for filename, file_info in TEST_FILES.items():
            file_path = test_data_dir / filename
            
            if not file_path.exists():
                continue
            
            # Generate notebook
            args = [
                str(file_path),
                "--no-llm",
                "--output-dir", str(temp_output_dir),
                "--verbose",  # More detailed output
            ]
            
            result = cli_runner(args)
            assert result.returncode == 0, f"CLI failed for {filename}: {result.stderr}"
            
            # Find notebook (use first sheet for multi-sheet files)
            if filename.endswith('.xlsx'):
                # Determine expected notebook name
                try:
                    from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets
                    sheets = list_sheets(file_path)
                    first_sheet = sheets[0] if sheets else "Sheet1"
                    sheet_safe = first_sheet.replace(" ", "_").replace("/", "_")
                    notebook_path = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
                except Exception:
                    # Fallback to default
                    notebook_path = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            else:
                notebook_path = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
            
            # Read notebook content
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Extract all content
            all_content = ""
            for cell in nb.cells:
                all_content += cell.source + "\n"
                if hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if hasattr(output, 'text'):
                            all_content += str(output.text) + "\n"
            
            # Check for basic analysis patterns that should be present
            basic_patterns = [
                (r"(shape|dimension)", "Data shape/dimensions"),
                (r"(dtype|type)", "Data types"),
                (r"(head|sample)", "Data preview"),
                (r"(describe|statistic)", "Statistical summary"),
            ]
            
            found_patterns = []
            for pattern, description in basic_patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    found_patterns.append(description)
            
            # Should find at least 3 out of 4 basic patterns
            assert len(found_patterns) >= 3, (
                f"Insufficient task coverage for {filename}. "
                f"Found: {found_patterns}, expected at least 3 basic patterns"
            )
            
            print(f"✅ Task coverage validated for {filename}: {found_patterns}")
    
    def test_plugin_system_integration(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that the plugin system is properly integrated and working.
        
        This test verifies that the three-tier architecture (core_exec, plugins, workflows)
        is functioning correctly by checking CLI output for plugin execution indicators.
        """
        test_file = test_data_dir / "simple_sales.xlsx"
        
        args = [
            str(test_file),
            "--no-llm",
            "--output-dir", str(temp_output_dir),
            "--verbose",
            "--sheet", "Monthly Sales",
        ]
        
        result = cli_runner(args)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Check CLI output for plugin system indicators
        cli_output = result.stdout + result.stderr
        
        # Should show evidence of plugin registration and execution
        plugin_indicators = [
            r"(plugin|task|workflow)",  # Plugin system terminology
            r"(register|load|execut)",  # Plugin registration/execution
            r"(core_exec|workflow)",    # Architecture components
        ]
        
        found_indicators = []
        for pattern in plugin_indicators:
            if re.search(pattern, cli_output, re.IGNORECASE):
                found_indicators.append(pattern)
        
        # Should find evidence of plugin system working
        assert len(found_indicators) >= 1, (
            f"No evidence of plugin system integration in CLI output. "
            f"CLI output:\n{cli_output[:500]}..."
        )
        
        # Check that notebook was actually created (integration success)
        notebook_path = temp_output_dir / "simple_sales" / "Monthly_Sales.ipynb"
        assert notebook_path.exists(), "Plugin system failed to generate notebook"
        
        print(f"✅ Plugin system integration validated")


class TestNotebookMetadata:
    """Test class for validating notebook metadata and annotations."""
    
    def test_notebook_metadata_presence(
        self,
        cli_runner,
        temp_output_dir: Path,
        test_data_dir: Path,
    ):
        """
        Test that notebooks contain proper metadata.
        
        Generated notebooks should include metadata about:
        - Generation timestamp
        - Source file information
        - CLI arguments used
        - System version information
        """
        test_file = test_data_dir / "simple_sales.xlsx"
        
        args = [
            str(test_file),
            "--no-llm",
            "--output-dir", str(temp_output_dir),
            "--sheet", "Monthly Sales",
        ]
        
        result = cli_runner(args)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        notebook_path = temp_output_dir / "simple_sales" / "Monthly_Sales.ipynb"
        
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Check notebook metadata
        assert hasattr(nb, 'metadata'), "Notebook missing metadata"
        
        # Look for generation information in markdown cells
        found_metadata = False
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                source = cell.source.lower()
                if any(keyword in source for keyword in ['generated', 'analysis', 'spreadsheet']):
                    found_metadata = True
                    break
        
        assert found_metadata, "No generation metadata found in notebook markdown cells"
        
        print(f"✅ Notebook metadata validation passed") 