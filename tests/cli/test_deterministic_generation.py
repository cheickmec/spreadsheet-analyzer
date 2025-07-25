"""Tests for deterministic reference notebook generation.

This module generates reference notebooks using the CLI with --no-llm flag.
The generated notebooks are stored in Git for regression testing and serve
as living documentation of the system's deterministic capabilities.

The tests are designed to be:
1. Reproducible - same input always produces same output
2. Cost-free - no LLM API calls
3. Comprehensive - covers all test files and sheets
4. Version-controlled - outputs stored in Git
"""

import pytest
import shutil
from pathlib import Path
from typing import Dict, List, Any
import filecmp
import json

from .conftest import TEST_FILES


class TestDeterministicGeneration:
    """Test class for generating deterministic reference notebooks."""
    
    @pytest.mark.parametrize("filename", TEST_FILES.keys())
    def test_generate_reference_notebook_all_sheets(
        self,
        filename: str,
        cli_runner,
        get_test_file_info,
        reference_notebooks_dir: Path,
        test_data_dir: Path,
        temp_output_dir: Path,
    ):
        """
        Generate reference notebooks for all sheets in a test file.
        
        This test runs the CLI with --no-llm on each test file and generates
        notebooks for all sheets. The outputs are stored as reference notebooks.
        
        Args:
            filename: Test file to process
            cli_runner: Fixture for running CLI commands
            get_test_file_info: Fixture for getting test file information
            reference_notebooks_dir: Directory for reference notebooks
            test_data_dir: Directory containing test data
            temp_output_dir: Temporary directory for outputs
        """
        # Get file information
        file_info = get_test_file_info(filename)
        file_path = file_info['file_path']
        sheets = file_info['actual_sheets']
        
        # Create reference subdirectory for this file
        file_ref_dir = reference_notebooks_dir / file_path.stem
        file_ref_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sheet (or the file itself for CSV)
        for sheet in sheets:
            # Build CLI arguments
            args = [
                str(file_path),
                "--no-llm",  # Deterministic mode
                "--output-dir", str(temp_output_dir),
                "--verbose",
            ]
            
            # Add sheet specification for Excel files
            if sheet is not None:
                args.extend(["--sheet", sheet])
            
            # Run CLI
            result = cli_runner(args)
            
            # Check that CLI succeeded
            assert result.returncode == 0, (
                f"CLI failed for {filename}, sheet '{sheet}': "
                f"stdout={result.stdout}, stderr={result.stderr}"
            )
            
            # Determine expected output path
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                expected_notebook = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
            else:
                # CSV files use the filename as the notebook name
                expected_notebook = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            # Verify notebook was created
            assert expected_notebook.exists(), (
                f"Expected notebook not found: {expected_notebook}. "
                f"CLI output: {result.stdout}"
            )
            
            # Copy to reference location
            reference_notebook = file_ref_dir / expected_notebook.name
            shutil.copy2(expected_notebook, reference_notebook)
            
            print(f"✅ Generated reference notebook: {reference_notebook}")
    
    @pytest.mark.parametrize("filename", TEST_FILES.keys())
    def test_reference_notebook_reproducibility(
        self,
        filename: str,
        cli_runner,
        get_test_file_info,
        reference_notebooks_dir: Path,
        temp_output_dir: Path,
    ):
        """
        Test that running the CLI multiple times produces identical results.
        
        This ensures true deterministic behavior - the same input should always
        produce byte-for-byte identical notebook outputs.
        
        Args:
            filename: Test file to process
            cli_runner: Fixture for running CLI commands  
            get_test_file_info: Fixture for getting test file information
            reference_notebooks_dir: Directory for reference notebooks
            temp_output_dir: Temporary directory for outputs
        """
        # Get file information
        file_info = get_test_file_info(filename)
        file_path = file_info['file_path']
        sheets = file_info['actual_sheets']
        
        # Test each sheet
        for sheet in sheets:
            # Build CLI arguments
            args = [
                str(file_path),
                "--no-llm",
                "--output-dir", str(temp_output_dir),
            ]
            
            if sheet is not None:
                args.extend(["--sheet", sheet])
            
            # Run CLI twice
            result1 = cli_runner(args)
            assert result1.returncode == 0, f"First run failed: {result1.stderr}"
            
            # Determine notebook paths
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                notebook1 = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
            else:
                notebook1 = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            # Move first result
            notebook1_backup = temp_output_dir / f"{notebook1.name}.first"
            shutil.move(notebook1, notebook1_backup)
            
            # Run CLI second time
            result2 = cli_runner(args)
            assert result2.returncode == 0, f"Second run failed: {result2.stderr}"
            
            # Compare results
            assert notebook1.exists(), "Second run didn't create notebook"
            assert filecmp.cmp(notebook1_backup, notebook1, shallow=False), (
                f"Notebook outputs differ between runs for {filename}, sheet '{sheet}'. "
                f"This indicates non-deterministic behavior!"
            )
            
            print(f"✅ Reproducibility verified for {filename}, sheet '{sheet}'")
    
    def test_generate_all_reference_notebooks(
        self,
        cli_runner,
        reference_notebooks_dir: Path,
        test_data_dir: Path,
        temp_output_dir: Path,
    ):
        """
        Generate all reference notebooks in one comprehensive test.
        
        This is a convenience test that generates notebooks for all test files
        and all sheets in a single run. Useful for initial setup or regeneration.
        """
        total_notebooks = 0
        failed_notebooks = []
        
        print(f"\n🏗️  Generating all reference notebooks...")
        print(f"   📁 Test data: {test_data_dir}")
        print(f"   📝 Reference dir: {reference_notebooks_dir}")
        
        for filename, file_info in TEST_FILES.items():
            file_path = test_data_dir / filename
            
            # Skip if test file doesn't exist
            if not file_path.exists():
                print(f"⚠️  Skipping {filename} - file not found")
                continue
            
            # Get actual sheets for this file
            if filename.endswith('.xlsx') and file_info['sheets'][0] is not None:
                try:
                    from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets
                    sheets = list_sheets(file_path)
                except Exception as e:
                    print(f"⚠️  Could not detect sheets for {filename}: {e}")
                    sheets = file_info['sheets']
            else:
                sheets = file_info['sheets']
            
            # Create reference subdirectory
            file_ref_dir = reference_notebooks_dir / file_path.stem
            file_ref_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📊 Processing {filename} ({len(sheets)} sheets)")
            
            for sheet in sheets:
                try:
                    # Build CLI arguments
                    args = [
                        str(file_path),
                        "--no-llm",
                        "--output-dir", str(temp_output_dir),
                        "--verbose",
                    ]
                    
                    if sheet is not None:
                        args.extend(["--sheet", sheet])
                        print(f"   📄 Sheet: {sheet}")
                    else:
                        print(f"   📄 File: {filename}")
                    
                    # Run CLI
                    result = cli_runner(args)
                    
                    if result.returncode != 0:
                        error_msg = f"{filename}/{sheet}: {result.stderr}"
                        failed_notebooks.append(error_msg)
                        print(f"   ❌ Failed: {result.stderr}")
                        continue
                    
                    # Find and copy the generated notebook
                    if sheet is not None:
                        sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                        source_notebook = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
                    else:
                        source_notebook = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
                    
                    if not source_notebook.exists():
                        error_msg = f"{filename}/{sheet}: notebook not created at {source_notebook}"
                        failed_notebooks.append(error_msg)
                        print(f"   ❌ Notebook not found: {source_notebook}")
                        continue
                    
                    # Copy to reference location
                    reference_notebook = file_ref_dir / source_notebook.name
                    shutil.copy2(source_notebook, reference_notebook)
                    
                    total_notebooks += 1
                    print(f"   ✅ Generated: {reference_notebook.name}")
                    
                except Exception as e:
                    error_msg = f"{filename}/{sheet}: unexpected error: {e}"
                    failed_notebooks.append(error_msg)
                    print(f"   ❌ Error: {e}")
        
        # Summary
        print(f"\n📊 Generation Summary:")
        print(f"   ✅ Generated: {total_notebooks} notebooks")
        print(f"   ❌ Failed: {len(failed_notebooks)} notebooks")
        
        if failed_notebooks:
            print(f"\n❌ Failed notebooks:")
            for error in failed_notebooks:
                print(f"   - {error}")
        
        # Save generation metadata
        metadata = {
            'total_generated': total_notebooks,
            'failed_count': len(failed_notebooks),
            'failed_notebooks': failed_notebooks,
            'test_files_processed': list(TEST_FILES.keys()),
        }
        
        metadata_file = reference_notebooks_dir / "generation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📝 Metadata saved to: {metadata_file}")
        
        # The test should succeed even if some notebooks failed,
        # but log the failures for investigation
        if failed_notebooks:
            print(f"\n⚠️  Some notebooks failed to generate. Check logs above.")
        
        assert total_notebooks > 0, "No reference notebooks were generated"
        print(f"\n✅ Reference notebook generation complete!")


@pytest.mark.slow
class TestReferenceNotebookRegression:
    """Test class for regression testing against reference notebooks."""
    
    @pytest.mark.parametrize("filename", TEST_FILES.keys())
    def test_regression_against_reference(
        self,
        filename: str,
        cli_runner,
        get_test_file_info,
        reference_notebooks_dir: Path,
        temp_output_dir: Path,
        validate_notebook,
    ):
        """
        Test that current CLI output matches stored reference notebooks.
        
        This is the key regression test - it ensures that changes to the codebase
        don't break the deterministic output generation.
        
        Args:
            filename: Test file to process
            cli_runner: Fixture for running CLI commands
            get_test_file_info: Fixture for getting test file information  
            reference_notebooks_dir: Directory with reference notebooks
            temp_output_dir: Temporary directory for outputs
            validate_notebook: Fixture for notebook validation
        """
        # Get file information
        file_info = get_test_file_info(filename)
        file_path = file_info['file_path']
        sheets = file_info['actual_sheets']
        
        # Check if reference notebooks exist
        file_ref_dir = reference_notebooks_dir / file_path.stem
        if not file_ref_dir.exists():
            pytest.skip(f"No reference notebooks found for {filename}. Run generation test first.")
        
        for sheet in sheets:
            # Determine reference notebook path
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                ref_notebook = file_ref_dir / f"{sheet_safe}.ipynb"
            else:
                ref_notebook = file_ref_dir / f"{file_path.stem}.ipynb"
            
            if not ref_notebook.exists():
                pytest.skip(f"No reference notebook for {filename}/{sheet}")
            
            # Run CLI to generate current output
            args = [
                str(file_path),
                "--no-llm",
                "--output-dir", str(temp_output_dir),
            ]
            
            if sheet is not None:
                args.extend(["--sheet", sheet])
            
            result = cli_runner(args)
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Find current output
            if sheet is not None:
                sheet_safe = sheet.replace(" ", "_").replace("/", "_")
                current_notebook = temp_output_dir / file_path.stem / f"{sheet_safe}.ipynb"
            else:
                current_notebook = temp_output_dir / file_path.stem / f"{file_path.stem}.ipynb"
            
            assert current_notebook.exists(), f"Current notebook not found: {current_notebook}"
            
            # Validate both notebooks
            ref_validation = validate_notebook(ref_notebook)
            current_validation = validate_notebook(current_notebook)
            
            assert ref_validation['valid'], f"Reference notebook is invalid: {ref_validation['issues']}"
            assert current_validation['valid'], f"Current notebook is invalid: {current_validation['issues']}"
            
            # Compare notebooks byte-for-byte
            if not filecmp.cmp(ref_notebook, current_notebook, shallow=False):
                # Notebooks differ - this is a regression
                pytest.fail(
                    f"Current output differs from reference for {filename}/{sheet}.\n"
                    f"Reference: {ref_notebook}\n"
                    f"Current: {current_notebook}\n"
                    f"This indicates a regression in deterministic output generation."
                )
            
            print(f"✅ Regression test passed for {filename}/{sheet}") 