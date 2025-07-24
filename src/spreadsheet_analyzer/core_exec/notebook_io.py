"""
Generic notebook I/O operations.

This module provides domain-agnostic notebook file operations:
- Reading and writing notebook files  
- Format conversion and validation
- Error handling and recovery
- Path management and safety checks

No domain-specific logic - pure notebook I/O primitives.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import nbformat
from nbformat.validator import ValidationError

from .notebook_builder import NotebookBuilder, NotebookCell, CellType


class NotebookFormatError(Exception):
    """Raised when notebook format is invalid."""
    pass


class NotebookIO:
    """
    Generic notebook I/O operations.
    
    Handles reading, writing, and format conversion for Jupyter notebooks
    without any domain-specific assumptions. Provides safe file operations
    and format validation.
    
    Usage:
        io = NotebookIO()
        builder = io.read_notebook("analysis.ipynb")
        builder.add_markdown_cell("# New section")
        io.write_notebook(builder, "updated.ipynb")
    """

    @staticmethod
    def read_notebook(file_path: Union[str, Path]) -> NotebookBuilder:
        """
        Read a notebook file and convert to NotebookBuilder.
        
        Args:
            file_path: Path to the notebook file
            
        Returns:
            NotebookBuilder instance with notebook content
            
        Raises:
            FileNotFoundError: If notebook file doesn't exist
            NotebookFormatError: If notebook format is invalid
            PermissionError: If file cannot be read
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Notebook file not found: {file_path}")
            
        if not file_path.suffix.lower() == '.ipynb':
            raise NotebookFormatError(f"File must have .ipynb extension: {file_path}")

        try:
            # Read and validate the notebook
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook_dict = json.load(f)
                
            # Validate using nbformat
            try:
                nbformat.validate(notebook_dict)
            except ValidationError as e:
                raise NotebookFormatError(f"Invalid notebook format: {e}") from e
                
            # Extract kernel information
            kernelspec = notebook_dict.get("metadata", {}).get("kernelspec", {})
            kernel_name = kernelspec.get("name", "python3")
            kernel_display_name = kernelspec.get("display_name", "Python 3")
            
            # Create builder
            builder = NotebookBuilder(kernel_name, kernel_display_name)
            
            # Convert cells
            for cell_dict in notebook_dict.get("cells", []):
                cell = NotebookIO._dict_to_cell(cell_dict)
                if cell:
                    builder.add_cell(cell)
                    
            return builder
            
        except json.JSONDecodeError as e:
            raise NotebookFormatError(f"Invalid JSON in notebook file: {e}") from e
        except Exception as e:
            raise NotebookFormatError(f"Error reading notebook: {e}") from e

    @staticmethod
    def write_notebook(
        builder: NotebookBuilder, 
        file_path: Union[str, Path],
        overwrite: bool = False
    ) -> Path:
        """
        Write a NotebookBuilder to a file.
        
        Args:
            builder: NotebookBuilder instance to write
            file_path: Path where to write the notebook
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path object of the written file
            
        Raises:
            FileExistsError: If file exists and overwrite=False
            PermissionError: If file cannot be written
            NotebookFormatError: If notebook content is invalid
        """
        file_path = Path(file_path)
        
        # Ensure .ipynb extension
        if not file_path.suffix.lower() == '.ipynb':
            file_path = file_path.with_suffix('.ipynb')
            
        # Check overwrite policy
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {file_path}")
            
        # Validate notebook structure
        validation_errors = builder.validate()
        if validation_errors:
            raise NotebookFormatError(f"Invalid notebook structure: {validation_errors}")
            
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and validate
            notebook_dict = builder.to_dict()
            
            try:
                nbformat.validate(notebook_dict)
            except ValidationError as e:
                raise NotebookFormatError(f"Generated notebook is invalid: {e}") from e
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
                
            return file_path
            
        except Exception as e:
            raise NotebookFormatError(f"Error writing notebook: {e}") from e

    @staticmethod
    def convert_outputs_to_nbformat(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert raw kernel outputs to proper nbformat output objects.
        
        This method handles the conversion of kernel execution results
        to the proper nbformat structure, ensuring compatibility.
        
        Args:
            outputs: List of raw output dictionaries from kernel execution
            
        Returns:
            List of properly formatted nbformat output objects
        """
        formatted_outputs = []
        
        for output in outputs:
            if not isinstance(output, dict):
                continue
                
            output_type = output.get("output_type") or output.get("type")
            if not output_type:
                continue
                
            try:
                if output_type == "stream":
                    formatted_output = {
                        "output_type": "stream",
                        "name": output.get("name", "stdout"),
                        "text": NotebookIO._ensure_text_list(output.get("text", ""))
                    }
                    
                elif output_type == "execute_result":
                    formatted_output = {
                        "output_type": "execute_result",
                        "execution_count": output.get("execution_count", 1),
                        "data": output.get("data", {}),
                        "metadata": output.get("metadata", {})
                    }
                    
                elif output_type == "display_data":
                    formatted_output = {
                        "output_type": "display_data",
                        "data": output.get("data", {}),
                        "metadata": output.get("metadata", {})
                    }
                    
                elif output_type == "error":
                    formatted_output = {
                        "output_type": "error",
                        "ename": output.get("ename", "Error"),
                        "evalue": output.get("evalue", "Unknown error"),
                        "traceback": output.get("traceback", [])
                    }
                    
                else:
                    # Skip unknown output types
                    continue
                    
                formatted_outputs.append(formatted_output)
                
            except Exception:
                # Skip malformed outputs rather than failing completely
                continue
                
        return formatted_outputs

    @staticmethod
    def _dict_to_cell(cell_dict: Dict[str, Any]) -> Optional[NotebookCell]:
        """
        Convert a dictionary representation to a NotebookCell.
        
        Args:
            cell_dict: Dictionary representation of a cell
            
        Returns:
            NotebookCell instance or None if conversion fails
        """
        try:
            cell_type_str = cell_dict.get("cell_type")
            if not cell_type_str:
                return None
                
            # Map string to enum
            cell_type_map = {
                "code": CellType.CODE,
                "markdown": CellType.MARKDOWN,
                "raw": CellType.RAW
            }
            
            cell_type = cell_type_map.get(cell_type_str)
            if not cell_type:
                return None
                
            # Get source (handle both string and list formats)
            source = cell_dict.get("source", [])
            if isinstance(source, str):
                source = source.split('\n')
                # Convert to proper format (add newlines except last line)
                formatted_source = []
                for i, line in enumerate(source):
                    if i < len(source) - 1:
                        formatted_source.append(line + '\n')
                    else:
                        formatted_source.append(line)
                source = formatted_source
            elif isinstance(source, list):
                source = list(source)  # Make a copy
            else:
                source = [""]
                
            # Extract metadata
            metadata = cell_dict.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                
            # Handle code cell specifics
            outputs = None
            execution_count = None
            
            if cell_type == CellType.CODE:
                outputs = cell_dict.get("outputs", [])
                if not isinstance(outputs, list):
                    outputs = []
                    
                execution_count = cell_dict.get("execution_count")
                if execution_count is not None:
                    try:
                        execution_count = int(execution_count)
                    except (ValueError, TypeError):
                        execution_count = None
                        
            return NotebookCell(
                cell_type=cell_type,
                source=source,
                metadata=metadata,
                outputs=outputs,
                execution_count=execution_count
            )
            
        except Exception:
            # Return None for malformed cells rather than crashing
            return None

    @staticmethod
    def _ensure_text_list(text: Union[str, List[str]]) -> List[str]:
        """
        Ensure text output is in list format as required by nbformat.
        
        Args:
            text: Text content as string or list
            
        Returns:
            Text as list of strings
        """
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            return [str(item) for item in text]
        else:
            return [str(text)]

    @staticmethod
    def validate_notebook_file(file_path: Union[str, Path]) -> List[str]:
        """
        Validate a notebook file and return any issues found.
        
        Args:
            file_path: Path to notebook file to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        file_path = Path(file_path)
        
        try:
            # Check file exists and is readable
            if not file_path.exists():
                issues.append(f"File does not exist: {file_path}")
                return issues
                
            if not file_path.is_file():
                issues.append(f"Path is not a file: {file_path}")
                return issues
                
            if not file_path.suffix.lower() == '.ipynb':
                issues.append(f"File should have .ipynb extension: {file_path}")
                
            # Try to read and validate
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook_dict = json.load(f)
                
            # Validate format
            try:
                nbformat.validate(notebook_dict)
            except ValidationError as e:
                issues.append(f"Invalid nbformat: {e}")
                
            # Check basic structure
            if "cells" not in notebook_dict:
                issues.append("Missing 'cells' key")
            elif not isinstance(notebook_dict["cells"], list):
                issues.append("'cells' should be a list")
                
            if "metadata" not in notebook_dict:
                issues.append("Missing 'metadata' key")
                
            if "nbformat" not in notebook_dict:
                issues.append("Missing 'nbformat' key")
                
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")
        except Exception as e:
            issues.append(f"Error reading file: {e}")
            
        return issues

    @staticmethod
    def create_empty_notebook(
        kernel_name: str = "python3",
        kernel_display_name: str = "Python 3"
    ) -> NotebookBuilder:
        """
        Create an empty notebook with proper metadata.
        
        Args:
            kernel_name: Kernel spec name
            kernel_display_name: Human-readable kernel name
            
        Returns:
            Empty NotebookBuilder instance
        """
        return NotebookBuilder(kernel_name, kernel_display_name)

    @staticmethod
    def merge_notebooks(*builders: NotebookBuilder) -> NotebookBuilder:
        """
        Merge multiple notebooks into a single notebook.
        
        Args:
            *builders: NotebookBuilder instances to merge
            
        Returns:
            New NotebookBuilder with all cells from input notebooks
        """
        if not builders:
            return NotebookIO.create_empty_notebook()
            
        # Use first notebook's kernel settings
        first = builders[0]
        merged = NotebookBuilder(first.kernel_name, first.kernel_display_name)
        
        # Add all cells from all notebooks
        for builder in builders:
            for cell in builder.cells:
                merged.add_cell(cell)
                
        return merged 