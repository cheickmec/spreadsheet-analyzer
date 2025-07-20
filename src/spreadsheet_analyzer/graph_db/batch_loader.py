#!/usr/bin/env python3
"""
Batch loader for processing all spreadsheets and loading them into Neo4j.

This script finds all Excel files in the repository, analyzes them,
and loads the dependency graphs into Neo4j.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from spreadsheet_analyzer.pipeline.stages.stage_3_formulas import stage_3_formula_analysis
from spreadsheet_analyzer.pipeline.types import PipelineContext

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the batch loader."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/app/logs/batch_loader.log"),
        ],
    )


def wait_for_neo4j(max_retries: int = 30, delay: int = 2) -> bool:
    """
    Wait for Neo4j to be ready.

    Args:
        max_retries: Maximum number of connection attempts
        delay: Seconds between attempts

    Returns:
        True if connection successful, False otherwise
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.exception("neo4j package not installed. Please install with: pip install neo4j")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    logger.info(f"Waiting for Neo4j at {uri}...")

    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            driver.close()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(delay)
            else:
                logger.exception(f"Failed to connect to Neo4j after {max_retries} attempts")
                return False
    else:
        return True


def find_spreadsheet_files(base_dir: Path) -> list[Path]:
    """
    Find all Excel files in the repository.

    Args:
        base_dir: Base directory to search

    Returns:
        List of Excel file paths
    """
    excel_extensions = {".xlsx", ".xlsm", ".xls"}
    excel_files = []

    # Common test/sample directories
    search_dirs = [
        base_dir / "test-files",
        base_dir / "tests" / "fixtures",
        base_dir / "examples",
        base_dir / "samples",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            logger.info(f"Searching for Excel files in {search_dir}")
            for ext in excel_extensions:
                files = list(search_dir.rglob(f"*{ext}"))
                excel_files.extend(files)

    # Remove duplicates and sort
    excel_files = sorted(set(excel_files))

    logger.info(f"Found {len(excel_files)} Excel files")
    return excel_files


def process_spreadsheet(file_path: Path) -> dict[str, Any]:
    """
    Process a single spreadsheet and return analysis results.

    Args:
        file_path: Path to Excel file

    Returns:
        Analysis results or error information
    """
    logger.info(f"Processing {file_path.name}...")

    try:
        # Run formula analysis
        result = stage_3_formula_analysis(file_path)

        if hasattr(result, "error"):
            logger.error(f"Analysis failed for {file_path.name}: {result.error}")
            return {
                "status": "error",
                "file": str(file_path),
                "error": result.error,
            }

        analysis = result.value

        # Extract statistics
        stats = {
            "status": "success",
            "file": str(file_path),
            "file_name": file_path.name,
            "formula_count": len(analysis.dependency_graph),
            "circular_references": len(analysis.circular_references),
            "volatile_formulas": len(analysis.volatile_formulas),
            "external_references": len(analysis.external_references),
            "max_depth": analysis.max_dependency_depth,
            "complexity_score": analysis.formula_complexity_score,
        }

        logger.info(
            f"Successfully analyzed {file_path.name}: "
            f"{stats['formula_count']} formulas, "
            f"depth={stats['max_depth']}, "
            f"complexity={stats['complexity_score']}"
        )
        
        return {"analysis": analysis, **stats}
    except Exception as e:
        logger.exception(f"Unexpected error processing {file_path.name}")
        return {
            "status": "error",
            "file": str(file_path),
            "error": str(e),
        }


def load_to_neo4j(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Load all analysis results into Neo4j.

    Args:
        results: List of analysis results

    Returns:
        Summary of loading process
    """
    try:
        from spreadsheet_analyzer.graph_db.loader import GraphDatabaseLoader
        from spreadsheet_analyzer.pipeline.types import PipelineResult
    except ImportError:
        logger.exception("GraphDatabaseLoader not available. Please install neo4j package.")
        return {"status": "error", "error": "Neo4j not installed"}

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    logger.info(f"Connecting to Neo4j at {uri}")

    summary = {
        "total_files": len(results),
        "successful": 0,
        "failed": 0,
        "total_nodes": 0,
        "total_relationships": 0,
    }

    try:
        with GraphDatabaseLoader(uri, user, password) as loader:
            # Clear existing data on first load
            clear_existing = True

            for result in results:
                if result["status"] != "success" or "analysis" not in result:
                    summary["failed"] += 1
                    continue

                try:
                    # Create a minimal PipelineContext
                    context = PipelineContext(file_path=Path(result["file"]), start_time=datetime.now(), options={})

                    # Create a PipelineResult wrapper
                    pipeline_result = PipelineResult(
                        context=context,
                        integrity=None,
                        security=None,
                        structure=None,
                        formulas=result["analysis"],
                        content=None,
                        execution_time=0.0,
                        success=True,
                        errors=(),  # Empty errors tuple
                    )

                    # Load into Neo4j
                    load_stats = loader.load_pipeline_result(pipeline_result, clear_existing=clear_existing)

                    # Only clear on first successful load
                    clear_existing = False

                    summary["successful"] += 1
                    summary["total_nodes"] += load_stats.get("nodes_created", 0)
                    summary["total_relationships"] += load_stats.get("relationships_created", 0)

                    logger.info(
                        f"Loaded {result['file_name']}: "
                        f"{load_stats.get('nodes_created', 0)} nodes, "
                        f"{load_stats.get('relationships_created', 0)} relationships"
                    )

                except Exception:
                    logger.exception(f"Failed to load {result.get('file_name', 'unknown')}")
                    summary["failed"] += 1

    except Exception as e:
        logger.exception("Failed to connect to Neo4j")
        return {"status": "error", "error": str(e)}

    return summary


def main():
    """Main entry point for batch loader."""
    setup_logging()

    logger.info("Starting spreadsheet batch loader")

    # Wait for Neo4j to be ready
    if not wait_for_neo4j():
        logger.error("Could not connect to Neo4j. Exiting.")
        sys.exit(1)

    # Find all spreadsheet files
    base_dir = Path("/app")
    excel_files = find_spreadsheet_files(base_dir)

    if not excel_files:
        logger.warning("No Excel files found to process")
        return

    # Process each file
    logger.info(f"Processing {len(excel_files)} Excel files...")
    results = []

    for file_path in excel_files:
        result = process_spreadsheet(file_path)
        results.append(result)

    # Summary of processing
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")

    logger.info(f"Processing complete: {successful} successful, {failed} failed")

    # Load successful results into Neo4j
    if successful > 0:
        logger.info("Loading results into Neo4j...")
        load_summary = load_to_neo4j(results)

        if load_summary.get("status") == "error":
            logger.error(f"Failed to load to Neo4j: {load_summary.get('error')}")
        else:
            logger.info(
                f"Neo4j loading complete: "
                f"{load_summary['successful']}/{load_summary['total_files']} files loaded, "
                f"{load_summary['total_nodes']} nodes, "
                f"{load_summary['total_relationships']} relationships"
            )

    # Print summary report
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files found: {len(excel_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed to process: {failed}")

    if successful > 0:
        print("\nSuccessfully processed files:")
        for result in results:
            if result["status"] == "success":
                print(f"  - {result['file_name']}: {result['formula_count']} formulas")

    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result["status"] == "error":
                print(f"  - {result['file']}: {result['error']}")

    print("=" * 60)

    logger.info("Batch loader completed")


if __name__ == "__main__":
    main()
