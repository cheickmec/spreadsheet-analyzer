"""Graph database integration for spreadsheet dependency analysis."""

# Always available - doesn't require Neo4j
from spreadsheet_analyzer.graph_db.query_engine import SpreadsheetQueryEngine, create_query_engine

__all__ = ["SpreadsheetQueryEngine", "create_query_engine"]

# Optional Neo4j-dependent modules
try:
    from spreadsheet_analyzer.graph_db.loader import GraphDatabaseLoader  # noqa: F401
    from spreadsheet_analyzer.graph_db.query_interface import GraphQueryInterface  # noqa: F401

    __all__.extend(["GraphDatabaseLoader", "GraphQueryInterface"])
except ImportError:
    # Neo4j not installed - only basic query engine is available
    pass
