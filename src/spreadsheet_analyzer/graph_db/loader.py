"""
Graph Database Loader for Excel Analysis Results.

This module handles loading pipeline results into Neo4j for
efficient querying and analysis.
"""

import logging
from typing import Any

from neo4j import GraphDatabase

from spreadsheet_analyzer.pipeline.types import FormulaAnalysis, PipelineResult

logger = logging.getLogger(__name__)


class GraphDatabaseLoader:
    """Loads spreadsheet analysis results into Neo4j graph database."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j at %s", uri)

    def close(self):
        """Close database connection."""
        self.driver.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def load_pipeline_result(self, result: PipelineResult, *, clear_existing: bool = True) -> dict[str, Any]:
        """
        Load complete pipeline result into Neo4j.

        Args:
            result: Pipeline analysis result
            clear_existing: Whether to clear existing graph data

        Returns:
            Summary statistics of loaded data
        """
        if not result.formulas:
            logger.warning("No formula analysis to load")
            return {"status": "no_data"}

        with self.driver.session() as session:
            if clear_existing:
                self._clear_graph(session)

            # Create constraints and indexes
            self._create_constraints(session)

            # Load the formula analysis
            stats = self._load_formula_analysis(session, result.formulas)

            # Pre-compute PageRank
            pagerank_stats = self._compute_pagerank(session)

            # Generate summary
            summary = self._generate_summary(session)
            summary.update(
                {
                    "nodes_created": stats["nodes_created"],
                    "relationships_created": stats["relationships_created"],
                    "pagerank_computed": pagerank_stats["nodes_ranked"],
                }
            )

            return summary

    def _clear_graph(self, session) -> None:
        """Clear all nodes and relationships."""
        logger.info("Clearing existing graph data")
        session.run("MATCH (n) DETACH DELETE n")

    def _create_constraints(self, session) -> None:
        """Create database constraints and indexes for performance."""
        logger.info("Creating constraints and indexes")

        # Create unique constraint on cell keys
        session.run("""
            CREATE CONSTRAINT cell_key_unique IF NOT EXISTS
            FOR (c:Cell) REQUIRE c.key IS UNIQUE
        """)

        # Create unique constraint on range keys
        session.run("""
            CREATE CONSTRAINT range_key_unique IF NOT EXISTS
            FOR (r:Range) REQUIRE r.key IS UNIQUE
        """)

        # Create indexes for common queries
        session.run("CREATE INDEX cell_sheet IF NOT EXISTS FOR (c:Cell) ON (c.sheet)")
        session.run("CREATE INDEX cell_ref IF NOT EXISTS FOR (c:Cell) ON (c.ref)")
        session.run("CREATE INDEX range_sheet IF NOT EXISTS FOR (r:Range) ON (r.sheet)")

    def _load_formula_analysis(self, session, analysis: FormulaAnalysis) -> dict[str, int]:
        """Load formula dependency graph into Neo4j."""
        logger.info("Loading formula analysis with %d nodes", len(analysis.dependency_graph))

        nodes_created = 0
        relationships_created = 0

        # Batch create nodes
        node_batch = []
        for node_key, node in analysis.dependency_graph.items():
            node_data = {
                "key": node_key,
                "sheet": node.sheet,
                "ref": node.cell,
                "formula": node.formula or "",
                "depth": node.depth if node.depth is not None else 0,
                "node_type": "cell",  # Default since stage_3 doesn't set this
            }

            # Skip range-specific properties for now since stage_3 doesn't populate them

            node_batch.append(node_data)

            # Batch insert every 1000 nodes
            if len(node_batch) >= 1000:
                nodes_created += self._batch_create_nodes(session, node_batch)
                node_batch = []

        # Insert remaining nodes
        if node_batch:
            nodes_created += self._batch_create_nodes(session, node_batch)

        # Create relationships
        for node_key, node in analysis.dependency_graph.items():
            if node.dependencies:
                # Batch create DEPENDS_ON relationships
                deps = [{"from": node_key, "to": f"{dep.sheet}!{dep.cell}"} for dep in node.dependencies]
                relationships_created += self._batch_create_relationships(session, deps, "DEPENDS_ON")

        logger.info("Created %d nodes and %d relationships", nodes_created, relationships_created)

        return {"nodes_created": nodes_created, "relationships_created": relationships_created}

    def _batch_create_nodes(self, session, node_batch: list[dict]) -> int:
        """Batch create nodes based on their type."""
        cell_nodes = [n for n in node_batch if n["node_type"] == "cell"]
        range_nodes = [n for n in node_batch if n["node_type"] == "range"]

        count = 0

        if cell_nodes:
            result = session.run(
                """
                UNWIND $nodes AS node
                CREATE (c:Cell {
                    key: node.key,
                    sheet: node.sheet,
                    ref: node.ref,
                    formula: node.formula,
                    depth: node.depth if node.depth is not None else 0
                })
                RETURN count(c) as created
            """,
                nodes=cell_nodes,
            )
            count += result.single()["created"]

        if range_nodes:
            result = session.run(
                """
                UNWIND $nodes AS node
                CREATE (r:Range {
                    key: node.key,
                    sheet: node.sheet,
                    ref: node.ref,
                    type: node.range_type,
                    size: node.range_size,
                    start_cell: node.start_cell,
                    end_cell: node.end_cell,
                    depth: node.depth if node.depth is not None else 0
                })
                RETURN count(r) as created
            """,
                nodes=range_nodes,
            )
            count += result.single()["created"]

        return count

    def _batch_create_relationships(self, session, relationships: list[dict], rel_type: str) -> int:
        """Batch create relationships."""
        result = session.run(
            f"""
            UNWIND $rels AS rel
            MATCH (from {{key: rel.from}})
            MATCH (to {{key: rel.to}})
            CREATE (from)-[:{rel_type}]->(to)
            RETURN count(*) as created
        """,
            rels=relationships,
        )

        record = result.single()
        return int(record["created"]) if record else 0

    def _compute_pagerank(self, session) -> dict[str, Any]:
        """Pre-compute PageRank scores for all nodes."""
        logger.info("Computing PageRank scores")

        # Create in-memory graph projection
        session.run("""
            CALL gds.graph.project.cypher(
                'formula-graph',
                'MATCH (n) WHERE n:Cell OR n:Range RETURN id(n) AS id',
                'MATCH (n)-[:DEPENDS_ON]->(m) RETURN id(n) AS source, id(m) AS target'
            )
        """)

        # Run PageRank algorithm
        result = session.run("""
            CALL gds.pageRank.write('formula-graph', {
                writeProperty: 'pagerank',
                maxIterations: 100,
                dampingFactor: 0.85
            })
            YIELD nodePropertiesWritten, ranIterations
            RETURN nodePropertiesWritten, ranIterations
        """)

        stats = result.single()

        # Drop the projection
        session.run("CALL gds.graph.drop('formula-graph')")

        return {"nodes_ranked": stats["nodePropertiesWritten"], "iterations": stats["ranIterations"]}

    def _generate_summary(self, session) -> dict[str, Any]:
        """Generate summary statistics of the loaded graph."""
        result = session.run("""
            MATCH (n)
            WITH
                count(DISTINCT CASE WHEN n:Cell THEN n END) as cell_count,
                count(DISTINCT CASE WHEN n:Range THEN n END) as range_count,
                count(DISTINCT CASE WHEN n:Cell AND n.formula <> '' THEN n END) as formula_count
            MATCH ()-[r:DEPENDS_ON]->()
            WITH cell_count, range_count, formula_count, count(r) as edge_count
            MATCH (n)
            WHERE n.pagerank IS NOT NULL
            WITH cell_count, range_count, formula_count, edge_count,
                 max(n.pagerank) as max_pagerank,
                 avg(n.pagerank) as avg_pagerank
            MATCH (n)
            WITH cell_count, range_count, formula_count, edge_count,
                 max_pagerank, avg_pagerank,
                 max(n.depth) as max_depth
            OPTIONAL MATCH p=(n)-[:DEPENDS_ON*]->(n)
            WITH cell_count, range_count, formula_count, edge_count,
                 max_pagerank, avg_pagerank, max_depth,
                 count(DISTINCT n) as circular_nodes
            RETURN {
                total_nodes: cell_count + range_count,
                cell_nodes: cell_count,
                range_nodes: range_count,
                formula_nodes: formula_count,
                total_edges: edge_count,
                max_depth: max_depth,
                max_pagerank: max_pagerank,
                avg_pagerank: avg_pagerank,
                circular_reference_nodes: circular_nodes
            } as summary
        """)

        record = result.single()
        return dict(record["summary"]) if record else {}
