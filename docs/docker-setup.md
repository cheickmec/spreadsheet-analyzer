# Docker Environment Setup for Spreadsheet Analysis

This guide explains how to set up and use the containerized environment for analyzing spreadsheet dependencies with Neo4j and visualization tools.

## Overview

The Docker setup includes:

- **Neo4j 5.25.1**: Graph database for storing and querying dependencies
- **Spreadsheet Loader**: Python service that processes Excel files and loads them into Neo4j
- **NeoDash**: Professional dashboard tool for graph visualization
- **Custom Visualization**: Neovis.js-based interactive graph explorer

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB of RAM available
- Port availability: 7474, 7687, 3000, 5005

## Quick Start

1. **Clone the repository and navigate to the project directory**

   ```bash
   cd spreadsheet-analyzer
   ```

1. **Copy the environment file**

   ```bash
   cp .env.example .env
   # Edit .env to change passwords and settings
   ```

1. **Start all services**

   ```bash
   docker compose up -d
   ```

1. **Wait for services to be ready** (about 30-60 seconds)

   ```bash
   docker compose logs -f spreadsheet-loader
   ```

1. **Access the interfaces**

   - Neo4j Browser: http://localhost:7474 (login: neo4j/spreadsheet123)
   - Custom Visualization: http://localhost:3000
   - NeoDash: http://localhost:5005

## Services

### Neo4j Database

The core graph database storing all dependency information.

**Features:**

- Latest Neo4j 5.25.1 Community Edition
- APOC procedures enabled
- Optimized memory settings for production use
- Persistent data storage

**Access:**

- Browser UI: http://localhost:7474
- Bolt protocol: bolt://localhost:7687
- Default credentials: neo4j/spreadsheet123

### Spreadsheet Loader

Automated service that processes all Excel files in the repository.

**Features:**

- Finds all .xlsx, .xlsm, and .xls files
- Analyzes formula dependencies
- Loads results into Neo4j
- Generates processing reports

**Monitoring:**

```bash
# View processing logs
docker compose logs spreadsheet-loader

# Re-run the loader
docker compose restart spreadsheet-loader
```

### NeoDash

Professional Neo4j dashboard tool for creating custom visualizations.

**Features:**

- Drag-and-drop dashboard builder
- Multiple visualization types
- Saved queries and reports
- Export capabilities

**Access:** http://localhost:5005

### Custom Visualization

Interactive graph explorer built with Neovis.js.

**Features:**

- Predefined queries for common analyses
- Custom Cypher query support
- Interactive node exploration
- Real-time graph manipulation

**Access:** http://localhost:3000

## Common Operations

### View All Loaded Spreadsheets

In Neo4j Browser, run:

```cypher
MATCH (n:Cell)
RETURN DISTINCT n.sheet as Sheet, COUNT(n) as CellCount
ORDER BY CellCount DESC
```

### Find High-Impact Cells

```cypher
MATCH (n)
WHERE n.pagerank IS NOT NULL
RETURN n.key as Cell, n.sheet as Sheet, n.pagerank as Impact
ORDER BY n.pagerank DESC
LIMIT 20
```

### Explore Circular References

```cypher
MATCH p=(n:Cell)-[:DEPENDS_ON*]->(n)
RETURN p
LIMIT 10
```

### View Cross-Sheet Dependencies

```cypher
MATCH (n1:Cell)-[:DEPENDS_ON]->(n2:Cell)
WHERE n1.sheet <> n2.sheet
RETURN n1.sheet as FromSheet, n2.sheet as ToSheet, COUNT(*) as Dependencies
ORDER BY Dependencies DESC
```

## Memory Configuration

Adjust Neo4j memory settings in `docker-compose.yml` based on your system:

**For 8GB systems:**

```yaml
- NEO4J_server_memory_pagecache_size=1G
- NEO4J_server_memory_heap_max__size=1G
```

**For 16GB+ systems:**

```yaml
- NEO4J_server_memory_pagecache_size=4G
- NEO4J_server_memory_heap_max__size=4G
```

## Troubleshooting

### Services won't start

Check port availability:

```bash
# Check if ports are in use
lsof -i :7474,7687,3000,5005
```

### Neo4j connection failed

Wait for Neo4j to fully start:

```bash
# Check Neo4j logs
docker compose logs neo4j

# Test connection
docker exec spreadsheet-neo4j neo4j status
```

### Loader can't find files

Ensure Excel files are in the correct directories:

- `test-files/`
- `tests/fixtures/`
- `examples/`
- `samples/`

### Out of memory errors

Reduce batch size in loader or increase Docker memory:

```bash
# Check Docker memory limits
docker system info | grep -i memory

# Increase memory in Docker Desktop settings
```

## Data Persistence

All Neo4j data is persisted in Docker volumes:

- `neo4j_data`: Database files
- `neo4j_logs`: Log files
- `neo4j_import`: Import directory

To backup:

```bash
docker run --rm -v spreadsheet-analyzer_neo4j_data:/data -v $(pwd):/backup alpine tar czf /backup/neo4j-backup.tar.gz -C / data
```

To restore:

```bash
docker run --rm -v spreadsheet-analyzer_neo4j_data:/data -v $(pwd):/backup alpine tar xzf /backup/neo4j-backup.tar.gz -C /
```

## Advanced Configuration

### Using Enterprise Edition

1. Obtain a Neo4j Enterprise license
1. Update `docker-compose.yml`:
   ```yaml
   image: neo4j:5.25.1-enterprise
   environment:
     - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
   ```

### Custom Plugins

Add plugins to the `neo4j/plugins/` directory before starting:

```bash
mkdir -p neo4j/plugins
# Copy .jar files to neo4j/plugins/
docker compose restart neo4j
```

### Production Deployment

For production use:

1. Change all default passwords
1. Enable SSL/TLS
1. Configure proper backup strategy
1. Monitor resource usage
1. Set up log rotation

## Query Examples

### Formula Complexity Analysis

```cypher
// Find most complex formulas by dependency count
MATCH (n:Cell)
WHERE n.formula IS NOT NULL
WITH n, size((n)-[:DEPENDS_ON]->()) as deps
RETURN n.key as Cell, n.formula as Formula, deps as Dependencies
ORDER BY deps DESC
LIMIT 20
```

### Impact Analysis

```cypher
// What cells would be affected if Sheet1!A1 changes?
MATCH (start:Cell {key: "Sheet1!A1"})
MATCH path = (start)<-[:DEPENDS_ON*1..3]-(dependent)
RETURN path
```

### Range Analysis

```cypher
// Find all formulas using large ranges
MATCH (c:Cell)-[:DEPENDS_ON]->(r:Range)
WHERE r.size > 100
RETURN c.key as Formula, r.ref as Range, r.size as Size
ORDER BY r.size DESC
```

## Shutting Down

To stop all services:

```bash
docker compose down
```

To stop and remove all data:

```bash
docker compose down -v
```

## Performance Tips

1. **Initial Load**: The first run processes all spreadsheets. Subsequent runs are faster.
1. **Query Performance**: Create indexes for frequently queried properties
1. **Visualization**: Limit initial queries to < 100 nodes for best performance
1. **Memory**: Monitor Neo4j memory usage and adjust settings as needed

## Security Considerations

1. **Change Default Passwords**: Always change the default Neo4j password
1. **Network Isolation**: Use Docker networks to isolate services
1. **Access Control**: Restrict port access in production
1. **Data Encryption**: Enable encryption for sensitive data

## Next Steps

1. Explore the visualization interfaces
1. Write custom Cypher queries for your analysis needs
1. Create NeoDash dashboards for recurring reports
1. Export insights for further analysis

For more information on Neo4j and Cypher queries, see:

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/developer/cypher/)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/)
