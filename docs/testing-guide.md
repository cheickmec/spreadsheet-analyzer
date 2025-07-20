# Testing Guide for Neo4j Integration

## Overview

This guide explains how to test the Neo4j integration both locally and with Docker.

## Local Testing (Without Docker)

Run the local test suite to verify all components work correctly:

```bash
# Install Neo4j driver if not already installed
uv pip install neo4j

# Run local tests
PYTHONPATH=src uv run test_local_neo4j.py
```

This tests:

- ✅ Neo4j package installation
- ✅ Loader module imports
- ✅ Spreadsheet processing capabilities
- ✅ Query engine functionality

## Docker Testing

### Prerequisites

1. **Start Docker Desktop** on your system
1. **Verify Docker is running**:
   ```bash
   docker --version
   docker compose version
   ```

### Starting Services

```bash
# Option 1: Using the setup script
./scripts/setup-docker.sh

# Option 2: Using docker compose directly
docker compose up -d

# Option 3: Using Make
make up
```

### Verifying Services

```bash
# Check service status
docker compose ps

# View logs
docker compose logs -f

# Test with automated script
./test_docker_env.sh
```

### Manual Verification

1. **Neo4j Browser**

   - Open http://localhost:7474
   - Login: neo4j / spreadsheet123
   - Run: `MATCH (n) RETURN COUNT(n)`

1. **Custom Visualization**

   - Open http://localhost:3000
   - Should see interactive graph interface

1. **NeoDash**

   - Open http://localhost:5005
   - Professional dashboard interface

### Testing Data Publishing

```bash
# Run the full connectivity test
docker exec spreadsheet-loader python test_neo4j_connectivity.py

# Or from host with correct environment
NEO4J_URI=bolt://localhost:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=spreadsheet123 \
PYTHONPATH=src uv run test_neo4j_connectivity.py
```

## Troubleshooting

### Docker Not Running

If you see:

```
Cannot connect to the Docker daemon at unix:///...
```

**Solution**: Start Docker Desktop

### Port Conflicts

If services fail to start:

```bash
# Check what's using the ports
lsof -i :7474,7687,3000,5005

# Stop conflicting services or change ports in docker-compose.yml
```

### Neo4j Connection Failed

If Neo4j won't connect:

```bash
# Check Neo4j logs
docker compose logs neo4j

# Restart Neo4j
docker compose restart neo4j

# Wait for health check
docker compose ps
```

### No Data Loaded

If no spreadsheets are loaded:

```bash
# Check loader logs
docker compose logs spreadsheet-loader

# Manually trigger reload
docker compose restart spreadsheet-loader

# Or run specific file
docker exec spreadsheet-loader python -m spreadsheet_analyzer.graph_db.batch_loader
```

## Test Results Summary

### Local Tests (Confirmed Working ✅)

- Neo4j Python driver: 5.28.1
- Batch loader processes files correctly
- Query engine works without Neo4j
- Found and processed 2446 formulas from test files

### Docker Tests (Ready When Docker Starts)

- Neo4j 5.25.1 configured with production settings
- Automatic spreadsheet loading on startup
- Three visualization interfaces ready
- All scripts and configurations in place

## Performance Expectations

- **Initial Load**: 30-60 seconds for all test files
- **Query Response**: < 100ms for most queries
- **Visualization**: Interactive with < 100 nodes

## Next Steps

1. Start Docker Desktop
1. Run `docker compose up -d`
1. Wait 1 minute for services to initialize
1. Access http://localhost:7474 to explore data
1. Try visualization at http://localhost:3000

## Useful Commands

```bash
# Quick status check
make status

# View all loaded data
make inspect-data

# Open Cypher shell
make cypher-shell

# Backup Neo4j data
make backup

# Complete cleanup
make clean
```
