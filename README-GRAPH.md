# Graph Database Setup for Spreadsheet Analyzer

This document provides a quick guide to setting up the Neo4j graph database environment for visualizing and analyzing spreadsheet dependencies.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
./scripts/setup-docker.sh
```

This will:

- Check Docker installation
- Create necessary directories
- Start all services
- Load spreadsheets into Neo4j
- Provide access URLs

### Option 2: Manual Setup

```bash
# Copy environment file
cp .env.example .env

# Start services
docker compose up -d

# Watch the loader process
docker compose logs -f spreadsheet-loader
```

### Option 3: Using Make

```bash
# Setup and start
make dev-setup
make up

# View status
make status

# View logs
make logs
```

## 🌐 Access Points

Once running, access these interfaces:

| Service                  | URL                   | Description                     |
| ------------------------ | --------------------- | ------------------------------- |
| **Neo4j Browser**        | http://localhost:7474 | Native Neo4j interface          |
| **Custom Visualization** | http://localhost:3000 | Interactive dependency explorer |
| **NeoDash**              | http://localhost:5005 | Dashboard builder               |

Default login: `neo4j` / `spreadsheet123`

## 📊 Key Features

### Automated Analysis

- Finds all Excel files in the repository
- Extracts formula dependencies
- Identifies circular references
- Calculates complexity metrics
- Loads everything into Neo4j

### Visualization Options

1. **Neo4j Browser**: Run Cypher queries directly
1. **Custom Viz**: Interactive graph with predefined queries
1. **NeoDash**: Build custom dashboards

### Query Examples

Find most connected cells:

```cypher
MATCH (n:Cell)
WHERE n.pagerank > 0.5
RETURN n
LIMIT 25
```

Show circular references:

```cypher
MATCH p=(n:Cell)-[:DEPENDS_ON*]->(n)
RETURN p
```

Cross-sheet dependencies:

```cypher
MATCH (a:Cell)-[:DEPENDS_ON]->(b:Cell)
WHERE a.sheet <> b.sheet
RETURN a, b
```

## 🛠️ Common Operations

### Reload Spreadsheets

```bash
make reload
# or
docker compose restart spreadsheet-loader
```

### Access Cypher Shell

```bash
make cypher-shell
# or
docker exec -it spreadsheet-neo4j cypher-shell -u neo4j -p spreadsheet123
```

### Backup Data

```bash
make backup
```

### View Service Status

```bash
make status
```

## 📋 Requirements

- Docker & Docker Compose
- 8GB+ RAM recommended
- Ports: 7474, 7687, 3000, 5005

## 🔧 Configuration

Edit `docker-compose.yml` to adjust:

- Memory settings
- Port mappings
- Neo4j version
- Visualization options

## 📚 Documentation

- [Full Docker Setup Guide](docs/docker-setup.md)
- [Query Engine Design](docs/design/query-engine-design.md)
- [LLM Integration](docs/design/llm-function-definitions.md)

## 🆘 Troubleshooting

If services don't start:

```bash
# Check logs
docker compose logs neo4j

# Check ports
lsof -i :7474,7687,3000,5005

# Restart everything
docker compose down
docker compose up -d
```

## 🎯 Next Steps

1. **Explore**: Open Neo4j Browser and run some queries
1. **Visualize**: Try the custom visualization at http://localhost:3000
1. **Dashboard**: Create a dashboard in NeoDash
1. **Integrate**: Use the Python API for programmatic access

## 📝 Notes

- First run may take a few minutes to process all spreadsheets
- Data persists in Docker volumes between restarts
- Change default passwords before production use
- Memory settings in `docker-compose.yml` can be adjusted based on your system
