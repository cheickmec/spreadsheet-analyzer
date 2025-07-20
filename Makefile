# Makefile for Spreadsheet Analyzer Docker Environment

.PHONY: help up down restart logs clean build status backup restore

# Default target
help:
	@echo "Spreadsheet Analyzer Docker Management"
	@echo "====================================="
	@echo "Available commands:"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make logs      - View logs for all services"
	@echo "  make clean     - Stop services and remove volumes"
	@echo "  make build     - Build/rebuild all images"
	@echo "  make status    - Show service status"
	@echo "  make backup    - Backup Neo4j data"
	@echo "  make restore   - Restore Neo4j data from backup"
	@echo ""
	@echo "Service-specific commands:"
	@echo "  make neo4j-logs    - View Neo4j logs"
	@echo "  make loader-logs   - View loader logs"
	@echo "  make reload        - Re-run the spreadsheet loader"
	@echo "  make shell-neo4j   - Open shell in Neo4j container"
	@echo "  make cypher-shell  - Open Cypher shell"

# Start all services
up:
	docker compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@echo "Services started. Access points:"
	@echo "  Neo4j Browser: http://localhost:7474"
	@echo "  Visualization: http://localhost:3000"
	@echo "  NeoDash:       http://localhost:5005"

# Stop all services
down:
	docker compose down

# Restart all services
restart:
	docker compose restart

# View logs
logs:
	docker compose logs -f

# Clean everything
clean:
	docker compose down -v
	@echo "All services stopped and data removed"

# Build/rebuild images
build:
	docker compose build --no-cache

# Show service status
status:
	@echo "Service Status:"
	@docker compose ps
	@echo ""
	@echo "Neo4j Status:"
	@docker exec spreadsheet-neo4j neo4j status 2>/dev/null || echo "Neo4j not running"

# Neo4j specific commands
neo4j-logs:
	docker compose logs -f neo4j

# Loader specific commands
loader-logs:
	docker compose logs -f spreadsheet-loader

reload:
	docker compose restart spreadsheet-loader
	docker compose logs -f spreadsheet-loader

# Shell access
shell-neo4j:
	docker exec -it spreadsheet-neo4j bash

shell-loader:
	docker exec -it spreadsheet-loader bash

# Cypher shell
cypher-shell:
	docker exec -it spreadsheet-neo4j cypher-shell -u neo4j -p spreadsheet123

# Backup Neo4j data
backup:
	@mkdir -p backups
	@BACKUP_FILE="backups/neo4j-backup-$$(date +%Y%m%d-%H%M%S).tar.gz"; \
	docker run --rm \
		-v spreadsheet-analyzer_neo4j_data:/data \
		-v $$(pwd)/backups:/backup \
		alpine tar czf /backup/$$(basename $$BACKUP_FILE) -C / data && \
	echo "Backup created: $$BACKUP_FILE"

# Restore Neo4j data
restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=backups/neo4j-backup-YYYYMMDD-HHMMSS.tar.gz"; \
		exit 1; \
	fi
	@if [ ! -f "$(BACKUP_FILE)" ]; then \
		echo "Backup file not found: $(BACKUP_FILE)"; \
		exit 1; \
	fi
	@echo "Restoring from $(BACKUP_FILE)..."
	@docker compose down
	@docker volume rm spreadsheet-analyzer_neo4j_data 2>/dev/null || true
	@docker volume create spreadsheet-analyzer_neo4j_data
	@docker run --rm \
		-v spreadsheet-analyzer_neo4j_data:/data \
		-v $$(pwd):/backup \
		alpine tar xzf /backup/$(BACKUP_FILE) -C /
	@echo "Restore complete. Starting services..."
	@make up

# Development helpers
dev-setup:
	cp .env.example .env
	@echo "Environment file created. Please edit .env to set passwords."

# Quick data inspection
inspect-data:
	@echo "Quick data inspection..."
	@docker exec spreadsheet-neo4j cypher-shell -u neo4j -p spreadsheet123 \
		"MATCH (n) RETURN labels(n)[0] as Type, COUNT(n) as Count;" 2>/dev/null || \
		echo "Neo4j not ready or no data loaded"

# Load a specific file
load-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make load-file FILE=path/to/file.xlsx"; \
		exit 1; \
	fi
	docker exec spreadsheet-loader python -m spreadsheet_analyzer.graph_db.single_loader "$(FILE)"

# Performance monitoring
monitor:
	@echo "Neo4j Memory Usage:"
	@docker exec spreadsheet-neo4j neo4j-admin server memory-recommendation
	@echo ""
	@echo "Container Stats:"
	@docker stats --no-stream spreadsheet-neo4j spreadsheet-loader spreadsheet-viz spreadsheet-neodash
