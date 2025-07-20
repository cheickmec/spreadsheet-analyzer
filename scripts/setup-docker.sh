#!/bin/bash
# Setup script for Docker environment

set -e

echo "Spreadsheet Analyzer Docker Setup"
echo "================================="
echo

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not updated."
    echo "   Please update Docker to the latest version."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "   Please edit .env to change default passwords!"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p neo4j/conf neo4j/licenses logs backups

# Check if any containers are already running
if docker compose ps --quiet | grep -q .; then
    echo "⚠️  Some containers are already running."
    read -p "Do you want to restart them? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Restarting containers..."
        docker compose down
    else
        echo "ℹ️  Keeping existing containers running"
        exit 0
    fi
fi

# Pull latest images
echo "📥 Pulling Docker images..."
docker compose pull

# Start services
echo "🚀 Starting services..."
docker compose up -d

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker exec spreadsheet-neo4j neo4j status &> /dev/null; then
        echo "✅ Neo4j is ready!"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "❌ Neo4j failed to start. Check logs with: docker compose logs neo4j"
    exit 1
fi

echo
echo "✅ All services started successfully!"
echo
echo "Access points:"
echo "  🌐 Neo4j Browser:   http://localhost:7474"
echo "  📊 Visualization:   http://localhost:3000"
echo "  📈 NeoDash:        http://localhost:5005"
echo
echo "Default credentials:"
echo "  Username: neo4j"
echo "  Password: spreadsheet123 (change in .env!)"
echo
echo "Useful commands:"
echo "  View logs:        docker compose logs -f"
echo "  Stop services:    docker compose down"
echo "  Restart loader:   docker compose restart spreadsheet-loader"
echo "  Open Cypher:      make cypher-shell"
echo
echo "The spreadsheet loader is now processing Excel files..."
echo "Check progress with: docker compose logs -f spreadsheet-loader"
