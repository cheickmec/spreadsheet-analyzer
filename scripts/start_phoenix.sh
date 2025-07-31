#!/bin/bash
# Start Phoenix for Spreadsheet Analyzer Observability

set -e

echo "🚀 Starting Arize Phoenix for Spreadsheet Analyzer..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not installed."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check if Phoenix is already running
if docker ps | grep -q spreadsheet-analyzer-phoenix; then
    echo "✅ Phoenix is already running!"
    echo "   Access the UI at: http://localhost:6006"
    exit 0
fi

# Start Phoenix
echo "Starting Phoenix container..."
$COMPOSE_CMD -f docker-compose.phoenix.yml up -d

# Wait for Phoenix to be ready
echo "Waiting for Phoenix to start..."
for i in {1..30}; do
    if curl -s http://localhost:6006/health > /dev/null 2>&1; then
        echo "✅ Phoenix is ready!"
        echo ""
        echo "📊 Access Phoenix UI at: http://localhost:6006"
        echo ""
        echo "🔧 To use Phoenix with the analyzer:"
        echo "   python -m spreadsheet_analyzer.notebook_cli data.xlsx --phoenix-mode docker"
        echo ""
        echo "🛑 To stop Phoenix:"
        echo "   $COMPOSE_CMD -f docker-compose.phoenix.yml down"
        echo ""
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "⚠️  Phoenix is taking longer than expected to start."
echo "   Check the logs with: docker logs spreadsheet-analyzer-phoenix"
exit 1
