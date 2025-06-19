# Makefile for Cardiology Protocols Vector Database

.PHONY: help setup start stop restart logs health clean install test

# Default target
help:
	@echo "🏥 Cardiology Protocols Vector Database"
	@echo ""
	@echo "Available commands:"
	@echo "  setup     - Initial setup (create dirs, start containers)"
	@echo "  start     - Start Qdrant container"
	@echo "  stop      - Stop Qdrant container"
	@echo "  restart   - Restart Qdrant container"
	@echo "  logs      - View container logs"
	@echo "  health    - Check Qdrant health"
	@echo "  clean     - Stop containers and remove volumes"
	@echo "  install   - Install Python dependencies"
	@echo "  process   - Process documents and create embeddings"
	@echo "  search    - Test search functionality"
	@echo "  backup    - Create backup of database"
	@echo "  test      - Run tests"

# Initial setup
setup:
	@echo "🚀 Setting up Cardiology Protocols Database..."
	@chmod +x setup.sh
	@./setup.sh

# Start containers
start:
	@echo "▶️  Starting Qdrant..."
	@docker-compose up -d

# Stop containers
stop:
	@echo "⏹️  Stopping Qdrant..."
	@docker-compose down

# Restart containers
restart:
	@echo "🔄 Restarting Qdrant..."
	@docker-compose restart

# View logs
logs:
	@echo "📋 Viewing Qdrant logs..."
	@docker-compose logs -f qdrant

# Health check
health:
	@echo "🏥 Checking Qdrant health..."
	@curl -f http://localhost:6333/health && echo "✅ Qdrant is healthy" || echo "❌ Qdrant is not responding"

# Clean everything
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v
	@echo "⚠️  Database data preserved in qdrant_storage/"

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	@pip install -r requirements.txt

# Process documents
process:
	@echo "⚙️  Processing documents and creating embeddings..."
	@python scripts/embedder.py ./data/processed/

# Test search
search:
	@echo "🔍 Testing search functionality..."
	@python scripts/vectorstore.py "cardiac arrest management"

# Create backup
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/qdrant_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz qdrant_storage/
	@echo "✅ Backup created in backups/ directory"

# Run tests
test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v

# Development commands
dev-start: start
	@echo "🛠️  Development environment ready!"
	@echo "Dashboard: http://localhost:6333/dashboard"

# Show database info
info:
	@echo "📊 Database Information:"
	@echo "Dashboard: http://localhost:6333/dashboard"
	@echo "API Endpoint: http://localhost:6333"
	@echo "Health Check: http://localhost:6333/health"
	@echo ""
	@curl -s http://localhost:6333/collections 2>/dev/null | python -m json.tool || echo "❌ Qdrant not responding"

# Quick status
status:
	@echo "📈 Container Status:"
	@docker-compose ps
