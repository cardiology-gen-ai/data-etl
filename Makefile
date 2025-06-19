# Makefile for Cardiology Protocols Vector Database
.PHONY: help setup start stop restart logs health clean install test run status
# Default target
help:
	@echo "🏥 Cardiology Protocols Vector Database"
	@echo ""
	@echo "Available commands:"
	@echo "  start     - Start Qdrant container (handles existing containers)"
	@echo "  setup     - Setup the environment and start Qdrant"
	@echo "  stop      - Stop Qdrant container"
	@echo "  restart   - Restart Qdrant container"
	@echo "  logs      - Show Qdrant container logs"
	@echo "  status    - Show container status"
	@echo "  health    - Check Qdrant health"
	@echo "  install   - Install Python dependencies"
	@echo "  convert   - Convert pdfs to markdown"
	@echo "  process   - Process documents and create embeddings"
	@echo "  search    - Test search functionality"
	@echo "  backup    - Create backup of database"
	@echo "  clean     - Remove stopped containers and clean up"
	@echo "  run       - Execute complete pipeline (convert → chunk → embed → store)"

# Start containers (handles existing containers gracefully)
start:
	@echo "▶️  Starting Qdrant..."
	@if docker ps -a --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
		if docker ps --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
			echo "✅ Qdrant container is already running"; \
		else \
			echo "🔄 Starting existing Qdrant container..."; \
			docker start qdrant; \
			echo "✅ Qdrant container started successfully"; \
		fi \
	else \
		echo "🆕 Creating and starting new Qdrant container..."; \
		docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
			-v "$$PWD/qdrant_storage:/qdrant/storage:z" \
			qdrant/qdrant; \
		echo "✅ Qdrant container created and started"; \
	fi
	@echo "📊 Dashboard: http://localhost:6333/dashboard"
	@echo "🔍 API: http://localhost:6333"

# Setup the environment and start Qdrant
setup:
	@echo "🚀 Setting up Cardiology Protocols Vector Database..."
	@chmod +x setup.sh
	@./setup.sh

# Stop containers
stop:
	@echo "⏹️  Stopping Qdrant..."
	@if docker ps --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
		docker stop qdrant; \
		echo "✅ Qdrant container stopped"; \
	else \
		echo "ℹ️  Qdrant container is not running"; \
	fi

# Restart containers
restart:
	@echo "🔄 Restarting Qdrant..."
	@$(MAKE) stop
	@sleep 2
	@$(MAKE) start

# View logs
logs:
	@echo "📋 Viewing Qdrant logs..."
	@docker logs -f qdrant

# Show container status
status:
	@echo "📈 Container Status:"
	@echo "==================="
	@if docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "qdrant"; then \
		docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|qdrant)"; \
	else \
		echo "❌ No Qdrant container found"; \
	fi

# Health check
health:
	@echo "🏥 Checking Qdrant health..."
	@if docker ps --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
		curl -f http://localhost:6333/healthz > /dev/null 2>&1 && \
		echo "✅ Qdrant is healthy and responding" || \
		echo "❌ Qdrant container is running but not responding (may still be starting up)"; \
	else \
		echo "❌ Qdrant container is not running"; \
	fi

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	@pip3 install -r requirements.txt

# Convert pdfs to markdown files
convert:
	@echo "⚙️ Converting pdf documents into markdown files..."
	@python3 src/converter.py ./pdfdocs

# Process documents
process:
	@echo "⚙️ Processing documents and creating embeddings..."
	@python3 src/embedder.py ./mddocs

# Test search
search:
	@echo "🔍 Testing search functionality..."
	@python3 src/vectorstore.py "cardiac arrest management"

# Create backup
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/qdrant_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz qdrant_storage/
	@echo "✅ Backup created in backups/ directory"

# Clean up stopped containers and unused resources
clean:
	@echo "🧹 Cleaning up..."
	@if docker ps -a --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
		if ! docker ps --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
			echo "🗑️  Removing stopped Qdrant container..."; \
			docker rm qdrant; \
		else \
			echo "⚠️  Qdrant container is running. Stop it first with 'make stop'"; \
		fi \
	fi
	@echo "✅ Cleanup completed"

# Execute complete pipeline
run:
	@echo "🔄 Starting complete cardiology protocols pipeline..."
	@echo ""
	@echo "Step 1/4: Ensuring Qdrant is running..."
	@$(MAKE) start
	@echo ""
	@echo "Step 2/4: Converting PDFs to Markdown..."
	@$(MAKE) convert
	@echo ""
	@echo "Step 3/4: Processing documents (chunking + embedding + storing)..."
	@$(MAKE) process
	@echo ""
	@echo "Step 4/4: Verifying pipeline completion..."
	@$(MAKE) health
	@echo ""
	@echo "🎉 Pipeline completed successfully!"
	@echo "📊 Dashboard: http://localhost:6333/dashboard"
	@echo "🔍 Test search with: make search"

# Show database info
info:
	@echo "📊 Database Information:"
	@echo "Dashboard: http://localhost:6333/dashboard"
	@echo "API Endpoint: http://localhost:6333"
	@echo "Health Check: http://localhost:6333/health"
	@echo ""
	@if curl -s http://localhost:6333/collections > /dev/null 2>&1; then \
		curl -s http://localhost:6333/collections | python -m json.tool; \
	else \
		echo "❌ Qdrant not responding or not running"; \
	fi

# Logs from container
logs:
	@echo "📋 Qdrant Container Logs:"
	@echo "========================="
	@if docker ps --format "table {{.Names}}" | grep -q "^qdrant$$"; then \
		docker logs --tail 50 -f qdrant; \
	else \
		echo "❌ Qdrant container is not running"; \
	fi