#!/bin/bash

# Cardiology Protocols Vector Database Setup Script
# This script sets up the complete environment for medical protocol semantic search

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Welcome message
echo -e "${PURPLE}"
cat << "EOF"
🏥 ====================================================
   Cardiology Protocols Vector Database Setup
   ESC Guidelines → Semantic Search Pipeline
====================================================
EOF
echo -e "${NC}"

print_step "Starting setup process..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Please run this script from the project root directory."
    exit 1
fi

# Check prerequisites
print_step "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

print_success "Docker is available and running"

# Check Docker Compose
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    print_success "Docker Compose (standalone) is available"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    print_success "Docker Compose (plugin) is available"
else
    print_error "Docker Compose is not available. Please install Docker Compose."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1,2)
    print_success "Python 3 is available (version: $PYTHON_VERSION)"
else
    print_warning "Python 3 not found. You'll need it to run the ETL pipeline."
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 is available"
else
    print_warning "pip3 not found. You'll need it to install Python dependencies."
fi

# Create necessary directories
print_step "Creating project directories..."

# Core directories
mkdir -p qdrant_storage
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p backups

# Create config directory if it doesn't exist
mkdir -p config

print_success "Project directories created"

# Check for config files
print_step "Checking configuration files..."

CONFIG_MISSING=false

if [ ! -f "config/development.yaml" ]; then
    print_warning "config/development.yaml missing"
    CONFIG_MISSING=true
fi

if [ ! -f "config/production.yaml" ]; then
    print_warning "config/production.yaml missing"
    CONFIG_MISSING=true
fi

if [ ! -f "config/README.md" ]; then
    print_warning "config/README.md missing"
    CONFIG_MISSING=true
fi

if [ "$CONFIG_MISSING" = true ]; then
    print_info "Some config files are missing. The system will use default settings."
    print_info "Check the config/ directory and ensure you have the required YAML files."
else
    print_success "All configuration files found"
fi

# Check for source files
print_step "Checking source directories..."

if [ -d "mddocs" ] && [ "$(ls -A mddocs 2>/dev/null)" ]; then
    MD_COUNT=$(find mddocs -name "*.md" | wc -l)
    print_success "Found $MD_COUNT markdown files in mddocs/"
else
    print_warning "No markdown files found in mddocs/ directory"
    print_info "You'll need to convert PDFs to markdown before running the embedding process"
fi

if [ -d "pdfdocs" ] && [ "$(ls -A pdfdocs 2>/dev/null)" ]; then
    PDF_COUNT=$(find pdfdocs -name "*.pdf" | wc -l)
    print_success "Found $PDF_COUNT PDF files in pdfdocs/"
else
    print_warning "No PDF files found in pdfdocs/ directory"
    print_info "Download ESC protocols from the shared Google Drive folder"
fi

# Check if Qdrant is already running
print_step "Checking for existing Qdrant containers..."

if docker ps --format "table {{.Names}}" | grep -q "qdrant"; then
    print_warning "Qdrant container is already running"
    echo "Existing containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|qdrant)"
    echo ""
    read -p "Do you want to restart the containers? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Stopping existing containers..."
        $COMPOSE_CMD down
    else
        print_info "Keeping existing containers running"
        print_info "Use '$COMPOSE_CMD down' to stop them if needed"
        exit 0
    fi
fi

# Start Docker containers
print_step "Starting Qdrant container..."

# Pull the latest image first
print_info "Pulling latest Qdrant image..."
$COMPOSE_CMD pull

# Start containers
$COMPOSE_CMD up -d

if [ $? -eq 0 ]; then
    print_success "Containers started successfully"
else
    print_error "Failed to start containers"
    print_info "Checking logs for errors..."
    $COMPOSE_CMD logs
    exit 1
fi

# Wait for Qdrant to be ready
print_step "Waiting for Qdrant to be ready..."

max_attempts=60
attempt=1
PORT=6333

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:$PORT/health &> /dev/null; then
        print_success "Qdrant is ready and healthy!"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Qdrant failed to start after $max_attempts attempts ($(($max_attempts * 2)) seconds)"
        print_info "Checking container logs:"
        echo "----------------------------------------"
        $COMPOSE_CMD logs qdrant | tail -20
        echo "----------------------------------------"
        print_info "Try running: $COMPOSE_CMD restart"
        exit 1
    fi
    
    # Show progress
    if [ $((attempt % 10)) -eq 0 ]; then
        echo -n "  Attempt $attempt/$max_attempts - still waiting"
    else
        echo -n "."
    fi
    
    sleep 2
    ((attempt++))
done

echo  # New line after progress dots

# Verify Qdrant is working
print_step "Verifying Qdrant functionality..."

# Test basic API
if curl -s http://localhost:$PORT/collections | python3 -m json.tool &> /dev/null; then
    print_success "Qdrant API is responding correctly"
else
    print_warning "Qdrant API response seems unusual, but container is running"
fi

# Check if Python requirements are installed
print_step "Checking Python dependencies..."

if [ -f "requirements.txt" ]; then
    if command -v pip3 &> /dev/null; then
        print_info "Installing Python dependencies..."
        pip3 install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "pip3 not available. Install manually with: pip3 install -r requirements.txt"
    fi
else
    print_warning "requirements.txt not found"
fi

# Success message and next steps
echo ""
echo -e "${GREEN}"
cat << "EOF"
🎉 =========================================
   Setup Complete! 
=========================================
EOF
echo -e "${NC}"

print_success "Qdrant vector database is running"

echo ""
echo -e "${CYAN}📊 Access Points:${NC}"
echo "   🌐 Dashboard:    http://localhost:6333/dashboard"
echo "   🔗 API:          http://localhost:6333"
echo "   💚 Health:       http://localhost:6333/health"

echo ""
echo -e "${CYAN}📁 Directory Structure:${NC}"
echo "   📄 PDFs:         ./pdfdocs/"
echo "   📝 Markdown:     ./mddocs/"
echo "   🗄️  Database:     ./qdrant_storage/"
echo "   ⚙️  Config:       ./config/"

echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"

if [ ! -d "mddocs" ] || [ -z "$(ls -A mddocs 2>/dev/null)" ]; then
    echo "   1. 📥 Download ESC protocol PDFs to pdfdocs/"
    echo "   2. 🔄 Convert PDFs to markdown:"
    echo "      cd src && python3 converter.py ../pdfdocs ../mddocs page_breaks"
fi

if [ -d "mddocs" ] && [ "$(ls -A mddocs 2>/dev/null)" ]; then
    echo "   3. 🧠 Create vector embeddings:"
    echo "      python3 embedder.py ./mddocs"
else
    echo "   3. 🧠 After conversion, create embeddings:"
    echo "      python3 embedder.py ./mddocs"
fi

echo "   4. 🔍 Test semantic search:"
echo "      python3 vectorstore.py 'cardiac arrest protocols'"

echo ""
echo -e "${YELLOW}🛠️  Useful Commands:${NC}"
echo "   • View logs:     $COMPOSE_CMD logs -f qdrant"
echo "   • Restart:       $COMPOSE_CMD restart"
echo "   • Stop:          $COMPOSE_CMD down"
echo "   • Status:        $COMPOSE_CMD ps"
echo "   • Health check:  curl http://localhost:6333/health"

echo ""
echo -e "${YELLOW}📚 Resources:${NC}"
echo "   • ESC Guidelines: https://www.escardio.org/Guidelines"
echo "   • Shared Drive:   [Your Google Drive Link]"
echo "   • Documentation:  ./README.md"

echo ""
echo -e "${PURPLE}💡 Pro Tip: Use 'make' commands for convenience (if Makefile present)${NC}"
if [ -f "Makefile" ]; then
    echo "   • make start    • make logs    • make health    • make process"
fi

echo ""
print_info "Happy searching! 🔍✨"
