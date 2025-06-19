#!/bin/bash

# Cardiology Protocols ETL Pipeline Setup Script
# This script sets up the complete environment for the cardiology protocols vector database

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "🏥 Cardiology Protocols ETL Pipeline Setup"
    echo "==========================================="
    echo -e "${NC}"
    
    # Step 1: Check for Python3 availability
    print_status "Checking Python3 availability..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python3 found: $PYTHON_VERSION"
    else
        print_error "Python3 is not installed or not in PATH"
        print_error "Please install Python3 (version 3.8 or higher) before running this script"
        exit 1
    fi
    
    # Check Python version (should be 3.8+)
    PYTHON_VERSION_NUM=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version $PYTHON_VERSION_NUM meets requirements (>= $REQUIRED_VERSION)"
    else
        print_warning "Python version $PYTHON_VERSION_NUM may not be compatible. Recommended: Python 3.8+"
    fi
    
    # Step 2: Check and update pip3
    print_status "Checking pip3 availability..."
    if command_exists pip3; then
        print_success "pip3 found"
        print_status "Updating pip3 to latest version..."
        python3 -m pip install --upgrade pip
        print_success "pip3 updated successfully"
    else
        print_error "pip3 is not installed or not in PATH"
        print_error "Please install pip3 before running this script"
        exit 1
    fi
    
    # Step 3: Install Python requirements
    print_status "Installing Python dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Python dependencies installed successfully"
    else
        print_error "requirements.txt not found in current directory"
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Step 4: Check for Docker availability
    print_status "Checking Docker availability..."
    if command_exists docker; then
        print_success "Docker found"
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            print_success "Docker daemon is running"
        else
            print_error "Docker daemon is not running. Please start Docker first"
            exit 1
        fi
    else
        print_error "Docker is not installed or not in PATH"
        print_error "Please install Docker before running this script"
        exit 1
    fi
    
    # Step 5: Check for Make availability
    print_status "Checking Make availability..."
    if command_exists make; then
        print_success "Make found"
    else
        print_error "Make is not installed or not in PATH"
        print_error "Please install Make before running this script"
        exit 1
    fi
    
    # Step 6: Create necessary directories
    print_status "Creating necessary directories..."
    
    DIRECTORIES=(
        "pdfdocs"
        "mddocs"
        "qdrant_storage"
        "qdrant_config"
        "src"
        "backups"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
    
    # Step 7: Set proper permissions for qdrant_storage
    print_status "Setting permissions for qdrant_storage..."
    chmod 755 qdrant_storage
    print_success "Permissions set for qdrant_storage"
    
    # Step 8: Check if Makefile exists
    print_status "Checking Makefile..."
    if [ ! -f "Makefile" ]; then
        print_error "Makefile not found in current directory"
        print_error "Please ensure you're running this script from the project root"
        exit 1
    fi
    print_success "Makefile found"
    
    # Step 9: Pull Qdrant Docker image
    print_status "Pulling Qdrant Docker image..."
    docker pull qdrant/qdrant:latest
    print_success "Qdrant Docker image pulled successfully"
    
    # Step 10: Start the container using make start
    print_status "Preparing to start Qdrant container..."
    
    # Check if container already exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^qdrant$"; then
        if docker ps --format "table {{.Names}}" | grep -q "^qdrant$"; then
            print_success "Qdrant container is already running!"
            print_status "Skipping container creation - proceeding to completion message"
        else
            print_status "Found existing stopped Qdrant container - it will be started automatically"
        fi
    else
        print_status "No existing Qdrant container found - a new one will be created"
    fi
    
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    
    # Check if container is already running to provide appropriate next steps
    if docker ps --format "table {{.Names}}" | grep -q "^qdrant$"; then
        echo "1. ✅ Qdrant container is already running!"
        echo "2. 📊 Access the dashboard at: http://localhost:6333/dashboard"
        echo "3. 🚀 Run the complete ETL pipeline: make run"
        echo "4. 🔍 Test search functionality: make search"
        echo ""
        echo -e "${BLUE}System is ready to use!${NC}"
    else
        echo "1. 🚀 Starting Qdrant container..."
        echo "2. 📊 Once running, access the dashboard at: http://localhost:6333/dashboard"
        echo "3. 🚀 To run the complete ETL pipeline: make run"
        echo "4. 🔍 To test search functionality: make search"
        echo ""
        # Start the container (this will handle existing containers gracefully)
        make start
        echo ""
        echo -e "${GREEN}🎉 Qdrant is now running!${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Quick commands:${NC}"
    echo "  make run      - Run complete pipeline (convert → embed → store)"
    echo "  make search   - Test search functionality"
    echo "  make status   - Check container status"
    echo "  make stop     - Stop Qdrant container"
    echo "  make health   - Check Qdrant health"
    echo "  make logs     - View container logs"
}

# Function to handle script interruption
cleanup() {
    print_warning "Setup interrupted by user"
    exit 1
}

# Trap Ctrl+C
trap cleanup INT

# Check if running as root (not recommended for Docker on some systems)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This may cause permission issues with Docker volumes."
    print_warning "Consider running as a regular user with Docker permissions."
fi

# Check if we're in the right directory (basic check)
if [ ! -f "requirements.txt" ] || [ ! -f "Makefile" ]; then
    print_error "This doesn't appear to be the cardiology protocols ETL project directory"
    print_error "Please run this script from the project root where requirements.txt and Makefile exist"
    exit 1
fi

# Run main setup
main "$@"