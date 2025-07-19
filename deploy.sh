#!/bin/bash

# InsightGenie Deployment Script
# Supports local, Docker, and cloud deployment

set -e

echo "🚀 InsightGenie Deployment Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

check_requirements() {
    echo "🔍 Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
    fi
    print_success "Python 3 found"
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        print_error "pip is required but not installed"
    fi
    print_success "pip found"
}

setup_environment() {
    echo "🛠️  Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Activated virtual environment"
    
    # Install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Installed dependencies"
}

check_api_key() {
    echo "🔑 Checking API configuration..."
    
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY not set"
        echo "Please set your OpenAI API key:"
        echo "export OPENAI_API_KEY='your-key-here'"
        echo "Or create a .env file with OPENAI_API_KEY=your-key-here"
        echo ""
        echo "You can continue without it, but AI analysis will be disabled."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "OpenAI API key found"
    fi
}

deploy_local() {
    echo "💻 Starting local deployment..."
    
    check_requirements
    setup_environment
    check_api_key
    
    echo "🚀 Starting InsightGenie..."
    python src/main.py
}

deploy_docker() {
    echo "🐳 Starting Docker deployment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
    fi
    print_success "Docker found"
    
    # Build image
    echo "🔨 Building Docker image..."
    docker build -t insightgenie:latest .
    print_success "Docker image built"
    
    # Run container
    echo "🚀 Starting container..."
    docker run -d \
        --name insightgenie \
        -p 7860:7860 \
        -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
        insightgenie:latest
    
    print_success "Container started on http://localhost:7860"
    echo "To stop: docker stop insightgenie"
    echo "To view logs: docker logs insightgenie"
}

deploy_production() {
    echo "🏭 Production deployment setup..."
    
    cat << EOF
Production Deployment Options:

1. 🌐 Heroku
   git add .
   git commit -m "Deploy to production"
   git push heroku main
   heroku config:set OPENAI_API_KEY="your-key"

2. ☁️  AWS/GCP/Azure
   Use the provided Dockerfile:
   docker build -t insightgenie .
   docker tag insightgenie your-registry/insightgenie
   docker push your-registry/insightgenie

3. 🎯 Gradio Cloud
   gradio deploy src/main.py

4. 🔧 Custom Server
   python src/main.py --host 0.0.0.0 --port 8000

Environment Variables for Production:
- OPENAI_API_KEY: Your OpenAI API key
- MAX_FILE_SIZE_MB: File upload limit (default: 50)
- LOG_LEVEL: Logging level (default: INFO)
EOF
}

run_tests() {
    echo "🧪 Running tests..."
    
    if [ ! -d "venv" ]; then
        setup_environment
    fi
    
    source venv/bin/activate
    
    # Install test dependencies
    pip install pytest pytest-cov
    
    # Run tests
    python -m pytest tests/ -v --cov=src --cov-report=html
    
    print_success "Tests completed. Coverage report in htmlcov/index.html"
}

show_help() {
    cat << EOF
InsightGenie Deployment Script

Usage: ./deploy.sh [OPTION]

Options:
  local       Deploy locally (default)
  docker      Deploy with Docker
  production  Show production deployment options
  test        Run test suite
  help        Show this help message

Examples:
  ./deploy.sh local      # Run locally
  ./deploy.sh docker     # Run in Docker
  ./deploy.sh test       # Run tests

Environment Variables:
  OPENAI_API_KEY         Required for AI analysis

For more information, visit:
https://github.com/yourusername/insightgenie
EOF
}

# Main script logic
case "${1:-local}" in
    "local")
        deploy_local
        ;;
    "docker")
        deploy_docker
        ;;
    "production")
        deploy_production
        ;;
    "test")
        run_tests
        ;;
    "help")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac