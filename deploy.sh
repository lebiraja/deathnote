#!/bin/bash
# Complete deployment script for Life Expectancy Prediction App

set -e

echo "========================================================================"
echo "Life Expectancy Prediction - Complete Deployment"
echo "========================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_info "Docker is running ✓"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p backend/models backend/logs backend/.env

# Create .env file if it doesn't exist
if [ ! -f backend/.env ]; then
    print_info "Creating .env file..."
    cat > backend/.env << EOL
# Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 16)

# CORS Settings
ALLOWED_ORIGINS=http://localhost,http://localhost:80,http://localhost:8000
EOL
    print_info ".env file created ✓"
else
    print_info ".env file already exists ✓"
fi

# Step 1: Train the model
print_info ""
print_info "========================================================================"
print_info "STEP 1: Training Machine Learning Model"
print_info "========================================================================"
print_info "Building training container and training the model..."
print_info "This may take a few minutes..."

docker compose --profile training up model-trainer --build

if [ $? -eq 0 ]; then
    print_info "Model training completed successfully ✓"
    
    # Check if model files were created
    if [ -f backend/models/best_model.pkl ]; then
        print_info "Model files created successfully ✓"
    else
        print_error "Model files were not created. Please check the logs."
        exit 1
    fi
else
    print_error "Model training failed. Please check the logs above."
    exit 1
fi

# Step 2: Build all services
print_info ""
print_info "========================================================================"
print_info "STEP 2: Building Docker Images"
print_info "========================================================================"
print_info "Building backend, frontend, and Redis services..."

docker compose build backend frontend

if [ $? -eq 0 ]; then
    print_info "All images built successfully ✓"
else
    print_error "Image building failed. Please check the logs above."
    exit 1
fi

# Step 3: Start all services
print_info ""
print_info "========================================================================"
print_info "STEP 3: Starting All Services"
print_info "========================================================================"
print_info "Starting backend, frontend, and Redis..."

docker compose up -d backend frontend redis

if [ $? -eq 0 ]; then
    print_info "All services started successfully ✓"
else
    print_error "Failed to start services. Please check the logs above."
    exit 1
fi

# Wait for services to be healthy
print_info ""
print_info "Waiting for services to be healthy..."
sleep 10

# Check service health
print_info ""
print_info "========================================================================"
print_info "Service Status"
print_info "========================================================================"

docker compose ps

print_info ""
print_info "========================================================================"
print_info "Deployment Complete!"
print_info "========================================================================"
print_info ""
print_info "🌐 Frontend:  http://localhost"
print_info "🔌 Backend:   http://localhost:8000"
print_info "📚 API Docs:  http://localhost:8000/docs"
print_info "🔍 Health:    http://localhost:8000/api/v1/health"
print_info ""
print_info "========================================================================"
print_info "Useful Commands:"
print_info "========================================================================"
print_info "View logs:           docker compose logs -f"
print_info "View backend logs:   docker compose logs -f backend"
print_info "View frontend logs:  docker compose logs -f frontend"
print_info "Stop services:       docker compose down"
print_info "Restart services:    docker compose restart"
print_info "Retrain model:       docker compose --profile training up model-trainer"
print_info ""
print_info "========================================================================"
