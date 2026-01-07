#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main

# Pull LFS files if using Git LFS
if command -v git-lfs &> /dev/null; then
    echo "📦 Pulling LFS files..."
    git lfs pull || echo "No LFS files to pull"
fi

# Download from S3 if configured
if [ -f "download-assets.sh" ]; then
    echo "☁️ Downloading assets from S3..."
    ./download-assets.sh
fi

# Rebuild containers
echo "🔨 Building Docker images..."
docker-compose build

# Stop old containers
echo "🛑 Stopping old containers..."
docker-compose down

# Start new containers
echo "✅ Starting new containers..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 15

# Health checks
echo "🏥 Running health checks..."

# Check backend
if curl -f http://localhost:8000/api/v1/health 2>/dev/null; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️ Backend health check failed"
    docker-compose logs backend --tail=50
fi

# Check frontend
if curl -f http://localhost:80 2>/dev/null; then
    echo "✅ Frontend is accessible!"
else
    echo "⚠️ Frontend check failed"
    docker-compose logs frontend --tail=50
fi

# Show status
echo "📊 Container status:"
docker-compose ps

# Show recent logs
echo "📝 Recent logs:"
docker-compose logs --tail=30

echo "✨ Deployment complete!"
