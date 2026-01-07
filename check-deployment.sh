#!/bin/bash
# Quick verification script to check if everything is ready

echo "========================================="
echo "Life Expectancy - Deployment Check"
echo "========================================="
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is NOT running"
    echo "   Please start Docker Desktop/Engine"
    exit 1
else
    echo "✅ Docker is running"
fi

# Check required files
echo ""
echo "Checking required files..."
FILES=(
    "docker-compose.yml"
    "backend/Dockerfile"
    "backend/train_model.py"
    "backend/requirements.txt"
    "backend/.env"
    "frontend/Dockerfile"
    "frontend/nginx.conf"
    "deploy.sh"
)

all_files_present=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
        all_files_present=false
    fi
done

# Check data files
echo ""
echo "Checking data files..."
if [ -f "backend/data/life-expectancy.csv" ] || [ -f "backend/data/life-expectancy-merged.csv" ]; then
    echo "✅ Training data available"
else
    echo "❌ No training data found in backend/data/"
    all_files_present=false
fi

# Check models directory
echo ""
if [ -d "backend/models" ]; then
    echo "✅ Models directory exists"
    if [ -f "backend/models/best_model.pkl" ]; then
        echo "✅ Model already trained"
        model_trained=true
    else
        echo "⚠️  Model not yet trained (will be created during deployment)"
        model_trained=false
    fi
else
    echo "✅ Models directory will be created"
    model_trained=false
fi

echo ""
echo "========================================="
if [ "$all_files_present" = true ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Ready to deploy. Run:"
    echo "  ./deploy.sh"
    echo ""
    echo "Or manually:"
    if [ "$model_trained" = false ]; then
        echo "  1. docker compose --profile training up model-trainer --build"
    fi
    echo "  2. docker compose up -d"
    exit 0
else
    echo "❌ Some checks failed"
    echo "Please fix the issues above before deploying"
    exit 1
fi
