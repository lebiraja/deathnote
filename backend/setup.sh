#!/bin/bash

# Setup script for Life Expectancy Prediction API

set -e

echo "🚀 Setting up Life Expectancy Prediction API..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
if [ "$1" == "--dev" ]; then
    echo "🛠️  Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data logs

# Copy models if they exist in parent directory
echo "🤖 Checking for ML models..."
if [ -f ../gradient_boosting_model.pkl ]; then
    echo "Copying gradient_boosting_model.pkl..."
    cp ../gradient_boosting_model.pkl models/
fi
if [ -f ../preprocessor.pkl ]; then
    echo "Copying preprocessor.pkl..."
    cp ../preprocessor.pkl models/
fi
if [ -f ../scaler.pkl ]; then
    echo "Copying scaler.pkl..."
    cp ../scaler.pkl models/
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload"
echo ""
echo "Or with Docker:"
echo "  docker-compose up -d"
echo ""
echo "📚 API Documentation: http://localhost:8000/api/docs"
