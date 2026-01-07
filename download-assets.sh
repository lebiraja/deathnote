#!/bin/bash
# Download models and datasets from AWS S3
# Configure your S3 bucket name and paths below

set -e

# Configuration
S3_BUCKET="your-bucket-name"
S3_MODELS_PATH="models"
S3_DATA_PATH="data"
LOCAL_MODELS_DIR="backend/models"
LOCAL_DATA_DIR="backend/data"

echo "☁️ Downloading assets from S3..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$LOCAL_MODELS_DIR"
mkdir -p "$LOCAL_DATA_DIR"

# Download models
echo "📦 Downloading models from s3://$S3_BUCKET/$S3_MODELS_PATH/..."
aws s3 sync "s3://$S3_BUCKET/$S3_MODELS_PATH/" "$LOCAL_MODELS_DIR/" \
    --exclude "*" \
    --include "*.pkl" \
    --include "*.joblib" \
    --delete

# Download datasets
echo "📊 Downloading datasets from s3://$S3_BUCKET/$S3_DATA_PATH/..."
aws s3 sync "s3://$S3_BUCKET/$S3_DATA_PATH/" "$LOCAL_DATA_DIR/" \
    --exclude "*" \
    --include "*.csv" \
    --include "*.json" \
    --delete

echo "✅ Assets downloaded successfully!"

# Show what was downloaded
echo ""
echo "📁 Downloaded models:"
ls -lh "$LOCAL_MODELS_DIR/"

echo ""
echo "📁 Downloaded datasets:"
ls -lh "$LOCAL_DATA_DIR/"
