#!/bin/bash
# Upload models and datasets to AWS S3
# Run this locally to upload your trained models and datasets

set -e

# Configuration
S3_BUCKET="your-bucket-name"
S3_MODELS_PATH="models"
S3_DATA_PATH="data"
LOCAL_MODELS_DIR="backend/models"
LOCAL_DATA_DIR="backend/data"

echo "☁️ Uploading assets to S3..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if directories exist
if [ ! -d "$LOCAL_MODELS_DIR" ]; then
    echo "❌ Models directory not found: $LOCAL_MODELS_DIR"
    exit 1
fi

if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "❌ Data directory not found: $LOCAL_DATA_DIR"
    exit 1
fi

# Upload models
echo "📦 Uploading models to s3://$S3_BUCKET/$S3_MODELS_PATH/..."
aws s3 sync "$LOCAL_MODELS_DIR/" "s3://$S3_BUCKET/$S3_MODELS_PATH/" \
    --exclude "*" \
    --include "*.pkl" \
    --include "*.joblib" \
    --delete

# Upload datasets
echo "📊 Uploading datasets to s3://$S3_BUCKET/$S3_DATA_PATH/..."
aws s3 sync "$LOCAL_DATA_DIR/" "s3://$S3_BUCKET/$S3_DATA_PATH/" \
    --exclude "*" \
    --include "*.csv" \
    --include "*.json" \
    --delete

echo "✅ Assets uploaded successfully!"

# Show S3 bucket contents
echo ""
echo "📁 S3 Models:"
aws s3 ls "s3://$S3_BUCKET/$S3_MODELS_PATH/" --human-readable

echo ""
echo "📁 S3 Datasets:"
aws s3 ls "s3://$S3_BUCKET/$S3_DATA_PATH/" --human-readable
