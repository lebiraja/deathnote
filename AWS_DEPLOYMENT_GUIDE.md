# AWS Deployment Guide - Life Expectancy Prediction App

> [!IMPORTANT]
> This guide covers deploying your Life Expectancy Prediction application to AWS EC2 using Docker and SSH, with automated CI/CD via GitHub Actions.

## Table of Contents
1. [AWS Instance Selection](#1-aws-instance-selection)
2. [Model & Dataset Strategy](#2-model--dataset-strategy)
3. [Initial Server Setup](#3-initial-server-setup)
4. [Docker Deployment via SSH](#4-docker-deployment-via-ssh)
5. [GitHub Actions CI/CD Pipeline](#5-github-actions-cicd-pipeline)
6. [Monitoring & Maintenance](#6-monitoring--maintenance)

---

## 1. AWS Instance Selection

### Recommended Instance Types

Based on your application requirements (ML model training + Flask API + React frontend):

#### **Option A: t3.medium (Recommended for Production)**
- **vCPUs**: 2
- **RAM**: 4 GB
- **Storage**: 30-50 GB EBS (gp3)
- **Cost**: ~$30/month
- **Best for**: Running pre-trained models, API serving, small-medium datasets
- **Use when**: Models are already trained, only serving predictions

#### **Option B: t3.large (For Training + Serving)**
- **vCPUs**: 2
- **RAM**: 8 GB
- **Storage**: 50-100 GB EBS (gp3)
- **Cost**: ~$60/month
- **Best for**: Occasional model retraining + serving
- **Use when**: You need to retrain models on the server

#### **Option C: c5.xlarge (Heavy Training Workloads)**
- **vCPUs**: 4 (compute-optimized)
- **RAM**: 8 GB
- **Storage**: 100 GB EBS (gp3)
- **Cost**: ~$120/month
- **Best for**: Frequent model training, large datasets
- **Use when**: Dataset > 1GB or complex ensemble models

### Storage Recommendations

| Component | Size Estimate | Storage Type |
|-----------|--------------|--------------|
| OS + Docker | 10 GB | EBS gp3 |
| Application Code | 1-2 GB | EBS gp3 |
| Dataset (CSV) | 5-50 GB | EBS gp3 or S3 |
| Trained Models | 100 MB - 5 GB | EBS gp3 or S3 |
| Logs & Cache | 5-10 GB | EBS gp3 |
| **Total Recommended** | **30-100 GB** | **EBS gp3** |

> [!TIP]
> Use **EBS gp3** for better price/performance ratio compared to gp2. Enable EBS encryption for security.

---

## 2. Model & Dataset Strategy

### Problem: Large Files in Git

> [!WARNING]
> **DO NOT** commit large model files (.pkl) or datasets (.csv) to Git if they exceed 100 MB.

### Solution Options

#### **Strategy 1: Git LFS (Large File Storage)** ⭐ Recommended
```bash
# Install Git LFS locally
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.csv"
git lfs track "backend/models/*"
git lfs track "backend/data/*"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"

# Push large files
git add backend/models/*.pkl backend/data/*.csv
git commit -m "Add models and datasets via LFS"
git push
```

**Pros**: 
- Seamless Git workflow
- Version control for models
- GitHub Actions compatible

**Cons**: 
- GitHub LFS: 1 GB free, then $5/50GB/month
- Bandwidth charges apply

#### **Strategy 2: AWS S3 Storage**
Store models and datasets in S3, download during deployment:

```bash
# Upload to S3
aws s3 cp backend/models/ s3://your-bucket/models/ --recursive
aws s3 cp backend/data/ s3://your-bucket/data/ --recursive

# Download on server (in deployment script)
aws s3 sync s3://your-bucket/models/ /app/backend/models/
aws s3 sync s3://your-bucket/data/ /app/backend/data/
```

**Pros**: 
- Cost-effective ($0.023/GB/month)
- No Git repo bloat
- Fast downloads within AWS

**Cons**: 
- Requires AWS CLI setup
- Extra deployment step

#### **Strategy 3: Train on Server**
Don't commit models, train them on first deployment:

```yaml
# In docker-compose.yml
services:
  model-trainer:
    profiles:
      - training
    # ... training configuration
```

```bash
# On server, first time only
docker-compose --profile training up model-trainer
```

**Pros**: 
- No large file transfers
- Always fresh models

**Cons**: 
- Requires compute resources
- Longer initial deployment

### **Recommended Approach**

For your project, I recommend **Strategy 1 (Git LFS)** for models < 2GB, and **Strategy 2 (S3)** for datasets > 1GB:

```
Git LFS: *.pkl files (models)
AWS S3: *.csv files (datasets)
```

---

## 3. Initial Server Setup

### Step 1: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Configure Instance**:
   - **Name**: `life-expectancy-app`
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance Type**: `t3.medium` (or as per your choice)
   - **Key Pair**: Create new or use existing (download `.pem` file)
   - **Network Settings**:
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere
     - Allow HTTPS (port 443) from anywhere
     - Allow Custom TCP (port 8000) from anywhere (for API)
   - **Storage**: 50 GB gp3 EBS

3. **Launch Instance** and note the **Public IP**

### Step 2: Connect to Server

```bash
# Set permissions for key file
chmod 400 ~/Downloads/your-key.pem

# Connect via SSH
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 3: Install Docker & Docker Compose

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version

# Logout and login again for group changes
exit
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 4: Install Git & AWS CLI (Optional)

```bash
# Install Git
sudo apt install git -y

# Install AWS CLI (if using S3)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
# Enter: Access Key, Secret Key, Region (e.g., us-east-1), Output format (json)
```

### Step 5: Setup Application Directory

```bash
# Create app directory
mkdir -p ~/app
cd ~/app

# Clone your repository
git clone https://github.com/YOUR_USERNAME/deathnote.git
cd deathnote
```

---

## 4. Docker Deployment via SSH

### Manual Deployment Steps

#### Step 1: Prepare Environment Variables

```bash
# Create backend .env file
cat > backend/.env << 'EOF'
ENVIRONMENT=production
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379/0
MODEL_PATH=/app/models
DATA_PATH=/app/data
EOF
```

#### Step 2: Download Models & Data

**If using Git LFS**:
```bash
# Pull LFS files
git lfs pull
```

**If using S3**:
```bash
# Download from S3
aws s3 sync s3://your-bucket/models/ backend/models/
aws s3 sync s3://your-bucket/data/ backend/data/
```

**If training on server**:
```bash
# Train models (first time only)
docker-compose --profile training up model-trainer
```

#### Step 3: Build and Run Containers

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Step 4: Verify Deployment

```bash
# Test backend health
curl http://localhost:8000/api/v1/health

# Test frontend
curl http://localhost:80

# From your local machine
curl http://YOUR_EC2_PUBLIC_IP:8000/api/v1/health
```

### Deployment Script

Create a deployment script for easy updates:

```bash
# Create deploy.sh
cat > deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main

# Pull LFS files if using Git LFS
if command -v git-lfs &> /dev/null; then
    echo "📦 Pulling LFS files..."
    git lfs pull
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

# Show status
echo "📊 Container status:"
docker-compose ps

# Show logs
echo "📝 Recent logs:"
docker-compose logs --tail=50

echo "✨ Deployment complete!"
EOF

chmod +x deploy.sh
```

**Usage**:
```bash
./deploy.sh
```

---

## 5. GitHub Actions CI/CD Pipeline

### Step 1: Setup GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `EC2_HOST` | Your EC2 public IP | Server address |
| `EC2_USERNAME` | `ubuntu` | SSH username |
| `EC2_SSH_KEY` | Contents of `.pem` file | Private SSH key |
| `AWS_ACCESS_KEY_ID` | Your AWS key | For S3 access (optional) |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret | For S3 access (optional) |

### Step 2: Create GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS EC2

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  deploy:
    name: Deploy Application
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          lfs: true  # Enable Git LFS
      
      - name: Checkout LFS objects
        run: git lfs pull
      
      - name: Setup SSH
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.EC2_SSH_KEY }}" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          ssh-keyscan -H ${{ secrets.EC2_HOST }} >> ~/.ssh/known_hosts
      
      - name: Deploy to EC2
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_USERNAME: ${{ secrets.EC2_USERNAME }}
        run: |
          ssh -i ~/.ssh/deploy_key ${EC2_USERNAME}@${EC2_HOST} << 'ENDSSH'
            set -e
            
            # Navigate to app directory
            cd ~/app/deathnote
            
            # Pull latest changes
            echo "📥 Pulling latest code..."
            git pull origin main
            
            # Pull LFS files
            echo "📦 Pulling LFS files..."
            git lfs pull
            
            # Rebuild and restart containers
            echo "🔨 Rebuilding containers..."
            docker-compose build
            
            echo "🛑 Stopping old containers..."
            docker-compose down
            
            echo "✅ Starting new containers..."
            docker-compose up -d
            
            # Health check
            echo "🏥 Running health check..."
            sleep 10
            curl -f http://localhost:8000/api/v1/health || exit 1
            
            echo "✨ Deployment successful!"
          ENDSSH
      
      - name: Cleanup
        if: always()
        run: rm -f ~/.ssh/deploy_key
      
      - name: Notify deployment status
        if: always()
        run: |
          if [ ${{ job.status }} == 'success' ]; then
            echo "✅ Deployment succeeded!"
          else
            echo "❌ Deployment failed!"
            exit 1
          fi
```

### Step 3: Advanced CI/CD with Testing

For production-grade deployments with tests:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v
      
      - name: Run linting
        run: |
          cd backend
          flake8 app/ --max-line-length=120
  
  build:
    name: Build Docker Images
    needs: test
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      
      - name: Build backend
        run: docker build -t life-expectancy-backend ./backend
      
      - name: Build frontend
        run: docker build -t life-expectancy-frontend ./frontend
  
  deploy:
    name: Deploy to Production
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      # ... (same as previous deploy job)
```

### Step 4: Setup Deployment Notifications (Optional)

Add Slack/Discord notifications:

```yaml
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Deployment to AWS EC2'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 6. Monitoring & Maintenance

### Setup Logging

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend

# Save logs to file
docker-compose logs > deployment.log
```

### Setup Monitoring with Portainer (Optional)

```bash
# Install Portainer
docker volume create portainer_data
docker run -d -p 9000:9000 --name portainer --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce

# Access at http://YOUR_EC2_IP:9000
```

### Backup Strategy

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=~/backups/$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup models
cp -r ~/app/deathnote/backend/models $BACKUP_DIR/

# Backup data
cp -r ~/app/deathnote/backend/data $BACKUP_DIR/

# Backup to S3
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/backups/

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /home/ubuntu/backup.sh
```

### SSL/HTTPS Setup with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### Resource Monitoring

```bash
# Check disk usage
df -h

# Check memory
free -h

# Check Docker stats
docker stats

# Check container health
docker-compose ps
```

---

## Quick Reference Commands

```bash
# Deploy/Update application
./deploy.sh

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Start services
docker-compose up -d

# Rebuild specific service
docker-compose build backend
docker-compose up -d backend

# Access container shell
docker-compose exec backend bash

# Check health
curl http://localhost:8000/api/v1/health
```

---

## Troubleshooting

### Issue: Port already in use
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Issue: Out of disk space
```bash
# Clean Docker
docker system prune -a --volumes

# Clean old images
docker image prune -a
```

### Issue: Container won't start
```bash
# Check logs
docker-compose logs backend

# Check container status
docker-compose ps

# Restart container
docker-compose restart backend
```

### Issue: Git LFS bandwidth limit
```bash
# Use S3 instead
aws s3 sync s3://your-bucket/models/ backend/models/
```

---

## Cost Optimization Tips

1. **Use Reserved Instances**: Save up to 72% for 1-3 year commitments
2. **Stop instance when not needed**: Use AWS Instance Scheduler
3. **Use S3 Intelligent-Tiering**: Automatic cost optimization for data
4. **Enable EBS snapshots**: Cheaper than keeping full volumes
5. **Use CloudWatch alarms**: Monitor and optimize resource usage

---

## Security Best Practices

1. ✅ Use SSH keys, not passwords
2. ✅ Restrict Security Group to specific IPs
3. ✅ Enable AWS CloudTrail for audit logs
4. ✅ Use IAM roles instead of access keys when possible
5. ✅ Enable EBS encryption
6. ✅ Regular security updates: `sudo apt update && sudo apt upgrade`
7. ✅ Use HTTPS with SSL certificates
8. ✅ Implement rate limiting in API
9. ✅ Use environment variables for secrets
10. ✅ Regular backups to S3

---

## Next Steps

1. ✅ Launch EC2 instance
2. ✅ Setup SSH access
3. ✅ Install Docker & Docker Compose
4. ✅ Clone repository
5. ✅ Configure environment variables
6. ✅ Deploy application
7. ✅ Setup GitHub Actions
8. ✅ Configure domain & SSL
9. ✅ Setup monitoring
10. ✅ Configure backups

---

**Need Help?** Check the logs, review this guide, or consult AWS documentation.

**Happy Deploying! 🚀**
