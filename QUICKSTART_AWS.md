# Quick Start - AWS Deployment

This is a condensed guide to get your Life Expectancy Prediction app deployed to AWS quickly. For detailed explanations, see [AWS_DEPLOYMENT_GUIDE.md](./AWS_DEPLOYMENT_GUIDE.md).

## Prerequisites

- AWS Account
- GitHub repository with your code
- Local machine with Git and SSH

## Step 1: Launch EC2 Instance (5 minutes)

1. Go to **AWS Console** → **EC2** → **Launch Instance**
2. Configure:
   - **Name**: `life-expectancy-app`
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance Type**: `t3.medium`
   - **Key Pair**: Create new (download `.pem` file)
   - **Security Group**: Allow ports 22, 80, 443, 8000
   - **Storage**: 50 GB gp3
3. Click **Launch Instance**
4. Note your **Public IP address**

## Step 2: Connect & Setup Server (10 minutes)

```bash
# Connect to server
chmod 400 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again
exit
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_IP

# Install Git
sudo apt update && sudo apt install git -y

# Clone repository
mkdir -p ~/app && cd ~/app
git clone https://github.com/YOUR_USERNAME/deathnote.git
cd deathnote
```

## Step 3: Handle Large Files (Choose One)

### Option A: Git LFS (Recommended for models < 2GB)

**On your local machine:**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.csv"

# Commit and push
git add .gitattributes
git add backend/models/*.pkl backend/data/*.csv
git commit -m "Add models via LFS"
git push
```

**On server:**
```bash
# Pull LFS files
git lfs pull
```

### Option B: AWS S3 (Recommended for datasets > 1GB)

**On your local machine:**
```bash
# Install AWS CLI
# (See AWS_DEPLOYMENT_GUIDE.md for installation)

# Configure AWS
aws configure

# Edit upload-assets.sh and set your bucket name
nano upload-assets.sh
# Change: S3_BUCKET="your-bucket-name"

# Upload to S3
./upload-assets.sh
```

**On server:**
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS
aws configure

# Edit download-assets.sh and set your bucket name
nano download-assets.sh
# Change: S3_BUCKET="your-bucket-name"

# Download from S3
./download-assets.sh
```

### Option C: Train on Server

```bash
# Train models on first deployment
docker-compose --profile training up model-trainer
```

## Step 4: Deploy Application (5 minutes)

```bash
# Create backend environment file
cat > backend/.env << 'EOF'
ENVIRONMENT=production
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379/0
MODEL_PATH=/app/models
DATA_PATH=/app/data
EOF

# Build and start containers
docker-compose build
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Step 5: Verify Deployment (2 minutes)

```bash
# On server
curl http://localhost:8000/api/v1/health
curl http://localhost:80

# From your local machine
curl http://YOUR_EC2_IP:8000/api/v1/health
# Open browser: http://YOUR_EC2_IP
```

✅ **Your app is now live!**

## Step 6: Setup GitHub Actions CI/CD (10 minutes)

### Add GitHub Secrets

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `EC2_HOST` | Your EC2 public IP |
| `EC2_USERNAME` | `ubuntu` |
| `EC2_SSH_KEY` | Contents of your `.pem` file |

### Commit Workflow File

The workflow file is already created at `.github/workflows/deploy.yml`. Just commit and push:

```bash
git add .github/workflows/deploy.yml
git commit -m "Add GitHub Actions deployment workflow"
git push
```

### Test Deployment

Make any change and push to `main` branch:

```bash
echo "# Test" >> README.md
git add README.md
git commit -m "Test auto-deployment"
git push
```

Go to **GitHub** → **Actions** tab to watch the deployment!

## Daily Usage

### Update Application

**Option 1: Automatic (via GitHub Actions)**
```bash
# Just push to main branch
git push origin main
# GitHub Actions will automatically deploy
```

**Option 2: Manual (on server)**
```bash
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_IP
cd ~/app/deathnote
./deploy.sh
```

### View Logs

```bash
docker-compose logs -f
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Restart Services

```bash
docker-compose restart
docker-compose restart backend
```

### Stop Services

```bash
docker-compose down
```

### Start Services

```bash
docker-compose up -d
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs backend
docker-compose ps
docker-compose restart backend
```

### Port already in use
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Out of disk space
```bash
docker system prune -a --volumes
```

## Cost Estimate

- **t3.medium EC2**: ~$30/month
- **50 GB EBS gp3**: ~$4/month
- **Data Transfer**: ~$1-5/month
- **Total**: ~$35-40/month

## Next Steps

- [ ] Setup domain name (Route 53)
- [ ] Configure SSL/HTTPS (Let's Encrypt)
- [ ] Setup monitoring (CloudWatch)
- [ ] Configure backups (S3)
- [ ] Setup alerts (SNS)

---

For detailed explanations and advanced configurations, see [AWS_DEPLOYMENT_GUIDE.md](./AWS_DEPLOYMENT_GUIDE.md).
