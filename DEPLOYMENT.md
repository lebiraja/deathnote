# Deployment Guide — Life Expectancy Prediction App

This guide covers everything you need to deploy this project — from training and managing the ML model to running in production on AWS EC2 with Docker.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [ML Model Management](#2-ml-model-management)
3. [Docker & Docker Compose](#3-docker--docker-compose)
4. [Environment Variables](#4-environment-variables)
5. [Local Development Setup](#5-local-development-setup)
6. [AWS EC2 Deployment](#6-aws-ec2-deployment)
7. [CI/CD with GitHub Actions](#7-cicd-with-github-actions)
8. [Deploying a New or Updated Model](#8-deploying-a-new-or-updated-model)
9. [Health Checks & Monitoring](#9-health-checks--monitoring)
10. [SSL / HTTPS Setup](#10-ssl--https-setup)
11. [Scaling & Performance](#11-scaling--performance)
12. [Troubleshooting](#12-troubleshooting)
13. [Best Practices Checklist](#13-best-practices-checklist)

---

## 1. Project Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    AWS EC2 Instance                  │
│                                                     │
│  ┌──────────┐    ┌──────────────┐   ┌───────────┐  │
│  │ Frontend │───▶│   Backend    │──▶│   Redis   │  │
│  │  (nginx) │    │  (FastAPI)   │   │  (Cache)  │  │
│  │  :80     │    │  :8000       │   │  :6379    │  │
│  └──────────┘    └──────┬───────┘   └───────────┘  │
│                         │                           │
│                  ┌──────▼───────┐                   │
│                  │  ML Models   │                   │
│                  │ /app/models/ │                   │
│                  │  *.pkl files │                   │
│                  └──────────────┘                   │
└─────────────────────────────────────────────────────┘
```

**Services:**
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `frontend` | Node 20 + nginx:alpine | 80 | React UI served via nginx |
| `backend` | Python 3.11-slim | 8000 | FastAPI inference server |
| `redis` | redis:7-alpine | 6380 (host) | Prediction cache (5-min TTL) |
| `model-trainer` | Same as backend | — | One-off model training job |

**ML Stack:**
- Framework: scikit-learn (Gradient Boosting Regressor)
- Serialization: joblib (`.pkl` files)
- Models live at: `backend/models/`
  - `gradient_boosting_model.pkl` — primary model (~470 KB)
  - `scaler.pkl` — StandardScaler (~1.4 KB)
  - `preprocessor.pkl` — LabelEncoders (~1.9 KB)

---

## 2. ML Model Management

### 2.1 Model Files & What They Do

```
backend/models/
├── gradient_boosting_model.pkl   ← Primary production model
├── scaler.pkl                    ← Scales numeric features before inference
├── preprocessor.pkl              ← Encodes categorical features
├── linear_model.pkl              ← Backup/fallback model
└── random_forest_model.pkl       ← Backup/fallback model
```

All three files (`gradient_boosting_model.pkl`, `scaler.pkl`, `preprocessor.pkl`) are **required** for inference. If any is missing, the API will fail to start.

### 2.2 Training the Model

**Option A — Train inside Docker (recommended for reproducibility):**
```bash
# Ensure your training data is in backend/data/
# Accepts any of: life-expectancy-merged.csv, life-expectancy-enhanced.csv, life-expectancy.csv
docker compose --profile training run --rm model-trainer
```
This runs `train_model.py` inside a container and writes `.pkl` files to `backend/models/` via volume mount.

**Option B — Train locally:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

After training, verify the files exist:
```bash
ls -lh backend/models/
```

### 2.3 Model Versioning — Best Practice

Never overwrite your production model in place without a backup. Use this pattern:

```bash
# 1. Backup current production model
cp backend/models/gradient_boosting_model.pkl \
   backend/models/gradient_boosting_model_v1.0.0.pkl

# 2. Train new model (outputs to models/ directory)
docker compose --profile training run --rm model-trainer

# 3. Evaluate the new model before deploying
# (check the R², RMSE, MAE printed at end of training)

# 4. The new model is now at gradient_boosting_model.pkl
# If something goes wrong, roll back:
cp backend/models/gradient_boosting_model_v1.0.0.pkl \
   backend/models/gradient_boosting_model.pkl
```

**Naming convention for versioned models:**
```
gradient_boosting_model_v<MAJOR>.<MINOR>.<PATCH>_<YYYYMMDD>.pkl
# Example:
gradient_boosting_model_v1.2.0_20260411.pkl
```

### 2.4 Storing Models — Git vs S3

The `.pkl` files are small (~470 KB total) and are tracked in git. This works fine for this project. For larger models:

**If model grows > 50 MB — use S3:**
```bash
# Upload to S3
aws s3 cp backend/models/ s3://your-bucket/models/ --recursive

# Download on EC2 (add to deploy.sh)
aws s3 cp s3://your-bucket/models/ ~/deathnote/backend/models/ --recursive
```

**If model grows > 2 GB — use Git LFS:**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "track .pkl files with LFS"
```

### 2.5 How the Model Is Loaded at Startup

The API loads all model artifacts once at startup via the FastAPI lifespan hook:

```
app startup
    └── ModelManager.load_models()
            ├── joblib.load("/app/models/gradient_boosting_model.pkl")
            ├── joblib.load("/app/models/scaler.pkl")
            └── joblib.load("/app/models/preprocessor.pkl")
```

Models are cached in memory for the lifetime of the container. The `/api/v1/health` endpoint reports `model_loaded: true/false`.

---

## 3. Docker & Docker Compose

### 3.1 Docker Compose File Explained

```yaml
# docker-compose.yml (top-level — production)

services:
  model-trainer:         # Only runs when you explicitly trigger it
    profiles: [training]

  backend:               # FastAPI API server
    ports: ["8000:8000"]
    volumes:
      - ./backend/models:/app/models:ro   # read-only — model files
      - ./backend/data:/app/data:ro
      - ./backend/logs:/app/logs
    depends_on:
      redis: { condition: service_healthy }

  frontend:              # React app served by nginx
    ports: ["80:80"]
    depends_on: [backend]

  redis:                 # Cache layer
    ports: ["6380:6379"] # host:container (6380 avoids conflicts)
    command: redis-server --appendonly yes   # persistent storage
```

### 3.2 Key Docker Commands

```bash
# Build all images
docker compose build

# Start all services (detached)
docker compose up -d

# Start fresh (no cache — use on code changes)
docker compose build --no-cache && docker compose up -d

# Stop everything
docker compose down

# Stop and remove volumes (wipes Redis cache)
docker compose down -v

# Train the model (one-off)
docker compose --profile training run --rm model-trainer

# View logs
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend

# Restart a single service
docker compose restart backend

# Enter a running container
docker compose exec backend bash
docker compose exec redis redis-cli
```

### 3.3 Backend Dockerfile — How It Works

The backend uses a **two-stage build** to keep the final image small:

```
Stage 1 (builder):
  python:3.11-slim
  └── pip install -r requirements.txt → /root/.local

Stage 2 (production):
  python:3.11-slim
  ├── Install runtime system libs (OpenBLAS, LAPACK — needed by numpy/scikit-learn)
  ├── Copy /root/.local from builder
  ├── Copy application code
  ├── Create non-root user (appuser)
  └── CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Why OpenBLAS/LAPACK?** scikit-learn uses BLAS for matrix operations. Without these, model inference will either fail or be slow.

### 3.4 Frontend Dockerfile — How It Works

```
Stage 1 (builder):
  node:20-alpine
  └── npm ci && npm run build → /app/dist

Stage 2 (production):
  nginx:alpine
  ├── Copy /app/dist → /usr/share/nginx/html
  └── nginx proxies /api/* → http://backend:8000
```

The nginx config handles two jobs:
1. Serve the built React app as static files
2. Proxy all `/api/*` requests to the backend container

---

## 4. Environment Variables

### 4.1 Creating the Backend `.env` File

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Security — CHANGE THIS IN PRODUCTION
SECRET_KEY=your-very-long-random-secret-key-here

# Model paths (inside container — do not change)
MODEL_PATH=/app/models/gradient_boosting_model.pkl
SCALER_PATH=/app/models/scaler.pkl
PREPROCESSOR_PATH=/app/models/preprocessor.pkl
MODEL_VERSION=1.0.0

# Redis (matches docker-compose service name)
REDIS_URL=redis://redis:6379/0
CACHE_ENABLED=true
CACHE_TTL=300

# CORS — add your domain here for production
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# Rate limiting
API_RATE_LIMIT=100/hour

# Optional — Sentry error tracking
# SENTRY_DSN=https://xxx@sentry.io/xxx
```

### 4.2 Generating a Secure Secret Key

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 4.3 Critical: Never Commit `.env` to Git

Verify `.env` is in `.gitignore`:
```bash
grep ".env" .gitignore
```
If it's not there, add it:
```bash
echo "backend/.env" >> .gitignore
```

---

## 5. Local Development Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd deathnote

# 2. Create environment file
cp backend/.env.example backend/.env
# Edit SECRET_KEY in backend/.env

# 3. Make sure model files exist
ls backend/models/
# If empty, train the model first:
docker compose --profile training run --rm model-trainer

# 4. Start all services
docker compose up -d

# 5. Check everything is running
docker compose ps

# 6. Verify health
curl http://localhost:8000/api/v1/health

# 7. Open the app
open http://localhost:80
```

**Frontend dev (hot reload):**
```bash
cd frontend
npm install
npm run dev
# App available at http://localhost:5173
# API calls proxied to http://localhost:8000 via vite.config.ts
```

---

## 6. AWS EC2 Deployment

### 6.1 EC2 Instance Recommendations

| Use Case | Instance Type | vCPU | RAM | Cost (est.) |
|----------|--------------|------|-----|-------------|
| Development / Testing | t3.small | 2 | 2 GB | ~$15/mo |
| Production (this app) | t3.medium | 2 | 4 GB | ~$30/mo |
| High Traffic | t3.large | 2 | 8 GB | ~$60/mo |

**Storage:** 30 GB EBS gp3 minimum. Use 50 GB if you plan to store training data on disk.

**AMI:** Ubuntu 22.04 LTS (recommended)

### 6.2 Security Group Rules

| Type | Protocol | Port | Source | Purpose |
|------|----------|------|--------|---------|
| SSH | TCP | 22 | Your IP only | Remote access |
| HTTP | TCP | 80 | 0.0.0.0/0 | Web traffic |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Secure web traffic |
| Custom | TCP | 8000 | 0.0.0.0/0 | Direct API access (optional) |

> Do NOT expose port 6380 (Redis) to the internet.

### 6.3 Initial Server Setup (One-Time)

SSH into your EC2 instance and run these steps once:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add ubuntu user to docker group (no sudo needed)
sudo usermod -aG docker ubuntu
newgrp docker

# Verify
docker --version
docker compose version

# Install Git
sudo apt-get install -y git

# Clone the repo
git clone <your-repo-url> ~/deathnote
cd ~/deathnote
```

### 6.4 Deploying the App

```bash
cd ~/deathnote

# Set up environment file
cp backend/.env.example backend/.env
nano backend/.env   # Set SECRET_KEY, CORS_ORIGINS with your domain

# If model files are not in the repo (or need re-training):
docker compose --profile training run --rm model-trainer

# Verify model files exist
ls -lh backend/models/

# Build and start all services
docker compose build --no-cache
docker compose up -d

# Check service status
docker compose ps

# Check health
curl http://localhost:8000/api/v1/health
curl http://localhost/
```

### 6.5 Updating the App (Manual)

```bash
cd ~/deathnote
git pull origin main
docker compose build --no-cache
docker compose down && docker compose up -d
docker compose ps
curl http://localhost:8000/api/v1/health
```

---

## 7. CI/CD with GitHub Actions

### 7.1 How It Works

Every push to `main` triggers `.github/workflows/ci-cd.yml`:

```
Push to main branch
    └── GitHub Actions runner (ubuntu-latest)
            ├── Checkout code
            ├── Setup SSH key from secrets
            └── SSH into EC2
                    ├── cd ~/deathnote
                    ├── git pull origin main
                    ├── docker-compose down
                    ├── docker-compose build --no-cache
                    └── docker-compose up -d
```

### 7.2 Required GitHub Secrets

Go to: **Repository → Settings → Secrets and variables → Actions**

| Secret Name | Value | How to Get It |
|-------------|-------|---------------|
| `EC2_SSH_KEY` | Contents of your `.pem` key file | `cat your-key.pem` |
| `EC2_HOST` | EC2 public IP address | AWS Console → EC2 → Instances |
| `EC2_USERNAME` | `ubuntu` | Default for Ubuntu AMIs |

**Setting the SSH key:**
```bash
# On your local machine, copy the .pem file content
cat your-key.pem
# Paste the ENTIRE output (including -----BEGIN/END RSA PRIVATE KEY-----) as EC2_SSH_KEY
```

### 7.3 Important: Model Files and CI/CD

The CI/CD pipeline does `git pull` — it will only update files tracked in git.

- If model `.pkl` files are committed to git: they will be updated automatically.
- If model `.pkl` files are NOT in git (stored in S3/LFS): add a download step to the workflow.

**Adding S3 model download to CI/CD:**
```yaml
# Add this step before docker-compose build in ci-cd.yml
- name: Download models from S3
  run: |
    ssh ${EC2_USERNAME}@${EC2_HOST} << 'EOF'
      aws s3 cp s3://your-bucket/models/ ~/deathnote/backend/models/ --recursive
    EOF
```

---

## 8. Deploying a New or Updated Model

This is the most important operational procedure. Follow these steps every time you have a new model version.

### 8.1 Standard Model Update Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Train new model locally or in Docker                │
│  2. Evaluate — confirm metrics are better or acceptable  │
│  3. Backup current production model                     │
│  4. Replace model files in backend/models/              │
│  5. Commit to git (or upload to S3)                     │
│  6. Push to main → CI/CD deploys                        │
│  7. Verify health endpoint after deploy                 │
│  8. Monitor error logs for 15 minutes                   │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Step-by-Step Model Deployment

**Step 1: Train and evaluate**
```bash
# Train (inside Docker for reproducibility)
docker compose --profile training run --rm model-trainer

# Output will show metrics — confirm R² is acceptable
# Example output:
# gradient_boosting:
#   RMSE: 2.1234
#   MAE:  1.6543
#   R²:   0.9456
```

**Step 2: Backup and version the old model**
```bash
VERSION="v1.0.0"
DATE=$(date +%Y%m%d)

cd backend/models
cp gradient_boosting_model.pkl gradient_boosting_model_${VERSION}_${DATE}.pkl
cp scaler.pkl scaler_${VERSION}_${DATE}.pkl
cp preprocessor.pkl preprocessor_${VERSION}_${DATE}.pkl
```

**Step 3: Commit and push**
```bash
cd /path/to/deathnote
git add backend/models/gradient_boosting_model.pkl
git add backend/models/scaler.pkl
git add backend/models/preprocessor.pkl
git commit -m "model: update to v1.1.0 - R2=0.951"
git push origin main
```
This triggers CI/CD which pulls and rebuilds on EC2.

**Step 4: Verify after deploy**
```bash
# Check health endpoint
curl https://yourdomain.com/api/v1/health
# Expected: {"status": "healthy", "model_loaded": true, ...}

# Test a prediction
curl -X POST https://yourdomain.com/api/v1/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender":"Male","Height":175,"Weight":70,"BMI":22.9,
       "Physical_Activity":3,"Smoking_Status":"Never",
       "Alcohol_Consumption":"Occasional","Diet":"Balanced",
       "Blood_Pressure":"Normal","Cholesterol":"Normal",
       "Diabetes":0,"Hypertension":0,"Heart_Disease":0,"Asthma":0}'
```

**Step 5: Rollback if needed**
```bash
# On EC2
cd ~/deathnote/backend/models
cp gradient_boosting_model_v1.0.0_20260411.pkl gradient_boosting_model.pkl
cp scaler_v1.0.0_20260411.pkl scaler.pkl
cp preprocessor_v1.0.0_20260411.pkl preprocessor.pkl

# Restart backend to reload the model
docker compose restart backend

# Verify
curl http://localhost:8000/api/v1/health
```

### 8.3 Zero-Downtime Model Reload

The current setup requires a container restart to load a new model (since models are loaded at startup). For zero-downtime reloads in future iterations, consider:

1. **Polling approach**: Have `ModelManager` check a version file every N minutes and reload if version changed.
2. **Hot reload endpoint**: Add an authenticated `POST /api/v1/admin/reload-model` endpoint.
3. **Blue/Green deployment**: Run two backends, switch nginx upstream.

---

## 9. Health Checks & Monitoring

### 9.1 Health Endpoints

```bash
# Basic health check (used by Docker healthcheck)
GET /api/v1/health
# Response: {"status": "healthy", "version": "1.0.0", "model_loaded": true}

# Detailed system health
GET /api/v1/health/detailed
# Response: includes CPU%, memory%, disk%, model status
```

### 9.2 Checking Service Status

```bash
# All service statuses
docker compose ps

# Container resource usage (CPU, memory)
docker stats

# Recent logs (last 100 lines)
docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
docker compose logs --tail=100 redis

# Follow logs in real-time
docker compose logs -f backend

# Application log file (inside container)
docker compose exec backend cat /app/logs/app.log
```

### 9.3 Key Metrics to Watch

| Metric | Where to Check | Alert Threshold |
|--------|---------------|-----------------|
| Backend health | `GET /api/v1/health` | `model_loaded: false` |
| Memory usage | `docker stats` | > 80% of instance RAM |
| Disk usage | `df -h` | > 80% |
| API errors | `docker compose logs backend` | Any 500 errors |
| Redis | `docker compose exec redis redis-cli ping` | Not returning PONG |

### 9.4 Useful One-Liners

```bash
# Count prediction errors in logs
docker compose logs backend | grep "ERROR" | wc -l

# Check Redis cache hit rate
docker compose exec redis redis-cli INFO stats | grep keyspace_hits

# Check backend memory usage
docker stats life-expectancy-backend --no-stream

# Disk space check
df -h /
```

---

## 10. SSL / HTTPS Setup

### 10.1 Using Certbot (Let's Encrypt) — Free SSL

> Requires a domain name pointing to your EC2 IP.

```bash
# On EC2
sudo apt-get install -y certbot python3-certbot-nginx

# Stop nginx temporarily (port 80 must be free)
docker compose down frontend

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Certificates are saved to:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

### 10.2 Update nginx to Use SSL

Add to `frontend/nginx.conf`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # ... rest of existing config
}
```

Mount certs in `docker-compose.yml` frontend service:
```yaml
frontend:
  volumes:
    - /etc/letsencrypt:/etc/letsencrypt:ro
  ports:
    - "80:80"
    - "443:443"
```

### 10.3 Update CORS for HTTPS

In `backend/.env`:
```env
CORS_ORIGINS=["https://yourdomain.com","https://www.yourdomain.com"]
```

### 10.4 Auto-Renew SSL Certificate

```bash
# Test renewal
sudo certbot renew --dry-run

# Auto-renew is handled by systemd timer — verify it's active
sudo systemctl status certbot.timer

# After renewal, restart nginx
sudo certbot renew --post-hook "docker compose -f ~/deathnote/docker-compose.yml restart frontend"
```

---

## 11. Scaling & Performance

### 11.1 Scaling the Backend (Multiple Workers)

Uvicorn is already configured for 4 workers in `config.py`. To increase:

```yaml
# docker-compose.yml — backend command override
backend:
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 8
```

**Rule of thumb:** `workers = (2 × CPU cores) + 1`

For a t3.medium (2 vCPU): `workers = 5`

### 11.2 Scaling with Multiple Backend Replicas

```yaml
# docker-compose.yml
backend:
  deploy:
    replicas: 2
```

> Note: If using replicas, remove the static `container_name` from the backend service, as it conflicts with multiple instances.

### 11.3 Redis Cache Tuning

The cache stores prediction results for 5 minutes (TTL=300). Identical inputs return cached results without running the model.

```bash
# Check cache size
docker compose exec redis redis-cli DBSIZE

# Clear cache (force fresh predictions)
docker compose exec redis redis-cli FLUSHDB

# Monitor cache in real-time
docker compose exec redis redis-cli MONITOR
```

### 11.4 Instance Sizing

If you see high memory usage or slow responses:

```bash
# Check current usage
docker stats --no-stream

# Upgrade the EC2 instance type (stop → change type → start)
# t3.small → t3.medium → t3.large
```

---

## 12. Troubleshooting

### Backend Won't Start

**Symptom:** `docker compose ps` shows backend as unhealthy or restarting.

```bash
# Check logs
docker compose logs backend

# Common causes:
# 1. Model file missing
ls backend/models/gradient_boosting_model.pkl

# 2. .env file missing
ls backend/.env

# 3. Redis not ready yet (usually resolves on its own)
docker compose logs redis

# Fix: restart with fresh build
docker compose down && docker compose up -d --build
```

### Model Not Loading

**Symptom:** `GET /api/v1/health` returns `model_loaded: false`.

```bash
# Check model files exist inside the container
docker compose exec backend ls -la /app/models/

# Check the volume mount is correct
docker compose config | grep -A5 volumes

# If files are there but model still fails to load:
docker compose logs backend | grep "ERROR\|model\|pkl"
```

### Prediction Endpoint Returns 500

```bash
# Get the full error
docker compose logs backend | grep -A10 "500\|ERROR\|Exception"

# Common causes:
# 1. Input data doesn't match training data format (column names, categories)
# 2. Scaler or preprocessor mismatch with model version
# 3. Memory issue — check docker stats
```

### Frontend Shows Blank Page or API Errors

```bash
# Check nginx is running
docker compose logs frontend

# Test API directly (bypass nginx)
curl http://localhost:8000/api/v1/health

# Check nginx proxy config
docker compose exec frontend cat /etc/nginx/conf.d/default.conf

# Common cause: CORS error — update CORS_ORIGINS in .env to match your domain
```

### Redis Connection Refused

```bash
# Check Redis is healthy
docker compose exec redis redis-cli ping
# Expected: PONG

# Check REDIS_URL in .env matches docker-compose service name
grep REDIS_URL backend/.env
# Should be: REDIS_URL=redis://redis:6379/0

# Restart Redis
docker compose restart redis
```

### Disk Full

```bash
# Check disk usage
df -h /

# Remove unused Docker images/containers
docker system prune -f

# Remove old log files
docker compose exec backend find /app/logs -name "*.log.*" -delete

# Check large files
du -sh backend/models/* backend/data/* 2>/dev/null | sort -rh | head -20
```

### CI/CD Deployment Fails

```bash
# Check GitHub Actions logs in the browser
# Repository → Actions → (failed run)

# Common issues:
# 1. EC2_SSH_KEY has wrong format — must include header/footer lines
# 2. EC2 security group doesn't allow GitHub Actions IPs on port 22
#    Fix: Allow 0.0.0.0/0 on SSH temporarily, or use AWS Systems Manager
# 3. Docker build fails due to disk space on EC2
#    Fix: docker system prune -f on EC2, then re-run
```

---

## 13. Best Practices Checklist

### Before Every Deployment
- [ ] `backend/.env` exists with a real `SECRET_KEY` (not the default)
- [ ] `CORS_ORIGINS` includes your production domain
- [ ] All three model files exist: `gradient_boosting_model.pkl`, `scaler.pkl`, `preprocessor.pkl`
- [ ] `backend/.env` is NOT committed to git

### Model Updates
- [ ] Evaluated new model metrics before deploying
- [ ] Backed up old model files with version suffix
- [ ] Tested prediction endpoint after deployment
- [ ] Monitored logs for 15+ minutes post-deploy

### Security
- [ ] Port 22 (SSH) restricted to known IPs, not 0.0.0.0/0
- [ ] Redis port (6380) NOT exposed publicly
- [ ] HTTPS enabled in production (not just HTTP)
- [ ] `SECRET_KEY` is randomly generated (use `python3 -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] Docker containers run as non-root (`appuser`)

### Reliability
- [ ] `restart: unless-stopped` on all services (already configured)
- [ ] Health checks in place (already configured)
- [ ] Redis persistence enabled — `appendonly yes` (already configured)
- [ ] Log rotation configured — 10 MB max, 5 backups (already configured)

### Maintenance
- [ ] Run `docker system prune -f` monthly to reclaim disk space
- [ ] Monitor disk usage (`df -h`) — alert at 80%
- [ ] Renew SSL certificates before expiry (Let's Encrypt: 90-day validity)
- [ ] Keep a versioned backup of working model files before every update

---

## Quick Reference

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# Train model
docker compose --profile training run --rm model-trainer

# View logs
docker compose logs -f backend

# Health check
curl http://localhost:8000/api/v1/health

# Restart backend only (e.g., after model update)
docker compose restart backend

# Full redeploy
docker compose down && docker compose build --no-cache && docker compose up -d

# Roll back model
cp backend/models/gradient_boosting_model_<version>.pkl \
   backend/models/gradient_boosting_model.pkl && \
docker compose restart backend
```
