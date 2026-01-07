# Life Expectancy Prediction - Docker Deployment Guide

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

Run the automated deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

This script will:
1. Train the ML model inside a Docker container
2. Build all Docker images (backend, frontend)
3. Start all services (backend, frontend, Redis)
4. Display service URLs and status

### Option 2: Manual Step-by-Step Deployment

#### Step 1: Train the Model

First, train the machine learning model inside a Docker container:

```bash
# Build and run the model training service
docker compose --profile training up model-trainer --build
```

This will:
- Create the models directory if it doesn't exist
- Train multiple ML models (Random Forest, Gradient Boosting, XGBoost)
- Save the best model to `backend/models/best_model.pkl`
- Save preprocessor and metadata

**Expected output:** You should see model training logs and "MODEL TRAINING COMPLETED SUCCESSFULLY!"

#### Step 2: Build All Services

Build the Docker images for backend and frontend:

```bash
docker compose build
```

#### Step 3: Start All Services

Start the backend, frontend, and Redis:

```bash
docker compose up -d
```

Or run in foreground to see logs:

```bash
docker compose up
```

## 🌐 Access the Application

Once deployed, access the application at:

- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## 📊 Service Architecture

```
┌─────────────────┐
│    Frontend     │  Port 80 (Nginx + React)
│   (Vite/React)  │
└────────┬────────┘
         │
         │ /api/* requests
         ▼
┌─────────────────┐
│     Backend     │  Port 8000 (FastAPI)
│    (FastAPI)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Redis      │  Port 6379 (Caching)
└─────────────────┘
```

## 🛠️ Useful Commands

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f redis
```

### Service Management

```bash
# Stop all services
docker compose down

# Restart all services
docker compose restart

# Restart specific service
docker compose restart backend

# View running containers
docker compose ps

# View resource usage
docker stats
```

### Model Management

```bash
# Retrain the model
docker compose --profile training up model-trainer

# Rebuild and retrain
docker compose --profile training up model-trainer --build

# Check model files
ls -lh backend/models/
```

### Rebuild Services

```bash
# Rebuild everything
docker compose down
docker compose build --no-cache
docker compose up -d

# Rebuild specific service
docker compose build backend --no-cache
docker compose up -d backend
```

## 🔍 Troubleshooting

### Model Training Failed

1. Check if data exists:
   ```bash
   ls -lh backend/data/
   ```

2. View training logs:
   ```bash
   docker compose --profile training up model-trainer
   ```

3. Check permissions:
   ```bash
   chmod -R 755 backend/models
   ```

### Backend Not Starting

1. Check if model exists:
   ```bash
   ls backend/models/best_model.pkl
   ```

2. Check backend logs:
   ```bash
   docker compose logs backend
   ```

3. Verify Redis is running:
   ```bash
   docker compose ps redis
   ```

### Frontend Can't Connect to Backend

1. Check nginx configuration:
   ```bash
   docker compose exec frontend cat /etc/nginx/conf.d/default.conf
   ```

2. Test backend directly:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

3. Check Docker network:
   ```bash
   docker network inspect deathnote_app-network
   ```

### Port Conflicts

If ports 80, 8000, or 6379 are already in use, modify `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Use port 8001 on host
  
  frontend:
    ports:
      - "8080:80"    # Use port 8080 on host
```

## 📁 Project Structure

```
.
├── deploy.sh                    # Automated deployment script
├── docker-compose.yml          # Main orchestration file
├── backend/
│   ├── Dockerfile              # Backend container image
│   ├── train_model.py          # Model training script
│   ├── requirements.txt        # Python dependencies
│   ├── app/                    # FastAPI application
│   ├── data/                   # Training datasets
│   ├── models/                 # Trained ML models (created by training)
│   └── logs/                   # Application logs
└── frontend/
    ├── Dockerfile              # Frontend container image
    ├── nginx.conf              # Nginx configuration
    ├── package.json            # Node dependencies
    └── src/                    # React application
```

## 🔐 Environment Configuration

Create `backend/.env` file:

```env
# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Redis
REDIS_URL=redis://redis:6379/0

# Security (generate random keys)
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# CORS
ALLOWED_ORIGINS=http://localhost,http://localhost:80
```

Generate random keys:
```bash
# Secret key
openssl rand -hex 32

# API key
openssl rand -hex 16
```

## 📈 Testing the Application

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-02T..."
}
```

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Status": "Developed",
    "Life_expectancy": 75.5,
    "Adult_Mortality": 120,
    "infant_deaths": 5,
    "Alcohol": 8.5,
    "percentage_expenditure": 1000,
    "Hepatitis_B": 90,
    "Measles": 100,
    "BMI": 25.5,
    "under_five_deaths": 8,
    "Polio": 95,
    "Total_expenditure": 6.5,
    "Diphtheria": 92,
    "HIV_AIDS": 0.1,
    "GDP": 15000,
    "Population": 5000000,
    "thinness_1_19_years": 3.5,
    "thinness_5_9_years": 3.2,
    "Income_composition_of_resources": 0.75,
    "Schooling": 12.5
  }'
```

### 3. View API Documentation

Open in browser: http://localhost:8000/docs

### 4. Test Frontend

Open in browser: http://localhost

## 🧹 Cleanup

### Remove All Containers and Volumes

```bash
docker compose down -v
```

### Remove Images

```bash
docker compose down
docker rmi life-expectancy-backend life-expectancy-frontend
```

### Clean Everything (including models)

```bash
docker compose down -v
rm -rf backend/models/*
rm -rf backend/logs/*
```

## 🔄 Development Workflow

### Making Backend Changes

```bash
# Edit code in backend/
# Rebuild and restart
docker compose build backend
docker compose up -d backend
docker compose logs -f backend
```

### Making Frontend Changes

```bash
# Edit code in frontend/src/
# Rebuild and restart
docker compose build frontend
docker compose up -d frontend
```

### Retraining the Model

```bash
# After updating training data or scripts
docker compose --profile training up model-trainer --build
docker compose restart backend
```

## 📊 Monitoring

### View Metrics

```bash
# System metrics
docker stats

# Application health
watch -n 5 'curl -s http://localhost:8000/api/v1/health | jq'
```

### View Logs in Real-time

```bash
# All services with timestamps
docker compose logs -f --timestamps

# Specific service
docker compose logs -f backend --tail=100
```

## 🎯 Production Considerations

For production deployment:

1. **Use proper secrets management** (not .env files)
2. **Enable HTTPS** (configure SSL certificates in nginx)
3. **Set up proper monitoring** (Prometheus, Grafana)
4. **Use production-grade Redis** (persistent storage)
5. **Implement rate limiting** (already configured in backend)
6. **Set up log aggregation** (ELK stack, CloudWatch)
7. **Use Docker Swarm or Kubernetes** for orchestration

## 📝 License

See main project LICENSE file.
