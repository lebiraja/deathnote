# Quick Start - Life Expectancy Prediction

## 🎯 One-Command Deployment

```bash
./deploy.sh
```

That's it! The script will:
1. ✅ Train the ML model in a container
2. ✅ Build all Docker images  
3. ✅ Start all services
4. ✅ Show you the URLs

## 🌐 Access Points

- **App**: http://localhost
- **API**: http://localhost:8000/docs
- **Health**: http://localhost:8000/api/v1/health

## 🔧 Common Commands

```bash
# View logs
docker compose logs -f

# Stop everything
docker compose down

# Retrain model
docker compose --profile training up model-trainer

# Restart services
docker compose restart
```

## 📋 Manual Steps (if needed)

```bash
# 1. Train model
docker compose --profile training up model-trainer --build

# 2. Start services
docker compose up -d

# 3. Check status
docker compose ps
```

## ❓ Troubleshooting

**No model found?**
```bash
docker compose --profile training up model-trainer
```

**Services not starting?**
```bash
docker compose logs backend
```

**Port conflict?**
Edit `docker-compose.yml` and change port numbers.

---

📖 Full documentation: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
