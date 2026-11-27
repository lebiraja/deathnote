# deathnote

Life expectancy estimation from health and lifestyle inputs: a FastAPI backend and a React frontend.

A rebuild is in progress: a survival model trained on NHANES data with linked mortality, and a new interface. The current model was trained on random data, so its output means nothing.

## Run locally

```bash
cp backend/.env.example backend/.env
docker compose up -d --build
```

Frontend: http://localhost. API docs: http://localhost:8000/api/docs.

This is an educational project and not medical advice.
