# deathnote

Estimates life expectancy from age, sleep, habits and health, against Indian death rates. A FastAPI backend, a React frontend and a survival model trained on NHANES with linked mortality.

An educational project. It is not medical advice. Read [the model card](docs/model-card.md) before trusting any number.

## Run

```bash
docker compose up -d --build
```

App: http://localhost:8080. API docs: http://localhost:8047/api/docs. Change ports with `WEB_PORT` and `API_PORT`.

## Test and train

```bash
docker compose run --rm test                  # backend tests
docker compose run --rm train                 # download NHANES, retrain, rewrite the artifact
cd frontend && npm ci && npm test             # frontend unit tests
cd frontend && npm run e2e                    # browser tests; needs the app running and Google Chrome
```

Documentation: [docs/](docs/README.md).
