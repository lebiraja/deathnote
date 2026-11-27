# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Life expectancy prediction from 14 health and lifestyle inputs (scikit-learn, synthetic data). README says the service is down.

## Layout

`backend/` (FastAPI) and `frontend/` (React 19, Vite), wired by `docker-compose.yml`. A v2 rebuild is in progress (NHANES survival model, new UI). `backend/*.md` and `frontend/README.md` are stale; trust the code.

## Commands

```bash
cp backend/.env.example backend/.env          # compose requires this file
docker compose up -d --build                  # frontend :80, backend :8000, redis host :6380
docker compose run --rm backend pytest        # pytest is only in requirements-dev.txt, not in the image
cd frontend && npm run dev | build | lint
```

API docs: `http://localhost:8000/api/docs`.

## Backend flow

`app/main.py` → `api/v1/router.py` → `endpoints/prediction.py` → `services/ml_service.py` (rename to Title_Case, label-encode with the preprocessor dict, order by `_feature_names`, scale, predict) → `services/recommendation_service.py` (rule-based). `ml/model_manager.py` loads the pickles lazily. `endpoints/report.py` builds the PDF with reportlab from client-supplied data. Routes: `/api/v1/predictions/predict`, `/api/v1/report`, `/api/v1/health[/detailed]`, `/api/v1/metrics/{system,model}`.

## Model artifacts (two incompatible sets)

| Set | sklearn | Alcohol_Consumption | Diet |
|---|---|---|---|
| `backend/models/` (served; trained on random data) | 1.8.0 | None, Light, Moderate, Heavy | Poor, Average, Good |
| `things/` (LFS; older set) | 1.3.0 | None, Moderate, High | Poor, Average, Healthy |

The React form uses the `things/` vocabulary.

## Known breakage

- UI values `High` (alcohol) and `Healthy` (diet) are rejected by the served encoder (HTTP 400).
- `backend/train_model.py` calls methods absent from `model.py` (`train_multiple_models`) and saves a different preprocessor format than `ml_service.py` reads.
- Tests send an old schema (`age`, no height/weight/alcohol).
- No dataset is tracked (`*.csv` ignored).
- `vite.config.ts` proxies `/api` to :5000; FastAPI is on :8000.
- `PredictionInput` has no range or enum validation. `confidence`/`accuracy` are hardcoded 0.87.
- Health `model_loaded` is true even when nothing is loaded. Redis is only pinged; no caching. `core/security.py` is unused.

## Delivery

No CI and no deployment (Lebi, 2026-11-27). The app runs in local Docker only.
