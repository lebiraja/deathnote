# CLAUDE.md

Life expectancy estimates for Indian users from age, sleep, habits and health. `backend/` (FastAPI, survival model), `frontend/` (React 19, Vite, Tailwind 4). Details in `docs/`.

## Commands

```bash
docker compose up -d --build                   # app :8080, API :8047
docker compose run --rm test                   # backend tests
docker compose run --rm test pytest tests/test_api.py -k what_if
docker compose run --rm train                  # retrain, rewrites app/ml/artifacts/model.json
cd frontend && npm test | npm run lint | npm run build | npm run e2e
```

## Rules that bite

- `app/ml/features.py` defines the encoding for both training and serving. Changing it requires retraining.
- A worse answer must never add years (`UNCONSTRAINED_PREFIXES`, tests enforce it).
- Request/response shapes live in `app/schemas/prediction.py` and `frontend/src/lib/schema.ts`; change both.
- Figures in `app/ml/india.py` are sourced. Do not add numbers without a source.
- `backend/ml/checksums.json` pins the CDC files.
- Never store or log user answers.
- Use original artwork only; nothing from the Death Note franchise.
- No CI and no deployment (Lebi, 2026-11-27). Local Docker only.
