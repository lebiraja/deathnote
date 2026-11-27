# Architecture

```
browser ── nginx (frontend) ── /api/v1 ──> FastAPI (backend) ──> model.json + India life table

NHANES + mortality (CDC) ──> ml.download ──> ml.data ──> ml.train ──> backend/app/ml/artifacts/model.json
```

| Path | Role |
|---|---|
| `backend/ml/` | Download (with checksums), harmonise, fit, evaluate |
| `backend/app/ml/features.py` | Encoding and coefficient limits, shared by training and serving |
| `backend/app/ml/survival.py` | Gompertz hazard used in training |
| `backend/app/ml/india.py` | SRS life table, Indian prevalences, chewing tobacco estimate |
| `backend/app/ml/model.py` | Loads the artifact; risk relative to the average Indian |
| `backend/app/services/` | Estimates, factor breakdown, what-if, advice, PDF |
| `backend/app/api/v1/router.py` | Routes |
| `frontend/src/pages/` | Cover, four question pages, result |
| `frontend/src/lib/` | Zod schemas, API client, form validation |

No database. User answers are held in browser memory, sent for calculation, and never stored or logged.
