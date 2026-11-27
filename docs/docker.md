# Docker

| Service | Purpose | Host port |
|---|---|---|
| `backend` | FastAPI | 127.0.0.1:8047 (`API_PORT`) |
| `frontend` | nginx: built app and `/api/v1` proxy | 127.0.0.1:8080 (`WEB_PORT`) |
| `test` | pytest (profile `tools`) | none |
| `train` | download and train (profile `tools`) | none |

Optional `backend/.env` (see `.env.example`): `LOG_LEVEL`, `CORS_ORIGINS`.

The network uses `10.213.47.0/24`; override with `DEATHNOTE_SUBNET`. `train` writes into `backend/` as `UID:GID` (default 1000:1000).

No deployment or CI: the project runs locally.
