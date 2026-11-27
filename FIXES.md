# Fixes

## 2026-11-27
- **Form values rejected (HTTP 400).** The served encoder lacked "High" alcohol and "Healthy" diet. Fixed by one typed schema on each side (`backend/app/schemas/prediction.py`, `frontend/src/lib/schema.ts`). Verified by API and browser tests.
- **Content-security-policy error in the browser console.** Zod tried to use eval. Fixed with `jitless` in `frontend/src/lib/schema.ts`. Verified by the browser test.
