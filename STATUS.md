# Status

## 2026-11-27: v2 rebuild
- Removed the legacy apps, AWS deploy workflow and stale docs.
- New survival model (NHANES + linked mortality) on an Indian baseline, new API, notebook-themed frontend.
- Verified: 47 backend tests in Docker, 10 frontend unit tests, 2 browser flows (desktop, mobile), retrain from empty data gives identical coefficients.
- Open: India-adjusted estimates are unvalidated; no uncertainty on coefficients; accessibility not audited with a screen reader.
