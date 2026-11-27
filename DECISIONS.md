# Decisions

## 2026-11-27
- Data: NHANES with linked mortality; Indian baseline from SRS 2020-24 (Lebi: "make it for India").
- Lifespans anchored to a life table: the raw extrapolation gave median ages around 88-92. The table is Indian, not the CDC table first proposed.
- Coefficients limited so worse answers never add years; underweight capped.
- No CI and no deployment (Lebi).
- Deviations from the plan: no gradient-boosted benchmark; no sleep-shift correction (only reported); missing blood pressure and cholesterol use the reference value at serving; logging and exception modules replaced rather than reused; hand-built components instead of the shadcn CLI.
- Delivered on one branch instead of one PR per phase.
