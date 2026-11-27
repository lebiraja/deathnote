# API

Base path `/api/v1`. Full schema at `/api/docs`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | `status`, `model_version` |
| POST | `/predictions/predict` | `Profile` | estimate, average, curve, factors, advice |
| POST | `/predictions/what-if` | `{profile, changes}` | before, after, `change_years`, curve (`you` = after, `average` = before) |
| POST | `/report` | `Profile` | PDF, recomputed on the server |

## Profile

`age` 20-79, `sex` female/male, `height_cm` 120-220, `weight_kg` 30-250, `bedtime` and `wake_time` `HH:MM` (2-16 h apart), `smoking` never/former/current, `chews_tobacco` bool, `alcohol` never/former/light/moderate/heavy, `moderate_minutes` and `vigorous_minutes` 0-3000 per week, `diet` excellent/very_good/good/fair/poor, `diabetes`, `hypertension`, `heart_disease`, `asthma` bool, optional `systolic_bp`, `total_cholesterol`.

`changes` takes the changeable fields: weight, sleep times, smoking, chewing, alcohol, exercise, diet.

What-if returns 422 for: smoker to never smoker, never smoker to former, and switching alcohol to never or former.
