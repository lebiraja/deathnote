"""Load the JSON artifact and turn a profile into risk relative to the average Indian."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.ml import india
from app.ml.features import design_matrix

# The comparison person: relative risk 1 before adjusting to the population average.
REFERENCE: dict[str, Any] = {
    "bmi": 22.0, "sleep_hours": 7.5, "smoking": "never", "chews_tobacco": False,
    "alcohol": "never", "activity_minutes": 300.0, "diet": "very_good",
    "diabetes": 0, "hypertension": 0, "heart_disease": 0, "asthma": 0,
    "systolic_bp": 115.0, "total_cholesterol": 180.0,
}


@dataclass(frozen=True)
class Model:
    version: str
    coefficients: dict[str, float]
    metrics: dict[str, Any]
    score_limits: tuple[float, float]

    def log_score(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = design_matrix(pd.DataFrame(rows))
        weights = np.array([self.coefficients[c] for c in matrix.columns])
        return matrix.to_numpy(float) @ weights

    def average_indian_risk(self, sex: str) -> float:
        """Mean risk of Indian adults relative to the reference person.

        Uses published prevalence for tobacco, hypertension and diabetes and treats them
        as independent. Other factors are assumed to sit at the reference level.
        """
        c = self.coefficients
        pairs = (
            (india.SMOKING_PREVALENCE[sex], math.exp(c["smoking_ever"] + c["smoking_current"])),
            (india.CHEWING_PREVALENCE[sex], india.CHEWING_RELATIVE_RISK[sex]),
            (india.HYPERTENSION_PREVALENCE[sex], math.exp(c["hypertension"])),
            (india.DIABETES_PREVALENCE[sex], math.exp(c["diabetes"])),
        )
        return math.prod(1 + share * (risk - 1) for share, risk in pairs)

    def log_effects(self, profile: dict[str, Any], resets: list[dict[str, Any]]) -> list[float]:
        """For each reset, how much the replaced answers raise the log risk (before clipping)."""
        rows = [
            {key: REFERENCE[key] if row.get(key) is None else row[key] for key in REFERENCE}
            for row in [profile, *({**profile, **reset} for reset in resets)]
        ]
        scores = self.log_score(rows)
        # Chewing is not a model column; add its published estimate.
        chewing = [
            math.log(india.CHEWING_RELATIVE_RISK[profile["sex"]])
            if row["chews_tobacco"] and row["smoking"] != "current" else 0.0
            for row in rows
        ]
        totals = [float(score) + extra for score, extra in zip(scores, chewing)]
        return [totals[0] - other for other in totals[1:]]

    def relative_risks(self, profiles: list[dict[str, Any]]) -> list[float]:
        """Each profile's death rate relative to the average Indian of the same age and sex."""
        rows = [
            {key: REFERENCE[key] if p.get(key) is None else p[key] for key in REFERENCE} for p in profiles
        ]
        scores = np.clip(self.log_score(rows), *self.score_limits) - self.log_score([REFERENCE])[0]
        results = []
        for profile, score in zip(profiles, scores):
            sex = profile["sex"]
            risk = math.exp(score)
            # The published chewing estimate compares with never users, so it is not stacked on smoking.
            if profile["chews_tobacco"] and profile["smoking"] != "current":
                risk *= india.CHEWING_RELATIVE_RISK[sex]
            results.append(risk / self.average_indian_risk(sex))
        return results


def load_model(path: Path) -> Model:
    data = json.loads(path.read_text())
    return Model(data["version"], data["coefficients"], data["metrics"], tuple(data["score_limits"]))


@lru_cache
def get_model() -> Model:
    return load_model(get_settings().model_path)
