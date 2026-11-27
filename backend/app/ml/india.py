"""India's baseline mortality and the published figures used to adapt the model.

Life table: Office of the Registrar General of India, SRS Based Abridged Life Tables
2020-24, all-India, probability of dying (nqx) by sex. The period includes the
COVID-19 years.
"""

from functools import lru_cache

import numpy as np

from app.ml.survival import FloatArray

MAX_AGE = 110.0
STEP = 0.05
# Start age of each interval; the last one ends at 85.
AGES = (0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85)
NQX: dict[str, tuple[float, ...]] = {
    "male": (0.02668, 0.00387, 0.00235, 0.00275, 0.00464, 0.00668, 0.00906, 0.01243, 0.01741,
             0.02510, 0.03356, 0.04980, 0.06946, 0.11646, 0.16213, 0.23413, 0.30753, 0.39829),
    "female": (0.02677, 0.00411, 0.00260, 0.00290, 0.00429, 0.00544, 0.00549, 0.00668, 0.00852,
               0.01164, 0.01854, 0.02886, 0.05311, 0.07729, 0.12360, 0.18788, 0.25781, 0.35301),
}
LIFE_EXPECTANCY_AT_BIRTH = {"male": 68.7, "female": 72.8}

# Tobacco: Global Adult Tobacco Survey 2016-17 (India), share of adults.
SMOKING_PREVALENCE = {"male": 0.190, "female": 0.020}
CHEWING_PREVALENCE = {"male": 0.296, "female": 0.128}
# NFHS-5 (2019-21), adults 15+, as reported in a secondary analysis (PMC12364263).
HYPERTENSION_PREVALENCE = {"male": 0.240, "female": 0.213}
DIABETES_PREVALENCE = {"male": 0.156, "female": 0.135}
# Smokeless tobacco vs never users, Mumbai Cohort Study, as tabulated by
# Shaikh et al., Nicotine Tob Res 2025 (doi:10.1093/ntr/ntaf121). NHANES cannot estimate this.
CHEWING_RELATIVE_RISK = {"male": 1.35, "female": 1.37}


@lru_cache
def baseline(sex: str) -> tuple[FloatArray, FloatArray]:
    """Age grid and cumulative hazard from birth. Beyond 85, a Gompertz continuation."""
    widths = np.diff(AGES)
    rates = -np.log1p(-np.array(NQX[sex])) / widths
    grid = np.arange(0, MAX_AGE + STEP, STEP)
    middle = grid + STEP / 2
    hazard = rates[np.clip(np.searchsorted(AGES, middle, side="right") - 1, 0, len(rates) - 1)]
    # Slope of the log hazard over the adult intervals from 60 to 85.
    midpoints = np.array(AGES[13:18]) + 2.5
    slope, level = np.polyfit(midpoints, np.log(rates[13:18]), 1)
    old = middle >= 85
    hazard[old] = np.exp(level + slope * middle[old])
    cumulative = np.concatenate([[0.0], np.cumsum(hazard[:-1] * STEP)])
    return grid, cumulative


class LifeCurve:
    """Survival from the current age, given a risk relative to the average Indian."""

    def __init__(self, sex: str, current_age: float, relative_risk: float) -> None:
        self.current_age = current_age
        self.grid, cumulative = baseline(sex)
        start = np.interp(current_age, self.grid, cumulative)
        self.hazard = relative_risk * (cumulative - start)

    def survival(self, age: float) -> float:
        return float(np.exp(-max(np.interp(age, self.grid, self.hazard), 0.0)))

    def age_at_survival(self, probability: float) -> float:
        """Age by which survival has fallen to `probability`, capped at MAX_AGE."""
        ahead = self.grid >= self.current_age
        return float(np.interp(-np.log(probability), self.hazard[ahead], self.grid[ahead]))
