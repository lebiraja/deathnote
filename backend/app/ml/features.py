"""Feature encoding. Training and serving both build the design matrix here."""

import numpy as np
import pandas as pd

SMOKING = ("never", "former", "current")
ALCOHOL = ("never", "former", "light", "moderate", "heavy", "unknown")
DIET = ("excellent", "very_good", "good", "fair", "poor")
SEX = ("female", "male")
CONDITIONS = ("diabetes", "hypertension", "heart_disease", "asthma")
SLEEP_KNOTS = (5.0, 6.0, 7.0, 8.0, 9.0)
SLEEP_RANGE = (3.0, 12.0)
# BMI thresholds follow the lower Asian Indian cut-offs for overweight (23).
BMI_UNDER, BMI_OVER, BMI_OBESE = 18.5, 23.0, 30.0
BMI_RANGE = (14.0, 55.0)
SBP_RANGE = (85.0, 220.0)
CHOLESTEROL_RANGE = (100.0, 350.0)

# Which design-matrix columns belong to each user-facing factor.
FACTOR_PREFIXES: dict[str, tuple[str, ...]] = {
    "sleep": ("sleep_",),
    "smoking": ("smoking_",),
    "alcohol": ("alcohol_",),
    "activity": ("activity_",),
    "diet": ("diet_",),
    "weight": ("bmi_",),
    "diabetes": ("diabetes",),
    "hypertension": ("hypertension",),
    "heart_disease": ("heart_disease",),
    "asthma": ("asthma",),
    "blood_pressure": ("sbp_",),
    "cholesterol": ("cholesterol_",),
}

# Columns whose coefficient may not be negative: a worse answer never adds years.
# Sleep is free (its effect is U-shaped) and so is the "alcohol unknown" training category.
UNCONSTRAINED_PREFIXES = ("sleep_", "alcohol_unknown")
# Upper limits where NHANES is implausibly extreme. Underweight is capped at the hazard
# ratio of about 1.5 reported for BMI 15-18.5 by the Global BMI Mortality Collaboration
# (Lancet 2016); the unconstrained NHANES fit is several times higher.
COEFFICIENT_CAPS = {"bmi_under": 0.405}


def spline(values: np.ndarray, knots: tuple[float, ...]) -> list[np.ndarray]:
    """Restricted cubic spline basis: linear beyond the outer knots."""
    first, last, before_last = knots[0], knots[-1], knots[-2]
    scale = (last - first) ** 2

    def cube(knot: float) -> np.ndarray:
        return np.clip(values - knot, 0, None) ** 3

    columns = [values - knots[len(knots) // 2]]
    for knot in knots[:-2]:
        term = (
            cube(knot)
            - cube(before_last) * (last - knot) / (last - before_last)
            + cube(last) * (before_last - knot) / (last - before_last)
        )
        columns.append(term / scale)
    return columns


def impute_optional(frame: pd.DataFrame, table: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Fill missing blood pressure and cholesterol with the sex and age-band mean."""
    filled = frame.copy()
    band = (filled["age"] // 10 * 10).clip(20, 70).astype(int)
    keys = filled["sex"] + "_" + band.astype(str)
    for column, means in table.items():
        filled[column] = filled[column].astype(float).fillna(keys.map(means))
    return filled


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode rows with complete values into model columns.

    Apart from sleep, columns are written as "how much worse than the best level",
    so non-negative coefficients give effects that only move in the expected direction.
    """
    columns: dict[str, np.ndarray] = {}
    sleep = frame["sleep_hours"].to_numpy(float).clip(*SLEEP_RANGE)
    for i, basis in enumerate(spline(sleep, SLEEP_KNOTS)):
        columns[f"sleep_{i}"] = basis

    bmi = frame["bmi"].to_numpy(float).clip(*BMI_RANGE)
    columns["bmi_under"] = np.clip(BMI_UNDER - bmi, 0, None) / 3
    columns["bmi_over"] = np.clip(bmi - BMI_OVER, 0, None) / 5
    columns["bmi_obese"] = np.clip(bmi - BMI_OBESE, 0, None) / 5

    def flag(column: str, *levels: str) -> np.ndarray:
        return frame[column].isin(levels).to_numpy(float)

    columns["smoking_ever"] = flag("smoking", "former", "current")
    columns["smoking_current"] = flag("smoking", "current")
    columns["alcohol_any_history"] = flag("alcohol", "former", "light", "moderate", "heavy")
    columns["alcohol_former"] = flag("alcohol", "former")
    columns["alcohol_heavy"] = flag("alcohol", "heavy")
    columns["alcohol_unknown"] = flag("alcohol", "unknown")
    for rank in range(1, len(DIET)):
        columns[f"diet_at_least_{DIET[rank]}"] = flag("diet", *DIET[rank:])

    minutes = frame["activity_minutes"].to_numpy(float)
    for name, threshold in (("none", 1.0), ("below_150", 150.0), ("below_300", 300.0)):
        columns[f"activity_{name}"] = (minutes < threshold).astype(float)

    for condition in CONDITIONS:
        columns[condition] = frame[condition].to_numpy(float)

    sbp = frame["systolic_bp"].to_numpy(float).clip(*SBP_RANGE)
    columns["sbp_raised"] = np.clip(sbp - 120, 0, None) / 20
    columns["sbp_high"] = np.clip(sbp - 140, 0, None) / 20
    cholesterol = frame["total_cholesterol"].to_numpy(float).clip(*CHOLESTEROL_RANGE)
    columns["cholesterol_raised"] = np.clip(cholesterol - 200, 0, None) / 40
    columns["cholesterol_high"] = np.clip(cholesterol - 240, 0, None) / 40
    return pd.DataFrame(columns, index=frame.index)
