"""Survival maths.

Training fits a Gompertz proportional-hazards model on age:
    hazard(age | x) = exp(intercept + gamma_sex * (age - AGE_ORIGIN) + x . beta)
Serving keeps only the relative risk exp(x . beta) and applies it to India's life table.
"""

import numpy as np
from numpy.typing import NDArray

AGE_ORIGIN = 60.0
FloatArray = NDArray[np.float64]


def cumulative_hazard(
    start: FloatArray | float, end: FloatArray | float, log_rate: FloatArray | float, gamma: FloatArray | float
) -> FloatArray:
    """Hazard accumulated between two ages. `log_rate` is the log hazard at AGE_ORIGIN."""
    a = np.asarray(start, dtype=float) - AGE_ORIGIN
    b = np.asarray(end, dtype=float) - AGE_ORIGIN
    return np.exp(log_rate) * (np.exp(gamma * b) - np.exp(gamma * a)) / gamma
