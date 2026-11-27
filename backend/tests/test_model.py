import numpy as np
import pandas as pd
import pytest

from app.ml.features import UNCONSTRAINED_PREFIXES
from app.ml.india import LIFE_EXPECTANCY_AT_BIRTH, LifeCurve, baseline
from app.ml.model import get_model
from ml import data


@pytest.mark.parametrize("sex", ["male", "female"])
def test_life_table_reproduces_official_life_expectancy(sex: str) -> None:
    grid, cumulative = baseline(sex)
    expectancy = np.trapezoid(np.exp(-cumulative), grid)
    assert expectancy == pytest.approx(LIFE_EXPECTANCY_AT_BIRTH[sex], abs=1.5)


def test_higher_risk_means_earlier_median() -> None:
    ages = [LifeCurve("male", 40, risk).age_at_survival(0.5) for risk in (0.5, 1, 2)]
    assert ages == sorted(ages, reverse=True)


def test_age_at_survival_inverts_survival() -> None:
    curve = LifeCurve("female", 35, 1.2)
    assert curve.survival(curve.age_at_survival(0.25)) == pytest.approx(0.25, abs=0.005)


def test_artifact_respects_constraints() -> None:
    for name, value in get_model().coefficients.items():
        if not name.startswith(UNCONSTRAINED_PREFIXES):
            assert value >= 0, name


def test_artifact_beats_age_and_sex_baseline() -> None:
    metrics = get_model().metrics
    assert metrics["cv_c_index"] > metrics["cv_c_index_age_sex"]


def test_smoking_codes() -> None:
    frame = pd.DataFrame({"SMQ020": [2, 1, 1, 1, 9], "SMQ040": [np.nan, 3, 1, 2, np.nan]})
    assert data.smoking(frame).tolist()[:4] == ["never", "former", "current", "current"]
    assert pd.isna(data.smoking(frame).iloc[4])


def test_alcohol_codes_before_2017() -> None:
    frame = pd.DataFrame({
        "ALQ101": [2, 1, 1, 1, 1], "ALQ110": [2, np.nan, np.nan, np.nan, np.nan],
        "ALQ120Q": [np.nan, 0, 2, 5, 999], "ALQ120U": [np.nan, np.nan, 1, 1, np.nan],
        "ALQ130": [np.nan, np.nan, 2, 4, np.nan],
    })
    assert data.alcohol(frame).tolist() == ["never", "former", "light", "heavy", "unknown"]


def test_alcohol_codes_2017() -> None:
    frame = pd.DataFrame({"ALQ111": [2, 1, 1, 1], "ALQ121": [np.nan, 0, 4, 1], "ALQ130": [np.nan, np.nan, 5, 3]})
    assert data.alcohol(frame).tolist() == ["never", "former", "moderate", "heavy"]


def test_activity_counts_vigorous_twice() -> None:
    frame = pd.DataFrame({
        "PAQ635": [2, 1], "PAQ640": [np.nan, 5], "PAD645": [np.nan, 20],
        "PAQ650": [2, 1], "PAQ655": [np.nan, 2], "PAD660": [np.nan, 30],
        "PAQ665": [2, 2], "PAQ670": [np.nan, np.nan], "PAD675": [np.nan, np.nan],
    })
    assert data.activity_minutes(frame).tolist() == [0, 220]
