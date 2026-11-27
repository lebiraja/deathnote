"""Turn a validated profile into estimates, a survival curve and a factor breakdown."""

from typing import Any

from app.ml.features import FACTOR_PREFIXES
from app.ml.india import LifeCurve
from app.ml.model import REFERENCE, Model
from app.schemas.prediction import (
    CurvePoint, Estimate, Factor, Habits, Prediction, Profile, WhatIfResponse, hours_asleep,
)
from app.services.advice import advise

# Factor -> the profile fields reset to the reference answer when measuring its effect.
FACTOR_FIELDS: dict[str, tuple[str, ...]] = {
    "sleep": ("sleep_hours",), "smoking": ("smoking",), "chewing_tobacco": ("chews_tobacco",),
    "alcohol": ("alcohol",), "activity": ("activity_minutes",), "diet": ("diet",), "weight": ("bmi",),
    "diabetes": ("diabetes",), "hypertension": ("hypertension",), "heart_disease": ("heart_disease",),
    "asthma": ("asthma",), "blood_pressure": ("systolic_bp",), "cholesterol": ("total_cholesterol",),
}
assert set(FACTOR_PREFIXES) <= set(FACTOR_FIELDS)


class InvalidChange(ValueError):
    """A what-if change that the data cannot answer."""


def to_features(profile: Profile) -> dict[str, Any]:
    row = profile.model_dump()
    row["sleep_hours"] = hours_asleep(profile.bedtime, profile.wake_time)
    row["bmi"] = profile.weight_kg / (profile.height_cm / 100) ** 2
    row["activity_minutes"] = profile.moderate_minutes + 2 * profile.vigorous_minutes
    return row


def current_age(profile: Profile) -> float:
    return profile.age + 0.5


def estimate(curve: LifeCurve) -> Estimate:
    median = curve.age_at_survival(0.5)
    return Estimate(
        median_age=round(median, 1),
        lower_age=round(curve.age_at_survival(0.75), 1),
        upper_age=round(curve.age_at_survival(0.25), 1),
        years_remaining=round(median - curve.current_age, 1),
    )


def curve_points(you: LifeCurve, average: LifeCurve) -> list[CurvePoint]:
    start = int(you.current_age)
    ages = sorted({start, *range(start + 1, 101)})
    return [
        CurvePoint(
            age=age,
            you=round(you.survival(max(age, you.current_age)), 4),
            average=round(average.survival(max(age, you.current_age)), 4),
        )
        for age in ages
    ]


def factor_breakdown(model: Model, row: dict[str, Any], age: float) -> list[Factor]:
    """Split the gap to the reference person across factors, in proportion to their log risk.

    The parts add up to the whole gap (up to rounding), also when the total risk is clipped.
    """
    reference = {**row, **{key: REFERENCE[key] for key in REFERENCE}}
    risk, reference_risk = model.relative_risks([row, reference])
    gap = (
        LifeCurve(row["sex"], age, risk).age_at_survival(0.5)
        - LifeCurve(row["sex"], age, reference_risk).age_at_survival(0.5)
    )
    resets = [{f: REFERENCE[f] for f in fields} for fields in FACTOR_FIELDS.values()]
    effects = model.log_effects(row, resets)
    total = sum(effects)
    if abs(total) < 1e-9:
        return []
    factors = [
        Factor(key=key, years=round(gap * effect / total, 1))
        for key, effect in zip(FACTOR_FIELDS, effects)
    ]
    return sorted((f for f in factors if f.years != 0), key=lambda f: f.years)


def predict(model: Model, profile: Profile) -> Prediction:
    row = to_features(profile)
    age = current_age(profile)
    you = LifeCurve(profile.sex, age, model.relative_risks([row])[0])
    average = LifeCurve(profile.sex, age, 1.0)
    return Prediction(
        estimate=estimate(you),
        average=estimate(average),
        sleep_hours=round(row["sleep_hours"], 2),
        bmi=round(row["bmi"], 1),
        curve=curve_points(you, average),
        factors=factor_breakdown(model, row, age),
        advice=advise(profile, row["sleep_hours"], row["bmi"]),
        model_version=model.version,
    )


def what_if(model: Model, profile: Profile, changes: Habits) -> WhatIfResponse:
    if profile.smoking != "never" and changes.smoking == "never":
        raise InvalidChange("Someone who has smoked can become a former smoker, not a never smoker")
    if profile.smoking == "never" and changes.smoking == "former":
        raise InvalidChange("A never smoker cannot become a former smoker")
    # Former drinkers in the data include people who stopped because they were ill.
    if changes.alcohol != profile.alcohol and changes.alcohol in ("never", "former"):
        raise InvalidChange("The data cannot estimate the effect of giving up alcohol; compare drinking levels")
    changed = Profile.model_validate({**profile.model_dump(), **changes.model_dump()})
    rows = [to_features(profile), to_features(changed)]
    before_risk, after_risk = model.relative_risks(rows)
    age = current_age(profile)
    before = LifeCurve(profile.sex, age, before_risk)
    after = LifeCurve(profile.sex, age, after_risk)
    first, second = estimate(before), estimate(after)
    return WhatIfResponse(
        before=first,
        after=second,
        change_years=round(after.age_at_survival(0.5) - before.age_at_survival(0.5), 1),
        curve=curve_points(after, before),
        sleep_hours=round(rows[1]["sleep_hours"], 2),
        bmi=round(rows[1]["bmi"], 1),
    )
