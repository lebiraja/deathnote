"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sex = Literal["female", "male"]
Smoking = Literal["never", "former", "current"]
Alcohol = Literal["never", "former", "light", "moderate", "heavy"]
Diet = Literal["excellent", "very_good", "good", "fair", "poor"]
CLOCK = r"^([01]\d|2[0-3]):[0-5]\d$"
MIN_SLEEP, MAX_SLEEP = 2.0, 16.0


def hours_asleep(bedtime: str, wake_time: str) -> float:
    def minutes(clock: str) -> int:
        hours, mins = clock.split(":")
        return int(hours) * 60 + int(mins)

    return ((minutes(wake_time) - minutes(bedtime)) % (24 * 60)) / 60


class Habits(BaseModel):
    """Answers a person can change."""

    model_config = ConfigDict(extra="forbid")

    weight_kg: float = Field(ge=30, le=250)
    bedtime: str = Field(pattern=CLOCK)
    wake_time: str = Field(pattern=CLOCK)
    smoking: Smoking
    chews_tobacco: bool
    alcohol: Alcohol
    moderate_minutes: int = Field(ge=0, le=3000, description="Per week, leisure and travel")
    vigorous_minutes: int = Field(ge=0, le=3000, description="Per week")
    diet: Diet

    @model_validator(mode="after")
    def plausible_sleep(self) -> "Habits":
        hours = hours_asleep(self.bedtime, self.wake_time)
        if not MIN_SLEEP <= hours <= MAX_SLEEP:
            raise ValueError(f"bedtime and wake_time must be {MIN_SLEEP:g} to {MAX_SLEEP:g} hours apart")
        return self


class Profile(Habits):
    age: int = Field(ge=20, le=79)
    sex: Sex
    height_cm: float = Field(ge=120, le=220)
    diabetes: bool
    hypertension: bool
    heart_disease: bool
    asthma: bool
    systolic_bp: float | None = Field(default=None, ge=70, le=260)
    total_cholesterol: float | None = Field(default=None, ge=80, le=500)


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Profile
    changes: Habits


class Estimate(BaseModel):
    median_age: float
    lower_age: float = Field(description="1 in 4 similar people would not reach this age")
    upper_age: float = Field(description="1 in 4 similar people would live beyond this age")
    years_remaining: float


class CurvePoint(BaseModel):
    age: int
    you: float
    average: float


class Factor(BaseModel):
    key: str
    years: float = Field(description="This factor's share of the gap to a person with reference answers")


class Advice(BaseModel):
    title: str
    detail: str


class Prediction(BaseModel):
    estimate: Estimate
    average: Estimate
    sleep_hours: float
    bmi: float
    curve: list[CurvePoint]
    factors: list[Factor]
    advice: list[Advice]
    model_version: str


class WhatIfResponse(BaseModel):
    before: Estimate
    after: Estimate
    change_years: float
    curve: list[CurvePoint]
    sleep_hours: float
    bmi: float
