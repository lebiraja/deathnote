"""Harmonise the NHANES cycles into one analysis table."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.download import CYCLES

# ALQ121 (2017-18) answer code -> drinking days per week.
DAYS_PER_WEEK_2017 = {
    1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0,
    6: 2.5 * 12 / 52, 7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52,
}


def read_xpt(path: Path) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport")
    numeric = frame.select_dtypes("number").columns
    # pandas reads SAS zero as 5.4e-79
    frame[numeric] = frame[numeric].mask(frame[numeric].abs() < 1e-70, 0.0)
    return frame.set_index("SEQN")


def read_mortality(path: Path) -> pd.DataFrame:
    """Fixed-width layout from the NCHS public-use linked mortality read programs."""
    raw = pd.read_fwf(
        path,
        colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
        names=["SEQN", "eligible", "dead", "months"],
        na_values=".",
    )
    return raw.set_index("SEQN")


def yes_no(series: pd.Series) -> pd.Series:
    """1 = yes, 2 = no; refused and unknown become missing."""
    return series.map({1: 1.0, 2: 0.0})


def sleep_hours(slq: pd.DataFrame) -> pd.Series:
    if "SLD012" in slq:
        return slq["SLD012"]
    return slq["SLD010H"].where(slq["SLD010H"] <= 24)


def smoking(smq: pd.DataFrame) -> pd.Series:
    status = pd.Series(np.nan, index=smq.index, dtype=object)
    status[smq["SMQ020"] == 2] = "never"
    status[(smq["SMQ020"] == 1) & (smq["SMQ040"] == 3)] = "former"
    status[(smq["SMQ020"] == 1) & smq["SMQ040"].isin([1, 2])] = "current"
    return status


def drink_category(weekly: pd.Series) -> pd.Series:
    labels = np.select([weekly <= 7, weekly <= 14], ["light", "moderate"], "heavy")
    return pd.Series(labels, index=weekly.index, dtype=object).where(weekly.notna())


def alcohol(alq: pd.DataFrame) -> pd.Series:
    per_day = alq["ALQ130"].where(alq["ALQ130"] < 100)
    if "ALQ121" in alq:
        days = alq["ALQ121"].map(DAYS_PER_WEEK_2017)
        never = alq["ALQ111"] == 2
        none_last_year = alq["ALQ121"] == 0
    else:
        quantity = alq["ALQ120Q"].where(alq["ALQ120Q"] <= 366)
        per_week = alq["ALQ120U"].map({1: 1.0, 2: 12 / 52, 3: 1 / 52})
        days = (quantity * per_week).clip(upper=7).where(quantity > 0)
        # Fewer than 12 drinks ever counts as never.
        never = (alq["ALQ101"] == 2) & (alq["ALQ110"] == 2)
        none_last_year = (quantity == 0) | ((alq["ALQ101"] == 2) & (alq["ALQ110"] == 1) & quantity.isna())
    status = drink_category(days * per_day)
    status[none_last_year] = "former"
    status[never] = "never"
    return status.fillna("unknown")


def activity_minutes(paq: pd.DataFrame) -> pd.Series:
    """Weekly moderate-equivalent minutes from travel and leisure (vigorous counts double)."""
    total = pd.Series(0.0, index=paq.index)
    for flag, days, minutes, weight in (
        ("PAQ635", "PAQ640", "PAD645", 1),  # walking or cycling
        ("PAQ650", "PAQ655", "PAD660", 2),  # vigorous recreation
        ("PAQ665", "PAQ670", "PAD675", 1),  # moderate recreation
    ):
        week = paq[days].where(paq[days] <= 7) * paq[minutes].where(paq[minutes] <= 1440) * weight
        total += week.where(paq[flag] == 1, 0.0).where(paq[flag].isin([1, 2]))
    return total


def load_cycle(directory: Path, suffix: str, year: int) -> pd.DataFrame:
    def table(stem: str) -> pd.DataFrame:
        return read_xpt(directory / f"{stem}_{suffix}.xpt")

    demo, bmx, bpx = table("DEMO"), table("BMX"), table("BPX")
    mcq, alq = table("MCQ"), table("ALQ")
    heart = mcq[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"]]
    columns = {
        "age_years": demo["RIDAGEYR"],
        "sex": demo["RIAGENDR"].map({1: "male", 2: "female"}),
        "weight_survey": demo["WTMEC2YR"] / len(CYCLES),
        "bmi": bmx["BMXBMI"],
        "systolic_bp": bpx.filter(regex=r"^BPXSY\d$").mean(axis=1),
        "total_cholesterol": table("TCHOL")["LBXTC"],
        "sleep_hours": sleep_hours(table("SLQ")),
        "smoking": smoking(table("SMQ")),
        "alcohol": alcohol(alq),
        "activity_minutes": activity_minutes(table("PAQ")),
        "diet": table("DBQ")["DBQ700"].map({1: "excellent", 2: "very_good", 3: "good", 4: "fair", 5: "poor"}),
        "diabetes": table("DIQ")["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0}),
        "hypertension": yes_no(table("BPQ")["BPQ020"]),
        "heart_disease": (heart == 1).any(axis=1).astype(float).where((heart == 1).any(axis=1) | (heart == 2).all(axis=1)),
        "asthma": yes_no(mcq["MCQ010"]),
    }
    mortality = read_mortality(directory / f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat")
    frame = pd.DataFrame(columns).join(mortality, how="inner")
    frame["alcohol"] = frame["alcohol"].fillna("unknown")
    frame["cycle"] = year
    return frame


REQUIRED = (
    "age_years", "sex", "bmi", "sleep_hours", "smoking", "activity_minutes", "diet",
    "diabetes", "hypertension", "heart_disease", "asthma", "dead", "months",
)


def load(directory: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the analysis table and the participant count after each filter."""
    everyone = pd.concat([load_cycle(directory, s, y) for s, y in CYCLES.items()])
    adults = everyone[everyone["age_years"].between(20, 79)]
    eligible = adults[(adults["eligible"] == 1) & (adults["weight_survey"] > 0)]
    complete = eligible.dropna(subset=list(REQUIRED)).copy()
    # Ages are whole years; assume the midpoint, and count at least half a month of follow-up.
    complete["age"] = complete["age_years"]
    complete["entry_age"] = complete["age_years"] + 0.5
    complete["exit_age"] = complete["entry_age"] + complete["months"].clip(lower=0.5) / 12
    flow = {
        "participants": len(everyone),
        "aged_20_79": len(adults),
        "mortality_eligible_examined": len(eligible),
        "complete": len(complete),
        "deaths": int(complete["dead"].sum()),
    }
    return complete, flow
