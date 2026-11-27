"""Fit the survival model, evaluate it and write the JSON artifact."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.ml.features import COEFFICIENT_CAPS, UNCONSTRAINED_PREFIXES, design_matrix, impute_optional
from app.ml.survival import AGE_ORIGIN, cumulative_hazard
from ml.data import load

RIDGE = 1.0
OPTIONAL = ("systolic_bp", "total_cholesterol")
ARTIFACT = Path(__file__).resolve().parents[1] / "app" / "ml" / "artifacts" / "model.json"


def imputation_table(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    band = (frame["age"] // 10 * 10).clip(20, 70).astype(int)
    keys = frame["sex"] + "_" + band.astype(str)
    return {c: frame.groupby(keys)[c].mean().round(2).to_dict() for c in OPTIONAL}


def fit(frame: pd.DataFrame, matrix: pd.DataFrame, constrain: bool = True) -> dict:
    """Weighted maximum likelihood with delayed entry and a small ridge penalty.

    Parameters: intercept and slope for each sex, then one coefficient per column.
    """
    x = matrix.to_numpy(float)
    male = (frame["sex"] == "male").to_numpy(float)
    entry = frame["entry_age"].to_numpy(float)
    end = frame["exit_age"].to_numpy(float)
    dead = frame["dead"].to_numpy(float)
    weight = frame["weight_survey"].to_numpy(float)
    weight = weight / weight.mean()
    a, b = entry - AGE_ORIGIN, end - AGE_ORIGIN

    def objective(p: np.ndarray) -> tuple[float, np.ndarray]:
        c_f, c_m, g_f, g_m, beta = p[0], p[1], p[2], p[3], p[4:]
        gamma = np.where(male == 1, g_m, g_f)
        log_rate = np.where(male == 1, c_m, c_f) + x @ beta
        hazard = cumulative_hazard(entry, end, log_rate, gamma)
        loglik = np.sum(weight * (dead * (log_rate + gamma * b) - hazard))
        residual = weight * (dead - hazard)
        d_hazard = np.exp(log_rate) * (b * np.exp(gamma * b) - a * np.exp(gamma * a)) / gamma - hazard / gamma
        d_gamma = weight * (dead * b - d_hazard)
        gradient = np.concatenate([
            [residual @ (1 - male), residual @ male, d_gamma @ (1 - male), d_gamma @ male],
            x.T @ residual - RIDGE * beta,
        ])
        return -(loglik - 0.5 * RIDGE * beta @ beta), -gradient

    start = np.concatenate([[-4.5, -4.2, 0.09, 0.09], np.zeros(x.shape[1])])
    bounds = [(None, None)] * 2 + [(0.02, 0.2)] * 2 + [
        (0, COEFFICIENT_CAPS.get(c, 20)) if constrain and not c.startswith(UNCONSTRAINED_PREFIXES) else (-20, 20) for c in matrix.columns
    ]
    result = minimize(objective, start, jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"fit did not converge: {result.message}")
    p = result.x
    return {
        "intercept": {"female": p[0], "male": p[1]},
        "gamma": {"female": p[2], "male": p[3]},
        "coefficients": dict(zip(matrix.columns, p[4:].tolist())),
    }


def linear_terms(model: dict, frame: pd.DataFrame, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    log_rate = frame["sex"].map(model["intercept"]).to_numpy(float)
    log_rate = log_rate + matrix.to_numpy(float) @ np.array([model["coefficients"][c] for c in matrix.columns])
    return log_rate, frame["sex"].map(model["gamma"]).to_numpy(float)


def concordance(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C: among comparable pairs, how often the earlier death had the higher risk."""
    agree, total = 0.0, 0
    for i in np.flatnonzero(event == 1):
        later = (time > time[i]) | ((time == time[i]) & (event == 0))
        total += int(later.sum())
        agree += float((risk[i] > risk[later]).sum()) + 0.5 * float((risk[i] == risk[later]).sum())
    return agree / total


def predicted_risk(model: dict, frame: pd.DataFrame, matrix: pd.DataFrame, years: float) -> np.ndarray:
    log_rate, gamma = linear_terms(model, frame, matrix)
    entry = frame["entry_age"].to_numpy(float)
    return 1 - np.exp(-cumulative_hazard(entry, entry + years, log_rate, gamma))


def observed_risk(frame: pd.DataFrame, years: float) -> float:
    """Weighted Kaplan-Meier risk of death within `years`."""
    time = (frame["exit_age"] - frame["entry_age"]).to_numpy()
    order = np.argsort(time)
    time, dead = time[order], frame["dead"].to_numpy()[order]
    weight = frame["weight_survey"].to_numpy()[order]
    at_risk = weight[::-1].cumsum()[::-1]
    steps = (dead == 1) & (time <= years)
    return float(1 - np.prod(1 - weight[steps] / at_risk[steps]))


def calibration(frame: pd.DataFrame, risk: np.ndarray, years: float) -> list[dict]:
    groups = pd.qcut(risk, 5, labels=False)
    return [
        {
            "group": int(g) + 1,
            "predicted": round(float(np.average(risk[groups == g], weights=frame["weight_survey"][groups == g])), 4),
            "observed": round(observed_risk(frame[groups == g], years), 4),
        }
        for g in range(5)
    ]


def score(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str] | None = None, constrain: bool = True
) -> tuple[float, dict]:
    """Fit on `train`, return C-index on `test`. `columns=[]` gives the age + sex baseline."""
    table = imputation_table(train)
    train, test = impute_optional(train, table), impute_optional(test, table)
    x_train, x_test = design_matrix(train), design_matrix(test)
    if columns is not None:
        x_train, x_test = x_train[columns], x_test[columns]
    model = fit(train, x_train, constrain)
    log_rate, gamma = linear_terms(model, test, x_test)
    risk = log_rate + gamma * (test["entry_age"].to_numpy() - AGE_ORIGIN)
    time = (test["exit_age"] - test["entry_age"]).to_numpy()
    return concordance(risk, time, test["dead"].to_numpy()), model


def evaluate(frame: pd.DataFrame) -> dict:
    folds = np.random.default_rng(42).integers(0, 5, len(frame))
    full, baseline, free = [], [], []
    risks = {5: np.zeros(len(frame)), 10: np.zeros(len(frame))}
    for k in range(5):
        train, test = frame[folds != k], frame[folds == k]
        c, model = score(train, test)
        full.append(c)
        baseline.append(score(train, test, [])[0])
        free.append(score(train, test, constrain=False)[0])
        filled = impute_optional(test, imputation_table(train))
        for years, values in risks.items():
            values[folds == k] = predicted_risk(model, filled, design_matrix(filled), years)
    latest = frame["cycle"].max()
    holdout_train, holdout = frame[frame["cycle"] < latest], frame[frame["cycle"] == latest]
    early = frame[frame["cycle"] <= 2009]  # followed for about ten years
    return {
        "cv_c_index": round(float(np.mean(full)), 4),
        "cv_c_index_folds": [round(float(v), 4) for v in full],
        "cv_c_index_age_sex": round(float(np.mean(baseline)), 4),
        "cv_c_index_unconstrained": round(float(np.mean(free)), 4),
        "holdout_cycle": int(latest),
        "holdout_c_index": round(score(holdout_train, holdout)[0], 4),
        "holdout_deaths": int(holdout["dead"].sum()),
        "calibration_5y": calibration(frame, risks[5], 5),
        "calibration_10y_2007_2010": calibration(early, risks[10][frame["cycle"] <= 2009], 10),
    }


def sensitivity(frame: pd.DataFrame) -> dict:
    """Refit without deaths in the first two years, to probe reverse causation."""
    kept = frame[~((frame["dead"] == 1) & (frame["months"] < 24))]
    kept = impute_optional(kept, imputation_table(kept))
    return fit(kept, design_matrix(kept))["coefficients"]


def sleep_shift(frame: pd.DataFrame) -> dict:
    after = frame["cycle"] >= 2015
    return {
        name: round(float(np.average(part["sleep_hours"], weights=part["weight_survey"])), 3)
        for name, part in (("mean_2007_2014", frame[~after]), ("mean_2015_2018", frame[after]))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    frame, flow = load(args.data)
    print(json.dumps(flow, indent=2))
    table = imputation_table(frame)
    filled = impute_optional(frame, table)
    model = fit(filled, design_matrix(filled))
    reference = {
        "missing_share": {c: round(float(frame[c].isna().mean()), 4) for c in OPTIONAL},
        "alcohol_unknown_share": round(float((frame["alcohol"] == "unknown").mean()), 4),
        "sleep_measure": sleep_shift(frame),
    }
    scores = design_matrix(filled).to_numpy() @ np.array(list(model["coefficients"].values()))
    limits = [round(float(v), 4) for v in np.quantile(scores, [0.005, 0.995])]
    artifact = {
        "version": "2.0.0",
        "reference_population": "Effects estimated on US adults; baseline mortality at serving is Indian.",
        "trained_on": date.today().isoformat(),
        "source": "NHANES 2007-2018 with NCHS public-use linked mortality (follow-up to 2019-12-31)",
        **model,
        # Serving clips scores here: beyond it, stacked risks are not supported by the data.
        "score_limits": limits,
        "flow": flow,
        "data_notes": reference,
        "metrics": evaluate(frame),
        "sensitivity_without_early_deaths": sensitivity(frame),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
