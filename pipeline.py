"""
pipeline.py
===========
End-to-end pipeline:

    raw CSV (DataFrame) -> feature_mapper (understand + standardise)
                        -> feature engineering (same as training)
                        -> model -> Prediction + Fraud_Probability + summary

Usage:
    from pipeline import run_pipeline
    result = run_pipeline(df, threshold=0.5)
    result.results_df.to_csv("output_results.csv", index=False)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from feature_mapper import KNOWN_TYPES, MappingError, understand_and_standardize

MODEL_PATH = os.environ.get(
    "FRAUD_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_detection_model.pkl"),
)
CHUNK = 500_000

# Feature order used by the current model (also used by train_model.py when training anew)
DEFAULT_FEATURE_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "balance_error_orig", "balance_error_dest", "orig_balance_change", "dest_balance_change",
    "hour", "day", "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


class PipelineError(ValueError):
    """A user-facing problem (bad file, missing columns...)."""


@dataclass
class PipelineResult:
    results_df: pd.DataFrame
    summary: dict
    mapping_details: list
    unmapped_columns: list
    data_report: dict
    warnings: list = field(default_factory=list)


@lru_cache(maxsize=2)
def load_model(path: str = MODEL_PATH) -> dict:
    """Load the saved model package {'model', 'model_name', 'feature_columns'}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    pkg = joblib.load(path)
    if not all(k in pkg for k in ("model", "feature_columns")):
        raise ValueError("Model package must contain 'model' and 'feature_columns'.")
    return pkg


# --------------------------------------------------------------------------- #
# Feature engineering — MUST stay identical to train_model.py / the notebook
# --------------------------------------------------------------------------- #
def engineer_features(std: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    amount = std["Amount"].astype(float)
    old_o, new_o = std["Old_Balance_Orig"].astype(float), std["New_Balance_Orig"].astype(float)
    old_d, new_d = std["Old_Balance_Dest"].astype(float), std["New_Balance_Dest"].astype(float)
    step = std["Time_Step"].astype(float)

    X = pd.DataFrame({
        "step": step,
        "amount": amount,
        "oldbalanceOrg": old_o,
        "newbalanceOrig": new_o,
        "oldbalanceDest": old_d,
        "newbalanceDest": new_d,
        "balance_error_orig": old_o - amount - new_o,
        "balance_error_dest": old_d + amount - new_d,
        "orig_balance_change": old_o - new_o,
        "dest_balance_change": new_d - old_d,
        "hour": step % 24,
        "day": step // 24,
    }, index=std.index)
    for t in KNOWN_TYPES:                                    # one-hot, same names as get_dummies
        X[f"type_{t}"] = (std["Type"] == t).astype(int)
    return X.reindex(columns=feature_columns, fill_value=0)


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), CHUNK):
        out[i:i + CHUNK] = model.predict_proba(X.iloc[i:i + CHUNK])[:, 1]
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame, threshold: float = 0.5, overrides: dict | None = None,
                 model_path: str | None = None) -> PipelineResult:
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise PipelineError("The uploaded file is empty.")
    if not 0.0 < threshold < 1.0:
        raise PipelineError("Threshold must be between 0 and 1.")

    try:
        std, mapping, report, warns = understand_and_standardize(df, overrides)
    except MappingError as e:
        raise PipelineError(str(e)) from e

    pkg = load_model(model_path or MODEL_PATH)
    X = engineer_features(std, pkg["feature_columns"])
    proba = _predict_proba(pkg["model"], X)
    pred = proba >= threshold

    out = df.copy()
    pred_col, prob_col = _free_name(out, "Prediction"), _free_name(out, "Fraud_Probability")
    out[pred_col] = np.where(pred, "Suspicious", "Normal")
    out[prob_col] = np.round(proba, 6)

    summary = _summarize(std, pred, proba, threshold, pkg, pred_col, prob_col)
    return PipelineResult(out, summary, mapping.details, mapping.unmapped_columns, report, warns)


def _free_name(df: pd.DataFrame, name: str) -> str:
    while name in df.columns:
        name = "AI_" + name
    return name


def _summarize(std, pred, proba, threshold, pkg, pred_col, prob_col) -> dict:
    n, s = len(std), int(pred.sum())
    amt = std["Amount"].to_numpy()
    summary = {
        "model_name": pkg.get("model_name", type(pkg["model"]).__name__),
        "threshold": threshold,
        "total_transactions": n,
        "suspicious_count": s,
        "normal_count": n - s,
        "suspicious_pct": round(100 * s / n, 3) if n else 0.0,
        "total_amount": float(amt.sum()),
        "suspicious_amount": float(amt[pred].sum()),
        "prediction_column": pred_col,
        "probability_column": prob_col,
    }
    by_type = []
    tmp = pd.DataFrame({"Type": std["Type"].to_numpy(), "sus": pred})
    for t, g in tmp.groupby("Type"):
        by_type.append({"type": t, "total": int(len(g)), "suspicious": int(g["sus"].sum()),
                        "pct": round(100 * g["sus"].mean(), 2)})
    summary["by_type"] = sorted(by_type, key=lambda r: -r["suspicious"])

    if "Is_Fraud" in std.columns:                           # file has labels -> report quality
        y = std["Is_Fraud"].to_numpy().astype(int)
        tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        summary["evaluation"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                 "precision": round(prec, 4), "recall": round(rec, 4),
                                 "f1": round(f1, 4)}
    return summary
