"""
train_model.py — retrain the fraud model on PaySim (or any labelled CSV).

    python train_model.py --data PS_20174392719_1491204439457_log.csv
    python train_model.py --data paysim.csv --sample 1000000 --out fraud_detection_model.pkl

It reuses feature_mapper + pipeline.engineer_features, so training and serving use EXACTLY the
same preprocessing (this was the bug in the notebook's /predict route).
Model selection uses F1 on the untouched test set (not PR-AUC, which was a near-tie).
"""
import argparse
import gc

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_mapper import understand_and_standardize
from pipeline import DEFAULT_FEATURE_COLUMNS, engineer_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labelled transactions CSV")
    ap.add_argument("--out", default="fraud_detection_model.pkl")
    ap.add_argument("--sample", type=int, default=0, help="use a random sample of N rows (0 = all)")
    ap.add_argument("--ratio", type=int, default=20, help="normal:fraud ratio after undersampling")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=42)
    std, mapping, _, _ = understand_and_standardize(df)
    if "Is_Fraud" not in std.columns:
        raise SystemExit("No label column found (e.g. isFraud). Cannot train.")
    X = engineer_features(std, DEFAULT_FEATURE_COLUMNS)
    y = std["Is_Fraud"].astype(int)
    del df, std; gc.collect()
    print(f"Rows: {len(X):,} | fraud: {int(y.sum()):,} ({100 * y.mean():.3f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # undersample normal rows in TRAIN only; the test set keeps its real distribution
    fraud_idx = y_tr[y_tr == 1].index
    normal_idx = y_tr[y_tr == 0].sample(min((y_tr == 0).sum(), len(fraud_idx) * a.ratio),
                                        random_state=42).index
    idx = fraud_idx.union(normal_idx)
    Xb, yb = X_tr.loc[idx], y_tr.loc[idx]
    print(f"Balanced train: {len(Xb):,} rows")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(yb == 0).sum() / max((yb == 1).sum(), 1), eval_metric="logloss",
            tree_method="hist", random_state=42, n_jobs=-1),
    }
    rows = {}
    for name, m in models.items():
        print(f"\nTraining {name} ...")
        m.fit(Xb, yb)
        p, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        rows[name] = {"Precision": precision_score(y_te, p, zero_division=0),
                      "Recall": recall_score(y_te, p, zero_division=0),
                      "F1": f1_score(y_te, p, zero_division=0),
                      "ROC-AUC": roc_auc_score(y_te, proba),
                      "PR-AUC": average_precision_score(y_te, proba)}
        print(classification_report(y_te, p, zero_division=0))

    table = pd.DataFrame(rows).T.round(4)
    print("\nModel comparison (test set):\n", table)
    best = table["F1"].idxmax()
    print(f"\nSelected model (best F1): {best}")
    joblib.dump({"model": models[best], "model_name": best,
                 "feature_columns": DEFAULT_FEATURE_COLUMNS,
                 "metrics": table.to_dict(orient="index")}, a.out)
    table.to_csv(a.out.replace(".pkl", "_comparison.csv"))
    print("Saved:", a.out)


if __name__ == "__main__":
    main()
