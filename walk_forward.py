"""
walk_forward.py — walk-forward (rolling out-of-sample) validation for ML models.

The standard train_test_split(shuffle=True) used at training time interleaves
test rows between training rows in time — for market data that is lookahead
leakage and inflates AUC. Walk-forward validation answers the question that
actually matters: "would a model trained only on the past have predicted the
future?"

Method: sort the dataset by date, cut it into sequential blocks, and for each
fold train on everything up to the block boundary and test strictly on the
next block. Compare in-sample vs out-of-sample AUC and the stability of the
out-of-sample folds:

  verdict     criteria
  ROBUST      mean OOS AUC ≥ 0.55, fold std < 0.03, IS−OOS gap < 0.05
  MODERATE    mean OOS AUC ≥ 0.53 and IS−OOS gap < 0.08
  WEAK        mean OOS AUC ≥ 0.51
  OVERFITTED  OOS near coin-flip (< 0.51) or IS−OOS gap ≥ 0.10

ml_trainer.py calls validate() after training and refuses to deploy a model
whose verdict is OVERFITTED.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _make_model():
    """Same model family/hyperparameters as ml_trainer's production model."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )


def _verdict(mean_oos: float, std_oos: float, gap: float) -> str:
    if mean_oos < 0.51 or gap >= 0.10:
        return "OVERFITTED"
    if mean_oos >= 0.55 and std_oos < 0.03 and gap < 0.05:
        return "ROBUST"
    if mean_oos >= 0.53 and gap < 0.08:
        return "MODERATE"
    return "WEAK"


def validate(df: pd.DataFrame, features: list, y: pd.Series,
             date_col: str = "date", n_folds: int = 5,
             min_train_frac: float = 0.30) -> dict:
    """
    Walk-forward validation. df must contain date_col; y is the binary target
    aligned to df. Returns a report dict with per-fold metrics and a verdict.
    """
    if date_col not in df.columns:
        return {"verdict": "SKIPPED", "reason": f"no {date_col} column in dataset"}

    order = df[date_col].argsort(kind="mergesort")
    X_all = (df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
             .iloc[order].reset_index(drop=True))
    y_all = y.iloc[order].reset_index(drop=True)
    dates = df[date_col].iloc[order].reset_index(drop=True)

    n = len(X_all)
    first_train_end = int(n * min_train_frac)
    fold_size = (n - first_train_end) // n_folds
    if fold_size < 200:
        return {"verdict": "SKIPPED",
                "reason": f"dataset too small for {n_folds} folds ({n} rows)"}

    folds = []
    for i in range(n_folds):
        train_end = first_train_end + i * fold_size
        test_end  = train_end + fold_size
        X_tr, y_tr = X_all.iloc[:train_end], y_all.iloc[:train_end]
        X_te, y_te = X_all.iloc[train_end:test_end], y_all.iloc[train_end:test_end]
        if y_te.nunique() < 2 or y_tr.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = _make_model()
        model.fit(X_tr_s, y_tr)

        is_auc  = roc_auc_score(y_tr, model.predict_proba(X_tr_s)[:, 1])
        oos_auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])
        folds.append({
            "fold":        i + 1,
            "train_rows":  len(X_tr),
            "test_rows":   len(X_te),
            "test_period": f"{dates.iloc[train_end].date()} → {dates.iloc[test_end-1].date()}",
            "in_sample_auc":     round(float(is_auc), 4),
            "out_of_sample_auc": round(float(oos_auc), 4),
        })
        print(f"  Fold {i+1}/{n_folds}: train {len(X_tr):,} → test {len(X_te):,} "
              f"({folds[-1]['test_period']})  IS {is_auc:.4f} / OOS {oos_auc:.4f}")

    if not folds:
        return {"verdict": "SKIPPED", "reason": "no valid folds"}

    oos = [f["out_of_sample_auc"] for f in folds]
    iss = [f["in_sample_auc"] for f in folds]
    mean_oos, std_oos = float(np.mean(oos)), float(np.std(oos))
    gap = float(np.mean(iss)) - mean_oos
    verdict = _verdict(mean_oos, std_oos, gap)

    return {
        "verdict":            verdict,
        "mean_oos_auc":       round(mean_oos, 4),
        "std_oos_auc":        round(std_oos, 4),
        "mean_is_auc":        round(float(np.mean(iss)), 4),
        "is_oos_gap":         round(gap, 4),
        "n_folds":            len(folds),
        "folds":              folds,
    }
