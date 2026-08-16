"""
dip_scorer.py — apply the ML dip confidence model to live signals.

Loads models/dip_confidence.pkl and scores each signal with a
bounce probability (0.0–1.0). Signals with dip_confidence ≥ 0.60
are higher-conviction buys.

Usage (standalone test):
    python3 dip_scorer.py
"""
import json
import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np

MODEL_PATH   = "models/dip_confidence.pkl"
CONFIDENCE_THRESHOLD = 0.60   # signals above this are high-conviction dip buys


def load_model():
    """Load the trained ML model. Returns None if model not found."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"]


# THE feature extractor lives in signal_features.py — one definition shared with
# breakout_scorer.py. Re-exported here so existing callers keep working; never
# re-inline it (tests/test_scorers.py fails CI if any file defines its own).
from signal_features import signal_to_features   # noqa: F401


def score_signals(signals: list) -> list:
    """
    Add dip_confidence score (0.0–1.0) to each signal.
    Returns the signals list with dip_confidence added.
    If model not found, dip_confidence = None for all signals.
    """
    model, scaler, features = load_model()

    if model is None or not signals:
        for sig in signals:
            sig["dip_confidence"] = None
        return signals

    feature_matrix = []
    for sig in signals:
        feat = signal_to_features(sig)
        row  = [feat.get(f, 0.0) for f in features]
        feature_matrix.append(row)

    X = np.array(feature_matrix, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    probas    = model.predict_proba(X_scaled)[:, 1]

    for sig, prob in zip(signals, probas):
        sig["dip_confidence"] = round(float(prob), 3)

    return signals


def confidence_label(score: float) -> str:
    if score is None:        return "unknown (no model)"
    if score >= 0.70:        return "HIGH ✓✓"
    if score >= 0.60:        return "elevated ✓"
    if score >= 0.50:        return "moderate"
    return                          "low"


if __name__ == "__main__":
    import glob, json

    # Test on latest signals file
    files = sorted(f for f in glob.glob("signals/2026-*.json")
                   if "_macro" not in f and "_intraday" not in f)
    if not files:
        print("No signals file found. Run research.py first.")
        exit(1)

    with open(files[-1]) as f:
        d = json.load(f)

    print(f"Scoring {len(d['signals'])} signals from {files[-1]}\n")
    scored = score_signals(d["signals"])

    print(f"{'Symbol':7s}  {'Score':>6s}  {'Confidence':>12s}  {'Label'}")
    print("─" * 55)
    for s in sorted(scored, key=lambda x: x.get("dip_confidence") or 0, reverse=True):
        conf = s.get("dip_confidence")
        label = confidence_label(conf)
        conf_str = f"{conf:.3f}" if conf is not None else "n/a"
        print(f"  {s['symbol']:5s}  {s['score']:>+5d}  {conf_str:>12s}  {label}")
