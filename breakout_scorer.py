"""
breakout_scorer.py — apply the ML breakout confidence model to live signals.

Loads models/breakout_confidence.pkl and scores each signal with a
breakout success probability (0.0–1.0).
"""
import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

MODEL_PATH = "models/breakout_confidence.pkl"

def load_model():
    """Load the trained ML model. Returns None if model not found."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"]


# THE feature extractor lives in signal_features.py — one definition shared with
# dip_scorer.py. Re-exported here so existing callers keep working; never
# re-inline it (tests/test_scorers.py fails CI if any file defines its own).
from signal_features import signal_to_features   # noqa: F401


def score_signals(signals: list) -> list:
    """
    Add breakout_confidence score (0.0–1.0) to each signal.
    """
    model, scaler, features = load_model()

    if model is None or not signals:
        for sig in signals:
            sig["breakout_confidence"] = None
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
        sig["breakout_confidence"] = round(float(prob), 3)

    return signals
