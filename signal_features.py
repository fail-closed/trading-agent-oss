"""signal_features.py — THE inference-time feature extractor, defined once.

`dip_scorer` and `breakout_scorer` each carried a verbatim copy of this function
(breakout_scorer's even said "Mirroring the feature extraction in dip_scorer.py
for consistency" — a comment is not a mechanism). That is the same shape as the
MAX_ALLOC_SCALE incident and the −8% stop: one rule computed in two files, which
in this repo has drifted every single time.

Drift here would be quiet and expensive. Both models are scored with
`[feat[name] for name in features]` against the feature ORDER stored in the
pickle, so a divergence doesn't raise — it feeds one model a value from the
wrong column and returns a confident-looking probability that gates real buys.

**Two feature paths exist by necessity, and they must agree in meaning:**

  training  `ml_trainer.compute_features_series()` — from historical bar series
  inference this module                            — from a live research.py signal

They can't share code (different inputs entirely), so `tests/test_scorers.py`
pins their shared contract instead: the keys here must equal `ml_trainer.FEATURES`.

**A naming trap worth knowing.** `dip_depth_pct` here is signed — negative when
price sits ABOVE the SMA20 — matching `ml_trainer.py:133` exactly, which is what
the models were trained on. The `dip_depth_pct` FIELD that `research.py`
publishes into the signals file is a different convention (zeroed unless
`is_dip`), meant for display. Same name, two meanings; do not "reconcile" them
by copying one into the other.
"""

# Volume arrives as a −1/0/+1 signal, not a ratio; these are the midpoints of
# the bands research.py used to produce it (>1.5x bullish, <0.8x bearish).
_VOLUME_RATIO = {1: 1.7, -1: 0.7, 0: 1.0}

# Not available from a single signal snapshot — the trainer computes them from
# bar history. Held at neutral so the column exists and stays constant rather
# than becoming noise.
_UNAVAILABLE_AT_INFERENCE = {"mom_5d": 0.0, "mom_20d": 0.0, "vol_20d": 20.0}


def signal_to_features(sig: dict) -> dict:
    """Extract model features from one research.py signal dict.

    Every lookup is defaulted: signals vary by symbol and by day, and a KeyError
    here would kill the whole scoring pass rather than one symbol. Denominators
    are guarded for the same reason — a zero price or a missing SMA must degrade
    to a neutral number, never raise.
    """
    sigs = sig.get("signals", {}) or {}
    rsi_sig = sigs.get("rsi", {}) or {}
    ma_sig = sigs.get("ma_crossover", {}) or {}
    macd_sig = sigs.get("macd", {}) or {}
    bb_sig = sigs.get("bollinger", {}) or {}
    vol_sig = sigs.get("volume", {}) or {}

    price = sig.get("price", 1.0)
    macd_gap = macd_sig.get("macd", 0) - macd_sig.get("signal_line", 0)
    macd_gap_pct = macd_gap / price * 100 if price else 0

    sma50, sma200 = ma_sig.get("sma50", 0), ma_sig.get("sma200", 1)
    sma50_vs_200 = (sma50 - sma200) / sma200 * 100 if sma200 else 0

    sma20 = sig.get("sma20", price)
    # Signed on purpose — see the module docstring. Positive = below the SMA20.
    dip_depth = (sma20 - price) / sma20 * 100 if sma20 else 0

    return {
        "rsi":            rsi_sig.get("value", 50.0),
        "sma50_vs_200":   sma50_vs_200,
        "macd_gap_pct":   macd_gap_pct,
        "pct_b":          bb_sig.get("pct_b", 0.5),
        "dip_depth_pct":  dip_depth,
        "volume_ratio":   _VOLUME_RATIO.get(vol_sig.get("signal", 0), 1.0),
        "raw_score":      sig.get("score", 0),
        "weighted_score": sig.get("weighted_score", 0),
        "support_score":  sig.get("support_score", 0) or 0,
        **_UNAVAILABLE_AT_INFERENCE,
    }
