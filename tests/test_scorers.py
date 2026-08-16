"""dip_scorer / breakout_scorer — feature extraction and safe degradation.

These gate real decisions: `dip_confidence` below 0.45 auto-skips a buy, and the
0.55/0.68 bands choose ladder-vs-full sizing. Two failure modes matter most.

**Degrading to None, not to a number.** With no model file every signal must get
`dip_confidence = None`. If it ever degraded to 0.0 instead, decide.py's
"< 0.45 → skip" rule would silently refuse EVERY dip buy and the agent would
look like it was being disciplined.

**Feature ORDER.** The row is built as `[feat[f] for f in features]` against the
order stored in the pickle. If extraction and that list disagree, the model gets
RSI where it expects momentum and still returns a confident-looking number —
wrong answers, no error.

The two modules also carry a copy-pasted `signal_to_features` ("Mirroring the
feature extraction in dip_scorer.py"), which is the same duplicate-rule shape as
MAX_ALLOC_SCALE and the −8% stop. There is a test for that below.
"""
import types

import numpy as np
import pytest

import breakout_scorer
import dip_scorer


def _sig(**over):
    base = {
        "symbol": "AAA", "price": 100.0, "sma20": 110.0, "score": 2,
        "weighted_score": 3.0, "support_score": 40,
        "signals": {
            "rsi": {"value": 28.0},
            "ma_crossover": {"sma50": 105.0, "sma200": 100.0},
            "macd": {"macd": 1.5, "signal_line": 1.0},
            "bollinger": {"pct_b": 0.05},
            "volume": {"signal": 1},
        },
    }
    base.update(over)
    return base


class _Model:
    """Records the feature matrix it is handed, returns a fixed probability."""

    def __init__(self, p=0.73):
        self.seen = None
        self._p = p

    def predict_proba(self, X):
        self.seen = X
        return np.column_stack([1 - np.full(len(X), self._p), np.full(len(X), self._p)])


class _Scaler:
    def transform(self, X):
        return X          # identity, so assertions see the raw feature values


def _with_model(monkeypatch, module, features, model=None):
    model = model or _Model()
    monkeypatch.setattr(module, "load_model", lambda: (model, _Scaler(), features))
    return model


# ── Safe degradation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("module,field", [(dip_scorer, "dip_confidence"),
                                          (breakout_scorer, "breakout_confidence")])
def test_missing_model_yields_None_not_zero(monkeypatch, module, field):
    """0.0 would read as 'model says no' and silently skip every buy."""
    monkeypatch.setattr(module, "load_model", lambda: (None, None, None))
    out = module.score_signals([_sig(), _sig(symbol="BBB")])
    assert [s[field] for s in out] == [None, None]


@pytest.mark.parametrize("module", [dip_scorer, breakout_scorer])
def test_empty_signal_list_is_safe(monkeypatch, module):
    _with_model(monkeypatch, module, ["rsi"])
    assert module.score_signals([]) == []


def test_scores_land_on_the_signals(monkeypatch):
    _with_model(monkeypatch, dip_scorer, ["rsi", "pct_b"], _Model(0.812))
    out = dip_scorer.score_signals([_sig()])
    assert out[0]["dip_confidence"] == 0.812


# ── Feature extraction correctness ──────────────────────────────────────────

def test_feature_derivations():
    f = dip_scorer.signal_to_features(_sig())
    assert f["rsi"] == 28.0
    assert f["pct_b"] == 0.05
    # price 100 below sma20 110 → a 9.09% dip, POSITIVE = deeper
    assert f["dip_depth_pct"] == pytest.approx(9.0909, abs=1e-3)
    # sma50 105 vs sma200 100 → +5%
    assert f["sma50_vs_200"] == pytest.approx(5.0)
    # (macd 1.5 − signal 1.0) / price 100 × 100 = 0.5
    assert f["macd_gap_pct"] == pytest.approx(0.5)


def test_dip_depth_is_negative_when_price_is_above_the_mean():
    f = dip_scorer.signal_to_features(_sig(price=120.0, sma20=100.0))
    assert f["dip_depth_pct"] < 0, "an extended price must not read as a dip"


@pytest.mark.parametrize("vol_signal,expected", [(1, 1.7), (0, 1.0), (-1, 0.7)])
def test_volume_signal_maps_to_a_ratio(vol_signal, expected):
    s = _sig()
    s["signals"]["volume"] = {"signal": vol_signal}
    assert dip_scorer.signal_to_features(s)["volume_ratio"] == expected


def test_missing_fields_fall_back_instead_of_raising():
    """research.py signals vary by symbol and by day; a KeyError here would kill
    the whole scoring pass, not one symbol."""
    f = dip_scorer.signal_to_features({"symbol": "AAA"})
    assert f["rsi"] == 50.0 and f["pct_b"] == 0.5
    assert all(isinstance(v, (int, float)) for v in f.values())


@pytest.mark.parametrize("bad", [{"price": 0}, {"sma20": 0},
                                 {"signals": {"ma_crossover": {"sma50": 5, "sma200": 0}}}])
def test_zero_denominators_do_not_raise(bad):
    dip_scorer.signal_to_features(_sig(**bad))          # must simply not raise


# ── Feature ORDER — the silent-wrong-answer failure ─────────────────────────

def test_matrix_columns_follow_the_models_feature_order(monkeypatch):
    """Reversing the order must reverse the columns. If extraction ignored the
    pickle's order, the model would score garbage and still look confident."""
    order = ["pct_b", "rsi"]
    model = _with_model(monkeypatch, dip_scorer, order)
    dip_scorer.score_signals([_sig()])
    assert model.seen.tolist() == [[0.05, 28.0]]

    model2 = _with_model(monkeypatch, dip_scorer, list(reversed(order)))
    dip_scorer.score_signals([_sig()])
    assert model2.seen.tolist() == [[28.0, 0.05]]


def test_unknown_feature_names_become_zero_not_an_exception(monkeypatch):
    model = _with_model(monkeypatch, dip_scorer, ["rsi", "a_feature_we_dropped"])
    dip_scorer.score_signals([_sig()])
    assert model.seen.tolist() == [[28.0, 0.0]]


def test_nan_and_inf_are_sanitised(monkeypatch):
    model = _with_model(monkeypatch, dip_scorer, ["rsi", "pct_b"])
    s = _sig()
    s["signals"]["rsi"] = {"value": float("nan")}
    s["signals"]["bollinger"] = {"pct_b": float("inf")}
    dip_scorer.score_signals([s])
    assert np.isfinite(model.seen).all(), "NaN/inf must never reach the model"


# ── One extractor, not two ─────────────────────────────────────────────────

def test_both_scorers_share_one_extractor():
    """Not 'they agree' — literally the same function object. The two modules
    used to hold verbatim copies, with breakout_scorer's even commenting that it
    was 'mirroring' dip_scorer's; that is the MAX_ALLOC_SCALE / −8%-stop shape,
    and in this repo it has drifted every time."""
    import signal_features
    assert dip_scorer.signal_to_features is breakout_scorer.signal_to_features
    assert dip_scorer.signal_to_features is signal_features.signal_to_features


def test_no_module_defines_its_own_extractor():
    """Tripwire. If this fails, import from signal_features — do not re-inline."""
    import re
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=root,
                             capture_output=True, text=True).stdout.split()
    offenders = [f for f in tracked
                 if f not in ("signal_features.py",) and not f.startswith("tests/")
                 and re.search(r"^def signal_to_features", (root / f).read_text(), re.M)]
    assert not offenders, (
        f"{offenders} define their own signal_to_features — import it from "
        "signal_features.py instead. Two copies of a feature extractor drift "
        "silently: the row is built as [feat[name] for name in features], so a "
        "divergence feeds a model the wrong column and still returns a "
        "confident-looking probability that gates real buys.")


def test_extractor_matches_the_trained_feature_contract():
    """The inference path (signal_features) and the training path
    (ml_trainer.compute_features_series, built from bar history) cannot share
    code — different inputs — so the contract they share is the feature NAME SET.
    If they disagree, `[feat.get(f, 0.0) for f in features]` silently feeds 0.0
    for the missing one instead of failing.

    Read from SOURCE, never imported: ml_trainer pulls in scikit-learn at module
    scope, and CI deliberately keeps sklearn/xgboost out to stay light. Importing
    it here turned both main and live CI red on 2026-08-14.
    """
    import ast
    from pathlib import Path

    import signal_features

    src = (Path(__file__).resolve().parent.parent / "ml_trainer.py").read_text()
    features = next(
        ast.literal_eval(node.value)
        for node in ast.parse(src).body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "FEATURES" for t in node.targets))

    produced = set(signal_features.signal_to_features({"symbol": "AAA"}))
    assert produced == set(features), (
        f"extractor keys vs ml_trainer.FEATURES — missing: {set(features) - produced}, "
        f"extra: {produced - set(features)}")


def test_dip_depth_keeps_the_TRAINING_convention():
    """Signed, matching ml_trainer.py:133. research.py publishes a FIELD of the
    same name using a different (display) convention — zeroed unless is_dip — so
    this is easy to 'reconcile' by mistake. The models were trained on the signed
    version; changing it would silently shift every prediction."""
    import signal_features
    extended = signal_features.signal_to_features(
        {"price": 120.0, "sma20": 100.0})["dip_depth_pct"]
    assert extended == pytest.approx(-20.0), (
        "dip_depth_pct must stay signed at inference to match training")


def test_confidence_labels_track_the_documented_bands():
    assert dip_scorer.confidence_label(None) == "unknown (no model)"
    assert "HIGH" in dip_scorer.confidence_label(0.71)
    assert "elevated" in dip_scorer.confidence_label(0.61)
    assert dip_scorer.confidence_label(0.55) == "moderate"
    assert dip_scorer.confidence_label(0.20) == "low"
