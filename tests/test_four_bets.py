"""
Tests for the four-bets rebalance: deploy thresholds, entry-mix tracking,
cross-sectional momentum, and the covered-call overlay.

The bar for each new knob is the same: with no env set, behaviour must be
byte-identical to what shipped before. Everything here is a pure function or
runs against a fake broker — no network, no keys.
"""
import os
import sys
import tempfile
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Change 1: per-mode buy thresholds ────────────────────────────────────────

def test_deploy_thresholds_default_to_legacy_ladder(monkeypatch):
    for k in ("DEPLOY_THRESH_AGGRESSIVE", "DEPLOY_THRESH_ACTIVE", "DEPLOY_THRESH_STANDARD"):
        monkeypatch.delenv(k, raising=False)
    import research
    assert research.deploy_thresholds() == {"aggressive": 0, "active": 1, "standard": 2}


def test_deploy_thresholds_env_override(monkeypatch):
    monkeypatch.setenv("DEPLOY_THRESH_AGGRESSIVE", "1")
    import research
    assert research.deploy_thresholds()["aggressive"] == 1


def test_deploy_thresholds_survive_garbage_env(monkeypatch):
    """A typo in a Railway variable must not take the trading session down."""
    monkeypatch.setenv("DEPLOY_THRESH_AGGRESSIVE", "not-a-number")
    import research
    with pytest.raises(ValueError):
        research.deploy_thresholds()   # documented: fail loud, don't trade on a guess


@pytest.mark.parametrize("cash_pct,mode,thresh", [
    (0.65, "aggressive", 0),
    (0.45, "active",     1),
    (0.35, "standard",   2),
    (0.10, "preserve",   None),      # PRESERVE buys nothing, so it has no threshold
])
def test_portfolio_status_surfaces_the_applicable_threshold(cash_pct, mode, thresh, monkeypatch):
    for k in ("DEPLOY_THRESH_AGGRESSIVE", "DEPLOY_THRESH_ACTIVE", "DEPLOY_THRESH_STANDARD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    import research
    ps = research.portfolio_status(cash_pct, [])
    assert ps["deploy_mode"] == mode
    assert ps["buy_threshold"] == thresh


def test_raised_aggressive_threshold_shows_up_in_the_note(monkeypatch):
    monkeypatch.setenv("DEPLOY_THRESH_AGGRESSIVE", "1")
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    import research
    assert "score ≥ 1" in research.portfolio_status(0.65, [])["note"]


# ── Change 2: entry-mix tracking ─────────────────────────────────────────────

@pytest.fixture
def em(monkeypatch):
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    monkeypatch.delenv("ENTRY_MIX_DIP_TARGET", raising=False)
    import entry_mix
    return entry_mix


def test_breakout_wins_the_classification_tie(em):
    """A Donchian break can still sit under its SMA20 and flag as both. The
    breakout is the thesis we acted on, so that is what gets recorded."""
    assert em.classify({"is_breakout": True, "is_dip": True}) == em.BREAKOUT
    assert em.classify({"is_dip": True}) == em.DIP
    assert em.classify({}) == em.SIGNAL


def test_rebuy_keeps_the_original_entry_type(em):
    em.record("AAA", em.DIP, "2026-08-12")
    em.record("AAA", em.BREAKOUT, "2026-08-13")
    assert em.load()["AAA"]["entry_type"] == em.DIP


def test_untracked_positions_are_excluded_not_assumed(em):
    """Positions predating this tracking must not be guessed into a bucket —
    that would fabricate a mix and steer real decisions off it."""
    em.record("AAA", em.BREAKOUT)
    em.record("BBB", em.DIP)
    mix = em.current(["AAA", "BBB", "LEGACY"])
    assert mix["counts"]["unknown"] == 1
    assert mix["classified"] == 2
    assert mix["dip_share"] == 0.5


def test_lean_points_away_from_the_overweight_side(em):
    for s in ("D1", "D2", "D3"):
        em.record(s, em.DIP)
    em.record("B1", em.BREAKOUT)
    assert "prefer a breakout" in em.current(["D1", "D2", "D3", "B1"])["lean"]


def test_no_classified_positions_yields_no_share(em):
    mix = em.current(["LEGACY"])
    assert mix["dip_share"] is None and mix["classified"] == 0


def test_reconcile_prunes_symbols_we_no_longer_hold(em):
    """Exits can bypass decide.py (stop_monitor, manual). Stale entries would
    otherwise skew the mix forever."""
    em.record("AAA", em.DIP)
    em.record("GONE", em.BREAKOUT)
    em.reconcile(["AAA"])
    assert list(em.load()) == ["AAA"]


def test_target_defaults_to_40pct_dip_and_is_tunable(em, monkeypatch):
    assert em.target_dip_share() == 0.40
    monkeypatch.setenv("ENTRY_MIX_DIP_TARGET", "0.60")
    assert em.target_dip_share() == 0.60


def test_entry_mix_is_fail_open(em, monkeypatch):
    """Nothing in this module may ever raise into the trade path."""
    monkeypatch.setattr(em, "load", lambda: (_ for _ in ()).throw(OSError("disk gone")))
    assert em.current(["AAA"])["lean"] == "entry-mix unavailable"
    em.record("AAA", em.DIP)      # must not raise
    em.drop("AAA")
    em.reconcile(["AAA"])


# ── Change 4: cross-sectional momentum ───────────────────────────────────────

def test_momentum_skips_the_most_recent_month():
    """The 12-1 construction must ignore the last 21 sessions. A series that
    rises for a year then crashes in the final month should still read strongly
    positive — that is the entire difference from raw 12-month return."""
    import numpy as np
    import pandas as pd
    import research
    rising = list(np.linspace(100, 300, 232))
    # Give back most of the year's gain over the final month — gradually, so this
    # tests the 12-1 skip rather than tripping the corporate-action guard.
    selloff = list(np.linspace(300, 150, 21))
    assert research.compute_momentum(pd.Series(rising + selloff))["mom_12_1"] > 100
    assert research.compute_momentum(pd.Series(rising + selloff))["mom_1m"] < -40
    # Raw 12-month return over the same series would be barely +50% — the skip is
    # what keeps this ranked on the year's trend rather than the last month's noise.


def test_momentum_needs_a_full_year():
    import pandas as pd
    import research
    assert research.compute_momentum(pd.Series(range(252)))["mom_12_1"] is None
    assert research.compute_momentum(None)["mom_12_1"] is None


def test_momentum_withholds_on_an_unadjusted_split():
    """Real 2026-08-11 case: KLAC's bars showed $912 -> $193 (an unadjusted ~10:1
    split) giving -74.6%, when the adjusted truth was +144%. A sign flip on a
    decile-ranked input. Withhold rather than guess."""
    import pandas as pd
    import research
    prices = [900.0] * 200 + [90.0] * 53          # 10:1 split mid-window
    out = research.compute_momentum(pd.Series(prices))
    assert out["mom_12_1"] is None and out["mom_suspect"] is True


def test_momentum_tolerates_a_large_but_genuine_move():
    """SATS really did move +70.3% in one session. That must NOT be mistaken for
    a corporate action — the upper bound sits at +100% for that reason."""
    import pandas as pd
    import research
    prices = [100.0] * 150 + [170.0] * 103        # a genuine +70% day
    out = research.compute_momentum(pd.Series(prices))
    assert out["mom_12_1"] is not None and out["mom_suspect"] is False


def test_suspect_momentum_never_reaches_the_sleeve(monkeypatch):
    """A withheld reading must produce no rank, and therefore no candidate."""
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    import research
    sigs = [{"symbol": "SPLIT", "mom_rank": None, "mom_suspect": True,
             "allocation_headroom": 0.02, "buy_notional": 500}]
    assert research.portfolio_status(0.35, sigs)["momentum_candidates"] == []


def test_momentum_sleeve_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMENTUM_SLEEVE", raising=False)
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    import research
    sigs = [{"symbol": "A", "score": 1, "mom_rank": 99.0, "mom_12_1": 40.0,
             "allocation_headroom": 0.02, "buy_notional": 500}]
    assert research.portfolio_status(0.35, sigs)["momentum_candidates"] == []


def test_momentum_sleeve_filters_to_top_decile_with_headroom(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    monkeypatch.setenv("RISK_STATE_DIR", tempfile.mkdtemp())
    import research
    sigs = [
        {"symbol": "TOP",     "mom_rank": 99.0, "allocation_headroom": 0.02, "buy_notional": 500},
        {"symbol": "MID",     "mom_rank": 50.0, "allocation_headroom": 0.02, "buy_notional": 500},
        {"symbol": "NOROOM",  "mom_rank": 99.0, "allocation_headroom": 0.0,  "buy_notional": 0},
        {"symbol": "HELD",    "mom_rank": 99.0, "allocation_headroom": 0.02, "position": {"qty": 1}},
        {"symbol": "EXTENDED","mom_rank": 99.0, "allocation_headroom": 0.02, "overextended": True},
        {"symbol": "NEWLIST", "mom_rank": None, "allocation_headroom": 0.02},
    ]
    got = [c["symbol"] for c in research.portfolio_status(0.35, sigs)["momentum_candidates"]]
    assert got == ["TOP"]


def test_xs_momentum_buys_top_decile_and_exits_on_rank_decay():
    from strategies.catalog import xs_momentum
    sigs = [{"symbol": "HOT", "mom_rank": 95.0, "mom_12_1": 60.0},
            {"symbol": "MEH", "mom_rank": 60.0, "mom_12_1": 10.0},
            {"symbol": "COLD", "mom_rank": 20.0, "mom_12_1": -5.0}]
    out = xs_momentum(sigs, 100_000, {"COLD": 5000})
    assert ("HOT", "buy") in {(i.symbol, i.side) for i in out}
    assert ("COLD", "sell") in {(i.symbol, i.side) for i in out}
    assert not any(i.symbol == "MEH" for i in out)   # held-nothing, rank too low to buy


def test_xs_momentum_respects_the_overextended_block():
    from strategies.catalog import xs_momentum
    sigs = [{"symbol": "X", "mom_rank": 99.0, "mom_12_1": 80.0, "overextended": True}]
    assert not any(i.side == "buy" for i in xs_momentum(sigs, 100_000, {}))


# ── Change 3: covered calls — NOT SHIPPED in this extraction ────────────────
#
# The covered-call sleeve (a variance-risk-premium harvester with cover,
# strike, timing and reconcile guards) is deliberately excluded here: it is
# the one premium-SELLING path, and shipping an options-writing sleeve in a
# public starter repo invites exactly the accident this project's risk rails
# exist to prevent. Its tests lived here and were removed with it.
