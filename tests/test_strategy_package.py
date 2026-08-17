"""`strategies/` — the package that was invisible to CI until 2026-08-16.

`test_coverage_floor` scanned top-level `*.py` only, so this whole package went
unchecked from the day it was created. Widening that scan (OPERATIONS §6 row 18)
surfaced `strategies.base` and `strategies.daytrade` as untested, and
`strategies/daytrade.py`'s own docstring claims "the whole strategy is
unit-testable without market access" — a claim nothing had ever tested.

Priority here is the ORDER of overrides. `build_intents` resolves three
competing conditions (flatten-by-close, sleeve stop, new entries) and getting
that order wrong is how a sleeve holds overnight or buys into its own stop-out.
"""
import pytest

from strategies.base import TradeIntent as BaseIntent
from strategies import daytrade as dt

RISK = {
    "sleeve_stop_pct": 0.02,
    "sleeve_take_profit_pct": 0.01,
    "max_concurrent_positions": 3,
    "per_position_pct_of_sleeve": 0.20,
    "per_position_stop_pct": 0.01,
}


def sig(symbol, score=3, orb=1, vwap=1, mom=1, pct=1.0):
    return {"symbol": symbol, "score": score,
            "signals": {"opening_range": {"signal": orb}, "vwap": {"signal": vwap},
                        "momentum": {"signal": mom, "pct": pct}}}


# ── the entry gate ───────────────────────────────────────────────────────────

def test_entry_needs_all_three_signals_aligned():
    assert dt.qualifies(sig("A"))
    for kw in ("orb", "vwap", "mom"):
        assert not dt.qualifies(sig("A", **{kw: 0})), f"{kw} alone must veto the entry"


def test_entry_needs_the_score_floor_even_with_all_three():
    """Three aligned signals but a score below 3 is not a breakout — the score
    carries information the three booleans do not."""
    assert not dt.qualifies(sig("A", score=2))


# ── sleeve-level gates ───────────────────────────────────────────────────────

@pytest.mark.parametrize("equity,expected", [
    (10_000, "ok"), (10_100, "halt"), (10_150, "halt"), (9_800, "liquidate"), (9_700, "liquidate"),
])
def test_sleeve_state_thresholds(equity, expected):
    assert dt.sleeve_state(equity, 10_000, RISK)["action"] == expected


def test_sleeve_state_survives_a_zero_baseline():
    """day_start_equity of 0 must not raise — a divide would kill the 5-min loop."""
    assert dt.sleeve_state(1000, 0, RISK) == {"action": "ok", "pl_pct": 0.0}


def test_halt_blocks_new_entries_but_does_not_sell():
    out = dt.build_intents([sig("A")], {"HELD"}, 10_100, 10_000, RISK, near_close=False)
    assert out["sleeve"]["action"] == "halt"
    assert out["intents"] == [], "a +1% halt locks the win; it does not liquidate"


# ── override order: the part most likely to be got wrong ─────────────────────

def test_flatten_by_close_beats_everything():
    """Even with the sleeve profitable and fresh candidates on the board, the
    only thing that may happen near the close is going flat. No overnight holds."""
    out = dt.build_intents([sig("A")], {"HELD"}, 10_050, 10_000, RISK, near_close=True)
    assert [(i.symbol, i.side, i.qty) for i in out["intents"]] == [("HELD", "sell", "all")]


def test_sleeve_stop_liquidates_and_never_buys_in_the_same_cycle():
    """The −2% path must not also open entries — that would buy into the exact
    condition that just triggered the stop."""
    out = dt.build_intents([sig("A")], {"HELD"}, 9_700, 10_000, RISK, near_close=False)
    assert all(i.side == "sell" for i in out["intents"])
    assert {i.symbol for i in out["intents"]} == {"HELD"}


def test_liquidate_with_nothing_held_falls_through_to_no_entries():
    out = dt.build_intents([sig("A")], set(), 9_700, 10_000, RISK, near_close=False)
    assert out["intents"] == []


# ── capacity and ranking ─────────────────────────────────────────────────────

def test_capacity_limits_entries_and_ranks_by_momentum():
    sigs = [sig("SLOW", pct=0.5), sig("FAST", pct=9.0), sig("MID", pct=3.0)]
    out = dt.build_intents(sigs, {"HELD"}, 10_000, 10_000, RISK, near_close=False)
    assert [i.symbol for i in out["intents"]] == ["FAST", "MID"], "2 open slots, strongest first"


def test_already_held_symbols_are_not_re_entered():
    out = dt.build_intents([sig("HELD")], {"HELD"}, 10_000, 10_000, RISK, near_close=False)
    assert out["intents"] == []


def test_full_book_takes_no_entries():
    out = dt.build_intents([sig("A")], {"X", "Y", "Z"}, 10_000, 10_000, RISK, near_close=False)
    assert out["intents"] == []


def test_entries_carry_size_and_a_per_position_stop():
    out = dt.build_intents([sig("A")], set(), 10_000, 10_000, RISK, near_close=False)
    i = out["intents"][0]
    assert i.side == "buy" and i.notional == 2000.0     # 20% of sleeve
    assert i.stop_pct == 0.01, "a day-trade entry without its stop is an overnight risk"


# ── the shared contract ──────────────────────────────────────────────────────

def test_trade_intent_contract_is_what_the_runner_consumes():
    """strategy_runner.py reads these field names off every strategy's return
    value. Renaming one breaks every strategy at once, silently."""
    i = BaseIntent(symbol="A", side="buy", notional=100.0, qty=None, reason="r")
    assert (i.symbol, i.side, i.notional, i.qty, i.reason) == ("A", "buy", 100.0, None, "r")
    assert BaseIntent("A", "sell").notional is None
