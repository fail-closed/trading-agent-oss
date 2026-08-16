"""Ladder tranche-2 recovery (external audit item 1.1).

The bug: `trade.py` stops polling for a fill after 30s, and decide.py placed
tranche 2 only when a fill price came back. A tranche 1 that filled at 31s — the
normal case for a limit order — lost its 40% tranche permanently and silently.

What matters here is that intent survives, that a resolved tranche is never
retried (double-buy), and that nothing is placed when tranche 1 never filled.
"""
import json
import types

import pytest

import ladder


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGER_DIR", str(tmp_path))
    return tmp_path


def _order(status, filled_qty=0, avg=0.0):
    return types.SimpleNamespace(status=types.SimpleNamespace(value=status),
                                 filled_qty=filled_qty, filled_avg_price=avg)


def _placer(calls, status="placed"):
    def place(symbol, notional, entry_price):
        calls.append((symbol, notional, entry_price))
        return {"status": status, "limit_price": round(entry_price * 0.96, 2)}
    return place


def test_pending_survives_a_slow_fill_and_places_at_the_actual_price():
    ladder.record_pending("CAT", "oid-1", 2000.0)
    assert [r["symbol"] for r in ladder.pending()] == ["CAT"]

    calls = []
    out = ladder.resolve(lambda oid: _order("filled", 6.0, 250.0), _placer(calls))
    assert calls == [("CAT", 2000.0, 250.0)], "tranche 2 must anchor on the real fill"
    assert out[0]["result"] == "placed"
    assert ladder.pending() == [], "resolved tranche must not be retried"


def test_still_open_order_is_kept_for_the_next_session():
    ladder.record_pending("CAT", "oid-1", 2000.0)
    calls = []
    out = ladder.resolve(lambda oid: _order("new"), _placer(calls))
    assert calls == [] and out == []
    assert len(ladder.pending()) == 1, "an unfilled order must stay pending"


def test_partial_fill_then_cancel_still_places_on_what_filled():
    """A DAY limit that partially fills and is canceled at the close DID buy —
    the ladder's second leg is still the right trade."""
    ladder.record_pending("CAT", "oid-1", 2000.0)
    calls = []
    ladder.resolve(lambda oid: _order("canceled", 2.5, 248.0), _placer(calls))
    assert calls == [("CAT", 2000.0, 248.0)]
    assert ladder.pending() == []


@pytest.mark.parametrize("status", ["canceled", "expired", "rejected"])
def test_terminal_without_any_fill_drops_the_tranche(status):
    ladder.record_pending("CAT", "oid-1", 2000.0)
    calls = []
    out = ladder.resolve(lambda oid: _order(status, 0, 0), _placer(calls))
    assert calls == [], "nothing was bought — tranche 2 would be an unplanned entry"
    assert out[0]["result"] == "dropped"
    assert ladder.pending() == []


def test_stale_pending_is_aged_out(monkeypatch):
    ladder.record_pending("CAT", "oid-1", 2000.0)
    rows = ladder.pending()
    rows[0]["created_at"] = "2020-01-01T10:00:00-05:00"
    ladder._save(rows)
    calls = []
    out = ladder.resolve(lambda oid: _order("new"), _placer(calls))
    assert calls == [] and out[0]["result"] == "dropped"
    assert "stale" in out[0]["detail"]
    assert ladder.pending() == []


def test_unknown_order_is_retried_then_aged_out():
    def boom(oid):
        raise RuntimeError("order not found")

    ladder.record_pending("CAT", "oid-1", 2000.0)
    assert ladder.resolve(boom, _placer([])) == []
    assert len(ladder.pending()) == 1, "a transient broker error must not drop intent"

    rows = ladder.pending()
    rows[0]["created_at"] = "2020-01-01T10:00:00-05:00"
    ladder._save(rows)
    out = ladder.resolve(boom, _placer([]))
    assert out[0]["result"] == "dropped" and ladder.pending() == []


def test_a_failed_placement_is_not_retried_forever():
    """place_ladder_tranche2 fails CLOSED on risk-guard/cash/allocation doubt.
    Retrying next session would re-run those checks against a stale −4% anchor."""
    ladder.record_pending("CAT", "oid-1", 2000.0)
    calls = []
    out = ladder.resolve(lambda oid: _order("filled", 6.0, 250.0),
                         _placer(calls, status="skipped"))
    assert out[0]["result"] == "skipped"
    assert ladder.pending() == []


def test_multiple_symbols_resolve_independently():
    ladder.record_pending("CAT", "oid-1", 1000.0)
    ladder.record_pending("BKR", "oid-2", 500.0)
    calls = []
    orders = {"oid-1": _order("filled", 4.0, 250.0), "oid-2": _order("new")}
    ladder.resolve(lambda oid: orders[oid], _placer(calls))
    assert calls == [("CAT", 1000.0, 250.0)]
    assert [r["symbol"] for r in ladder.pending()] == ["BKR"]


def test_recording_the_same_order_twice_does_not_duplicate():
    ladder.record_pending("CAT", "oid-1", 2000.0)
    ladder.record_pending("CAT", "oid-1", 2000.0)
    assert len(ladder.pending()) == 1


def test_corrupt_state_file_is_survivable(_isolate):
    (_isolate / ladder.STATE_FILE).write_text("{{{not json")
    assert ladder.pending() == []
    ladder.record_pending("CAT", "oid-1", 2000.0)
    assert len(ladder.pending()) == 1


def test_decide_wires_the_recovery_before_it_decides():
    """Recovery must run at session start — every intraday cycle is another
    chance for a tranche 1 that filled after the poll window."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent.joinpath("decide.py").read_text()
    assert "resolve_pending_ladders()" in src
    assert "ladder.record_pending(" in src
    assert src.index("resolve_pending_ladders()") < src.index("exits = execute_exits(")
