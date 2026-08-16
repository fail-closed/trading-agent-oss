"""Snapshot-diff reconciliation (external audit item 2.2).

`ledger.reconcile()` used to check only `symbol in known` — so a 10:1 split, a
covered-call assignment, or an exit that reported success without selling all
returned `ok: True`. Quantity and cost-basis drift were invisible.

The audit proposed Σbuys − Σsells. That cannot work here: buys are placed by
NOTIONAL (the broker picks the share count), partial fills are normal, and
assignment removes shares with no equity order — the running total would drift
into permanent false alarms. These tests pin the snapshot-diff behavior instead,
including the false-positive cases that would cause alert fatigue.
"""
import json

import pytest

import ledger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", str(tmp_path))
    return tmp_path


def _seed(positions):
    """First call seeds baseline + snapshot and never alerts."""
    rep = ledger.reconcile("core", positions)
    assert rep["ok"] and rep.get("note") == "baseline seeded"


def _issues(positions):
    return [d["issue"] for d in ledger.reconcile("core", positions)["drift"]]


def test_stable_book_is_clean():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    assert _issues({"CAT": {"qty": 4.0, "avg_entry": 300.0}}) == []


def test_unknown_symbol_still_flagged():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    issues = _issues({"CAT": {"qty": 4.0, "avg_entry": 300.0},
                      "MYSTERY": {"qty": 3.0, "avg_entry": 50.0}})
    assert any("no order in ledger" in i for i in issues)


def test_unexplained_quantity_change_is_caught():
    """The old reconcile returned ok: True here — the symbol was 'known'."""
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    issues = _issues({"CAT": {"qty": 2.0, "avg_entry": 300.0}})
    assert any("quantity changed 4 → 2 with no order" in i for i in issues)


def test_split_is_named_not_just_flagged():
    """KLAC's unadjusted 10:1 read as a −75% collapse in our own data (§6)."""
    _seed({"KLAC": {"qty": 2.0, "avg_entry": 900.0}})
    issues = _issues({"KLAC": {"qty": 20.0, "avg_entry": 90.0}})
    assert any("10:1 split" in i for i in issues), issues


def test_reverse_split_is_named():
    _seed({"AAA": {"qty": 20.0, "avg_entry": 5.0}})
    issues = _issues({"AAA": {"qty": 4.0, "avg_entry": 25.0}})
    assert any("1:5 reverse split" in i for i in issues), issues


def test_cost_basis_jump_without_a_buy_is_flagged():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    issues = _issues({"CAT": {"qty": 4.0, "avg_entry": 500.0}})
    assert any("cost basis moved" in i for i in issues)


def test_vanished_position_without_a_sell_is_flagged():
    """An exit that reported success but didn't sell leaves this fingerprint."""
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    assert any("disappeared" in i for i in _issues({}))


# ── False positives are the real risk: alert fatigue kills the alarm ────────

def test_change_explained_by_an_order_is_not_drift():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    ledger.record({"event": "submitted", "account": "core", "symbol": "CAT",
                   "side": "buy", "notional": 600.0})
    assert _issues({"CAT": {"qty": 6.0, "avg_entry": 300.0}}) == []


def test_sold_position_with_a_sell_order_is_not_drift():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    ledger.record({"event": "submitted", "account": "core", "symbol": "CAT",
                   "side": "sell", "qty": "all"})
    assert _issues({}) == []


def test_dividend_reinvestment_dust_is_ignored():
    """Live DRIP adds a fraction of a share with no order of ours; alerting on
    it every day would train us to ignore the alarm."""
    _seed({"CAT": {"qty": 100.0, "avg_entry": 300.0}})
    assert _issues({"CAT": {"qty": 100.4, "avg_entry": 300.0}}) == []


def test_price_move_alone_is_never_drift():
    """Only qty and BASIS matter — basis doesn't move when the market does."""
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    assert _issues({"CAT": {"qty": 4.0, "avg_entry": 300.0}}) == []


def test_new_position_from_a_buy_is_not_drift():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    ledger.record({"event": "submitted", "account": "core", "symbol": "BKR",
                   "side": "buy", "notional": 500.0})
    assert _issues({"CAT": {"qty": 4.0, "avg_entry": 300.0},
                    "BKR": {"qty": 10.0, "avg_entry": 50.0}}) == []


# ── Compatibility & robustness ─────────────────────────────────────────────

def test_legacy_plain_qty_dict_still_works():
    """reconcile_positions.py now sends the rich form, but the plain {sym: qty}
    shape must keep working for any other caller."""
    _seed({"CAT": 4.0})
    assert _issues({"CAT": 4.0}) == []
    assert any("quantity changed" in i for i in _issues({"CAT": 2.0}))


def test_corrupt_snapshot_degrades_to_symbol_check(_isolate):
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    (_isolate / "recon_snapshot.json").write_text("{ not json")
    assert _issues({"CAT": {"qty": 2.0, "avg_entry": 300.0}}) == []   # no crash


def test_snapshot_advances_so_drift_is_reported_once():
    _seed({"CAT": {"qty": 4.0, "avg_entry": 300.0}})
    assert len(_issues({"CAT": {"qty": 2.0, "avg_entry": 300.0}})) == 1
    assert _issues({"CAT": {"qty": 2.0, "avg_entry": 300.0}}) == [], \
        "a reported drift must not re-alert every 8 hours forever"
