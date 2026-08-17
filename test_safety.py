"""
Tests for the real-money safety layer: account live-gating, the risk circuit
breaker, and order idempotency. Pure-logic — no network. Run: pytest -q
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import accounts
import ledger
import risk_guard


# ── 1. Live gating defaults to paper ──────────────────────────────────────────

def test_is_live_defaults_false(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    assert accounts.is_live("core") is False


def test_is_live_requires_all_conditions(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_ACCOUNTS", "core")
    # live keys NOT set → still paper
    monkeypatch.delenv("ALPACA_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET_KEY", raising=False)
    assert accounts.is_live("core") is False
    # now all three conditions hold → live
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "k")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "s")
    assert accounts.is_live("core") is True
    # account not opted in → paper even with keys + flag
    monkeypatch.setenv("LIVE_ACCOUNTS", "options")
    assert accounts.is_live("core") is False


# ── 2. Risk circuit breaker (pure evaluate) ───────────────────────────────────

L = risk_guard.DEFAULT_LIMITS


def test_sell_always_allowed():
    ok, _ = risk_guard.evaluate("core", "sell", None, 90000, 100000, 100000, 0, 0, L)
    assert ok  # risk-reducing even when down big


def test_daily_loss_limit_blocks(monkeypatch):
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    ok, reason = risk_guard.evaluate("core", "buy", 1000, 95000, 100000, 100000, 0, 0, L)
    assert not ok and "daily loss" in reason


def test_monthly_profit_lock_blocks_buys_not_sells():
    # +6% MTD ≥ 5% lock
    ok_buy, r = risk_guard.evaluate("core", "buy", 1000, 106000, 106000, 100000, 0, 0, L)
    assert not ok_buy and "profit-lock" in r
    ok_sell, _ = risk_guard.evaluate("core", "sell", None, 106000, 106000, 100000, 0, 0, L)
    assert ok_sell


def test_deposit_not_counted_as_profit_lock():
    # A $3k deposit into a $1k account (equity 3992, anchor 999.70) would read as
    # +299% MTD and trip the profit-lock — netting the deposit out (month_contrib)
    # leaves the real ~-0.8% trading return, so buys flow. (Live bug, 2026-06-29.)
    blocked, r = risk_guard.evaluate("core", "buy", 200, 3992.0, 3992.0, 999.70, 0, 0, L)
    assert not blocked and "profit-lock" in r          # without netting → blocked
    ok, _ = risk_guard.evaluate("core", "buy", 200, 3992.0, 3992.0, 999.70, 0, 0, L,
                                month_contrib=3000.0)
    assert ok                                          # deposit netted out → allowed


def test_deposit_netting_never_fabricates_loss_halt():
    # A real -20% loss masked by a $3k deposit (equity 3800, anchor 1000): the
    # profit-lock must NOT fire, and the deposit netting must NOT manufacture a
    # monthly loss halt (loss halts use raw return, which is +280% here).
    ok, r = risk_guard.evaluate("core", "buy", 50, 3800.0, 3800.0, 1000.0, 0, 0, L,
                                month_contrib=3000.0)
    assert ok and "loss halt" not in r and "profit-lock" not in r


def test_max_order_size_blocks():
    ok, reason = risk_guard.evaluate("core", "buy", 20000, 100000, 100000, 100000, 0, 0, L)
    assert not ok and "% of equity" in reason


def test_max_orders_per_day_blocks():
    ok, reason = risk_guard.evaluate("core", "buy", 1000, 100000, 100000, 100000, 0,
                                     L["max_orders_per_day"], L)
    assert not ok and "max orders" in reason


def test_max_position_blocks():
    ok, reason = risk_guard.evaluate("core", "buy", 5000, 100000, 100000, 100000,
                                     held_value=9000, orders_today=0, limits=L)
    assert not ok and "position would exceed" in reason


def test_normal_buy_passes():
    ok, _ = risk_guard.evaluate("core", "buy", 5000, 100000, 100000, 100000, 0, 0, L)
    assert ok


def test_kill_switch_blocks(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "true")
    ok, reason = risk_guard.evaluate("core", "buy", 1000, 100000, 100000, 100000, 0, 0, L)
    assert not ok and "KILL_SWITCH" in reason


# ── 3. Idempotency ────────────────────────────────────────────────────────────

def test_client_order_id_deterministic():
    a = ledger.client_order_id("core", "SPY", "buy", notional=5000)
    b = ledger.client_order_id("core", "SPY", "buy", notional=5000)
    c = ledger.client_order_id("core", "SPY", "buy", notional=6000)
    assert a == b           # same intended order → same id (dedupe on re-run)
    assert a != c           # different size → different id
    assert a.startswith("ta-core-")


# ── non-finite inputs must fail CLOSED, but never gate a sell ────────────────
#
# Open item from the 2026-06-21 security audit ("NaN-notional risk bypass"),
# closed 2026-08-17. NaN defeats every comparison-based guard: `NaN > cap` is
# False for any cap, and NaN is truthy, so `if notional and notional > cap`
# APPROVES it. Infinity was always caught (`inf > cap` is True) — which is
# exactly why the gap survived review.

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -500, "abc", [1]])
def test_buy_with_a_non_finite_notional_fails_closed(bad):
    ok, why = risk_guard.evaluate("core", "buy", bad, 100000, 100000, 100000, 0, 0, L)
    assert not ok, f"{bad!r} notional was APPROVED by the risk gate"
    assert "invalid notional" in why


def test_non_finite_equity_also_fails_closed():
    ok, _ = risk_guard.evaluate("core", "buy", 5000, float("nan"), 100000, 100000, 0, 0, L)
    assert not ok


def test_a_sell_is_never_gated_on_input_hygiene():
    """The critical ordering. A first cut of the guard sat ABOVE the sell bypass
    and blocked sells with a bad notional — which would stop a stop-loss firing,
    the one thing this module guarantees can always happen. Risk-reducing orders
    must clear regardless of how mangled the inputs are."""
    for bad in (float("nan"), float("inf"), -1, "abc", None):
        ok, _ = risk_guard.evaluate("core", "sell", bad, 100000, 100000, 100000, 0, 0, L)
        assert ok, f"sell was blocked by a {bad!r} notional — a stop-loss could not fire"


def test_a_valid_buy_is_unaffected():
    ok, why = risk_guard.evaluate("core", "buy", 5000, 100000, 100000, 100000, 0, 0, L)
    assert ok and why == "ok"


def test_none_notional_still_allowed_for_sizing_free_checks():
    """`check_order(side='buy')` is called without a notional in places that only
    want the halt/kill-switch verdict. None must stay a valid 'not applicable'."""
    ok, _ = risk_guard.evaluate("core", "buy", None, 100000, 100000, 100000, 0, 0, L)
    assert ok
