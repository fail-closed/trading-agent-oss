"""
risk_guard.py — central pre-order circuit breaker for real-money safety.

EVERY order, on every account, must pass check_order() before it reaches the
broker. This is the one place risk is enforced in code (not prose): a global
kill-switch, a per-day trading halt, a daily-loss limit, the monthly +5%
profit-lock / −X% halt, max order size, max orders/day, and max position.

State (equity anchors, halts) persists on the volume (RISK_STATE_DIR, default
'signals') so limits survive restarts. Pure-logic helpers take state as args so
they unit-test without files or network.

Usage (from trade.py / options_trade.py):
    import risk_guard
    ok, reason = risk_guard.check_order(account, side, notional=..., equity=...,
                                        held_value=..., orders_today=...)
    if not ok: fail(reason)
"""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE_DIR = os.getenv("RISK_STATE_DIR", "signals")
STATE_FILE = "risk_state.json"

# Defaults — overridable per account via risk_limits.json
DEFAULT_LIMITS = {
    "daily_loss_limit_pct":    0.04,   # halt the account for the day at −4%
    "monthly_profit_lock_pct": 0.05,   # at +5% MTD, stop opening new positions
    "monthly_loss_halt_pct":   0.06,   # at −6% MTD, halt the account for the month
    "max_order_pct_of_equity": 0.10,   # no single order > 10% of account equity
    "max_orders_per_day":      15,     # circuit-break runaway loops
    "max_position_pct":        0.10,   # no position > 10% of account equity
}


def _state_path() -> Path:
    return Path(STATE_DIR) / STATE_FILE


def _load() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save(st: dict):
    # Atomic write: this is real-money risk state (daily-loss anchors, halts)
    # written on every order. A truncated write would wipe the day's guard memory.
    from io_utils import write_json_atomic
    write_json_atomic(str(_state_path()), st)


_ENV_LIMITS = {                     # env var -> limit key
    "RISK_DAILY_LOSS_PCT":     "daily_loss_limit_pct",
    "RISK_MONTHLY_LOSS_PCT":   "monthly_loss_halt_pct",
    "RISK_MONTHLY_PROFIT_PCT": "monthly_profit_lock_pct",
    "RISK_MAX_ORDER_PCT":      "max_order_pct_of_equity",
    "RISK_MAX_POSITION_PCT":   "max_position_pct",
    "RISK_MAX_ORDERS_DAY":     "max_orders_per_day",
}


def _limits(account: str) -> dict:
    """DEFAULT_LIMITS < risk_limits.json < environment.

    Env overrides let one profile run a different risk appetite without a code
    change or a divergent config file, and let a change be reverted by unsetting
    a variable. risk_limits.json still wins over the defaults; env wins over both."""
    cfg = {}
    p = Path("risk_limits.json")
    if p.exists():
        try:
            cfg = json.loads(p.read_text()).get("per_account", {}).get(account, {})
        except Exception:
            cfg = {}
    out = {**DEFAULT_LIMITS, **cfg}
    for env_key, lim_key in _ENV_LIMITS.items():
        raw = os.getenv(env_key, "").strip()
        if raw:
            try:
                out[lim_key] = int(raw) if lim_key == "max_orders_per_day" else float(raw)
            except ValueError:
                pass
    return out


def benchmark_mtd(account: str) -> float:
    """Benchmark return since this account's month anchor, or None.

    Only consulted when RISK_MONTHLY_HALT_VS_BENCH is on. Cached to the risk
    state dir per (date, anchor) so the order path costs at most one network
    call per day, and returns None on ANY failure — callers then fall back to the
    absolute halt, which is the stricter behaviour. A benchmark outage must never
    widen the safety net."""
    if os.getenv("RISK_MONTHLY_HALT_VS_BENCH", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    sym = os.getenv("BENCHMARK_SYMBOL", "SPY")
    since = anchor_since(account, "month")
    today = datetime.now(ET).strftime("%Y-%m-%d")
    cache_path = Path(STATE_DIR) / "bench_mtd.json"
    key = f"{sym}|{since}|{today}"
    try:
        if cache_path.exists():
            hit = json.loads(cache_path.read_text()).get(key)
            if hit is not None:
                return float(hit)
    except Exception:
        pass
    try:
        import accounts
        from datetime import date as _date, timedelta as _td
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        k, s = accounts.data_keys()
        d0 = _date.fromisoformat(since) - _td(days=7)
        bars = StockHistoricalDataClient(k, s).get_stock_bars(StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame.Day,
            start=datetime(d0.year, d0.month, d0.day),
            feed=accounts.data_feed())).data.get(sym, [])
        pts = [(b.timestamp.date().isoformat(), float(b.close)) for b in bars]
        base = next((p for d, p in pts if d >= since), None)
        if not base or not pts:
            return None
        val = pts[-1][1] / base - 1.0
        try:
            cur = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        except Exception:
            cur = {}
        cur = {key: val}                       # single entry — keys are date-scoped
        from io_utils import write_json_atomic
        write_json_atomic(str(cache_path), cur)
        return val
    except Exception as e:
        print(f"  [risk_guard] benchmark MTD unavailable ({str(e)[:60]}) — "
              f"falling back to absolute monthly halt", file=__import__("sys").stderr)
        return None


def kill_switch_on() -> bool:
    """Hard global stop — env KILL_SWITCH=true or a flag in risk_limits.json."""
    if os.getenv("KILL_SWITCH", "false").lower() == "true":
        return True
    p = Path("risk_limits.json")
    if p.exists():
        try:
            return bool(json.loads(p.read_text()).get("global", {}).get("kill_switch"))
        except Exception:
            return False
    return False


# ── Equity anchors (day / month) ──────────────────────────────────────────────

def anchor_equity(account: str, current: float, period: str) -> float:
    """Return the account's equity at the start of this day/month, setting it on
    first observation of a new period. period: 'day' | 'month'."""
    now = datetime.now(ET)
    key = now.strftime("%Y-%m-%d") if period == "day" else now.strftime("%Y-%m")
    st = _load()
    anchors = st.setdefault("anchors", {}).setdefault(account, {})
    bucket = anchors.setdefault(period, {})
    if bucket.get("key") != key:
        bucket["key"], bucket["value"] = key, current
        bucket["set_at"] = now.isoformat()   # boundary for netting out in-period deposits
        _save(st)
    return bucket["value"]


def anchor_since(account: str, period: str) -> str:
    """YYYY-MM-DD after which in-period external contributions (deposits/withdrawals)
    should be netted out of the profit-lock return: the anchor's `set_at` date if
    recorded, else the start of the current period (legacy anchors predating set_at).
    Same-day-as-anchor deposits are excluded by the caller's strict `>` — they are
    already baked into the anchor value."""
    b = _load().get("anchors", {}).get(account, {}).get(period, {})
    sa = b.get("set_at")
    if sa:
        return sa[:10]
    now = datetime.now(ET)
    return now.strftime("%Y-%m-01") if period == "month" else now.strftime("%Y-%m-%d")


def set_halt(account: str, reason: str, scope: str = "day"):
    st = _load()
    now = datetime.now(ET)
    key = now.strftime("%Y-%m-%d") if scope == "day" else now.strftime("%Y-%m")
    st.setdefault("halts", {})[account] = {"scope": scope, "key": key, "reason": reason,
                                           "at": now.isoformat()}
    _save(st)


def halt_reason(account: str):
    """Return active halt reason for the account, or None (auto-expires by period)."""
    st = _load()
    h = st.get("halts", {}).get(account)
    if not h:
        return None
    now = datetime.now(ET)
    key = now.strftime("%Y-%m-%d") if h.get("scope") == "day" else now.strftime("%Y-%m")
    return h["reason"] if h.get("key") == key else None


# ── The gate ──────────────────────────────────────────────────────────────────

def _is_finite_amount(x) -> bool:
    """True only for a real, finite, non-negative money amount.

    THE one place that decides what counts as a valid amount, so trade.py and
    risk_guard cannot disagree about it (OPERATIONS §8 habit 6). Rejects NaN,
    ±inf, negatives, and anything non-numeric — a string that sneaks in would
    otherwise raise mid-comparison inside the order path rather than fail closed.
    """
    import math
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= 0


def evaluate(account, side, notional, equity, day_start, month_start,
             held_value, orders_today, limits, month_contrib=0.0,
             bench_mtd_pct=None) -> tuple:
    """Pure decision logic (no I/O). Returns (ok: bool, reason: str).

    month_contrib: net external cash flow (deposits − withdrawals) into the account
    since the month anchor was set. Netted out of the profit-lock check ONLY so a
    deposit isn't mistaken for a trading gain."""
    # Kill-switch is the only thing that blocks EVERYTHING (human takes over).
    if kill_switch_on():
        return False, "KILL_SWITCH active — all trading halted"

    if equity is not None and not _is_finite_amount(equity):
        return False, f"invalid equity ({equity!r}) — not a finite non-negative number"

    # Sells/exits are always permitted otherwise — risk-reducing, and must work
    # even when the account is loss-halted so stop-losses can fire and you're
    # never trapped in a position.
    if side == "sell":
        return True, "ok (sell/exit always permitted)"

    # BUYS ONLY, and deliberately placed after the sell bypass above. An earlier
    # version of this check sat before it and blocked SELLS with a bad notional —
    # which would stop a stop-loss firing, the one thing this file guarantees can
    # always happen. Risk-reducing orders must never be gated on input hygiene.
    #
    # NaN fails OPEN through every numeric comparison below it: `NaN > x` is False
    # for any x, and NaN is truthy, so `if notional and notional > cap` PASSES.
    # Infinity was already caught (`inf > cap` is True), which is precisely why
    # this looked fine. Open since the 2026-06-21 audit ("NaN-notional risk
    # bypass"); nothing exercised it until 2026-08-17.
    #
    # Scope, stated honestly: trade.py rejects a non-finite notional at its own
    # entry (before it ever calls here), so this is defence in depth, not the last
    # line. What it fixes is the gate REPORTING "ok" for an order that can never
    # be placed — an approval in the decision log for a trade that does not exist.
    if notional is not None and not _is_finite_amount(notional):
        return False, f"invalid notional ({notional!r}) — not a finite non-negative number"
    if equity is not None and not _is_finite_amount(equity):
        return False, f"invalid equity ({equity!r}) — not a finite non-negative number"

    daily = (equity / day_start - 1.0) if day_start else 0.0
    monthly = (equity / month_start - 1.0) if month_start else 0.0
    # Profit-lock must not fire on deposited cash — a deposit raises equity but is not
    # a trading gain. Net out external contributions for the profit-lock check ONLY.
    # Loss halts stay on raw return: a deposit inflates equity upward, so it can never
    # *fabricate* a loss (only mask one — per-position stop-losses cover that), whereas
    # netting an over-counted deposit into a loss check could falsely halt the account
    # for the whole period.
    monthly_locked = (((equity - max(0.0, month_contrib)) / month_start - 1.0)
                      if month_start else 0.0)

    if daily <= -abs(limits["daily_loss_limit_pct"]):
        return False, f"daily loss limit hit ({daily*100:.1f}%) — account halted for the day"

    # Monthly halt, optionally measured against the benchmark.
    #
    # Absolute monthly loss is the wrong trigger for a long-only equity book: on
    # an 80%-invested portfolio a -6% month occurs in 11% of months from ordinary
    # market movement (6y of SPY), and each one halts the account for the REST of
    # the month — forcing it out after a drawdown and out of the recovery. That is
    # pro-cyclical: it does the opposite of what the strategy needs.
    #
    # Falling 6% while the market falls 8% is the strategy WORKING. Falling 6%
    # while the market rises is the strategy BREAKING. With bench_mtd_pct supplied
    # the halt measures that difference, so it catches malfunction instead of
    # weather. Absolute stays the fallback whenever the benchmark is unavailable —
    # a data failure must not silently remove the safety net.
    monthly_vs_bench = (monthly - bench_mtd_pct) if bench_mtd_pct is not None else None
    if monthly_vs_bench is not None:
        if monthly_vs_bench <= -abs(limits["monthly_loss_halt_pct"]):
            return False, (f"monthly halt: {monthly*100:.1f}% vs benchmark "
                           f"{bench_mtd_pct*100:+.1f}% = {monthly_vs_bench*100:.1f}% excess "
                           f"— underperforming, account halted for the month")
    elif monthly <= -abs(limits["monthly_loss_halt_pct"]):
        return False, f"monthly loss halt ({monthly*100:.1f}%) — account halted for the month"
    if monthly_locked >= limits["monthly_profit_lock_pct"]:
        return False, (f"monthly +{monthly_locked*100:.1f}% ≥ {limits['monthly_profit_lock_pct']*100:.0f}% "
                       f"profit-lock — new buys paused (high-conviction override only)")
    if orders_today >= limits["max_orders_per_day"]:
        return False, f"max orders/day reached ({orders_today})"
    if notional and equity and notional > equity * limits["max_order_pct_of_equity"] * 1.001:
        return False, (f"order ${notional:,.0f} > {limits['max_order_pct_of_equity']*100:.0f}% "
                       f"of equity (${equity*limits['max_order_pct_of_equity']:,.0f})")
    if equity and (held_value + (notional or 0)) > equity * limits["max_position_pct"] * 1.001:
        return False, f"position would exceed {limits['max_position_pct']*100:.0f}% of equity"
    return True, "ok"


def check_order(account: str, side: str, notional: float = None, equity: float = 0.0,
                held_value: float = 0.0, orders_today: int = 0,
                high_conviction: bool = False, month_contrib: float = 0.0,
                bench_mtd_pct: float = None) -> tuple:
    """File-backed wrapper: resolves anchors/halts, evaluates, auto-sets halts.

    month_contrib: net external cash flow into the account since the month anchor was
    set (see accounts.net_contributions). Netted out of the profit-lock only."""
    # A loss-halt blocks new buys but never exits — sells always get through
    # (kill-switch, checked in evaluate(), is the only all-stop).
    if side != "sell" and halt_reason(account):
        return False, f"account halted: {halt_reason(account)}"

    limits = _limits(account)
    day_start = anchor_equity(account, equity, "day")
    month_start = anchor_equity(account, equity, "month")
    if bench_mtd_pct is None and side != "sell":
        bench_mtd_pct = benchmark_mtd(account)   # None unless the flag is on

    ok, reason = evaluate(account, side, notional, equity, day_start, month_start,
                          held_value, orders_today, limits, month_contrib=month_contrib,
                          bench_mtd_pct=bench_mtd_pct)

    # Persist halts so they stick across restarts / the rest of the period.
    if not ok and "loss limit hit" in reason:
        set_halt(account, reason, scope="day")
    if not ok and "monthly loss halt" in reason:
        set_halt(account, reason, scope="month")

    # High-conviction override only bypasses the *profit-lock*, never a loss halt
    # or the kill-switch.
    if not ok and high_conviction and "profit-lock" in reason:
        return True, "ok (high-conviction override of profit-lock)"
    return ok, reason
