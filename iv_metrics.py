"""
iv_metrics.py — per-name implied volatility level, rank, and skew.

WHY THIS EXISTS
---------------
We already pay for options data (Alpaca chains, ThetaData) and use it only to
pick contracts and run backtests. The chain also prices the single most
reliable premium we measured: implied vol exceeded subsequent realised vol by
+3.86 points on 84.9% of days over 2021-05..2026-06 (strategy_lens.py §6).

That measurement was index-level. To act on it per name you need to know whether
*this* underlying's vol is expensive relative to its own history — which is what
IV rank is. The absolute IV number is nearly useless across names: 30% is cheap
for a semi and dear for a utility.

Two outputs:

  iv_rank   where today's 30-day ATM IV sits in its own trailing range, 0-100.
            High = options expensive here = good to WRITE (covered calls),
            bad to buy. Low = the reverse.
  skew      25-delta put IV minus 25-delta call IV, in vol points. High skew
            means downside protection is bid — crash fear is being priced.
            It moves ahead of spot more often than the level does.

BOOTSTRAP HONESTY
-----------------
IV rank needs history we do not have on day one. Rather than fake it from a
single snapshot, this module persists one ATM IV reading per symbol per day and
returns `iv_rank: None` until MIN_HISTORY_DAYS have accumulated. A null is
correct and readable; a rank computed from four observations is neither.

  import iv_metrics
  iv_metrics.enrich(signals, trading_client, data_client)
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HISTORY_FILE = "iv_history.json"

TARGET_DTE_LOW = 20        # the 30-day window the VRP literature uses
TARGET_DTE_HIGH = 45
MIN_HISTORY_DAYS = 40      # below this, iv_rank stays None
MAX_HISTORY_DAYS = 400     # roughly a year of trailing range
SNAPSHOT_BATCH = 100


def enabled() -> bool:
    return os.getenv("IV_METRICS", "").strip().lower() in ("1", "true", "yes", "on")


def _cache_dir():
    # Read the env on every call rather than caching at import: a module-scope
    # capture freezes the path to whichever import happened first, which silently
    # breaks anything that sets LEDGER_DIR afterwards (and leaks state in tests).
    from pathlib import Path
    return Path(os.getenv("LEDGER_DIR", "signals"))


def _path():
    return _cache_dir() / HISTORY_FILE


def _load() -> dict:
    import json
    p = _path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save(hist: dict):
    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_path()), hist)
    except Exception:
        pass


def _chain_snapshot(tc, dc, symbol: str, spot: float) -> list:
    """Contracts in the target DTE window with IV and delta attached."""
    from alpaca.data.requests import OptionSnapshotRequest
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetOptionContractsRequest

    today = datetime.now(ET).date()
    try:
        req = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=today + timedelta(days=TARGET_DTE_LOW),
            expiration_date_lte=today + timedelta(days=TARGET_DTE_HIGH),
            strike_price_gte=str(round(spot * 0.80, 2)),
            strike_price_lte=str(round(spot * 1.20, 2)),
            limit=250,
        )
        contracts = tc.get_option_contracts(req).option_contracts or []
    except Exception as e:
        print(f"  [iv] {symbol} chain failed: {str(e)[:70]}", file=sys.stderr)
        return []

    meta = {c.symbol: {
        "strike": float(c.strike_price),
        "type": c.type.value if hasattr(c.type, "value") else str(c.type),
    } for c in contracts}

    rows = []
    syms = list(meta)
    for i in range(0, len(syms), SNAPSHOT_BATCH):
        try:
            snaps = dc.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=syms[i:i + SNAPSHOT_BATCH]))
        except Exception as e:
            print(f"  [iv] {symbol} snapshot failed: {str(e)[:70]}", file=sys.stderr)
            continue
        for occ, snap in (snaps or {}).items():
            iv = getattr(snap, "implied_volatility", None)
            if snap is None or iv is None:
                continue
            delta = None
            if getattr(snap, "greeks", None) and snap.greeks.delta is not None:
                delta = float(snap.greeks.delta)
            rows.append({**meta[occ], "iv": float(iv), "delta": delta})
    return rows


def _atm_iv_and_skew(rows: list, spot: float) -> dict:
    """ATM IV (average of the nearest-strike call and put) and 25-delta skew."""
    out = {"atm_iv": None, "skew_25d": None}
    if not rows or not spot:
        return out

    calls = [r for r in rows if r["type"] == "call"]
    puts = [r for r in rows if r["type"] == "put"]

    def _nearest_strike(rs):
        return min(rs, key=lambda r: abs(r["strike"] - spot)) if rs else None

    c_atm, p_atm = _nearest_strike(calls), _nearest_strike(puts)
    atm = [r["iv"] for r in (c_atm, p_atm) if r]
    if atm:
        out["atm_iv"] = round(sum(atm) / len(atm) * 100, 2)

    # 25-delta wings. Put deltas are negative, so compare on magnitude.
    def _near_delta(rs, target):
        havedelta = [r for r in rs if r.get("delta") is not None]
        if not havedelta:
            return None
        return min(havedelta, key=lambda r: abs(abs(r["delta"]) - target))

    p25, c25 = _near_delta(puts, 0.25), _near_delta(calls, 0.25)
    if p25 and c25:
        out["skew_25d"] = round((p25["iv"] - c25["iv"]) * 100, 2)
    return out


def _rank(history: list, current: float) -> "float | None":
    """Percentile of `current` within its own trailing range, 0-100."""
    vals = [v for v in history if v is not None]
    if len(vals) < MIN_HISTORY_DAYS or current is None:
        return None
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return 50.0
    return round((current - lo) / (hi - lo) * 100, 1)


def enrich(signals: list, tc=None, dc=None) -> int:
    """Attach iv_* fields to each signal and persist today's reading.

    Returns the number of symbols priced. No-op unless IV_METRICS is on.
    """
    if not enabled():
        return 0
    try:
        if tc is None or dc is None:
            import accounts
            from alpaca.data.historical.option import OptionHistoricalDataClient
            acct = "options" if accounts.available("options") else "core"
            key, sec = accounts.get_keys(acct)
            tc = tc or accounts.trading_client(acct)
            dc = dc or OptionHistoricalDataClient(key, sec)
    except Exception as e:
        print(f"  [iv] client setup failed: {str(e)[:80]}", file=sys.stderr)
        return 0

    today = datetime.now(ET).strftime("%Y-%m-%d")
    hist = _load()
    priced = 0

    for s in signals:
        sym = s.get("symbol")
        spot = s.get("price")
        if not sym or not spot:
            continue

        entry = hist.setdefault(sym, {"dates": [], "atm_iv": []})
        # Already recorded today (a second session) — reuse, don't re-poll.
        if entry["dates"] and entry["dates"][-1] == today:
            atm = entry["atm_iv"][-1]
            measured = {"atm_iv": atm, "skew_25d": s.get("iv_skew_25d")}
        else:
            measured = _atm_iv_and_skew(_chain_snapshot(tc, dc, sym, spot), spot)
            if measured["atm_iv"] is not None:
                entry["dates"].append(today)
                entry["atm_iv"].append(measured["atm_iv"])
                # Keep the trailing window bounded.
                entry["dates"] = entry["dates"][-MAX_HISTORY_DAYS:]
                entry["atm_iv"] = entry["atm_iv"][-MAX_HISTORY_DAYS:]
                priced += 1

        s["iv_atm"] = measured["atm_iv"]
        s["iv_skew_25d"] = measured["skew_25d"]
        s["iv_rank"] = _rank(entry["atm_iv"], measured["atm_iv"])
        s["iv_history_days"] = len(entry["atm_iv"])

    _save(hist)
    return priced


if __name__ == "__main__":
    import json
    os.environ.setdefault("IV_METRICS", "true")
    from dotenv import load_dotenv
    load_dotenv()
    import accounts
    from alpaca.data.historical.option import OptionHistoricalDataClient

    acct = "options" if accounts.available("options") else "core"
    key, sec = accounts.get_keys(acct)
    sigs = []
    for arg in (sys.argv[1:] or ["AAPL:230", "GS:1030"]):
        sym, _, px = arg.partition(":")
        sigs.append({"symbol": sym, "price": float(px) if px else None})
    print(f"priced {enrich(sigs, accounts.trading_client(acct), OptionHistoricalDataClient(key, sec))}")
    for s in sigs:
        print(json.dumps({k: v for k, v in s.items() if k.startswith("iv") or k == "symbol"}))
