"""
fundamentals.py — lightweight fundamentals lens via the yfinance dep you already
have (flag: FUNDAMENTALS_LENS). No OpenBB / numpy-2 baggage.

Attaches a compact `fundamentals` dict to each signal so the analyst decomposition
gets a valuation/quality lens. Cached once per symbol per day on the volume
(yfinance .info is slow + rate-limited), and fully fail-open — any error yields an
empty dict and never breaks research.

  import fundamentals
  if fundamentals.enabled():
      fundamentals.enrich(signals)     # mutates each signal: s["fundamentals"] = {...}
"""
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_DIR = os.getenv("LEDGER_DIR", "signals")   # volume-backed
CACHE_FILE = "fundamentals_cache.json"

# yfinance .info key → our compact field name. Only present values are kept.
_FIELDS = {
    "trailingPE": "pe",
    "forwardPE": "forward_pe",
    "pegRatio": "peg",
    "profitMargins": "profit_margin",
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
    "revenueGrowth": "revenue_growth",
    "marketCap": "market_cap",
    # Short interest — positioning, not price. Every other input we have is a
    # transform of OHLCV, and our own dip model tops out at ~0.50 AUC on that
    # feature set; shorts are one of the few free inputs that carry genuinely
    # orthogonal information (they are also the most informed trader cohort).
    # sharesShortPriorMonth is what makes this useful: the CHANGE in short
    # interest says what positioning is doing now, while the level mostly
    # reflects the structural borrow of that name.
    "sharesShort": "shares_short",
    "sharesShortPriorMonth": "shares_short_prior",
    "shortRatio": "short_days_to_cover",
    "shortPercentOfFloat": "short_pct_float",
}


def enabled() -> bool:
    return os.getenv("FUNDAMENTALS_LENS", "").strip().lower() in ("1", "true", "yes", "on")


def _cache_path():
    from pathlib import Path
    return Path(CACHE_DIR) / CACHE_FILE


def _load_cache() -> dict:
    import json
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _fetch(symbol: str) -> dict:
    """One symbol's compact fundamentals via yfinance. {} on any failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        out = {}
        for src, dst in _FIELDS.items():
            v = info.get(src)
            if v is not None:
                out[dst] = round(v, 4) if isinstance(v, float) else v

        # Month-over-month change in short interest, as a percentage. Rising =
        # bearish positioning building; falling = shorts covering, which can
        # itself be fuel for a move up.
        cur, prior = out.get("shares_short"), out.get("shares_short_prior")
        if cur is not None and prior:
            try:
                out["short_interest_change_pct"] = round((cur / prior - 1) * 100, 2)
            except (TypeError, ZeroDivisionError):
                pass
        return out
    except Exception as e:
        print(f"  [fundamentals] {symbol} fetch failed: {str(e)[:80]}", file=sys.stderr)
        return {}


def enrich(signals: list) -> int:
    """Attach s['fundamentals'] to each signal, using a daily cache. Returns # fetched live."""
    import json
    today = datetime.now(ET).strftime("%Y-%m-%d")
    cache = _load_cache()
    fetched = 0
    for s in signals:
        sym = s.get("symbol")
        if not sym:
            continue
        entry = cache.get(sym)
        if entry and entry.get("date") == today:
            s["fundamentals"] = entry.get("data", {})
            continue
        data = _fetch(sym)
        cache[sym] = {"date": today, "data": data}
        s["fundamentals"] = data
        if data:
            fetched += 1
    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_cache_path()), cache)
    except Exception:
        pass
    return fetched
