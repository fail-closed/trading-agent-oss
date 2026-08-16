"""
corporate_actions.py — split and dividend history, so price series mean what
they appear to mean.

WHY THIS EXISTS
---------------
Our bar source does not always back-adjust splits. On 2026-08-11 KLAC's history
read $912 twelve months ago against $193 now — a −75% year — when the truth was
$91 → $200, i.e. **+144%**. A 10:1 split on 2026-06-12 sat unadjusted in the
middle of the lookback. That is a sign flip on a decile-ranked input, and it put
the universe's strongest momentum name at the very bottom of the ranking.

`research.compute_momentum` currently defends itself by *abstaining* when it sees
a split-sized gap. That is safe but lossy: the name drops out of the
cross-section entirely, so we lose a real signal to protect against a data
artifact. With an actual corporate-actions feed we can **adjust** instead, keep
the name, and reserve the abstain path for gaps we genuinely cannot explain.

Also carries the dividend calendar, which matters for the covered-call overlay:
American calls are most often assigned early the day before an ex-dividend date,
when the dividend exceeds the call's remaining time value. Writing a call across
an ex-div date invites exactly that.

Cached once per symbol per day on the volume, fully fail-open.

  import corporate_actions as ca
  splits = ca.splits_for("KLAC")          # {"2026-06-12": 10.0}
  adj    = ca.adjust_close(close, splits) # back-adjusted series
"""
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_FILE = "corporate_actions_cache.json"

# Only actions inside a momentum lookback can distort it; two years is ample.
LOOKBACK_DAYS = 730


def enabled() -> bool:
    """Default ON — this is a data-correctness fix, not a strategy opinion."""
    return os.getenv("CORPORATE_ACTIONS", "true").strip().lower() in ("1", "true", "yes", "on")


def _cache_dir():
    # Read the env on every call rather than caching at import: a module-scope
    # capture freezes the path to whichever import happened first, which silently
    # breaks anything that sets LEDGER_DIR afterwards (and leaks state in tests).
    from pathlib import Path
    return Path(os.getenv("LEDGER_DIR", "signals"))


def _cache_path():
    return _cache_dir() / CACHE_FILE


def _load_cache() -> dict:
    import json
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_cache_path()), cache)
    except Exception:
        pass


def _fetch(symbol: str) -> dict:
    """{splits: {iso: ratio}, next_ex_div: iso|None} for one symbol."""
    try:
        import pandas as pd
        import yfinance as yf

        t = yf.Ticker(symbol)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
        out = {"splits": {}, "next_ex_div": None}

        try:
            sp = t.splits
            if sp is not None and len(sp):
                idx = sp.index
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                for ts, ratio in zip(idx, sp.values):
                    if ts >= cutoff and float(ratio) > 0:
                        out["splits"][ts.date().isoformat()] = float(ratio)
        except Exception:
            pass

        try:
            cal = t.calendar or {}
            exd = cal.get("Ex-Dividend Date")
            if exd is not None:
                out["next_ex_div"] = exd.isoformat() if hasattr(exd, "isoformat") else str(exd)
        except Exception:
            pass

        return out
    except Exception as e:
        print(f"  [corp-actions] {symbol} fetch failed: {str(e)[:80]}", file=sys.stderr)
        return {"splits": {}, "next_ex_div": None}


def fetch_all(symbols: list) -> dict:
    """{symbol: {splits, next_ex_div}} using a daily cache."""
    if not enabled():
        return {}
    today = datetime.now(ET).strftime("%Y-%m-%d")
    cache = _load_cache()
    out = {}
    dirty = False
    for sym in symbols:
        entry = cache.get(sym)
        if entry and entry.get("date") == today:
            out[sym] = entry.get("data", {})
            continue
        data = _fetch(sym)
        cache[sym] = {"date": today, "data": data}
        out[sym] = data
        dirty = True
    if dirty:
        _save_cache(cache)
    return out


def splits_for(symbol: str) -> dict:
    return (fetch_all([symbol]).get(symbol) or {}).get("splits", {})


def adjust_close(close, splits: dict):
    """Back-adjust a close series for splits so the whole window is comparable.

    A 10:1 split on date D means every price BEFORE D is quoted in pre-split
    units and is 10× too large relative to today. Dividing those by the product
    of all ratios that occur after them puts the series on one scale.

    Returns the input unchanged when there is nothing to do, so callers can pass
    an empty dict without branching.
    """
    if not splits or close is None or len(close) == 0:
        return close
    try:
        import pandas as pd

        idx = close.index
        # Normalise both sides to naive dates — the bar index may be tz-aware,
        # tz-naive, or plain dates depending on the provider.
        try:
            idx_dates = pd.DatetimeIndex(idx).tz_localize(None)
        except (TypeError, AttributeError):
            idx_dates = pd.DatetimeIndex(idx)

        factor = pd.Series(1.0, index=close.index)
        for iso, ratio in splits.items():
            try:
                d = pd.Timestamp(iso)
                r = float(ratio)
            except (TypeError, ValueError):
                continue
            if r <= 0:
                continue
            # Bars strictly before the split are quoted pre-split.
            factor[idx_dates < d] *= r
        return close / factor
    except Exception as e:
        print(f"  [corp-actions] adjust failed, using raw series: {str(e)[:80]}",
              file=sys.stderr)
        return close


def ex_div_within(symbol_data: dict, days: int, today=None) -> bool:
    """True if an ex-dividend date falls inside the next `days`.

    Used by the covered-call overlay: American calls are most often assigned
    early the day before ex-div, when the dividend exceeds remaining time value.
    """
    exd = (symbol_data or {}).get("next_ex_div")
    if not exd:
        return False
    try:
        today = today or datetime.now(ET).date()
        delta = (datetime.fromisoformat(str(exd)).date() - today).days
        return 0 <= delta <= days
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    import json
    syms = sys.argv[1:] or ["KLAC", "WDC", "AAPL"]
    print(json.dumps(fetch_all(syms), indent=2, default=str))
