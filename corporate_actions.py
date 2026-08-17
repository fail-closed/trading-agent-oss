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
        out = {"splits": {}, "next_ex_div": None, "ok": False}

        # `ok` exists because {} previously meant BOTH "no splits" and "the
        # lookup failed". Those demand opposite responses: the first is a clean
        # signal, the second means the price series may be unadjusted and every
        # indicator derived from it is suspect. At 30 names an occasional failure
        # was noise; at 1,000 it is routine, and silently reading it as "no
        # splits" is a fail-open on the exact defect this module exists to catch.
        try:
            sp = t.splits
            if sp is not None and len(sp):
                idx = sp.index
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                for ts, ratio in zip(idx, sp.values):
                    if ts >= cutoff and float(ratio) > 0:
                        out["splits"][ts.date().isoformat()] = float(ratio)
            out["ok"] = True
        except Exception as e:
            out["ok"] = False
            out["error"] = str(e)[:80]

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
    todo = []
    for sym in symbols:
        entry = cache.get(sym)
        if entry and entry.get("date") == today:
            out[sym] = entry.get("data", {})
        else:
            todo.append(sym)

    # Fetched in parallel. Sequentially this is one network round-trip per symbol:
    # fine for 30 names, ~1,000 round-trips for a liquidity-ranked universe, which
    # is slow enough that partial failure becomes the normal case rather than the
    # exception. Bounded at 8 workers to stay polite to the data source.
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=int(os.getenv("CORP_ACTION_WORKERS", "8"))) as ex:
            futures = {ex.submit(_fetch, s): s for s in todo}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    data = fut.result()
                except Exception as e:                       # pragma: no cover
                    data = {"splits": {}, "next_ex_div": None, "ok": False,
                            "error": str(e)[:80]}
                out[sym] = data
                # Only a SUCCESSFUL lookup is cached for the day. Caching a
                # failure would turn one rate-limit into a whole session — and a
                # silent one, since the cached {} reads as "no splits".
                if data.get("ok"):
                    cache[sym] = {"date": today, "data": data}
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


def coverage(actions: dict) -> dict:
    """How much of the universe we actually have split data for.

    research.py prints this. A universe where 300 of 1,000 lookups failed is one
    where 300 names may be running on unadjusted prices, and that has to be
    visible rather than inferred from a suspiciously low split count.
    """
    total = len(actions)
    ok = sum(1 for v in actions.values() if v.get("ok"))
    with_splits = sum(1 for v in actions.values() if v.get("splits"))
    return {"total": total, "ok": ok, "failed": total - ok,
            "with_splits": with_splits,
            "pct_ok": round(ok / total * 100, 1) if total else 0.0}


def data_ok(actions: dict, symbol: str) -> bool:
    """False when we could not establish this symbol's split history — the caller
    should treat its moving averages as unverified, not as clean."""
    return bool((actions.get(symbol) or {}).get("ok"))
