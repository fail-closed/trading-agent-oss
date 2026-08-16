"""
earnings_calendar.py — real earnings dates, so the blackout rule has data behind it.

WHY THIS EXISTS
---------------
CLAUDE.md lists this under NON-NEGOTIABLE risk rules:

    "Never buy a stock that has earnings within 48 hours."

Until now nothing in the repo knew when earnings were. The only source was
`macro.get("earnings_watch")` in decide.py — a list the LLM writes from whatever
news headlines happened to surface that morning. A hard risk rule was being
enforced by a model guessing. This module replaces the guess with a calendar,
and `blackout_symbols()` lets the trade path enforce it in code rather than in a
prompt the model can reason its way around.

It also carries the surprise history, which is the input to post-earnings
announcement drift (PEAD) — buying confirmed positive surprises for ~60 days
after the print. PEAD is a *trend* bet, and trend is the family that actually
paid in our own universe (strategy_lens.py). Cheap to add once the calendar is
here; `pead_window` is emitted as advisory and is not yet a buy trigger.

Cached once per symbol per day on the volume (yfinance is slow and rate-limited)
and fully fail-open — any error yields empty data and never breaks research.

  import earnings_calendar as ec
  ec.enrich(signals)                 # adds earnings_* fields to each signal
  ec.blackout_symbols(signals)       # -> {"AAPL", ...} for the hard gate
"""
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_FILE = "earnings_cache.json"

# How many days before a print to refuse new buys. The rule says 48h; kept as a
# variable so a profile can widen it without a deploy.
DEFAULT_BLACKOUT_DAYS = 2
# PEAD looks at the drift AFTER a confirmed beat, for roughly a quarter.
PEAD_MAX_DAYS = 60
PEAD_MIN_SURPRISE = 5.0    # percent


def enabled() -> bool:
    """Default ON. This backs a NON-NEGOTIABLE risk rule, so the safe failure
    mode is 'have the data', not 'have the flag'. Set EARNINGS_CALENDAR=false
    to disable (e.g. if yfinance starts rate-limiting hard)."""
    return os.getenv("EARNINGS_CALENDAR", "true").strip().lower() in ("1", "true", "yes", "on")


def blackout_days() -> int:
    try:
        return int(os.getenv("EARNINGS_BLACKOUT_DAYS", DEFAULT_BLACKOUT_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_BLACKOUT_DAYS


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


def _fetch(symbol: str) -> "dict | None":
    """Next earnings date + most recent surprise for one symbol.

    Returns {} when the lookup SUCCEEDED but the name has no earnings (an ETF, a
    closed-end fund), and None when the lookup FAILED. The caller must not cache
    a None: caching a failure blinds a NON-NEGOTIABLE risk rule for a whole day.
    """
    try:
        import yfinance as yf
        import pandas as pd

        t = yf.Ticker(symbol)
        out = {}

        # get_earnings_dates carries both the schedule and the surprise history.
        try:
            df = t.get_earnings_dates(limit=12)
        except Exception:
            df = None

        if df is not None and len(df):
            now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
            future = df[df.index > now]
            past = df[df.index <= now]
            if len(future):
                out["next"] = future.index.min().date().isoformat()
            if len(past):
                last = past.index.max()
                out["last"] = last.date().isoformat()
                # .loc on a duplicated index returns a DataFrame, not a row, and
                # everything downstream then works on a Series — which is how
                # PAGS raised "truth value of a Series is ambiguous". Take the
                # first matching row explicitly instead of trusting the shape.
                rows = past.loc[[last]] if last in past.index else past.tail(1)
                surprise = rows["Surprise(%)"].iloc[0] if "Surprise(%)" in rows else None
                if surprise is not None and not pd.isna(surprise):
                    out["last_surprise_pct"] = round(float(surprise), 2)

        # calendar is a cheap cross-check and a fallback when the history call fails.
        if "next" not in out:
            try:
                cal = t.calendar or {}
                dates = cal.get("Earnings Date")
                # May be a list, a Series, or None depending on yfinance version —
                # never use plain truthiness on it.
                dates = [] if dates is None else list(dates)
                if len(dates):
                    out["next"] = min(dates).isoformat()
            except Exception:
                pass
        return out
    except Exception as e:
        print(f"  [earnings] {symbol} fetch failed: {str(e)[:80]}", file=sys.stderr)
        return None


def _days_between(iso_date: str, today) -> "int | None":
    try:
        return (datetime.fromisoformat(iso_date).date() - today).days
    except (TypeError, ValueError):
        return None


def enrich(signals: list, skip: set = None) -> int:
    """Attach earnings fields to each signal. Returns the number fetched live.

    `skip` holds symbols with no earnings by construction (ETFs). Passing them
    costs a wasted round trip and prints a scary-looking 404 for something that
    is entirely normal.

    Fields added:
      earnings_date        next scheduled print (ISO) or None
      days_to_earnings     calendar days until it (negative = already passed)
      earnings_blackout    True when a buy must be refused
      days_since_earnings  for PEAD
      last_surprise_pct    most recent EPS surprise
      pead_window          advisory: confirmed beat, still inside the drift window
    """
    if not enabled():
        for s in signals:
            s["earnings_blackout"] = False
        return 0

    import json
    today = datetime.now(ET).date()
    today_str = today.isoformat()
    cache = _load_cache()
    window = blackout_days()
    fetched = 0

    skip = {x.upper() for x in (skip or set())}
    for s in signals:
        sym = s.get("symbol")
        if not sym:
            continue
        if sym.upper() in skip:
            # An ETF has no earnings, so it can never be in a blackout.
            s["earnings_date"] = None
            s["days_to_earnings"] = None
            s["days_since_earnings"] = None
            s["last_surprise_pct"] = None
            s["earnings_blackout"] = False
            s["pead_window"] = False
            continue
        entry = cache.get(sym)
        unknown = False
        if entry and entry.get("date") == today_str:
            data = entry.get("data", {})
        else:
            result = _fetch(sym)
            if result is None:
                # DO NOT cache a failure. Caching it would leave this name with
                # no earnings data — and therefore no blackout — for the rest of
                # the day. Caught live: PAGS hit a transient parse error, was
                # cached empty, and reported earnings THAT SAME DAY with
                # earnings_blackout=False. There are three sessions a day, so a
                # retry self-heals; a cached failure does not.
                data, unknown = {}, True
            else:
                data = result
                cache[sym] = {"date": today_str, "data": data}
                fetched += 1

        nxt = data.get("next")
        last = data.get("last")
        d_to = _days_between(nxt, today) if nxt else None
        d_since = -_days_between(last, today) if last else None
        surprise = data.get("last_surprise_pct")

        s["earnings_date"] = nxt
        s["days_to_earnings"] = d_to
        s["days_since_earnings"] = d_since
        s["last_surprise_pct"] = surprise
        # Surfaced rather than silently absent. We do NOT block on unknown: a
        # yfinance outage would otherwise freeze all buying, and the rest of the
        # risk stack still applies. But the model should see that the check could
        # not be performed rather than read a null as "no earnings scheduled".
        s["earnings_unknown"] = unknown
        # Blackout only on a KNOWN upcoming print. An unknown date is not treated
        # as safe-by-default anywhere the caller cares — see blackout_symbols().
        s["earnings_blackout"] = bool(d_to is not None and 0 <= d_to <= window)
        s["pead_window"] = bool(
            d_since is not None and 1 <= d_since <= PEAD_MAX_DAYS
            and surprise is not None and surprise >= PEAD_MIN_SURPRISE
        )

    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_cache_path()), cache)
    except Exception:
        pass
    return fetched


def blackout_symbols(signals: list) -> set:
    """Symbols a BUY must be refused on today, for the code-level gate."""
    return {s["symbol"] for s in signals
            if s.get("symbol") and s.get("earnings_blackout")}


if __name__ == "__main__":
    import json
    syms = sys.argv[1:] or ["AAPL", "GS", "CAT"]
    sigs = [{"symbol": s} for s in syms]
    n = enrich(sigs)
    print(f"fetched {n} live; blackout={sorted(blackout_symbols(sigs))}")
    for s in sigs:
        print(json.dumps({k: v for k, v in s.items()}, default=str))
