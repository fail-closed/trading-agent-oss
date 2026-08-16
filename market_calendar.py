"""market_calendar.py — is today a US trading day?

The scheduler crons run Mon–Fri, but that still fires on weekday market holidays
(e.g. Juneteenth, July 4, Thanksgiving) when the market is closed — research then
pulls stale data and orders can't fill. This module answers "is <date> a real
trading day?" from Alpaca's official trading calendar (which encodes holidays and
half-days), cached to a JSON on the volume so we don't hit the API every job.

Fail-open: if the calendar can't be fetched, weekdays are treated as trading days
so a transient API blip never silently halts the agent. Weekends are always closed.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE = Path(os.getenv("LEDGER_DIR", "signals")) / "market_calendar.json"


def _fetch(year: int) -> dict:
    """Trading days for `year` from Alpaca → {'YYYY-MM-DD': {'open','close'}}."""
    import accounts
    from alpaca.trading.requests import GetCalendarRequest
    tc = accounts.trading_client("core")
    cal = tc.get_calendar(GetCalendarRequest(start=f"{year}-01-01", end=f"{year}-12-31"))
    days = {}
    for c in cal:
        d = c.date.strftime("%Y-%m-%d") if hasattr(c.date, "strftime") else str(c.date)[:10]
        days[d] = {"open": str(getattr(c, "open", "")), "close": str(getattr(c, "close", ""))}
    return days


def _trading_days() -> dict:
    """Cached map of this year's trading days; refreshes when missing or year rolls over."""
    year = datetime.now(ET).year
    try:
        if CACHE.exists():
            data = json.loads(CACHE.read_text())
            if data.get("year") == year and data.get("days"):
                return data["days"]
    except Exception:
        pass
    days = _fetch(year)          # may raise — caller handles fail-open
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"year": year, "days": days}))
    except Exception:
        pass
    return days


def is_trading_day(date_str: str = None) -> bool:
    """True if `date_str` (ET, default today) is a US equity trading day."""
    date_str = date_str or datetime.now(ET).strftime("%Y-%m-%d")
    try:
        return date_str in _trading_days()
    except Exception:
        # fail-open: don't halt the agent on a calendar fetch glitch — weekdays trade.
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5


if __name__ == "__main__":
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"{today}: {'TRADING DAY' if is_trading_day() else 'MARKET CLOSED'}")
