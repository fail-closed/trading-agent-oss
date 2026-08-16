"""ladder.py — makes ladder tranche 2 survive a slow tranche-1 fill.

The ladder splits a moderate-confidence dip buy into 60% now + 40% as a GTC
limit 4% below the fill. Tranche 2 can only be priced once tranche 1 has a fill
price — and `trade.py` polls for that for 30 seconds, then returns without one.

Before this module, `decide.py` placed tranche 2 only `if r.get("fill_price")`.
A tranche 1 that filled at 31 seconds — or any limit order that filled later in
the session, which is the normal case for a limit order placed at the touch —
silently lost its second tranche FOREVER. The account was left holding 60% of
the intended position with nothing recording that the other 40% was owed.

Now the intent is persisted and retried: every decide.py session (10:00 and each
intraday cycle) resolves what is pending before making new decisions.

  pending order still open      → leave it, try again next session
  filled (or partially filled)  → place tranche 2 at 4% below the ACTUAL average
  canceled/expired/rejected, no fill → drop it, nothing was bought
  older than MAX_AGE_DAYS       → drop it, the -4% level is stale

Safe to run twice: `place_ladder_tranche2` builds a deterministic
client_order_id (salt "ladder2"), so a duplicate is rejected by the broker.
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE_FILE = "pending_ladders.json"
MAX_AGE_DAYS = 3          # after this the -4% anchor is stale, not an entry

# Terminal broker states — the order will never fill any further.
TERMINAL = {"filled", "canceled", "cancelled", "expired", "rejected",
            "done_for_day", "stopped", "suspended"}


def _path() -> Path:
    return Path(os.getenv("LEDGER_DIR", "signals")) / STATE_FILE


def _load() -> list:
    try:
        data = json.loads(_path().read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(rows: list) -> None:
    """Best-effort atomic write. Losing this file costs a second tranche, never
    money — it must never raise into the trading path."""
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        from io_utils import write_text_atomic
        write_text_atomic(str(p), json.dumps(rows, indent=1))
    except Exception as e:
        print(f"ladder: could not persist pending tranches (non-fatal): {e}")


def record_pending(symbol: str, order_id: str, tranche2_notional: float,
                   account: str = "core") -> None:
    """Remember that `symbol` still owes a tranche 2 once its tranche 1 fills."""
    rows = [r for r in _load() if r.get("order_id") != order_id]
    rows.append({"symbol": symbol, "order_id": str(order_id),
                 "tranche2_notional": round(float(tranche2_notional), 2),
                 "account": account, "created_at": datetime.now(ET).isoformat()})
    _save(rows)
    print(f"    Tranche 2 deferred — ${tranche2_notional:.0f} pending fill of {order_id}")


def pending() -> list:
    return _load()


def _age_days(row) -> float:
    try:
        created = datetime.fromisoformat(row["created_at"])
        return (datetime.now(ET) - created).total_seconds() / 86400.0
    except Exception:
        return 0.0


def resolve(get_order, place_tranche2) -> list:
    """Retry every deferred tranche 2. Returns a list of outcome dicts.

    `get_order(order_id)` → the broker order (raise/None if unknown).
    `place_tranche2(symbol, notional, entry_price)` → dict with a "status" key.
    Injected rather than imported so this is testable without a broker.
    """
    rows, keep, outcomes = _load(), [], []
    for row in rows:
        sym, oid = row.get("symbol"), row.get("order_id")
        try:
            order = get_order(oid)
            status = getattr(order.status, "value", str(order.status)).lower()
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
            avg = float(getattr(order, "filled_avg_price", 0) or 0)
        except Exception as e:
            # Unknown order — don't guess. Age it out rather than retry forever.
            if _age_days(row) > MAX_AGE_DAYS:
                outcomes.append({"symbol": sym, "result": "dropped",
                                 "detail": f"order not found: {str(e)[:60]}"})
            else:
                keep.append(row)
            continue

        if filled_qty > 0 and avg > 0 and status in TERMINAL:
            res = place_tranche2(sym, row["tranche2_notional"], avg)
            outcomes.append({"symbol": sym, "result": res.get("status", "unknown"),
                             "detail": res.get("error", ""), "entry": avg})
            continue                       # resolved either way — do not retry
        if status in TERMINAL:             # terminal with no fill at all
            outcomes.append({"symbol": sym, "result": "dropped",
                             "detail": f"tranche 1 {status} unfilled"})
            continue
        if _age_days(row) > MAX_AGE_DAYS:
            outcomes.append({"symbol": sym, "result": "dropped",
                             "detail": f"still {status} after {MAX_AGE_DAYS}d — stale anchor"})
            continue
        keep.append(row)                   # still working — try again next session

    if keep != rows:
        _save(keep)
    return outcomes
