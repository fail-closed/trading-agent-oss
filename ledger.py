"""
ledger.py — durable order audit trail + broker reconciliation.

Every order attempt (blocked, submitted, rejected, filled) is appended to an
immutable JSONL ledger on the volume — the system of record for real money,
unlike ephemeral stdout logs. reconcile() compares our recorded expectations
against Alpaca's actual positions and flags drift (the KLAC-type problem).

  import ledger
  ledger.record({"event":"submitted","account":"core","symbol":"SPY",...})
  cid = ledger.client_order_id("core","SPY","buy",notional=5000)
  ledger.orders_today("core")          # count for the circuit breaker
  ledger.reconcile("core", alpaca_positions)
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
LEDGER_DIR = os.getenv("LEDGER_DIR", "signals")
LEDGER_FILE = "orders.jsonl"


def _path() -> Path:
    return Path(LEDGER_DIR) / LEDGER_FILE


def record(entry: dict) -> dict:
    """Append one event to the JSONL ledger (timestamped, ET-dated)."""
    Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
    now = datetime.now(ET)
    entry = {"ts": now.isoformat(), "date": now.strftime("%Y-%m-%d"), **entry}
    with _path().open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def _read_all() -> list:
    p = _path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def orders_today(account: str, events=("submitted",)) -> int:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    return sum(1 for e in _read_all()
               if e.get("date") == today and e.get("account") == account
               and e.get("event") in events)


def client_order_id(account: str, symbol: str, side: str,
                    notional: float = None, qty=None, salt: str = "") -> str:
    """
    Deterministic id for idempotency: a re-run of the SAME intended order (same
    account/date/symbol/side/size) produces the SAME id, which Alpaca rejects as
    a duplicate — preventing double-fills after a mid-session restart.
    """
    today = datetime.now(ET).strftime("%Y%m%d")
    size = f"n{round(float(notional),2)}" if notional else f"q{qty}"
    raw = f"{account}|{today}|{symbol.upper()}|{side}|{size}|{salt}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"ta-{account}-{today}-{symbol.upper()}-{side}-{h}"


def _baseline_path() -> Path:
    return Path(LEDGER_DIR) / "recon_baseline.json"


def _baseline() -> dict:
    p = _baseline_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _snapshot_path() -> Path:
    return Path(LEDGER_DIR) / "recon_snapshot.json"


def _snapshot() -> dict:
    try:
        p = _snapshot_path()
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _save_snapshot(snap: dict) -> None:
    try:
        Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
        from io_utils import write_text_atomic
        write_text_atomic(str(_snapshot_path()), json.dumps(snap, indent=1))
    except Exception:
        pass        # reconciliation must never break the pipeline


def _events_since(account: str, symbol: str, ts: str) -> int:
    """Order events for this symbol after `ts` — what could legitimately explain
    a position change."""
    return sum(1 for e in _read_all()
               if e.get("account") == account and e.get("symbol") == symbol
               and e.get("event") in ("submitted", "filled")
               and str(e.get("ts", "")) > str(ts))


# A quantity change smaller than this is ignored: broker dust and (on live)
# dividend reinvestment move share counts slightly without any order of ours.
QTY_TOLERANCE_PCT = 0.01
QTY_TOLERANCE_ABS = 0.001
# avg_entry_price is broker-computed and moves ONLY when you buy more — so a
# jump with no buy event is a corporate action, not trading.
BASIS_JUMP_PCT = 0.30
# Clean ratios that fingerprint a split rather than a fill.
SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 20)


def _split_fingerprint(old_qty, new_qty, old_basis, new_basis):
    """Name the corporate action when the numbers look like one: shares scale by
    a clean ratio while cost basis scales by its reciprocal. KLAC's unadjusted
    10:1 read as a −75% price collapse in our own history on 2026-08-11."""
    if not (old_qty and new_qty):
        return None
    ratio = new_qty / old_qty
    for r in SPLIT_RATIOS:
        for cand, label in ((float(r), f"{r}:1 split"), (1.0 / r, f"1:{r} reverse split")):
            if abs(ratio - cand) / cand > 0.02:
                continue
            if old_basis and new_basis:
                expect = 1.0 / cand
                if abs((new_basis / old_basis) - expect) / expect < 0.05:
                    return label
            else:
                return label + " (qty ratio only)"
    return None


def reconcile(account: str, broker_positions: dict) -> dict:
    """
    Compare the broker's actual positions to what we can explain.

    Three kinds of drift, in order of how badly we need to know:

    1. **Unknown symbol** — the broker holds something we never ordered and that
       wasn't in the baseline (unexpected fill, wrong account).
    2. **Unexplained quantity change** — a held position's share count moved with
       no order event since the last snapshot. This is the KLAC split case, a
       covered-call assignment, or an exit that reported success and didn't sell.
    3. **Unexplained cost-basis jump** — avg_entry_price moved >30% with no buy.
       The broker only recomputes basis on a purchase, so this is a corporate
       action by definition.

    Deliberately NOT sum(buys) − sum(sells): buys are placed by NOTIONAL (the
    broker decides the share count), partial fills are normal, and covered-call
    assignment legitimately removes shares without an equity order — a running
    total would drift into permanent false alarms. Comparing consecutive
    snapshots asks a question we can actually answer.

    When a change IS explained by an order, we re-snapshot and move on: with
    notional orders the exact share count is not ours to predict, and claiming
    to verify it would be false precision.

    First run per account seeds baseline + snapshot and reports ok.
    `broker_positions`: {symbol: qty} or {symbol: {"qty":…, "avg_entry":…}}.
    """
    def _qty(v):
        return float(v["qty"]) if isinstance(v, dict) else float(v)

    def _basis(v):
        return float(v.get("avg_entry") or 0) if isinstance(v, dict) else 0.0

    submitted = {e["symbol"] for e in _read_all()
                 if e.get("account") == account and e.get("event") == "submitted" and e.get("symbol")}

    now = datetime.now(ET).isoformat()
    current = {s: {"qty": _qty(v), "avg_entry": _basis(v)} for s, v in broker_positions.items()}

    bl = _baseline()
    snap = _snapshot()
    if account not in bl:                       # cold start → seed, don't alert
        bl[account] = sorted(broker_positions)
        Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
        _baseline_path().write_text(json.dumps(bl, indent=2))
        snap[account] = {"ts": now, "positions": current}
        _save_snapshot(snap)
        return {"account": account, "broker_symbols": len(broker_positions),
                "ledger_symbols": len(submitted), "drift": [], "ok": True,
                "note": "baseline seeded"}

    known = submitted | set(bl.get(account, []))
    drift = [{"symbol": s, "issue": "broker holds position with no order in ledger",
              "broker_qty": v["qty"]}
             for s, v in current.items() if s not in known]

    prev = snap.get(account, {})
    prev_pos, prev_ts = prev.get("positions", {}), prev.get("ts", "")
    for sym, was in prev_pos.items():
        old_qty, old_basis = float(was.get("qty", 0)), float(was.get("avg_entry") or 0)
        now_pos = current.get(sym)
        explained = _events_since(account, sym, prev_ts)

        if now_pos is None:
            if not explained:
                drift.append({"symbol": sym, "broker_qty": 0,
                              "issue": f"position of {old_qty:g} shares disappeared "
                                       "with no sell order in the ledger"})
            continue

        new_qty, new_basis = now_pos["qty"], now_pos["avg_entry"]
        delta = abs(new_qty - old_qty)
        if not explained and delta > QTY_TOLERANCE_ABS and delta > old_qty * QTY_TOLERANCE_PCT:
            action = _split_fingerprint(old_qty, new_qty, old_basis, new_basis)
            drift.append({"symbol": sym, "broker_qty": new_qty,
                          "issue": (f"quantity changed {old_qty:g} → {new_qty:g} with no order"
                                    + (f" — looks like a {action}" if action else ""))})
        elif not explained and old_basis and new_basis and \
                abs(new_basis - old_basis) / old_basis > BASIS_JUMP_PCT:
            drift.append({"symbol": sym, "broker_qty": new_qty,
                          "issue": f"cost basis moved ${old_basis:.2f} → ${new_basis:.2f} "
                                   "with no buy order — corporate action?"})

    snap[account] = {"ts": now, "positions": current}
    _save_snapshot(snap)
    return {"account": account, "broker_symbols": len(broker_positions),
            "ledger_symbols": len(submitted), "drift": drift, "ok": len(drift) == 0}
