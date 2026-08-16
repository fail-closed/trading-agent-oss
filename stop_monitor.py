"""
stop_monitor.py — frequent mechanical stop-loss AND profit-lock for the core account.

Alpaca does NOT allow broker-side GTC stop orders on fractional shares, so small
accounts (e.g. the live pilot, where the strategy controls ~$4k and almost every
position is < 1 share) can't have a broker-held stop. This fills that gap every
few minutes during market hours:

  1. HARD EXIT — any position at or below the stop floor is sold (all sizes).
  2. TRAILING PROFIT LOCK — fractional positions only, because whole-share
     positions already have a real GTC stop that decide.py ratchets. Without
     this, a fractional position could run +18% and give it all back: it had
     the −8% floor and nothing else. `decide.py`'s trail_stops() skips
     `floor(qty) < 1` by broker necessity, which on the live account meant
     essentially the WHOLE BOOK never locked a profit (audit item 4.2).

Both use the shared rule in `stops.py` — the same ratchet decide.py applies to
broker-held stops, so the two paths cannot drift. The high-water stop level per
fractional position is persisted (a broker order can't hold it for us) in
`<LEDGER_DIR>/frac_stops.json`, and pruned when the position closes so a
re-entry starts fresh.

Covered calls are safe by arithmetic: a contract needs 100 whole shares, so a
fractional position can never be covering one.

Cheap (no LLM), so it can run often. Carved-out symbols are never touched.
Live-aware via accounts.trading_client('core').

  python3 stop_monitor.py
"""
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import accounts
import stops

ET = ZoneInfo("America/New_York")
FRAC_STATE_FILE = "frac_stops.json"


def _state_path() -> Path:
    return Path(os.getenv("LEDGER_DIR", "signals")) / FRAC_STATE_FILE


def _load_state() -> dict:
    try:
        return json.loads(_state_path().read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    """Best-effort atomic write — losing this file costs profit locks, never money."""
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        from io_utils import write_text_atomic
        write_text_atomic(str(p), json.dumps(state, indent=1))
    except Exception as e:
        print(f"stop_monitor: could not persist frac stops (non-fatal): {e}", file=sys.stderr)


def _signal_context() -> tuple:
    """(atr_pct by symbol, kill_switch) from today's signals — best effort.

    Same source decide.py's trail_stops() reads, so both ratchets see identical
    inputs. Missing file → static schedule, which is the safe default.
    """
    try:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        with open(f"signals/{today}.json") as f:
            data = json.load(f)
        macro = data.get("macro_context") or {}
        kill = "MACRO KILL-SWITCH ACTIVE" in (macro.get("trading_guidance") or "")
        return {s["symbol"]: s.get("atr_pct") for s in data.get("signals", [])}, kill
    except Exception:
        return {}, False


def _sell_all(symbol: str, reason: str) -> None:
    """Exit via trade.py so risk_guard + the ledger see the order (never direct)."""
    subprocess.run(["python3", "trade.py", "--symbol", symbol,
                    "--side", "sell", "--qty", "all", "--account", "core"],
                   capture_output=True, text=True)
    try:
        import alerts
        alerts.send(reason)
    except Exception:
        pass


def main():
    try:
        tc = accounts.trading_client("core")
    except Exception as e:
        print(f"stop_monitor: core account unavailable ({e})")
        return
    try:
        if not tc.get_clock().is_open:
            return
    except Exception as e:
        print(f"stop_monitor: clock check failed ({e})")
        return

    carved = accounts.carved_out()
    atr_map, kill_switch = _signal_context()
    hard_floor = stops.hard_exit_pct(kill_switch)
    state = _load_state()
    frac = state.setdefault("core", {})

    stopped, locked_out, held = [], [], set()
    checked = 0
    for p in tc.get_all_positions():
        sym = p.symbol.upper()
        if sym in carved:
            continue
        checked += 1
        try:
            plpc = float(p.unrealized_plpc)
            qty = float(p.qty)
        except Exception:
            continue

        # ── 1. Hard exit floor (every position, whatever its size) ────────────
        if plpc <= hard_floor:
            print(f"*** STOP {sym}: {plpc*100:.1f}% ≤ {hard_floor*100:.1f}% → selling all")
            _sell_all(sym, f"🛑 STOP-LOSS {sym} {plpc*100:.1f}% — sold all (stop_monitor)")
            stopped.append(f"{sym} ({plpc*100:.1f}%)")
            frac.pop(sym, None)          # position gone — don't inherit its high-water mark
            continue

        # ── 2. Trailing profit lock — fractional positions only ──────────────
        # Whole-share positions have a broker GTC stop that decide.py ratchets;
        # managing them here too would fight that order.
        if math.floor(qty) >= 1:
            frac.pop(sym, None)
            continue
        held.add(sym)

        want_pct, label = stops.ratchet_stop_pct(plpc, atr_map.get(sym), kill_switch)
        prev_pct = frac.get(sym, {}).get("stop_pct", hard_floor)
        lock_pct = max(prev_pct, want_pct)          # the ratchet only ever moves UP
        if lock_pct > prev_pct:
            frac[sym] = {"stop_pct": lock_pct, "label": label,
                         "updated": datetime.now(ET).isoformat()}
            print(f"  ↑ FRAC LOCK {sym}: {prev_pct*100:+.1f}% → {lock_pct*100:+.1f}% "
                  f"({label}) — position {plpc*100:+.1f}%")

        # Only the profit-lock band belongs to this branch; the floor above owns
        # everything at or below hard_floor, so the two can never double-fire.
        if lock_pct > hard_floor and plpc <= lock_pct:
            print(f"*** PROFIT LOCK {sym}: {plpc*100:+.1f}% ≤ locked {lock_pct*100:+.1f}% → selling all")
            _sell_all(sym, f"🔒 PROFIT LOCK {sym} {plpc*100:+.1f}% — sold all at the "
                           f"{lock_pct*100:+.1f}% trailing stop (fractional, stop_monitor)")
            locked_out.append(f"{sym} ({plpc*100:+.1f}%)")
            frac.pop(sym, None)
            held.discard(sym)

    # Prune closed positions so a re-entry never inherits an old high-water mark.
    for sym in [s for s in frac if s not in held]:
        frac.pop(sym, None)
    _save_state(state)

    stamp = datetime.now(ET).strftime("%H:%M")
    if stopped or locked_out:
        print(f"stop_monitor {stamp}: stopped {stopped} locked {locked_out}")
    else:
        print(f"stop_monitor {stamp}: {checked} position(s) above floor "
              f"({hard_floor*100:.1f}%), {len(frac)} fractional lock(s) tracked")

    _publish_status(checked, stopped + locked_out)   # best-effort; never affects the logic above


PUBLISH_MIN_MINUTES = 60      # liveness heartbeat cadence when nothing changes
PUBLISH_STATE_FILE = "stopmon_publish.json"


def _should_publish(checked, stopped, now) -> bool:
    """True when this tick is worth a commit: something fired, the book changed
    size, or the last publish is older than the liveness window."""
    path = Path(os.getenv("LEDGER_DIR", "signals")) / PUBLISH_STATE_FILE
    try:
        last = json.loads(path.read_text())
    except Exception:
        last = {}
    changed = bool(stopped) or last.get("checked") != checked
    stale = True
    try:
        stale = (now - datetime.fromisoformat(last["ts"])).total_seconds() >= PUBLISH_MIN_MINUTES * 60
    except Exception:
        pass
    if not (changed or stale):
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from io_utils import write_text_atomic
        write_text_atomic(str(path), json.dumps({"ts": now.isoformat(), "checked": checked}))
    except Exception:
        pass          # if we can't record it, publishing again is the safe error
    return True


def _publish_status(checked, stopped) -> None:
    """Best-effort: push a tiny rolling status so the dashboard can show the monitor
    is alive (status/stopmon-<scheduler>.json, overwritten each run). Never raises.
    Only called when the market is open (main returns early when closed).

    EDGE-TRIGGERED. This used to commit to GitHub on every 5-minute tick: ~78
    commits/day per scheduler, ~156 across both, for a file whose content was
    usually byte-identical apart from a timestamp. Now it pushes when something
    actually happened (a stop or lock fired, the position count changed) or once
    an hour for liveness — roughly 8 commits/day/scheduler.

    Not solvable by writing to `/data` as the audit suggested: the dashboard is a
    separate Railway service with its OWN volume, which is exactly why this goes
    through GitHub at all.
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            return
        scheduler = (os.getenv("ARTIFACT_PREFIX", "").strip("/") or "paper")
        now = datetime.now(ET)
        if not _should_publish(checked, stopped, now):
            return
        body = json.dumps({"ts": now.isoformat(), "scheduler": scheduler,
                           "checked": checked, "stopped": stopped,
                           "stop_pct": abs(stops.hard_exit_pct())})
        from github import Github, GithubException
        repo = Github(token).get_repo("YOUR_GH_USER/YOUR_REPO")
        path = f"status/stopmon-{scheduler}.json"
        msg = f"stop monitor status {scheduler}"
        try:
            existing = repo.get_contents(path)
            repo.update_file(path, msg, body, existing.sha)
        except GithubException as e:
            if getattr(e, "status", None) == 404:
                repo.create_file(path, msg, body)
    except Exception:
        pass


if __name__ == "__main__":
    main()
