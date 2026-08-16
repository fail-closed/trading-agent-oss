"""
entry_mix.py — track how each open position was entered, and steer the balance.

CLAUDE.md has always asked for a blend of dip entries (buying weakness) and
breakout entries (buying strength), but nothing measured the actual blend, so
the instruction was unenforceable — the model had no idea what it was already
holding, and the 60/40 target was a guess.

Two things changed that:

1. It is now measured. Every executed BUY stamps an entry type, so the current
   mix of open positions is a fact in the session context rather than a vibe.

2. The target moved from 60% dip to 40% dip. A 5-year decomposition of our own
   universe (strategy_lens.py), benchmarked against equal-weighting all 198
   watchlist names, found:

       breakout entries   alpha +6.6%/yr (t 1.87), Sharpe 1.29, maxDD -14.8%
       dip entries        alpha +1.2%/yr (t 0.32), Sharpe 0.94, maxDD -17.8%

   Neither clears a 95% bar on five years of data, so this is a tilt and not a
   reversal — dips stay a large minority of the book. The dip sleeve also loads
   on short-term reversion (beta +0.14, t 8.5) and liquidity provision
   (beta +0.04, t 2.8), which were the two of the four primitive bets that did
   NOT pay in this universe; the breakout sleeve loads on trend (beta +0.10,
   t 9.0), which did.

Fail-open by design: every function swallows its own errors and degrades to
"unknown". Nothing here may ever block a trade.
"""
import json
import os
from pathlib import Path

STATE_FILE = "entry_types.json"

BREAKOUT = "breakout"
DIP      = "dip"
SIGNAL   = "signal"      # scored in on neither a dip nor a breakout setup


def _path() -> Path:
    # Read the env on every call rather than caching at import. Capturing it at
    # module scope means the path is frozen by whichever import happened first,
    # which breaks anything that sets RISK_STATE_DIR after import.
    return Path(os.getenv("RISK_STATE_DIR", "signals")) / STATE_FILE


def load() -> dict:
    try:
        p = _path()
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _save(state: dict) -> None:
    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_path()), state)
    except Exception:
        pass


def classify(signal: dict) -> str:
    """Entry type for a signal we are about to buy.

    Breakout wins ties: a name can be flagged both (a Donchian break that is
    still under its SMA20), and the breakout thesis is what we are acting on.
    """
    if not isinstance(signal, dict):
        return SIGNAL
    if signal.get("is_breakout"):
        return BREAKOUT
    if signal.get("is_dip"):
        return DIP
    return SIGNAL


def record(symbol: str, entry_type: str, date: str = None) -> None:
    """Stamp an executed BUY. Re-buying a held name keeps the original type."""
    try:
        state = load()
        sym   = symbol.upper()
        if sym not in state:
            state[sym] = {"entry_type": entry_type, "date": date}
            _save(state)
    except Exception:
        pass


def drop(symbol: str) -> None:
    """Forget a symbol once the position is fully closed."""
    try:
        state = load()
        if state.pop(symbol.upper(), None) is not None:
            _save(state)
    except Exception:
        pass


def reconcile(held_symbols) -> None:
    """Drop entries for symbols we no longer hold (covers exits that bypass decide.py,
    e.g. stop_monitor, and any manual intervention)."""
    try:
        held  = {s.upper() for s in held_symbols}
        state = load()
        pruned = {k: v for k, v in state.items() if k in held}
        if len(pruned) != len(state):
            _save(pruned)
    except Exception:
        pass


def target_dip_share() -> float:
    """Target share of open positions entered as dips. Env-tunable.

    Default 0.40 (was an unenforced 0.60 in prose). Set ENTRY_MIX_DIP_TARGET=0.60
    to restore the old target without a deploy.
    """
    try:
        return min(1.0, max(0.0, float(os.getenv("ENTRY_MIX_DIP_TARGET", "0.40"))))
    except (TypeError, ValueError):
        return 0.40


def current(held_symbols) -> dict:
    """Mix of currently-open positions, plus which side to lean on next.

    Positions opened before this tracking existed count as `unknown` and are
    excluded from the shares, so the guidance is based only on what we actually
    know rather than assuming a type.
    """
    try:
        held  = {s.upper() for s in held_symbols}
        state = load()
        counts = {BREAKOUT: 0, DIP: 0, SIGNAL: 0, "unknown": 0}
        for sym in held:
            counts[state.get(sym, {}).get("entry_type", "unknown")] += 1

        classified = counts[BREAKOUT] + counts[DIP]
        target     = target_dip_share()
        dip_share  = (counts[DIP] / classified) if classified else None

        if dip_share is None:
            lean = ("no classified positions yet — either entry type is fine, "
                    "breakout preferred on a tie")
        elif dip_share > target + 0.10:
            lean = (f"over-weight dips ({dip_share*100:.0f}% vs {target*100:.0f}% target) "
                    f"— prefer a breakout candidate")
        elif dip_share < target - 0.10:
            lean = (f"under-weight dips ({dip_share*100:.0f}% vs {target*100:.0f}% target) "
                    f"— a dip entry is fine here")
        else:
            lean = f"on target ({dip_share*100:.0f}% dip vs {target*100:.0f}%) — pick on merit"

        return {
            "counts":           counts,
            "classified":       classified,
            "dip_share":        round(dip_share, 3) if dip_share is not None else None,
            "target_dip_share": target,
            "lean":             lean,
        }
    except Exception:
        return {"counts": {}, "classified": 0, "dip_share": None,
                "target_dip_share": 0.40, "lean": "entry-mix unavailable"}
