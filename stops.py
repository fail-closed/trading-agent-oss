"""stops.py — THE stop-loss and trailing-ratchet rule, defined exactly once.

Before this module the −8% stop lived in five places with three different
override paths (`LIVE_STOP_PCT`, `ATR_STOP_MULT`, bare constants), and the
trailing ratchet lived only inside `decide.py`. That is the MAX_ALLOC_SCALE
failure shape (OPERATIONS.md §6 row 11): one rule computed inline in several
files, drifting silently until a position exits at the wrong price.

Three layers, deliberately kept distinct — they are NOT one number:

  initial_stop_pct()  the GTC stop distance placed at fill time  (trade.py)
  ratchet_stop_pct()  where the stop belongs given the current gain
                      (decide.py for whole-share broker stops; stop_monitor.py
                      for fractional positions the broker can't hold a stop on)
  hard_exit_pct()     the sell-everything floor on unrealized P&L
                      (decide.py session exits, stop_monitor.py every 5 min)

`tests/test_stop_rule.py` fails CI if any other file re-defines these.
"""
import os

# Magnitudes (positive). Signed values come from the functions below.
STOP_LOSS_PCT = 0.08          # the −8% floor in CLAUDE.md's exit table
KILL_STOP_PCT = 0.05          # tightened floor while the macro kill-switch is on

# Profit-taking, the other half of the mechanical exit table in CLAUDE.md.
# Here rather than in decide.py so every consumer — the session exits AND the
# dashboard's target price — reads one number. The dashboard used to hardcode
# `avg_entry * 1.20`, which is fine until the day this changes.
TAKE_PROFIT_FULL = 0.20       # +20% → sell all
TAKE_PROFIT_PARTIAL = 0.10    # +10% → sell half

# Trailing ratchet — (trigger_gain, stop_level_relative_to_entry), descending.
# The stop only ever moves UP; for broker-held stops the GTC order enforces that,
# for fractional positions stop_monitor.py persists the high-water mark.
TRAILING_STOP_SCHEDULE = [
    (0.15, 0.08),   # +15% gain → lock +8%
    (0.10, 0.05),   # +10% gain → lock +5%
    (0.05, 0.00),   # +5%  gain → lock break-even
]

# ATR-based variants (preferred when the signal file carries atr_pct).
_ATR_MULTS = {            # (tp3, tp2, tp1, initial_sl) in ATR units
    False: (3.0, 2.0, 1.0, 2.5),        # normal
    True:  (2.0, 1.0, 0.5, 1.5),        # kill-switch: tighter
}
ATR_STOP_FLOOR = 0.04     # clamp for the ATR initial stop
ATR_STOP_CEIL = 0.15


def hard_exit_pct(kill_switch: bool = False) -> float:
    """SIGNED unrealized-P&L level at or below which we sell the whole position.

    `LIVE_STOP_PCT` overrides the floor for EVERY consumer (it used to be read
    only by stop_monitor.py, so setting it moved the 5-minute monitor's stop
    while decide.py's session exit stayed at −8%). Unset on both services as of
    2026-08-14, so routing it here is zero behavior change today.
    """
    if kill_switch:
        return -KILL_STOP_PCT
    try:
        return -abs(float(os.getenv("LIVE_STOP_PCT", STOP_LOSS_PCT)))
    except (TypeError, ValueError):
        return -STOP_LOSS_PCT


def atr_stop_enabled() -> bool:
    """True when ATR_STOP_MULT is configured — lets callers decide whether it is
    worth fetching ATR bars WITHOUT re-reading the knob (one reader, here)."""
    try:
        return float(os.getenv("ATR_STOP_MULT", "0") or 0) > 0
    except ValueError:
        return False


def initial_stop_pct(atr14: float = None, price: float = None,
                     base: float = None) -> float:
    """POSITIVE stop distance for the GTC order placed at fill.

    Volatility-adjusted when ATR_STOP_MULT is set and ATR is known (noisy names
    get room, slow names get a tighter stop), clamped to 4%–15%; otherwise the
    flat −8%. `base` lets trade.py's --stop_loss_pct override the flat default.
    """
    base = STOP_LOSS_PCT if base is None else abs(base)
    try:
        mult = float(os.getenv("ATR_STOP_MULT", "0") or 0)
    except ValueError:
        mult = 0.0
    if mult > 0 and atr14 and price and price > 0:
        return min(max(mult * atr14 / price, ATR_STOP_FLOOR), ATR_STOP_CEIL)
    return base


def ratchet_stop_pct(pl_pct: float, atr_pct: float = None,
                     kill_switch: bool = False) -> tuple:
    """Where the stop belongs, relative to ENTRY, for a position now at `pl_pct`.

    Returns (stop_pct, label). stop_pct is signed and relative to the entry
    price: −0.08 = 8% below entry, 0.0 = break-even, +0.08 = 8% profit locked.
    ATR-scaled when atr_pct is known, else the static schedule. Callers are
    responsible for never moving a stop DOWN.
    """
    tag = " ⚠KILL" if kill_switch else ""
    if atr_pct:
        tp3, tp2, tp1, sl = _ATR_MULTS[bool(kill_switch)]
        if pl_pct >= tp3 * atr_pct:
            pct = 1.5 * atr_pct
            return pct, f"+{pct*100:.1f}% (ATR-trail{tag})"
        if pl_pct >= tp2 * atr_pct:
            pct = 0.5 * atr_pct
            return pct, f"+{pct*100:.1f}% (ATR-trail{tag})"
        if pl_pct >= tp1 * atr_pct:
            return 0.0, f"break-even (ATR-trail{tag})"
        pct = -sl * atr_pct
        return pct, f"{pct*100:.1f}% (ATR-init{tag})"

    for trigger, level in TRAILING_STOP_SCHEDULE:
        # The kill-switch reaches each profit lock 30% sooner.
        if pl_pct >= (trigger * 0.7 if kill_switch else trigger):
            return level, (f"break-even{tag}" if level == 0 else f"+{level*100:.0f}%{tag}")
    pct = hard_exit_pct(kill_switch)
    return pct, f"{pct*100:.1f}% (Static{tag})"
