"""
microstructure.py — volume profile, liquidity sweeps, and an order-flow proxy.

Pure functions (numpy/pandas only) so they unit-test without market access,
matching the project's options_models.py / walk_forward.py pattern.

Three concepts, all derived from OHLCV bars we already fetch:

  1. Volume Profile (compute_volume_profile) — volume distributed across PRICE
     levels rather than time. Yields POC (most-traded price), the 70% value
     area (VAH/VAL), and where the current price sits relative to them.
     Objective, volume-backed support/resistance.

  2. Liquidity Sweep (detect_liquidity_sweep) — price pierces a prior swing
     low/high (grabbing stops) then closes back through it (a reclaim). A
     bullish sweep at support is a high-probability reversal/entry-timing cue.

  3. Order-flow proxy (compute_cvd_proxy) — cumulative volume delta approximated
     from intraday bar polarity (close vs open). HONEST LIMITATION: this is a
     bar-polarity shadow of real order flow, not aggressor-classified tick
     delta. Real order flow needs tick trades+quotes or L2 depth (a paid data
     upgrade) — see the note at the bottom of this file.
"""
import numpy as np
import pandas as pd

VALUE_AREA_PCT = 0.70   # standard 70% value area
NEAR_PCT       = 0.015  # "near" a level = within 1.5%


# ── 1. Volume Profile ─────────────────────────────────────────────────────────

def compute_volume_profile(high: pd.Series, low: pd.Series, volume: pd.Series,
                           price: float, lookback: int = 60, bins: int = 50) -> dict:
    """
    Bar-based volume profile: distribute each bar's volume uniformly across the
    price bins its [low, high] range overlaps. Returns POC, value area, the
    nearest high-volume node, and the current price's position.
    """
    h = high.tail(lookback).to_numpy(dtype=float)
    l = low.tail(lookback).to_numpy(dtype=float)
    v = volume.tail(lookback).to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(l) & np.isfinite(v) & (v > 0)
    h, l, v = h[mask], l[mask], v[mask]
    if len(h) < 20:
        return {"available": False}

    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return {"available": False}

    edges   = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vp      = np.zeros(bins)

    for bh, bl, bv in zip(h, l, v):
        # bins this bar's range overlaps; spread volume evenly across them
        first = max(0, np.searchsorted(edges, bl, side="right") - 1)
        last  = min(bins - 1, np.searchsorted(edges, bh, side="right") - 1)
        n = last - first + 1
        if n > 0:
            vp[first:last + 1] += bv / n

    total = vp.sum()
    if total <= 0:
        return {"available": False}

    poc_i = int(vp.argmax())
    poc   = float(centers[poc_i])

    # Expand the value area outward from the POC until it holds 70% of volume.
    lo_i = hi_i = poc_i
    acc = vp[poc_i]
    target = total * VALUE_AREA_PCT
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        below = vp[lo_i - 1] if lo_i > 0 else -1
        above = vp[hi_i + 1] if hi_i < bins - 1 else -1
        if above >= below:
            hi_i += 1; acc += vp[hi_i]
        else:
            lo_i -= 1; acc += vp[lo_i]
    val, vah = float(centers[lo_i]), float(centers[hi_i])

    # High-volume nodes (bins ≥ 1.5× mean) → cluster support/resistance
    mean_v = vp.mean()
    hvn = [round(float(centers[i]), 2) for i in range(bins) if vp[i] >= 1.5 * mean_v]
    nearest_hvn = min(hvn, key=lambda x: abs(x - price)) if hvn else None

    if price > vah:
        position = "above_value"
    elif price < val:
        position = "below_value"
    else:
        position = "in_value"

    return {
        "available":      True,
        "poc":            round(poc, 2),
        "vah":            round(vah, 2),
        "val":            round(val, 2),
        "position":       position,
        "dist_to_poc_pct": round((price - poc) / poc * 100, 2),
        "near_poc":       abs(price - poc) / poc <= NEAR_PCT,
        "nearest_hvn":    nearest_hvn,
        "near_hvn":       bool(nearest_hvn and abs(price - nearest_hvn) / price <= NEAR_PCT),
        "hvn_count":      len(hvn),
    }


# ── 2. Liquidity Sweep ────────────────────────────────────────────────────────

def _swing_levels(high: np.ndarray, low: np.ndarray, k: int):
    """Pivot highs/lows: extremum of a ±k bar window. Returns (lows, highs) of
    (index, price), excluding the last k bars (not yet confirmable)."""
    lows, highs = [], []
    n = len(high)
    for i in range(k, n - k):
        win_lo = low[i - k:i + k + 1]
        win_hi = high[i - k:i + k + 1]
        if low[i] == win_lo.min():
            lows.append((i, float(low[i])))
        if high[i] == win_hi.max():
            highs.append((i, float(high[i])))
    return lows, highs


def detect_liquidity_sweep(high: pd.Series, low: pd.Series, close: pd.Series,
                           lookback: int = 60, k: int = 3, recent: int = 3) -> dict:
    """
    Detect a stop-hunt + reclaim in the last `recent` bars:
      bullish — a bar's low pierces a prior swing low but close reclaims above it
      bearish — a bar's high pierces a prior swing high but close rejects below it
    """
    h = high.tail(lookback).to_numpy(dtype=float)
    l = low.tail(lookback).to_numpy(dtype=float)
    c = close.tail(lookback).to_numpy(dtype=float)
    n = len(h)
    if n < 2 * k + recent + 2:
        return {"sweep": None}

    lows, highs = _swing_levels(h, l, k)

    for j in range(n - recent, n):
        # prior swing lows established before this bar
        prior_lows = [p for (idx, p) in lows if idx < j - k]
        for lvl in prior_lows:
            if l[j] < lvl and c[j] > lvl:                 # pierced then reclaimed
                return {"sweep": "bullish", "level": round(lvl, 2),
                        "bars_ago": n - 1 - j,
                        "note": f"swept liquidity below {lvl:.2f} and reclaimed"}
        prior_highs = [p for (idx, p) in highs if idx < j - k]
        for lvl in prior_highs:
            if h[j] > lvl and c[j] < lvl:                 # pierced then rejected
                return {"sweep": "bearish", "level": round(lvl, 2),
                        "bars_ago": n - 1 - j,
                        "note": f"swept liquidity above {lvl:.2f} and rejected"}
    return {"sweep": None}


# ── 3. Order-flow proxy (intraday) ────────────────────────────────────────────

def compute_cvd_proxy(open_: pd.Series, close: pd.Series, high: pd.Series,
                      low: pd.Series, volume: pd.Series) -> dict:
    """
    Cumulative Volume Delta approximated from bar polarity (a bar's whole volume
    is counted as buying if it closed up, selling if down). Also flags
    price/CVD divergence over the session.

    PROXY ONLY — not aggressor-classified tick delta. See module note.
    """
    o = open_.to_numpy(dtype=float); c = close.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)
    if len(c) < 4 or np.nansum(v) <= 0:
        return {"available": False}

    sign  = np.sign(c - o)               # +1 up bar, −1 down bar, 0 doji
    delta = sign * v
    cvd   = float(np.nansum(delta))
    buy_v = float(np.nansum(v[sign > 0]))
    tot_v = float(np.nansum(v))
    buying_pressure = buy_v / tot_v if tot_v else 0.5

    # Divergence: compare price vs cumulative-delta direction over the session halves
    cum = np.nancumsum(delta)
    mid = len(c) // 2
    price_up = c[-1] > c[mid]
    cvd_up   = cum[-1] > cum[mid]
    if price_up and not cvd_up:
        divergence = "bearish"     # price rising on weakening net buying
    elif (not price_up) and cvd_up:
        divergence = "bullish"     # price falling but net buying building
    else:
        divergence = None

    return {
        "available":        True,
        "cvd":              round(cvd, 0),
        "buying_pressure":  round(buying_pressure, 3),
        "divergence":       divergence,
    }


# ── Future upgrade: real order flow ───────────────────────────────────────────
# compute_cvd_proxy infers buy/sell from candle direction. True order flow needs
# trades classified by aggressor side (uptick/Lee-Ready) or L2 depth — i.e. a
# tick-level trades+quotes feed (Polygon paid tier / Databento). When that data
# is added, replace the polarity heuristic here with per-trade classification;
# the field names (cvd, buying_pressure, divergence) can stay so nothing
# downstream changes.
