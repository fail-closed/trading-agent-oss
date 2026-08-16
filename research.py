"""
research.py — fetch market data and compute technical signals for all watchlist stocks.
Outputs signals/YYYY-MM-DD.json for the trading agent to read.

Signals (5 components, score −5 to +5):
  RSI(14)         oversold<30 (+1) / overbought>70 (−1)
  MA Crossover    golden cross (+1) / death cross (−1)
  MACD(12,26,9)   MACD > signal (+1) / MACD < signal (−1)
  Bollinger(20)   pct_b < 0.10 (+1) / pct_b > 0.90 (−1)
  Volume          > 1.5× 20-day avg (+1) / < 0.8× avg (−1)

Adaptive weights derived from 92,823-outcome knowledge build.
Market regime: VIX level included to guide threshold adjustments.
News sentiment: last 24h headlines fetched for actionable symbols (|score| ≥ 1).

Data sources:
  DataClient (Polygon if key set, else yfinance) — historical daily bars
  Polygon news API / yfinance                    — recent headlines
  yfinance                                       — VIX
  Alpaca                                         — account state, positions, live quotes
"""
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from data_client import DataClient


# ── Adaptive weights from knowledge build ────────────────────────────────────
# Derived from 92,823 daily outcomes. Bearish signals get lower weights because
# they still predict next-day gains ~60% of the time (market upward bias).
SIGNAL_WEIGHTS = {
    "rsi":  {"bullish": 1.5, "bearish": 0.3},  # RSI oversold: 63.2% win, +1.92% avg
    "ma":   {"bullish": 1.0, "bearish": 0.4},  # Death cross still 60.2% bullish
    "macd": {"bullish": 1.0, "bearish": 0.3},  # Bearish MACD 60.6% bullish
    "bb":   {"bullish": 1.5, "bearish": 0.3},  # BB lower band: +1.11% avg return
    "vol":  {"bullish": 1.0, "bearish": 0.5},  # Volume confirmation
}

# VIX regime thresholds
VIX_REGIMES = [
    (35, "extreme_fear",    2, "Raise buy threshold +2 — only highest conviction setups"),
    (25, "elevated_fear",   1, "Raise buy threshold +1 — tighter entry bar"),
    (20, "neutral_cautious",0, "Standard thresholds, remain alert"),
    (15, "neutral",         0, "Standard thresholds apply"),
    (0,  "complacency",     0, "Standard thresholds — market calm"),
]


# ── Signal computation ────────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return round(float((100 - 100/(1+rs)).iloc[-1]), 2)


def compute_macd(close: pd.Series):
    macd_line   = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def compute_sma(close: pd.Series, period: int) -> float:
    return float(close.rolling(period).mean().iloc[-1])


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return float(upper.iloc[-1]), float(ma.iloc[-1]), float(lower.iloc[-1])


# A partial session's volume is not a low-volume day. research.py runs at 09:45,
# ~15 minutes into the session, so a provider whose latest bar is the CURRENT
# (incomplete) day reports a fraction of a normal day's shares. Compared against
# a 20-day average of COMPLETE days that reads ~0.1×, which the rules treat as
# "low volume → −1", and CLAUDE.md makes a −1 a hard skip even in deployment mode.
#
# This is not hypothetical: on 2026-08-13 live had 43 of 44 symbols at −1 (SPY,
# QQQ, AAPL, MSFT, NVDA all "0.1× 20d avg") while paper — same code, same time,
# but with a POLYGON_API_KEY so it used complete prior-day bars — had 10 of 31 and
# a healthy 0.7–1.7× spread. Live had been silently disqualifying ~98% of its
# candidates before any other gate ran, which is why it bought about one name a
# day no matter what its position sizing or buy caps were set to.
#
# Below this ratio the reading is treated as an incomplete bar rather than a real
# signal: no honest full session trades a fifth of its 20-day average, so a value
# that low is evidence about the data, not the tape. Neutral (0) is the safe
# answer — it neither manufactures a buy signal nor hard-blocks the symbol.
PARTIAL_BAR_RATIO = 0.20


def compute_volume_signal(volume: pd.Series) -> tuple:
    """Compare latest volume to 20-day average."""
    if len(volume) < 21 or volume.iloc[-1] == 0:
        return 0, 1.0, "volume data unavailable"
    avg_20d = float(volume.iloc[-21:-1].mean())   # exclude today
    today   = float(volume.iloc[-1])
    if avg_20d == 0:
        return 0, 1.0, "volume baseline zero"
    rel = today / avg_20d
    if rel < PARTIAL_BAR_RATIO:
        return 0, round(rel, 2), (f"volume inconclusive ({rel:.2f}× 20d avg — "
                                  f"partial/stale bar, not a low-volume day)")
    if rel > 1.5:   return  1, round(rel, 2), f"high volume ({rel:.1f}× 20d avg)"
    elif rel < 0.8: return -1, round(rel, 2), f"low volume ({rel:.1f}× 20d avg)"
    else:           return  0, round(rel, 2), f"normal volume ({rel:.1f}× 20d avg)"


def compute_support_levels(close: pd.Series, high: pd.Series, low: pd.Series,
                           price: float) -> dict:
    """
    Detect if current price is near a historical support level.
    A dip at support is higher probability than a dip in open air.

    Checks:
      - 52-week range position (near low = strong support)
      - Previous local lows (price bounced here before)
      - Fibonacci retracement of last major swing
      - Round number proximity ($50/$100/$200 etc.)

    Returns support_score (0-100), at_support bool, and detail dict.
    """
    score  = 0
    detail = {}

    # 1. 52-week range position (strongest signal)
    high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
    low_52w  = float(low.tail(252).min())  if len(low)  >= 252 else float(low.min())
    rng = high_52w - low_52w
    pct_from_low = (price - low_52w) / rng if rng > 0 else 0.5
    detail["52w_range_pct"]  = round(pct_from_low * 100, 1)
    detail["52w_low"]        = round(low_52w, 2)
    detail["52w_high"]       = round(high_52w, 2)
    if pct_from_low < 0.10:   score += 40   # near 52-week low — very strong support
    elif pct_from_low < 0.25: score += 20   # lower quartile — moderate support

    # 2. Previous local lows in last 6 months (price bounced here before)
    recent_low = low.tail(126)   # ~6 months of trading days
    local_mins = []
    for i in range(2, len(recent_low) - 2):
        v = recent_low.iloc[i]
        if v < recent_low.iloc[i-1] and v < recent_low.iloc[i-2] \
           and v < recent_low.iloc[i+1] and v < recent_low.iloc[i+2]:
            local_mins.append(float(v))

    nearest_prev_low = None
    for lvl in sorted(local_mins):
        if abs(price - lvl) / price < 0.03:   # within 3% of a previous low
            score += 30
            nearest_prev_low = round(lvl, 2)
            break
    detail["near_prev_low"] = nearest_prev_low

    # 3. Fibonacci retracement of last major swing
    # Swing = distance from the 6-month low to the 6-month high. The lookbacks
    # MUST match: this read a 126-bar high against a 252-bar low, contradicting
    # its own comment and stretching the swing whenever the 12-month low sat
    # outside the 6-month window — which pushed every retracement level down and
    # made support_score fire in the wrong places.
    FIB_LOOKBACK = 126
    swing_high = float(high.tail(FIB_LOOKBACK).max())
    swing_low  = float(low.tail(FIB_LOOKBACK).min())
    swing      = swing_high - swing_low
    fib_hit    = None
    if swing > 0 and price < swing_high:
        for pct, label in [(0.618, "61.8%"), (0.500, "50.0%"), (0.382, "38.2%")]:
            fib_level = swing_high - swing * pct
            if abs(price - fib_level) / price < 0.025:   # within 2.5%
                score += 20
                fib_hit = f"{label} at ${fib_level:.2f}"
                break
    detail["fibonacci"] = fib_hit

    # 4. Round number proximity (institutional order clustering)
    round_levels = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100,
                    120, 125, 150, 175, 200, 250, 300, 400, 500,
                    600, 750, 1000, 1250, 1500, 2000, 2500]
    round_hit = None
    for lvl in round_levels:
        if abs(price - lvl) / price < 0.015:   # within 1.5%
            score += 10
            round_hit = lvl
            break
    detail["round_number"] = round_hit

    score = min(score, 100)
    return {
        "support_score": score,
        "at_support":    score >= 35,   # at least one meaningful support level
        "detail":        detail,
    }


def compute_breakout_signal(close: pd.Series, sma50: float, sma200: float,
                            pct_b: float, vol_signal: int, score: int,
                            overextended: bool) -> tuple:
    """
    Detect momentum breakout setups — buying strength, not weakness.
    Opposite of dip buying: price rising on expanding volume.

    Two triggers:
    1. Fresh golden cross (SMA50 just crossed above SMA200 within 5 days)
       + score ≥ 2 + volume confirms + not overextended
    2. Very strong signal (score ≥ 3) + volume confirms + not deeply extended

    Returns (is_breakout: bool, fresh_golden_cross: bool, days_since_cross: int | None)
    """
    # Check for fresh golden cross
    fresh_golden_cross  = False
    days_since_cross    = None

    if sma50 > sma200 and len(close) >= 210:
        sma50_s  = close.rolling(50).mean()
        sma200_s = close.rolling(200).mean()
        for days_ago in range(1, 8):           # look back up to 7 days
            idx = -(days_ago + 1)
            if abs(idx) > len(sma50_s):
                break
            if (not pd.isna(sma50_s.iloc[idx]) and
                    sma50_s.iloc[idx] <= sma200_s.iloc[idx]):
                fresh_golden_cross = True
                days_since_cross   = days_ago
                break

    # Breakout type 1: fresh golden cross + confirmed by other signals
    breakout_on_cross = (
        fresh_golden_cross and
        score >= 2 and
        vol_signal >= 1 and        # volume expanding = real move
        pct_b >= 0.30 and          # price above lower third of Bollinger
        not overextended            # not already at extreme extension
    )

    # Breakout type 2: very strong signal regardless of cross
    # (score ≥ 3 means ≥3 signals bullish — rare, high conviction)
    breakout_on_strength = (
        score >= 3 and
        vol_signal >= 1 and
        pct_b >= 0.30 and
        not overextended
    )

    is_breakout = breakout_on_cross or breakout_on_strength
    return is_breakout, fresh_golden_cross, days_since_cross


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """True Range = max(H-L, |H-Cp|, |L-Cp|)"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 4)


def compute_mtf_alignment(close: pd.Series, daily_bias: str) -> dict:
    """
    Multi-timeframe alignment: does the weekly trend agree with the daily bias?
    Weekly trend = last weekly close vs its 20-week SMA. Alignment across
    timeframes is a confidence modifier, not a standalone signal.
    """
    if len(close) < 110 or not isinstance(close.index, pd.DatetimeIndex):
        return {"weekly_trend": None, "aligned": None}
    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < 21:
        return {"weekly_trend": None, "aligned": None}
    sma20w = weekly.rolling(20).mean()
    weekly_trend = "uptrend" if float(weekly.iloc[-1]) > float(sma20w.iloc[-1]) else "downtrend"
    return {"weekly_trend": weekly_trend, "aligned": weekly_trend == daily_bias}


def compute_bb_squeeze(close: pd.Series) -> tuple:
    """
    Bollinger Band width percentile over the trailing year.
    Low band width (a "squeeze") precedes volatility expansion — breakouts
    fired from a squeeze have more room to run.
    Returns (bb_width_percentile 0-100 or None, squeeze: bool).
    """
    if len(close) < 120:
        return None, False
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    width = (4 * std20) / sma20.where(sma20 > 0)
    w = width.dropna().tail(252)
    if len(w) < 60:
        return None, False
    pct = float((w < w.iloc[-1]).mean() * 100)
    return round(pct, 1), pct < 20.0


def compute_donchian_breakout(close: pd.Series, high: pd.Series, period: int = 20) -> bool:
    """Classic Donchian channel breakout: today's close above the prior 20-day high."""
    src = high if len(high) >= period + 1 else close
    if len(src) < period + 1 or len(close) < 1:
        return False
    prior_high = float(src.iloc[-(period + 1):-1].max())
    return bool(float(close.iloc[-1]) > prior_high)


def compute_momentum(close: pd.Series, splits: dict = None) -> dict:
    """Cross-sectional 12-1 momentum: the 12-month return, skipping the most
    recent month.

    The skip is the whole trick. Raw 12-month return is contaminated by
    short-term reversal in the last few weeks — the names that ran hardest most
    recently tend to give some back — so the classical construction drops the
    last 21 sessions. That makes this genuinely different from our existing
    trend signals: MA crossover and MACD are *fast* trend reads on the last
    weeks, while this is a *slow* one that deliberately ignores them.

    Why it earns a place: in the 5-year decomposition of our own universe
    (strategy_lens.py), a long/short 12-1 book had only **+0.11 correlation** to
    what we currently trade — by far the most independent of the four primitive
    bets. Blending 25% of it into the current book lifted Sharpe 1.08 → 1.18 and
    cut max drawdown 17.5% → 15.5%. Its own Sharpe (0.59) is unremarkable; the
    diversification is the point.

    Needs ~253 sessions. research.py fetches 18mo (~378), so this is populated
    for any name with a full history and None for recent listings.

    UNADJUSTED SPLITS: our bar source does not always back-adjust splits, and a
    split anywhere inside the lookback silently inverts the reading. Real case,
    2026-08-11: KLAC showed $912 twelve months ago against $193 today, giving
    −74.6% — while the split-adjusted truth was $91 → $200, i.e. **+144%**. That
    is a sign flip on a decile-ranked name, which is exactly the input that would
    put it at the top or bottom of the sleeve.

    research.py's existing `gap_alert` only compares the last two bars, so a
    split two months back passes it.

    Two defences, in order of preference:

    1. `splits` (from corporate_actions.fetch_all) back-adjusts the series so the
       reading is CORRECT. This is strictly better than skipping it — abstaining
       drops the name from the cross-section, which is how we lost the universe's
       strongest momentum name to a data artifact.
    2. If no split data is available, or a gap remains that no known corporate
       action explains, the whole window is scanned and any single session outside
       [−35%, +100%] withholds the reading. Calibrated on the live watchlist, where
       KLAC's split day was −89.4% and the worst genuine move was −19.8% (SATS's
       real +70.3% day sits safely under the upper bound). Deliberately
       conservative: a genuine −40% collapse also withholds, and momentum on a
       name that just halved is not worth trusting anyway.
    """
    if close is None or len(close) < 253:
        return {"mom_12_1": None, "mom_1m": None, "mom_suspect": False}
    try:
        if splits:
            import corporate_actions
            close = corporate_actions.adjust_close(close, splits)

        window = close.iloc[-253:]
        step = window.pct_change().dropna()
        if len(step) and (step.min() <= -0.35 or step.max() >= 1.00):
            # Survived adjustment (or none was available) — an unexplained gap.
            return {"mom_12_1": None, "mom_1m": None, "mom_suspect": True}

        px_now, px_1m, px_12m = float(close.iloc[-1]), float(close.iloc[-22]), float(close.iloc[-253])
        if px_12m <= 0 or px_1m <= 0:
            return {"mom_12_1": None, "mom_1m": None, "mom_suspect": False}
        return {
            "mom_12_1":    round((px_1m / px_12m - 1) * 100, 2),
            "mom_1m":      round((px_now / px_1m - 1) * 100, 2),
            "mom_suspect": False,
        }
    except (IndexError, ValueError, TypeError):
        return {"mom_12_1": None, "mom_1m": None, "mom_suspect": False}


def compute_weighted_score(s_rsi: int, s_ma: int, s_macd: int, s_bb: int, s_vol: int = 0) -> float:
    """
    Empirically-weighted composite score.
    Bearish components are downweighted because knowledge build shows they
    still predict next-day gains ~60% of the time.
    """
    total = 0.0
    for sig, key in [(s_rsi, "rsi"), (s_ma, "ma"), (s_macd, "macd"), (s_bb, "bb"), (s_vol, "vol")]:
        w = SIGNAL_WEIGHTS[key]["bullish"] if sig > 0 else SIGNAL_WEIGHTS[key]["bearish"]
        total += sig * w
    return round(total, 2)


def score_signals(price, rsi_val, sma50, sma200, macd_val, macd_sig,
                  bb_upper, bb_middle, bb_lower, vol_signal=0, vol_note=""):
    score   = 0
    details = {}

    # Directional Bias (Regime-Aware): is the primary trend up or down?
    uptrend = bool(price > sma200)
    details["directional_bias"] = "uptrend" if uptrend else "downtrend"

    # RSI
    if rsi_val < 30:
        score += 1; rs, rn = 1,  f"oversold ({rsi_val:.1f} < 30)"
    elif rsi_val > 70:
        if uptrend:
            # In an uptrend, overbought RSI is strength/momentum, not exhaustion
            rs, rn = 0, f"overbought strength ({rsi_val:.1f} > 70) — ignored in uptrend"
        else:
            score -= 1; rs, rn = -1, f"overbought ({rsi_val:.1f} > 70)"
    else:
        rs, rn = 0,  f"neutral ({rsi_val:.1f})"
    details["rsi"] = {"value": round(rsi_val, 2), "signal": rs, "note": rn}

    # MA crossover
    if sma50 > sma200: score += 1; ms, mn = 1,  f"golden cross (SMA50 {sma50:.2f} > SMA200 {sma200:.2f})"
    else:              score -= 1; ms, mn = -1, f"death cross (SMA50 {sma50:.2f} < SMA200 {sma200:.2f})"
    details["ma_crossover"] = {"sma50": round(sma50, 2), "sma200": round(sma200, 2), "signal": ms, "note": mn}

    # MACD — require minimum gap of 0.05% of price to filter razor-thin crossovers
    macd_gap     = macd_val - macd_sig
    min_gap      = abs(price) * 0.0005   # 0.05% of price
    if abs(macd_gap) < min_gap:
        cs, cn = 0, f"flat — gap {macd_gap:.4f} below min threshold ({min_gap:.4f})"
    elif macd_gap > 0:
        score += 1; cs, cn = 1, f"bullish ({macd_val:.4f} > signal {macd_sig:.4f})"
    else:
        score -= 1; cs, cn = -1, f"bearish ({macd_val:.4f} < signal {macd_sig:.4f})"
    details["macd"] = {"macd": round(macd_val, 4), "signal_line": round(macd_sig, 4),
                        "signal": cs, "gap": round(macd_gap, 4), "note": cn}

    # Bollinger Bands
    band_width  = bb_upper - bb_lower
    pct_b       = (price - bb_lower) / band_width if band_width > 0 else 0.5
    overextended = pct_b > 1.3   # significantly above upper band — hard skip flag
    if pct_b < 0.10:
        score += 1; bs, bn = 1,  f"near lower band (pct_b {pct_b:.2f})"
    elif pct_b > 0.90:
        if uptrend:
            # Strength continuation in uptrend
            bs, bn = 0, f"upper band strength (pct_b {pct_b:.2f}) — ignored in uptrend"
        else:
            score -= 1; bs, bn = -1, f"near upper band (pct_b {pct_b:.2f})"
    else:
        bs, bn = 0,  f"mid-range (pct_b {pct_b:.2f})"
    if overextended:
        bn += f" ⚠ OVEREXTENDED (>{1.3}) — hard skip for new buys"
    details["bollinger"] = {
        "upper": round(bb_upper, 2), "middle": round(bb_middle, 2), "lower": round(bb_lower, 2),
        "pct_b": round(pct_b, 3), "signal": bs, "overextended": overextended, "note": bn,
    }

    # Volume confirmation (5th signal)
    score += vol_signal
    if vol_signal == 1:    vs, vn = 1,  f"confirms — {vol_note}"
    elif vol_signal == -1: vs, vn = -1, f"warns — {vol_note}"
    else:                  vs, vn = 0,  f"neutral — {vol_note}"
    details["volume"] = {"signal": vs, "note": vn}

    # Adaptive weighted score
    weighted = compute_weighted_score(rs, ms, cs, bs, vs)
    details["_weighted_score"] = weighted

    return score, details


def action_from_score(score: int, min_buy: int = 2, max_sell: int = -2) -> str:
    if score >= min_buy:  return "BUY"
    if score <= max_sell: return "SELL"
    return "HOLD"


def deploy_bands() -> dict:
    """Cash-percentage band edges for the deploy modes, env-overridable.

    Defaults reproduce the original hardcoded 0.30/0.40/0.60 ladder and a 70%
    invested target, so nothing changes unless a service sets these. They are
    env-driven so the live and paper profiles can run different deployment
    postures without a code change, and so a posture change is reversible by
    unsetting a variable rather than shipping a deploy.

    PRESERVE must stay at or above the hard cash reserve (watchlist
    risk.min_cash_reserve_pct, enforced independently in trade.py) — otherwise
    the agent is told to keep buying into orders trade.py will reject."""
    return {
        "preserve": float(os.getenv("DEPLOY_PRESERVE_PCT", "0.30")),
        "standard": float(os.getenv("DEPLOY_STANDARD_PCT", "0.40")),
        "active":   float(os.getenv("DEPLOY_ACTIVE_PCT",   "0.60")),
        "target":   float(os.getenv("TARGET_INVESTED_PCT", "0.70")),
        # Buys allowed per session in AGGRESSIVE mode. Was a literal "5" in
        # CLAUDE.md, which both profiles share — so one profile could not deploy
        # harder than the other without changing the rulebook for both.
        "max_buys": int(os.getenv("MAX_BUYS_PER_SESSION", "5")),
    }


def deploy_thresholds() -> dict:
    """Minimum composite score to buy in each deploy mode, env-overridable.

    Defaults reproduce the original hardcoded 0/1/2 ladder, so nothing changes
    unless a service sets these.

    Why these are tunable: a 5-year factor decomposition of our own universe
    (strategy_lens.py) regressed each deploy mode's entries against an
    equal-weighted hold of the same 198 names. At score ≥ 2 the dip sleeve had
    +1.2%/yr alpha (t 0.32 — indistinguishable from zero). At score ≥ 0, the
    AGGRESSIVE-mode gate, alpha was −2.0%/yr with beta 1.03 — i.e. that mode
    was buying the universe with extra steps and slightly worse selection.
    AGGRESSIVE is also the mode that deploys the most capital, so the weakest
    entry bar sat on the largest notional. Raising it to 1 is the cheapest
    correction; it is a variable rather than a constant so it can be reverted
    without a deploy and so paper and live can differ while it is validated."""
    return {
        "aggressive": int(os.getenv("DEPLOY_THRESH_AGGRESSIVE", "0")),
        "active":     int(os.getenv("DEPLOY_THRESH_ACTIVE",     "1")),
        "standard":   int(os.getenv("DEPLOY_THRESH_STANDARD",   "2")),
    }


def portfolio_status(cash_pct: float, signals: list, macro_brief: dict = None) -> dict:
    """
    Compute portfolio deployment status and identify dip-buying candidates.
    Deploy mode activates when cash exceeds the target buffer (see deploy_bands).
    """
    invested_pct   = round(1 - cash_pct, 4)
    _b = deploy_bands()
    _t = deploy_thresholds()
    target_invested = _b["target"]
    gap            = round(target_invested - invested_pct, 4)

    # Macro Kill-Switch Override
    kill_switch = bool(macro_brief and "MACRO KILL-SWITCH ACTIVE" in macro_brief.get("trading_guidance", ""))

    if kill_switch:
        mode = "preserve"
        note = f"⚠ MACRO KILL-SWITCH ACTIVE. Forcefully shifting to PRESERVE mode to protect capital."
    elif cash_pct > _b["active"]:
        mode = "aggressive"   # far above cash target — deploy urgently
        note = (f"Cash at {cash_pct*100:.0f}% — target {(1-target_invested)*100:.0f}%. "
                f"Buy dips at score ≥ {_t['aggressive']}.")
    elif cash_pct > _b["standard"]:
        mode = "active"       # above cash target — look for dips
        note = (f"Cash at {cash_pct*100:.0f}% — above {(1-target_invested)*100:.0f}% target. "
                f"Buy validated dips at score ≥ {_t['active']}.")
    elif cash_pct > _b["preserve"]:
        mode = "standard"     # near target — normal signal-based buying
        note = (f"Cash at {cash_pct*100:.0f}% — near {(1-target_invested)*100:.0f}% target. "
                f"Normal signal threshold applies (score ≥ {_t['standard']}).")
    else:
        mode = "preserve"     # at or below floor — stop deploying
        note = (f"Cash at {cash_pct*100:.0f}% — at/below the {_b['preserve']*100:.0f}% floor. "
                f"Preserve cash, no new buys.")

    # Breakout candidates: buying strength (fresh golden cross or score ≥3 + volume)
    breakout_candidates = [
        {
            "symbol":          s["symbol"],
            "score":           s.get("score", 0),
            "weighted":        s.get("weighted_score", 0),
            "is_dip":          s.get("is_dip", False),
            "fresh_cross":     s.get("fresh_golden_cross", False),
            "days_since_cross":s.get("days_since_cross"),
            "headroom":        s.get("allocation_headroom", 0),
            "buy_notional":    s.get("buy_notional", 0),
        }
        for s in signals
        if s.get("is_breakout") and s.get("allocation_headroom", 0) > 0
           and not s.get("position") and not s.get("earnings_blackout")
    ]
    breakout_candidates.sort(key=lambda x: (x["score"], x["weighted"]), reverse=True)

    # Deployment candidates: dip stocks with room to buy
    candidates = [
        {
            "symbol":       s["symbol"],
            "score":        s["score"],
            "weighted":     s.get("weighted_score", 0),
            "is_dip":       s.get("is_dip", False),
            "dip_depth":    s.get("dip_depth_pct", 0),
            "headroom":     s.get("allocation_headroom", 0),
            "buy_notional": s.get("buy_notional", 0),
        }
        for s in signals
        if s.get("is_dip") and s.get("allocation_headroom", 0) > 0
        and s.get("score", 0) >= -1 and not s.get("position")
        and not s.get("earnings_blackout")
    ]
    candidates.sort(key=lambda x: (x["dip_depth"], x["weighted"]), reverse=True)

    # Cross-sectional momentum sleeve (MOMENTUM_SLEEVE=true, default off).
    # Top-decile 12-1 names with room to buy. Kept as a SEPARATE list rather than
    # folded into the composite score on purpose: its value is that it is
    # uncorrelated (+0.11) with what the score already picks, and averaging it
    # into the score would destroy exactly that independence. Deliberately does
    # not require a dip or a breakout — this bet is neither.
    momentum_candidates = []
    if os.getenv("MOMENTUM_SLEEVE", "false").strip().lower() in ("1", "true", "yes", "on"):
        try:
            min_rank = float(os.getenv("MOMENTUM_MIN_RANK", "90"))
        except (TypeError, ValueError):
            min_rank = 90.0
        momentum_candidates = [
            {
                "symbol":       s["symbol"],
                "mom_12_1":     s.get("mom_12_1"),
                "mom_rank":     s.get("mom_rank"),
                "score":        s.get("score", 0),
                "weighted":     s.get("weighted_score", 0),
                "headroom":     s.get("allocation_headroom", 0),
                "buy_notional": s.get("buy_notional", 0),
            }
            for s in signals
            if s.get("mom_rank") is not None and s["mom_rank"] >= min_rank
            and s.get("allocation_headroom", 0) > 0
            and not s.get("position") and not s.get("overextended")
            and not s.get("earnings_blackout")
        ]
        momentum_candidates.sort(key=lambda x: x["mom_rank"], reverse=True)

    # Entry-type balance of the book we already hold. Advisory: it tells the
    # decision session which side of the dip/breakout blend is short, so the
    # instruction in CLAUDE.md is based on the actual book instead of a guess.
    try:
        import entry_mix
        held = [s["symbol"] for s in signals if s.get("position")]
        entry_mix.reconcile(held)
        mix = entry_mix.current(held)
    except Exception:
        mix = None

    return {
        "invested_pct":   invested_pct,
        "cash_pct":       round(cash_pct, 4),
        "target_invested": target_invested,
        "gap_to_fill":    max(gap, 0),
        "deploy_mode":         mode,
        "note":                note,
        # Surfaced so the decision prompt uses THIS session's live thresholds
        # rather than the illustrative numbers written into CLAUDE.md.
        "deploy_bands":        {"preserve_at_or_below": _b["preserve"],
                                "standard_below":       _b["standard"],
                                "active_below":         _b["active"],
                                "target_invested":      target_invested,
                                "max_buys_per_session": _b["max_buys"]},
        # Minimum score to buy in each mode, and the one that applies right now.
        # PRESERVE has no threshold — it buys nothing.
        "buy_thresholds":      _t,
        "buy_threshold":       _t.get(mode),
        "entry_mix":           mix,
        "dip_candidates":      candidates[:10],
        "breakout_candidates": breakout_candidates[:10],
        "momentum_candidates": momentum_candidates[:5],
    }


# ── News sentiment ───────────────────────────────────────────────────────────

def _fetch_news_polygon(sym: str, polygon_key: str, hours: int = 24) -> list:
    """Fetch recent headlines via Polygon news API."""
    try:
        from polygon import RESTClient
        client = RESTClient(api_key=polygon_key)
        since  = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        articles = client.list_ticker_news(
            ticker=sym, limit=5,
            published_utc__gte=since,
            sort="published_utc", order="desc",
        )
        return [
            {
                "title":     a.title,
                "published": a.published_utc[:16] if a.published_utc else "",
                "source":    a.publisher.name if a.publisher else "",
            }
            for a in articles
        ][:5]
    except Exception:
        return []


def _fetch_news_yfinance(sym: str) -> list:
    """Fetch recent headlines via yfinance (fallback)."""
    try:
        import yfinance as yf
        raw = yf.Ticker(sym).news or []
        headlines = []
        for n in raw[:5]:
            # Handle both yfinance news formats
            content = n.get("content", n)
            title   = content.get("title") or n.get("title", "")
            pub     = content.get("pubDate") or ""
            source  = (content.get("provider") or {}).get("displayName") or n.get("publisher", "")
            if title:
                headlines.append({"title": title, "published": str(pub)[:16], "source": source})
        return headlines
    except Exception:
        return []


def fetch_news_for_symbols(symbols: list, max_workers: int = 8) -> dict:
    """
    Parallel-fetch recent headlines for a list of symbols.
    Uses Polygon if POLYGON_API_KEY is set, otherwise yfinance.
    Returns {symbol: [{"title":..., "published":..., "source":...}]}
    """
    polygon_key = os.getenv("POLYGON_API_KEY", "").strip()

    def _fetch(sym):
        if polygon_key:
            items = _fetch_news_polygon(sym, polygon_key)
            if not items:          # Polygon returned nothing, try yfinance
                items = _fetch_news_yfinance(sym)
        else:
            items = _fetch_news_yfinance(sym)
        return sym, items

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, items = fut.result()
            results[sym] = items
    return results


# ── VIX regime ────────────────────────────────────────────────────────────────

def _vix_from_macro() -> "float | None":
    """Last-resort VIX: macro_context.py runs at 9:00 (before research at 9:45) and
    reliably writes VIX into signals/<date>_macro.json. Read it locally (same volume)."""
    try:
        path = f"signals/{datetime.now(ET).strftime('%Y-%m-%d')}_macro.json"
        if not os.path.exists(path):
            return None
        v = (json.load(open(path)).get("market_data", {}).get("vix", {}) or {}).get("price")
        return round(float(v), 2) if v else None
    except Exception:
        return None


def _fetch_vix_value() -> "tuple[float | None, str]":
    """Layered VIX fetch — yfinance is flaky, so don't rely on one call/endpoint.
    Returns (vix, source). Tries: download (×2) → fast_info → today's macro brief."""
    import time
    for attempt in range(2):
        try:
            d = yf.download("^VIX", period="5d", interval="1d", auto_adjust=True, progress=False)
            if not d.empty:
                return round(float(d["Close"].iloc[-1]), 2), "yfinance.download"
        except Exception:
            pass
        if attempt == 0:
            time.sleep(2)
    # fast_info: a different Yahoo endpoint that often works when download() rate-limits
    try:
        fi = yf.Ticker("^VIX").fast_info
        v = (fi.get("lastPrice") if hasattr(fi, "get") else None) or getattr(fi, "last_price", None)
        if v:
            return round(float(v), 2), "yfinance.fast_info"
    except Exception:
        pass
    v = _vix_from_macro()
    if v is not None:
        return v, "macro_brief"
    return None, "all sources failed"


def fetch_vix_regime() -> dict:
    """VIX regime with resilient multi-source fetch (VIX is an index, not Polygon-gated)."""
    vix, source = _fetch_vix_value()
    if vix is None:
        return {"vix": None, "regime": "unknown", "threshold_adjustment": 0, "note": "VIX fetch failed"}

    for threshold, regime, adjustment, note in VIX_REGIMES:
        if vix > threshold:
            return {"vix": vix, "regime": regime, "threshold_adjustment": adjustment,
                    "note": note, "source": source}
    return {"vix": vix, "regime": "complacency", "threshold_adjustment": 0,
            "note": "Market very calm", "source": source}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import accounts
    try:
        api_key, secret_key = accounts.get_keys("core")
    except Exception:
        print("ERROR: core account keys not set", file=sys.stderr)
        sys.exit(1)

    with open("watchlist.json") as f:
        wl = json.load(f)
    symbols = [s["symbol"] for s in wl["stocks"]]
    risk    = wl["risk"]

    # Split history, fetched once for the whole universe. Our bar source does not
    # always back-adjust, and an unadjusted split inside the momentum lookback
    # inverts the reading (KLAC read −75% when the truth was +144%). With this we
    # correct the series instead of discarding the name.
    try:
        import corporate_actions
        corp_actions = corporate_actions.fetch_all(symbols)
        _n_splits = sum(1 for v in corp_actions.values() if v.get("splits"))
        print(f"  Corporate actions: {_n_splits}/{len(symbols)} symbols have splits "
              f"in the lookback", file=sys.stderr)
    except Exception as e:
        corp_actions = {}
        print(f"  corporate actions skipped (non-fatal): {e}", file=sys.stderr)

    # ── Alpaca: account state, positions, live quotes (core account, live-aware) ─
    trading_client = accounts.trading_client("core")
    data_client    = StockHistoricalDataClient(api_key, secret_key)

    account         = trading_client.get_account()
    raw_equity      = float(account.portfolio_value)
    cash            = float(account.cash)

    positions_raw = trading_client.get_all_positions()
    _carved = accounts.carved_out()  # symbols fenced off from all strategy control
    # Net out carved-out holdings (e.g. the live account's legacy QQQ) so the
    # strategy sizes, sets deployment mode, and allocates against ONLY the
    # capital it actually controls — INVESTABLE equity, not total account value.
    carved_mv       = sum(float(p.market_value) for p in positions_raw
                          if p.symbol.upper() in _carved)
    portfolio_value = raw_equity - carved_mv
    cash_pct        = cash / portfolio_value if portfolio_value > 0 else 0
    positions = [
        {
            "symbol":            p.symbol,
            "qty":               float(p.qty),
            "avg_entry_price":   float(p.avg_entry_price),
            "market_value":      float(p.market_value),
            "unrealized_pl":     float(p.unrealized_pl),
            "unrealized_pl_pct": round(float(p.unrealized_plpc), 4),
            "current_price":     float(p.current_price),
            "peak_gain_pct":     round(max(float(p.unrealized_plpc), 0), 4),  # for trailing stops
        }
        for p in positions_raw
        if p.symbol.upper() not in _carved
    ]

    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=accounts.data_feed())
    quotes        = data_client.get_stock_latest_quote(quote_request)

    # ── Macro context (geopolitical / economic calendar brief) ───────────────
    macro_brief = None
    macro_path  = f"signals/{datetime.now(ET).strftime('%Y-%m-%d')}_macro.json"
    if os.path.exists(macro_path):
        with open(macro_path) as f:
            macro_data  = json.load(f)
            macro_brief = macro_data.get("brief")
        print(f"  Macro brief loaded: score={macro_brief.get('macro_score',0):+d} "
              f"risk={macro_brief.get('risk_level','?')}", file=sys.stderr)
    else:
        print("  No macro brief for today (run macro_context.py at 9 AM ET)", file=sys.stderr)

    # ── VIX regime ────────────────────────────────────────────────────────────
    print("Fetching VIX regime...", file=sys.stderr)
    market_regime = fetch_vix_regime()
    print(f"  VIX: {market_regime['vix']} — {market_regime['regime']}", file=sys.stderr)

    # ── Technical indicators ─────────────────────────────────────────────────
    dc = DataClient()
    # Built-in Polygon indicators only make sense on Starter (unlimited parallel calls).
    # Free tier: 6 calls/symbol vs 1 bar call → slower AND competes for quota.
    use_polygon_indicators = (dc.provider == "polygon:starter")

    if use_polygon_indicators:
        print(f"Fetching Polygon built-in indicators for {len(symbols)} symbols [{dc.provider}]...", file=sys.stderr)
        indicators_bulk = dc._provider.get_indicators_bulk(symbols)
        raw = pd.DataFrame()  # not needed when using Polygon indicators
    else:
        print(f"Downloading daily bars for {len(symbols)} symbols [{dc.provider}]...", file=sys.stderr)
        raw = dc.get_daily_bars(symbols, period="18mo")
        indicators_bulk = {}

    # ── Compute signals ───────────────────────────────────────────────────────
    signals = []
    for entry in wl["stocks"]:
        sym       = entry["symbol"]
        # Per-stock cap comes from the ONE shared rule so sizing here can never
        # disagree with trade.py's enforcement again — the scale knob, its
        # rationale (small live account, shared watchlist), and the guard-cap
        # ceiling are all documented on accounts.effective_max_allocation.
        import accounts as _accounts
        max_alloc = _accounts.effective_max_allocation(entry)

        if use_polygon_indicators:
            ind = indicators_bulk.get(sym, {})
            bars_df = ind.get("bars_df", pd.DataFrame())

            if not ind or ind.get("rsi") is None or ind.get("sma200") is None:
                print(f"WARNING: Incomplete Polygon indicators for {sym}", file=sys.stderr)
                continue

            rsi_val   = ind["rsi"]
            sma20     = ind.get("sma20") or 0.0
            sma50     = ind.get("sma50") or 0.0
            sma200    = ind["sma200"]
            macd_val  = ind.get("macd_val") or 0.0
            macd_sig  = ind.get("macd_sig") or 0.0
            close_val = ind.get("close") or 0.0

            if not bars_df.empty and len(bars_df) >= 20:
                close  = bars_df["Close"].dropna()
                high   = bars_df["High"].dropna()  if "High"   in bars_df.columns else close
                low    = bars_df["Low"].dropna()   if "Low"    in bars_df.columns else close
                volume = bars_df["Volume"].dropna() if "Volume" in bars_df.columns else pd.Series()
                bb_upper, bb_middle, bb_lower = compute_bollinger(close)
            else:
                bb_upper = bb_middle = bb_lower = close_val
                high = low = volume = pd.Series()

        else:
            sym_df = dc.sym_df(raw, sym)
            if sym_df.empty or len(sym_df) < 220:
                print(f"WARNING: Insufficient bar data for {sym} ({len(sym_df)} bars)", file=sys.stderr)
                continue

            close  = sym_df["Close"].dropna()
            high   = sym_df["High"].dropna()   if "High"   in sym_df.columns else close
            low    = sym_df["Low"].dropna()    if "Low"    in sym_df.columns else close
            volume = sym_df["Volume"].dropna() if "Volume" in sym_df.columns else pd.Series()
            close_val = float(close.iloc[-1])

            rsi_val            = compute_rsi(close)
            sma20              = compute_sma(close, 20)
            sma50              = compute_sma(close, 50)
            sma200             = compute_sma(close, 200)
            macd_val, macd_sig = compute_macd(close)
            bb_upper, bb_middle, bb_lower = compute_bollinger(close)
            atr_val = compute_atr(high, low, close)

        if sym in quotes and quotes[sym].ask_price:
            price = float(quotes[sym].ask_price)
            ask   = float(quotes[sym].ask_price)
            bid   = float(quotes[sym].bid_price) if quotes[sym].bid_price else price
        else:
            price = close_val if use_polygon_indicators else float(close.iloc[-1])
            ask   = price
            bid   = price

        vol_signal, rel_vol, vol_note = compute_volume_signal(
            bars_df["Volume"].dropna() if (use_polygon_indicators and not bars_df.empty and "Volume" in bars_df.columns)
            else volume
        )

        score, details = score_signals(
            price, rsi_val, sma50, sma200, macd_val, macd_sig, bb_upper, bb_middle, bb_lower,
            vol_signal, vol_note,
        )

        # Adjusted action considering VIX threshold
        adj_min_buy  = risk["min_buy_signal_score"]  + market_regime["threshold_adjustment"]
        adj_max_sell = risk["max_sell_signal_score"]
        action = action_from_score(score, adj_min_buy, adj_max_sell)

        pos           = next((p for p in positions if p["symbol"] == sym), None)
        current_alloc = float(pos["market_value"]) / portfolio_value if pos else 0.0
        headroom      = max_alloc - current_alloc

        # Breakout detection — buying strength on expanding volume
        bb_pct_b = details.get("bollinger", {}).get("pct_b", 0.5)
        overextended = details.get("bollinger", {}).get("overextended", False)
        is_breakout, fresh_golden_cross, days_since_cross = compute_breakout_signal(
            close, sma50, sma200, bb_pct_b, vol_signal, score, overextended
        )

        # Gap anomaly: >30% move vs prior close suggests an unadjusted corporate
        # action (split) — KLAC 2026-06-12 lesson. Alert and flag so the decision
        # session and resting GTC stops can be sanity-checked before sell logic.
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
        gap_alert = bool(prev_close and abs(price / prev_close - 1) > 0.30)
        if gap_alert:
            import alerts
            gap_pct = (price / prev_close - 1) * 100
            alerts.send(f"⚠️ {sym} gapped {gap_pct:+.0f}% vs prior close "
                        f"(${prev_close:,.2f} → ${price:,.2f}) — possible unadjusted "
                        f"split. Verify position data and cancel resting stops for "
                        f"{sym} before any sell logic runs.")

        # Donchian channel breakout — third breakout path: close above the prior
        # 20-day high with volume confirmation and a non-bearish composite
        donchian_breakout = compute_donchian_breakout(close, high)
        if (donchian_breakout and vol_signal == 1 and score >= 1
                and not overextended and bb_pct_b < 1.3):
            is_breakout = True

        # BB squeeze — was the breakout fired from a volatility compression?
        bb_width_percentile, squeeze = compute_bb_squeeze(close)

        # Microstructure: volume profile (POC/value area) + liquidity sweep
        from microstructure import compute_volume_profile, detect_liquidity_sweep
        vol_profile = compute_volume_profile(high, low, volume, price) \
            if len(high) >= 20 and len(volume) >= 20 else {"available": False}
        liq = detect_liquidity_sweep(high, low, close) \
            if len(high) >= 30 else {"sweep": None}

        # Microstructure score (−1..+1): condenses liquidity sweep + volume-profile
        # position into one number. Advisory by default; when MICRO_SCORE=true a
        # bearish reading (≤ −0.5) knocks the composite score down 1 and cancels a
        # breakout BEFORE the LLM sees it — so a +3 into a bearish sweep becomes +2.
        _micro = 0.0
        if liq.get("sweep") == "bullish":   _micro += 0.5
        elif liq.get("sweep") == "bearish": _micro -= 0.5
        _vp_pos = (vol_profile or {}).get("position")
        if _vp_pos == "below_value":   _micro += 0.3
        elif _vp_pos == "above_value": _micro -= 0.3
        if (vol_profile or {}).get("near_poc") or (vol_profile or {}).get("near_hvn"):
            _micro += 0.2
        microstructure_score = round(max(-1.0, min(1.0, _micro)), 2)
        micro_adjusted = False
        if os.getenv("MICRO_SCORE", "false").lower() == "true" and microstructure_score <= -0.5:
            score = max(-5, score - 1)
            is_breakout = False
            micro_adjusted = True

        # Multi-timeframe alignment — weekly trend vs daily bias
        mtf = compute_mtf_alignment(close, details.get("directional_bias", ""))

        # Support level detection — is price at a historically significant floor?
        support = compute_support_levels(close, high, low, price) \
                  if len(close) >= 50 and len(high) >= 50 and len(low) >= 50 \
                  else {"support_score": 0, "at_support": False, "detail": {}}

        # Dip detection: price below 20-day SMA, not overbought, and ideally at support.
        # Depth is volatility-normalized when DIP_ATR_MIN>0 (flag, default off): the
        # pullback must be ≥ DIP_ATR_MIN ATRs below SMA20 (Keltner-style), so a "dip"
        # means the same thing in a sleepy utility and a jumpy semi. Backtest (5y
        # watchlist): vs the raw gate, ~1.0–1.5 ATR improves Sharpe + drawdown, with
        # the benefit concentrated in choppy regimes. DIP_ATR_MIN=0 = legacy behavior.
        dip_atr = round((sma20 - price) / atr_val, 3) if atr_val else 0.0
        _dip_atr_min = float(os.getenv("DIP_ATR_MIN", "0") or 0)
        if _dip_atr_min > 0 and atr_val > 0:
            _depth_ok = (sma20 - price) >= _dip_atr_min * atr_val
        else:
            _depth_ok = price < sma20
        # ── Corporate-action guard ───────────────────────────────────────────
        # A split halves (or multiplies) the price while the moving averages
        # still hold pre-split values, so every price-vs-average indicator turns
        # extreme at once and manufactures a perfect-looking dip. KLAC on
        # 2026-06-12 (10:1 — price $241 vs SMA20 $1,817) and MNST on 2026-08-13
        # (2:1 — price $46.47 vs SMA20 $90.20, RSI 11.6, pct_b −0.21, score +2)
        # both produced phantom BUY signals this way, and MNST was bought
        # against an explicit directive because the setup looked compelling.
        #
        # The signature is unambiguous: no ordinary session moves price a third
        # of the way from its own 20-day average. Detect it in code so it cannot
        # be reasoned away session-to-session. If it IS a genuine one-day
        # collapse rather than a split, refusing to buy is still correct.
        _split_pct = float(os.getenv("SPLIT_GUARD_PCT", "0.35"))
        price_vs_sma20 = ((price - sma20) / sma20) if sma20 else 0.0
        split_suspect = bool(sma20) and abs(price_vs_sma20) >= _split_pct
        if split_suspect:
            print(f"  SPLIT GUARD: {sym} price ${price:,.2f} is {price_vs_sma20*100:+.1f}% "
                  f"vs SMA20 ${sma20:,.2f} — indicators unreliable, buys blocked",
                  file=sys.stderr)

        # A suspect symbol is never a dip or a breakout: those flags are computed
        # from the same contaminated averages that triggered the guard.
        is_dip       = bool(_depth_ok and rsi_val < 55 and score >= -1) and not split_suspect
        dip_depth    = round((sma20 - price) / sma20 * 100, 2) if is_dip else 0.0

        # High-quality dip: below SMA20 AND at a historical support level
        is_supported_dip = is_dip and support["at_support"]

        # Slow cross-sectional trend — the least correlated bet we have to what
        # we already trade. Ranked across the universe once all signals are built.
        mom = compute_momentum(close, (corp_actions.get(sym) or {}).get("splits"))

        signals.append({
            "symbol":              sym,
            "price":               round(price, 4),
            "ask":                 round(ask, 4),
            "bid":                 round(bid, 4),
            "score":               score,
            "weighted_score":      details.pop("_weighted_score"),
            "directional_bias":    details.pop("directional_bias"),
            "atr_val":             round(atr_val, 4),
            "atr_pct":             round(atr_val / price, 4),
            "action":              action,
            "sma20":               round(sma20, 2),
            # Hard buy-block: decide.py refuses any BUY on a suspect symbol, the
            # same way it refuses one below the dip_confidence floor.
            "split_suspect":       split_suspect,
            "price_vs_sma20_pct":  round(price_vs_sma20 * 100, 2),
            "is_dip":              is_dip,
            "is_supported_dip":    is_supported_dip,
            "dip_depth_pct":       dip_depth,
            "dip_atr":             dip_atr,   # advisory: pullback depth in ATRs below SMA20
            "support_score":       support["support_score"],
            "support_detail":      support["detail"],
            "is_breakout":         is_breakout,
            "fresh_golden_cross":  fresh_golden_cross,
            "days_since_cross":    days_since_cross,
            "gap_alert":           gap_alert,
            "volume_profile":      vol_profile,
            "liquidity_sweep":     liq.get("sweep"),
            "liquidity_sweep_detail": liq if liq.get("sweep") else None,
            "microstructure_score": microstructure_score,
            "micro_adjusted":      micro_adjusted,
            "donchian_breakout":   donchian_breakout,
            "bb_width_percentile": bb_width_percentile,
            "squeeze":             squeeze,
            "weekly_trend":        mtf["weekly_trend"],
            "mtf_aligned":         mtf["aligned"],
            "mom_12_1":            mom["mom_12_1"],   # 12-month return, skipping last month
            "mom_1m":              mom["mom_1m"],     # the skipped month, for context
            "mom_suspect":         mom.get("mom_suspect", False),  # unadjusted split in window
            "overextended":        overextended,
            "max_allocation":      max_alloc,
            "current_allocation":  round(current_alloc, 4),
            "allocation_headroom": round(max(headroom, 0), 4),
            "buy_notional":        round(portfolio_value * max(headroom, 0), 2),
            "signals":             details,
            "position":            pos,
        })

    # ── Cross-sectional momentum rank ────────────────────────────────────────
    # 12-1 momentum only means something relative to the rest of the universe —
    # +18% over a year is a top decile in one market and a bottom decile in
    # another. Rank it here, where the whole cross-section is in hand, into a
    # 0–100 percentile. Names without a full year of history get None rather
    # than a default, so a recent listing can't fake a rank.
    _ranked = sorted((s for s in signals if s.get("mom_12_1") is not None),
                     key=lambda s: s["mom_12_1"])
    _n = len(_ranked)
    for s in signals:
        s["mom_rank"] = None
    for i, s in enumerate(_ranked):
        s["mom_rank"] = round(100.0 * i / (_n - 1), 1) if _n > 1 else 50.0

    # ── ML dip and breakout confidence scoring ───────────────────────────────
    try:
        from dip_scorer import score_signals as ml_dip_score
        from breakout_scorer import score_signals as ml_breakout_score
        signals = ml_dip_score(signals)
        signals = ml_breakout_score(signals)
        scored_dip = sum(1 for s in signals if s.get("dip_confidence") is not None)
        scored_brk = sum(1 for s in signals if s.get("breakout_confidence") is not None)
        print(f"  ML confidence added: {scored_dip} dip, {scored_brk} breakout", file=sys.stderr)
    except Exception as e:
        print(f"  ML scoring unavailable: {e}", file=sys.stderr)

    # ── News sentiment: fetch headlines for actionable + held symbols ────────────
    news_targets = [
        s["symbol"] for s in signals
        if abs(s["score"]) >= 1 or s.get("position")
    ]
    if news_targets:
        print(f"Fetching news for {len(news_targets)} actionable symbols...", file=sys.stderr)
        news_map = fetch_news_for_symbols(news_targets)
        for sig in signals:
            headlines = news_map.get(sig["symbol"], [])
            if headlines:
                sig["news"] = headlines
    # ─────────────────────────────────────────────────────────────────────────

    # ── Orthogonal data enrichment ───────────────────────────────────────────
    # Everything above this line is a transform of OHLCV, and our own dip model
    # tops out near 0.50 AUC on that feature set — the price-derived well is dry.
    # These four sources carry information that is NOT in the price: a scheduled
    # event calendar, legally-disclosed insider decisions, dealer-priced forward
    # vol, and short positioning. Each is independently flag-gated and fail-open.
    #
    # Ordering matters: earnings must land BEFORE portfolio_status so a name in
    # its blackout window never reaches the dip/breakout/momentum candidate lists.
    try:
        import earnings_calendar
        # ETFs have no earnings — skip them rather than burn a round trip and
        # print a 404 for something entirely normal.
        _etfs = {s["symbol"].upper() for s in wl["stocks"]
                 if str(s.get("sector", "")).upper() == "ETF"}
        n = earnings_calendar.enrich(signals, skip=_etfs)
        blacked = earnings_calendar.blackout_symbols(signals)
        print(f"  Earnings calendar: {n} fetched live"
              + (f" | BLACKOUT: {sorted(blacked)}" if blacked else " | no blackouts"),
              file=sys.stderr)
    except Exception as e:
        print(f"  earnings enrich skipped (non-fatal): {e}", file=sys.stderr)

    try:
        import insider_flow
        if insider_flow.enabled():
            n = insider_flow.enrich(signals)
            clusters = [s["symbol"] for s in signals
                        if (s.get("insider") or {}).get("cluster_buy")]
            print(f"  Insider flow: {n} fetched"
                  + (f" | CLUSTER BUYS: {clusters}" if clusters else ""),
                  file=sys.stderr)
    except Exception as e:
        print(f"  insider enrich skipped (non-fatal): {e}", file=sys.stderr)

    try:
        import iv_metrics
        if iv_metrics.enabled():
            n = iv_metrics.enrich(signals)
            print(f"  IV metrics: priced {n} underlyings", file=sys.stderr)
    except Exception as e:
        print(f"  iv enrich skipped (non-fatal): {e}", file=sys.stderr)

    today  = datetime.now(ET).strftime("%Y-%m-%d")
    port_status = portfolio_status(cash_pct, signals, macro_brief)
    print(f"  Deploy mode: {port_status['deploy_mode']} | "
          f"Invested: {port_status['invested_pct']*100:.0f}% | "
          f"Dip candidates: {len(port_status['dip_candidates'])}", file=sys.stderr)

    # Optional fundamentals lens (flag: FUNDAMENTALS_LENS) — attach compact yfinance
    # fundamentals per symbol (cached daily, fail-open) for the analyst decomposition.
    try:
        import fundamentals
        if fundamentals.enabled():
            n = fundamentals.enrich(signals)
            print(f"  Fundamentals lens: enriched {len(signals)} signals ({n} fetched live)", file=sys.stderr)
    except Exception as e:
        print(f"  fundamentals enrich skipped (non-fatal): {e}", file=sys.stderr)

    output = {
        "date":             today,
        "generated_at":     datetime.now(ET).isoformat(),
        "market_regime":    market_regime,
        "macro_context":    macro_brief,   # None if macro_context.py hasn't run yet
        "portfolio_status": port_status,
        "account": {
            "portfolio_value": round(portfolio_value, 2),   # INVESTABLE (carved-out netted out)
            "cash":            round(cash, 2),
            "cash_pct":        round(cash_pct, 4),
            "buying_power":    round(float(account.buying_power), 2),
            "total_equity":    round(raw_equity, 2),         # real account value incl. carved-out
            "carved_out_value": round(carved_mv, 2),         # market value of fenced-off holdings
        },
        "positions":   positions,
        "signals":     signals,
        "risk_params": risk,
    }

    os.makedirs("signals", exist_ok=True)
    out_path = f"signals/{today}.json"
    # Atomic write: decide.py reads this file next — a truncated write (timeout/
    # OOM/redeploy mid-dump) would crash it. See io_utils.
    from io_utils import write_json_atomic
    write_json_atomic(out_path, output)

    print(json.dumps(output, indent=2))
    print(f"\nSignals saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
