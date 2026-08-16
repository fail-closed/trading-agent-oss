"""
strategies/catalog.py — registry of candidate daily strategies.

Each is a pure decide() over today's signals. To add a candidate: write a
decide function, register it in REGISTRY, map a paper account to it in
strategy_map.json, and strategy_runner.py runs it daily on that account —
compared head-to-head against the live swing baseline on the scoreboard.

Sizing convention: buys use ~PER_BUY of equity (capped by cash); the per-order
risk_guard + watchlist max_allocation still apply downstream.
"""
from strategies.base import TradeIntent

PER_BUY = 0.05          # 5% of equity per new position
MAX_POSITIONS = 12
VALUE_DIP_NEW_PER_DAY = 2   # value_dip: take the N best supported dips per day (relative rank)


def _held(positions):
    return set(positions or {})


def swing_lite(signals, equity, positions):
    """Plain composite-score swing: buy score≥2 (not overextended), sell score≤−2."""
    held = _held(positions)
    out = []
    for s in signals:
        sym, score = s["symbol"], s.get("score", 0)
        if score <= -2 and sym in held:
            out.append(TradeIntent(sym, "sell", qty="all", reason=f"score {score}"))
    if len(held) < MAX_POSITIONS:
        buys = [s for s in signals if s.get("score", 0) >= 2
                and not s.get("overextended") and s["symbol"] not in held]
        buys.sort(key=lambda s: s.get("weighted_score", 0), reverse=True)
        for s in buys[:MAX_POSITIONS - len(held)]:
            out.append(TradeIntent(s["symbol"], "buy", notional=round(equity * PER_BUY, 2),
                                   reason=f"score +{s['score']} (w {s.get('weighted_score')})"))
    return out


def value_dip(signals, equity, positions):
    """Conviction value: buy the day's best supported dips by ML confidence; exit on
    score≤−2. Uses RELATIVE ranking (top-N by dip_confidence), not an absolute gate:
    the dip-confidence model's output is compressed around ~0.5 (real AUC ≈0.50), so a
    fixed ≥0.60 cutoff was unreachable and value_dip never traded. Ranking the supported
    dips and taking the best few keeps the 'conviction value' thesis while staying
    calibrated to whatever the model actually emits (incl. after a retrain)."""
    held = _held(positions)
    out = []
    for s in signals:
        if s.get("score", 0) <= -2 and s["symbol"] in held:
            out.append(TradeIntent(s["symbol"], "sell", qty="all", reason="score ≤ −2"))
    room = MAX_POSITIONS - len(held)
    if room > 0:
        cands = [s for s in signals
                 if s.get("is_supported_dip") and s.get("score", 0) >= 1
                 and not s.get("overextended") and s["symbol"] not in held]
        # Best supported dips first (relative confidence rank), capped per day.
        cands.sort(key=lambda s: (s.get("dip_confidence") or 0), reverse=True)
        for s in cands[:min(room, VALUE_DIP_NEW_PER_DAY)]:
            out.append(TradeIntent(s["symbol"], "buy", notional=round(equity * PER_BUY, 2),
                                   reason=f"top supported dip (ml {s.get('dip_confidence') or 0:.2f})"))
    return out


def breakout_momentum(signals, equity, positions):
    """Momentum: buy confirmed breakouts (is_breakout), exit on score≤−2."""
    held = _held(positions)
    out = []
    for s in signals:
        if s.get("score", 0) <= -2 and s["symbol"] in held:
            out.append(TradeIntent(s["symbol"], "sell", qty="all", reason="score ≤ −2"))
    if len(held) < MAX_POSITIONS:
        cands = [s for s in signals if s.get("is_breakout") and not s.get("overextended")
                 and s["symbol"] not in held]
        cands.sort(key=lambda s: s.get("weighted_score", 0), reverse=True)
        for s in cands[:MAX_POSITIONS - len(held)]:
            out.append(TradeIntent(s["symbol"], "buy", notional=round(equity * PER_BUY, 2),
                                   reason="breakout"))
    return out


def xs_momentum(signals, equity, positions):
    """Cross-sectional 12-1 momentum: hold the universe's top-decile trend names.

    The one candidate here that is not a variation on the existing composite
    score. In the 5-year decomposition of our own universe (strategy_lens.py) a
    long/short 12-1 book correlated just +0.11 with what we actually trade — the
    most independent of the four primitive bets — and blending 25% of it into
    the current book lifted Sharpe 1.08 → 1.18 while cutting max drawdown from
    17.5% to 15.5%. Its standalone Sharpe (0.59) is mediocre; the low
    correlation is the whole reason to run it.

    Note what is deliberately absent: no dip condition, no breakout condition,
    no score floor beyond avoiding the deeply broken. Momentum is its own thesis
    and gating it on the others would just reproduce them. Exits on rank decay
    rather than score, for the same reason.
    """
    held = _held(positions)
    out  = []

    ranked = {s["symbol"]: s.get("mom_rank") for s in signals
              if s.get("mom_rank") is not None}

    # Exit when a name falls out of the top half — momentum's sell discipline is
    # relative decay, not an absolute score.
    for s in signals:
        sym = s["symbol"]
        if sym in held and ranked.get(sym) is not None and ranked[sym] < 50:
            out.append(TradeIntent(sym, "sell", qty="all",
                                   reason=f"momentum rank {ranked[sym]:.0f} < 50"))

    if len(held) < MAX_POSITIONS:
        cands = [s for s in signals
                 if (s.get("mom_rank") or 0) >= 90
                 and not s.get("overextended") and s["symbol"] not in held]
        cands.sort(key=lambda s: s.get("mom_rank", 0), reverse=True)
        for s in cands[:MAX_POSITIONS - len(held)]:
            out.append(TradeIntent(s["symbol"], "buy", notional=round(equity * PER_BUY, 2),
                                   reason=f"12-1 momentum rank {s.get('mom_rank'):.0f} "
                                          f"({s.get('mom_12_1')}%)"))
    return out


REGISTRY = {
    "swing_lite":        swing_lite,
    "value_dip":         value_dip,
    "breakout_momentum": breakout_momentum,
    "xs_momentum":       xs_momentum,
}
