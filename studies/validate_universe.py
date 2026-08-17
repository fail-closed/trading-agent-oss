"""
validate_universe.py — does this universe's entries beat holding the universe?

    python -m studies.validate_universe --years 3
    python -m studies.validate_universe --years 3 --hold 20 --costs-bps 5

THE ONE QUESTION THIS ANSWERS
-----------------------------
Not "is the strategy profitable" — in a rising market almost anything is. The
question is whether *selecting* with these rules beats *equal-weighting the same
names over the same days*. That benchmark is the whole point: a signal that returns
+0.4% over five days in a week the universe returned +0.5% has negative alpha
while looking like a winner.

This is the discipline `memory_v2` applies to live trades, applied to history.

IT REUSES THE PRODUCTION SIGNAL FUNCTIONS ON PURPOSE
----------------------------------------------------
Every indicator here is imported from `research.py`. A backtest with its own
implementation measures a strategy that does not exist — and the two drift, always
in the flattering direction, because the backtest is the one being tuned. If the
numbers below are wrong, they are wrong in the same way production is wrong, which
is the only kind of wrong that is useful.

WHAT IT CANNOT TELL YOU
-----------------------
* It is **in-sample by construction**. These rules were developed on data that
  overlaps this window. A good result here is necessary, not sufficient.
* It ignores execution beyond a flat cost assumption: no partial fills, no gaps,
  no borrow, no market impact past the cost in basis points.
* It uses split-adjusted closes from the data vendor. Where that adjustment is
  wrong, this is wrong, and `corporate_actions` exists because it sometimes is.
* Survivorship: a universe built from *today's* tradable assets excludes
  everything that has since delisted, which flatters every long strategy. This
  harness does NOT correct for it and cannot — the delisted names are simply not
  in the asset list any more. Treat the numbers as an upper bound.

A t-statistic below ~2 means the result is indistinguishable from luck at this
sample size. The report says so in words, because a Sharpe ratio printed without
its t-stat is the most common way a backtest lies.
"""
import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADING_DAYS = 252


def load_universe(path=None):
    doc = json.loads((Path(path) if path else ROOT / "watchlist.json").read_text())
    return [s["symbol"] for s in doc.get("stocks", [])]


def fetch_history(symbols, years, batch=200):
    """{symbol: DataFrame(close, high, low, volume)} of split-adjusted daily bars."""
    import pandas as pd
    import accounts
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment

    k, s = accounts.data_keys()
    dc = StockHistoricalDataClient(k, s)
    start = datetime.now(timezone.utc) - timedelta(days=int(years * 365.25) + 400)
    out = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            data = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start,
                adjustment=Adjustment.ALL, feed=accounts.data_feed())).data
        except Exception as e:
            print(f"  bars batch {i//batch+1} failed ({str(e)[:60]})", file=sys.stderr)
            continue
        for sym, bars in data.items():
            if len(bars) < 260:                      # need a year before the first signal
                continue
            out[sym] = pd.DataFrame({
                "close": [float(b.close) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "volume": [float(b.volume) for b in bars],
            }, index=[b.timestamp.date() for b in bars])
        print(f"  loaded {min(i+batch, len(symbols))}/{len(symbols)}", file=sys.stderr)
    return out


def signals_on(df, i):
    """Every indicator for row `i`, using ONLY rows up to i — no lookahead.

    The slice is the whole safety argument: `df.iloc[:i+1]` cannot see the future,
    so a bug that leaks tomorrow's price would have to be inside research.py, where
    production would share it.
    """
    import research as R
    hist = df.iloc[:i + 1]
    close, high, low, vol = hist["close"], hist["high"], hist["low"], hist["volume"]
    if len(close) < 210:
        return None
    price = float(close.iloc[-1])
    sma20, sma50, sma200 = (R.compute_sma(close, 20), R.compute_sma(close, 50),
                            R.compute_sma(close, 200))
    if not (sma20 and sma50 and sma200):
        return None
    rsi = R.compute_rsi(close)
    macd_val, macd_sig = R.compute_macd(close)
    # Signatures taken from research.py, not assumed: compute_bollinger returns
    # (upper, middle, lower) and %B is derived here exactly as production does.
    bb_u, bb_m, bb_l = R.compute_bollinger(close)
    pct_b = ((price - bb_l) / (bb_u - bb_l)) if (bb_u - bb_l) else 0.5
    vol_sig, _vol_rel, vol_note = R.compute_volume_signal(vol)
    score, _details = R.score_signals(price, rsi, sma50, sma200, macd_val, macd_sig,
                                      bb_u, bb_m, bb_l, vol_sig, vol_note)
    overextended = pct_b > 1.3
    is_brk = bool(R.compute_breakout_signal(close, sma50, sma200, pct_b,
                                            vol_sig, score, overextended)[0])
    # Same dip definition production uses: below SMA20, RSI < 55, score >= -1.
    is_dip = bool(price < sma20 and rsi < 55 and score >= -1)
    return {"score": score, "is_dip": is_dip, "is_breakout": is_brk,
            "overextended": overextended, "price": price}


def forward_return(df, i, hold):
    if i + hold >= len(df):
        return None
    a, b = float(df["close"].iloc[i]), float(df["close"].iloc[i + hold])
    return (b / a - 1.0) if a else None


def run(hist, hold=5, min_score=2, costs_bps=5.0, stride=5):
    """Walk history; for every (symbol, day) record the archetype and its forward
    return, alongside the equal-weight universe return over the SAME days."""
    per_day = {}
    for sym, df in hist.items():
        for i in range(210, len(df) - hold, stride):
            d = df.index[i]
            per_day.setdefault(d, {"bench": [], "entries": []})
            fr = forward_return(df, i, hold)
            if fr is None:
                continue
            per_day[d]["bench"].append(fr)             # every name = the benchmark
            s = signals_on(df, i)
            if not s or s["overextended"]:
                continue
            if s["is_breakout"]:
                arch = "breakout"
            elif s["is_dip"] and s["score"] >= min_score:
                arch = "dip"
            elif s["score"] >= min_score:
                arch = "score_only"
            else:
                continue
            per_day[d]["entries"].append((arch, fr, sym))

    cost = costs_bps / 10_000 * 2                       # entry + exit
    excess = {}
    for d, rec in per_day.items():
        if not rec["bench"]:
            continue
        bench = statistics.fmean(rec["bench"])
        for arch, fr, sym in rec["entries"]:
            excess.setdefault(arch, []).append(fr - cost - bench)
    return excess, per_day


def stats(xs, hold):
    """n, mean excess per trade, annualised Sharpe of the excess, and a t-stat."""
    n = len(xs)
    if n < 2:
        return {"n": n}
    mean, sd = statistics.fmean(xs), statistics.stdev(xs)
    periods = TRADING_DAYS / hold
    return {
        "n": n,
        "mean_excess_pct": round(mean * 100, 3),
        "annual_excess_pct": round(mean * periods * 100, 2),
        "sharpe": round((mean / sd) * math.sqrt(periods), 2) if sd else 0.0,
        "t_stat": round(mean / (sd / math.sqrt(n)), 2) if sd else 0.0,
    }


def report(excess, per_day, hold, universe_n, years):
    lines = [
        "",
        f"  Universe {universe_n} names · {years}y · {hold}-day hold · "
        f"{len(per_day)} sample days",
        f"  Benchmark: equal-weight the SAME universe over the SAME days.",
        "",
        f"  {'entry':<12}{'n':>7}{'excess/trade':>14}{'annualised':>12}"
        f"{'Sharpe':>9}{'t':>7}  verdict",
        f"  {'-'*12}{'-'*7}{'-'*14}{'-'*12}{'-'*9}{'-'*7}  {'-'*22}",
    ]
    for arch in ("breakout", "dip", "score_only"):
        xs = excess.get(arch, [])
        st = stats(xs, hold)
        if st.get("n", 0) < 2:
            lines.append(f"  {arch:<12}{st.get('n', 0):>7}{'—':>14}"
                         f"{'—':>12}{'—':>9}{'—':>7}  too few entries")
            continue
        t = abs(st["t_stat"])
        verdict = ("significant" if t >= 2.0 else
                   "suggestive, not significant" if t >= 1.0 else
                   "indistinguishable from luck")
        if st["annual_excess_pct"] < 0:
            verdict = "NEGATIVE — " + verdict
        lines += [f"  {arch:<12}{st['n']:>7}{st['mean_excess_pct']:>13.3f}%"
                  f"{st['annual_excess_pct']:>11.2f}%{st['sharpe']:>9.2f}"
                  f"{st['t_stat']:>7.2f}  {verdict}"]
    lines += [
        "",
        "  A t-stat below ~2 means this sample cannot distinguish the result from",
        "  luck. Sharpe without n and t is the most common way a backtest lies.",
        "",
        "  IN-SAMPLE: these rules were developed on data overlapping this window.",
        "  A good result here is necessary, not sufficient. Survivorship also",
        "  flatters it — the universe excludes names that have since delisted.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--min-score", type=int, default=2)
    ap.add_argument("--costs-bps", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=5,
                    help="sample every Nth day; 1 is exhaustive and slow")
    ap.add_argument("--watchlist", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N symbols only (smoke test)")
    a = ap.parse_args(argv)

    syms = load_universe(a.watchlist)
    if a.limit:
        syms = syms[:a.limit]
    print(f"  universe: {len(syms)} symbols", file=sys.stderr)
    hist = fetch_history(syms, a.years)
    if not hist:
        print("  no history loaded — check credentials and connectivity", file=sys.stderr)
        return 1
    print(f"  usable histories: {len(hist)}/{len(syms)}", file=sys.stderr)
    excess, per_day = run(hist, a.hold, a.min_score, a.costs_bps, a.stride)
    print(report(excess, per_day, a.hold, len(hist), a.years))
    return 0


if __name__ == "__main__":
    sys.exit(main())
