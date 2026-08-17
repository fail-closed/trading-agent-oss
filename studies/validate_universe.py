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

HOW SIGNIFICANCE IS JUDGED, AND WHY NOT THE OBVIOUS WAY
-------------------------------------------------------
Two corrections separate this from the naive version, and at a 1,000-name
universe they are the difference between a headline and a non-result:

* **Clustering by date.** Trades opened on the same day are not independent —
  they share that day's sector and style moves. A t-stat over pooled trades
  treats 8 correlated bets as 8 independent draws. The report's verdict uses
  `t·day`: excess averaged within each day, tested across days. See
  `daily_means`.
* **Three archetypes, one dataset.** Testing breakout, dip and score_only and
  then reporting the best is three coin flips. The bar for "significant" is
  raised above 2.0 accordingly.

A Sharpe ratio printed without its t-stat is the most common way a backtest
lies; a t-stat computed over correlated observations is the second.
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


def daily_means(per_day, cost):
    """Excess averaged WITHIN each day, per archetype: {arch: [day_mean, ...]}.

    THIS IS THE CORRECTION THAT MATTERS FOR SIGNIFICANCE. Trades opened on the
    same day are not independent observations — they share that day's market,
    sector and factor moves. Subtracting the equal-weight benchmark removes the
    common market component, but sector and style co-movement survive it. A
    t-statistic computed over pooled trades therefore treats a few hundred
    correlated bets as a few hundred independent draws, and overstates
    significance by up to sqrt(trades per day).

    Averaging within the day and testing across days is the standard
    cluster-by-date fix: the sample size becomes the number of days, which is the
    number of genuinely independent draws the walk produced. It is the
    conservative choice — perfectly correlated same-day trades would make it
    exactly right, and uncorrelated ones would make it merely pessimistic.
    """
    out = {}
    for _d, rec in per_day.items():
        if not rec.get("bench"):
            continue
        bench = statistics.fmean(rec["bench"])
        buckets = {}
        for arch, fr, _sym in rec.get("entries", []):
            buckets.setdefault(arch, []).append(fr - cost - bench)
        for arch, xs in buckets.items():
            out.setdefault(arch, []).append(statistics.fmean(xs))
    return out


def _corr(xs, ys):
    """Pearson r, or None when the sample cannot support one."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        return round(statistics.correlation(xs, ys), 3)
    except (statistics.StatisticsError, ValueError):
        return None


# A day on which at least this many names fired is treated as "wide". Above it the
# day's own mean is an average of many names, so it is a far quieter estimate than
# a day carrying a single trade.
WIDE_BREADTH = 21


def breadth(per_day, cost):
    """Per archetype: how many names fired each day, and how that skew moves the mean.

    WHY THIS IS WORTH A SECTION OF ITS OWN. The pooled mean weights each day by how
    many trades it produced; the day-clustered mean weights every day equally.
    Those answer different questions:

        trade-weighted — "what did the average TRADE earn?"  (fixed size per NAME)
        day-weighted   — "what did the average DAY earn?"     (fixed size per SESSION)

    Neither is wrong. This system caps buys per session, so the day-weighted figure
    is the closer analogue of what it would have earned — but it has a defect worth
    naming, and the 1,000-name run is what exposed it.

    THE DEFECT: breadth is wildly skewed. On the real run a third of dip days
    carried exactly ONE trade, while the widest carried 288. Day-weighting hands
    those 89 singleton days a third of the total weight, and a singleton day's
    "mean" is one trade with ~8% standard deviation. So the day-weighted mean can
    flip sign purely on singleton noise, with no relationship whatsoever between
    breadth and outcome.

    WHAT THIS DELIBERATELY DOES NOT CLAIM: an earlier version of this function
    asserted that a negative breadth/excess correlation meant the archetype "fires
    widest on its worst days". On the real data that correlation was -0.039
    (Spearman -0.021) — no relationship — while the sign still flipped. The
    correlation is retained below for completeness but is near-useless on a count
    variable this skewed, and nothing here relies on it. The singleton share and
    the wide-day mean are what actually explain the divergence.
    """
    counts, means = {}, {}
    for _d, rec in per_day.items():
        if not rec.get("bench"):
            continue
        bench = statistics.fmean(rec["bench"])
        buckets = {}
        for arch, fr, _sym in rec.get("entries", []):
            buckets.setdefault(arch, []).append(fr - cost - bench)
        for arch, xs in buckets.items():
            counts.setdefault(arch, []).append(len(xs))
            means.setdefault(arch, []).append(statistics.fmean(xs))

    out = {}
    for arch, ms in means.items():
        cs = counts[arch]
        total = sum(cs)
        singles = [m for c, m in zip(cs, ms) if c == 1]
        wide = [(c, m) for c, m in zip(cs, ms) if c >= WIDE_BREADTH]
        wide_trades = sum(c for c, _ in wide)
        out[arch] = {
            "days": len(ms),
            "median_breadth": round(statistics.median(cs), 1),
            "max_breadth": max(cs),
            "corr_breadth_excess": _corr(cs, ms),   # see docstring: not load-bearing
            # trade-weighted = pooled: each day counts in proportion to its trades
            "trade_weighted_pct": round(
                sum(c * m for c, m in zip(cs, ms)) / total * 100, 3) if total else 0.0,
            "day_weighted_pct": round(statistics.fmean(ms) * 100, 3),
            # the singleton problem, quantified
            "singleton_days": len(singles),
            "singleton_day_share": round(len(singles) / len(ms), 3),
            "singleton_mean_pct": round(statistics.fmean(singles) * 100, 3) if singles else None,
            "singleton_sd_pct": (round(statistics.stdev(singles) * 100, 2)
                                 if len(singles) > 1 else None),
            # the quiet estimate: days broad enough that the day-mean is itself an average
            "wide_days": len(wide),
            "wide_trades": wide_trades,
            "wide_mean_pct": (round(sum(c * m for c, m in wide) / wide_trades * 100, 3)
                              if wide_trades else None),
        }
    return out


def breadth_note(per_day, cost):
    """Lines for the report — emitted ONLY when a weighting changes the sign, since
    that is the case a reader must not miss.

    It reports the singleton share and the wide-day mean rather than a correlation,
    because on the real run the correlation was flat while the sign flipped.
    """
    b = breadth(per_day, cost)
    flipped = [a for a, v in b.items()
               if v["trade_weighted_pct"] * v["day_weighted_pct"] < 0]
    if not flipped:
        return []
    lines = ["", "  BREADTH WARNING — weighting changes the SIGN for: "
                 + ", ".join(sorted(flipped)), ""]
    for arch in sorted(flipped):
        v = b[arch]
        lines += [
            f"    {arch:<12} per-trade {v['trade_weighted_pct']:+.3f}%   "
            f"per-day {v['day_weighted_pct']:+.3f}%   "
            f"breadth median {v['median_breadth']:.0f} max {v['max_breadth']}",
            f"    {'':<12} single-trade days: {v['singleton_days']} of {v['days']} "
            f"({v['singleton_day_share']*100:.0f}% of the day-weight), "
            f"mean {v['singleton_mean_pct']:+.3f}% sd {v['singleton_sd_pct']}%",
            f"    {'':<12} days with >={WIDE_BREADTH} names "
            f"({v['wide_days']} days, {v['wide_trades']} trades): "
            f"{v['wide_mean_pct']:+.3f}%  <- the quiet estimate",
        ]
    lines += [
        "",
        "  Per-trade assumes fixed size per NAME; per-day assumes fixed size per",
        "  SESSION. This system caps buys per session, so per-day is the closer",
        "  analogue — but breadth is skewed enough that single-trade days can carry",
        "  a third of the day-weight at ~8% standard deviation each, which is enough",
        "  to flip the sign on noise alone. Read the wide-day figure as the estimate",
        "  with the least sampling noise in it, and treat the divergence as a reason",
        "  to trust NEITHER headline rather than to pick the flattering one.",
    ]
    return lines


def _t(xs):
    """Two-sided t against zero, or None when the sample cannot support one."""
    if len(xs) < 2:
        return None
    mean, sd = statistics.fmean(xs), statistics.stdev(xs)
    return round(mean / (sd / math.sqrt(len(xs))), 2) if sd else 0.0


def stats(xs, hold):
    """n, mean excess per trade, annualised Sharpe of the excess, and a t-stat.

    `t_stat` here is the POOLED, per-trade figure. It is retained because it is
    what most backtests print, but `daily_means` explains why the verdict is not
    based on it.
    """
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


def report(excess, per_day, hold, universe_n, years, costs_bps=5.0, archetypes=3):
    daily = daily_means(per_day, costs_bps / 10_000 * 2)
    lines = [
        "",
        f"  Universe {universe_n} names · {years}y · {hold}-day hold · "
        f"{len(per_day)} sample days",
        f"  Benchmark: equal-weight the SAME universe over the SAME days.",
        "",
        f"  {'entry':<12}{'n':>7}{'days':>6}{'excess/trade':>14}{'annualised':>12}"
        f"{'Sharpe':>9}{'t·pool':>8}{'t·day':>7}  verdict",
        f"  {'-'*12}{'-'*7}{'-'*6}{'-'*14}{'-'*12}{'-'*9}{'-'*8}{'-'*7}  {'-'*30}",
    ]
    # Three archetypes tested on one dataset: the threshold for "significant"
    # rises accordingly, or we would be reporting the best of three coin flips.
    bar = 2.0 + (0.4 if archetypes >= 3 else 0.0)
    for arch in ("breakout", "dip", "score_only"):
        xs = excess.get(arch, [])
        st = stats(xs, hold)
        if st.get("n", 0) < 2:
            lines.append(f"  {arch:<12}{st.get('n', 0):>7}{'—':>6}{'—':>14}"
                         f"{'—':>12}{'—':>9}{'—':>8}{'—':>7}  too few entries")
            continue
        dxs = daily.get(arch, [])
        td = _t(dxs)
        # The verdict uses the day-clustered t when there is one. Falling back to
        # the pooled figure is flagged, because it is the optimistic one.
        t = abs(td) if td is not None else abs(st["t_stat"])
        verdict = ("significant" if t >= bar else
                   "suggestive, not significant" if t >= 1.0 else
                   "indistinguishable from luck")
        if st["annual_excess_pct"] < 0:
            verdict = "NEGATIVE — " + verdict
        if td is None:
            verdict += " (pooled t — no day clusters)"
        lines += [f"  {arch:<12}{st['n']:>7}{len(dxs):>6}"
                  f"{st['mean_excess_pct']:>13.3f}%"
                  f"{st['annual_excess_pct']:>11.2f}%{st['sharpe']:>9.2f}"
                  f"{st['t_stat']:>8.2f}"
                  f"{(f'{td:.2f}' if td is not None else '—'):>7}  {verdict}"]
    lines += breadth_note(per_day, costs_bps / 10_000 * 2)
    lines += [
        "",
        f"  VERDICTS USE t·day AND A BAR OF {bar}, NOT t·pool AND 2.0.",
        "  t·pool treats every trade as an independent observation. Trades opened on",
        "  the same day share that day's sector and style moves, so t·pool is",
        "  inflated by up to sqrt(trades per day) — here that is the difference",
        "  between a headline and a non-result. t·day averages within each day and",
        "  tests across days, so n becomes the number of independent draws. The bar",
        "  is raised above 2.0 because three archetypes were tested on one dataset.",
        "",
        "  Sharpe and excess/trade are per-trade figures and are NOT corrected for",
        "  this. They describe the average trade, not the confidence in it.",
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
    # A 1,000-name walk is ~190k indicator computations and takes minutes. Asking a
    # second question of the same walk should not mean paying for it twice — and
    # re-running to answer a follow-up is how people end up reporting two results
    # computed from subtly different data.
    ap.add_argument("--dump", default=None, metavar="PATH",
                    help="pickle (excess, per_day) after the walk for later analysis")
    ap.add_argument("--load", default=None, metavar="PATH",
                    help="skip fetching and walking; re-report a previous --dump")
    a = ap.parse_args(argv)

    if a.load:
        import pickle
        with open(a.load, "rb") as fh:
            excess, per_day, meta = pickle.load(fh)
        print(f"  loaded walk of {meta['universe_n']} names from {a.load}",
              file=sys.stderr)
        print(report(excess, per_day, meta["hold"], meta["universe_n"],
                     meta["years"], meta["costs_bps"]))
        return 0

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
    if a.dump:
        import pickle
        meta = {"hold": a.hold, "years": a.years, "costs_bps": a.costs_bps,
                "universe_n": len(hist), "stride": a.stride,
                "min_score": a.min_score}
        with open(a.dump, "wb") as fh:
            pickle.dump((excess, per_day, meta), fh)
        print(f"  dumped walk to {a.dump}", file=sys.stderr)
    print(report(excess, per_day, a.hold, len(hist), a.years, a.costs_bps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
