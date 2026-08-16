"""
memory_v2.py — close the trade-learning loop (flag: MEMORY_V2).

Two additions to TRADING_MEMORY.md, both fail-open (never break the EOD journal):

  A. Outcome Scorecard — each EOD, score executed BUYs from ~5 trading days ago by
     their REALIZED return (entry → price now), accumulate to a durable
     signals/trade_outcomes.jsonl, and render a rolling "## Outcome Scorecard". This
     turns the memory from "what I did" into "what actually worked".
  B. Lessons Learned — durable rules the agent surfaces (decide.py `lesson` field)
     promoted into a persistent "## Lessons Learned" section that does NOT age out of
     the 15-session window.

Section writes are done by replacing/inserting whole "## " sections via regex, so a
partially-formatted file can't be silently corrupted by string-index math.
"""
import glob
import json
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OUTCOMES_PATH = os.path.join(os.getenv("LEDGER_DIR", "signals"), "trade_outcomes.jsonl")
# Debate-gate counterfactual (Phase 0, docs/TRADINGAGENTS_PLAN.md). A SEPARATE
# file from trade_outcomes.jsonl on purpose: these rows include buys that were
# never executed, and folding them into the Outcome Scorecard would tell the
# trading agent it made trades it did not make. This file feeds the human review,
# never TRADING_MEMORY.md.
DEBATE_OUTCOMES_PATH = os.path.join(os.getenv("LEDGER_DIR", "signals"), "debate_outcomes.jsonl")
SCORE_LO, SCORE_HI = 5, 10   # score a BUY 5–10 calendar days after entry (≈5 trading days)
KEEP = 60                    # scorecard window
BENCHMARK = os.getenv("BENCHMARK_SYMBOL", "SPY")   # what a trade must beat to count


def enabled() -> bool:
    return os.getenv("MEMORY_V2", "").strip().lower() in ("1", "true", "yes", "on")


# ── section editing (robust: operate on whole "## " sections) ─────────────────

def upsert_section(content: str, title: str, body: str) -> str:
    """Replace the `## {title}` section's body, or insert it after ## Performance."""
    section = f"## {title}\n{body.rstrip()}\n"
    pat = re.compile(rf"^## {re.escape(title)}\n.*?(?=^## |\Z)", re.DOTALL | re.MULTILINE)
    if pat.search(content):
        return pat.sub(section + "\n", content, count=1)
    perf = re.compile(r"^## Performance\n.*?(?=^## |\Z)", re.DOTALL | re.MULTILINE)
    m = perf.search(content)
    if m:
        return content[:m.end()] + section + "\n" + content[m.end():]
    return content.rstrip() + "\n\n" + section + "\n"


def add_lesson(content: str, lesson: str, cap: int = 20) -> str:
    """Prepend a new, non-duplicate lesson bullet to ## Lessons Learned (persistent)."""
    lesson = (lesson or "").strip().rstrip(".")
    if not lesson:
        return content
    pat = re.compile(r"^## Lessons Learned\n(.*?)(?=^## |\Z)", re.DOTALL | re.MULTILINE)
    m = pat.search(content)
    norm = re.sub(r"\s+", " ", lesson.lower())[:40]      # normalized prefix
    if m:
        existing = [l for l in m.group(1).splitlines() if l.strip().startswith("- ")]
        for l in existing:                                # dedup: prefix overlaps either way
            el = re.sub(r"\s+", " ", l.lower().lstrip("- ").strip())[:40]
            if el and (el in norm or norm in el):
                return content
        existing.insert(0, f"- {lesson}")
        body = "\n".join(existing[:cap])
        return content[:m.start(1)] + body + "\n\n" + content[m.end(1):]
    return upsert_section(content, "Lessons Learned", f"- {lesson}")


# ── A. outcome scoring ────────────────────────────────────────────────────────

def _latest_prices(symbols: list) -> dict:
    if not symbols:
        return {}
    try:
        import accounts
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        k, s = accounts.data_keys()
        dc = StockHistoricalDataClient(k, s)
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=accounts.data_feed()))
        out = {}
        for sym, quote in q.items():
            mid = ((quote.bid_price or 0) + (quote.ask_price or 0)) / 2
            out[sym] = mid or float(quote.ask_price or quote.bid_price or 0)
        return {s: p for s, p in out.items() if p}
    except Exception as e:
        print(f"  [memory_v2] price fetch failed: {str(e)[:80]}")
        return {}


def _bench_closes(start: str) -> dict:
    """{date: SPY close} from `start` onward — the benchmark leg of every outcome.

    A BUY is only informative relative to what the market did over the same days.
    Scoring `ret_pct > 0` as a win counts market drift as skill: in a window where
    SPY rose ~4%, most randomly-chosen stocks are up after five days, so a 100%
    'win rate' can represent zero or negative edge. Every outcome now carries the
    benchmark's return over its own holding period."""
    try:
        import accounts
        from datetime import date as _date, timedelta as _td
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        k, s = accounts.data_keys()
        d0 = _date.fromisoformat(start) - _td(days=7)      # pad for weekends/holidays
        bars = StockHistoricalDataClient(k, s).get_stock_bars(StockBarsRequest(
            symbol_or_symbols=BENCHMARK, timeframe=TimeFrame.Day,
            start=datetime(d0.year, d0.month, d0.day),
            feed=accounts.data_feed())).data.get(BENCHMARK, [])
        return {b.timestamp.date().isoformat(): float(b.close) for b in bars}
    except Exception as e:
        print(f"  [memory_v2] benchmark fetch failed: {str(e)[:80]}")
        return {}


def _bench_ret(closes: dict, spy_now: float, trade_date: str):
    """SPY % return from `trade_date` (first session on/after) to now, or None."""
    if not closes or not spy_now:
        return None
    for d in sorted(closes):                    # first session on/after the entry
        if d >= trade_date:
            base = closes[d]
            return ((spy_now - base) / base * 100) if base else None
    return None


def score_outcomes(today: str) -> list:
    """Score not-yet-scored executed BUYs aged 5–10 days; append to trade_outcomes.jsonl.
    Returns all outcome rows (for the scorecard)."""
    rows, existing, fills = [], set(), set()
    if os.path.exists(OUTCOMES_PATH):
        for line in open(OUTCOMES_PATH):
            line = line.strip()
            if line:
                try:
                    o = json.loads(line); rows.append(o); existing.add(o.get("key"))
                    # Track the underlying fill too, so a later session's re-report
                    # of the same BUY doesn't get scored again under a new key.
                    fills.add((o.get("trade_date"), o.get("symbol"), o.get("entry")))
                except Exception:
                    pass
    t0 = datetime.strptime(today, "%Y-%m-%d")
    pending = {}   # symbol -> [(key, date, time, entry)]
    for f in glob.glob("decisions/*.json"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})_", os.path.basename(f))
        if not m:
            continue
        d = m.group(1)
        age = (t0 - datetime.strptime(d, "%Y-%m-%d")).days
        if not (SCORE_LO <= age <= SCORE_HI):
            continue
        try:
            log = json.load(open(f))
        except Exception:
            continue
        for dec in log.get("decisions", []):
            if dec.get("action") == "BUY" and dec.get("executed") and dec.get("entry_price"):
                key = f"{d}|{log.get('time','')}|{dec['symbol']}"
                if key in existing:
                    continue
                pending.setdefault(dec["symbol"], []).append((key, d, log.get("time", ""), float(dec["entry_price"])))
    prices = _latest_prices(list(pending) + [BENCHMARK])
    spy_now = prices.get(BENCHMARK)
    earliest = min((d for items in pending.values() for _, d, _, _ in items), default=today)
    closes = _bench_closes(earliest) if pending else {}
    new = []
    for sym, items in pending.items():
        px = prices.get(sym)
        if not px:
            continue
        for key, d, t, entry in items:
            if entry <= 0:
                continue
            if (d, sym, round(entry, 2)) in fills:
                continue                      # same fill, re-reported by a later session
            fills.add((d, sym, round(entry, 2)))
            ret = (px - entry) / entry * 100
            bench = _bench_ret(closes, spy_now, d)
            row = {"key": key, "scored_on": today, "trade_date": d, "symbol": sym,
                   "entry": round(entry, 2), "price": round(px, 2),
                   "ret_pct": round(ret, 2)}
            if bench is not None:
                # excess = the only figure that separates skill from market drift
                row["bench_pct"] = round(bench, 2)
                row["excess_pct"] = round(ret - bench, 2)
            new.append(row)
    if new:
        os.makedirs(os.path.dirname(OUTCOMES_PATH), exist_ok=True)
        with open(OUTCOMES_PATH, "a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        rows += new
        print(f"  [memory_v2] scored {len(new)} trade outcome(s)")
    return rows


# ── A2. debate-gate counterfactual ───────────────────────────────────────────

def _debate_rows_from_logs(today: str) -> list:
    """Every vetted decision aged SCORE_LO..SCORE_HI days, with its verdict.

    Scores BOTH cohorts off the same basis — the decision-time signal price that
    `save_decisions` records for every decision, executed or not. That symmetry is
    what makes the comparison mean anything: a `skip` under a LIVE gate never
    traded, so there is no fill to compare against, and using fills for `proceed`
    while using signal prices for `skip` would measure slippage, not judgement."""
    out = []
    t0 = datetime.strptime(today, "%Y-%m-%d")
    for f in sorted(glob.glob("decisions/*.json")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})_", os.path.basename(f))
        if not m:
            continue
        d = m.group(1)
        if not (SCORE_LO <= (t0 - datetime.strptime(d, "%Y-%m-%d")).days <= SCORE_HI):
            continue
        try:
            log = json.load(open(f))
        except Exception:
            continue
        for dec in log.get("decisions", []):
            if not dec.get("debate_verdict") or not dec.get("entry_price"):
                continue
            out.append({
                "key": f"{d}|{log.get('time','')}|{dec['symbol']}",
                "trade_date": d, "time": log.get("time", ""), "symbol": dec["symbol"],
                "verdict": dec["debate_verdict"],
                "confidence": dec.get("debate_confidence"),
                "applied": bool(dec.get("debate_applied")),
                "executed": bool(dec.get("executed")),
                "entry": float(dec["entry_price"]),
            })
    return out


def score_debate_outcomes(today: str) -> list:
    """Score not-yet-scored debate verdicts aged 5–10 days. Returns all rows."""
    rows, existing = [], set()
    if os.path.exists(DEBATE_OUTCOMES_PATH):
        for line in open(DEBATE_OUTCOMES_PATH):
            line = line.strip()
            if line:
                try:
                    o = json.loads(line); rows.append(o); existing.add(o.get("key"))
                except Exception:
                    pass
    pending = [r for r in _debate_rows_from_logs(today) if r["key"] not in existing]
    if not pending:
        return rows
    prices = _latest_prices(sorted({r["symbol"] for r in pending}) + [BENCHMARK])
    spy_now = prices.get(BENCHMARK)
    closes = _bench_closes(min(r["trade_date"] for r in pending))
    new = []
    for r in pending:
        px = prices.get(r["symbol"])
        if not px or r["entry"] <= 0:
            continue
        ret = (px - r["entry"]) / r["entry"] * 100
        row = dict(r, scored_on=today, price=round(px, 2), ret_pct=round(ret, 2))
        bench = _bench_ret(closes, spy_now, r["trade_date"])
        if bench is not None:
            row["bench_pct"] = round(bench, 2)
            row["excess_pct"] = round(ret - bench, 2)
        new.append(row)
    if new:
        os.makedirs(os.path.dirname(DEBATE_OUTCOMES_PATH), exist_ok=True)
        with open(DEBATE_OUTCOMES_PATH, "a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        rows += new
        print(f"  [memory_v2] scored {len(new)} debate verdict(s)")
    return rows


def debate_report(rows: list) -> str:
    """Proceed-vs-skip cohort comparison, for the human gate review.

    Deliberately NOT written into TRADING_MEMORY.md: that file is the trading
    agent talking to itself (OPERATIONS §0, four text surfaces), and telling it
    that its own veto layer is under audit invites it to game the experiment."""
    seen, uniq = set(), []
    for r in rows:                                    # intraday sessions re-list
        k = (r.get("trade_date"), r.get("symbol"))    # the same buy; first wins
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    if not uniq:
        return "Debate gate: no verdicts scored yet."

    def stats(subset):
        ex = [r["excess_pct"] for r in subset if r.get("excess_pct") is not None]
        if not ex:
            return None
        return len(ex), sum(ex) / len(ex), sum(1 for e in ex if e > 0) / len(ex) * 100

    lines = [f"Debate gate counterfactual — {len(uniq)} verdict(s), "
             f"excess vs {BENCHMARK} over ~5 trading days"]
    for label, subset in (("proceed", [r for r in uniq if r["verdict"] == "proceed"]),
                          ("skip   ", [r for r in uniq if r["verdict"] == "skip"])):
        s = stats(subset)
        lines.append(f"  {label}: n={s[0]:<3} avg excess {s[1]:+.2f}%  beat-mkt {s[2]:.0f}%"
                     if s else f"  {label}: n=0")
    p, k = stats([r for r in uniq if r["verdict"] == "proceed"]), \
           stats([r for r in uniq if r["verdict"] == "skip"])
    if p and k:
        gap = p[1] - k[1]
        lines.append(f"  → gate edge: {gap:+.2f}%/trade "
                     f"({'skips underperformed — gate adds value' if gap > 0 else 'skips OUTPERFORMED — gate is costing us trades'})")
        if min(p[0], k[0]) < 10:
            lines.append(f"  ⚠ underpowered: smallest cohort n={min(p[0], k[0])}; "
                         f"treat as directional only until both clear ~20")
    else:
        lines.append("  → both cohorts need scored rows before an edge can be computed")
    shadow = sum(1 for r in uniq if not r["applied"])
    lines.append(f"  basis: decision-time signal price for both cohorts; "
                 f"{shadow}/{len(uniq)} row(s) from shadow sessions (gate inactive)")
    return "\n".join(lines)


def dedupe_outcomes(rows: list) -> list:
    """Collapse re-reports of the same fill to one outcome.

    The key is `date|session_time|symbol`, but each intraday session's decision log
    re-lists a BUY already executed earlier that day — so one WDC purchase on
    2026-06-17 produced four identical rows (10:00, 11:15, 14:15, 15:45, all entry
    713.04) and carried 4/6 of the entire scorecard. Statistics computed over that
    describe one trade wearing four hats.

    Dedupe on (date, symbol, entry) so a genuine second fill at a different price —
    e.g. the 60/40 ladder — still counts separately."""
    seen, out = set(), []
    for r in rows:
        k = (r.get("trade_date"), r.get("symbol"), r.get("entry"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def render_scorecard(rows: list, today: str) -> str:
    recent = dedupe_outcomes(rows)[-KEEP:]
    if not recent:
        return ("_Realized ~5-trading-day outcomes of executed BUYs (auto-scored). "
                "None scored yet — accrues as trades mature._")
    n = len(recent)
    wins = sum(1 for r in recent if r["ret_pct"] > 0)
    avg = sum(r["ret_pct"] for r in recent) / n
    best = max(recent, key=lambda r: r["ret_pct"])
    worst = min(recent, key=lambda r: r["ret_pct"])

    # Excess vs the benchmark leads, because it is the figure that means anything.
    # Raw return follows as context. Rows scored before benchmarking existed have
    # no excess_pct, so the excess stats are computed over whatever subset has it.
    ex = [r for r in recent if r.get("excess_pct") is not None]
    if ex:
        m = len(ex)
        ex_win = sum(1 for r in ex if r["excess_pct"] > 0)
        ex_avg = sum(r["excess_pct"] for r in ex) / m
        ex_best = max(ex, key=lambda r: r["excess_pct"])
        ex_worst = min(ex, key=lambda r: r["excess_pct"])
        verdict = ("beating" if ex_avg > 0 else "trailing")
        head = (f"_Realized ~5-trading-day outcomes of executed BUYs, measured as EXCESS "
                f"return vs {BENCHMARK} over the same days (auto-scored). Last {today}._\n"
                f"- **Excess vs {BENCHMARK}: {ex_avg:+.2f}% avg | beat-market rate {ex_win/m*100:.0f}% "
                f"({ex_win}/{m})** — {verdict} the market\n"
                f"- Best vs mkt: {ex_best['symbol']} {ex_best['excess_pct']:+.1f}% | "
                f"Worst: {ex_worst['symbol']} {ex_worst['excess_pct']:+.1f}%\n"
                f"- Raw (for context, inflated by market drift): {n} trades | "
                f"{wins/n*100:.0f}% up | {avg:+.2f}% avg")
        if m < n:
            head += f"\n- _{n - m} older trade(s) predate benchmarking and are excluded from the excess figures._"
        return head
    return (f"_Realized ~5-trading-day outcomes of executed BUYs (auto-scored). Last {today}._\n"
            f"- Trades scored: {n} | Win rate: {wins/n*100:.0f}% | Avg return: {avg:+.2f}%\n"
            f"- Best: {best['symbol']} {best['ret_pct']:+.1f}% | Worst: {worst['symbol']} {worst['ret_pct']:+.1f}%\n"
            f"- _Raw return only — benchmark unavailable; these numbers include market drift._")


def apply(today: str, content: str, lesson: str = "") -> str:
    """Add the Outcome Scorecard + any new Lesson to the memory content. Fail-open."""
    # Accrues BEFORE the MEMORY_V2 gate and returns nothing into `content`: the
    # debate ledger answers a developer's question (is the veto layer worth its
    # cost?), not the trading agent's, so it must neither depend on the memory
    # flag nor leak into the memory file.
    try:
        score_debate_outcomes(today)
    except Exception as e:
        print(f"  [memory_v2] debate scoring skipped: {str(e)[:100]}")
    if not enabled():
        return content
    try:
        rows = score_outcomes(today)
        content = upsert_section(content, "Outcome Scorecard", render_scorecard(rows, today))
    except Exception as e:
        print(f"  [memory_v2] scorecard skipped: {str(e)[:100]}")
    try:
        content = add_lesson(content, lesson)
    except Exception as e:
        print(f"  [memory_v2] lesson skipped: {str(e)[:100]}")
    return content


if __name__ == "__main__":
    # `python3 memory_v2.py --debate-report` — the Phase 0 read-out for the gate
    # review. Reads the accumulated ledger only; scores nothing, writes nothing.
    import sys
    if "--debate-report" in sys.argv:
        rows = []
        if os.path.exists(DEBATE_OUTCOMES_PATH):
            for line in open(DEBATE_OUTCOMES_PATH):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        print(debate_report(rows))
    else:
        print("usage: python3 memory_v2.py --debate-report")
