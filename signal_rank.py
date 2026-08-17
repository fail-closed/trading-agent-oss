"""
signal_rank.py — choose WHICH signals reach the model, under a hard budget.

WHY THIS EXISTS
---------------
`decide.py` used to send the model every symbol with `abs(score) >= 1`. Measured
across 15 real sessions that is **73% of the universe** — a sensible filter for 30
names and a broken one for 1,000, in two separate ways:

  COST scales linearly with the universe. Input ~210 tokens/symbol, output a
  measured 72 tokens per returned decision.

  TRUNCATION arrives first. `max_tokens=16000` divided by 72 is ~221 decisions,
  so a universe past roughly 300 names produces a reply that is cut off — and
  `decide.py` raises rather than trading on half a response. The wall is
  structural, not budgetary.

So selection moves from a threshold to a RANKED BUDGET: rank candidates, take the
best N, and let cost be flat in universe size instead of proportional.

WHY QUOTAS, NOT JUST A RANKING
------------------------------
A pure top-N by score hands every slot to whichever archetype happens to score
higher, and silently starves the other. That is not hypothetical here. Measured on
the same sessions:

    breakout   0.60 per session   Sharpe 1.29, alpha +6.6%/yr
    dip        6.40 per session   Sharpe 0.94, alpha +1.2%/yr (t 0.32)

Dips outnumber breakouts more than 10:1 while carrying the weaker evidence. The
documented target mix is 60% breakout — which at 3 buys a session needs 1.8
breakouts and has a supply of 0.6. The target is not being missed by choice; it is
arithmetically unreachable, and an unquota'd ranker would cement that.

So each archetype gets reserved slots, and dips are CAPPED rather than allowed to
fill the budget by weight of numbers.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not decide anything. It selects what the model is allowed to see, after
`decide.py`'s existing filters (sector cap, overextended block) have already run.
Every held position bypasses ranking entirely — a holding missing from the prompt
cannot be sold, which would be a far worse failure than a missed entry.

    import signal_rank
    kept, report = signal_rank.select(filtered_signals)
"""
import os

# Output budget. max_tokens=16000 / 72 measured tokens per decision ≈ 221 hard
# ceiling; 150 leaves headroom for the session summary, memory observation and a
# reply that runs longer than average. Raising this without raising max_tokens
# reintroduces truncation.
DEFAULT_BUDGET = 150

# Reserved slots per archetype, applied in this order. Breakout is first and
# generously reserved because it is the scarce, better-evidenced setup; dip is
# capped for the opposite reason.
QUOTAS = (
    ("breakout", 40),
    ("momentum", 10),
    ("dip", 60),
)


def budget() -> int:
    try:
        return max(10, int(os.getenv("PROMPT_SIGNAL_BUDGET", DEFAULT_BUDGET)))
    except ValueError:
        return DEFAULT_BUDGET


def archetype(s: dict) -> str:
    """One label per signal, in priority order.

    A signal can be several things at once — a name can be a breakout AND sit in
    the momentum decile. It is classified as the scarcest thing it qualifies as,
    so a breakout is never consumed out of the dip quota.
    """
    if s.get("is_breakout"):
        return "breakout"
    mr = s.get("mom_rank")
    if mr is not None and mr >= _min_mom_rank():
        return "momentum"
    if s.get("is_dip"):
        return "dip"
    return "other"


def _min_mom_rank() -> float:
    try:
        return float(os.getenv("MOMENTUM_MIN_RANK", "90"))
    except ValueError:
        return 90.0


def rank_key(s: dict):
    """Sort key, best first.

    `weighted_score` before `score`: score is a −5…+5 integer, so at 1,000 names
    hundreds tie at +2 and the tie-break decides everything. `weighted_score` is
    the empirically-derived finer measure and is what should carry the ordering.
    `dip_confidence` breaks remaining ties; symbol last so the result is stable
    and testable rather than dependent on dict order.
    """
    return (
        -float(s.get("weighted_score") or 0),
        -int(s.get("score") or 0),
        -float(s.get("dip_confidence") or 0),
        str(s.get("symbol") or ""),
    )


def select(signals: list, cap: int = None) -> tuple:
    """(kept, report). Pure — no I/O, no env beyond the knobs above.

    Order of operations matters and is asserted by tests:
      1. every held position is kept, unranked and uncapped
      2. remaining slots are filled per-archetype by quota, best-ranked first
      3. any leftover slots go to the best of whatever remains
    """
    cap = budget() if cap is None else cap
    held = [s for s in signals if s.get("position")]
    rest = [s for s in signals if not s.get("position")]

    kept = list(held)
    report = {"universe": len(signals), "held": len(held), "budget": cap,
              "by_archetype": {}, "dropped": 0, "budget_exhausted": False}

    room = cap - len(kept)
    if room <= 0:
        # More holdings than the budget. Keep them all anyway — see the module
        # docstring: an unsellable position is worse than an oversized prompt.
        report["budget_exhausted"] = True
        report["dropped"] = len(rest)
        return kept, report

    buckets = {}
    for s in rest:
        buckets.setdefault(archetype(s), []).append(s)
    for v in buckets.values():
        v.sort(key=rank_key)

    taken = set()
    for name, quota in QUOTAS:
        if room <= 0:
            break
        picked = buckets.get(name, [])[:min(quota, room)]
        for s in picked:
            taken.add(id(s))
        kept.extend(picked)
        room -= len(picked)
        report["by_archetype"][name] = len(picked)

    # Leftover budget → best remaining regardless of archetype, so a session with
    # no breakouts does not waste 40 slots.
    if room > 0:
        spare = sorted((s for s in rest if id(s) not in taken), key=rank_key)[:room]
        kept.extend(spare)
        report["by_archetype"]["spare"] = len(spare)
        room -= len(spare)

    report["dropped"] = len(signals) - len(kept)
    report["budget_exhausted"] = report["dropped"] > 0
    return kept, report


def summarise(report: dict) -> str:
    """One line for the session log. Silent truncation is how a funnel starts
    lying about coverage, so what was dropped is always stated."""
    by = ", ".join(f"{k} {v}" for k, v in report.get("by_archetype", {}).items() if v)
    line = (f"Prompt budget {report['budget']}: kept {report['universe'] - report['dropped']}"
            f"/{report['universe']} ({report['held']} held" + (f", {by}" if by else "") + ")")
    if report.get("dropped"):
        line += f" — {report['dropped']} omitted by rank"
    return line
