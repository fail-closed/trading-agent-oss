"""
archetype_gate.py — a buy needs a THESIS, not just a high score.

WHAT THIS BLOCKS
----------------
The composite score is a count of five agreeing indicators. Reaching +2 says the
indicators agree; it does not say what the trade IS. Every buying mode this system
documents is a specific claim about *why* the price should move:

    dip       — it fell to a level buyers have defended before
    breakout  — momentum is accelerating on expanding volume
    momentum  — slow 12-1 trend, deliberately independent of the other two

A buy that clears the score threshold while matching none of those has no thesis
behind it. It is, in the backtest's language, `score_only`. This gate refuses it.

THE EVIDENCE
------------
`studies/validate_universe.py` on a 987-name universe, 3 years, 5bp each way:

    entry          n       days   annualised   t·day
    breakout       5,820    277       +9.32%    2.02
    dip            5,854    272       -6.73%    0.57
    score_only    19,523    424       -9.22%   -1.42

`score_only` was **63% of all entries** and the worst bucket. Two honest caveats:

* **It is not statistically significant.** t·day -1.42 against a 2.4 bar. What
  argues for acting anyway is the *consistency*: it was negative in every breadth
  band — -0.425% on single-name days, -0.403% at 21-60 names, -0.164% above 61 —
  so the sign does not depend on which slice you look at.
* **The sample is in-sample and survivorship-flattered.** Both of which flatter it,
  and it was still the worst bucket.

The stronger argument is not the t-stat at all: two thirds of the capital this
system would deploy was going into the one entry type nobody designed. Closing
that is making the code match the documented strategy.

WHY IT IS A CODE GATE AND NOT A PROMPT LINE
-------------------------------------------
CLAUDE.md is edited in the same repo as the code and read by a model that can
reason its way around a sentence. The earnings blackout and the corporate-action
guard both started as prose and both were violated in live sessions before they
became code. A rule that only exists in the prompt is a preference.

FAIL-CLOSED, DELIBERATELY
-------------------------
No determinable archetype means no buy. That is the safe direction here — the cost
of a false block is a missed trade, which CLAUDE.md already ranks as cheaper than a
bad one. Note this differs from `entry_mix.classify`, which is fail-OPEN by design
and documents that it may never block a trade; that function is for *reporting* the
book's mix, so the two must not be conflated. This module is the gate; that one is
the tally.

SELLS ARE NEVER AFFECTED. Exiting is always permitted, as with every other gate.
"""
import os

BREAKOUT = "breakout"
DIP      = "dip"
MOMENTUM = "momentum"

# Order matters: a name can satisfy more than one, and the first match is the
# thesis we are acting on. Breakout wins over dip for the same reason
# entry_mix.classify prefers it — a Donchian break still under its SMA20 is being
# bought for the break, not the discount.
ARCHETYPES = (BREAKOUT, DIP, MOMENTUM)

DEFAULT_MIN_MOM_RANK = 90.0


def _truthy(name: str, default: str) -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """Default ON. This closed a path that was two thirds of deployed capital and
    measurably the weakest; shipping it default-off would mean the finding sits in
    a study nobody runs."""
    return _truthy("REQUIRE_ARCHETYPE", "true")


def momentum_sleeve_on() -> bool:
    """Momentum counts as a thesis only when its sleeve is actually running.

    Otherwise a high-`mom_rank` name that is neither dip nor breakout would slip
    through this gate as "momentum" while the sleeve that governs it — its 1-per-
    session cap, its ranking, its candidate list — is switched off. That would be
    an unmanaged momentum book arriving through the back door.
    """
    return _truthy("MOMENTUM_SLEEVE", "false")


def min_mom_rank() -> float:
    try:
        return float(os.getenv("MOMENTUM_MIN_RANK", str(DEFAULT_MIN_MOM_RANK)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_MOM_RANK


def archetype(signal):
    """The thesis behind buying this signal, or None if there isn't one.

    `is_supported_dip` needs no separate case: research.py derives it as
    `is_dip and at_support`, so it implies `is_dip`.
    """
    if not isinstance(signal, dict):
        return None
    if signal.get("is_breakout"):
        return BREAKOUT
    if signal.get("is_dip"):
        return DIP
    # `mom_rank` is already None whenever `mom_suspect` is true — research.py
    # withholds the figure rather than reporting a split-contaminated one. The
    # explicit check below is belt-and-braces: a sign-flipped momentum rank is
    # exactly the input that would put us in the wrong name with confidence.
    if momentum_sleeve_on() and not signal.get("mom_suspect"):
        mr = signal.get("mom_rank")
        if isinstance(mr, (int, float)) and mr >= min_mom_rank():
            return MOMENTUM
    return None


def check(signal):
    """(allowed, archetype, detail) for a proposed BUY.

    When the gate is disabled the archetype is still reported, so logs and the
    journal record what the trade was even where nothing is being blocked.
    """
    arch = archetype(signal)
    if arch:
        return True, arch, arch
    if not enabled():
        return True, None, "no archetype (gate disabled via REQUIRE_ARCHETYPE)"
    score = signal.get("score") if isinstance(signal, dict) else None
    where = f"score {score:+d}" if isinstance(score, int) else "score n/a"
    return False, None, (
        f"no archetype — not a dip, not a breakout"
        f"{'' if momentum_sleeve_on() else ' (momentum sleeve off)'}; "
        f"{where} alone is not a thesis")
