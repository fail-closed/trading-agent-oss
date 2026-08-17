"""Ranked prompt budget — what the model is allowed to see.

Two failures this must make impossible, in priority order:

  A HELD POSITION MISSING FROM THE PROMPT CANNOT BE SOLD. That is strictly worse
  than a missed entry, so holdings bypass ranking and the budget entirely.

  SILENT TRUNCATION. `max_tokens=16000` over a measured 72 output tokens per
  decision is a ~221-decision ceiling. Exceeding it does not degrade the answer,
  it destroys it — `decide.py` raises rather than trade on half a reply.

And one that is about strategy rather than mechanics: an unquota'd top-N would
hand every slot to dips, which outnumber breakouts 10:1 while carrying the weaker
evidence (Sharpe 0.94 / t 0.32 against 1.29 / +6.6%/yr). The quotas exist so
selection cannot quietly abandon the better-evidenced setup.
"""
import pytest

import signal_rank as sr


def sig(sym, ws=0.0, score=0, dip=False, brk=False, mom=None, held=False, conf=0.5):
    s = {"symbol": sym, "weighted_score": ws, "score": score,
         "is_dip": dip, "is_breakout": brk, "dip_confidence": conf}
    if mom is not None:
        s["mom_rank"] = mom
    if held:
        s["position"] = {"qty": 1}
    return s


# ── holdings are untouchable ─────────────────────────────────────────────────

def test_every_held_position_survives_even_past_the_budget():
    """The most important assertion in this file. 200 holdings against a budget of
    10 must all come through — an omitted holding is an unsellable holding."""
    held = [sig(f"H{i}", held=True) for i in range(200)]
    kept, rep = sr.select(held, cap=10)
    assert len(kept) == 200
    assert {s["symbol"] for s in kept} == {s["symbol"] for s in held}
    assert rep["budget_exhausted"]


def test_holdings_are_kept_regardless_of_how_badly_they_rank():
    """A losing position ranks bottom by weighted_score. It is exactly the one you
    most need the model to see."""
    sigs = [sig("LOSER", ws=-9.0, held=True)] + [sig(f"G{i}", ws=5.0) for i in range(50)]
    kept, _ = sr.select(sigs, cap=5)
    assert "LOSER" in {s["symbol"] for s in kept}


def test_holdings_do_not_consume_archetype_quotas():
    held = [sig(f"H{i}", held=True, dip=True) for i in range(20)]
    dips = [sig(f"D{i}", dip=True, ws=1.0) for i in range(60)]
    kept, rep = sr.select(held + dips, cap=100)
    assert rep["held"] == 20
    assert rep["by_archetype"].get("dip", 0) == 60, "quota was eaten by holdings"


# ── the budget is hard ───────────────────────────────────────────────────────

def test_budget_is_never_exceeded_by_candidates():
    sigs = [sig(f"S{i}", ws=float(i), dip=True) for i in range(1000)]
    kept, rep = sr.select(sigs, cap=150)
    assert len(kept) == 150
    assert rep["dropped"] == 850


def test_a_thousand_name_universe_stays_inside_the_token_ceiling():
    """The whole point of the change: cost and truncation become flat in universe
    size. 221 decisions is the measured hard ceiling."""
    sigs = ([sig(f"B{i}", brk=True, ws=3.0) for i in range(50)] +
            [sig(f"D{i}", dip=True, ws=1.0) for i in range(700)] +
            [sig(f"O{i}", ws=0.5) for i in range(250)])
    kept, _ = sr.select(sigs)
    assert len(kept) <= 221, "would truncate the model's reply"
    assert len(kept) == sr.DEFAULT_BUDGET


def test_budget_is_configurable_and_floored(monkeypatch):
    monkeypatch.setenv("PROMPT_SIGNAL_BUDGET", "25")
    assert sr.budget() == 25
    monkeypatch.setenv("PROMPT_SIGNAL_BUDGET", "1")
    assert sr.budget() == 10, "a floor stops a typo emptying the prompt"
    monkeypatch.setenv("PROMPT_SIGNAL_BUDGET", "not-a-number")
    assert sr.budget() == sr.DEFAULT_BUDGET


# ── quotas: the strategy-preserving part ─────────────────────────────────────

def test_dips_cannot_crowd_out_breakouts():
    """700 dips against 5 breakouts, budget 60. Without quotas the dips take
    everything and the sleeve with the alpha gets nothing."""
    sigs = ([sig(f"D{i}", dip=True, ws=9.0) for i in range(700)] +   # dips score HIGHER
            [sig(f"B{i}", brk=True, ws=1.0) for i in range(5)])
    kept, rep = sr.select(sigs, cap=60)
    brks = [s for s in kept if s["symbol"].startswith("B")]
    assert len(brks) == 5, "every breakout must be kept — they are the scarce setup"
    assert rep["by_archetype"]["breakout"] == 5


def test_dip_intake_is_capped_even_when_dips_are_all_there_is():
    sigs = [sig(f"D{i}", dip=True, ws=float(i)) for i in range(500)]
    kept, rep = sr.select(sigs, cap=150)
    assert rep["by_archetype"]["dip"] == 60, "dip quota should cap intake"
    # the rest of the budget is spare capacity, not more dips-by-another-name
    assert rep["by_archetype"].get("spare", 0) == 90
    assert len(kept) == 150


def test_momentum_gets_reserved_slots():
    sigs = ([sig(f"D{i}", dip=True, ws=9.0) for i in range(200)] +
            [sig(f"M{i}", mom=95.0, ws=0.1) for i in range(30)])
    kept, rep = sr.select(sigs, cap=80)
    assert rep["by_archetype"]["momentum"] == 10


def test_unused_quota_is_not_wasted():
    """A session with no breakouts must not idle 40 slots."""
    sigs = [sig(f"D{i}", dip=True, ws=float(i)) for i in range(300)]
    kept, rep = sr.select(sigs, cap=100)
    assert rep["by_archetype"].get("breakout", 0) == 0
    assert len(kept) == 100, "leftover budget should be reallocated"


def test_a_breakout_is_never_charged_to_the_dip_quota():
    """Names are frequently both. Classification takes the scarcest label so the
    dip cap cannot consume breakout supply."""
    both = sig("BOTH", brk=True, dip=True, ws=1.0)
    assert sr.archetype(both) == "breakout"


def test_momentum_threshold_follows_its_own_knob(monkeypatch):
    monkeypatch.setenv("MOMENTUM_MIN_RANK", "80")
    assert sr.archetype(sig("X", mom=85.0)) == "momentum"
    monkeypatch.setenv("MOMENTUM_MIN_RANK", "95")
    assert sr.archetype(sig("X", mom=85.0)) == "other"


# ── ordering ─────────────────────────────────────────────────────────────────

def test_weighted_score_orders_ahead_of_integer_score():
    """score is a −5…+5 integer; at scale hundreds tie at +2 and the tie-break
    would decide everything. weighted_score must carry the ordering."""
    sigs = [sig("COARSE", ws=1.0, score=4, dip=True),
            sig("FINE", ws=4.5, score=2, dip=True)]
    kept, _ = sr.select(sigs, cap=1)
    assert kept[0]["symbol"] == "FINE"


def test_ties_break_deterministically():
    """Same result on every run, or the tests are theatre and the behaviour drifts."""
    a = [sig("BBB", ws=1.0, dip=True), sig("AAA", ws=1.0, dip=True)]
    first = [s["symbol"] for s in sr.select(list(a), cap=1)[0]]
    for _ in range(5):
        assert [s["symbol"] for s in sr.select(list(reversed(a)), cap=1)[0]] == first


def test_missing_fields_do_not_raise():
    """Signals arrive from JSON; absent keys are normal, not exceptional."""
    kept, _ = sr.select([{"symbol": "BARE"}, {"symbol": "ALSO"}], cap=5)
    assert len(kept) == 2


# ── reporting ────────────────────────────────────────────────────────────────

def test_report_states_what_was_dropped():
    """A funnel that truncates silently reads as full coverage — the exact
    failure mode this repo keeps logging."""
    sigs = [sig(f"D{i}", dip=True) for i in range(400)]
    _, rep = sr.select(sigs, cap=50)
    line = sr.summarise(rep)
    assert "50/400" in line
    assert "350 omitted" in line


def test_report_is_quiet_when_nothing_was_dropped():
    sigs = [sig("A", dip=True), sig("B", brk=True)]
    _, rep = sr.select(sigs, cap=50)
    assert "omitted" not in sr.summarise(rep)


def test_small_universe_is_unchanged_in_practice():
    """Today's 198-name universe trims to ~145 eligible — under the budget, so the
    ranker must be a no-op there. This change must not alter current behaviour."""
    sigs = [sig(f"S{i}", ws=float(i % 5), dip=(i % 3 == 0)) for i in range(145)]
    kept, rep = sr.select(sigs)
    assert len(kept) == 145 and rep["dropped"] == 0
