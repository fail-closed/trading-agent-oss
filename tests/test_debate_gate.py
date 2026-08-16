"""Debate gate: mode resolution, shadow mode's harmlessness, cohort scoring.

Phase 0 of docs/TRADINGAGENTS_PLAN.md. The gate has been vetoing buys with no
counterfactual since it shipped; shadow mode exists to produce one. The property
that matters most here is negative: **shadow mode must never change a decision.**
If a future edit routes shadow through apply_to_decisions, the experiment stops
being an experiment and starts being an unreviewed trading change — so that is
the first thing tested.
"""
import json

import pytest

import debate
import memory_v2


# ── mode resolution ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("buys,shadow,invested,expected", [
    ("",          "",     None, "off"),
    ("",          "true", None, "shadow"),
    ("true",      "",     None, "live"),
    ("true",      "true", None, "live"),      # live wins — don't pay for both
    ("at:0.80",   "",     0.50, "off"),
    ("at:0.80",   "true", 0.50, "shadow"),    # below threshold → measure instead
    ("at:0.80",   "",     0.85, "live"),
    ("at:0.80",   "true", 0.85, "live"),
    ("at:0.80",   "true", None, "shadow"),    # invested unknown → gate can't arm
    ("at:garbage", "true", 0.99, "shadow"),   # unparseable → off, so shadow
])
def test_mode_resolution(monkeypatch, buys, shadow, invested, expected):
    monkeypatch.setenv("DEBATE_BUYS", buys)
    monkeypatch.setenv("DEBATE_SHADOW", shadow)
    assert debate.mode(invested) == expected


def test_mode_note_only_narrates_the_threshold_form(monkeypatch):
    """A plain on/off gate needs no explanation; `at:` does, because its state
    depends on the book and a bare "OFF" would read as a broken config."""
    monkeypatch.setenv("DEBATE_SHADOW", "")
    monkeypatch.setenv("DEBATE_BUYS", "true")
    assert debate.mode_note(0.9) == ""
    monkeypatch.setenv("DEBATE_BUYS", "at:0.80")
    note = debate.mode_note(0.5)
    assert "OFF" in note and "50.0%" in note


# ── shadow mode must be inert ────────────────────────────────────────────────

def _decisions():
    return [
        {"symbol": "AAA", "action": "BUY", "reason": "score +3", "trade_args": ["--notional", "500"]},
        {"symbol": "BBB", "action": "BUY", "reason": "score +2", "trade_args": ["--notional", "500"]},
        {"symbol": "CCC", "action": "HOLD", "reason": "no setup", "trade_args": None},
    ]


VERDICTS = {
    "AAA": {"symbol": "AAA", "verdict": "proceed", "confidence": 0.8, "bull": "b", "bear": "r"},
    "BBB": {"symbol": "BBB", "verdict": "skip", "confidence": 0.7, "bull": "b", "bear": "overbought"},
}


def test_stamp_verdicts_never_changes_an_action():
    d = _decisions()
    before = [(x["symbol"], x["action"], x["trade_args"]) for x in d]
    n = debate.stamp_verdicts(d, VERDICTS, applied=False)
    assert n == 2
    assert [(x["symbol"], x["action"], x["trade_args"]) for x in d] == before
    assert d[1]["debate_verdict"] == "skip" and d[1]["debate_applied"] is False
    assert "debate_downgraded" not in d[1]      # the marker only apply_ may set
    assert "debate_verdict" not in d[2]         # unvetted decision untouched


def test_apply_still_downgrades_and_marks():
    d = _decisions()
    debate.stamp_verdicts(d, VERDICTS, applied=True)
    assert debate.apply_to_decisions(d, VERDICTS) == 1
    assert d[0]["action"] == "BUY"                       # proceed survives
    assert d[1]["action"] == "SKIP" and d[1]["trade_args"] is None
    assert d[1]["debate_downgraded"] is True
    assert "DEBATE-SKIP" in d[1]["reason"]


def test_missing_verdict_is_fail_open():
    """A vetting error returns {} — every proposed buy must survive untouched."""
    d = _decisions()
    assert debate.stamp_verdicts(d, {}, applied=True) == 0
    assert debate.apply_to_decisions(d, {}) == 0
    assert [x["action"] for x in d] == ["BUY", "BUY", "HOLD"]


# ── cohort scoring ───────────────────────────────────────────────────────────

def _write_log(tmp_path, date, decisions, mode="shadow"):
    d = tmp_path / "decisions"
    d.mkdir(exist_ok=True)
    (d / f"{date}_1000.json").write_text(json.dumps(
        {"date": date, "time": "10:00", "debate_mode": mode, "decisions": decisions}))


def test_rows_are_read_only_inside_the_scoring_window(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dec = [{"symbol": "AAA", "action": "BUY", "entry_price": 100.0, "executed": True,
            "debate_verdict": "proceed", "debate_confidence": 0.8, "debate_applied": False}]
    _write_log(tmp_path, "2026-08-10", dec)       # 6 days before → in window
    _write_log(tmp_path, "2026-08-15", dec)       # 1 day before  → too fresh
    rows = memory_v2._debate_rows_from_logs("2026-08-16")
    assert [r["trade_date"] for r in rows] == ["2026-08-10"]
    assert rows[0]["entry"] == 100.0


def test_unvetted_decisions_are_not_rows(tmp_path, monkeypatch):
    """Only decisions the debate actually saw belong in the counterfactual."""
    monkeypatch.chdir(tmp_path)
    _write_log(tmp_path, "2026-08-10", [
        {"symbol": "AAA", "action": "BUY", "entry_price": 100.0, "executed": True,
         "debate_verdict": "proceed", "debate_applied": False},
        {"symbol": "BBB", "action": "HOLD", "entry_price": 50.0, "executed": False},
    ])
    assert [r["symbol"] for r in memory_v2._debate_rows_from_logs("2026-08-16")] == ["AAA"]


def test_vetoed_buys_are_scored_despite_never_executing(tmp_path, monkeypatch):
    """The whole point: under a LIVE gate the skip cohort has no fill, and if the
    scorer required `executed` the losing side of the experiment would be
    invisible — which is exactly the blind spot Phase 0 exists to close."""
    monkeypatch.chdir(tmp_path)
    _write_log(tmp_path, "2026-08-10", [
        {"symbol": "BBB", "action": "SKIP", "entry_price": 50.0, "executed": False,
         "debate_verdict": "skip", "debate_applied": True, "debate_downgraded": True},
    ], mode="live")
    rows = memory_v2._debate_rows_from_logs("2026-08-16")
    assert len(rows) == 1
    assert rows[0]["verdict"] == "skip" and rows[0]["executed"] is False
    assert rows[0]["applied"] is True


def _row(sym, date, verdict, excess, applied=False):
    return {"trade_date": date, "symbol": sym, "verdict": verdict,
            "excess_pct": excess, "applied": applied, "ret_pct": excess}


def test_report_computes_the_edge_in_the_right_direction():
    rows = [_row("A", "2026-08-10", "proceed", 3.0), _row("B", "2026-08-10", "proceed", 1.0),
            _row("C", "2026-08-10", "skip", -2.0), _row("D", "2026-08-10", "skip", -4.0)]
    out = memory_v2.debate_report(rows)
    assert "+5.00%/trade" in out                      # proceed +2.0 vs skip −3.0
    assert "gate adds value" in out
    assert "underpowered" in out                      # n=2 per cohort

    flipped = [_row("A", "2026-08-10", "proceed", -3.0), _row("C", "2026-08-10", "skip", 4.0)]
    assert "costing us trades" in memory_v2.debate_report(flipped)


def test_report_dedupes_intraday_re_reports():
    """10:00, 12:45 and 15:45 all re-list the same buy — counted once, or one
    trade wears three hats and carries the statistics (the bug dedupe_outcomes
    was written for)."""
    rows = [_row("A", "2026-08-10", "proceed", 5.0)] * 3 + [_row("C", "2026-08-10", "skip", 0.0)]
    out = memory_v2.debate_report(rows)
    assert "2 verdict(s)" in out and "n=1" in out


def test_report_is_honest_when_a_cohort_is_empty():
    out = memory_v2.debate_report([_row("A", "2026-08-10", "proceed", 1.0)])
    assert "n=0" in out and "before an edge can be computed" in out
    assert memory_v2.debate_report([]).startswith("Debate gate: no verdicts")


# ── the call site that composes them (decide.py) ─────────────────────────────
#
# The tests above prove debate.py's pieces behave. They do NOT prove decide.py
# wires them together correctly — and the wiring is where shadow mode could
# silently acquire teeth, because that is the only place apply_to_decisions() is
# reachable from. A QA review on 2026-08-16 found this seam untested.

def test_save_decisions_persists_the_verdict_fields(tmp_path, monkeypatch):
    """memory_v2 scores from the decision LOG, so a verdict that never reaches
    disk is a verdict that never gets measured — the gate would look untested
    for a second reason, with nothing failing."""
    import decide
    monkeypatch.chdir(tmp_path)
    signals = {"account": "paper", "signals": [{"symbol": "AAA", "price": 101.5}]}
    result = {
        "session_summary": "s", "memory_observation": "", "lesson": "",
        "debate_mode": "shadow",
        "decisions": [{
            "symbol": "AAA", "action": "BUY", "reason": "score +3",
            "debate_verdict": "skip", "debate_confidence": 0.7, "debate_applied": False,
        }],
    }
    decide.save_decisions("2026-08-16", "10:00", signals, result, 1, {"AAA": {"status": "submitted"}})
    log = json.load(open(tmp_path / "decisions" / "2026-08-16_1000.json"))
    assert log["debate_mode"] == "shadow"
    d = log["decisions"][0]
    assert d["debate_verdict"] == "skip"
    assert d["debate_confidence"] == 0.7
    assert d["debate_applied"] is False
    assert d["debate_downgraded"] is False
    # entry_price is the decision-time signal price, and it must be written for
    # every decision — under a LIVE gate a vetoed buy has no fill, so this is the
    # only basis on which the skip cohort can ever be scored.
    assert d["entry_price"] == 101.5


def test_save_decisions_records_a_vetoed_buy_with_a_price(tmp_path, monkeypatch):
    """A live-gate veto flips the action to SKIP and never trades. If that row
    lost its price, the losing side of the experiment would be unmeasurable —
    the exact blind spot Phase 0 exists to close."""
    import decide
    monkeypatch.chdir(tmp_path)
    signals = {"account": "paper", "signals": [{"symbol": "BBB", "price": 50.0}]}
    result = {
        "session_summary": "s", "memory_observation": "", "lesson": "",
        "debate_mode": "live",
        "decisions": [{
            "symbol": "BBB", "action": "SKIP", "reason": "DEBATE-SKIP: overbought",
            "trade_args": None, "debate_verdict": "skip", "debate_confidence": 0.7,
            "debate_applied": True, "debate_downgraded": True,
        }],
    }
    decide.save_decisions("2026-08-16", "10:00", signals, result, 0, {})
    d = json.load(open(tmp_path / "decisions" / "2026-08-16_1000.json"))["decisions"][0]
    assert d["executed"] is False and d["entry_price"] == 50.0
    assert d["debate_downgraded"] is True and d["debate_applied"] is True
    # …and memory_v2 must be able to pick it up from that log alone.
    assert [r["symbol"] for r in memory_v2._debate_rows_from_logs("2026-08-22")] == ["BBB"]


def test_unvetted_session_writes_null_verdicts_not_missing_keys(tmp_path, monkeypatch):
    """Gate off: the keys must still exist and be null. A KeyError-shaped absence
    would make the scorer's `.get()` and a real 'no verdict' indistinguishable."""
    import decide
    monkeypatch.chdir(tmp_path)
    signals = {"account": "paper", "signals": [{"symbol": "CCC", "price": 10.0}]}
    result = {"session_summary": "s", "memory_observation": "", "lesson": "",
              "debate_mode": "off",
              "decisions": [{"symbol": "CCC", "action": "HOLD", "reason": "no setup"}]}
    decide.save_decisions("2026-08-16", "10:00", signals, result, 0, {})
    d = json.load(open(tmp_path / "decisions" / "2026-08-16_1000.json"))["decisions"][0]
    assert d["debate_verdict"] is None and d["debate_applied"] is None
    assert memory_v2._debate_rows_from_logs("2026-08-22") == []
