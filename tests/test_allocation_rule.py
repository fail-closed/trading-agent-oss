"""The allocation cap is ONE rule with one implementation.

On 2026-08-13, MAX_ALLOC_SCALE was added to research.py's sizing but not to
trade.py's enforcement: the sizer wrote live orders at 2× the allocation and
the gate rejected them. The rule now lives only in
accounts.effective_max_allocation(); these tests pin its math and — more
importantly — trip if anyone re-introduces an inline copy.
"""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import accounts

REPO = pathlib.Path(__file__).resolve().parent.parent
# Every file allowed to mention MAX_ALLOC_SCALE. Anything else referencing it
# is an inline copy of the rule waiting to drift.
ALLOWED = {"accounts.py"}


def test_no_inline_copies_of_the_allocation_rule():
    """Flags a file that READS the knob or ASSIGNS its own copy. Prose naming it
    (docstrings retelling the incident, comments pointing at the shared helper)
    is documentation, not a second definition — stops.py cites this incident by
    name and must stay legal."""
    import re
    reads = re.compile(r"os\.(?:environ|getenv)[^\n]*MAX_ALLOC_SCALE")
    assigns = re.compile(r"^\s*MAX_ALLOC_SCALE\s*=(?!=)")
    offenders = []
    for p in REPO.glob("*.py"):
        if p.name in ALLOWED:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if reads.search(line) or assigns.search(line):
                offenders.append(f"{p.name}:{i}")
    assert not offenders, (
        f"{offenders} reference MAX_ALLOC_SCALE directly — call "
        "accounts.effective_max_allocation() instead; an inline copy is how the "
        "sizer and the gate drifted apart on 2026-08-13"
    )


def test_consumers_call_the_shared_rule():
    # research (sizing), trade (enforcement), decide (ladder tranche 2) must all
    # go through the one helper.
    for name in ("research.py", "trade.py", "decide.py"):
        src = (REPO / name).read_text()
        assert "effective_max_allocation" in src, f"{name} no longer calls the shared rule"


@pytest.mark.parametrize("raw,scale,cap,expected", [
    (0.03, 1.0, 0.10, 0.03),    # default: unscaled watchlist value
    (0.03, 2.0, 0.10, 0.06),    # live profile: doubled
    (0.10, 2.0, 0.10, 0.10),    # never above the guard's position cap
    (0.05, 4.0, 0.10, 0.10),    # extreme scale still capped
])
def test_rule_math(monkeypatch, raw, scale, cap, expected):
    monkeypatch.setenv("MAX_ALLOC_SCALE", str(scale))
    monkeypatch.setenv("RISK_MAX_POSITION_PCT", str(cap))
    assert accounts.effective_max_allocation({"max_allocation": raw}) == pytest.approx(expected)


def test_cap_honours_risk_limits_json(monkeypatch, tmp_path):
    # The ceiling comes from risk_guard's resolved limits, so a risk_limits.json
    # override tightens the sizer too — the inline copies read only the env var
    # and would have sized above a json-configured guard cap.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "risk_limits.json").write_text(
        '{"per_account": {"core": {"max_position_pct": 0.05}}}')
    monkeypatch.delenv("RISK_MAX_POSITION_PCT", raising=False)
    monkeypatch.setenv("MAX_ALLOC_SCALE", "2.0")
    assert accounts.effective_max_allocation({"max_allocation": 0.05}) == pytest.approx(0.05)
