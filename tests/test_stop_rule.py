"""The stop rule exists ONCE — tripwire + behavior tests.

Template: tests/test_allocation_rule.py, written after MAX_ALLOC_SCALE was found
half-wired (sizer 2×, gates 1×). The stop had the same shape: −8% defined in
decide.py, trade.py and stop_monitor.py with three different override paths
(`LIVE_STOP_PCT`, `ATR_STOP_MULT`, bare constants), and the trailing ratchet
existed only inside decide.py — so fractional positions, which are essentially
the entire live book, never locked a profit at all.

If this test fails, do NOT re-inline the constant to make it pass. Route the new
consumer through `stops.py`.
"""
import math
import re
import subprocess
from pathlib import Path

import pytest

import stops

ROOT = Path(__file__).resolve().parent.parent
OWNER = "stops.py"
ALLOWED = {OWNER, "tests/test_stop_rule.py"}


def _tracked_py():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return [f for f in out if f not in ALLOWED]


# ── Tripwire: nobody else may define the rule ────────────────────────────────

@pytest.mark.parametrize("token", ["TRAILING_STOP_SCHEDULE", "LIVE_STOP_PCT", "ATR_STOP_MULT"])
def test_no_file_outside_stops_defines_the_rule(token):
    """Flags real drift — a file READING the knob from env or ASSIGNING its own
    copy. Prose that merely names it (docstrings, comments) is documentation,
    not a second definition, and stays legal."""
    reads = re.compile(rf"os\.(?:environ|getenv)[^\n]*{token}")
    assigns = re.compile(rf"^\s*{token}\s*=(?!=)")
    offenders = []
    for name in _tracked_py():
        for i, line in enumerate((ROOT / name).read_text().splitlines(), 1):
            if line.strip().startswith("#") or "stops." in line:
                continue
            if reads.search(line) or assigns.search(line):
                offenders.append(f"{name}:{i}: {line.strip()[:90]}")
    assert not offenders, (
        f"{token} is referenced outside {OWNER}:\n  " + "\n  ".join(offenders) +
        f"\n\nThe stop rule lives in {OWNER} and is consumed via "
        "stops.hard_exit_pct() / initial_stop_pct() / ratchet_stop_pct(). "
        "Re-inlining it is how the −8% floor drifted across three files and how "
        "MAX_ALLOC_SCALE ended up half-wired (OPERATIONS.md §6 row 11).")


def test_consumers_import_the_shared_module():
    for name in ("decide.py", "trade.py", "stop_monitor.py"):
        assert "import stops" in (ROOT / name).read_text(), \
            f"{name} must consume the shared stop rule, not its own constant"


def test_claude_md_still_states_the_enforced_floor():
    """CLAUDE.md is the production trading prompt — it must not promise the LLM a
    stop the code doesn't enforce."""
    assert "| Position ≤ −8% unrealized | Sell all (stop-loss) |" in (ROOT / "CLAUDE.md").read_text()
    assert stops.hard_exit_pct() == pytest.approx(-0.08)


# ── Behavior: the floor ──────────────────────────────────────────────────────

def test_hard_floor_and_kill_switch():
    assert stops.hard_exit_pct() == pytest.approx(-0.08)
    assert stops.hard_exit_pct(kill_switch=True) == pytest.approx(-0.05)


def test_live_stop_pct_overrides_every_consumer(monkeypatch):
    monkeypatch.setenv("LIVE_STOP_PCT", "0.06")
    assert stops.hard_exit_pct() == pytest.approx(-0.06)
    monkeypatch.setenv("LIVE_STOP_PCT", "garbage")
    assert stops.hard_exit_pct() == pytest.approx(-0.08)   # never crash the monitor


# ── Behavior: the initial GTC distance ───────────────────────────────────────

def test_initial_stop_is_flat_without_atr_knob(monkeypatch):
    monkeypatch.delenv("ATR_STOP_MULT", raising=False)
    assert stops.initial_stop_pct(atr14=5.0, price=100.0) == pytest.approx(0.08)
    assert stops.atr_stop_enabled() is False


def test_initial_stop_scales_and_clamps_with_atr(monkeypatch):
    monkeypatch.setenv("ATR_STOP_MULT", "2.0")
    assert stops.atr_stop_enabled() is True
    assert stops.initial_stop_pct(atr14=3.0, price=100.0) == pytest.approx(0.06)
    assert stops.initial_stop_pct(atr14=0.5, price=100.0) == pytest.approx(0.04)   # floor
    assert stops.initial_stop_pct(atr14=20.0, price=100.0) == pytest.approx(0.15)  # ceil
    assert stops.initial_stop_pct(atr14=0, price=0) == pytest.approx(0.08)         # degrade


def test_explicit_base_survives_when_atr_absent(monkeypatch):
    monkeypatch.delenv("ATR_STOP_MULT", raising=False)
    assert stops.initial_stop_pct(base=0.05) == pytest.approx(0.05)


# ── Behavior: the ratchet (shared by decide.py and stop_monitor.py) ──────────

@pytest.mark.parametrize("gain,expected", [
    (0.00, -0.08), (0.04, -0.08),      # below the first trigger → the floor
    (0.05, 0.00), (0.09, 0.00),        # +5% → break-even
    (0.10, 0.05), (0.14, 0.05),        # +10% → lock +5%
    (0.15, 0.08), (0.40, 0.08),        # +15% → lock +8%
])
def test_static_ratchet_schedule(gain, expected):
    assert stops.ratchet_stop_pct(gain)[0] == pytest.approx(expected)


def test_ratchet_is_monotonic_in_gain():
    levels = [stops.ratchet_stop_pct(g / 100)[0] for g in range(0, 60)]
    assert levels == sorted(levels), "a bigger gain must never produce a looser stop"


def test_kill_switch_reaches_locks_sooner_and_floors_tighter():
    assert stops.ratchet_stop_pct(0.02)[0] == pytest.approx(-0.05, abs=1e-9) or \
           stops.ratchet_stop_pct(0.02, kill_switch=True)[0] == pytest.approx(-0.05)
    # +3.5% is below the normal +5% trigger but clears the kill-switch's 0.7×
    assert stops.ratchet_stop_pct(0.036)[0] == pytest.approx(-0.08)
    assert stops.ratchet_stop_pct(0.036, kill_switch=True)[0] == pytest.approx(0.0)


def test_atr_branch_scales_with_volatility():
    # atr_pct 4%: +12% gain clears 3×ATR → lock 1.5×ATR = +6%
    assert stops.ratchet_stop_pct(0.12, atr_pct=0.04)[0] == pytest.approx(0.06)
    # +8% clears 2×ATR → lock 0.5×ATR = +2%
    assert stops.ratchet_stop_pct(0.08, atr_pct=0.04)[0] == pytest.approx(0.02)
    # +5% clears 1×ATR → break-even
    assert stops.ratchet_stop_pct(0.05, atr_pct=0.04)[0] == pytest.approx(0.0)
    # +1% clears nothing → initial ATR stop at −2.5×ATR
    assert stops.ratchet_stop_pct(0.01, atr_pct=0.04)[0] == pytest.approx(-0.10)


def test_labels_are_human_readable():
    assert "break-even" in stops.ratchet_stop_pct(0.06)[1]
    assert "+8%" in stops.ratchet_stop_pct(0.20)[1]
    assert "⚠KILL" in stops.ratchet_stop_pct(0.20, kill_switch=True)[1]


# ── The fractional gap this closes ──────────────────────────────────────────

def test_fractional_positions_get_a_lock_the_broker_cannot_hold():
    """The whole point of audit item 4.2: a 0.65-share position at +18% has no
    broker GTC stop (Alpaca rejects fractional GTC), so before this change its
    only protection was the −8% floor — it could round-trip the entire gain."""
    qty = 0.65
    assert math.floor(qty) < 1, "premise: broker cannot hold a stop on this"
    lock, _ = stops.ratchet_stop_pct(0.18)
    assert lock == pytest.approx(0.08)
    assert lock > stops.hard_exit_pct(), "the lock must sit above the hard floor"
