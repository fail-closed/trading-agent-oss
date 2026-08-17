"""A buy needs a thesis. These tests hold the gate to that, and to its edges.

The gate closes the `score_only` path — 63% of entries and the worst bucket on a
987-name validation run. Four things must be true and stay true:

  IT BLOCKS THE THESIS-FREE BUY. A high score with no dip, breakout or momentum
  claim behind it does not reach the broker.

  IT DOES NOT BLOCK THE THREE REAL ARCHETYPES. A gate that also kills breakouts
  would destroy the one sleeve the same study found positive.

  IT NEVER BLOCKS A SELL. Every other gate in this system exempts exits, because
  refusing to let a stop-loss out is the one failure that compounds.

  MOMENTUM COUNTS ONLY WHEN ITS SLEEVE IS RUNNING. Otherwise a high mom_rank name
  slips through as "momentum" while the caps that govern the sleeve are switched
  off — an unmanaged momentum book through the back door.
"""
import pytest

import archetype_gate as ag


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The gate reads env on every call by design; isolate each test from it."""
    for k in ("REQUIRE_ARCHETYPE", "MOMENTUM_SLEEVE", "MOMENTUM_MIN_RANK"):
        monkeypatch.delenv(k, raising=False)


def _sig(**kw):
    base = {"symbol": "AAA", "score": 3, "is_dip": False, "is_breakout": False,
            "mom_rank": None, "mom_suspect": False}
    return {**base, **kw}


# ── the path being closed ────────────────────────────────────────────────────

def test_a_high_score_with_no_thesis_is_blocked():
    ok, arch, why = ag.check(_sig(score=4))
    assert not ok and arch is None
    assert "not a dip, not a breakout" in why


def test_even_a_perfect_score_is_blocked_without_an_archetype():
    """The score is the bar, not the reason. +5 with no setup is still no setup."""
    assert not ag.check(_sig(score=5))[0]


def test_the_block_reason_names_the_score_so_the_log_is_readable():
    _ok, _a, why = ag.check(_sig(score=2))
    assert "score +2" in why


# ── the three real archetypes must survive ───────────────────────────────────

def test_a_dip_is_allowed():
    ok, arch, _ = ag.check(_sig(is_dip=True))
    assert ok and arch == ag.DIP


def test_a_breakout_is_allowed():
    ok, arch, _ = ag.check(_sig(is_breakout=True))
    assert ok and arch == ag.BREAKOUT


def test_a_supported_dip_needs_no_special_case():
    """research.py derives is_supported_dip as `is_dip and at_support`, so it
    implies is_dip. If that ever stops being true this test is the tripwire."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "research.py").read_text()
    assert re.search(r"is_supported_dip\s*=\s*is_dip\s+and\b", src), \
        "is_supported_dip no longer implies is_dip — the gate needs its own case"
    ok, arch, _ = ag.check(_sig(is_dip=True, is_supported_dip=True))
    assert ok and arch == ag.DIP


def test_breakout_wins_when_a_name_is_both():
    """Matches entry_mix.classify: a Donchian break still under its SMA20 is being
    bought for the break, not the discount."""
    ok, arch, _ = ag.check(_sig(is_dip=True, is_breakout=True))
    assert ok and arch == ag.BREAKOUT


# ── momentum: only when the sleeve is on ─────────────────────────────────────

def test_momentum_is_allowed_when_the_sleeve_is_on(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    ok, arch, _ = ag.check(_sig(mom_rank=95.0))
    assert ok and arch == ag.MOMENTUM


def test_momentum_does_not_count_when_the_sleeve_is_off():
    """The sleeve's 1-per-session cap and candidate ranking live elsewhere. Letting
    momentum through this gate while those are off is an unmanaged book."""
    ok, _arch, why = ag.check(_sig(mom_rank=99.0))
    assert not ok
    assert "momentum sleeve off" in why


def test_momentum_below_the_rank_floor_is_not_a_thesis(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    assert not ag.check(_sig(mom_rank=89.9))[0]
    assert ag.check(_sig(mom_rank=90.0))[0], "the floor is inclusive"


def test_the_rank_floor_honours_its_env_override(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    monkeypatch.setenv("MOMENTUM_MIN_RANK", "95")
    assert not ag.check(_sig(mom_rank=92.0))[0]
    assert ag.check(_sig(mom_rank=96.0))[0]


def test_a_corrupt_rank_floor_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("MOMENTUM_MIN_RANK", "not-a-number")
    assert ag.min_mom_rank() == ag.DEFAULT_MIN_MOM_RANK


def test_a_split_suspect_momentum_rank_is_never_a_thesis(monkeypatch):
    """KLAC read -75% when the adjusted truth was +144% — a sign flip on a
    decile-ranked input. research.py withholds mom_rank when suspect; this is the
    second lock on the same door."""
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    assert not ag.check(_sig(mom_rank=99.0, mom_suspect=True))[0]


def test_a_non_numeric_rank_is_not_a_thesis(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SLEEVE", "true")
    for bad in (None, "95", [95], float("nan")):
        assert not ag.check(_sig(mom_rank=bad))[0], f"mom_rank={bad!r} slipped through"


# ── the flag ─────────────────────────────────────────────────────────────────

def test_the_gate_defaults_to_ON():
    """Default-off would leave the finding in a study nobody runs."""
    assert ag.enabled()
    assert not ag.check(_sig(score=4))[0]


def test_the_gate_can_be_switched_off_and_says_so(monkeypatch):
    monkeypatch.setenv("REQUIRE_ARCHETYPE", "false")
    ok, arch, why = ag.check(_sig(score=4))
    assert ok and arch is None
    assert "gate disabled" in why


def test_a_disabled_gate_still_reports_the_archetype(monkeypatch):
    """So the journal records what the trade WAS even where nothing is blocked."""
    monkeypatch.setenv("REQUIRE_ARCHETYPE", "false")
    ok, arch, _ = ag.check(_sig(is_breakout=True))
    assert ok and arch == ag.BREAKOUT


# ── malformed input fails closed ─────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, [], "AAA", 42])
def test_a_non_signal_has_no_archetype(bad):
    assert ag.archetype(bad) is None


def test_an_empty_signal_is_blocked():
    ok, arch, _ = ag.check({})
    assert not ok and arch is None


def test_check_of_a_non_dict_does_not_raise():
    """decide.py calls this on whatever came back in the signals file."""
    for bad in (None, [], "AAA", 42):
        ok, arch, why = ag.check(bad)
        assert not ok and arch is None and why


# ── the contract that separates this from entry_mix ──────────────────────────

def test_entry_mix_remains_fail_open_and_is_not_used_as_the_gate():
    """entry_mix.classify documents that it may never block a trade, and returns
    SIGNAL for the thesis-free case. Using it as the gate would silently couple a
    reporting tally to an execution decision — and inverting its fail-open
    contract. They must stay separate functions."""
    import entry_mix
    assert entry_mix.classify({}) == entry_mix.SIGNAL, "classify must not block"
    assert entry_mix.classify(None) == entry_mix.SIGNAL
    # The gate is the one that refuses.
    assert not ag.check({})[0]


def test_decide_gates_buys_only_and_never_sells():
    """The archetype block must be NESTED inside a BUY branch. A gate that caught
    SELL would stop a stop-loss getting out, which is the one failure that
    compounds.

    Checked by walking the AST rather than matching adjacent source lines: an
    earlier version of this test regex-matched the first mention of
    `archetype_check`, which is the sig_meta construction, and so proved nothing
    about where the gate actually sits.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "decide.py").read_text()
    tree = ast.parse(src)

    def mentions(node, needle):
        return any(isinstance(n, ast.Constant) and n.value == needle
                   for n in ast.walk(node))

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    # The gate's own statement is the tuple-unpack `_ok, _arch, _why = ...`; the
    # other mention of archetype_check is the sig_meta dict, which is not a guard.
    gate = [n for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
            and any(isinstance(c, ast.Constant) and c.value == "archetype_check"
                    for c in ast.walk(n))]
    assert len(gate) == 1, f"expected one archetype gate, found {len(gate)}"

    # Walk up to the NEAREST enclosing `if`. Outer blocks may legitimately mention
    # both BUY and SELL; the guard immediately around the gate may not.
    node = gate[0]
    while node in parents and not isinstance(parents[node], ast.If):
        node = parents[node]
    guard = parents.get(node)
    assert isinstance(guard, ast.If), "the archetype gate sits under no `if` at all"
    assert mentions(guard.test, "BUY"), \
        f"line {guard.lineno}: the archetype gate is not guarded by a BUY check"
    assert not mentions(guard.test, "SELL"), \
        f"line {guard.lineno}: the archetype gate must never apply to a SELL"


def test_decide_treats_a_missing_signal_row_as_blocked():
    """meta.get returns None when the symbol has no signal row; the fallback must
    be a block, not a silent pass."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "decide.py").read_text()
    assert 'meta.get("archetype_check") or (' in src
    assert "False, None, \"no signal row for this symbol\"" in src, \
        "a symbol with no signal row must fail closed"
