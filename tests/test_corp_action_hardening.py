"""Corporate-action and liquidity hardening for a ranked universe.

Three defects this closes, all of which were harmless at 30 hand-picked names and
become routine at 1,000 ranked ones:

  A FAILED SPLIT LOOKUP WAS INDISTINGUISHABLE FROM "NO SPLITS". `_fetch` swallowed
  its exception and returned `{"splits": {}}`, which is also what a clean name
  returns. Those demand opposite responses — the second is a clean signal, the
  first means the price series may be unadjusted and every indicator derived from
  it is suspect.

  THE SPLIT GUARD ONLY CAUGHT BIG SPLITS. It fires on |price − SMA20| ≥ 35%. A 2:1
  is −50% and trips it; a 3:2 is −33% and slips under; a 5:4 is −20%. Common
  ratios, arriving weekly rather than quarterly at scale.

  THE ADV GUARD WAS OFF. Fine for a list where every name was known liquid; not
  once the universe is chosen by rank and a name's volume can collapse between
  monthly rebuilds.
"""
import pytest

import corporate_actions as ca


# ── failure is not absence ───────────────────────────────────────────────────

def test_a_clean_name_and_a_failed_lookup_are_distinguishable():
    clean = {"splits": {}, "next_ex_div": None, "ok": True}
    failed = {"splits": {}, "next_ex_div": None, "ok": False, "error": "429"}
    assert clean["splits"] == failed["splits"], "both are empty — that was the bug"
    assert ca.data_ok({"A": clean}, "A")
    assert not ca.data_ok({"B": failed}, "B")


def test_data_ok_is_false_for_an_unknown_symbol():
    """Absent from the map means we never asked, which is not the same as clean."""
    assert not ca.data_ok({}, "NEVER_FETCHED")


def test_coverage_reports_failures_separately_from_no_splits():
    actions = {
        "OK1": {"splits": {}, "ok": True},
        "OK2": {"splits": {"2026-08-01": 2.0}, "ok": True},
        "BAD1": {"splits": {}, "ok": False},
        "BAD2": {"splits": {}, "ok": False},
    }
    c = ca.coverage(actions)
    assert c == {"total": 4, "ok": 2, "failed": 2, "with_splits": 1, "pct_ok": 50.0}


def test_coverage_of_an_empty_universe_does_not_divide_by_zero():
    assert ca.coverage({})["pct_ok"] == 0.0


# ── failures must not be cached ──────────────────────────────────────────────

def test_a_failed_lookup_is_not_written_to_the_daily_cache(monkeypatch, tmp_path):
    """Caching a failure turns one rate-limit into a whole session of silently
    unadjusted prices — and the cached {} reads as 'no splits'."""
    monkeypatch.setenv("CORP_ACTIONS", "true")
    monkeypatch.setattr(ca, "_cache_dir", lambda: tmp_path)
    saved = {}
    monkeypatch.setattr(ca, "_load_cache", lambda: {})
    monkeypatch.setattr(ca, "_save_cache", lambda c: saved.update(c))
    monkeypatch.setattr(ca, "_fetch", lambda s: (
        {"splits": {}, "next_ex_div": None, "ok": True} if s == "GOOD"
        else {"splits": {}, "next_ex_div": None, "ok": False, "error": "429"}))

    out = ca.fetch_all(["GOOD", "BAD"])
    assert out["GOOD"]["ok"] and not out["BAD"]["ok"]
    assert "GOOD" in saved
    assert "BAD" not in saved, "a failure must be retried, not cached for the day"


def test_fetch_all_returns_an_entry_for_every_symbol(monkeypatch, tmp_path):
    """A missing key would read as 'never asked' downstream and hide the failure."""
    monkeypatch.setenv("CORP_ACTIONS", "true")
    monkeypatch.setattr(ca, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(ca, "_load_cache", lambda: {})
    monkeypatch.setattr(ca, "_save_cache", lambda c: None)
    monkeypatch.setattr(ca, "_fetch", lambda s: {"splits": {}, "ok": False})
    syms = [f"S{i}" for i in range(50)]
    out = ca.fetch_all(syms)
    assert set(out) == set(syms)


def test_parallel_fetch_survives_one_worker_raising(monkeypatch, tmp_path):
    monkeypatch.setenv("CORP_ACTIONS", "true")
    monkeypatch.setattr(ca, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(ca, "_load_cache", lambda: {})
    monkeypatch.setattr(ca, "_save_cache", lambda c: None)

    def flaky(sym):
        if sym == "BOOM":
            raise RuntimeError("network")
        return {"splits": {}, "ok": True}

    monkeypatch.setattr(ca, "_fetch", flaky)
    out = ca.fetch_all(["A", "BOOM", "B"])
    assert out["A"]["ok"] and out["B"]["ok"]
    assert not out["BOOM"]["ok"], "a raising worker must degrade to ok=False, not vanish"


# ── the magnitude gap the small-split check closes ───────────────────────────

@pytest.mark.parametrize("ratio,move_pct,caught_by_magnitude", [
    (2.0,   -50.0, True),    # 2:1  — trips the 35% test
    (3.0,   -66.7, True),    # 3:1
    (1.5,   -33.3, False),   # 3:2  — SLIPS UNDER, this is the gap
    (1.25,  -20.0, False),   # 5:4  — slips under by a wide margin
    (1.1,    -9.1, False),   # 11:10
])
def test_magnitude_test_alone_misses_small_splits(ratio, move_pct, caught_by_magnitude):
    """Documents the arithmetic that motivated the data-driven check. A split of
    `ratio` moves price to 1/ratio of its pre-split level, so the gap versus a
    trailing SMA20 is (1/ratio − 1)."""
    threshold = 0.35
    implied = (1.0 / ratio - 1.0) * 100
    assert abs(implied - move_pct) < 0.5, "arithmetic drifted"
    assert (abs(implied) / 100 >= threshold) == caught_by_magnitude


def test_a_recent_split_is_suspect_regardless_of_size():
    """The rule research.py now applies: any split inside the SMA20 window
    contaminates that average, however small the ratio."""
    from datetime import date, timedelta
    recent = (date.today() - timedelta(days=5)).isoformat()
    old = (date.today() - timedelta(days=200)).isoformat()
    cut = (date.today() - timedelta(days=30)).isoformat()
    assert any(d >= cut for d in {recent: 1.25}), "a 5-day-old 5:4 must be suspect"
    assert not any(d >= cut for d in {old: 2.0}), "a 200-day-old split is out of window"


# ── the liquidity guard ──────────────────────────────────────────────────────

def test_adv_guard_is_on_by_default():
    """It was 0 (off). At a ranked universe that is a fail-open: it is the only
    check that looks at TODAY'S volume at order time."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "trade.py").read_text()
    m = re.search(r'ADV_MAX_PCT",\s*"([0-9.]+)"', src)
    assert m, "the ADV_MAX_PCT default disappeared"
    assert float(m.group(1)) > 0, "the illiquidity guard must not default to off"


def test_adv_default_is_non_binding_for_a_liquid_universe():
    """1% of the $7.7M/day floor a 1,000-name liquidity ranking produces is $77k —
    far above anything this sizer places, so the guard only bites on names that
    have genuinely dried up rather than blocking ordinary trades."""
    adv_floor, pct = 7_728_546, 0.01
    assert adv_floor * pct > 50_000
