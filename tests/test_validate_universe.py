"""The validation harness — above all, that it cannot see the future.

A backtest's failure modes are not crashes; they are flattering numbers. So the
tests here target the three ways this could lie:

  LOOKAHEAD. If a signal at day i is influenced by day i+1, every result is
  fiction. Proven directly: signals computed on history truncated at i must equal
  signals computed at i from the full series, whatever happens afterwards.

  THE WRONG BENCHMARK. Measuring against zero counts market drift as skill. The
  benchmark must be the same universe over the same days.

  A SHARPE WITHOUT ITS SAMPLE SIZE. The report must state n and t, and must say in
  words when a result cannot be distinguished from luck.
"""
import math

import pytest

pd = pytest.importorskip("pandas", reason="pandas not installed")
import studies.validate_universe as vu


def _series(n=400, start=100.0, step=0.1, vol=1_000_000):
    """A gently rising series long enough for SMA200 plus a year of warmup."""
    idx = pd.date_range("2023-01-02", periods=n, freq="B").date
    close = [start + step * i for i in range(n)]
    return pd.DataFrame({"close": close,
                         "high": [c * 1.01 for c in close],
                         "low": [c * 0.99 for c in close],
                         "volume": [vol] * n}, index=idx)


# ── lookahead: the one that matters ──────────────────────────────────────────

def test_signals_cannot_see_the_future():
    """Compute at i from the full series, then from history truncated at i. They
    must be identical — otherwise a future bar is leaking into a past decision."""
    df = _series(400)
    i = 300
    full = vu.signals_on(df, i)
    truncated = vu.signals_on(df.iloc[:i + 1], i)
    assert full is not None and full == truncated


def test_a_violent_future_move_does_not_change_the_past_signal():
    """The strongest form: replace everything after i with a crash. The signal at
    i must not move by a single field."""
    df = _series(400)
    i = 300
    before = vu.signals_on(df, i)
    crashed = df.copy()
    crashed.iloc[i + 1:, crashed.columns.get_loc("close")] = 1.0
    crashed.iloc[i + 1:, crashed.columns.get_loc("high")] = 1.0
    crashed.iloc[i + 1:, crashed.columns.get_loc("low")] = 1.0
    assert vu.signals_on(crashed, i) == before


def test_forward_return_reads_forward_and_refuses_to_run_off_the_end():
    df = _series(50)
    assert vu.forward_return(df, 10, 5) == pytest.approx(
        float(df["close"].iloc[15]) / float(df["close"].iloc[10]) - 1)
    assert vu.forward_return(df, 48, 5) is None, "must not wrap or extrapolate"


def test_insufficient_history_returns_nothing_rather_than_guessing():
    assert vu.signals_on(_series(100), 90) is None


# ── the benchmark ────────────────────────────────────────────────────────────

def test_excess_is_measured_against_the_same_universe_on_the_same_days():
    """Two names on an identical STEEP path. The raw forward return is large and
    positive; the excess over a universe of those same names must be ~zero.

    The earlier version of this test used a nearly-flat series, so raw return and
    excess were both tiny and removing the benchmark subtraction changed nothing —
    it passed whether the benchmark was applied or not. A steep path makes the two
    quantities differ by an order of magnitude, so the assertion has teeth."""
    hist = {"A": _series(400, step=2.0), "B": _series(400, step=2.0)}
    excess, per_day = vu.run(hist, hold=5, min_score=-99, costs_bps=0, stride=25)
    flat = [x for xs in excess.values() for x in xs]
    assert flat, "no entries generated"
    raw = [fr for rec in per_day.values() for _a, fr, _s in rec["entries"]]
    assert statisticsmean(raw) > 0.01, "fixture should produce a clear raw gain"
    assert max(abs(x) for x in flat) < 0.001, (
        f"excess should collapse to ~0 against an identical universe, got "
        f"{max(abs(x) for x in flat):.4f} — the raw return leaked in")


def test_costs_are_charged_on_both_sides():
    hist = {"A": _series(400), "B": _series(400)}
    free, _ = vu.run(hist, hold=5, min_score=-99, costs_bps=0, stride=25)
    paid, _ = vu.run(hist, hold=5, min_score=-99, costs_bps=10, stride=25)
    f = statisticsmean([x for xs in free.values() for x in xs])
    p = statisticsmean([x for xs in paid.values() for x in xs])
    assert f - p == pytest.approx(0.0020, abs=1e-6), "10bps each way should cost 20bps"


def statisticsmean(xs):
    return sum(xs) / len(xs)


def test_a_genuinely_better_name_shows_positive_excess():
    """Sanity in the other direction: if the entry rule happens to pick the name
    that outperforms, excess must be positive. A harness that can only return zero
    is not measuring anything."""
    strong, weak = _series(400, step=0.5), _series(400, step=0.01)
    hist = {"STRONG": strong, "WEAK": weak}
    excess, _ = vu.run(hist, hold=20, min_score=-99, costs_bps=0, stride=25)
    per_symbol = {}
    for d, rec in vu.run(hist, hold=20, min_score=-99, costs_bps=0, stride=25)[1].items():
        for arch, fr, sym in rec["entries"]:
            per_symbol.setdefault(sym, []).append(fr)
    assert statisticsmean(per_symbol["STRONG"]) > statisticsmean(per_symbol["WEAK"])


# ── statistics and honesty of the report ─────────────────────────────────────

def test_stats_math():
    xs = [0.01] * 100
    st = vu.stats(xs, hold=5)
    assert st["n"] == 100
    assert st["mean_excess_pct"] == pytest.approx(1.0)
    # zero variance → Sharpe and t are defined as 0 rather than infinity
    assert st["sharpe"] == 0.0 and st["t_stat"] == 0.0


def test_stats_annualises_by_holding_period():
    xs = [0.001] * 50
    five = vu.stats(xs, hold=5)["annual_excess_pct"]
    twenty = vu.stats(xs, hold=20)["annual_excess_pct"]
    assert five == pytest.approx(twenty * 4, rel=0.01), \
        "a 5-day hold compounds 4x as often as a 20-day one"


def test_stats_refuses_to_summarise_a_single_observation():
    assert vu.stats([0.01], hold=5) == {"n": 1}


def test_report_names_a_weak_result_as_luck():
    """The report must say so in words. A reader who does not know what a t-stat
    is must still be told the number means nothing."""
    import random
    random.seed(7)
    noise = [random.gauss(0.0001, 0.05) for _ in range(200)]
    out = vu.report({"dip": noise}, {"d": {}}, hold=5, universe_n=100, years=3)
    # The footer also contains the word "luck", so asserting on the whole output
    # passed even when the per-row verdict was deleted. Assert on the dip ROW.
    row = next(l for l in out.splitlines() if l.strip().startswith("dip"))
    assert "luck" in row.lower(), f"verdict missing from the row itself: {row!r}"


def test_report_flags_a_negative_result_as_negative():
    out = vu.report({"dip": [-0.02] * 200}, {"d": {}}, hold=5, universe_n=100, years=3)
    assert "NEGATIVE" in out


def test_report_always_states_the_benchmark_and_the_insample_caveat():
    out = vu.report({"breakout": [0.01] * 50}, {"d": {}}, hold=5, universe_n=100, years=3)
    assert "equal-weight the SAME universe" in out
    assert "IN-SAMPLE" in out
    assert "urvivorship" in out


def test_report_shows_n_alongside_every_sharpe():
    out = vu.report({"breakout": [0.01, 0.02, -0.01] * 40}, {"d": {}},
                    hold=5, universe_n=100, years=3)
    assert "Sharpe" in out and "t" in out
    assert "120" in out, "n must appear next to the ratio it qualifies"


# ── universe loading ─────────────────────────────────────────────────────────

def test_loads_symbols_from_a_watchlist(tmp_path):
    import json
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"stocks": [{"symbol": "AAA"}, {"symbol": "BBB"}]}))
    assert vu.load_universe(p) == ["AAA", "BBB"]


def test_overextended_entries_are_excluded():
    """pct_b > 1.3 is a hard skip in production; the backtest must honour it or it
    measures a strategy that takes trades production refuses.

    The earlier version asserted only that the KEY existed, which is true whether
    or not the exclusion is applied. This forces the flag true and asserts no
    entry is recorded."""
    df = _series(400)
    real = vu.signals_on

    def always_overextended(d, i):
        s = real(d, i)
        if s:
            s = {**s, "overextended": True, "is_breakout": True, "score": 5}
        return s

    vu.signals_on = always_overextended
    try:
        excess, per_day = vu.run({"A": df}, hold=5, min_score=-99, costs_bps=0, stride=25)
        entries = [e for rec in per_day.values() for e in rec["entries"]]
        assert not entries, "an overextended name must never become an entry"
        assert not excess
    finally:
        vu.signals_on = real
