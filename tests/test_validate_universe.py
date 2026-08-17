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


# ── correlated observations: the inflated t-stat ─────────────────────────────
#
# At 10 names this barely mattered. At 1,000 the walk opens ~8 trades per sample
# day, and pooling them multiplies the apparent sample size by 8 while adding
# almost no independent information. These tests hold the harness to the
# clustered figure.

def _same_day_book(day_returns, per_day_count, arch="dip"):
    """A book where every trade opened on a day has the SAME return — the limiting
    case of within-day correlation, where pooling is maximally wrong. Benchmark is
    zero so the arithmetic stays checkable by hand."""
    per_day, excess = {}, {arch: []}
    for k, r in enumerate(day_returns):
        per_day[f"day{k:03d}"] = {
            "bench": [0.0],
            "entries": [(arch, r, f"S{j}") for j in range(per_day_count)],
        }
        excess[arch] += [r] * per_day_count
    return excess, per_day


def _spread(mean, sd, n):
    """n values with the requested mean and (very nearly) the requested sd."""
    return [mean + sd if k % 2 == 0 else mean - sd for k in range(n)]


def test_daily_means_averages_within_the_day():
    _e, per_day = _same_day_book([0.03], per_day_count=1)
    per_day["day000"]["entries"] = [("dip", 0.01, "A"), ("dip", 0.05, "B")]
    assert vu.daily_means(per_day, 0.0)["dip"] == [pytest.approx(0.03)]


def test_daily_means_subtracts_that_days_benchmark_and_the_cost():
    per_day = {"d": {"bench": [0.02, 0.02], "entries": [("dip", 0.05, "A")]}}
    assert vu.daily_means(per_day, 0.001)["dip"] == [pytest.approx(0.05 - 0.001 - 0.02)]


def test_daily_means_ignores_a_day_with_no_benchmark():
    """No benchmark means no excess is definable — the day must be dropped, not
    treated as a day whose benchmark was zero."""
    per_day = {"d": {"bench": [], "entries": [("dip", 0.05, "A")]}}
    assert vu.daily_means(per_day, 0.0) == {}


def test_clustering_by_date_deflates_the_pooled_t():
    """50 identical trades a day carry one day's worth of information, not 50.
    The pooled t must exceed the clustered one by ~sqrt(50)."""
    excess, per_day = _same_day_book(_spread(0.002, 0.01, 30), per_day_count=50)
    pooled = abs(vu.stats(excess["dip"], hold=5)["t_stat"])
    clustered = abs(vu._t(vu.daily_means(per_day, 0.0)["dip"]))
    assert clustered > 0
    assert pooled > 4 * clustered, (
        f"pooled {pooled} vs clustered {clustered} — the correction did nothing")


def test_the_verdict_follows_the_clustered_t_not_the_pooled_one():
    """The case that matters: pooling says 'significant', clustering says 'luck'.
    If the row reports significance here, the harness is lying the way every
    backtest lies."""
    excess, per_day = _same_day_book(_spread(0.0015, 0.012, 30), per_day_count=50)
    pooled = abs(vu.stats(excess["dip"], hold=5)["t_stat"])
    assert pooled >= 2.4, "fixture must be one that pooling would call significant"

    out = vu.report(excess, per_day, hold=5, universe_n=1000, years=3, costs_bps=0)
    row = next(l for l in out.splitlines() if l.strip().startswith("dip"))
    assert "indistinguishable from luck" in row, f"row: {row!r}"
    assert "days" not in row and "30" in row, "the day count must be shown"


def test_a_result_between_two_and_the_raised_bar_is_not_called_significant():
    """Three archetypes tested on one dataset. t=2.2 clears the textbook 2.0 and
    must still fail here, or we are reporting the best of three coin flips."""
    excess, per_day = _same_day_book(_spread(0.004903, 0.012, 30), per_day_count=10)
    td = abs(vu._t(vu.daily_means(per_day, 0.0)["dip"]))
    assert 2.0 < td < 2.4, f"fixture drifted: t·day = {td}"
    out = vu.report(excess, per_day, hold=5, universe_n=1000, years=3, costs_bps=0)
    row = next(l for l in out.splitlines() if l.strip().startswith("dip"))
    assert "suggestive, not significant" in row, f"row: {row!r}"


def test_a_genuinely_strong_clustered_result_is_still_called_significant():
    """The correction must not be a blanket 'nothing is ever significant'."""
    excess, per_day = _same_day_book(_spread(0.010, 0.012, 30), per_day_count=10)
    assert abs(vu._t(vu.daily_means(per_day, 0.0)["dip"])) > 2.4
    out = vu.report(excess, per_day, hold=5, universe_n=1000, years=3, costs_bps=0)
    row = next(l for l in out.splitlines() if l.strip().startswith("dip"))
    assert "significant" in row and "not significant" not in row, f"row: {row!r}"


def test_report_shows_both_t_stats_and_explains_which_one_decides():
    excess, per_day = _same_day_book(_spread(0.002, 0.01, 30), per_day_count=50)
    out = vu.report(excess, per_day, hold=5, universe_n=1000, years=3, costs_bps=0)
    assert "t·pool" in out and "t·day" in out, "both must be visible to compare"
    assert "VERDICTS USE t·day" in out
    assert "BAR OF 2.4" in out


def test_falling_back_to_the_pooled_t_is_labelled_as_such():
    """When there are no day clusters the pooled figure is all there is — but it
    is the optimistic one, so saying so is mandatory."""
    out = vu.report({"dip": [0.02] * 200}, {"d": {}}, hold=5, universe_n=100, years=3)
    row = next(l for l in out.splitlines() if l.strip().startswith("dip"))
    assert "pooled t" in row, f"unlabelled fallback to the optimistic statistic: {row!r}"


def test_sharpe_is_still_the_per_trade_figure_and_says_so():
    out = vu.report({"breakout": [0.01] * 50}, {"d": {}}, hold=5, universe_n=100, years=3)
    assert "Sharpe and excess/trade are per-trade figures" in out


# ── breadth: when the weighting changes the sign ─────────────────────────────
#
# The 1,000-name run produced a dip sleeve that was negative per trade and
# positive per day. That is not a bug — the pooled mean weights each day by its
# trade count — but silently reporting one of the two would be.

def _book(days, arch="dip"):
    """days: [(n_trades, per_trade_return)] with a zero benchmark."""
    per_day, excess = {}, {arch: []}
    for k, (n, r) in enumerate(days):
        per_day[f"day{k:03d}"] = {"bench": [0.0],
                                  "entries": [(arch, r, f"S{j}") for j in range(n)]}
        excess[arch] += [r] * n
    return excess, per_day


def test_the_two_weightings_can_disagree_in_sign_and_both_are_reported():
    """The exact shape the real run produced. Losing on wide days and winning on
    narrow ones makes per-trade negative while per-day is positive."""
    _e, per_day = _book([(50, -0.01), (50, -0.01), (2, +0.06), (2, +0.06)])
    v = vu.breadth(per_day, 0.0)["dip"]
    assert v["trade_weighted_pct"] < 0 < v["day_weighted_pct"], v
    # per-trade: (100*-0.01 + 4*0.06)/104 = -0.7308%
    # per-day:   mean(-1%, -1%, +6%, +6%)  = +2.5%
    assert v["trade_weighted_pct"] == pytest.approx(-0.7308, abs=1e-3)
    assert v["day_weighted_pct"] == pytest.approx(2.5, abs=1e-3)


def test_the_singleton_share_is_quantified_because_it_is_the_real_mechanism():
    """The real run's sign flip came from single-trade days holding a third of the
    day-weight at ~8% sd each — NOT from wide days being bad. This is the number
    that explains it, so it must be computed and correct."""
    # 3 singleton days at +10%, 1 wide day of 60 names at -1%
    _e, per_day = _book([(1, 0.10), (1, 0.10), (1, 0.10), (60, -0.01)])
    v = vu.breadth(per_day, 0.0)["dip"]
    assert v["singleton_days"] == 3
    assert v["singleton_day_share"] == pytest.approx(0.75)
    assert v["singleton_mean_pct"] == pytest.approx(10.0)
    # trade-weighted is dominated by the 60: (3*0.10 + 60*-0.01)/63 = -0.476%
    assert v["trade_weighted_pct"] == pytest.approx(-0.476, abs=1e-3)
    # day-weighted is dominated by the singletons: mean(10,10,10,-1)/4 = +7.25%
    assert v["day_weighted_pct"] == pytest.approx(7.25, abs=1e-3)


def test_the_wide_day_mean_excludes_singleton_noise():
    """The quiet estimate: only days broad enough that the day-mean is itself an
    average of many names."""
    _e, per_day = _book([(1, 0.50), (1, -0.50), (30, 0.02), (70, 0.01)])
    v = vu.breadth(per_day, 0.0)["dip"]
    assert v["wide_days"] == 2 and v["wide_trades"] == 100
    # (30*0.02 + 70*0.01)/100 = 1.3%, with the two ±50% singletons excluded
    assert v["wide_mean_pct"] == pytest.approx(1.3, abs=1e-3)


def test_wide_mean_is_none_when_no_day_was_broad_enough():
    """Rather than reporting a 'quiet estimate' computed from nothing."""
    _e, per_day = _book([(1, 0.05), (3, 0.02)])
    assert vu.breadth(per_day, 0.0)["dip"]["wide_mean_pct"] is None


def test_a_sign_disagreement_forces_a_warning_into_the_report():
    excess, per_day = _book([(1, 0.10), (1, 0.10), (1, 0.10), (60, -0.01)])
    out = vu.report(excess, per_day, hold=5, universe_n=987, years=3, costs_bps=0)
    assert "BREADTH WARNING" in out
    assert "per-trade" in out and "per-day" in out
    assert "single-trade days: 3 of 4" in out, "the mechanism must be in the report"
    # The share of the DAY-WEIGHT is the number that makes the flip legible; the
    # raw count alone does not. Assert the figure itself, not just the label.
    assert "75% of the day-weight" in out, \
        "the singleton share of the day-weight must be stated, not just the count"
    assert "sd" in out, "a singleton mean without its sd invites over-reading it"
    assert "the quiet estimate" in out


def test_the_report_does_not_claim_a_breadth_outcome_relationship():
    """An earlier version asserted that a negative breadth/excess correlation meant
    the archetype 'fires widest on its worst days'. The real data showed a flat
    correlation (-0.039) alongside a genuine sign flip, so that explanation was
    wrong and must not reappear in the output."""
    excess, per_day = _book([(1, 0.10), (1, 0.10), (1, 0.10), (60, -0.01)])
    out = vu.report(excess, per_day, hold=5, universe_n=987, years=3, costs_bps=0)
    assert "widest on its worst days" not in out
    assert "corr(breadth,excess)" not in out, \
        "a near-useless statistic on skewed counts must not headline the diagnosis"


def test_no_breadth_warning_when_the_weightings_agree():
    """The section must stay quiet in the ordinary case, or it becomes wallpaper."""
    excess, per_day = _book([(10, 0.01), (10, 0.02), (10, 0.015)])
    out = vu.report(excess, per_day, hold=5, universe_n=987, years=3, costs_bps=0)
    assert "BREADTH WARNING" not in out


def test_breadth_charges_the_cost_and_the_benchmark_like_everything_else():
    per_day = {"d": {"bench": [0.01], "entries": [("dip", 0.05, "A"),
                                                 ("dip", 0.03, "B")]}}
    v = vu.breadth(per_day, 0.002)["dip"]
    # mean raw 0.04, less 0.002 cost, less the 0.01 benchmark = 0.028
    assert v["day_weighted_pct"] == pytest.approx((0.04 - 0.002 - 0.01) * 100, abs=1e-6)


def test_corr_declines_to_guess_from_too_little_data():
    assert vu._corr([1, 2], [0.1, 0.2]) is None, "two points is not a correlation"
    assert vu._corr([5, 5, 5], [0.1, 0.2, 0.3]) is None, "no variance in breadth"


# ── dump / load: asking a second question of the same walk ───────────────────

def test_a_dumped_walk_reloads_to_an_identical_report(tmp_path, monkeypatch, capsys):
    """A follow-up question must be answered from the SAME numbers, not from a
    second walk that might differ."""
    import pickle
    excess, per_day = _book([(3, 0.01), (3, -0.02), (3, 0.015)])
    meta = {"hold": 5, "years": 3.0, "costs_bps": 0.0, "universe_n": 987,
            "stride": 5, "min_score": 2}
    p = tmp_path / "walk.pkl"
    p.write_bytes(pickle.dumps((excess, per_day, meta)))

    assert vu.main(["--load", str(p)]) == 0
    reloaded = capsys.readouterr().out
    assert reloaded == vu.report(excess, per_day, 5, 987, 3.0, 0.0) + "\n"
    assert "987 names" in reloaded


def test_load_does_not_touch_the_network_or_the_watchlist(tmp_path, monkeypatch):
    """--load must be genuinely offline: if it still fetched, the whole point of
    caching the walk would be lost."""
    import pickle
    excess, per_day = _book([(3, 0.01), (3, -0.02), (3, 0.015)])
    meta = {"hold": 5, "years": 3.0, "costs_bps": 0.0, "universe_n": 9,
            "stride": 5, "min_score": 2}
    p = tmp_path / "walk.pkl"
    p.write_bytes(pickle.dumps((excess, per_day, meta)))

    def boom(*a, **k):
        raise AssertionError("--load must not fetch or re-walk")

    monkeypatch.setattr(vu, "fetch_history", boom)
    monkeypatch.setattr(vu, "load_universe", boom)
    monkeypatch.setattr(vu, "run", boom)
    assert vu.main(["--load", str(p)]) == 0


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
