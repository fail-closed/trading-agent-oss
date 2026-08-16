"""
Tests for the non-price data sources: earnings calendar, corporate actions,
credit regime, insider flow, and IV metrics.

All pure-function or fixture-driven — no network, no keys. The network paths are
exercised by the modules' own __main__ blocks during development; what matters
here is that the *decision logic* on top of them is right, and that every source
degrades safely when the data is missing.
"""
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# The code under test anchors every earnings calculation to the EXCHANGE day
# (earnings_calendar.enrich: `today = datetime.now(ET).date()`), so the fixtures
# must be built on the same clock. Using date.today() made "N days out" mean the
# runner's local day, which is a different date from ET for four hours every
# evening — CI genuinely failed on PR #42 at 20:15 ET for exactly this, passed on
# the next two runs at 10:17 and 10:37 ET, and fails all day on a JST laptop.
# A date-dependent test that is green only part of the day is worse than no test:
# CI gates deployment, so it can block a real-money fix at 9pm for no reason.
ET = ZoneInfo("America/New_York")


def _iso(days_from_today):
    return (datetime.now(ET).date() + timedelta(days=days_from_today)).isoformat()


# ── Earnings calendar + blackout ─────────────────────────────────────────────

@pytest.fixture
def ec(monkeypatch):
    monkeypatch.setenv("LEDGER_DIR", tempfile.mkdtemp())
    monkeypatch.delenv("EARNINGS_BLACKOUT_DAYS", raising=False)
    monkeypatch.delenv("EARNINGS_CALENDAR", raising=False)
    import earnings_calendar
    return earnings_calendar


def _seed(ec, monkeypatch, data_by_symbol):
    """Bypass the network: feed _fetch directly."""
    monkeypatch.setattr(ec, "_fetch", lambda sym: data_by_symbol.get(sym, {}))


def test_earnings_defaults_on_because_it_backs_a_hard_rule(ec):
    """This is the data behind a NON-NEGOTIABLE rule, so 'have the data' is the
    safe default, not 'have the flag'."""
    assert ec.enabled() is True


@pytest.mark.parametrize("days_out,expect_blackout", [
    (0, True), (1, True), (2, True), (3, False), (30, False),
])
def test_blackout_window(ec, monkeypatch, days_out, expect_blackout):
    _seed(ec, monkeypatch, {"AAA": {"next": _iso(days_out)}})
    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_blackout"] is expect_blackout
    assert (sigs[0]["symbol"] in ec.blackout_symbols(sigs)) is expect_blackout


def test_blackout_window_is_tunable(ec, monkeypatch):
    monkeypatch.setenv("EARNINGS_BLACKOUT_DAYS", "5")
    _seed(ec, monkeypatch, {"AAA": {"next": _iso(4)}})
    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_blackout"] is True


def test_past_earnings_never_blacks_out(ec, monkeypatch):
    """A print that already happened is not gap risk — it's the PEAD window."""
    _seed(ec, monkeypatch, {"AAA": {"last": _iso(-10), "last_surprise_pct": 8.0}})
    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_blackout"] is False
    assert sigs[0]["days_since_earnings"] == 10
    assert sigs[0]["pead_window"] is True


def test_pead_needs_a_real_beat_inside_the_window(ec, monkeypatch):
    _seed(ec, monkeypatch, {
        "SMALL": {"last": _iso(-10), "last_surprise_pct": 1.0},    # beat too small
        "STALE": {"last": _iso(-200), "last_surprise_pct": 20.0},  # drift is over
        "MISS":  {"last": _iso(-10), "last_surprise_pct": -8.0},   # a miss
    })
    sigs = [{"symbol": s} for s in ("SMALL", "STALE", "MISS")]
    ec.enrich(sigs)
    assert not any(s["pead_window"] for s in sigs)


def test_unknown_earnings_date_does_not_block(ec, monkeypatch):
    """No data must not silently freeze all buying — it degrades to 'no blackout'
    and the other risk rules still apply."""
    _seed(ec, monkeypatch, {"AAA": {}})
    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_blackout"] is False
    assert ec.blackout_symbols(sigs) == set()


def test_disabled_still_sets_the_field(ec, monkeypatch):
    """decide.py reads earnings_blackout unconditionally — it must never be absent."""
    monkeypatch.setenv("EARNINGS_CALENDAR", "false")
    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_blackout"] is False


# ── Corporate actions ────────────────────────────────────────────────────────

def test_split_adjustment_reverses_a_10_for_1(monkeypatch):
    """The real KLAC case: $900 pre-split, $90 post. Back-adjusted, the series
    must be continuous rather than showing a −90% day."""
    import pandas as pd
    import corporate_actions as ca

    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    close = pd.Series([900.0] * 5 + [90.0] * 5, index=idx)
    adj = ca.adjust_close(close, {"2026-01-06": 10.0})
    assert adj.iloc[0] == pytest.approx(90.0)
    assert adj.iloc[-1] == pytest.approx(90.0)
    assert adj.pct_change().dropna().abs().max() == pytest.approx(0.0)


def test_adjust_is_a_noop_without_splits(monkeypatch):
    import pandas as pd
    import corporate_actions as ca
    close = pd.Series([1.0, 2.0, 3.0])
    assert ca.adjust_close(close, {}) is close
    assert ca.adjust_close(close, None) is close


def test_adjust_survives_a_garbage_split_entry():
    """A bad ratio must not take research down — fall back to the raw series."""
    import pandas as pd
    import corporate_actions as ca
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    close = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)
    out = ca.adjust_close(close, {"not-a-date": "banana", "2026-01-03": 0})
    assert list(out) == [10.0, 10.0, 10.0, 10.0]


def test_momentum_uses_splits_to_correct_rather_than_discard():
    """The whole point: with split data KLAC is CORRECTED (+142%), not dropped.
    Without it, the guard still withholds."""
    import pandas as pd
    import research

    # A year flat at $900, then a 10:1 split 40 sessions ago, then flat at $90.
    prices = [900.0] * 220 + [90.0] * 40
    idx = pd.date_range("2025-06-01", periods=len(prices), freq="D")
    close = pd.Series(prices, index=idx)
    split_date = idx[220].date().isoformat()

    unguarded = research.compute_momentum(close)
    assert unguarded["mom_12_1"] is None and unguarded["mom_suspect"] is True

    fixed = research.compute_momentum(close, {split_date: 10.0})
    assert fixed["mom_suspect"] is False
    assert fixed["mom_12_1"] == pytest.approx(0.0, abs=1e-6)   # genuinely flat


def test_ex_div_window():
    import corporate_actions as ca
    today = date(2026, 8, 12)
    assert ca.ex_div_within({"next_ex_div": "2026-08-14"}, 5, today) is True
    assert ca.ex_div_within({"next_ex_div": "2026-09-30"}, 5, today) is False
    assert ca.ex_div_within({"next_ex_div": None}, 5, today) is False
    assert ca.ex_div_within({}, 5, today) is False


# ── Credit regime ────────────────────────────────────────────────────────────

def test_credit_regime_classification(monkeypatch):
    import macro_fred

    def fake(series):
        def _f(api_key, series_id, days_back=120, limit=40):
            return series.get(series_id, [])
        return _f

    # Spreads gapping out +1.2pp from a low base = severe, even though the level
    # alone (4.0) would read as normal. Widening is the tell, not the level.
    monkeypatch.setattr(macro_fred, "fetch_fred_history", fake({
        "BAMLH0A0HYM2": [(f"d{i}", 2.8) for i in range(20)] + [("now", 4.0)],
        "NFCI": [], "T10Y3M": [],
    }))
    assert macro_fred.get_credit_regime("k")["credit_stress"] == "severe"

    monkeypatch.setattr(macro_fred, "fetch_fred_history", fake({
        "BAMLH0A0HYM2": [(f"d{i}", 2.70) for i in range(30)],
        "NFCI": [], "T10Y3M": [],
    }))
    assert macro_fred.get_credit_regime("k")["credit_stress"] == "normal"


def test_credit_regime_degrades_without_a_key(monkeypatch):
    import macro_fred
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    out = macro_fred.get_credit_regime(None)
    assert out["hy_oas"] is None and out["credit_stress"] == "unknown"


def test_macro_regime_keeps_its_original_contract(monkeypatch):
    """macro_context.py reads these three keys — adding credit must not break them."""
    import macro_fred
    monkeypatch.setattr(macro_fred, "fetch_fred_series", lambda k, s, **kw: {"value": 0.5})
    monkeypatch.setattr(macro_fred, "fetch_fred_history", lambda *a, **kw: [])
    monkeypatch.setenv("FRED_API_KEY", "x")
    out = macro_fred.get_macro_regime()
    for key in ("yield_curve_spread", "kill_switch_active", "regime_note"):
        assert key in out


# ── Insider flow ─────────────────────────────────────────────────────────────

FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{director}</isDirector><isOfficer>{officer}</isOfficer>
      <isTenPercentOwner>{ten}</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _form4(code, shares=100, price=50, owner="JANE DOE", officer=1, director=0, ten=0, ad="A"):
    return FORM4.format(code=code, shares=shares, price=price, owner=owner,
                        officer=officer, director=director, ten=ten, ad=ad)


def test_parses_an_open_market_purchase():
    import insider_flow
    txs = insider_flow._parse_form4(_form4("P", shares=500, price=20))
    assert txs == [("P", 500.0, 20.0, "JANE DOE", True)]


def test_ten_percent_owners_are_not_counted_as_informed():
    """A 10% holder is usually an index fund or a bank's own broker-dealer, not
    someone with a view."""
    import insider_flow
    txs = insider_flow._parse_form4(_form4("P", officer=0, director=0, ten=1))
    assert txs[0][4] is False


def test_compensation_codes_are_distinguishable_from_decisions():
    """A/M/F/G are machinery. The caller filters on the code, so the parser must
    surface it faithfully rather than lumping everything into 'acquired'."""
    import insider_flow
    for code in ("A", "M", "F", "G"):
        assert insider_flow._parse_form4(_form4(code))[0][0] == code


def test_malformed_xml_returns_empty_not_raise():
    import insider_flow
    assert insider_flow._parse_form4("<not-xml") == []
    assert insider_flow._parse_form4("") == []


def test_insider_flow_off_by_default(monkeypatch):
    monkeypatch.delenv("INSIDER_FLOW", raising=False)
    import insider_flow
    assert insider_flow.enabled() is False
    assert insider_flow.enrich([{"symbol": "AAA"}]) == 0


# ── IV metrics ───────────────────────────────────────────────────────────────

def test_iv_rank_is_none_until_enough_history():
    """Honest bootstrapping: a rank from four observations is worse than no rank."""
    import iv_metrics
    assert iv_metrics._rank([20.0] * 10, 25.0) is None
    assert iv_metrics._rank([20.0] * 39, 25.0) is None
    assert iv_metrics._rank(list(range(40)), 39) == 100.0


def test_iv_rank_positions_within_the_trailing_range():
    import iv_metrics
    hist = [10.0] * 20 + [30.0] * 20        # range 10..30
    assert iv_metrics._rank(hist, 20.0) == 50.0
    assert iv_metrics._rank(hist, 10.0) == 0.0
    assert iv_metrics._rank(hist, 30.0) == 100.0


def test_iv_rank_handles_a_flat_history():
    import iv_metrics
    assert iv_metrics._rank([15.0] * 50, 15.0) == 50.0


def test_atm_iv_and_skew():
    import iv_metrics
    rows = [
        {"type": "call", "strike": 100, "iv": 0.30, "delta": 0.50},
        {"type": "put",  "strike": 100, "iv": 0.32, "delta": -0.50},
        {"type": "put",  "strike": 90,  "iv": 0.40, "delta": -0.25},
        {"type": "call", "strike": 110, "iv": 0.28, "delta": 0.25},
    ]
    out = iv_metrics._atm_iv_and_skew(rows, spot=100)
    assert out["atm_iv"] == pytest.approx(31.0)        # (30 + 32) / 2
    assert out["skew_25d"] == pytest.approx(12.0)      # 40 − 28, in vol points


def test_iv_metrics_off_by_default(monkeypatch):
    monkeypatch.delenv("IV_METRICS", raising=False)
    import iv_metrics
    assert iv_metrics.enabled() is False
    assert iv_metrics.enrich([{"symbol": "AAA", "price": 100}]) == 0


def test_atm_iv_handles_an_empty_chain():
    import iv_metrics
    assert iv_metrics._atm_iv_and_skew([], 100) == {"atm_iv": None, "skew_25d": None}


def test_etfs_are_skipped_not_fetched(ec, monkeypatch):
    """An ETF has no earnings; fetching one wastes a call and logs a 404 that
    looks like a failure."""
    called = []
    monkeypatch.setattr(ec, "_fetch", lambda sym: called.append(sym) or {})
    sigs = [{"symbol": "SPY"}, {"symbol": "AAPL"}]
    ec.enrich(sigs, skip={"SPY"})
    assert called == ["AAPL"]
    assert sigs[0]["earnings_blackout"] is False
    assert sigs[0]["earnings_date"] is None


def test_fred_errors_never_leak_the_api_key():
    """requests puts the full request URL in its exception text, and FRED takes
    the key as a query param — so an unsanitised print posts the secret into
    Railway's retained logs. Caught for real: a 502 from FRED printed the key."""
    import macro_fred
    leaky = ("502 Server Error: Bad Gateway for url: "
             "https://api.stlouisfed.org/fred/series/observations"
             "?series_id=NFCI&api_key=abc123secret&file_type=json")
    safe = macro_fred._safe(Exception(leaky))
    assert "abc123secret" not in safe
    assert "api_key=***" in safe
    assert "series_id=NFCI" in safe          # everything else stays diagnosable


def test_a_failed_fetch_is_never_cached(ec, monkeypatch):
    """Caught live: PAGS hit a transient parse error, was cached empty for the
    day, and reported earnings THAT SAME DAY with earnings_blackout=False. A
    cached failure silently disables a NON-NEGOTIABLE rule for 24h."""
    import json
    calls = []

    def flaky(sym):
        calls.append(sym)
        return None if len(calls) == 1 else {"next": _iso(1)}

    monkeypatch.setattr(ec, "_fetch", flaky)

    sigs = [{"symbol": "AAA"}]
    ec.enrich(sigs)
    assert sigs[0]["earnings_unknown"] is True
    assert sigs[0]["earnings_blackout"] is False        # we do not fail closed
    assert "AAA" not in json.loads(ec._cache_path().read_text())   # not cached

    # Next session retries and now finds the print -> blackout engages.
    sigs2 = [{"symbol": "AAA"}]
    ec.enrich(sigs2)
    assert sigs2[0]["earnings_blackout"] is True
    assert sigs2[0]["earnings_unknown"] is False


def test_a_successful_empty_result_IS_cached(ec, monkeypatch):
    """An ETF genuinely has no earnings. That is a successful lookup and must be
    cached, or we re-fetch it every session forever."""
    import json
    calls = []
    monkeypatch.setattr(ec, "_fetch", lambda s: calls.append(s) or {})
    ec.enrich([{"symbol": "ETFX"}])
    ec.enrich([{"symbol": "ETFX"}])
    assert calls == ["ETFX"]                                   # fetched once
    assert "ETFX" in json.loads(ec._cache_path().read_text())
