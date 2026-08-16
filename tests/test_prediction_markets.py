"""Prediction markets: selection quality, and the three ways this data lies.

Phase 2 of docs/TRADINGAGENTS_PLAN.md. Every assertion here traces to something
the FIRST live run got wrong — this module was written against the real
Polymarket API and each filter exists because its absence produced visible junk,
not because it seemed prudent.

No network: the fixtures are shaped like real gamma-API payloads, including the
detail that `outcomes` and `outcomePrices` arrive as JSON *strings*.
"""
import json

import pytest

import prediction_markets as pm


def _market(question, prob, liq=100_000, days=30, outcomes=("Yes", "No")):
    from datetime import datetime, timedelta, timezone
    end = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    return {
        "question": question,
        # The API returns these as JSON strings, not lists. Parsing them as lists
        # yields silent empty results, which looks identical to "no markets".
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps([str(prob), str(round(1 - prob, 4))]),
        "liquidityNum": liq,
        "endDate": end,
    }


def _events(markets):
    return [{"title": "t", "markets": markets}]


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("PREDICTION_MARKETS", "true")


def _fetch_with(monkeypatch, markets, tags=("fed",)):
    monkeypatch.setattr(pm, "_fetch_tag", lambda tag: _events(markets))
    return pm.fetch(tags=tags)


# ── the flag ─────────────────────────────────────────────────────────────────

def test_disabled_returns_nothing_without_calling_the_api(monkeypatch):
    monkeypatch.setenv("PREDICTION_MARKETS", "false")

    def boom(tag):
        raise AssertionError("must not hit the network when disabled")

    monkeypatch.setattr(pm, "_fetch_tag", boom)
    assert pm.fetch() == []


def test_api_failure_is_fail_open(monkeypatch):
    """macro_context must still produce a brief when Polymarket is down."""
    def boom(tag):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(pm, "_fetch_tag", boom)
    assert pm.fetch(tags=("fed", "oil")) == []


# ── lie #1: thin markets ─────────────────────────────────────────────────────

def test_thin_markets_are_dropped(monkeypatch):
    rows = _fetch_with(monkeypatch, [
        _market("Deep question?", 0.4, liq=500_000),
        _market("Thin question?", 0.4, liq=100),
    ])
    assert [r["question"] for r in rows] == ["Deep question?"]


def test_liquidity_floor_is_configurable(monkeypatch):
    monkeypatch.setenv("PREDICTION_MIN_LIQUIDITY", "600000")
    rows = _fetch_with(monkeypatch, [_market("Deep question?", 0.4, liq=500_000)])
    assert rows == []


# ── lie #2: near-certain rows crowd out live ones ────────────────────────────

def test_resolved_in_all_but_name_rows_are_dropped(monkeypatch):
    """One FOMC meeting lists a market per outcome bucket; most sit at ~0%.
    On the first live run those consumed 6 of 12 slots while a 16% chance of a
    US-Iran war fell below the cut."""
    rows = _fetch_with(monkeypatch, [
        _market("Will 11 Fed rate cuts happen?", 0.0),
        _market("Will the Fed hold rates?", 0.74),
        _market("Certain thing?", 0.995),
    ])
    assert [r["question"] for r in rows] == ["Will the Fed hold rates?"]


def test_small_tail_risks_survive(monkeypatch):
    """4% on an invasion is a real tail risk — the filter must not eat it."""
    rows = _fetch_with(monkeypatch, [_market("Will China invade Taiwan?", 0.04)])
    assert len(rows) == 1


# ── lie #3: false-positive topic matching ────────────────────────────────────

def test_theme_matching_uses_word_boundaries():
    """The first cut matched `war` inside "Warnock" and pulled a Senate primary
    in as a geopolitical risk."""
    assert pm._theme("Will Raphael Warnock win the nomination?") == "other"
    assert pm._theme("Will the U.S. invade Iran before 2027?") == "geopolitics"


@pytest.mark.parametrize("q,expected", [
    ("Will the U.S. invade Iran before 2027?", "geopolitics"),
    ("Will the Iranian regime fall before 2027?", "geopolitics"),
    ("Putin out as President of Russia by December 31?", "geopolitics"),
    ("Will China invade Taiwan by end of 2026?", "geopolitics"),
    ("Strait of Hormuz traffic returns to normal?", "energy"),
    ("Will there be no change in Fed interest rates?", "rates"),
    ("Will CPI inflation exceed 3%?", "inflation"),
    ("Will the US enter a recession in 2026?", "growth"),
    ("Will new tariffs be imposed on China?", "trade"),
])
def test_live_run_questions_classify_correctly(q, expected):
    """Every one of these was mislabelled `other` by the first implementation."""
    assert pm._theme(q) == expected


# ── selection shape ──────────────────────────────────────────────────────────

def test_one_liquid_theme_cannot_crowd_out_the_rest(monkeypatch):
    fed = [_market(f"Fed question {i}?", 0.5, liq=900_000 - i) for i in range(8)]
    geo = [_market("Will China invade Taiwan?", 0.04, liq=10_000)]
    rows = _fetch_with(monkeypatch, fed + geo)
    themes = [r["theme"] for r in rows]
    assert themes.count("rates") <= pm.MAX_PER_THEME
    assert "geopolitics" in themes


def test_unclassified_rows_are_capped_not_dropped(monkeypatch):
    """`other` is demoted to one slot — a Nobel Peace Prize market rode in on the
    geopolitics tag — but not banned, or a genuinely new macro topic we never
    thought to name would vanish silently."""
    noise = [_market(f"Unrelated thing {i}?", 0.5, liq=900_000 - i) for i in range(5)]
    rows = _fetch_with(monkeypatch, noise)
    assert len([r for r in rows if r["theme"] == "other"]) == 1


def test_long_dated_markets_are_excluded(monkeypatch):
    rows = _fetch_with(monkeypatch, [_market("Something by 2031?", 0.5, days=2000)])
    assert rows == []


def test_non_binary_and_non_yes_markets_are_skipped(monkeypatch):
    rows = _fetch_with(monkeypatch, [
        _market("Three-way?", 0.4, outcomes=("A", "B", "C")),
        _market("Not yes-first?", 0.4, outcomes=("Up", "Down")),
    ])
    assert rows == []


def test_duplicate_questions_across_tags_appear_once(monkeypatch):
    monkeypatch.setattr(pm, "_fetch_tag", lambda tag: _events([_market("Same Q?", 0.5)]))
    assert len(pm.fetch(tags=("fed", "economy", "oil"))) == 1


# ── downstream helpers ───────────────────────────────────────────────────────

def test_notable_does_not_claim_a_direction():
    """It selects on topic and materiality only. "Traffic returns to normal —
    44%" is good news and still qualifies; calling these "risks" (as the first
    version did) asserted a polarity the data does not carry."""
    rows = [
        {"question": "Strait of Hormuz traffic returns to normal?", "prob_yes": 0.44,
         "theme": "energy"},
        {"question": "Will the U.S. invade Iran?", "prob_yes": 0.16, "theme": "geopolitics"},
        {"question": "Will the Fed hold?", "prob_yes": 0.74, "theme": "rates"},
        {"question": "Tiny tail?", "prob_yes": 0.02, "theme": "geopolitics"},
    ]
    got = [r["question"] for r in pm.notable(rows)]
    assert "Will the U.S. invade Iran?" in got          # 16% clears the 15% bar
    assert "Strait of Hormuz traffic returns to normal?" in got
    assert "Will the Fed hold?" not in got              # rates is not a risk theme
    assert "Tiny tail?" not in got


def test_summary_is_safe_on_empty():
    assert pm.summary([]) == "Not available"


def test_summary_always_shows_liquidity():
    """A price without the money behind it invites reading a thin market as a
    forecast — the whole reason MIN_LIQUIDITY exists."""
    line = pm.summary([{"question": "Q?", "prob_yes": 0.44, "liquidity": 320194,
                        "days_to_resolve": 136, "theme": "energy"}])
    assert "44%" in line and "320,194" in line and "136d" in line
