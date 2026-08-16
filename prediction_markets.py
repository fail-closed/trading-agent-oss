"""
prediction_markets.py — forward-looking macro probabilities from Polymarket.

WHY THIS EXISTS
---------------
Look at what already feeds the macro read: VIX, yield curve, HY spreads, price
changes, headlines. Every one is either a transform of today's prices or a
description of something that already happened. `build_economic_calendar()` is
the only forward-looking piece and it is a hardcoded table of *approximate*
release dates — it knows an FOMC meeting is coming, not what the market thinks
will happen at it.

A prediction market price IS a forward-looking probability, produced by people
with money at stake. On 2026-08-16 "How many Fed rate cuts in 2026?" priced
**no cuts at 85%** — a rate view our whole stack could otherwise only infer
second-hand from the 10Y-2Y spread. That is genuinely new information, it costs
one unauthenticated HTTP call, and no LLM is involved in producing it.

WHAT IT IS NOT
--------------
Not a trading signal. Not a veto. It is context for the macro assessment, in
the same tier as `credit_note` — it can move `macro_score` and `risk_level`,
which raise or lower the buy threshold. It never names a stock.

THREE WAYS THIS DATA LIES, AND WHAT WE DO ABOUT EACH
----------------------------------------------------
1. **Thin markets are noise.** Anyone can move a $200 book. We drop anything
   below MIN_LIQUIDITY and report the price with its liquidity attached, so a
   number can always be weighed against the money standing behind it.
2. **Volume ranking surfaces junk.** The highest-volume markets on the platform
   are novelty bets ("Will Jesus Christ return before 2027?" — $65M). We select
   by *topic tag*, never by volume, and rank within the topic by liquidity.
3. **Substring keyword matching produces false positives.** The first version of
   this matched `war` inside "**War**nock" and pulled in a Senate primary as a
   geopolitical risk. Topic selection is done by Polymarket's own tags, and the
   one place keywords remain uses word boundaries.

Long-dated markets are also excluded: "before 2030" says nothing about this
month's threshold, and its price barely moves.

No cache, unlike every other enricher here. `macro_context` runs three times a
day (9:00 + two intraday refreshes), so this is 3 unauthenticated calls daily —
below any rate limit, and a cache would add a state file to back up in exchange
for nothing. The prices also move intraday, which is the point of refreshing.

  import prediction_markets
  pm = prediction_markets.fetch()            # [] on any failure
  text = prediction_markets.summary(pm)      # for the macro prompt
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"

# Polymarket's own topic tags. Selecting by tag rather than by keyword is what
# keeps novelty and sports markets out without a blocklist arms race.
DEFAULT_TAGS = ("fed", "inflation", "economy", "recession", "tariffs", "geopolitics", "oil")

MIN_LIQUIDITY = 5_000        # USD in the book; below this a price is one trader's opinion
MAX_DAYS_OUT = 400           # a market resolving past this barely moves on today's news
MAX_MARKETS = 12             # prompt budget — the macro call is already large
MAX_PER_THEME = 4            # see below
PER_TAG_EVENTS = 20
TIMEOUT = 15

# Prices this close to resolved carry no incremental information, and one Fed
# meeting spawns a dozen of them ("Will 11 Fed rate cuts happen in 2026? — 0%").
# Left unfiltered, those tail buckets consumed 6 of 12 slots on the first live
# run while a 16% chance of the US invading Iran sat below the cut. A 4% tail
# risk is worth a slot; a 0% one is not.
CERTAIN_LO, CERTAIN_HI = 0.01, 0.99

# Only used to label a row's theme for the prompt. Word-boundary matched — see
# the Warnock note above.
# NOTE the `s?` suffixes. `\btariff\b` does NOT match "tariffs" — the word
# boundary needs a non-word character after it, and `s` is a word character. A
# test for "Will new tariffs be imposed on China?" caught this labelled
# `geopolitics` (via "China"), because the trade pattern silently never fired.
_THEME_PATTERNS = (
    ("rates", r"\b(fed|rate cuts?|rate hikes?|fomc|powell|interest rates?)\b"),
    ("inflation", r"\b(cpi|inflation|pce|deflation)\b"),
    ("growth", r"\b(recessions?|gdp|unemployment|jobs report|payrolls)\b"),
    ("trade", r"\b(tariffs?|trade deals?|trade war|sanctions?|embargo)\b"),
    ("energy", r"\b(oil|opec|crude|gas prices?|hormuz|pipelines?)\b"),
    # The first live run labelled "Will the U.S. invade Iran before 2027?",
    # "Will China invade Taiwan…", "Putin out as President…" and "Will the
    # Iranian regime fall…" all as `other` — four of the most consequential rows
    # in the set, invisible to headline_risks(). Verbs and actors matter as much
    # as the noun "war".
    ("geopolitics", r"\b(war|invade|invasion|ceasefire|strike|nuclear|missile|"
                    r"regime|coup|troops|conflict|annex|blockade|"
                    r"iran|taiwan|russia|ukraine|putin|israel|china)\b"),
)


def enabled() -> bool:
    return os.getenv("PREDICTION_MARKETS", "").strip().lower() in ("1", "true", "yes", "on")


def _tags() -> tuple:
    raw = os.getenv("PREDICTION_MARKET_TAGS", "").strip()
    return tuple(t.strip() for t in raw.split(",") if t.strip()) if raw else DEFAULT_TAGS


def _min_liquidity() -> float:
    try:
        return float(os.getenv("PREDICTION_MIN_LIQUIDITY", MIN_LIQUIDITY))
    except ValueError:
        return MIN_LIQUIDITY


def _theme(question: str) -> str:
    q = (question or "").lower()
    for name, pat in _THEME_PATTERNS:
        if re.search(pat, q):
            return name
    return "other"


def _as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_prices(raw):
    """outcomePrices arrives as a JSON *string* ('["0.85", "0.15"]'), not a list."""
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    return [_as_float(p, None) for p in raw] if isinstance(raw, list) else []


def _days_out(end_iso: str):
    if not end_iso:
        return None
    try:
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        return (end - datetime.now(timezone.utc)).days
    except Exception:
        return None


def _fetch_tag(tag: str) -> list:
    import requests
    r = requests.get(GAMMA_EVENTS, timeout=TIMEOUT, params={
        "closed": "false", "limit": PER_TAG_EVENTS, "tag_slug": tag,
        "order": "volume", "ascending": "false",
    })
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def fetch(tags: tuple = None) -> list:
    """Liquid, near-dated macro markets as [{question, prob_yes, ...}]. [] on failure.

    Fail-open by construction: macro_context must produce a brief even when a
    third-party API is down, so every error path returns what we have so far."""
    if not enabled():
        return []
    rows, seen = [], set()
    floor = _min_liquidity()
    for tag in (tags or _tags()):
        try:
            events = _fetch_tag(tag)
        except Exception as e:
            print(f"  [prediction_markets] {tag}: {str(e)[:80]}", file=sys.stderr)
            continue
        for ev in events:
            for m in (ev.get("markets") or []):
                q = (m.get("question") or "").strip()
                if not q or q.lower() in seen:
                    continue
                prices = _parse_prices(m.get("outcomePrices"))
                # Binary Yes/No only. A multi-outcome market's first price is not
                # a probability of anything nameable in one line.
                outcomes = m.get("outcomes")
                if isinstance(outcomes, str):
                    import json
                    try:
                        outcomes = json.loads(outcomes)
                    except Exception:
                        outcomes = None
                if not (isinstance(outcomes, list) and len(outcomes) == 2
                        and str(outcomes[0]).lower() == "yes"):
                    continue
                if len(prices) != 2 or prices[0] is None:
                    continue
                if not (CERTAIN_LO < prices[0] < CERTAIN_HI):
                    continue
                liq = _as_float(m.get("liquidityNum") or m.get("liquidity"))
                if liq < floor:
                    continue
                days = _days_out(m.get("endDate") or ev.get("endDate"))
                if days is None or days < 0 or days > MAX_DAYS_OUT:
                    continue
                seen.add(q.lower())
                rows.append({
                    "question": q,
                    "prob_yes": round(prices[0], 3),
                    "liquidity": round(liq),
                    "days_to_resolve": days,
                    "theme": _theme(q),
                    "tag": tag,
                })
    rows.sort(key=lambda r: r["liquidity"], reverse=True)
    return _diversify(rows)


def _diversify(rows: list, cap: int = MAX_PER_THEME, total: int = MAX_MARKETS) -> list:
    """Take the most liquid `cap` per theme before filling the rest by liquidity.

    Straight liquidity ranking is dominated by whichever topic happens to be
    liquid today — a single FOMC meeting lists one market per outcome bucket, all
    of them deep, and they crowd out every other theme. The macro read wants
    breadth across themes more than the 5th-most-liquid rates market."""
    kept, seen_per_theme = [], {}
    for r in rows:
        t = r["theme"]
        # `other` = matched a macro TAG but none of our theme patterns. Usually
        # adjacent-but-irrelevant (a Nobel Peace Prize market rode in on the
        # geopolitics tag). Capped at one rather than dropped: the patterns are
        # a labelling aid, and making them the inclusion test would silently hide
        # any genuinely new macro topic we hadn't thought to name.
        limit = 1 if t == "other" else cap
        if seen_per_theme.get(t, 0) < limit:
            seen_per_theme[t] = seen_per_theme.get(t, 0) + 1
            kept.append(r)
    # The cap is HARD — there is deliberately no backfill. An earlier version
    # topped the list back up to `total` by liquidity, which quietly undid the
    # whole point: 8 deep Fed buckets refilled every slot the cap had just freed.
    # Returning 5 diverse rows beats returning 12 where 8 restate one meeting.
    return kept[:total]


def summary(rows: list) -> str:
    """Compact block for the macro prompt. Liquidity is shown on every line so
    the model can discount a thin market rather than read every price alike."""
    if not rows:
        return "Not available"
    out = []
    for r in rows:
        out.append(f"  [{r['theme']}] {r['question']} — {r['prob_yes'] * 100:.0f}% "
                   f"(liquidity ${r['liquidity']:,}, resolves in {r['days_to_resolve']}d)")
    return "\n".join(out)


NOTABLE_THEMES = ("growth", "geopolitics", "trade", "energy")
NOTABLE_THRESHOLD = 0.15


def notable(rows: list, threshold: float = NOTABLE_THRESHOLD) -> list:
    """Rows worth a human glance in the session log. **Direction is NOT implied.**

    This was `headline_risks()` and returned "risks" until the first live run
    made the flaw obvious: it flagged *"Strait of Hormuz traffic returns to
    normal by December 31 — 44%"* as a risk, when a high probability there is
    good news. Prediction-market questions have arbitrary polarity — "will
    <bad thing> happen" and "will <good thing> resume" both read as `prob_yes`,
    and nothing short of parsing the sentence tells them apart. So this function
    no longer claims to know which is which; it selects on topic and materiality
    only, and the reader (or the macro LLM, which sees the full question text)
    supplies the direction.

    15%, not 30%: tail events rarely price high even when they dominate the room.
    At 30% this returned nothing while the set held a 16% chance of the US
    invading Iran. `energy` is in scope because the watchlist holds BKR/VLO/PSX."""
    return [r for r in rows
            if r["theme"] in NOTABLE_THEMES and r["prob_yes"] >= threshold]


if __name__ == "__main__":
    os.environ.setdefault("PREDICTION_MARKETS", "true")
    got = fetch()
    print(f"{len(got)} market(s)\n")
    print(summary(got))
