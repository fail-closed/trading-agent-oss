"""
macro_context.py — daily macro intelligence brief.

Captures ALL major market-moving factors:
  News:        S&P, VIX, Gold, Oil, USD, Treasuries, sector ETFs, currencies,
               credit markets, defense, geopolitical proxies
  Market data: Yield curve (2Y-10Y spread), currency pairs, credit spreads,
               sector momentum, put/call proxy
  Calendar:    FOMC, CPI, NFP, PPI, PCE, GDP, Retail Sales, Fed speakers,
               earnings for watchlist + major index movers

Run at 9:00 AM ET daily. Outputs signals/YYYY-MM-DD_macro.json.
"""
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import anthropic
from macro_fred import get_macro_regime

load_dotenv()


# ── Expanded news sources ─────────────────────────────────────────────────────

MACRO_TICKERS = {
    # Broad market & fear
    "^GSPC":    "S&P 500 (broad market)",
    "^VIX":     "Volatility / fear index",
    # Safe havens
    "GC=F":     "Gold (safe haven / inflation hedge)",
    "^TNX":     "10-Year Treasury yield (rates)",
    "^IRX":     "2-Year Treasury (Fed expectations)",
    # Risk assets
    "CL=F":     "Oil WTI (geopolitical / growth)",
    "NG=F":     "Natural gas (energy / Russia-Europe)",
    # Currencies — critical for carry trades and risk sentiment
    "EURUSD=X": "EUR/USD (dollar strength / EU economy)",
    "JPY=X":    "USD/JPY (yen carry trade — unwind = risk-off)",
    "CNY=X":    "USD/CNH (China slowdown signal)",
    # Credit markets — early warning system
    "HYG":      "High yield bonds (credit stress / risk appetite)",
    "LQD":      "Investment grade bonds (rate sensitivity)",
    # Dollar
    "DX-Y.NYB": "US Dollar index",
}

SECTOR_TICKERS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLV": "Healthcare",
    "XLU": "Utilities",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "ITA": "Defense / Aerospace ETF",
    "SOXX": "Semiconductors ETF",
    "GLD": "Gold ETF (geopolitical hedge)",
    "TLT": "Long-term Treasuries (rate risk)",
}


# ── Market data snapshot ─────────────────────────────────────────────────────

def fetch_market_data() -> dict:
    """Fetch key market metrics that signal macro regime."""
    data = {}

    def get_price_chg(ticker):
        try:
            fi = yf.Ticker(ticker).fast_info
            p, c = fi.last_price, fi.previous_close
            return round(p, 4), round((p-c)/c*100, 3) if c else 0
        except Exception:
            return None, None

    tickers = {
        "sp500":     ("^GSPC",    "S&P 500"),
        "nasdaq":    ("^IXIC",    "NASDAQ"),
        "vix":       ("^VIX",     "VIX"),
        "yield_10y": ("^TNX",     "10Y Yield"),
        "yield_2y":  ("^IRX",     "2Y Yield"),
        "gold":      ("GC=F",     "Gold"),
        "oil":       ("CL=F",     "Oil WTI"),
        "usd":       ("DX-Y.NYB", "USD Index"),
        "eurusd":    ("EURUSD=X", "EUR/USD"),
        "usdjpy":    ("JPY=X",    "USD/JPY"),
        "hyg":       ("HYG",      "High Yield Bonds"),
        "tlt":       ("TLT",      "Long Bonds"),
        # Global equity indices — Asia/Europe trade BEFORE the US open, so their
        # session sets the overnight risk tone the US tends to follow into the cash
        # open (the "global lead" angle). US-listed ETF proxies are used for Asia so
        # the % change is fresh same-session rather than a stale prior local close.
        "eu_stoxx":  ("^STOXX50E", "Euro Stoxx 50 (Europe, same-day)"),
        "uk_ftse":   ("^FTSE",     "FTSE 100 (UK, same-day)"),
        "japan_ewj": ("EWJ",       "Japan (EWJ, US-listed proxy)"),
        "china_fxi": ("FXI",       "China large-cap (FXI, US-listed proxy)"),
        "em_eem":    ("EEM",       "Emerging markets (EEM, US-listed proxy)"),
    }

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(get_price_chg, v[0]): (k, v[1]) for k, v in tickers.items()}
        for fut in as_completed(futures):
            key, label = futures[fut]
            price, chg = fut.result()
            if price is not None:
                data[key] = {"label": label, "price": price, "chg_pct": chg}

    # Yield curve: 2Y-10Y spread (negative = inverted = recession warning)
    y2 = data.get("yield_2y", {}).get("price", 0)
    y10= data.get("yield_10y", {}).get("price", 0)
    if y2 and y10:
        spread = round(y10 - y2, 3)
        data["yield_curve"] = {
            "spread_10y_2y": spread,
            "signal": "inverted" if spread < 0 else ("flat" if spread < 0.5 else "normal"),
        }

    # Credit stress: HYG price decline = credit tightening = risk-off
    hyg = data.get("hyg", {}).get("chg_pct", 0)
    data["credit_stress"] = "elevated" if hyg and hyg < -0.5 else "normal"

    # Sector momentum: compute relative strength of key sector ETFs vs SPY
    sector_strength = {}
    spy_chg = data.get("sp500", {}).get("chg_pct", 0)
    with ThreadPoolExecutor(max_workers=8) as ex:
        sec_tickers = {"XLK":"Tech","XLE":"Energy","XLF":"Financials",
                       "XLI":"Industrials","XLV":"Healthcare","ITA":"Defense"}
        futures = {ex.submit(get_price_chg, tk): (tk, nm)
                   for tk, nm in sec_tickers.items()}
        for fut in as_completed(futures):
            tk, nm = futures[fut]
            _, chg = fut.result()
            if chg is not None and spy_chg is not None:
                sector_strength[nm] = round(chg - spy_chg, 3)  # relative to SPY

    data["sector_relative_strength"] = sector_strength
    return data


# ── Earnings calendar ─────────────────────────────────────────────────────────

# Major S&P 500 stocks whose earnings move entire sectors
MAJOR_EARNERS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","INTC",
    "JPM","BAC","GS","MS","WFC",
    "JNJ","PFE","LLY","UNH",
    "XOM","CVX","COP",
    "CAT","BA","RTX","LMT",
]

def fetch_earnings_calendar(watchlist_symbols: list, days_ahead: int = 10) -> list:
    """
    Fetch upcoming earnings dates for our positions AND major market movers.
    Returns list of {symbol, date, days_away} sorted by date.
    """
    all_syms = list(set(watchlist_symbols + MAJOR_EARNERS))
    earnings  = []

    def check_earnings(sym):
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None or cal.empty:
                return None
            # Earnings Date is typically the first column
            dates = cal.get("Earnings Date")
            if dates is None:
                return None
            for dt in (dates if hasattr(dates, '__iter__') else [dates]):
                if hasattr(dt, 'date'):
                    ev_date = dt.date()
                elif hasattr(dt, 'to_pydatetime'):
                    ev_date = dt.to_pydatetime().date()
                else:
                    continue
                delta = (ev_date - datetime.now(ET).date()).days
                if 0 <= delta <= days_ahead:
                    return {"symbol": sym, "date": str(ev_date), "days_away": delta}
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(check_earnings, s): s for s in all_syms}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                result["is_watchlist"] = fut.result()["symbol"] in watchlist_symbols
                earnings.append(result)

    return sorted(earnings, key=lambda x: x["days_away"])


# ── Economic calendar ─────────────────────────────────────────────────────────
# Key recurring events that historically move markets 1-3%

def build_economic_calendar() -> list:
    """Generate economic events for 2026 based on typical release schedules."""
    today = datetime.now(ET)
    year  = today.year
    events = []

    # FOMC meetings 2026 (approx)
    fomc_dates = [(1,28),(3,18),(5,6),(6,17),(7,29),(9,16),(11,4),(12,9)]
    for m, d in fomc_dates:
        try:
            ev = datetime(year, m, d).date()
            delta = (ev - today.date()).days
            if 0 <= delta <= 14:
                events.append({"name": "FOMC Meeting / Rate Decision",
                                "date": str(ev), "days_away": delta,
                                "impact": "rates/financials/growth/utilities",
                                "guidance": "Avoid adding rate-sensitive positions 2 days before"})
        except ValueError: pass

    # CPI — typically 2nd or 3rd week of each month
    cpi_dates = [(1,15),(2,12),(3,12),(4,10),(5,13),(6,11),(7,15),(8,13),(9,10),(10,14),(11,12),(12,10)]
    for m, d in cpi_dates:
        try:
            ev = datetime(year, m, d).date()
            delta = (ev - today.date()).days
            if 0 <= delta <= 7:
                events.append({"name": "CPI Inflation Report",
                                "date": str(ev), "days_away": delta,
                                "impact": "inflation/consumer/rates/bonds",
                                "guidance": "Higher CPI = risk-off; hold cash buffer"})
        except ValueError: pass

    # NFP — first Friday of each month
    for month in range(1, 13):
        try:
            first_day = datetime(year, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            nfp_date = (first_day + timedelta(days=days_to_friday)).date()
            delta = (nfp_date - today.date()).days
            if 0 <= delta <= 7:
                events.append({"name": "Non-Farm Payrolls (Jobs Report)",
                                "date": str(nfp_date), "days_away": delta,
                                "impact": "employment/consumer/rates/growth",
                                "guidance": "Strong NFP = Fed hawkish = rate-sensitive stocks fall"})
        except ValueError: pass

    # PCE (Fed's preferred inflation measure) — last Thursday/Friday of month
    for month in range(1, 13):
        try:
            # Approx last Friday of month
            last_day = datetime(year, month, 28)
            days_to_friday = (4 - last_day.weekday()) % 7
            pce_date = (last_day + timedelta(days=days_to_friday)).date()
            delta = (pce_date - today.date()).days
            if 0 <= delta <= 7:
                events.append({"name": "PCE Inflation (Fed's preferred measure)",
                                "date": str(pce_date), "days_away": delta,
                                "impact": "inflation/rates/growth",
                                "guidance": "PCE > 2.5% = Fed stays hawkish"})
        except ValueError: pass

    # Quarterly GDP
    gdp_dates = [(1,30),(4,30),(7,30),(10,30)]
    for m, d in gdp_dates:
        try:
            ev = datetime(year, m, d).date()
            delta = (ev - today.date()).days
            if 0 <= delta <= 7:
                events.append({"name": "GDP (Advance Estimate)",
                                "date": str(ev), "days_away": delta,
                                "impact": "growth/all-sectors",
                                "guidance": "GDP miss = cyclicals sell-off; GDP beat = risk-on"})
        except ValueError: pass

    # US elections / major political events
    political = [(11,3,"US Midterm Elections","all-sectors/defense/healthcare/energy")]
    for m, d, name, impact in political:
        try:
            ev = datetime(year, m, d).date()
            delta = (ev - today.date()).days
            if 0 <= delta <= 30:
                events.append({"name": name, "date": str(ev),
                                "days_away": delta, "impact": impact,
                                "guidance": "Increase cash reserves 1 week before major elections"})
        except ValueError: pass

    # ── Global scheduled events (2026) ───────────────────────────────────────
    # US equities don't trade in isolation — foreign central banks set the rates
    # backdrop for the dollar/carry trades, OPEC sets the oil tape, and major
    # elections/policy meetings drive cross-border risk. Approx 2026 dates;
    # (month, day, name, impact, lead_days, guidance).
    global_events = [
        # ECB Governing Council — euro/USD, EU-exposed multinationals, rate backdrop
        (1,30,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (3,12,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (4,30,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (6,11,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (7,30,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (9,17,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (10,29,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        (12,17,"ECB Rate Decision","rates/EUR-USD/financials/multinationals",2,"ECB surprise moves the dollar → US multinationals & rate-sensitives"),
        # Bank of Japan — yen carry trade; a hawkish BoJ = global risk-off (Aug-2024 unwind)
        (1,23,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (3,19,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (4,28,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (6,16,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (7,31,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (9,18,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (10,30,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        (12,18,"BoJ Rate Decision","JPY-carry/global-risk/tech",2,"Hawkish BoJ unwinds the yen carry trade → broad global risk-off; de-risk tech"),
        # Bank of England — GBP, UK-exposed names
        (2,5,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (3,26,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (5,7,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (6,18,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (8,6,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (9,17,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (11,5,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        (12,17,"BoE Rate Decision","rates/GBP/financials",2,"BoE move shifts GBP and the global rates narrative"),
        # OPEC+ ministerial meetings — set the oil tape → energy sleeve (BKR/VLO/PSX)
        (5,31,"OPEC+ Meeting","oil/energy",3,"OPEC+ supply decision drives crude → energy names (BKR/VLO/PSX)"),
        (11,30,"OPEC+ Meeting","oil/energy",3,"OPEC+ supply decision drives crude → energy names (BKR/VLO/PSX)"),
        # China — NPC plenum / major policy (stimulus → materials, industrials, semis demand)
        (3,5,"China NPC (policy/stimulus)","China-demand/materials/industrials/semis",5,"China stimulus signal lifts materials/industrials & global demand; absence = risk to cyclicals"),
        (10,20,"China Plenum (policy)","China-demand/materials/industrials/semis",5,"China policy shift moves global demand expectations for cyclicals/semis"),
    ]
    for m, d, name, impact, lead, guidance in global_events:
        try:
            ev = datetime(year, m, d).date()
            delta = (ev - today.date()).days
            if 0 <= delta <= lead:
                events.append({"name": name, "date": str(ev), "days_away": delta,
                                "impact": impact, "guidance": guidance, "scope": "global"})
        except ValueError: pass

    return sorted(events, key=lambda x: x["days_away"])


# ── News fetching ──────────────────────────────────────────────────────────────

def fetch_ticker_news(ticker: str, label: str, hours: int = 48) -> list:
    try:
        raw = yf.Ticker(ticker).news or []
        items = []
        for n in raw[:6]:
            content = n.get("content", n)
            title   = content.get("title") or n.get("title", "")
            pub     = content.get("pubDate", "") or ""
            source  = (content.get("provider") or {}).get("displayName", "") or n.get("publisher", "")
            if title:
                items.append({"source_label": label, "title": title,
                               "published": str(pub)[:16], "provider": source})
        return items
    except Exception:
        return []


def fetch_alpaca_news(symbols: list = None, limit: int = 20) -> list:
    """Fetch recent news via the native Alpaca NewsClient.

    Replaces the old `alpaca` CLI subprocess (the CLI binary isn't installed in
    the container). Best-effort: returns [] on any error, so macro news falls back
    to the yfinance ticker-news path. Uses accounts.data_keys() so it works on the
    live project too (which has only ALPACA_LIVE_* keys)."""
    try:
        import accounts
        ak, sk = accounts.data_keys()
    except Exception:
        ak, sk = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
    if not (ak and sk):
        return []
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
        nc = NewsClient(ak, sk)
        kwargs = {"limit": limit, "exclude_contentless": True}
        if symbols:
            kwargs["symbols"] = ",".join(symbols[:10])  # Alpaca limits symbols per call
        resp = nc.get_news(NewsRequest(**kwargs))
        articles = resp.data.get("news", []) if hasattr(resp, "data") else getattr(resp, "news", [])
        out = []
        for a in articles:
            headline = getattr(a, "headline", "") or ""
            if not headline:
                continue
            src = getattr(a, "source", "") or ""
            out.append({
                "source_label": f"Alpaca News ({src})",
                "title":        headline,
                "published":    str(getattr(a, "created_at", ""))[:16],
                "provider":     src,
                "symbols":      getattr(a, "symbols", []) or [],
            })
        return out
    except Exception:
        return []


def fetch_all_macro_news() -> list:
    all_tickers = {**MACRO_TICKERS, **SECTOR_TICKERS}
    headlines   = []
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(fetch_ticker_news, tk, lbl): tk
                   for tk, lbl in all_tickers.items()}
        for fut in as_completed(futures):
            headlines.extend(fut.result())
    seen, dedup = set(), []
    for h in headlines:
        key = h["title"][:60]
        if key not in seen:
            seen.add(key)
            dedup.append(h)
    dedup.sort(key=lambda x: x["published"], reverse=True)
    return dedup[:50]


# ── Claude assessment ─────────────────────────────────────────────────────────

def _loads_lenient(text: str) -> dict:
    """json.loads, tolerant of a prose preamble or ```json fence. See the twin in
    decide.py — a malformed body must degrade, never halt the session."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped.split("\n", 1)[1] if stripped.lower().startswith("json") else stripped
        return json.loads(stripped.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return json.loads(stripped[start:end + 1])
    raise json.JSONDecodeError("no JSON object found in response", text, 0)


MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "macro_score":     {"type": "integer",
                            "description": "-2 very bearish to +2 strongly bullish"},
        "risk_level":      {"type": "string",
                            "enum": ["low","moderate","elevated","high","extreme"]},
        "dominant_themes": {"type": "array", "items": {"type": "string"},
                            "description": "3-5 key themes driving markets today"},
        "sector_impacts": {
            "type": "object",
            "properties": {
                "Technology":    {"type": "string"},
                "Industrials":   {"type": "string"},
                "Financials":    {"type": "string"},
                "Healthcare":    {"type": "string"},
                "Energy":        {"type": "string"},
                "Consumer":      {"type": "string"},
                "Utilities":     {"type": "string"},
                "Real Estate":   {"type": "string"},
                "Defense":       {"type": "string"},
                "Semiconductors":{"type": "string"},
            },
            "required": ["Technology","Industrials","Financials","Healthcare",
                          "Energy","Consumer","Utilities","Real Estate",
                          "Defense","Semiconductors"],
            "additionalProperties": False
        },
        "geopolitical_flags": {"type": "array", "items": {"type": "string"}},
        "global_markets_signal": {"type": "string",
                                  "enum": ["risk_on","neutral","risk_off"],
                                  "description": "Overnight global equity lead (Asia/Europe) for the US open"},
        "global_markets_note": {"type": "string",
                                "description": "1-2 sentences: how Asia/Europe traded overnight and the read-through to the US open & our book"},
        "yield_curve_signal": {"type": "string",
                               "description": "What the yield curve shape signals for equities"},
        "currency_signals":   {"type": "array", "items": {"type": "string"},
                               "description": "Key FX moves and their equity implications"},
        "earnings_watch":     {"type": "array", "items": {"type": "string"},
                               "description": "Upcoming earnings that could move sector ETFs"},
        "trading_guidance":   {"type": "string",
                               "description": "2-3 sentence specific guidance for today's session"},
        "calendar_warnings":  {"type": "array", "items": {"type": "string"}},
        "threshold_adjustment": {"type": "integer",
                                 "description": "0=standard, +1=raise buy threshold, +2=very cautious"},
    },
    "required": ["macro_score","risk_level","dominant_themes","sector_impacts",
                 "geopolitical_flags","global_markets_signal","global_markets_note",
                 "yield_curve_signal","currency_signals",
                 "earnings_watch","trading_guidance","calendar_warnings",
                 "threshold_adjustment"],
    "additionalProperties": False
}

MACRO_SYSTEM = """You are a senior macro analyst for an autonomous equity trading system.
Given comprehensive market data, headlines, and calendar events, produce a daily macro brief.
Be specific about actionable implications for equities.

Take a GLOBAL outlook, not just US: US equities open AFTER Asia and Europe have
already traded, so the overnight global session is a leading tell for the US cash
open. Weigh the global equity block explicitly:
  - Broad-based global weakness (Europe + Asia both red, esp. with a stronger USD
    or hawkish foreign central bank) is a risk-off lead → bias toward caution /
    raising the buy threshold, even if US futures look flat.
  - Broad-based global strength is a risk-on lead → supportive of deploying.
  - DIVERGENCE matters: if Asia/Europe are weak but US is being held up by a few
    mega-caps, treat breadth as poor (fragile rally), not genuine strength.
  - Map regional moves to our book: China/EM weakness → caution on materials,
    industrials, semis (KLAC) & global-demand cyclicals; Europe weakness + strong
    USD → headwind for US multinationals; a hawkish BoJ / yen-carry unwind → broad
    global de-risk (especially tech).
Set global_markets_signal to risk_on / neutral / risk_off and reflect this lead in
macro_score, risk_level, and trading_guidance. Output valid JSON only."""


def assess_macro(headlines: list, events: list, market_data: dict,
                 earnings: list, watchlist: list, fred_regime: dict = None,
                 pred_markets: list = None) -> dict:
    client = anthropic.Anthropic()

    headlines_text = "\n".join(
        f"[{h['source_label']}] {h['title']} ({h['published'][:10]})"
        for h in headlines[:40]
    )
    events_text = "\n".join(
        f"  {e['days_away']}d: {e['name']} ({e['date']}) impacts: {e['impact']}"
        for e in events
    ) or "  None in next 14 days"

    earnings_text = "\n".join(
        f"  {e['days_away']}d: {e['symbol']} earnings{' [OUR POSITION]' if e['symbol'] in watchlist else ''}"
        for e in earnings[:10]
    ) or "  None in next 10 days"

    # FRED Macro Data
    fred_text = "Not available"
    if fred_regime:
        fred_text = (
            f"Yield Spread (10Y-2Y): {fred_regime.get('yield_curve_spread', '?')}% "
            f"| 10Y-3M: {fred_regime.get('t10y3m', '?')}% "
            f"| Status: {fred_regime.get('regime_note', '?')}"
        )
        # Credit leads equity: HY spreads widen while the index is still near
        # highs, so this is an earlier read on risk than VIX gives.
        if fred_regime.get("hy_oas") is not None:
            fred_text += f"\nCredit: {fred_regime.get('credit_note', '')}"

    # Prediction markets (flag: PREDICTION_MARKETS). "Not available" when the
    # flag is off or the API failed — the brief must never depend on a
    # third-party endpoint being up.
    try:
        import prediction_markets
        pm_text = prediction_markets.summary(pred_markets or [])
    except Exception:
        pm_text = "Not available"

    # Format market data summary
    yc  = market_data.get("yield_curve", {})
    md_text = f"""
Market snapshot:
  S&P 500: {market_data.get('sp500',{}).get('chg_pct',0):+.2f}%
  NASDAQ:  {market_data.get('nasdaq',{}).get('chg_pct',0):+.2f}%
  VIX:     {market_data.get('vix',{}).get('price',0):.1f}
  Gold:    {market_data.get('gold',{}).get('chg_pct',0):+.2f}%
  Oil:     {market_data.get('oil',{}).get('chg_pct',0):+.2f}%
  10Y:     {market_data.get('yield_10y',{}).get('price',0):.2f}%
  2Y:      {market_data.get('yield_2y',{}).get('price',0):.2f}%
  Yield curve (10Y-2Y): {yc.get('spread_10y_2y',0):+.3f}% ({yc.get('signal','unknown')})
  EUR/USD: {market_data.get('eurusd',{}).get('chg_pct',0):+.2f}%
  USD/JPY: {market_data.get('usdjpy',{}).get('chg_pct',0):+.2f}%
  HYG:     {market_data.get('hyg',{}).get('chg_pct',0):+.2f}% (credit: {market_data.get('credit_stress','normal')})

Global equity markets (Asia & Europe trade BEFORE the US open — read these as the
overnight risk lead the US tends to follow into the cash open):
  Europe  (Euro Stoxx 50): {market_data.get('eu_stoxx',{}).get('chg_pct',0):+.2f}%
  UK      (FTSE 100):      {market_data.get('uk_ftse',{}).get('chg_pct',0):+.2f}%
  Japan   (EWJ proxy):     {market_data.get('japan_ewj',{}).get('chg_pct',0):+.2f}%
  China   (FXI proxy):     {market_data.get('china_fxi',{}).get('chg_pct',0):+.2f}%
  EM      (EEM proxy):     {market_data.get('em_eem',{}).get('chg_pct',0):+.2f}%

Fundamental Macro (FRED):
  {fred_text}

Sector relative strength vs SPY:
{chr(10).join(f"  {k}: {v:+.2f}%" for k, v in market_data.get("sector_relative_strength", {}).items())}"""

    prompt = f"""Today is {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}.

{md_text}

Recent headlines (last 48h):
{headlines_text}

Upcoming economic events:
{events_text}

Prediction markets (real money on FORWARD outcomes; every other input above
describes the past or today's prices). Treat a price as a probability, weighted
by its liquidity — a thin market is one trader's opinion, not a forecast. These
inform macro_score / risk_level / threshold_adjustment; they never name a stock:
{pm_text}

Upcoming earnings:
{earnings_text}

Our portfolio: Technology (KLAC), Industrials (CAT/PWR/BA), Energy (BKR),
Consumer (EBAY/MNST), Real Estate (SPG), Healthcare (MCK), Utilities (ATO/CNP),
Financials (JPC), 56% cash ready to deploy.

Provide a comprehensive macro brief for today's trading session.
Set threshold_adjustment: 0=standard, 1=raise buy threshold by +1, 2=very cautious (raise by +2)."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        thinking={"type": "disabled"},
        system=MACRO_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": MACRO_SCHEMA}},
        messages=[{"role": "user", "content": prompt}]
    )
    text   = next(b.text for b in response.content if b.type == "text")
    result = _loads_lenient(text)
    result["generated_at"]   = datetime.now(ET).isoformat()
    result["headlines_count"] = len(headlines)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main(watchlist_symbols: list = None):
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if watchlist_symbols is None:
        try:
            import json as _json
            with open("watchlist.json") as f:
                wl = _json.load(f)
            watchlist_symbols = [s["symbol"] for s in wl["stocks"]]
        except Exception:
            watchlist_symbols = []

    print(f"Macro Context — {today}")

    print("Fetching market data snapshot...")
    market_data = fetch_market_data()
    yc = market_data.get("yield_curve", {})
    print(f"  Yield curve: {yc.get('spread_10y_2y',0):+.3f}% ({yc.get('signal','?')})")
    print(f"  Credit stress: {market_data.get('credit_stress','?')}")

    print("Fetching macro headlines...")
    yf_headlines     = fetch_all_macro_news()
    alpaca_headlines = fetch_alpaca_news(limit=20)   # broad market news via Alpaca NewsClient
    headlines        = yf_headlines + alpaca_headlines
    # Deduplicate again after combining sources
    seen, headlines_dedup = set(), []
    for h in sorted(headlines, key=lambda x: x["published"], reverse=True):
        key = h["title"][:60]
        if key not in seen:
            seen.add(key)
            headlines_dedup.append(h)
    headlines = headlines_dedup[:60]
    print(f"  {len(headlines)} headlines ({len(yf_headlines)} yfinance + {len(alpaca_headlines)} Alpaca News)")

    print("Checking economic calendar...")
    events = build_economic_calendar()
    if events:
        print(f"  {len(events)} event(s): {', '.join(e['name'] for e in events[:3])}")

    print("Checking earnings calendar...")
    earnings = fetch_earnings_calendar(watchlist_symbols, days_ahead=10)
    if earnings:
        watchlist_earnings = [e for e in earnings if e["symbol"] in watchlist_symbols]
        print(f"  {len(earnings)} upcoming earnings ({len(watchlist_earnings)} in our portfolio)")
        for e in watchlist_earnings[:3]:
            print(f"    ⚠ {e['symbol']} earnings in {e['days_away']} day(s) — {e['date']}")

    print("Fetching FRED economic data...")
    fred_regime = get_macro_regime()

    pred_markets = []
    try:
        import prediction_markets
        if prediction_markets.enabled():
            pred_markets = prediction_markets.fetch()
            # NOTABLE, not "risks" — see prediction_markets.notable(): question
            # polarity is arbitrary, so a high probability may be good news.
            notable = prediction_markets.notable(pred_markets)
            print(f"  Prediction markets: {len(pred_markets)} liquid macro market(s)"
                  + (f" | NOTABLE: "
                     + "; ".join(f"{r['question'][:44]} {r['prob_yes']*100:.0f}%" for r in notable[:2])
                     if notable else ""))
    except Exception as e:
        print(f"  prediction markets skipped (non-fatal): {str(e)[:90]}")

    brief = assess_macro(headlines, events, market_data, earnings, watchlist_symbols,
                         fred_regime, pred_markets)

    # Force cautious stance if Macro Kill-Switch is active
    if fred_regime.get("kill_switch_active"):
        brief["threshold_adjustment"] = 2
        brief["risk_level"] = "high"
        brief["trading_guidance"] = "⚠ MACRO KILL-SWITCH ACTIVE: " + brief["trading_guidance"]

    # Print summary
    score = brief["macro_score"]
    risk  = brief["risk_level"]
    adj   = brief.get("threshold_adjustment", 0)
    print(f"\n  Score: {score:+d}  |  Risk: {risk}  |  Threshold adj: +{adj}")
    print(f"  Themes: {', '.join(brief['dominant_themes'][:3])}")
    print(f"  Yield curve: {brief.get('yield_curve_signal','')}")
    if brief["geopolitical_flags"]:
        print(f"  Geo flags: {'; '.join(brief['geopolitical_flags'][:2])}")
    if brief["calendar_warnings"]:
        print(f"  Calendar: {'; '.join(brief['calendar_warnings'][:2])}")
    print(f"  Guidance: {brief['trading_guidance']}")

    # Save
    os.makedirs("signals", exist_ok=True)
    out = {
        "date":        today,
        "market_data": market_data,
        "headlines":   headlines,
        "events":      events,
        "earnings":    earnings,
        # Persist the FRED read. It was previously prompt-only, which made the
        # numbers behind a risk call unauditable after the fact and unavailable
        # to the dashboard. Kept under its own key rather than merged into
        # market_data, which already has an unrelated HYG-derived
        # `credit_stress` — two different measures under one name would be worse
        # than not persisting at all.
        "fred_regime": fred_regime,
        # Same reasoning as fred_regime above: persist the inputs behind a risk
        # call, not just the call. Without this, "why was the threshold raised on
        # 08-16?" is unanswerable after the markets reprice — and prediction
        # market prices are gone the moment they move, with no historical API on
        # the free tier to reconstruct them from.
        "prediction_markets": pred_markets,
        "brief":       brief,
    }
    out_path = f"signals/{today}_macro.json"
    from io_utils import write_json_atomic
    write_json_atomic(out_path, out)
    print(f"\nSaved to {out_path}")
    return brief


if __name__ == "__main__":
    main()
