"""
macro_fred.py — Fetch critical economic data from the Federal Reserve Economic Data (FRED) API.

Fetches:
1. T10Y2Y  10Y−2Y Treasury spread (yield curve inversion = recession signal)
2. T10Y3M  10Y−3M spread — the version that actually leads recessions best
3. BAMLH0A0HYM2  US high-yield option-adjusted spread (credit stress)
4. NFCI    Chicago Fed National Financial Conditions Index (>0 = tighter than average)

WHY THE EXTRA SERIES
--------------------
Our own dip-confidence model ranks VIX/regime as its single most predictive
feature — so regime is the one place where more data has already proven it pays.
VIX is an equity-only, spot measure of fear. Credit spreads and financial
conditions are broader and slower-moving, and credit typically leads equity
drawdowns: HY spreads widen while the index is still near highs.

The LEVEL of a spread mostly tells you where we are in the cycle. The CHANGE
tells you what is happening now, which is what a trading session needs — so
every series carries a 20-observation delta alongside its latest value.

All of it is free; FRED_API_KEY is the only requirement, and every field
degrades to None without it.

Output is consumed by macro_context.py to inform the overall risk level and 'Macro Kill-Switch'.
"""
import json
import os
import re
import sys
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
import time

warnings.filterwarnings("ignore")

import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

_KEY_RE = re.compile(r"(api_key=)[^&\s]+")


def _safe(err) -> str:
    """Error text with the API key stripped.

    requests puts the full request URL in its exception messages, and FRED takes
    the key as a query parameter — so a plain `print(e)` on any HTTP error posts
    the secret straight into stderr, which Railway captures into logs that are
    retained and searchable. Never log a raw requests error from this module.
    """
    return _KEY_RE.sub(r"\1***", str(err))

def fetch_fred_series(api_key: str, series_id: str, days_back: int = 60) -> dict:
    """Fetch recent observations for a specific FRED series."""
    start_date = (datetime.now(ET) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "desc",
        "limit": 10
    }
    
    try:
        response = requests.get(FRED_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Filter out empty observations (represented by '.' in FRED API)
        valid_obs = [obs for obs in data.get("observations", []) if obs.get("value") != "."]
        
        if valid_obs:
            latest = valid_obs[0]
            return {
                "date": latest["date"],
                "value": float(latest["value"])
            }
    except Exception as e:
        print(f"Error fetching {series_id} from FRED: {_safe(e)}", file=sys.stderr)
    
    return None

def fetch_fred_history(api_key: str, series_id: str, days_back: int = 120,
                       limit: int = 40) -> list:
    """Recent observations, oldest→newest, as [(date, value)]. [] on failure."""
    start_date = (datetime.now(ET) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": start_date, "sort_order": "desc", "limit": limit,
    }
    try:
        r = requests.get(FRED_API_URL, params=params, timeout=10)
        r.raise_for_status()
        obs = [o for o in r.json().get("observations", []) if o.get("value") != "."]
        return [(o["date"], float(o["value"])) for o in reversed(obs)]
    except Exception as e:
        print(f"Error fetching {series_id} history from FRED: {_safe(e)}", file=sys.stderr)
        return []


def _latest_and_change(series: list, lookback: int = 20) -> tuple:
    """(latest, change over `lookback` observations). (None, None) if unavailable."""
    if not series:
        return None, None
    latest = series[-1][1]
    if len(series) <= lookback:
        return latest, None
    return latest, round(latest - series[-1 - lookback][1], 3)


def get_credit_regime(api_key: str = None) -> dict:
    """Credit and financial-conditions read. Every field degrades to None."""
    api_key = api_key or os.getenv("FRED_API_KEY")
    out = {
        "hy_oas": None, "hy_oas_change_20d": None,
        "nfci": None, "nfci_change_20d": None,
        "t10y3m": None,
        "credit_stress": "unknown", "credit_note": "FRED unavailable",
    }
    if not api_key:
        return out

    hy = fetch_fred_history(api_key, "BAMLH0A0HYM2")
    nfci = fetch_fred_history(api_key, "NFCI", days_back=240)
    t3m = fetch_fred_history(api_key, "T10Y3M")

    out["hy_oas"], out["hy_oas_change_20d"] = _latest_and_change(hy)
    # NFCI is weekly, so 20 observations is ~5 months — use 4 for a ~1-month read.
    out["nfci"], out["nfci_change_20d"] = _latest_and_change(nfci, lookback=4)
    out["t10y3m"] = t3m[-1][1] if t3m else None

    lvl, chg, nfci_lvl = out["hy_oas"], out["hy_oas_change_20d"], out["nfci"]
    if lvl is None:
        return out

    # Widening matters more than level: spreads gapping out from a low base is
    # the classic pre-drawdown tell, while a high-but-stable spread is priced in.
    if (chg is not None and chg >= 1.00) or lvl >= 8.0:
        out["credit_stress"] = "severe"
    elif (chg is not None and chg >= 0.50) or lvl >= 5.5:
        out["credit_stress"] = "elevated"
    elif (chg is not None and chg <= -0.30) and lvl < 4.0:
        out["credit_stress"] = "easing"
    else:
        out["credit_stress"] = "normal"

    chg_txt = f"{chg:+.2f}pp over 20 obs" if chg is not None else "change n/a"
    nfci_txt = f", NFCI {nfci_lvl:+.2f}" if nfci_lvl is not None else ""
    out["credit_note"] = (f"HY OAS {lvl:.2f}% ({chg_txt}){nfci_txt} — "
                          f"credit {out['credit_stress']}.")
    return out


def get_macro_regime() -> dict:
    """Determine the macro regime using FRED data."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not found in environment. Returning default regime.", file=sys.stderr)
        return {
            "yield_curve_spread": None,
            "kill_switch_active": False,
            "regime_note": "FRED data unavailable (no API key)"
        }

    # Fetch Yield Curve (10Y-2Y Spread)
    # A negative value indicates an inverted yield curve (recession warning)
    yc_data = fetch_fred_series(api_key, "T10Y2Y")
    
    spread = yc_data["value"] if yc_data else None
    
    # Kill-Switch Logic: Severe inversion is a powerful recession lead indicator
    kill_switch = False
    note = "Macro regime stable."
    
    if spread is not None:
        if spread < -0.5:
            kill_switch = True
            note = f"SEVERE YIELD CURVE INVERSION ({spread}%). Macro Kill-Switch ACTIVE."
        elif spread < 0:
            note = f"Yield curve inverted ({spread}%). Cautious stance recommended."
        else:
            note = f"Yield curve normal ({spread}%)."
            
    # Credit is additive context, never a kill-switch on its own: the kill-switch
    # halts all buying, and a single widening print is not grounds for that.
    credit = get_credit_regime(api_key)

    return {
        "yield_curve_spread": spread,
        "kill_switch_active": kill_switch,
        "regime_note": note,
        "last_updated": datetime.now(ET).isoformat(),
        **credit,
    }

if __name__ == "__main__":
    regime = get_macro_regime()
    print(json.dumps(regime, indent=2))
