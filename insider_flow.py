"""
insider_flow.py — open-market insider buying and selling from SEC Form 4.

WHY THIS EXISTS
---------------
Every other input we feed the model is a transform of OHLCV, and our own
dip-confidence model tops out near 0.50 AUC on that feature set — the price-
derived well is dry. Form 4 is one of the few free inputs that is orthogonal by
construction: it is not a statistic computed from the price, it is a legally
compelled disclosure of what the people running the company did with their own
money, filed within two business days.

WHAT COUNTS, AND WHAT IS NOISE
------------------------------
Most Form 4 volume is compensation machinery and says nothing about conviction:

    A  grant/award            they were given it
    M  option exercise        mechanical, usually paired with a sale
    F  tax withholding        forced, at whatever price the calendar picked
    G  gift

Only two codes reflect a decision to transact at the market price:

    P  open-market PURCHASE   the informative one
    S  open-market SALE       weakly informative — insiders sell for a hundred
                              reasons (diversification, houses, divorce), which
                              is why the literature finds buying predictive and
                              selling much less so

So we count P and S separately and never net them into one number. The signal
worth acting on is a *cluster* buy: several distinct insiders buying within a
short window, which is far harder to explain away than one person topping up.

Free, but SEC asks for a descriptive User-Agent and ≤10 requests/second. Cached
once per symbol per day and fully fail-open.

  import insider_flow
  insider_flow.enrich(signals)     # adds s["insider"] = {...}
"""
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_FILE = "insider_cache.json"
CIK_FILE = "sec_cik_map.json"

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

LOOKBACK_DAYS = 90          # insider buys are a slow signal; a quarter is the window
MAX_FILINGS_PER_SYMBOL = 25  # cost cap — big caps file a lot of comp paperwork
REQUEST_PAUSE = 0.12        # SEC asks for <= 10 req/s; this is ~8

# A cluster needs this many DISTINCT insiders buying inside the lookback.
CLUSTER_MIN_BUYERS = 2


def enabled() -> bool:
    return os.getenv("INSIDER_FLOW", "").strip().lower() in ("1", "true", "yes", "on")


def _user_agent() -> str:
    """SEC requires a real contact. Overridable so a fork does not impersonate us."""
    return os.getenv("SEC_USER_AGENT", "trading-agent research you@example.com")


def _headers():
    return {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}


def _cache_dir():
    # Read the env on every call rather than caching at import: a module-scope
    # capture freezes the path to whichever import happened first, which silently
    # breaks anything that sets LEDGER_DIR afterwards (and leaks state in tests).
    from pathlib import Path
    return Path(os.getenv("LEDGER_DIR", "signals"))


def _cache_path(name: str):
    return _cache_dir() / name


def _load_json(name: str) -> dict:
    import json
    p = _cache_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_json(name: str, obj: dict):
    try:
        from io_utils import write_json_atomic
        write_json_atomic(str(_cache_path(name)), obj)
    except Exception:
        pass


def cik_map() -> dict:
    """{TICKER: zero-padded CIK}. Cached for a week — this file barely changes."""
    cached = _load_json(CIK_FILE)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if cached.get("fetched") and cached.get("map"):
        try:
            age = (datetime.strptime(today, "%Y-%m-%d")
                   - datetime.strptime(cached["fetched"], "%Y-%m-%d")).days
            if age < 7:
                return cached["map"]
        except ValueError:
            pass
    try:
        import requests
        raw = requests.get(SEC_TICKERS, headers=_headers(), timeout=30).json()
        m = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
        _save_json(CIK_FILE, {"fetched": today, "map": m})
        return m
    except Exception as e:
        print(f"  [insider] CIK map fetch failed: {str(e)[:80]}", file=sys.stderr)
        return cached.get("map", {})


def _parse_form4(xml_text: str) -> list:
    """[(code, shares, price, owner, is_officer_or_director)] from one Form 4."""
    import xml.etree.ElementTree as ET_

    out = []
    try:
        root = ET_.fromstring(xml_text)
    except Exception:
        return out

    owner, insider_rank = "", False
    for ro in root.iter("reportingOwner"):
        name = ro.find(".//rptOwnerName")
        if name is not None and name.text:
            owner = name.text.strip()
        rel = ro.find("reportingOwnerRelationship")
        if rel is not None:
            def _flag(tag):
                el = rel.find(tag)
                return bool(el is not None and (el.text or "").strip() in ("1", "true"))
            # Officers and directors are the informed cohort. A 10% holder is
            # usually an index fund or a bank's own broker-dealer entity.
            insider_rank = _flag("isOfficer") or _flag("isDirector")

    for tx in root.iter("nonDerivativeTransaction"):
        code_el = tx.find(".//transactionCode")
        if code_el is None or not code_el.text:
            continue
        code = code_el.text.strip().upper()

        def _num(path):
            el = tx.find(path)
            try:
                return float(el.text) if el is not None and el.text else None
            except (TypeError, ValueError):
                return None

        shares = _num(".//transactionShares/value")
        price = _num(".//transactionPricePerShare/value")
        out.append((code, shares, price, owner, insider_rank))
    return out


def _fetch_symbol(symbol: str, cik: str) -> dict:
    """Aggregate insider activity for one symbol over the lookback."""
    import requests

    empty = {"buys": 0, "sells": 0, "buy_shares": 0.0, "buy_value": 0.0,
             "sell_shares": 0.0, "distinct_buyers": 0, "cluster_buy": False,
             "filings_scanned": 0}
    try:
        sub = requests.get(SEC_SUBMISSIONS.format(cik=cik),
                           headers=_headers(), timeout=30).json()
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
    except Exception as e:
        print(f"  [insider] {symbol} submissions failed: {str(e)[:80]}", file=sys.stderr)
        return empty

    cutoff = (datetime.now(ET).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    buyers, res = set(), dict(empty)

    for form, date, acc, doc in zip(forms, dates, accs, docs):
        if form != "4" or date < cutoff:
            continue
        if res["filings_scanned"] >= MAX_FILINGS_PER_SYMBOL:
            break
        # primaryDocument points at the XSL-rendered view; the machine-readable
        # XML is the same filename without that directory prefix.
        raw_doc = doc.split("/")[-1]
        url = SEC_ARCHIVE.format(cik=int(cik), acc=acc.replace("-", ""), doc=raw_doc)
        try:
            time.sleep(REQUEST_PAUSE)
            r = requests.get(url, headers=_headers(), timeout=30)
            if r.status_code != 200:
                continue
            txs = _parse_form4(r.text)
        except Exception:
            continue

        res["filings_scanned"] += 1
        for code, shares, price, owner, is_insider in txs:
            if code == "P":
                res["buys"] += 1
                res["buy_shares"] += shares or 0
                res["buy_value"] += (shares or 0) * (price or 0)
                if is_insider and owner:
                    buyers.add(owner)
            elif code == "S":
                res["sells"] += 1
                res["sell_shares"] += shares or 0
            # A/M/F/G are compensation mechanics — deliberately ignored.

    res["distinct_buyers"] = len(buyers)
    res["cluster_buy"] = len(buyers) >= CLUSTER_MIN_BUYERS
    res["buy_value"] = round(res["buy_value"], 2)
    # Average ticket separates conviction from routine. Some boards (REITs
    # especially) run director plans that buy shares at market every quarter and
    # file them as P — SPG showed 28 buys across 11 insiders at ~$19k each, which
    # is a schedule, not a view. A handful of six-figure purchases is the signal.
    res["avg_buy_value"] = round(res["buy_value"] / res["buys"], 2) if res["buys"] else 0.0
    return res


def enrich(signals: list) -> int:
    """Attach s['insider'] to each signal using a daily cache. Returns # fetched."""
    if not enabled():
        return 0
    today = datetime.now(ET).strftime("%Y-%m-%d")
    cache = _load_json(CACHE_FILE)
    ciks = cik_map()
    fetched = 0

    for s in signals:
        sym = (s.get("symbol") or "").upper()
        if not sym:
            continue
        entry = cache.get(sym)
        if entry and entry.get("date") == today:
            s["insider"] = entry.get("data", {})
            continue
        cik = ciks.get(sym)
        if not cik:
            s["insider"] = {}
            continue
        data = _fetch_symbol(sym, cik)
        cache[sym] = {"date": today, "data": data}
        s["insider"] = data
        fetched += 1

    if fetched:
        _save_json(CACHE_FILE, cache)
    return fetched


if __name__ == "__main__":
    import json
    os.environ.setdefault("INSIDER_FLOW", "true")
    syms = sys.argv[1:] or ["GS", "CAT", "WDC"]
    sigs = [{"symbol": s} for s in syms]
    print(f"fetched {enrich(sigs)}")
    for s in sigs:
        print(f"{s['symbol']:6s} {json.dumps(s.get('insider', {}))}")
