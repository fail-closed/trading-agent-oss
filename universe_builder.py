"""
universe_builder.py — build watchlist.json by dollar volume, with the drops shown.

    python universe_builder.py --size 1000            # write watchlist.json
    python universe_builder.py --size 1000 --dry-run  # show what it would do

WHY DOLLAR VOLUME AND NOT MARKET CAP
------------------------------------
The constraint that actually bites is whether an order can be filled without
moving the price. A $40bn company that trades by appointment is worse for us than
a $2bn one that turns over $80m a day, because `trade.py`'s ADV guard rejects a
large fraction of thin volume and the limit-order slippage assumption
(`limit_order_slippage`, 0.2%) silently stops being true.

So the universe is ranked by 20-day average dollar volume, which is the same
quantity the execution guard measures.

WHY EVERY EXCLUSION IS COUNTED AND PRINTED
------------------------------------------
"Top 1,000 by liquidity" hides a lot of judgement: which exchanges, what minimum
price, whether leveraged ETFs are in. A builder that silently applies six filters
and prints one number is a builder nobody can audit. Every filter here reports how
many names it removed, so the universe is a decision you can inspect rather than
an artifact you inherit.

WHAT THIS DOES NOT DO
---------------------
It does not validate that a wider universe is a good idea. The published Sharpe
and alpha figures were measured on a 198-name universe and **do not transfer** —
they must be re-derived on whatever this produces before those numbers mean
anything. Building the universe is the cheap half; re-validating is the half that
decides whether it was worth doing.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WATCHLIST = ROOT / "watchlist.json"

# ── filters, each with a stated reason ───────────────────────────────────────

# Regulated US venues only. OTC has no consolidated tape worth trading against
# and our data feed's coverage of it is inconsistent.
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS", "NYSEARCA"}

# Below this, the bid-ask spread is routinely a larger fraction of the price than
# our entire per-trade edge, so a limit order at mid ± 0.2% either misses or fills
# badly. Not a view on the companies.
MIN_PRICE = 5.0

# Leveraged and inverse products decay by construction and break the assumptions
# behind an ATR stop and a 20-day SMA. Matched on the name, not the symbol,
# because tickers are not systematic.
NAME_EXCLUDE = ("2X", "3X", "-1X", "ULTRA", "ULTRASHORT", "INVERSE", "LEVERAGED",
                "BULL 2", "BEAR 2", "DAILY 2", "DAILY 3")

BARS_BATCH = 200          # symbols per bars request
LOOKBACK_DAYS = 40        # ~20 trading sessions plus slack for holidays


def _client():
    import accounts
    from alpaca.trading.client import TradingClient
    k, s = accounts.get_keys("core")
    return TradingClient(k, s, paper=True)


def list_tradable_assets():
    """Active, tradable US equities from the broker, with exclusions counted."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    tc = _client()
    raw = tc.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE,
                                             asset_class=AssetClass.US_EQUITY))
    drops = {"not_tradable": 0, "exchange": 0, "leveraged": 0, "no_symbol": 0}
    out = []
    for a in raw:
        sym = (getattr(a, "symbol", "") or "").upper()
        name = (getattr(a, "name", "") or "").upper()
        if not sym or "/" in sym or "." in sym:
            drops["no_symbol"] += 1                    # units, warrants, pref classes
            continue
        if not getattr(a, "tradable", False):
            drops["not_tradable"] += 1
            continue
        if (getattr(a, "exchange", "") or "").upper() not in ALLOWED_EXCHANGES:
            drops["exchange"] += 1
            continue
        if any(x in name for x in NAME_EXCLUDE):
            drops["leveraged"] += 1
            continue
        out.append({"symbol": sym, "name": getattr(a, "name", sym) or sym,
                    "fractionable": bool(getattr(a, "fractionable", False))})
    return out, len(raw), drops


def dollar_volumes(symbols, batch=BARS_BATCH):
    """{symbol: (adv_dollar, last_close)} over ~20 sessions. Missing data is
    omitted rather than defaulted — a name we cannot price is a name we cannot
    size, and defaulting it to zero would look identical to 'illiquid'."""
    import accounts
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    k, s = accounts.data_keys()
    dc = StockHistoricalDataClient(k, s)
    start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    out = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            bars = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=start, feed=accounts.data_feed())).data
        except Exception as e:
            print(f"  bars batch {i//batch + 1} failed ({str(e)[:70]}) — skipped",
                  file=sys.stderr)
            continue
        for sym, series in bars.items():
            vals = [(float(b.close), float(b.volume)) for b in series
                    if b.close and b.volume]
            if len(vals) < 10:                 # too little history to average
                continue
            recent = vals[-20:]
            adv = sum(c * v for c, v in recent) / len(recent)
            out[sym] = (adv, recent[-1][0])
        print(f"  priced {min(i+batch, len(symbols))}/{len(symbols)} symbols…",
              file=sys.stderr)
    return out


def build(size=1000, min_price=MIN_PRICE, require_fractionable=False):
    """Returns (universe, report). Pure enough to test: all I/O is in the two
    helpers above, which the tests stub."""
    assets, total, drops = list_tradable_assets()
    print(f"  broker listed {total} active US equities → {len(assets)} after "
          f"exclusions {drops}", file=sys.stderr)

    if require_fractionable:
        before = len(assets)
        assets = [a for a in assets if a["fractionable"]]
        drops["not_fractionable"] = before - len(assets)

    dv = dollar_volumes([a["symbol"] for a in assets])
    drops["no_price_data"] = len(assets) - len(dv)

    rows = []
    for a in assets:
        hit = dv.get(a["symbol"])
        if not hit:
            continue
        adv, price = hit
        if price < min_price:
            drops["below_min_price"] = drops.get("below_min_price", 0) + 1
            continue
        rows.append({**a, "adv_dollar": adv, "price": price})

    rows.sort(key=lambda r: r["adv_dollar"], reverse=True)
    chosen = rows[:size]
    drops["below_liquidity_rank"] = max(0, len(rows) - len(chosen))

    # One cap for everything. 1.5x equal-weight gives the sizer room to prefer a
    # strong setup without letting one name dominate; the 10% ceiling is the hard
    # limit risk_guard enforces anyway, so exceeding it here would be a lie.
    cap = round(min(0.10, (1.0 / max(len(chosen), 1)) * 1.5), 5)
    universe = [{"symbol": r["symbol"], "name": r["name"],
                 "sector": "Unknown", "max_allocation": cap} for r in chosen]

    report = {
        "requested": size, "selected": len(chosen), "candidates": len(rows),
        "broker_listed": total, "max_allocation": cap, "drops": drops,
        "adv_floor": round(chosen[-1]["adv_dollar"]) if chosen else 0,
        "adv_top": round(chosen[0]["adv_dollar"]) if chosen else 0,
    }
    return universe, report


def write(universe, report, path=WATCHLIST):
    """Rewrite watchlist.json, preserving the `risk` block — those are hand-tuned
    rails, not something a universe rebuild should silently replace."""
    existing = {}
    if Path(path).exists():
        try:
            existing = json.loads(Path(path).read_text())
        except Exception:
            existing = {}
    doc = {
        "_comment": (f"Built by universe_builder.py — top {report['selected']} US "
                     f"equities by 20-day average dollar volume. NOT a recommendation. "
                     f"The strategy's published Sharpe/alpha were measured on a "
                     f"198-name universe and do not transfer to this one."),
        "stocks": universe,
        "risk": existing.get("risk", {}),
        "meta": {"built_at": datetime.now(timezone.utc).isoformat(),
                 "adv_floor_usd": report["adv_floor"],
                 "selection": "20d average dollar volume",
                 **{f"dropped_{k}": v for k, v in report["drops"].items()}},
    }
    if not doc["risk"]:
        print("  WARNING: no `risk` block found to preserve — the rails are "
              "missing from watchlist.json", file=sys.stderr)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def summarise(report):
    lines = [f"Universe: {report['selected']}/{report['requested']} selected "
             f"from {report['broker_listed']} listed",
             f"  ADV range: ${report['adv_top']:,.0f} … ${report['adv_floor']:,.0f} per day",
             f"  max_allocation: {report['max_allocation']*100:.2f}% per name",
             "  dropped:"]
    for k, v in sorted(report["drops"].items(), key=lambda x: -x[1]):
        if v:
            lines.append(f"    {k:24s} {v:>6}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--fractionable-only", action="store_true",
                    help="only names the broker will sell in fractions — "
                         "necessary for small accounts where one whole share "
                         "may already exceed max_allocation")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    universe, report = build(a.size, a.min_price, a.fractionable_only)
    print()
    print(summarise(report))
    print()
    if a.dry_run:
        print("  DRY RUN — watchlist.json not written")
        print("  first 15:", ", ".join(s["symbol"] for s in universe[:15]))
        return 0
    write(universe, report)
    print(f"  wrote {WATCHLIST.name} with {len(universe)} symbols")
    print("  NEXT: re-run your own validation. The published Sharpe/alpha figures")
    print("        were measured on 198 names and do not transfer to this universe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
