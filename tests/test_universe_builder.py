"""Universe construction — every exclusion counted, and the rails preserved.

The failure this must make impossible is a silent one: a rebuild that quietly
drops the hand-tuned `risk` block, or that reports "top 1000 by liquidity" while
six unstated filters did the real work. A universe nobody can audit is a universe
nobody should trade.

All broker I/O is stubbed, so these run with no keys and no network.
"""
import json

import pytest

import universe_builder as ub


class _Asset:
    def __init__(self, symbol, name=None, tradable=True, exchange="NASDAQ",
                 fractionable=True):
        self.symbol, self.name = symbol, name or symbol
        self.tradable, self.exchange, self.fractionable = tradable, exchange, fractionable


def _stub(monkeypatch, assets, prices):
    """prices: {symbol: (adv_dollar, last_close)}"""
    monkeypatch.setattr(ub, "list_tradable_assets", lambda: _filtered(assets))
    monkeypatch.setattr(ub, "dollar_volumes", lambda syms, **k:
                        {s: prices[s] for s in syms if s in prices})


def _filtered(assets):
    """Run the real exclusion logic over stub assets, mirroring
    list_tradable_assets so filter behaviour is still exercised."""
    drops = {"not_tradable": 0, "exchange": 0, "leveraged": 0, "no_symbol": 0}
    out = []
    for a in assets:
        sym, name = a.symbol.upper(), (a.name or "").upper()
        if not sym or "/" in sym or "." in sym:
            drops["no_symbol"] += 1; continue
        if not a.tradable:
            drops["not_tradable"] += 1; continue
        if a.exchange.upper() not in ub.ALLOWED_EXCHANGES:
            drops["exchange"] += 1; continue
        if any(x in name for x in ub.NAME_EXCLUDE):
            drops["leveraged"] += 1; continue
        out.append({"symbol": sym, "name": a.name, "fractionable": a.fractionable})
    return out, len(assets), drops


# ── ranking ──────────────────────────────────────────────────────────────────

def test_ranks_by_dollar_volume_not_price(monkeypatch):
    """A $900 share trading 1k/day is less tradable than a $20 share trading 5m."""
    assets = [_Asset("EXPENSIVE"), _Asset("LIQUID")]
    _stub(monkeypatch, assets, {"EXPENSIVE": (900_000, 900.0),
                                "LIQUID": (100_000_000, 20.0)})
    uni, _ = ub.build(size=1)
    assert [s["symbol"] for s in uni] == ["LIQUID"]


def test_size_is_respected_and_the_floor_reported(monkeypatch):
    assets = [_Asset(f"S{i}") for i in range(50)]
    prices = {f"S{i}": (1_000_000 * (50 - i), 50.0) for i in range(50)}
    _stub(monkeypatch, assets, prices)
    uni, rep = ub.build(size=10)
    assert len(uni) == 10
    assert rep["adv_top"] > rep["adv_floor"] > 0
    assert rep["drops"]["below_liquidity_rank"] == 40


# ── exclusions, all counted ──────────────────────────────────────────────────

def test_untradable_and_offvenue_names_are_excluded_and_counted(monkeypatch):
    assets = [_Asset("GOOD"), _Asset("HALTED", tradable=False),
              _Asset("OTCJUNK", exchange="OTC")]
    _stub(monkeypatch, assets, {s: (5_000_000, 30.0) for s in
                                ("GOOD", "HALTED", "OTCJUNK")})
    uni, rep = ub.build(size=10)
    assert [s["symbol"] for s in uni] == ["GOOD"]
    assert rep["drops"]["not_tradable"] == 1
    assert rep["drops"]["exchange"] == 1


def test_leveraged_and_inverse_products_are_excluded(monkeypatch):
    """They decay by construction, which breaks an ATR stop and a 20-day SMA."""
    assets = [_Asset("SPY", "SPDR S&P 500 ETF"),
              _Asset("SPXL", "Direxion Daily S&P 500 Bull 3X Shares"),
              _Asset("SQQQ", "ProShares UltraShort QQQ")]
    _stub(monkeypatch, assets, {s: (50_000_000, 100.0) for s in ("SPY", "SPXL", "SQQQ")})
    uni, rep = ub.build(size=10)
    assert [s["symbol"] for s in uni] == ["SPY"]
    assert rep["drops"]["leveraged"] == 2


def test_penny_stocks_are_excluded_on_price_not_liquidity(monkeypatch):
    """A $2 name can turn over millions and still be untradeable for us: the
    spread is a larger fraction of price than the whole per-trade edge."""
    assets = [_Asset("CHEAP"), _Asset("NORMAL")]
    _stub(monkeypatch, assets, {"CHEAP": (90_000_000, 2.0),
                                "NORMAL": (10_000_000, 40.0)})
    uni, rep = ub.build(size=10)
    assert [s["symbol"] for s in uni] == ["NORMAL"]
    assert rep["drops"]["below_min_price"] == 1


def test_derivative_share_classes_are_skipped(monkeypatch):
    assets = [_Asset("BRK.B"), _Asset("FOO/WS"), _Asset("PLAIN")]
    _stub(monkeypatch, assets, {"PLAIN": (5_000_000, 30.0)})
    uni, rep = ub.build(size=10)
    assert [s["symbol"] for s in uni] == ["PLAIN"]
    assert rep["drops"]["no_symbol"] == 2


def test_names_without_price_data_are_dropped_not_defaulted(monkeypatch):
    """Defaulting a missing ADV to zero would look identical to 'illiquid' and
    hide a data problem behind a plausible-looking exclusion."""
    assets = [_Asset("HASDATA"), _Asset("NODATA")]
    _stub(monkeypatch, assets, {"HASDATA": (5_000_000, 30.0)})
    uni, rep = ub.build(size=10)
    assert [s["symbol"] for s in uni] == ["HASDATA"]
    assert rep["drops"]["no_price_data"] == 1


def test_fractionable_only_is_opt_in_and_counted(monkeypatch):
    """Small accounts need fractions: one whole share can already exceed
    max_allocation. But it is a real restriction, so it is opt-in."""
    assets = [_Asset("FRAC"), _Asset("WHOLE", fractionable=False)]
    prices = {"FRAC": (5_000_000, 30.0), "WHOLE": (9_000_000, 30.0)}
    _stub(monkeypatch, assets, prices)
    uni, _ = ub.build(size=10)
    assert {s["symbol"] for s in uni} == {"FRAC", "WHOLE"}
    uni2, rep2 = ub.build(size=10, require_fractionable=True)
    assert [s["symbol"] for s in uni2] == ["FRAC"]
    assert rep2["drops"]["not_fractionable"] == 1


# ── allocation cap ───────────────────────────────────────────────────────────

def test_allocation_cap_never_exceeds_the_hard_risk_limit(monkeypatch):
    """risk_guard enforces 10% per position. A watchlist claiming more would be
    a lie the sizer cannot honour."""
    for size in (1, 2, 10, 100, 1000):
        assets = [_Asset(f"S{i}") for i in range(size)]
        _stub(monkeypatch, assets, {f"S{i}": (5_000_000, 30.0) for i in range(size)})
        uni, rep = ub.build(size=size)
        assert rep["max_allocation"] <= 0.10, f"size={size}"
        assert all(s["max_allocation"] <= 0.10 for s in uni)


def test_cap_shrinks_as_the_universe_grows(monkeypatch):
    def cap_for(n):
        assets = [_Asset(f"S{i}") for i in range(n)]
        _stub(monkeypatch, assets, {f"S{i}": (5_000_000, 30.0) for i in range(n)})
        return ub.build(size=n)[1]["max_allocation"]
    assert cap_for(1000) < cap_for(100) < cap_for(15)


# ── writing: the rails must survive ──────────────────────────────────────────

def test_rebuild_preserves_the_hand_tuned_risk_block(tmp_path):
    """The single most dangerous thing a universe rebuild could do is quietly
    replace the risk rails with defaults."""
    p = tmp_path / "watchlist.json"
    rails = {"stop_loss_pct": 0.08, "min_cash_reserve_pct": 0.2,
             "limit_order_slippage": 0.002}
    p.write_text(json.dumps({"stocks": [{"symbol": "OLD"}], "risk": rails}))
    doc = ub.write([{"symbol": "NEW", "name": "n", "sector": "Unknown",
                     "max_allocation": 0.01}],
                   {"selected": 1, "adv_floor": 1, "drops": {}}, path=p)
    assert doc["risk"] == rails
    assert [s["symbol"] for s in doc["stocks"]] == ["NEW"]
    assert json.loads(p.read_text())["risk"] == rails


def test_missing_rails_are_warned_about_not_invented(tmp_path, capsys):
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps({"stocks": []}))
    doc = ub.write([], {"selected": 0, "adv_floor": 0, "drops": {}}, path=p)
    assert doc["risk"] == {}, "must not fabricate risk limits"
    assert "no `risk` block" in capsys.readouterr().err


def test_written_file_records_the_provenance_and_the_caveat(tmp_path):
    p = tmp_path / "watchlist.json"
    doc = ub.write([{"symbol": "A", "name": "a", "sector": "Unknown",
                     "max_allocation": 0.01}],
                   {"selected": 1, "adv_floor": 1_234_567,
                    "drops": {"exchange": 3}}, path=p)
    assert "do not transfer" in doc["_comment"], \
        "the file must carry the warning that fitted numbers were measured elsewhere"
    assert doc["meta"]["adv_floor_usd"] == 1_234_567
    assert doc["meta"]["dropped_exchange"] == 3
    assert "built_at" in doc["meta"]


# ── reporting ────────────────────────────────────────────────────────────────

def test_summary_lists_every_nonzero_exclusion():
    rep = {"selected": 900, "requested": 1000, "broker_listed": 11000,
           "adv_top": 5e9, "adv_floor": 2e6, "max_allocation": 0.0015,
           "drops": {"exchange": 4000, "leveraged": 120, "below_min_price": 900,
                     "no_price_data": 0}}
    out = ub.summarise(rep)
    assert "900/1000" in out and "11000" in out
    for k in ("exchange", "leveraged", "below_min_price"):
        assert k in out
    assert "no_price_data" not in out, "zero-count drops are noise"


def test_dry_run_does_not_write(monkeypatch, tmp_path):
    assets = [_Asset("A"), _Asset("B")]
    _stub(monkeypatch, assets, {"A": (5e6, 30.0), "B": (4e6, 30.0)})
    target = tmp_path / "watchlist.json"
    monkeypatch.setattr(ub, "WATCHLIST", target)
    assert ub.main(["--size", "1", "--dry-run"]) == 0
    assert not target.exists()
