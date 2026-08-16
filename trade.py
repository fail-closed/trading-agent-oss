"""
trade.py — place a validated limit order via Alpaca paper trading.

After a BUY fills, automatically places a GTC stop-loss at −8% of fill price.
Take-profit (+10% partial / +20% full) is handled by the 90-min decide.py cycles.

Usage:
  python trade.py --symbol SPY --side buy  --notional 5000
  python trade.py --symbol AAPL --side sell --qty 10
  python trade.py --symbol AAPL --side sell --qty all
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest, StopOrderRequest

import stops                # THE stop rule (shared with decide.py/stop_monitor.py)

STOP_LOSS_PCT    = stops.STOP_LOSS_PCT   # GTC stop placed at fill_price × (1 − pct)
FILL_POLL_SECS   = 30     # seconds to wait for fill before giving up on stop placement
FILL_POLL_EVERY  = 2      # poll interval in seconds


def fail(msg: str):
    print(json.dumps({"status": "error", "message": msg}))
    sys.exit(1)


def daily_stats(data_client, symbol: str, lookback: int = 20) -> dict:
    """Avg daily DOLLAR volume + ATR(14) from recent daily bars. Used by the
    illiquidity guard (ADV_MAX_PCT) and the ATR initial stop (ATR_STOP_MULT).
    Returns {} on failure so callers degrade gracefully."""
    from datetime import timedelta
    try:
        import accounts
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=datetime.now() - timedelta(days=lookback * 2 + 15),
                               feed=accounts.data_feed())
        bars = data_client.get_stock_bars(req).data.get(symbol, [])
        if len(bars) < 5:
            return {}
        closes = [float(b.close) for b in bars]
        highs  = [float(b.high) for b in bars]
        lows   = [float(b.low) for b in bars]
        vols   = [float(b.volume) for b in bars]
        adv_shares = sum(vols[-lookback:]) / min(lookback, len(vols))
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
               for i in range(1, len(bars))]
        atr14 = sum(trs[-14:]) / min(14, len(trs)) if trs else 0.0
        return {"adv_dollar": adv_shares * closes[-1], "atr14": atr14, "bars": len(bars)}
    except Exception as e:
        return {"error": str(e)[:120]}


def place_gtc_stop(trading_client: TradingClient, symbol: str,
                   fill_price: float, filled_qty: float,
                   stop_loss_pct: float = STOP_LOSS_PCT):
    """
    Place a GTC stop-loss order at fill_price × (1 − stop_loss_pct).
    Uses whole shares only (Alpaca GTC restriction on fractional shares).
    Returns the stop order dict or None if placement failed.
    """
    stop_price = round(fill_price * (1 - abs(stop_loss_pct)), 2)
    whole_qty  = math.floor(filled_qty)

    if whole_qty < 1:
        return None  # position too small for whole-share GTC

    try:
        stop_order = trading_client.submit_order(StopOrderRequest(
            symbol=symbol,
            qty=whole_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
        ))
        return {
            "stop_order_id":  str(stop_order.id),
            "stop_price":     stop_price,
            "stop_qty":       whole_qty,
            "stop_pct":       f"-{abs(stop_loss_pct)*100:.1f}%",
        }
    except Exception as e:
        # Non-fatal: the 90-min cycle will catch stop-loss via execute_exits()
        return {"stop_order_id": None, "stop_error": str(e)[:120]}


def poll_for_fill(trading_client: TradingClient, order_id: str,
                  timeout: int = FILL_POLL_SECS) -> tuple:
    """
    Poll until the order is filled or timeout expires.
    Returns (filled_qty, fill_price) or (None, None).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            o = trading_client.get_order_by_id(order_id)
            if o.status.value in ("filled", "partially_filled"):
                fq = float(o.filled_qty) if o.filled_qty else 0
                fp = float(o.filled_avg_price) if o.filled_avg_price else 0
                if fq > 0 and fp > 0:
                    return fq, fp
        except Exception:
            pass
        time.sleep(FILL_POLL_EVERY)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--side",     required=True, choices=["buy", "sell"])
    parser.add_argument("--notional", type=float,    help="Dollar amount to trade (buy side)")
    parser.add_argument("--qty",                     help="Number of shares to sell, or 'all'")
    parser.add_argument("--stop_loss_pct", type=float, default=STOP_LOSS_PCT,
                        help="Volatility-adjusted stop-loss percentage (e.g. 0.05 for -5%)")
    parser.add_argument("--account", default="core", choices=["core", "options", "daytrade"],
                        help="Which Alpaca paper account to trade on (default: core)")
    args = parser.parse_args()

    symbol = args.symbol.upper()

    import accounts
    import risk_guard
    import ledger

    # Carve-out: hard-block any order (buy OR sell) on a fenced-off symbol. Last
    # line of defense guaranteeing a carved-out holding (e.g. the live account's
    # legacy QQQ) is NEVER touched, regardless of upstream logic.
    if symbol in accounts.carved_out():
        fail(f"{symbol} is carved out (CARVE_OUT_SYMBOLS) — agent must not trade it")
    try:
        api_key, secret_key = accounts.get_keys(args.account)
    except (RuntimeError, ValueError) as e:
        fail(str(e))

    # Live ONLY when the account is explicitly promoted via accounts.is_live();
    # otherwise paper. This is the single real-money gate.
    trading_client = accounts.trading_client(args.account)
    data_client    = StockHistoricalDataClient(api_key, secret_key)
    is_live        = accounts.is_live(args.account)

    # ── Validate market is open ──────────────────────────────────────────────
    clock = trading_client.get_clock()
    if not clock.is_open:
        fail(f"Market is closed. Next open: {clock.next_open}")

    # ── Load watchlist risk params ───────────────────────────────────────────
    with open("watchlist.json") as f:
        wl = json.load(f)
    risk = wl["risk"]
    watchlist_entry = next((s for s in wl["stocks"] if s["symbol"] == symbol), None)
    if not watchlist_entry:
        fail(f"{symbol} is not in watchlist.json — add it before trading")

    # THE shared allocation rule — same call the sizer (research.py) and the
    # ladder leg (decide.py) make, so sizer and gate cannot drift apart again
    # (they did on 2026-08-13: scale applied to sizing only → every scaled buy
    # would have been rejected here).
    max_allocation = accounts.effective_max_allocation(watchlist_entry, args.account)

    # ── Account info ─────────────────────────────────────────────────────────
    account         = trading_client.get_account()
    # Net out carved-out holdings (e.g. live legacy QQQ) so sizing, the cash
    # reserve, the per-stock allocation cap, AND risk_guard's equity-based caps
    # all key off INVESTABLE equity — the capital actually under strategy control.
    _carved = accounts.carved_out()
    carved_mv = 0.0
    if _carved:
        carved_mv = sum(float(p.market_value) for p in trading_client.get_all_positions()
                        if p.symbol.upper() in _carved)
    portfolio_value = float(account.portfolio_value) - carved_mv
    cash            = float(account.cash)
    cash_pct        = cash / portfolio_value if portfolio_value > 0 else 0
    min_cash        = portfolio_value * risk["min_cash_reserve_pct"]

    # ── Current position ─────────────────────────────────────────────────────
    positions  = {p.symbol: p for p in trading_client.get_all_positions()}
    position   = positions.get(symbol)
    held_qty   = float(position.qty)          if position else 0.0
    held_value = float(position.market_value) if position else 0.0

    # Feature flags (default OFF → no behavior change; enabled per-service in env).
    adv_max_pct = float(os.getenv("ADV_MAX_PCT", "0") or 0)    # e.g. 0.01 = ≤1% of ADV
    bar_stats = (daily_stats(data_client, symbol)   # ATR knob owned by stops.py
                 if args.side == "buy" and (adv_max_pct > 0 or stops.atr_stop_enabled()) else {})

    # ── Latest quote ─────────────────────────────────────────────────────────
    quote_req = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=accounts.data_feed())
    quotes    = data_client.get_stock_latest_quote(quote_req)
    if symbol not in quotes:
        fail(f"Could not fetch quote for {symbol}")
    ask      = float(quotes[symbol].ask_price)
    bid      = float(quotes[symbol].bid_price)
    slippage = risk["limit_order_slippage"]

    # ══ BUY logic ════════════════════════════════════════════════════════════
    if args.side == "buy":
        if args.notional is None:
            fail("--notional is required for buy orders")
        notional = args.notional

        # Reject non-finite values FIRST: NaN/Inf defeat every comparison-based
        # guard below (any comparison with NaN is False under IEEE-754), so they
        # must be caught before the allocation / cash-reserve / minimum checks.
        if not math.isfinite(notional):
            fail(f"Notional must be a finite number, got {notional}")

        max_notional = portfolio_value * max_allocation
        if held_value + notional > max_notional * 1.01:
            fail(
                f"Order would exceed max allocation: current {held_value:.2f} + "
                f"new {notional:.2f} > {max_notional:.2f} ({max_allocation*100:.0f}% of {portfolio_value:.2f})"
            )

        if cash - notional < min_cash:
            fail(
                f"Order would breach 20% cash reserve: "
                f"cash {cash:.2f} - notional {notional:.2f} < min {min_cash:.2f}"
            )

        if notional < 1:
            fail("Notional must be at least $1")

        # Illiquidity guard: reject if the order is a large fraction of average
        # daily dollar volume (protects against thin names like CEFs — e.g. JPC).
        if adv_max_pct > 0:
            adv = bar_stats.get("adv_dollar", 0)
            if adv > 0 and notional > adv * adv_max_pct:
                fail(f"Illiquidity guard: ${notional:,.0f} > {adv_max_pct*100:.2f}% of ADV "
                     f"(${adv*adv_max_pct:,.0f}); 20d ADV ${adv:,.0f} for {symbol}")

        limit_price = round(ask * (1 + slippage), 2)
        # Capture the deterministic client_order_id so it's recorded in the ledger
        # too (links the audit trail to the broker order + makes dedup exact).
        coid = ledger.client_order_id(args.account, symbol, "buy", notional=round(notional, 2))
        order_req   = LimitOrderRequest(
            symbol=symbol,
            limit_price=limit_price,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )

    # ══ SELL logic ═══════════════════════════════════════════════════════════
    else:
        if held_qty == 0:
            fail(f"No position in {symbol} to sell")

        if args.qty is None:
            fail("--qty is required for sell orders (use 'all' to sell entire position)")

        sell_qty = held_qty if args.qty == "all" else float(args.qty)
        if sell_qty > held_qty:
            fail(f"Cannot sell {sell_qty} shares — only holding {held_qty}")

        sell_qty  = int(sell_qty) if sell_qty == int(sell_qty) else sell_qty

        # Cancel any resting orders on this symbol (e.g. a GTC stop) first — they
        # reserve shares (held_for_orders) and would otherwise cause the sell to
        # fail with "insufficient qty available" (the JPC bug, 2026-06-15).
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = trading_client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[symbol]))
            for o in open_orders:
                trading_client.cancel_order_by_id(o.id)
            if open_orders:
                time.sleep(1.0)   # let cancellations settle so shares free up
        except Exception as e:
            print(f"  warning: could not cancel resting orders for {symbol}: {e}", file=sys.stderr)

        limit_price = round(bid * (1 - slippage), 2)
        coid = ledger.client_order_id(args.account, symbol, "sell", qty=sell_qty)
        order_req   = LimitOrderRequest(
            symbol=symbol,
            limit_price=limit_price,
            qty=sell_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )

    # ── Risk circuit breaker — every order passes through here ────────────────
    guard_notional = round(notional, 2) if args.side == "buy" else None
    # Net out external cash flows (deposits/withdrawals) since the month anchor so the
    # profit-lock isn't tripped by a deposit — a $3k add would otherwise read as a
    # huge MTD "gain" and pause all new buys (it did, on the live sleeve, 2026-06-29).
    month_contrib = 0.0
    if args.side == "buy":
        try:
            month_contrib = accounts.net_contributions(
                trading_client, risk_guard.anchor_since(args.account, "month"))
        except Exception:
            month_contrib = 0.0
    ok, reason = risk_guard.check_order(
        args.account, args.side, notional=guard_notional, equity=portfolio_value,
        held_value=held_value, orders_today=ledger.orders_today(args.account),
        month_contrib=month_contrib)
    if not ok:
        ledger.record({"event": "blocked", "account": args.account, "symbol": symbol,
                       "side": args.side, "notional": guard_notional, "reason": reason,
                       "client_order_id": coid, "live": is_live})
        fail(f"RISK GUARD: {reason}")

    # ── Submit (audit both success AND failure) ───────────────────────────────
    try:
        order = trading_client.submit_order(order_req)
    except Exception as e:
        emsg = str(e)
        # Stale-coid recovery (sells only). The deterministic client_order_id is a
        # double-fill guard, but Alpaca enforces coid uniqueness across ALL orders
        # ever placed (including cancelled/filled ones). A legitimate same-day exit
        # retry of the same symbol/side/qty regenerates the SAME coid → rejected
        # with "client_order_id must be unique" (40010001) — the WDC bug: one 10:00
        # sell submitted, then four ~90-min retries all collided. Cancelling resting
        # orders frees the shares but not the coid. Resubmit once with a unique salt:
        # safe because the cancel-resting + held_qty checks above already rule out a
        # double-fill. Buys keep the strict guard (anti-double-buy on restart).
        dup = "client_order_id must be unique" in emsg or "40010001" in emsg
        if dup and args.side == "sell":
            coid = ledger.client_order_id(args.account, symbol, "sell", qty=sell_qty,
                                          salt=datetime.now(ET).strftime("%H%M%S%f"))
            order_req = LimitOrderRequest(
                symbol=symbol, limit_price=limit_price, qty=sell_qty,
                side=OrderSide.SELL, time_in_force=TimeInForce.DAY, client_order_id=coid)
            ledger.record({"event": "coid_retry", "account": args.account, "symbol": symbol,
                           "side": "sell", "qty": sell_qty, "client_order_id": coid,
                           "live": is_live, "note": "stale deterministic coid; resubmitting with salt"})
            try:
                order = trading_client.submit_order(order_req)
            except Exception as e2:
                ledger.record({"event": "submit_failed", "account": args.account, "symbol": symbol,
                               "side": "sell", "qty": sell_qty, "client_order_id": coid,
                               "live": is_live, "error": str(e2)[:200]})
                fail(f"Order submit failed (after coid retry): {str(e2)[:200]}")
        else:
            ledger.record({"event": "submit_failed", "account": args.account, "symbol": symbol,
                           "side": args.side, "notional": guard_notional,
                           "qty": (sell_qty if args.side == "sell" else None),
                           "client_order_id": coid, "live": is_live, "error": emsg[:200]})
            fail(f"Order submit failed: {emsg[:200]}")
    ledger.record({"event": "submitted", "account": args.account, "symbol": symbol,
                   "side": args.side, "order_id": str(order.id), "limit_price": limit_price,
                   "notional": guard_notional,
                   "qty": (sell_qty if args.side == "sell" else None),
                   "client_order_id": coid, "live": is_live})

    result = {
        "status":        "submitted",
        "order_id":      str(order.id),
        "symbol":        symbol,
        "side":          args.side,
        "limit_price":   limit_price,
        "qty":           str(order.qty)      if order.qty      else None,
        "notional":      str(order.notional) if order.notional else None,
        "time_in_force": "day",
        "submitted_at":  datetime.now(ET).isoformat(),
        "account_after": {
            "portfolio_value": round(portfolio_value, 2),
            "cash":            round(cash, 2),
            "cash_pct":        round(cash_pct, 4),
        },
    }

    # ── Auto stop-loss for buys ───────────────────────────────────────────────
    if args.side == "buy":
        filled_qty, fill_price = poll_for_fill(trading_client, str(order.id))
        if filled_qty and fill_price:
            # Initial stop distance: fixed --stop_loss_pct by default, OR a
            # volatility-adjusted ATR(14)×ATR_STOP_MULT stop (clamped 4%–15%) so
            # noisy high-beta names get room and slow names get a tighter stop.
            stop_pct = stops.initial_stop_pct(atr14=bar_stats.get("atr14", 0),
                                              price=fill_price,
                                              base=args.stop_loss_pct)
            stop_result = place_gtc_stop(trading_client, symbol, fill_price, filled_qty,
                                         stop_loss_pct=stop_pct)
            result["fill_price"]  = round(fill_price, 4)
            result["filled_qty"]  = round(filled_qty, 6)
            result["stop_loss"]   = stop_result or {"stop_order_id": None, "note": "poll timed out"}
        else:
            result["stop_loss"] = {
                "stop_order_id": None,
                "note": f"Order not filled within {FILL_POLL_SECS}s — stop will be placed at next cycle",
            }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
