"""
decide.py — Claude-powered trade decision engine.
Reads today's signals + trading memory + human directive, asks Claude to reason,
then executes qualifying trades via trade.py.

Exit rules executed BEFORE Claude (non-negotiable, no AI override):
  Trailing stop:      GTC stop trailed up as position gains (never moved down)
  Stop-loss:          position ≤ −8%  → sell all immediately
  Take-profit full:   position ≥ +20% → sell all
  Take-profit partial:position ≥ +10% → sell half

Trailing stop schedule:
  Position +5%  → stop moves to break-even (entry price)
  Position +10% → stop moves to +5% above entry
  Position +15% → stop moves to +8% above entry
  Position +20% → take-profit fires

CLAUDE.md is cached as a stable system prompt. Memory and directive go in the user message.
"""
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import copy

try:
    import anthropic
except ModuleNotFoundError:      # pragma: no cover - exercised in CI
    # Only ask_claude() needs the SDK. Importing at module scope makes decide.py
    # un-importable wherever it is absent — which breaks every test that touches
    # this module, and any CI that ships a light dependency set. Upstream that
    # turned CI red for a day and silently froze a CI-gated deploy.
    anthropic = None
from dotenv import load_dotenv

load_dotenv()

import alerts
import debate
import entry_mix
import ladder

# ── Exit thresholds (non-negotiable — executed before Claude) ─────────────────
# The stop floor and the trailing ratchet live in stops.py — ONE definition
# shared with trade.py (initial GTC distance) and stop_monitor.py (5-min hard
# exit + fractional-position ratchet). Never re-inline them here; tests/
# test_stop_rule.py fails CI if any file does.
import signal_rank
import stops
TAKE_PROFIT_FULL    = stops.TAKE_PROFIT_FULL       # +20% → sell all
TAKE_PROFIT_PARTIAL = stops.TAKE_PROFIT_PARTIAL    # +10% → sell half

# Sector Exposure Limit
MAX_SECTOR_EXPOSURE = 0.20   # 20% max per sector

# ML dip confidence thresholds
DIP_SKIP_BELOW   = 0.45   # below this: skip the buy (low confidence)
DIP_LADDER_ABOVE = 0.55   # above this: use ladder buying (split tranche)
DIP_FULL_ABOVE   = 0.68   # above this: full position (high conviction, no ladder needed)
LADDER_TRANCHE_1 = 0.60   # 60% now
LADDER_TRANCHE_2 = 0.40   # 40% via GTC limit at entry × 0.96

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "cash_check_passed": {
            "type": "boolean",
            "description": "True if cash_pct >= 0.20 (buys allowed)"
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["BUY", "SELL", "HOLD", "SKIP"]
                    },
                    "reason": {
                        "type": "string",
                        "description": "Concise reasoning citing specific signal values and relevant memory patterns"
                    },
                    "trade_args": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "CLI args for trade.py e.g. ['--symbol','SPY','--side','buy','--notional','10000']. Null for HOLD/SKIP."
                    },
                    "stop_loss_flag": {
                        "type": "boolean",
                        "description": "True if unrealized P&L <= -8%"
                    }
                },
                "required": ["symbol", "action", "reason", "trade_args", "stop_loss_flag"],
                "additionalProperties": False
            }
        },
        "options_decision": {
            "type": ["object", "null"],
            "description": "Options sleeve decision (max 1 buy/day). Null if no options candidates were provided in context.",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SKIP"]},
                "contract": {
                    "type": ["string", "null"],
                    "description": "OCC symbol of the candidate contract to buy (must come from the provided candidates). Null for SKIP."
                },
                "contracts": {
                    "type": ["integer", "null"],
                    "description": "Number of contracts to buy. Null for SKIP."
                },
                "reason": {"type": "string"}
            },
            "required": ["action", "contract", "contracts", "reason"],
            "additionalProperties": False
        },
        "session_summary": {
            "type": "string",
            "description": "One or two sentences: what was traded, why, what was skipped, any memory patterns applied"
        },
        "memory_observation": {
            "type": "string",
            "description": "One sentence noting any pattern from today that should be remembered (e.g. a signal that fired accurately or a false signal). Empty string if nothing notable."
        },
        "lesson": {
            "type": "string",
            "description": "A DURABLE, reusable trading RULE you are confident about and want to keep permanently (distinct from a one-off observation), e.g. 'illiquid CEFs: market-sell at EOD, GTC stops don't fill'. Empty string unless you have a genuinely durable rule to promote."
        }
    },
    "required": ["cash_check_passed", "decisions", "options_decision", "session_summary", "memory_observation", "lesson"],
    "additionalProperties": False
}

# Tier-A analyst decomposition (flag: ANALYST_VIEWS). Forces per-candidate stances
# across lenses in the SAME call (no extra LLM call) before deciding. Adapted from
# TradingAgents' analyst team, but reusing the data already in the prompt.
_ANALYST_VIEWS_PROP = {
    "type": "array",
    "description": "For each near-threshold / candidate symbol, a one-line stance per lens, then the net.",
    "items": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "technical": {"type": "string", "description": "from the 5-signal score, microstructure, MTF/squeeze"},
            "macro_sector": {"type": "string", "description": "from macro_context sector_impacts + VIX regime"},
            "news_sentiment": {"type": "string", "description": "from the signal's news field; say 'none' if absent"},
            "fundamentals": {"type": "string", "description": "from the signal's fundamentals field (PE, margins, ROE, debt); say 'none' if absent"},
            "net": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        },
        "required": ["symbol", "technical", "macro_sector", "news_sentiment", "fundamentals", "net"],
        "additionalProperties": False,
    },
}


def _analyst_views_on() -> bool:
    return os.getenv("ANALYST_VIEWS", "").strip().lower() in ("1", "true", "yes", "on")


def _loads_lenient(text: str) -> dict:
    """json.loads, tolerant of a prose preamble or ```json fence around the object.

    output_config's json_schema format guarantees clean JSON, so the plain parse is
    the normal path. This fallback exists because when that guarantee was removed
    (commit 525e72b4, 2026-07-09) every decide/macro run died on a bare json.loads
    for five weeks — the paper agent stopped buying entirely while its mechanical
    exits kept selling. A malformed body must degrade, never halt the session."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = text.strip()
    if stripped.startswith("```"):                      # ```json … ``` fence
        stripped = stripped.split("```")[1]
        stripped = stripped.split("\n", 1)[1] if stripped.lower().startswith("json") else stripped
        return json.loads(stripped.strip())
    start, end = stripped.find("{"), stripped.rfind("}")   # prose preamble/epilogue
    if start != -1 and end > start:
        return json.loads(stripped[start:end + 1])
    raise json.JSONDecodeError("no JSON object found in response", text, 0)


def _build_decision_schema() -> dict:
    """Base schema, plus analyst_views when ANALYST_VIEWS is enabled. Default: unchanged."""
    if not _analyst_views_on():
        return DECISION_SCHEMA
    s = copy.deepcopy(DECISION_SCHEMA)
    s["properties"]["analyst_views"] = _ANALYST_VIEWS_PROP
    s["required"] = s["required"] + ["analyst_views"]
    return s


INTRADAY_CONTEXT = """
You also have intraday signals from 15-minute bars. Scoring system (−4 to +4):
| Indicator        | Bullish (+1)                  | Bearish (−1)                  |
|------------------|-------------------------------|-------------------------------|
| VWAP             | Price above VWAP              | Price below VWAP              |
| 15-min RSI       | RSI < 30 (oversold)           | RSI > 70 (overbought)         |
| Opening range    | Price above OR high (breakout)| Price below OR low (breakdown)|
| Momentum vs open | > +0.5% from open             | < −0.5% from open             |

Use daily signals for trend direction. Use intraday signals for timing.
Only trade intraday in the direction of the daily trend unless signals are very strong.
"""


def load_memory() -> str:
    path = Path(__file__).parent / "TRADING_MEMORY.md"
    return path.read_text().strip() if path.exists() else ""


def load_and_clear_directive() -> str:
    path = Path(__file__).parent / "HUMAN_DIRECTIVE.md"
    if not path.exists():
        return ""
    content = path.read_text().strip()
    if "[No active directive]" in content or not content:
        return ""
    # Clear the directive after reading so it only applies once
    path.write_text(
        "# Human Directive\n"
        "_Edit this file to give Claude a one-session instruction at the next 10 AM cycle._\n"
        "_It will be read, acted on, then cleared automatically after the session._\n"
        "_Leave the line below unchanged if you have no directive._\n\n"
        "[No active directive]\n"
    )
    return content


def place_ladder_tranche2(symbol: str, notional: float, entry_price: float) -> dict:
    """
    Place the second ladder tranche: a GTC limit buy at entry × 0.96 (4% below entry).
    This fills if the stock dips further, giving a better average price.

    This used to call tc.submit_order() bare — the ONLY order path in the system
    outside trade.py, meaning no risk-guard, no cash-reserve or allocation check,
    no ledger row (invisible to reconcile + orders_today), and no deterministic
    client_order_id (a mid-session restart could double-place it). It now applies
    the same protections as trade.py, in the same order. Where trade.py fails
    open-ish (a rejected order is reported and the session continues), this leg
    fails CLOSED on any doubt: tranche 2 is an optional enhancement, and skipping
    it costs a slightly worse average price, not a missed position.
    """
    from dotenv import load_dotenv
    load_dotenv()
    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        import accounts
        import ledger
        import risk_guard

        account_name = "core"
        tc = accounts.trading_client(account_name)
        is_live = accounts.is_live(account_name)
        notional = round(notional, 2)
        limit_px = round(entry_price * 0.96, 2)   # 4% below entry

        # ── Same account math as trade.py: INVESTABLE equity (carved netted) ──
        acct = tc.get_account()
        all_pos = tc.get_all_positions()
        carved = accounts.carved_out()
        carved_mv = sum(float(p.market_value) for p in all_pos
                        if p.symbol.upper() in carved) if carved else 0.0
        portfolio_value = float(acct.portfolio_value) - carved_mv
        cash = float(acct.cash)
        held_value = next((float(p.market_value) for p in all_pos
                           if p.symbol == symbol), 0.0)

        # Cash reserve — the tranche reserves buying power at placement, so it
        # must clear the same floor a market buy would.
        wl = json.load(open(Path(__file__).parent / "watchlist.json"))
        min_cash = portfolio_value * wl["risk"]["min_cash_reserve_pct"]
        if cash - notional < min_cash:
            return {"status": "skipped",
                    "error": f"would breach cash reserve (cash {cash:.0f} − {notional:.0f} < {min_cash:.0f})"}

        # Allocation cap — scaled identically to research.py/trade.py. Tranche 1
        # already holds part of the cap, so this catches a T1+T2 total overshoot.
        entry = next((s for s in wl["stocks"] if s["symbol"] == symbol), None)
        if entry is None:
            return {"status": "skipped", "error": "symbol not in watchlist"}
        max_alloc = accounts.effective_max_allocation(entry, account_name)
        if held_value + notional > portfolio_value * max_alloc * 1.01:
            return {"status": "skipped",
                    "error": f"would exceed allocation cap ({max_alloc*100:.0f}%)"}

        # Risk guard — daily/monthly halts, order size, orders-per-day, kill switch.
        month_contrib = 0.0
        try:
            month_contrib = accounts.net_contributions(
                tc, risk_guard.anchor_since(account_name, "month"))
        except Exception:
            pass
        ok, reason = risk_guard.check_order(
            account_name, "buy", notional=notional, equity=portfolio_value,
            held_value=held_value, orders_today=ledger.orders_today(account_name),
            month_contrib=month_contrib)
        if not ok:
            ledger.record({"event": "blocked", "account": account_name, "symbol": symbol,
                           "side": "buy", "notional": notional,
                           "reason": f"ladder2: {reason}", "live": is_live})
            return {"status": "skipped", "error": f"risk guard: {reason}"}

        # Deterministic id (salt distinguishes it from tranche 1) → a mid-session
        # restart re-placing the same tranche is rejected by Alpaca as a duplicate.
        coid = ledger.client_order_id(account_name, symbol, "buy",
                                      notional=notional, salt="ladder2")
        order = tc.submit_order(LimitOrderRequest(
            symbol=symbol,
            limit_price=limit_px,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id=coid,
        ))
        ledger.record({"event": "submitted", "account": account_name, "symbol": symbol,
                       "side": "buy", "notional": notional, "order_id": str(order.id),
                       "client_order_id": coid, "limit_price": limit_px,
                       "kind": "ladder_tranche2", "live": is_live})
        return {"status": "placed", "order_id": str(order.id),
                "limit_price": limit_px, "notional": notional}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:100]}


def resolve_pending_ladders() -> list:
    """Retry deferred tranche 2s. Never raises — an enhancement must not be able
    to stop a session that has real exits to run."""
    try:
        if not ladder.pending():
            return []
        import accounts
        tc = accounts.trading_client("core")
        outcomes = ladder.resolve(
            get_order=lambda oid: tc.get_order_by_id(str(oid)),
            place_tranche2=place_ladder_tranche2)
        for o in outcomes:
            if o["result"] == "placed":
                print(f"  ↩ Tranche 2 recovered for {o['symbol']} "
                      f"(tranche 1 filled @ ${o.get('entry', 0):.2f})")
            else:
                print(f"  ↩ Tranche 2 {o['result']} for {o['symbol']}: {o.get('detail', '')}")
        return outcomes
    except Exception as e:
        print(f"  ladder recovery skipped (non-fatal): {e}", file=sys.stderr)
        return []


def get_sector_map() -> dict:
    path = Path(__file__).parent / "watchlist.json"
    if not path.exists():
        return {}
    with open(path) as f:
        wl = json.load(f)
    return {s["symbol"]: s.get("sector", "Unknown") for s in wl.get("stocks", [])}


def calculate_sector_exposure(positions: list, portfolio_value: float) -> dict:
    sector_map = get_sector_map()
    exposure = {}
    for pos in positions:
        sym = pos["symbol"]
        sector = sector_map.get(sym, "Unknown")
        val = pos["market_value"]
        exposure[sector] = exposure.get(sector, 0.0) + (val / portfolio_value)
    return {k: round(v, 4) for k, v in exposure.items()}


def _release_cover(symbol: str, shares_remaining: float) -> None:
    """Buy back covered calls before shedding the shares that cover them.

    Selling the stock first would leave a naked short call with unbounded loss.
    Fail-open: covered_calls.release() swallows its own errors, and
    covered_calls.reconcile() sweeps up anything left uncovered — an exit must
    never be blocked by the options overlay.
    """
    try:
        import covered_calls
        for a in covered_calls.release(symbol, shares_remaining):
            print(f"    covered call: bought back {a.get('qty')}× {a.get('symbol')} "
                  f"— {a.get('reason', '')}")
    except Exception as e:
        print(f"    covered-call release failed (non-fatal): {str(e)[:140]}", file=sys.stderr)


def execute_exits(signals_data: dict) -> list:
    """
    Execute stop-loss and profit-taking BEFORE calling Claude.
    Returns list of exits that were actioned so Claude is aware.
    Uses volatility-adjusted (ATR) thresholds when available.
    Tightens risk if Macro Kill-Switch is active.
    """
    exits = []
    positions = signals_data.get("positions", [])
    # Map symbol -> atr_pct for exit thresholds
    sig_map = {s["symbol"]: s.get("atr_pct") for s in signals_data.get("signals", [])}
    
    macro = signals_data.get("macro_context")
    kill_switch = bool(macro and "MACRO KILL-SWITCH ACTIVE" in macro.get("trading_guidance", ""))

    for pos in positions:
        sym    = pos["symbol"]
        pl_pct = pos["unrealized_pl_pct"]
        qty    = pos["qty"]
        atr_pct = sig_map.get(sym)

        # Dynamic thresholds based on ATR (e.g., stop = 2.5 * ATR)
        if atr_pct:
            # Tighten stops if Kill-Switch is active (cut losses faster)
            sl_mult = 1.5 if kill_switch else 2.5
            sl_thresh = -max(0.03 if kill_switch else 0.04, sl_mult * atr_pct)
            
            tp_partial = max(0.06, 1.5 * atr_pct)
            tp_full    = max(0.12, 3.0 * atr_pct)
        else:
            sl_thresh = stops.hard_exit_pct(kill_switch)
            tp_partial = TAKE_PROFIT_PARTIAL
            tp_full    = TAKE_PROFIT_FULL

        if pl_pct <= sl_thresh:
            reason = f"stop-loss ({pl_pct*100:.1f}% ≤ {sl_thresh*100:.1f}%)"
            print(f"  *** STOP-LOSS {sym}: {pl_pct*100:.1f}% → selling all ***")
            _release_cover(sym, 0)
            code, out, err = run_trade(["--symbol", sym, "--side", "sell", "--qty", "all"])
            exits.append({"symbol": sym, "type": "stop_loss", "pl_pct": pl_pct,
                           "action": "sell_all", "result": out or err})

        elif pl_pct >= tp_full:
            reason = f"take-profit full ({pl_pct*100:.1f}% ≥ {tp_full*100:.1f}%)"
            print(f"  *** TAKE-PROFIT {sym}: +{pl_pct*100:.1f}% → selling all ***")
            _release_cover(sym, 0)
            code, out, err = run_trade(["--symbol", sym, "--side", "sell", "--qty", "all"])
            exits.append({"symbol": sym, "type": "take_profit_full", "pl_pct": pl_pct,
                           "action": "sell_all", "result": out or err})

        elif pl_pct >= tp_partial:
            half = max(1, int(qty / 2))
            reason = f"take-profit partial ({pl_pct*100:.1f}% ≥ {tp_partial*100:.1f}%) — selling {half} of {qty}"
            print(f"  *** TAKE-PROFIT PARTIAL {sym}: +{pl_pct*100:.1f}% → selling {half} shares ***")
            _release_cover(sym, qty - half)
            code, out, err = run_trade(["--symbol", sym, "--side", "sell", "--qty", str(half)])
            exits.append({"symbol": sym, "type": "take_profit_partial", "pl_pct": pl_pct,
                           "action": f"sell_{half}", "result": out or err})

    return exits


def trail_stops(signals_data: dict) -> list:
    """
    Review all GTC stop orders and trail them upward when a position has gained.
    Never moves a stop DOWN — only locks in higher floors.
    Runs at every 90-min cycle, fully automated (no Claude involvement).
    """
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce
    import accounts

    try:
        tc = accounts.trading_client("core")
    except Exception:
        return []

    # Always use LIVE positions from Alpaca for current P&L — signals file can be stale
    live = tc.get_all_positions()
    _carved = accounts.carved_out()  # never manage/exit a fenced-off symbol
    positions = [
        {
            "symbol":            p.symbol,
            "unrealized_pl_pct": float(p.unrealized_plpc),
            "avg_entry_price":   float(p.avg_entry_price),
            "qty":               float(p.qty),
        }
        for p in live
        if p.symbol.upper() not in _carved
    ]
    if not positions:
        return []

    # Map symbol -> atr_pct from signals file
    try:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        with open(f"signals/{today}.json") as f:
            sig_data = json.load(f)
            sig_map = {s["symbol"]: s.get("atr_pct") for s in sig_data.get("signals", [])}
            macro = sig_data.get("macro_context")
            kill_switch = bool(macro and "MACRO KILL-SWITCH ACTIVE" in macro.get("trading_guidance", ""))
    except Exception:
        sig_map = {}
        kill_switch = False

    # Map symbol → active GTC stop order
    open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    stop_map    = {
        o.symbol: o for o in open_orders
        if o.order_type.value == "stop" and o.time_in_force.value == "gtc"
    }

    adjustments = []
    for pos in positions:
        sym     = pos["symbol"]
        pl_pct  = pos["unrealized_pl_pct"]   # e.g. 0.071 = +7.1%
        entry   = pos["avg_entry_price"]
        qty     = pos["qty"]
        atr_pct = sig_map.get(sym)

        # Where the stop belongs given the current gain — the shared rule, so
        # broker-held stops here and fractional stops in stop_monitor.py can
        # never disagree about the ratchet.
        new_stop_pct, label = stops.ratchet_stop_pct(pl_pct, atr_pct, kill_switch)

        new_stop_px = round(entry * (1 + new_stop_pct), 2)

        existing = stop_map.get(sym)
        current_stop_px = float(existing.stop_price) if existing else 0.0

        # Only trail UP — never reduce a stop that's already higher
        if new_stop_px <= current_stop_px:
            continue

        print(f"  ↑ TRAIL STOP {sym}: ${current_stop_px:.2f} → ${new_stop_px:.2f} "
              f"({label}) — position {pl_pct*100:+.1f}%")

        # Cancel old stop and place new one
        try:
            if existing:
                try:
                    tc.cancel_order_by_id(str(existing.id))
                except Exception:
                    pass # might already be canceling or canceled

                # Poll for up to 15 seconds for the order to be fully cleared
                cleared = False
                for _ in range(15):
                    time.sleep(1)
                    try:
                        o = tc.get_order_by_id(str(existing.id))
                        status = o.status.value if hasattr(o.status, "value") else str(o.status)
                        if status in ("canceled", "expired", "filled"):
                            cleared = True
                            break
                    except Exception:
                        cleared = True # order is gone from active list
                        break
                
                if not cleared:
                    print(f"  ✗ Trail stop for {sym} skipped: old order {existing.id} still not cleared", file=sys.stderr)
                    continue

            whole_qty = math.floor(qty)
            if whole_qty >= 1:
                tc.submit_order(StopOrderRequest(
                    symbol=sym, qty=whole_qty,
                    side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                    stop_price=new_stop_px,
                ))
                adjustments.append({
                    "symbol":     sym,
                    "old_stop":   current_stop_px,
                    "new_stop":   new_stop_px,
                    "pl_pct":     round(pl_pct * 100, 2),
                    "stop_label": label,
                })
        except Exception as e:
            print(f"  ✗ Trail stop failed for {sym}: {e}", file=sys.stderr)

    return adjustments


def run_trade(args: list) -> tuple:
    result = subprocess.run(
        ["python3", "trade.py"] + args,
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def run_options_trade(args: list) -> tuple:
    result = subprocess.run(
        ["python3", "options_trade.py"] + args,
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def execute_options_decision(opt: dict, options_data: dict, portfolio_value: float,
                             executed_alerts: list = None, rejected_alerts: list = None) -> bool:
    """
    Validate and execute Claude's options_decision. Code-level backstops mirror
    the CLAUDE.md rules so a hallucinated contract or oversized order can never
    reach the broker: contract must be a known candidate, selection_score ≥ 60,
    and size is clamped to the 1%-of-portfolio premium cap.
    Returns True if a buy was submitted.
    """
    if not opt or opt.get("action") != "BUY":
        if opt:
            print(f"  OPTIONS: SKIP — {opt.get('reason', '')}")
        return False

    cand = next((c for c in (options_data or {}).get("candidates", [])
                 if c.get("contract", {}).get("symbol") == opt.get("contract")), None)
    if cand is None:
        print(f"  OPTIONS: BUY rejected — {opt.get('contract')} is not in today's candidates")
        return False

    k = cand["contract"]
    if (k.get("selection_score") or 0) < 60:
        print(f"  OPTIONS: BUY skipped — selection_score {k.get('selection_score')} < 60")
        return False

    premium = k.get("premium_per_contract") or 0
    if premium <= 0:
        print("  OPTIONS: BUY skipped — missing premium_per_contract")
        return False
    max_contracts = int((portfolio_value * 0.01) // premium)
    n = min(int(opt.get("contracts") or 0), max_contracts)
    if n < 1:
        print(f"  OPTIONS: BUY skipped — ${premium:,.0f}/contract exceeds 1% portfolio sizing")
        return False

    print(f"  OPTIONS: BUY {n}× {k['symbol']} ({cand.get('setup', '')}) — {opt.get('reason', '')}")
    code, out, err = run_options_trade(
        ["--contract", k["symbol"], "--side", "buy", "--contracts", str(n)])
    if code == 0:
        try:
            r = json.loads(out)
            print(f"    Submitted — order {r['order_id']} @ limit ${r['limit_price']} "
                  f"(premium ~${r.get('est_premium', 0):,.0f}, sleeve {r.get('options_exposure_pct', 0)*100:.1f}%)")
            if executed_alerts is not None:
                executed_alerts.append(
                    f"✅ OPTIONS BUY {n}× {k['symbol']} @ ${r.get('limit_price')} "
                    f"(premium ~${r.get('est_premium', 0):,.0f}) — {cand.get('setup', '')}")
        except Exception:
            print(f"    {out}")
        return True
    try:
        r = json.loads(out)
        print(f"    REJECTED: {r['message']}")
        if rejected_alerts is not None:
            rejected_alerts.append(f"🚫 OPTIONS BUY {k['symbol']} rejected: {r['message'][:140]}")
    except Exception:
        print(f"    ERROR: {err or out}")
    return False


def ask_claude(signals_data: dict, claude_md: str, intraday_data=None,
               memory: str = "", directive: str = "", options_data=None) -> dict:
    client = anthropic.Anthropic()

    # Build preamble — memory and directive go in the user message to preserve system prompt cache
    preamble_parts = []

    # Macro context (geopolitical / economic calendar)
    macro = signals_data.get("macro_context")
    if macro:
        score    = macro.get("macro_score", 0)
        risk     = macro.get("risk_level", "unknown")
        themes   = ", ".join(macro.get("dominant_themes", []))
        guidance = macro.get("trading_guidance", "")
        geo      = "; ".join(macro.get("geopolitical_flags", [])) or "None flagged"
        cal      = "; ".join(macro.get("calendar_warnings", [])) or "None this week"

        sector_lines = "\n".join(
            f"    {sector}: {impact}"
            for sector, impact in macro.get("sector_impacts", {}).items()
        )
        adj          = macro.get("threshold_adjustment", 0)
        yc_signal    = macro.get("yield_curve_signal", "")
        fx_signals   = "; ".join(macro.get("currency_signals", [])) or "None notable"
        earn_watch   = "; ".join(macro.get("earnings_watch", [])) or "None this week"
        adj_note     = f"\n⚠ BUY THRESHOLD RAISED +{adj} by macro context" if adj > 0 else ""

        macro_block = (
            f"## Macro Context (Geopolitical / Economic / Calendar)\n"
            f"Score: {score:+d}  |  Risk: {risk}{adj_note}\n"
            f"Themes: {themes}\n"
            f"Guidance: {guidance}\n"
            f"Geopolitical: {geo}\n"
            f"Yield curve: {yc_signal}\n"
            f"Currency signals: {fx_signals}\n"
            f"Earnings watch: {earn_watch}\n"
            f"Calendar warnings: {cal}\n"
            f"Sector impacts:\n{sector_lines}"
        )
        preamble_parts.append(macro_block)

    # Market regime / VIX context
    regime = signals_data.get("market_regime", {})
    if regime and regime.get("vix"):
        vix   = regime["vix"]
        name  = regime["regime"]
        adj   = regime.get("threshold_adjustment", 0)
        note  = regime.get("note", "")
        adj_str = f"  → BUY threshold raised to +{2+adj} for this session" if adj > 0 else "  → Standard thresholds apply"
        preamble_parts.append(f"## Market Regime\nVIX: {vix} ({name})\n{adj_str}\n{note}")

    # Portfolio deployment status
    port = signals_data.get("portfolio_status", {})
    if port:
        mode       = port.get("deploy_mode", "standard")
        invested   = port.get("invested_pct", 0)
        gap        = port.get("gap_to_fill", 0)
        note       = port.get("note", "")
        dip_cands  = port.get("dip_candidates", [])
        brk_cands  = port.get("breakout_candidates", [])

        # Calculate sector exposure for context
        port_val = signals_data["account"]["portfolio_value"]
        exposure = calculate_sector_exposure(signals_data.get("positions", []), port_val)
        exp_str = ", ".join(f"{s}: {v*100:.1f}%" for s, v in exposure.items())

        dip_str = ", ".join(
            f"{c['symbol']}({c['dip_depth']:.1f}% below SMA20)" for c in dip_cands[:5]
        ) if dip_cands else "none"
        brk_str = ", ".join(
            f"{c['symbol']}({'fresh cross '+str(c['days_since_cross'])+'d ago' if c['fresh_cross'] else 'score+3'})"
            for c in brk_cands[:5]
        ) if brk_cands else "none"

        # The threshold is computed per-session (research.deploy_thresholds) and is
        # env-tunable, so state it explicitly rather than letting the model fall
        # back to the illustrative ladder in CLAUDE.md.
        thresh = port.get("buy_threshold")
        thresh_str = (f"Buy threshold this session: score ≥ {thresh}"
                      if thresh is not None else
                      "Buy threshold this session: no new buys (PRESERVE)")
        target = port.get("target_invested", 0.70)

        mix = port.get("entry_mix") or {}
        mix_str = ""
        if mix.get("counts"):
            c = mix["counts"]
            mix_str = (f"Entry mix: {c.get('breakout', 0)} breakout / {c.get('dip', 0)} dip / "
                       f"{c.get('signal', 0)} signal / {c.get('unknown', 0)} untracked "
                       f"→ {mix.get('lean', '')}\n")

        port_block = (
            f"## Portfolio Deployment Status\n"
            f"Mode: {mode.upper()} | Invested: {invested*100:.0f}% | "
            f"Target: {target*100:.0f}% | Gap: ${gap*100:.0f}K\n"
            f"{thresh_str}\n"
            f"{mix_str}"
            f"Sector Exposure: {exp_str}\n"
            f"{note}\n"
            f"Dip candidates:      {dip_str}\n"
            f"Breakout candidates: {brk_str}"
        )
        preamble_parts.append(port_block)

    # Options sleeve candidates (first session of the day only)
    if options_data and (options_data.get("candidates") or options_data.get("option_positions")):
        opts_payload = {
            "candidates":       options_data.get("candidates", []),
            "option_positions": options_data.get("option_positions", []),
        }
        opts_block = (
            "## Options Candidates (long premium sleeve)\n"
            f"Open premium: ${options_data.get('open_premium', 0):,.0f} "
            f"({options_data.get('open_premium_pct', 0)*100:.1f}% of portfolio, cap 5%)\n"
            "Apply the Options decision rules from your instructions. Output your choice in "
            "`options_decision` — the system executes it via options_trade.py. Max 1 buy, "
            "selection_score ≥ 60, premium ≤ 1% of portfolio.\n"
            + json.dumps(opts_payload, indent=2)
        )
        preamble_parts.append(opts_block)

    if memory:
        preamble_parts.append(f"## Trading Memory (your accumulated context)\n{memory}")
    if directive:
        preamble_parts.append(f"## Human Directive (one-session instruction — act on this)\n{directive}")
    preamble = ("\n\n".join(preamble_parts) + "\n\n---\n\n") if preamble_parts else ""

    # Trim signals: keep actionable, near-threshold, dip candidates, and held positions.
    # Apply Sector Correlation Filter: over-exposed sectors require higher scores.
    deploy_mode = signals_data.get("portfolio_status", {}).get("deploy_mode", "standard")
    port_val = signals_data["account"]["portfolio_value"]
    exposure = calculate_sector_exposure(signals_data.get("positions", []), port_val)
    sector_map = get_sector_map()

    def trim_signals(signals_list):
        result = []
        for s in signals_list:
            sym = s["symbol"]
            has_position  = bool(s.get("position"))
            overextended  = bool(s.get("overextended"))
            score         = s.get("score", 0)
            is_dip        = s.get("is_dip", False)
            is_breakout   = s.get("is_breakout", False)
            sector        = sector_map.get(sym, "Unknown")

            # Sector Cap: if sector > 20%, new buys require score >= 3
            sector_capped = exposure.get(sector, 0) >= MAX_SECTOR_EXPOSURE
            if sector_capped and not has_position:
                if score < 3 and not is_breakout:
                    continue  # Filtered by sector cap

            # Hard block: overextended with no position → skip entirely
            if overextended and not has_position:
                continue

            # Include if: near threshold, held, dip candidate, or breakout candidate
            if (abs(score) >= 1
                    or has_position
                    or (deploy_mode in ("aggressive", "active") and is_dip)
                    or is_breakout):
                # Inject sector cap warning into signal note for Claude
                if sector_capped and not has_position:
                    s["note"] = s.get("note", "") + f" ⚠ SECTOR CAP ({sector} {exposure[sector]*100:.1f}%) — score 3+ required"
                result.append(s)
        return result

    # Rank-and-budget AFTER the filters above. trim_signals answers "is this
    # eligible?"; signal_rank answers "does it fit?" — the second question only
    # started mattering when the universe grew past a few hundred names, where
    # `abs(score) >= 1` (73% of the universe, measured) truncates the reply.
    _eligible = trim_signals(signals_data.get("signals", []))
    _ranked, _rank_report = signal_rank.select(_eligible)
    print(f"  {signal_rank.summarise(_rank_report)}", file=sys.stderr)
    trimmed = {**signals_data, "signals": _ranked}
    hold_count = len(signals_data.get("signals", [])) - len(trimmed["signals"])

    if intraday_data:
        _intra_ranked, _ = signal_rank.select(
            trim_signals(intraday_data.get("signals", [])))
        trimmed_intra = {**intraday_data, "signals": _intra_ranked}
        signal_content = (
            f"Today is {signals_data['date']}. Time: {intraday_data['time']} ET.\n"
            f"({hold_count} score-0 symbols with no position omitted — all HOLD, no action needed.)\n\n"
            "## Daily signals (trend context)\n"
            + json.dumps(trimmed, indent=2)
            + "\n\n## Intraday signals — 15-min bars (timing)\n"
            + json.dumps(trimmed_intra, indent=2)
        )
        system_extra = INTRADAY_CONTEXT
    else:
        signal_content = (
            f"Today is {signals_data['date']}. "
            f"({hold_count} score-0 symbols with no position omitted — all HOLD.)\n"
            "Apply your decision rules to the signals below and output your trading decisions.\n\n"
            + json.dumps(trimmed, indent=2)
        )
        system_extra = ""

    # Tier-A analyst decomposition: ask for per-candidate per-lens stances first.
    if _analyst_views_on():
        signal_content += (
            "\n\n## Analyst decomposition (required)\n"
            "Before your decisions, populate `analyst_views` for each near-threshold / "
            "candidate symbol: a one-line stance from the technical lens (the 5-signal "
            "score + microstructure + MTF), the macro/sector lens (macro_context sector "
            "impacts + VIX regime), the news/sentiment lens (the signal's `news` field; "
            "'none' if absent), and the fundamentals lens (the signal's `fundamentals` "
            "field — PE, margins, ROE, debt; 'none' if absent), then a net "
            "bullish/neutral/bearish. Let these views inform your decisions — a buy needs "
            "the technical lens AND not be contradicted by a clearly bearish macro, news, "
            "or fundamentals lens."
        )

    # Model selection with an explicit fallback chain. CLAUDE_MODEL overrides the
    # primary; if a model id is unavailable (e.g. retired/renamed), fall through
    # to the next so a session never hard-fails on a single model reference.
    # Intraday "timing" calls may use a cheaper model via INTRADAY_MODEL (e.g.
    # claude-haiku-4-5) — set this ONLY on the paper service; the live service
    # leaves it unset so the real-money path stays on Sonnet for the full session.
    # The 10:00 primary decide always uses CLAUDE_MODEL (no intraday_data). The
    # Sonnet fallbacks below still apply, so a bad/cheap id degrades to Sonnet.
    _primary = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    if intraday_data and os.getenv("INTRADAY_MODEL", "").strip():
        _primary = os.getenv("INTRADAY_MODEL").strip()
    _models = [_primary]
    for _fb in ("claude-sonnet-4-6", "claude-3-5-sonnet-latest"):
        if _fb not in _models:
            _models.append(_fb)
    response, _last_err = None, None
    for _m in _models:
        try:
            response = client.messages.create(
                model=_m,
                max_tokens=16000,
                thinking={"type": "disabled"},
                system=[{
                    "type": "text",
                    "text": claude_md + system_extra,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"format": {"type": "json_schema",
                                          "schema": _build_decision_schema()}},
                messages=[{"role": "user", "content": preamble + signal_content}],
            )
            break
        except (anthropic.NotFoundError, anthropic.BadRequestError) as e:
            if "model" not in str(e).lower():
                raise
            _last_err = e
            print(f"  [decide] model {_m} unavailable, trying fallback…", file=sys.stderr)
    if response is None:
        raise _last_err or RuntimeError("no Claude model available")

    usage = response.usage
    cache_note = ""
    if usage.cache_read_input_tokens:
        cache_note = f" (cache hit: {usage.cache_read_input_tokens} tokens)"
    elif usage.cache_creation_input_tokens:
        cache_note = f" (cache write: {usage.cache_creation_input_tokens} tokens)"
    mode = "daily+intraday" if intraday_data else "daily"
    print(f"  Claude [{mode}]: {usage.input_tokens} input + {usage.output_tokens} output tokens{cache_note}")

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(
            f"Claude returned no text block (stop_reason={response.stop_reason}). "
            "Response may have been truncated — increase max_tokens."
        )
    return _loads_lenient(text_blocks[0])


def save_decisions(today: str, session_time: str, signals_data: dict,
                   result: dict, trades_executed: int, exec_results: dict = None):
    """Persist this session's decisions for journal.py to compute outcomes later."""
    os.makedirs("decisions", exist_ok=True)

    entry_prices = {s["symbol"]: s["price"] for s in signals_data["signals"]}

    log = {
        "date": today,
        "time": session_time,
        "account": signals_data["account"],
        "session_summary": result["session_summary"],      # model narrative, pre-execution
        "execution_summary": result.get("execution_summary", ""),   # what actually happened
        "memory_observation": result.get("memory_observation", ""),
        "lesson": result.get("lesson", ""),
        "options_decision": result.get("options_decision"),
        "trades_executed": trades_executed,
        "analyst_views": result.get("analyst_views"),   # Tier-A decomposition (if ANALYST_VIEWS on)
        "debate": result.get("debate"),                 # bull/bear verdicts (if the gate ran)
        "debate_mode": result.get("debate_mode"),       # live | shadow | off — memory_v2 scores on this
        "decisions": [
            {
                "symbol": d["symbol"],
                "action": d["action"],
                "reason": d["reason"],
                "entry_price": entry_prices.get(d["symbol"]),
                # Truth comes from the broker result, not from having intended to
                # trade. Falls back to the old inference only when exec_results is
                # absent (older callers), so behaviour degrades rather than breaks.
                "executed": ((exec_results or {}).get(d["symbol"], {}).get("status") == "submitted"
                             if exec_results is not None
                             else d["action"] in ("BUY", "SELL") and bool(d.get("trade_args"))),
                "execution_status": (exec_results or {}).get(d["symbol"], {}).get("status"),
                "execution_detail": (exec_results or {}).get(d["symbol"], {}).get("detail", ""),
                # Debate-gate counterfactual (Phase 0). Persisted per decision, not
                # just in the top-level `debate` blob, because the scorer needs the
                # verdict next to this decision's entry_price and executed flag.
                "debate_verdict": d.get("debate_verdict"),
                "debate_confidence": d.get("debate_confidence"),
                "debate_applied": d.get("debate_applied"),
                "debate_downgraded": d.get("debate_downgraded", False),
            }
            for d in result["decisions"]
        ],
    }

    path = f"decisions/{today}_{session_time.replace(':', '')}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def _signal_for(signals_data: dict, intraday_data, symbol: str) -> dict:
    """Find a symbol's signal dict (daily, falling back to intraday) for debate context."""
    for src in (signals_data, intraday_data):
        if not src:
            continue
        for s in src.get("signals", []):
            if s.get("symbol") == symbol:
                return s
    return {}


def _debate_context(signals_data: dict) -> str:
    """Compact session context (macro/regime/deploy mode) for the bull/bear judge."""
    m = signals_data.get("macro_context") or {}
    r = signals_data.get("market_regime") or {}
    p = signals_data.get("portfolio_status") or {}
    return (
        f"Macro score {m.get('macro_score', 0)} risk={m.get('risk_level','?')}; "
        f"VIX {r.get('vix','?')} ({r.get('regime','?')}); "
        f"deploy mode {p.get('deploy_mode','standard')}; "
        f"guidance: {m.get('trading_guidance','')}"
    )


def main():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    session_time = datetime.now(ET).strftime("%H:%M")
    signals_path = f"signals/{today}.json"

    from io_utils import load_valid_signals
    signals_data = load_valid_signals(signals_path, today)
    if signals_data is None:
        # Missing, corrupt (truncated write), or wrong-date → regenerate once.
        print("Signals file missing/invalid for today — running research.py now...")
        result = subprocess.run(["python3", "research.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: research.py failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print("research.py complete.\n")
        signals_data = load_valid_signals(signals_path, today)

    if signals_data is None:
        # Fail SAFE: never trade off missing/corrupt/stale signals. research.py
        # already succeeded above yet the file is still invalid → re-running won't
        # help, so exit 3 (EXIT_PERMANENT) to tell the scheduler NOT to retry this
        # session. The next scheduled cycle will try fresh.
        msg = f"🔴 decide.py aborted {today} {session_time} — no valid signals after re-run (fail-safe, no trades)."
        print(msg, file=sys.stderr)
        try:
            alerts.send(msg)
        except Exception:
            pass
        sys.exit(3)

    # Refresh macro context from the latest _macro.json on disk. research.py embeds
    # the 9:00 AM macro brief into the morning signals file, but macro_context.py is
    # re-run before each intraday session — so re-read it here to pick up same-day
    # regime shifts (e.g. geopolitical de-escalation) the morning snapshot can't see.
    macro_path = f"signals/{today}_macro.json"
    if os.path.exists(macro_path):
        with open(macro_path) as f:
            fresh_brief = json.load(f).get("brief")
        if fresh_brief:
            signals_data["macro_context"] = fresh_brief

    account = signals_data["account"]
    print(f"=== Trading Session {today} {session_time} ===")
    print(f"Portfolio: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f} ({account['cash_pct']*100:.1f}%)")

    intraday_path = f"signals/{today}_intraday.json"
    intraday_data = None
    if os.path.exists(intraday_path):
        with open(intraday_path) as f:
            intraday_data = json.load(f)

    # Options are handled by the dedicated options account / options_decide.py
    # (the 10:00 scheduler job runs it). decide.py is now pure equity (core).
    options_data = None

    # ── Step 1·0: Finish any ladder tranche 2 whose tranche 1 filled late ───────
    # trade.py gives up polling for a fill after 30s; a limit order that fills at
    # 31s (or any time later in the session) used to lose its second tranche
    # silently, leaving 60% of the intended position and no record of the rest.
    resolve_pending_ladders()

    # ── Step 1a: Trail stops upward for winning positions ────────────────────────
    trailed = trail_stops(signals_data)
    if trailed:
        print(f"\n  {len(trailed)} stop(s) trailed upward:")
        for t in trailed:
            print(f"    {t['symbol']:5s}  ${t['old_stop']:.2f} → ${t['new_stop']:.2f}  "
                  f"({t['stop_label']}, position +{t['pl_pct']:.1f}%)")

    # ── Step 1b: Non-negotiable exits (stop-loss / profit-taking) — before Claude ─
    exits = execute_exits(signals_data)
    if exits:
        print(f"\n  {len(exits)} automatic exit(s) executed before Claude session.")
        icon = {"stop_loss": "🛑", "take_profit_full": "🎯", "take_profit_partial": "🎯"}
        alerts.send(f"⚙️ Exits executed at {session_time} ET:\n" + "\n".join(
            f"{icon.get(e['type'], '•')} {e['symbol']} {e['type'].replace('_', ' ')} "
            f"({e['pl_pct']*100:+.1f}%) → {e['action'].replace('_', ' ')}"
            for e in exits))

    # ── Step 2: Load context for Claude ──────────────────────────────────────
    memory    = load_memory()
    directive = load_and_clear_directive()

    regime = signals_data.get("market_regime", {})
    if regime.get("vix"):
        print(f"  [VIX: {regime['vix']} — {regime['regime']}]")
    if directive:
        print(f"  [Directive active]: {directive[:80]}...")
    if memory:
        print("  [Trading memory loaded]")

    print("Consulting Claude...")

    claude_md = (Path(__file__).parent / "CLAUDE.md").read_text()
    result = ask_claude(signals_data, claude_md, intraday_data, memory, directive,
                        options_data=options_data)

    print()
    print(f"Summary: {result['session_summary']}")
    if result.get("memory_observation"):
        print(f"Memory note: {result['memory_observation']}")
    if not result["cash_check_passed"]:
        print("NOTE: Cash below 20% reserve — buys skipped.")
    print()

    # ── Step 2b: Bull/bear debate gate — vet proposed BUYs (DEBATE_BUYS/DEBATE_SHADOW) ──
    # One cheap batched call argues bull vs bear per candidate buy; a buy the bear
    # case wins is downgraded to SKIP. Fail-open: a vetting error leaves buys as-is.
    # invested_pct drives the `at:<pct>` form, which re-arms the gate on its own
    # once the book is deployed — see debate.enabled().
    #
    # In "shadow" the same call runs and every verdict is recorded, but nothing is
    # downgraded: both cohorts execute, so memory_v2 can measure whether the gate's
    # skips would actually have lost money. Only debate.mode() decides which it is.
    _ps = signals_data.get("portfolio_status") or {}
    _invested = _ps.get("invested_pct")
    _debate_mode = debate.mode(_invested)
    result["debate_mode"] = _debate_mode
    if (_note := debate.mode_note(_invested)):
        print(f"  {_note}")
    if _debate_mode != "off":
        buy_list = [
            {"symbol": d["symbol"], "reason": d["reason"],
             "signal": _signal_for(signals_data, intraday_data, d["symbol"])}
            for d in result["decisions"]
            if d["action"] == "BUY" and d.get("trade_args")
        ]
        if buy_list:
            verdicts = debate.vet_buys(buy_list, _debate_context(signals_data))
            result["debate"] = verdicts
            debate.stamp_verdicts(result["decisions"], verdicts, applied=_debate_mode == "live")
            if _debate_mode == "live":
                n = debate.apply_to_decisions(result["decisions"], verdicts)
                if n:
                    print(f"  Debate gate: {n}/{len(buy_list)} buy(s) downgraded to SKIP after bull/bear review.")
            else:
                _would = sum(1 for v in verdicts.values() if v.get("verdict") == "skip")
                print(f"  Debate SHADOW: vetted {len(verdicts)} buy(s), "
                      f"would have skipped {_would} — none downgraded (measurement only).")

    trades_executed = 0
    executed_alerts, rejected_alerts = [], []
    # symbol -> what the broker ACTUALLY did. save_decisions used to infer this
    # from the presence of trade_args, which records a risk-guard rejection or a
    # broker error as an executed trade — and memory_v2 then scores that phantom
    # as a real outcome. Only this dict knows the truth, so only it decides.
    exec_results = {}
    # Buys stopped by a gate BEFORE any order is built. Without these the
    # reconciliation could count the gap but not explain it: on 2026-08-13 it
    # correctly reported "proposed 7, submitted 5" and then printed an empty
    # reason, because VLO and PSX were dropped by the dip_confidence floor and
    # so never reached exec_results at all.
    skipped_pre_order = {}

    # Build lookup for ML confidence scores and ATR from signals data
    sig_meta = {s["symbol"]: {"conf": s.get("dip_confidence"), "atr_pct": s.get("atr_pct"),
                              "buy_notional": s.get("buy_notional"),
                              "entry_type": entry_mix.classify(s),
                              "earnings_blackout": bool(s.get("earnings_blackout")),
                              "days_to_earnings": s.get("days_to_earnings")}
                for s in signals_data.get("signals", [])}

    for decision in result["decisions"]:
        symbol     = decision["symbol"]
        action     = decision["action"]
        reason     = decision["reason"]
        trade_args = decision.get("trade_args")
        meta       = sig_meta.get(symbol, {})

        if decision.get("stop_loss_flag"):
            print(f"  *** STOP-LOSS: {symbol} — {reason} ***")

        if action in ("BUY", "SELL") and trade_args:
            conf      = meta.get("conf") if action == "BUY" else None
            conf_str  = f"  [ML conf={conf:.2f}]" if conf is not None else ""

            # SYSTEM-SIZED BUYS: never trust the LLM's chosen dollar amount —
            # override --notional with the system-computed buy_notional
            # (portfolio_value × headroom, investable-equity aware). The LLM picks
            # WHAT to buy; the system decides HOW MUCH. Critical on small accounts
            # (e.g. the $1k live pilot) where the LLM anchors to $10k+ sizes that
            # the allocation guard would reject — so nothing ever deployed.
            # EARNINGS BLACKOUT — enforced here, in code, not in the prompt.
            # CLAUDE.md has always listed "never buy within 48h of earnings" as
            # NON-NEGOTIABLE, but until earnings_calendar.py there was no calendar:
            # the only source was a list the LLM wrote from news headlines. A hard
            # risk rule enforced by a model's reading of the news is not a rule.
            # Sells are untouched — exiting is always permitted.
            if action == "BUY" and meta.get("earnings_blackout"):
                d = meta.get("days_to_earnings")
                print(f"  {symbol}: BUY BLOCKED — earnings in {d} day(s) "
                      f"(gap risk; blackout is non-negotiable)")
                rejected_alerts.append(
                    f"🚫 BUY {symbol} blocked — earnings in {d} day(s)")
                continue

            if action == "BUY":
                sys_notional = round(float(meta.get("buy_notional") or 0), 2)
                if sys_notional < 1:
                    print(f"  {symbol}: BUY SKIPPED — no allocation headroom (buy_notional ${sys_notional})")
                    continue
                ni = next((i + 1 for i, a in enumerate(trade_args) if a == "--notional"), None)
                trade_args = list(trade_args)
                if ni and ni < len(trade_args):
                    trade_args[ni] = str(sys_notional)
                else:
                    trade_args += ["--notional", str(sys_notional)]

            # Inject volatility-adjusted stop-loss if buying
            if action == "BUY" and meta.get("atr_pct"):
                sl_pct = max(0.04, 2.5 * meta["atr_pct"])
                trade_args += ["--stop_loss_pct", str(round(sl_pct, 4))]

            # Corporate-action guard — a split leaves price and the moving
            # averages on different scales, so every price-vs-average indicator
            # goes extreme at once and fabricates an ideal dip. This is a code
            # block rather than a directive line because MNST was bought on
            # exactly this pattern on 2026-08-13 *while a directive said skip it*,
            # the session having convinced itself the indicators had reset.
            if action == "BUY" and meta.get("split_suspect"):
                dev = meta.get("price_vs_sma20_pct")
                why = (f"price {dev:+.1f}% vs SMA20" if dev is not None
                       else "price/SMA20 scale mismatch")
                print(f"  {symbol}: BUY BLOCKED — corporate-action guard ({why}); "
                      f"indicators unreliable until the averages catch up")
                skipped_pre_order[symbol] = {"status": "blocked",
                                             "detail": f"split guard: {why}"}
                continue

            # ML confidence gate for buys (skip very low confidence dips)
            if action == "BUY" and conf is not None and conf < DIP_SKIP_BELOW:
                print(f"  {symbol}: BUY SKIPPED{conf_str} — ML confidence too low ({conf:.2f} < {DIP_SKIP_BELOW})")
                skipped_pre_order[symbol] = {"status": "skipped",
                                             "detail": f"dip_confidence {conf:.2f} < {DIP_SKIP_BELOW}"}
                continue

            # Ladder buying: split tranche when moderate confidence
            use_ladder = (action == "BUY" and conf is not None
                          and DIP_LADDER_ABOVE <= conf < DIP_FULL_ABOVE)

            if use_ladder:
                # Find notional in trade_args and split it
                notional_idx = next((i+1 for i, a in enumerate(trade_args)
                                     if a == "--notional"), None)
                if notional_idx and notional_idx < len(trade_args):
                    orig_notional = float(trade_args[notional_idx])
                    tranche1 = round(orig_notional * LADDER_TRANCHE_1, 2)
                    tranche2 = round(orig_notional * LADDER_TRANCHE_2, 2)
                    trade_args = list(trade_args)
                    trade_args[notional_idx] = str(tranche1)
                    print(f"  {symbol}: BUY (LADDER){conf_str} — "
                          f"${tranche1:.0f} now + ${tranche2:.0f} GTC @ −4% — {reason}")
                else:
                    use_ladder = False
                    print(f"  {symbol}: {action}{conf_str} — {reason}")
            else:
                print(f"  {symbol}: {action}{conf_str} — {reason}")

            code, out, err = run_trade(trade_args)
            if code == 0:
                try:
                    r = json.loads(out)
                    print(f"    Tranche 1 submitted — order {r['order_id']} @ limit ${r['limit_price']}")
                    amt = f"${float(r['notional']):,.0f}" if r.get("notional") else f"{r.get('qty', '?')} sh"
                    executed_alerts.append(
                        f"✅ {action} {symbol} {amt} @ limit ${r['limit_price']} — {reason[:100]}")
                    # Stamp/clear the entry type so the dip-vs-breakout balance of the
                    # book is measurable next session (fail-open — never blocks a trade).
                    if action == "BUY":
                        entry_mix.record(symbol, meta.get("entry_type", entry_mix.SIGNAL), today)
                    elif "all" in [str(a) for a in trade_args]:
                        entry_mix.drop(symbol)
                    # Place tranche 2 GTC if ladder mode. When tranche 1 hasn't
                    # filled inside trade.py's 30s poll there is no price to
                    # anchor −4% to yet — defer instead of dropping it, or the
                    # account keeps 60% of the position and forgets the rest.
                    if use_ladder:
                        if r.get("fill_price"):
                            t2 = place_ladder_tranche2(symbol, tranche2, r["fill_price"])
                            if t2["status"] == "placed":
                                print(f"    Tranche 2 GTC placed — ${tranche2:.0f} @ ${t2['limit_price']:.2f} (−4%)")
                            else:
                                print(f"    Tranche 2 failed: {t2.get('error','')}")
                        else:
                            ladder.record_pending(symbol, r["order_id"], tranche2)
                except Exception:
                    print(f"    {out}")
                trades_executed += 1
                exec_results[symbol] = {"status": "submitted", "detail": ""}
            else:
                try:
                    r = json.loads(out)
                    print(f"    REJECTED: {r['message']}")
                    rejected_alerts.append(f"🚫 {action} {symbol} rejected: {r['message'][:140]}")
                    exec_results[symbol] = {"status": "rejected", "detail": r["message"][:200]}
                except Exception:
                    print(f"    ERROR: {err or out}")
                    rejected_alerts.append(f"🚫 {action} {symbol} errored: {(err or out)[:140]}")
                    exec_results[symbol] = {"status": "error", "detail": (err or out)[:200]}
        elif action == "SKIP":
            print(f"  {symbol}: SKIP — {reason}")
        else:
            print(f"  {symbol}: HOLD — {reason}")

    # ── Options sleeve (first session only — options_data is None otherwise) ──
    if options_data is not None:
        if execute_options_decision(result.get("options_decision"), options_data,
                                    signals_data["account"]["portfolio_value"],
                                    executed_alerts, rejected_alerts):
            trades_executed += 1

    # ── WhatsApp alert: trades executed / rejected this session ──────────────
    if executed_alerts or rejected_alerts:
        alerts.send(f"📈 Session {today} {session_time} ET:\n"
                    + "\n".join(executed_alerts + rejected_alerts))

    # ── Decision → fill reconciliation ───────────────────────────────────────
    # session_summary is written by the model BEFORE the debate gate and before a
    # single order is sent, so it narrates intent. On 2026-08-12 it read "Executed
    # 4 trades: BUY PSX, BUY PWR, BUY CAH…" while zero orders were placed — the
    # debate gate had vetoed all three. Nothing compared the two, so the gap was
    # only findable by reading logs. Record the factual version next to the
    # narrative one, and say so out loud when they disagree.
    # Guarded, and it runs BEFORE save_decisions only to populate the record —
    # a reporting nicety must never prevent the decision log from being written
    # after real orders have gone out. That is the precise failure that cost six
    # sessions of the live audit trail in journal.py.
    try:
        proposed = [d["symbol"] for d in result["decisions"]
                    if d["action"] in ("BUY", "SELL") and d.get("trade_args")]
        submitted = [s for s, r in exec_results.items() if r["status"] == "submitted"]
        blocked   = {s: r for s, r in exec_results.items() if r["status"] != "submitted"}
        blocked.update(skipped_pre_order)          # gates that fired before the order
        vetoed    = [d["symbol"] for d in result["decisions"] if d.get("debate_downgraded")]
        # Anything proposed that neither submitted nor has a recorded reason. This
        # should be empty; if it is not, a gate is dropping buys silently and that
        # is worth seeing rather than quietly reconciling to zero.
        unexplained = [s for s in proposed
                       if s not in exec_results and s not in skipped_pre_order]
        execution_summary = (
            f"proposed {len(proposed)}, submitted {len(submitted)}"
            + (f", blocked {len(blocked)}" if blocked else "")
            + (f", debate-vetoed {len(vetoed)}" if vetoed else "")
            + (f", UNEXPLAINED {len(unexplained)}" if unexplained else "")
        )
        result["execution_summary"] = execution_summary
        result["execution_detail"] = {s: r["detail"] for s, r in blocked.items() if r.get("detail")}
        print()
        print(f"Execution: {execution_summary}")
        if len(submitted) != len(proposed) or vetoed:
            print("  ⚠️  Session summary describes intent, not outcome:")
            for s in vetoed:
                print(f"      {s}: debate-vetoed")
            for s, r in blocked.items():
                print(f"      {s}: {r['status']} — {r.get('detail','')}")
            for s in unexplained:
                print(f"      {s}: UNEXPLAINED — proposed but never attempted or logged")
    except Exception as e:
        print(f"  reconciliation failed (non-fatal): {type(e).__name__}: {e}", file=sys.stderr)

    save_decisions(today, session_time, signals_data, result, trades_executed,
                   exec_results)
    print()
    print(f"Session complete — {trades_executed} trade(s) executed.")


if __name__ == "__main__":
    main()
