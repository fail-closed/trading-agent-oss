"""
strategies/daytrade.py — intraday momentum/breakout decision logic (pure).

No broker calls — every function takes plain data and returns decisions, so the
whole strategy is unit-testable without market access. daytrade_runner.py wires
this to the live 'daytrade' Alpaca paper account.

Entry (built on research_intraday.py per-symbol signals):
  long when a name shows a clean intraday breakout —
    opening-range breakout (opening_range.signal == +1)  AND
    price above VWAP        (vwap.signal == +1)          AND
    positive momentum        (momentum.signal == +1)
  → intraday composite score ≥ 3 with those three aligned.
  Rank qualifying names by momentum %, take the top `max_positions`.

Risk (sleeve-level, from daytrade_watchlist.json risk block):
  +1% on the sleeve  → take profit, halt new entries for the day
  −2% on the sleeve  → liquidate everything, halt for the day
  −1% per position   → hard stop on each individual day-trade
  flat by close      → liquidate all before the bell, every day
"""
from dataclasses import dataclass, field


@dataclass
class TradeIntent:
    symbol: str
    side: str                     # "buy" | "sell"
    qty: str = None               # "all" or a number (sells)
    notional: float = None        # dollar amount (buys)
    reason: str = ""
    stop_pct: float = None        # per-position hard stop for buys


def sleeve_state(equity: float, day_start_equity: float, risk: dict) -> dict:
    """Classify the sleeve's intraday P&L into an action gate."""
    if day_start_equity <= 0:
        return {"action": "ok", "pl_pct": 0.0}
    pl = equity / day_start_equity - 1.0
    if pl <= -abs(risk["sleeve_stop_pct"]):
        action = "liquidate"      # -2%: dump everything, done for the day
    elif pl >= risk["sleeve_take_profit_pct"]:
        action = "halt"           # +1%: lock the win, no new entries
    else:
        action = "ok"
    return {"action": action, "pl_pct": round(pl, 4)}


def qualifies(sig: dict) -> bool:
    """A single intraday signal passes the breakout entry gate."""
    s = sig.get("signals", {})
    return (sig.get("score", 0) >= 3
            and s.get("opening_range", {}).get("signal") == 1
            and s.get("vwap", {}).get("signal") == 1
            and s.get("momentum", {}).get("signal") == 1)


def select_entries(intraday_signals: list, held_symbols: set,
                   risk: dict, sleeve_action: str) -> list:
    """
    Pick new day-trade entries from intraday signals.
    Returns [] when the sleeve is halted/liquidating or already at capacity.
    """
    if sleeve_action != "ok":
        return []
    max_pos = risk.get("max_concurrent_positions", 3)
    open_slots = max_pos - len(held_symbols)
    if open_slots <= 0:
        return []

    candidates = [s for s in intraday_signals
                  if qualifies(s) and s["symbol"] not in held_symbols]
    # rank by intraday momentum % (strongest thrust first)
    candidates.sort(key=lambda s: s.get("signals", {}).get("momentum", {}).get("pct", 0),
                    reverse=True)
    return candidates[:open_slots]


def build_intents(intraday_signals: list, held_symbols: set, sleeve_equity: float,
                  day_start_equity: float, risk: dict, near_close: bool) -> dict:
    """
    Full per-cycle decision. Returns:
      {sleeve: {...}, intents: [TradeIntent...], note: str}
    Execution order matters: flatten-by-close and sleeve-liquidate override entries.
    """
    state = sleeve_state(sleeve_equity, day_start_equity, risk)

    # 1. End-of-day flatten — no overnight holds, ever
    if near_close and held_symbols:
        return {"sleeve": state, "note": "flat-by-close: liquidating all",
                "intents": [TradeIntent(sym, "sell", qty="all",
                                        reason="flat by close") for sym in sorted(held_symbols)]}

    # 2. Sleeve stop-loss — dump everything
    if state["action"] == "liquidate" and held_symbols:
        return {"sleeve": state, "note": f"sleeve stop {state['pl_pct']*100:+.1f}% — liquidating all",
                "intents": [TradeIntent(sym, "sell", qty="all",
                                        reason=f"sleeve -2% stop ({state['pl_pct']*100:+.1f}%)")
                            for sym in sorted(held_symbols)]}

    # 3. Otherwise consider new entries (gated by halt/capacity inside select_entries)
    entries = select_entries(intraday_signals, held_symbols, risk, state["action"])
    per_pos = sleeve_equity * risk.get("per_position_pct_of_sleeve", 0.20)
    intents = [
        TradeIntent(s["symbol"], "buy", notional=round(per_pos, 2),
                    stop_pct=risk.get("per_position_stop_pct"),
                    reason=f"intraday breakout (score {s['score']}, "
                           f"mom {s.get('signals',{}).get('momentum',{}).get('pct',0):+.2f}%)")
        for s in entries
    ]
    note = ("halted (+1% hit) — holding, no new entries" if state["action"] == "halt"
            else f"{len(intents)} entry candidate(s)")
    return {"sleeve": state, "note": note, "intents": intents}
