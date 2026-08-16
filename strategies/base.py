"""
strategies/base.py — common interface for pluggable daily strategies.

A strategy is just a pure function of (signals, equity, positions) → list of
TradeIntent. It contains NO broker calls — strategy_runner.py executes the
intents on the strategy's mapped account through the same risk_guard + ledger
path as everything else. So every candidate strategy automatically inherits the
circuit breaker, idempotency, and audit trail, and can be run on a paper account
for testing then promoted by config — no execution code to rewrite.
"""
from dataclasses import dataclass


@dataclass
class TradeIntent:
    symbol: str
    side: str               # "buy" | "sell"
    notional: float = None  # buy: dollars
    qty: str = None         # sell: shares or "all"
    reason: str = ""


# A strategy = decide(signals: list[dict], equity: float, positions: dict) -> list[TradeIntent]
#   signals  : today's per-symbol signal dicts (from signals/YYYY-MM-DD.json)
#   equity   : the account's portfolio value
#   positions: {symbol: market_value} currently held on that account
