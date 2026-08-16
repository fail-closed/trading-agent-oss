# trading-agent

A rules-based, LLM-assisted paper-trading agent for US equities and ETFs. It
scores a watchlist on five technical signals each morning, applies risk rules
that live in code rather than in a prompt, and asks a model to make the final
call within those rails.

**Paper trading by default. Four independent gates stand between this repo and a
real order.** See [Real money](#real-money).

> **Not investment advice, and not a strategy that is known to work.** This is
> published as an engineering reference — how to wire an LLM into a trading loop
> without letting it near the risk rails. The thresholds and weights in
> `CLAUDE.md` were fitted to one 198-name universe over 2021–2026 and are
> reproduced so you can see the reasoning, not so you can run them. Markets
> change; fitted parameters do not. You can lose money. See [LICENSE](LICENSE) —
> there is no warranty.

---

## What's actually interesting here

Most public trading repos are a strategy. The strategy is the least durable part
of this one. These are the parts worth reading:

| Area | Why |
|---|---|
| **One rule, one owner** (`stops.py`) | The −8% stop and the trailing ladder were once defined in three files and drifted. Now there is one module, every consumer imports it, and `tests/test_stop_rule.py` fails if a second definition appears |
| **Risk rails outside the prompt** (`risk_guard.py`, `trade.py`) | The model proposes; code disposes. Cash floor, position caps, order limits and the daily-loss halt are enforced *after* the LLM, and it cannot argue past them |
| **Fractional-position stops** (`stop_monitor.py`) | Brokers won't hold a stop on 0.4 shares. The same ratchet is enforced in-process every 5 minutes, with the high-water mark persisted — otherwise sub-share positions have no protection at all |
| **Reconciliation that detects drift** (`ledger.reconcile`) | Not "is the symbol still there" but snapshot-diffing quantity and cost basis, with split fingerprinting, so an assignment or a partial exit can't pass silently |
| **A measurement discipline** (`memory_v2.py`) | Every trade is scored at T+5 as *excess* return vs SPY. Scoring against zero counts market drift as skill — a 100% "win rate" can be zero edge |
| **CI tripwires** (`tests/`) | Every module is tested or consciously exempted with a reason; every state file is backed up or declared disposable; env knobs read by two files fail the build. Each tripwire was validated by breaking it on purpose |

If you take one thing, take `tests/test_state_registry.py` and
`tests/test_coverage_floor.py`. They encode a rule that is easy to state and
hard to keep: **you may decide anything, but you may not fail to decide.**

## How it runs

```
09:45  research.py        fetch bars → score signals → signals/<date>.json
10:00  decide.py          mechanical exits FIRST, then the LLM proposes, then
                          risk_guard + trade.py enforce; orders are always limit
16:15  journal.py         write the day, score matured trades, update memory
```

`research.py` needs no LLM key — it is a pure signal engine and you can run it
standalone to see what it produces.

## The scoring system

Five independent signals, each −1/0/+1, summed to a composite −5…+5: RSI(14),
SMA50/200 crossover, MACD(12,26,9), Bollinger %B, and volume vs its 20-day
average. A second `weighted_score` applies empirically-derived weights — bearish
signals are down-weighted because in the tested universe they still preceded
next-day gains about 60% of the time.

Entries fall into three archetypes — **dip** (buying weakness at support),
**breakout** (buying strength on volume) and **momentum** (12-1, slow trend).
`CLAUDE.md` documents which of these carried the tested book and which did not,
including the strategies that were tested and **rejected**.

## Quick start

```bash
git clone <this repo> && cd trading-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca PAPER keys
python research.py          # no LLM key needed — prints today's signals
pytest -q                   # 208 tests
```

Edit `watchlist.json` before doing anything else. The ten symbols shipped are
liquid large caps chosen so a first run works — they are not a recommendation.

## Real money

`accounts.is_live()` requires **all four**, and defaults to paper if any is
missing:

```
I_UNDERSTAND_THIS_TRADES_REAL_MONEY=yes   # explicit acknowledgement
LIVE_TRADING=true                          # global master switch
LIVE_ACCOUNTS=core                         # per-account opt-in
ALPACA_LIVE_API_KEY / _SECRET_KEY          # separate live credentials
```

Three of those existed upstream. The acknowledgement gate was added for public
distribution, because a copied `.env` should not be one step away from real
orders.

If you do go live, the honest advice is the boring advice: run it on paper for a
quarter first, start with an amount you would be untroubled to lose entirely,
and read `risk_guard.py` and `stops.py` end to end before you fund anything.

## What is deliberately not here

Extracted from a private system; these were left behind on purpose.

| Omitted | Why |
|---|---|
| Options and covered-call sleeves | The one premium-*selling* path. Shipping an options writer in a starter repo invites the accident the rails exist to prevent |
| Day-trade / mean-reversion sleeves | Ran virtual-only upstream; no validated edge to pass on |
| Scheduler, dashboard, alerting, Postgres mirror | Deployment-specific. Wire your own cron |
| Backtests and the fitted ML model | Fitted to a universe that is not yours. `ml_trainer.py` is included so you can train your own |
| All journals, signals and trading history | Someone's actual book |

Seven modules are listed in `tests/test_coverage_floor.py:NO_TESTS` with reasons
— their upstream tests depend on infrastructure not shipped here. That list is
honest rather than empty by design.

## License

MIT. No warranty, express or implied. See [LICENSE](LICENSE).
