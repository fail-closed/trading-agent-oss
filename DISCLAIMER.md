# Disclaimer

**This software places orders with a brokerage. Read this before running it.**

It is provided for research and educational purposes and is **not investment
advice**. The authors are not registered investment advisers, brokers, or
dealers, and nothing here is a recommendation to buy or sell any security.

## What it is

An engineering reference for wiring a language model into a trading loop without
letting it near the risk rails. The interesting parts are the constraints — one
owner per rule, exits that run before the model is called, orders that are always
limit orders, and CI that refuses to let state or modules go unclassified.

## What it is not

A strategy that is known to work.

The thresholds and weights in [`CLAUDE.md`](CLAUDE.md) were fitted to one
198-name universe over 2021–2026. They are published so you can see the
reasoning, not so you can run them. Markets change; fitted parameters do not.
A backtest that looks good on the period it was fitted to is not evidence.

Note that this repository also documents strategies that were tested and
**rejected** — two of them genuine premia in the academic literature that simply
did not survive our cost base. That section is there because a rejected strategy
is the more durable artifact, and because it is the honest shape of this work.

## Risk

Trading involves substantial risk of loss, including loss of principal.

Automated trading can lose money faster than manual trading, and can do so
unattended. Ways this software can lose money that have nothing to do with
whether the strategy is sound:

- a defect in code you did not read
- stale, partial, or incorrect market data (see the incident notes — a partial
  daily bar once volume-blocked 43 of 44 symbols)
- an unadjusted corporate action making a split look like a crash
- a broker outage, a rate limit, or a rejected order at the wrong moment
- a scheduler that stops firing without telling you
- an API key that silently loses permission

The rails in this repository are designed to fail closed against several of
these. None of them is a guarantee.

## Your responsibility

You are solely responsible for any orders placed by software you run, whether or
not you were present when it ran, and whether or not you understood what it
would do.

Concretely, before risking real money:

1. Run it on a **paper account** for a meaningful period — a quarter, not a week.
2. Read [`risk_guard.py`](risk_guard.py), [`stops.py`](stops.py) and
   [`trade.py`](trade.py) end to end. If you would not be comfortable explaining
   what blocks a bad order, do not fund it.
3. Understand the four gates in `accounts.is_live()`. They default to paper, and
   they are deliberately tedious to open.
4. Start with an amount you would be entirely untroubled to lose.

## Jurisdiction

Automated trading may be regulated where you live. Tax treatment of frequent
trading varies. Neither is addressed anywhere in this repository, and neither is
the authors' responsibility to advise you on.

---

The software is licensed under the [MIT License](LICENSE), which includes its own
disclaimer of warranty and limitation of liability. This document is additional
context, not a modification of those terms.
