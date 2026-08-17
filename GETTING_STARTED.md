# Getting started

No coding required. Two commands, then a guided setup that asks you questions.

```bash
git clone https://github.com/fail-closed/trading-agent-oss.git
cd trading-agent-oss
python3 setup_wizard.py
```

That's it. The wizard does the rest, and **nothing is written until it has
verified your details work.** Press Ctrl-C at any point to back out.

---

## What you need first

| | | Cost |
|---|---|---|
| **Python 3.10+** | Already on most Macs and Linux. On Windows, get it from [python.org](https://python.org) | free |
| **An Alpaca paper account** | The broker. The wizard links you to it and walks you through | **free, no money required** |
| An Anthropic API key | *Optional.* Only needed for the part that decides trades | ~a few dollars a month |

You do **not** need to fund anything to start. A paper account trades fake cash
against real market prices, which is exactly what you want for the first months.

## What the wizard does, step by step

1. **Checks your Python** — stops with a clear message if it's too old
2. **Installs everything** into an isolated folder so it can't disturb anything else on your computer
3. **Takes your Alpaca paper keys** and immediately verifies them against the real
   broker — you'll see your play-money balance, so you know it worked
4. **Lets you choose what to trade**, or keep a starter list of ten large companies
5. **Optionally takes an AI key** and explains exactly what you get with and without one
6. **Sets sensible defaults** for everything else
7. **Runs the engine once**, live, so you see real output instead of taking our word
8. **Prints the daily schedule** you can copy in if you want it to run by itself

Run it again any time. It shows what's already set and keeps it unless you say
otherwise.

## After setup

```bash
.venv/bin/python research.py     # score your watchlist (no AI key needed)
.venv/bin/python decide.py       # exits first, then propose + enforce
.venv/bin/python journal.py      # write up the day
```

`research.py` is the one to start with. It places no orders — it just shows you
what the system sees, and you can run it as often as you like.

## Real money

**Don't rush this.** The wizard has a separate path:

```bash
python3 setup_wizard.py --live
```

It is deliberately harder than the paper setup. It will refuse unless you have
actually run paper sessions, it asks whether you've read the three files that
decide whether a bad order reaches your broker, it makes you name an amount you'd
be untroubled to lose, and it makes you type a sentence rather than press Enter.

That isn't box-ticking. Paper trading cannot cost you anything; live trading can
cost you everything you put in — and the ways it goes wrong are usually not "the
strategy was wrong". They're a bug in code nobody read, a stale price feed, a
stock split the system misread, or an unattended loop at 3am.

Read **[DISCLAIMER.md](DISCLAIMER.md)** first. It's short and it's honest.

### Stopping live trading

Either of these takes effect on the next run, immediately:

| Edit `.env` | Effect |
|---|---|
| `LIVE_TRADING=false` | back to paper |
| `KILL_SWITCH=true` | blocks all buying; selling still works, so stops can still fire |

The second is the one to use in a panic — it never traps you in a position.

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Python 3.10+ required` | old Python | install from [python.org](https://python.org), re-run |
| `rejected (401/403)` | wrong keys, or live keys on the paper endpoint | check the Alpaca dashboard says **Paper**, regenerate |
| `cannot reach Alpaca` | no internet, or a firewall | check your connection |
| The engine ran but found nothing | market holiday, or a symbol that no longer trades | harmless — nothing was ordered |
| `decide.py` fails, `research.py` works | no AI key | add `ANTHROPIC_API_KEY` to `.env`, or just use `research.py` |

Nothing in that table can lose you money on a paper account.

## What this is not

It is not a product, it has no support, and it is not investment advice. It's an
engineering reference published so you can read how a language model can be wired
into a trading loop without letting it near the risk rules.

The thresholds in [`CLAUDE.md`](CLAUDE.md) were fitted to one 198-name universe
over 2021–2026. They are shown so you can see the reasoning, **not** so you can
run them. Markets change; fitted numbers don't.
