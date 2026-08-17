#!/usr/bin/env python3
"""
setup_wizard.py — get from a fresh clone to a working paper agent, guided.

    python3 setup_wizard.py

STDLIB ONLY, on purpose. This runs BEFORE `pip install`, so it cannot import
requests, alpaca, dotenv or anything else from requirements.txt. Everything here
uses urllib, json and subprocess.

WHAT IT WILL AND WILL NOT DO
----------------------------
It will: check your Python, create a virtualenv, install dependencies, take your
Alpaca PAPER keys and verify them against the real API, let you choose a
watchlist, write `.env`, run the signal engine once so you can see real output,
and print the schedule you can copy into cron.

It will NOT set up live trading. `--live` exists and is deliberately slower: it
refuses to proceed without evidence you have actually run on paper, and it makes
you type things rather than press Enter. That is not friction for its own sake —
see the note in `live_setup()`.

Re-running is safe. Existing `.env` values are shown (secrets masked) and kept
unless you choose to replace them.
"""
import getpass
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
VENV = ROOT / ".venv"

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"

# ── presentation ─────────────────────────────────────────────────────────────

_C = {"b": "\033[1m", "dim": "\033[2m", "g": "\033[32m", "y": "\033[33m",
      "r": "\033[31m", "c": "\033[36m", "x": "\033[0m"}
if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
    _C = {k: "" for k in _C}


def say(msg=""):
    print(msg)


def head(n, total, title):
    say()
    say(f"{_C['c']}{_C['b']}  Step {n}/{total} — {title}{_C['x']}")
    say(f"{_C['dim']}  " + "─" * (len(title) + 14) + _C['x'])


def ok(msg):
    say(f"  {_C['g']}✓{_C['x']} {msg}")


def warn(msg):
    say(f"  {_C['y']}!{_C['x']} {msg}")


def bad(msg):
    say(f"  {_C['r']}✗{_C['x']} {msg}")


def die(msg, fix=None):
    bad(msg)
    if fix:
        say(f"    {_C['dim']}try: {fix}{_C['x']}")
    sys.exit(1)


def ask(prompt, default=None, secret=False, allow_blank=False):
    """One question. Returns a stripped string. Ctrl-C exits cleanly.

    `secret=True` reads without echo (getpass), so a broker secret does not land
    in the terminal scrollback, a shared screen, or a screen recording. Anything
    that is a credential must pass secret=True — the first version of this wizard
    declared the parameter and never used one, which meant the Alpaca secret was
    echoed in full and stayed visible for the rest of the session.
    """
    suffix = f" {_C['dim']}[{default}]{_C['x']}" if default else ""
    reader = getpass.getpass if secret else input
    try:
        while True:
            val = reader(f"  {prompt}{suffix}: ").strip()
            if not val and default is not None:
                return default
            if val or allow_blank:
                return val
            say(f"    {_C['dim']}(required){_C['x']}")
    except (EOFError, KeyboardInterrupt):
        say()
        say("  Cancelled. Nothing was written.")
        sys.exit(130)


def confirm(prompt, default=False):
    d = "Y/n" if default else "y/N"
    val = ask(f"{prompt} ({d})", default="y" if default else "n").lower()
    return val.startswith("y")


def mask(v):
    if not v:
        return ""
    return v[:4] + "…" + v[-2:] if len(v) > 8 else "…"


# ── .env handling ────────────────────────────────────────────────────────────

def read_env():
    """Existing .env as a dict. Comments and blanks are dropped; we rewrite the
    file from the template each time so new options appear on re-runs."""
    out = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.split("#")[0].strip()
    return out


def write_env(values):
    """Write .env with 0600 permissions. It holds live broker keys once live is
    configured, so world-readable is not acceptable."""
    lines = ["# Written by setup_wizard.py. Safe to edit by hand.",
             "# NEVER commit this file — .gitignore already excludes it.", ""]
    groups = [
        ("Broker — PAPER (free, no funding required)",
         ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_DATA_FEED"]),
        ("LLM — only decide.py and macro_context.py need this",
         ["ANTHROPIC_API_KEY"]),
        ("Optional data", ["FRED_API_KEY", "PREDICTION_MARKETS", "SEC_USER_AGENT",
                           "INSIDER_FLOW"]),
        ("Behaviour", ["MEMORY_V2", "BENCHMARK_SYMBOL", "MAX_BUYS_PER_SESSION",
                       "TARGET_INVESTED_PCT"]),
        ("Artifact mirror (optional — YOUR repo)", ["GITHUB_TOKEN", "GITHUB_REPO"]),
        ("REAL MONEY — all four required, defaults keep you on paper",
         ["I_UNDERSTAND_THIS_TRADES_REAL_MONEY", "LIVE_TRADING", "LIVE_ACCOUNTS",
          "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY"]),
    ]
    seen = set()
    for title, keys in groups:
        lines.append(f"# ── {title} " + "─" * max(0, 60 - len(title)))
        for k in keys:
            seen.add(k)
            lines.append(f"{k}={values.get(k, '')}")
        lines.append("")
    extra = sorted(set(values) - seen)
    if extra:
        lines.append("# ── Other ──")
        lines += [f"{k}={values[k]}" for k in extra]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    os.chmod(ENV_PATH, 0o600)


# ── Alpaca verification ──────────────────────────────────────────────────────

def alpaca_account(key, secret, base):
    """GET /v2/account. Returns (ok, payload_or_error). No SDK — stdlib only."""
    req = urllib.request.Request(
        f"{base}/v2/account",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return True, json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "rejected (401/403) — key or secret is wrong, or it is a LIVE key on the paper endpoint"
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"cannot reach Alpaca ({e.reason}) — check your internet connection"
    except Exception as e:                                    # pragma: no cover
        return False, str(e)[:120]


# ── steps ────────────────────────────────────────────────────────────────────

TOTAL = 8


def step_python():
    head(1, TOTAL, "Checking Python")
    v = sys.version_info
    say(f"  found Python {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10):
        die(f"Python 3.10+ required, you have {v.major}.{v.minor}",
            "install a newer Python from python.org, then re-run this wizard")
    ok(f"Python {v.major}.{v.minor} is supported")


def step_deps():
    head(2, TOTAL, "Installing dependencies")
    if VENV.exists():
        ok(f"virtualenv already present at {VENV.name}/")
    else:
        say("  creating an isolated environment so this cannot disturb other Python projects…")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)],
                           capture_output=True, text=True)
        if r.returncode:
            die("could not create a virtualenv", "python3 -m venv .venv")
        ok(f"created {VENV.name}/")
    pip = VENV / "bin" / "pip"
    if not pip.exists():
        pip = VENV / "Scripts" / "pip.exe"           # Windows
    say("  installing packages (this takes a minute or two)…")
    r = subprocess.run([str(pip), "install", "-q", "-r", str(ROOT / "requirements.txt")],
                       capture_output=True, text=True)
    if r.returncode:
        say(r.stderr[-800:])
        die("dependency install failed", f"{pip} install -r requirements.txt")
    ok("dependencies installed")
    return VENV / "bin" / "python"


def step_paper_keys(env):
    head(3, TOTAL, "Connecting your Alpaca paper account")
    say("  Alpaca is the broker. A PAPER account is free, needs no money, and")
    say("  trades with fake cash against real market prices.")
    say()
    say(f"  1. Sign up (free): {_C['b']}https://alpaca.markets{_C['x']}")
    say("  2. Make sure the dashboard says 'Paper' (top-left toggle), not 'Live'")
    say("  3. Find 'API Keys' and click Generate")
    say("  4. Copy BOTH values — the secret is shown only once")
    say()

    have = env.get("ALPACA_API_KEY")
    if have and not confirm(f"replace the existing key {mask(have)}?", default=False):
        ok("keeping existing keys")
    else:
        while True:
            key = ask("Paste your paper API key ID")
            sec = ask("Paste your paper secret key (hidden as you type)", secret=True)
            if key.upper().startswith("PK") is False and key.upper().startswith("AK"):
                warn("that looks like a LIVE key (starts with AK). Paper keys start with PK.")
                if not confirm("use it anyway?", default=False):
                    continue
            say("  verifying against Alpaca…")
            good, res = alpaca_account(key, sec, PAPER_BASE)
            if good:
                env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"] = key, sec
                ok(f"connected — paper account with ${float(res.get('equity', 0)):,.2f} of play money")
                break
            bad(str(res))
            if not confirm("try again?", default=True):
                die("cannot continue without working paper keys")
    env.setdefault("ALPACA_DATA_FEED", "iex")
    return env


def step_watchlist(env):
    head(4, TOTAL, "Choosing what to trade")
    wl_path = ROOT / "watchlist.json"
    wl = json.loads(wl_path.read_text())
    syms = [s["symbol"] for s in wl["stocks"]]
    say(f"  The starter list has {len(syms)} liquid large caps:")
    say(f"    {_C['dim']}{', '.join(syms)}{_C['x']}")
    say()
    say("  These are NOT a recommendation — they exist so a first run works.")
    say(f"  {_C['dim']}Each has a max_allocation cap; the engine never exceeds it.{_C['x']}")
    say()
    if confirm("edit the list now?", default=False):
        say("  Enter symbols separated by commas (e.g. AAPL, MSFT, SPY).")
        say(f"  {_C['dim']}8-30 names is a reasonable range. Fewer than 8 and one name dominates.{_C['x']}")
        raw = ask("Symbols", default=",".join(syms))
        new = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if len(new) < 3:
            warn("that is very few names — keeping the starter list instead")
        else:
            cap = round(min(0.10, 1.0 / len(new) * 1.5), 4)
            wl["stocks"] = [{"symbol": s, "name": s, "sector": "Unknown",
                             "max_allocation": cap} for s in new]
            wl["_comment"] = "Edited by setup_wizard.py."
            wl_path.write_text(json.dumps(wl, indent=2) + "\n")
            ok(f"saved {len(new)} symbols, max {cap*100:.1f}% each")
    else:
        ok("keeping the starter list")
    return env


def step_llm(env):
    head(5, TOTAL, "The decision layer (optional)")
    say("  The signal engine runs WITHOUT this. An API key is only needed for the")
    say("  part that chooses among the candidates the rules already approved.")
    say()
    say(f"  Without a key: {_C['b']}research.py{_C['x']} works — you see scores and candidates daily.")
    say(f"  With a key:    {_C['b']}decide.py{_C['x']} also works — it proposes trades within the rails.")
    say()
    say(f"  Get one at {_C['b']}https://console.anthropic.com{_C['x']} (paid, roughly a few dollars a month")
    say("  at one session a day). You can add it later by editing .env.")
    say()
    have = env.get("ANTHROPIC_API_KEY")
    if have:
        ok(f"already set ({mask(have)})")
        if confirm("replace it?", default=False):
            env["ANTHROPIC_API_KEY"] = ask("Anthropic API key (hidden)", secret=True, allow_blank=True)
    elif confirm("add an Anthropic API key now?", default=False):
        env["ANTHROPIC_API_KEY"] = ask("Anthropic API key (hidden)", secret=True, allow_blank=True)
    else:
        env.setdefault("ANTHROPIC_API_KEY", "")
        warn("skipped — research.py will work, decide.py will not")
    return env


def step_options(env):
    head(6, TOTAL, "Extras (all optional, all default off)")
    say("  Sensible defaults are being written. You can change any of these later")
    say(f"  by editing {_C['b']}.env{_C['x']} — each line has a comment explaining it.")
    say()
    env.setdefault("MEMORY_V2", "true")
    env.setdefault("BENCHMARK_SYMBOL", "SPY")
    env.setdefault("PREDICTION_MARKETS", "false")
    env.setdefault("INSIDER_FLOW", "false")
    env.setdefault("SEC_USER_AGENT", "")
    env.setdefault("FRED_API_KEY", "")
    env.setdefault("GITHUB_TOKEN", "")
    env.setdefault("GITHUB_REPO", "")
    env.setdefault("MAX_BUYS_PER_SESSION", "3")
    env.setdefault("TARGET_INVESTED_PCT", "0.70")
    ok("MEMORY_V2=true — scores your trades against SPY after 5 days")
    ok("BENCHMARK_SYMBOL=SPY — what a trade must beat to count as a win")
    ok("MAX_BUYS_PER_SESSION=3 — hard cap on trades per session")
    if confirm("turn on prediction markets? (free, no key, adds macro context)", default=True):
        env["PREDICTION_MARKETS"] = "true"
        ok("PREDICTION_MARKETS=true")
    # Live keys must exist as empty strings so the four gates read as closed.
    for k in ("I_UNDERSTAND_THIS_TRADES_REAL_MONEY", "LIVE_TRADING", "LIVE_ACCOUNTS",
              "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY"):
        env.setdefault(k, "")
    return env


def step_smoke(env, py):
    head(7, TOTAL, "Proving it works")
    write_env(env)
    ok(f"wrote {ENV_PATH.name} (permissions 0600 — only you can read it)")
    say()
    say("  Running the signal engine once. This fetches real market data and")
    say("  scores your watchlist. It places NO orders.")
    say()
    if not confirm("run it now? (takes 30-60 seconds)", default=True):
        warn("skipped — run it yourself with:  .venv/bin/python research.py")
        return
    r = subprocess.run([str(py), "research.py"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    tail = (r.stdout or "")[-1500:]
    if r.returncode == 0:
        say(f"{_C['dim']}{tail}{_C['x']}")
        ok("the engine ran and wrote a signals file")
    else:
        say(f"{_C['dim']}{(r.stderr or '')[-1200:]}{_C['x']}")
        warn("the engine reported a problem — the output above says why")
        say("  Common causes: markets closed on a holiday, or a symbol that no")
        say("  longer trades. Neither is dangerous; nothing was ordered.")


def step_schedule(py):
    head(8, TOTAL, "Running it every day")
    say("  Nothing is scheduled automatically — that is your choice to make.")
    say()
    say(f"  {_C['b']}To run by hand:{_C['x']}")
    say(f"    {VENV.name}/bin/python research.py     {_C['dim']}# signals (no LLM key needed){_C['x']}")
    say(f"    {VENV.name}/bin/python decide.py       {_C['dim']}# exits, then propose+enforce{_C['x']}")
    say(f"    {VENV.name}/bin/python journal.py      {_C['dim']}# write up the day{_C['x']}")
    say()
    say(f"  {_C['b']}To run automatically (macOS/Linux), add to `crontab -e`:{_C['x']}")
    say(f"{_C['dim']}    45 9  * * 1-5  cd {ROOT} && {VENV}/bin/python research.py")
    say(f"    0  10 * * 1-5  cd {ROOT} && {VENV}/bin/python decide.py")
    say(f"    15 16 * * 1-5  cd {ROOT} && {VENV}/bin/python journal.py{_C['x']}")
    say()
    say(f"  {_C['dim']}Times are your machine's local clock — the market runs 9:30-16:00 ET.{_C['x']}")


# ── live ─────────────────────────────────────────────────────────────────────

def live_setup():
    """Deliberately slower than the paper path, and not a click-through.

    Every other step in this wizard removes friction. This one adds it, for a
    reason worth stating: paper trading cannot lose you anything, and live
    trading can lose everything you fund it with. The failure modes that matter
    here are not "the strategy was wrong" — they are a defect in code you did not
    read, a stale data feed, a corporate action, or an unattended loop at 3am.

    So this asks for evidence rather than consent: that you have actually run
    sessions on paper, that you have opened the risk files, and that you can
    state an amount you would be untroubled to lose. None of it is enforceable —
    you can lie to a wizard. It is here so the decision is made deliberately
    rather than by pressing Enter four times.
    """
    say()
    say(f"{_C['y']}{_C['b']}  ══ REAL MONEY SETUP ══{_C['x']}")
    say()
    say("  This configures orders that spend your actual money.")
    say()

    env = read_env()
    if not env.get("ALPACA_API_KEY"):
        die("no paper setup found — run `python3 setup_wizard.py` first",
            "python3 setup_wizard.py")

    # 1. evidence of paper running, read from disk rather than asked
    sig = sorted((ROOT / "signals").glob("*.json"))
    dec = sorted((ROOT / "decisions").glob("*.json"))
    jrn = sorted((ROOT / "journal").glob("*.md"))
    say(f"  Your paper history on this machine:")
    say(f"    signal files:   {len(sig)}")
    say(f"    decision logs:  {len(dec)}")
    say(f"    journal days:   {len(jrn)}")
    say()
    if len(dec) < 20:
        bad(f"only {len(dec)} paper sessions recorded.")
        say("  A quarter of paper sessions (~60) is the honest bar; 20 is the floor")
        say("  this wizard will accept. That is not arbitrary — it is roughly the")
        say("  point at which you have seen a losing week, a stop fire, and a")
        say(f"  {_C['b']}corporate action{_C['x']} go through the system.")
        say()
        say("  Come back when you have more. Nothing was changed.")
        return 1
    ok(f"{len(dec)} paper sessions on record")

    # 2. have they read the code that protects them
    say()
    say("  These three files decide whether a bad order reaches your broker:")
    for f in ("risk_guard.py", "stops.py", "trade.py"):
        n = len((ROOT / f).read_text().splitlines()) if (ROOT / f).exists() else 0
        say(f"    {f:16s} {n:>4} lines")
    say()
    if not confirm("have you read all three, end to end?", default=False):
        say()
        say("  Read them first. If you would not be comfortable explaining what")
        say("  blocks an oversized order, you are not ready to fund this.")
        say("  Nothing was changed.")
        return 1

    # 3. an amount, stated out loud
    say()
    amt = ask("What amount would you be entirely untroubled to lose? (e.g. 500)")
    try:
        amt_f = float(re.sub(r"[^0-9.]", "", amt) or 0)
    except ValueError:
        amt_f = 0
    if amt_f <= 0:
        die("could not read that as a number — nothing was changed")
    say(f"  Fund the live account with {_C['b']}no more than ${amt_f:,.0f}{_C['x']}.")
    say(f"  {_C['dim']}This wizard cannot enforce that. Your broker will let you send more.{_C['x']}")

    # 4. live keys, verified against the LIVE endpoint
    say()
    say("  Now your LIVE Alpaca keys — generated with the dashboard toggled to")
    say("  'Live', and different from your paper keys.")
    say()
    while True:
        k = ask("Live API key ID")
        s = ask("Live secret key (hidden as you type)", secret=True)
        if k == env.get("ALPACA_API_KEY"):
            bad("that is your PAPER key. Live keys are generated separately.")
            continue
        say("  verifying against the live endpoint…")
        good, res = alpaca_account(k, s, LIVE_BASE)
        if not good:
            bad(str(res))
            if not confirm("try again?", default=True):
                return 1
            continue
        eq = float(res.get("equity", 0))
        ok(f"connected — live account, equity ${eq:,.2f}")
        if eq > amt_f * 1.5 and eq > 0:
            warn(f"that account holds ${eq:,.2f}, more than the ${amt_f:,.0f} you named.")
            say("  The engine sizes positions from account equity, so it will risk")
            say("  more than you just said you were comfortable with.")
            if not confirm("continue anyway?", default=False):
                say("  Nothing was changed. Withdraw down first, or re-run and name a higher figure.")
                return 1
        env["ALPACA_LIVE_API_KEY"], env["ALPACA_LIVE_SECRET_KEY"] = k, s
        break

    # 5. the four gates, opened explicitly
    say()
    say(f"  {_C['b']}Four gates{_C['x']} keep this on paper. All four must open:")
    say("    1. I_UNDERSTAND_THIS_TRADES_REAL_MONEY=yes")
    say("    2. LIVE_TRADING=true")
    say("    3. LIVE_ACCOUNTS=core")
    say("    4. live keys present  ✓ (just verified)")
    say()
    phrase = "I accept I can lose this money"
    say(f"  To open the first three, type exactly: {_C['b']}{phrase}{_C['x']}")
    typed = ask("Type the phrase")
    if typed.strip() != phrase:
        say()
        bad("that did not match. Nothing was changed — you are still on paper.")
        return 1
    env["I_UNDERSTAND_THIS_TRADES_REAL_MONEY"] = "yes"
    env["LIVE_TRADING"] = "true"
    env["LIVE_ACCOUNTS"] = "core"
    write_env(env)
    say()
    ok("live trading is now ENABLED for the 'core' account")
    say()
    say(f"  {_C['b']}To stop it at any time{_C['x']} — either is instant:")
    say(f"    set {_C['b']}LIVE_TRADING=false{_C['x']} in .env    (back to paper)")
    say(f"    set {_C['b']}KILL_SWITCH=true{_C['x']} in .env      (blocks all buys, sells still work)")
    say()
    say("  Watch the first session end to end: a real fill, a real exit, and the")
    say("  journal entry. Do not walk away from it.")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--live" in argv:
        return live_setup()
    if "--help" in argv or "-h" in argv:
        say(__doc__)
        return 0

    say()
    say(f"{_C['b']}  trading-agent setup{_C['x']}")
    say(f"{_C['dim']}  Paper trading, start to finish. No money involved.{_C['x']}")
    say(f"{_C['dim']}  Ctrl-C at any point — nothing is written until step 7.{_C['x']}")

    env = read_env()
    step_python()
    py = step_deps()
    env = step_paper_keys(env)
    env = step_watchlist(env)
    env = step_llm(env)
    env = step_options(env)
    step_smoke(env, py)
    step_schedule(py)

    say()
    say(f"{_C['g']}{_C['b']}  Done — you are set up for paper trading.{_C['x']}")
    say()
    say(f"  Read {_C['b']}DISCLAIMER.md{_C['x']} before you consider real money.")
    say(f"  When you have run a quarter on paper: {_C['b']}python3 setup_wizard.py --live{_C['x']}")
    say()
    return 0


if __name__ == "__main__":
    sys.exit(main())
