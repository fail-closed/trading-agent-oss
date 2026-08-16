"""
accounts.py — registry mapping each strategy to its own Alpaca paper account.

Three isolated paper accounts, one per strategy, so each account's natively
reported equity IS that strategy's performance — no co-mingling, no attribution
math:

  core     — swing/position equity (the ORIGINAL account). Uses the unprefixed
             ALPACA_API_KEY / ALPACA_SECRET_KEY so all existing behavior is
             unchanged if the new accounts aren't set up yet.
  options  — long-premium options sleeve, its own account + capital.
  daytrade — intraday momentum/breakout, its own account (funded $25k to stay
             clear of the Pattern Day Trader rule).

Keys come from env vars (set in Railway once the user creates the paper accounts
in the Alpaca dashboard and generates each one's key pair). Only the user can
create the accounts; this module just routes to whichever keys exist.

Usage:
    import accounts
    tc = accounts.trading_client("daytrade")     # raises if keys missing
    if accounts.available("options"): ...        # check before using
"""
import os

# account name → (api_key_env, secret_key_env) for PAPER
ACCOUNTS = {
    "core":     ("ALPACA_API_KEY",     "ALPACA_SECRET_KEY"),
    "options":  ("ALPACA_OPT_API_KEY", "ALPACA_OPT_SECRET_KEY"),
    "daytrade": ("ALPACA_DT_API_KEY",  "ALPACA_DT_SECRET_KEY"),
    # Staging slots for parallel paper R&D — each runs a candidate strategy on
    # its own paper account (strategy_map.json). No LIVE_KEYS entry → can NEVER
    # go live; they exist only to test strategies head-to-head before promotion.
    "staging1": ("ALPACA_STG1_API_KEY", "ALPACA_STG1_SECRET_KEY"),
    "staging2": ("ALPACA_STG2_API_KEY", "ALPACA_STG2_SECRET_KEY"),
    "staging3": ("ALPACA_STG3_API_KEY", "ALPACA_STG3_SECRET_KEY"),
}

# Separate LIVE key envs — only used when an account is explicitly promoted live.
LIVE_KEYS = {
    "core":     ("ALPACA_LIVE_API_KEY",     "ALPACA_LIVE_SECRET_KEY"),
    "options":  ("ALPACA_OPT_LIVE_API_KEY", "ALPACA_OPT_LIVE_SECRET_KEY"),
    "daytrade": ("ALPACA_DT_LIVE_API_KEY",  "ALPACA_DT_LIVE_SECRET_KEY"),
}


def _live_accounts() -> set:
    return {a.strip() for a in os.getenv("LIVE_ACCOUNTS", "").split(",") if a.strip()}


def data_feed():
    """Market-data feed for all historical/quote requests. With Alpaca Elite the
    keys are entitled to the full SIP consolidated tape; set ALPACA_DATA_FEED=sip
    to use it (more accurate volume/ADV, real-time quotes). Defaults to IEX so
    nothing breaks if the entitlement is missing."""
    from alpaca.data.enums import DataFeed
    return DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "iex").lower() == "sip" else DataFeed.IEX


def carved_out() -> set:
    """Symbols the agent must COMPLETELY ignore — never buy, never sell, never
    treat as a managed position, never reconcile-alarm on. Used to fence off a
    pre-existing holding that is NOT under strategy control (e.g. the live
    account's legacy QQQ position). Set via CARVE_OUT_SYMBOLS (comma-separated,
    case-insensitive); empty by default so the paper project is unaffected. The
    live project sets CARVE_OUT_SYMBOLS=QQQ. Enforced at trade.py (hard block),
    research.py / decide.py (filtered out of positions), and reconcile."""
    return {s.strip().upper() for s in os.getenv("CARVE_OUT_SYMBOLS", "").split(",") if s.strip()}


def effective_max_allocation(watchlist_entry: dict, account: str = "core") -> float:
    """THE per-stock allocation cap, as a fraction of investable equity.

    Single source of truth. This rule used to be computed inline in three
    places — research.py (sizing), trade.py (enforcement), decide.py (ladder
    tranche 2) — and on 2026-08-13 MAX_ALLOC_SCALE was added to the sizer only:
    research sized live's buys at 2× while trade.py still enforced 1×, so the
    sizer wrote orders the gate rejected. A rule that exists in several files
    WILL drift; every consumer now calls this, and a tripwire test fails if any
    file outside accounts.py references MAX_ALLOC_SCALE again.

    The ceiling comes from risk_guard's resolved limits (defaults <
    risk_limits.json < env), so the sizer can never propose what the guard
    would refuse — previously the inline copies read the env var directly and
    would have ignored a risk_limits.json override.
    """
    scale = float(os.getenv("MAX_ALLOC_SCALE", "1.0"))
    try:
        import risk_guard
        cap = float(risk_guard._limits(account)["max_position_pct"])
    except Exception:
        cap = float(os.getenv("RISK_MAX_POSITION_PCT", "0.10"))
    return min(watchlist_entry["max_allocation"] * scale, cap)


def is_live(account: str) -> bool:
    """
    REAL-MONEY GATE. An account trades live ONLY when ALL hold:
      0. I_UNDERSTAND_THIS_TRADES_REAL_MONEY=yes  (public-distribution gate)
      1. LIVE_TRADING=true  (global master switch, default off)
      2. account is listed in LIVE_ACCOUNTS  (explicit per-account opt-in)
      3. that account's live key pair is set
    Any missing condition → paper. This makes going live deliberate and
    impossible by accident.
    """
    # Fourth gate, added for the public distribution. The three below were
    # written for an operator who set up their own account and knows what
    # LIVE_TRADING does. A stranger cloning this repo has not made that choice
    # yet, and a copied .env or a tutorial that says "set LIVE_TRADING=true"
    # should not be one step away from real orders. This costs an informed
    # operator ten seconds and makes the accident considerably harder.
    if os.getenv("I_UNDERSTAND_THIS_TRADES_REAL_MONEY", "").lower() != "yes":
        return False
    if os.getenv("LIVE_TRADING", "false").lower() != "true":
        return False
    if account not in _live_accounts():
        return False
    kf, sf = LIVE_KEYS.get(account, (None, None))
    return bool(kf and os.getenv(kf) and os.getenv(sf))


def get_keys(account: str = "core") -> tuple:
    """Return (api_key, secret_key) for a named account — live keys if the account
    is promoted live (is_live), else paper keys. Raises a clear error if missing."""
    if account not in ACCOUNTS:
        raise ValueError(f"unknown account {account!r}; valid: {list(ACCOUNTS)}")
    kf, sf = (LIVE_KEYS if is_live(account) else ACCOUNTS)[account]
    key, sec = os.getenv(kf), os.getenv(sf)
    if not key or not sec:
        raise RuntimeError(
            f"Account '{account}' keys missing — set {kf} and {sf}.")
    return key, sec


def available(account: str) -> bool:
    """True if usable keys exist for this account (live keys if promoted, else paper)."""
    kf, sf = (LIVE_KEYS if is_live(account) else ACCOUNTS).get(account, (None, None))
    return bool(kf and os.getenv(kf) and os.getenv(sf))


def trading_client(account: str = "core"):
    """TradingClient for a named account — paper=False ONLY when the account is
    promoted live via the is_live() gate; paper=True otherwise."""
    from alpaca.trading.client import TradingClient
    key, sec = get_keys(account)
    return TradingClient(key, sec, paper=not is_live(account))


def data_keys() -> tuple:
    """Market-data keys — any valid pair works; prefer core, fall back to any set."""
    for acct in ("core", "options", "daytrade"):
        if available(acct):
            return get_keys(acct)
    raise RuntimeError("No Alpaca keys set for any account")


def net_contributions(trading_client, since_date: str) -> float:
    """Net external cash flow (deposits − withdrawals) posted STRICTLY AFTER
    `since_date` (YYYY-MM-DD), from the account's CSD/CSW/JNLC activities.

    A deposit adds equity but isn't a trading gain; a withdrawal removes equity
    but isn't a loss. Callers net this out of period-return calculations so cash
    flows are never mistaken for P&L (e.g. the monthly profit-lock). Hits the REST
    activities endpoint directly (the SDK request class isn't in this alpaca-py).
    Best-effort: returns 0.0 on any error or when there are no activities (paper
    accounts have none), i.e. it fails to the unadjusted behaviour."""
    try:
        import requests
        base = str(getattr(trading_client._base_url, "value", trading_client._base_url)).rstrip("/")
        k = getattr(trading_client, "_api_key", None)
        s = getattr(trading_client, "_secret_key", None)
        if not (base and k and s):
            return 0.0
        r = requests.get(f"{base}/v2/account/activities",
                         headers={"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s},
                         params={"activity_types": "CSD,CSW,JNLC"}, timeout=15)
        if r.status_code != 200:
            return 0.0
        net = 0.0
        for a in r.json():
            d = str(a.get("date") or a.get("transaction_time") or "")[:10]
            amt = float(a.get("net_amount") or 0)
            if d and d > since_date:
                net += amt
        return net
    except Exception:
        return 0.0
