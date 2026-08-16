"""
alerts.py — push event alerts to WhatsApp via the bridge service.

The whatsapp-bridge service exposes POST /send {text} which delivers a
message to the home WhatsApp channel. On Railway the trading agent reaches
it over private networking (BRIDGE_URL=http://whatsapp-bridge.railway.internal:8765).

Alerts are best-effort and must NEVER break the trading pipeline: if
BRIDGE_URL is unset (e.g. local runs) or the bridge is down, send() quietly
returns False.

Alert set (high-value only — events, not noise):
  - trades executed / rejected (decide.py, end of each session)
  - mechanical exits: stop-loss / take-profit (decide.py, options_trade.py)
  - price gap >30% vs prior close — possible unadjusted split (research.py)
  - scheduler job failure or timeout (scheduler.py)
"""
import os
import sys

import requests

BRIDGE_URL = os.getenv("BRIDGE_URL", "").rstrip("/")
# Shared secret for the bridge's token-gated /send (set when the bridge is
# exposed on a public domain so the live project can post over the internet).
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")

# Second channel (redundancy): a generic webhook (Discord/Slack-style) and/or
# Telegram. The self-hosted WhatsApp bridge desyncs (Baileys) — these give a
# fallback path so a failure alert still reaches you when the bridge is down.
# All default-unset → no behavior change until configured.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()   # Discord/Slack/etc.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _ok(resp) -> bool:
    """A 2xx is the ONLY evidence a channel accepted the message.

    Every sender below used to `requests.post(...)` and unconditionally return
    True, so an HTTP 500 read as delivered. That is exactly how five weeks of
    daily 'Job failed' alerts vanished in 2026-07/08: the bridge process was up
    and returning 500 ({"ok":false,"error":"Connection Closed"}) because its
    Baileys WhatsApp socket was dead, and alerts.send() reported success on
    every one. Never report delivery without checking the status."""
    if resp.status_code // 100 != 2:
        body = (resp.text or "")[:160].replace("\n", " ")
        raise RuntimeError(f"HTTP {resp.status_code} — {body}")
    return True


def _send_bridge(text: str) -> bool:
    if not BRIDGE_URL:
        return False
    headers = {"x-bridge-token": BRIDGE_TOKEN} if BRIDGE_TOKEN else None
    return _ok(requests.post(f"{BRIDGE_URL}/send", json={"text": text},
                             headers=headers, timeout=10))


def _send_webhook(text: str) -> bool:
    if not ALERT_WEBHOOK_URL:
        return False
    # "content" satisfies Discord, "text" satisfies Slack/Mattermost — send both.
    return _ok(requests.post(ALERT_WEBHOOK_URL, json={"content": text, "text": text},
                             timeout=10))


def _send_telegram(text: str) -> bool:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False
    return _ok(requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10))


def send(text: str) -> bool:
    """
    Fan an alert out to every configured channel (WhatsApp bridge + webhook +
    Telegram). Best-effort and independent: one channel failing never stops the
    others, and the call never raises. Returns True if ANY channel delivered.
    """
    if not text:
        return False
    delivered, configured, errors = False, 0, []
    for name, fn in (("bridge", _send_bridge), ("webhook", _send_webhook), ("telegram", _send_telegram)):
        try:
            if fn(text):
                delivered = True
                configured += 1
        except Exception as e:
            configured += 1
            errors.append(f"{name}: {e}")
            print(f"  [alerts] {name} send failed: {e}", file=sys.stderr)
    if configured and not delivered:
        # Loud, because the thing that just failed is how you find out things fail.
        print(f"  [alerts] ⚠️  ALERT UNDELIVERED on all {configured} channel(s) — "
              f"{'; '.join(errors) or 'no channel accepted'}", file=sys.stderr)
    return delivered
