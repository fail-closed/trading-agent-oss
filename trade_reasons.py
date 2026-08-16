"""
trade_reasons.py — publish a compact per-symbol buy-rationale digest to GitHub.

The dashboard's Trade Ledger (/v2/ledger) shows WHY each stock was bought, but the
dashboard has no local files — it reads from GitHub. decide.py writes the full LLM
rationale to decisions/{date}_{HHMM}.json, but those aren't pushed. This aggregates
them into one small signals/trade_reasons.json (latest executed BUY reason per symbol)
and pushes it, so the ledger can show the thesis + whether it played out.

Runs daily from journal.py (after decide has written the day's decisions). Idempotent.

  python3 trade_reasons.py            # build + push the digest now (also a backfill)
"""
import glob
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
REPO_NAME = "YOUR_GH_USER/YOUR_REPO"
# Namespace by ARTIFACT_PREFIX (live project sets "live/") so a live run never
# clobbers the paper digest on GitHub. Paper → signals/trade_reasons.json.
OUT_PATH = f"{os.getenv('ARTIFACT_PREFIX', '')}signals/trade_reasons.json"


def build(decisions_glob: str = "decisions/*.json") -> dict:
    """Latest executed BUY rationale per symbol from all decision logs (chronological,
    so the most recent buy thesis wins). Returns {date, reasons: {SYMBOL: {...}}}."""
    reasons = {}
    for f in sorted(glob.glob(decisions_glob)):   # filename sorts chronologically
        try:
            d = json.load(open(f))
        except Exception:
            continue
        date, time = d.get("date"), d.get("time")
        for dec in d.get("decisions", []):
            if dec.get("action") == "BUY" and dec.get("executed"):
                reasons[dec["symbol"].upper()] = {
                    "date": date, "time": time, "action": "BUY",
                    "reason": dec.get("reason", ""),
                    "entry_price": dec.get("entry_price"),
                }
    return {"generated_at": datetime.now(ET).isoformat(),
            "count": len(reasons), "reasons": reasons}


def _push(content: str):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set — skipping push", file=sys.stderr)
        return
    from github import Github, GithubException
    repo = Github(token).get_repo(REPO_NAME)
    try:
        existing = repo.get_contents(OUT_PATH)
        if existing.decoded_content.decode("utf-8") != content:
            repo.update_file(OUT_PATH, "trade reasons digest", content, existing.sha)
    except GithubException as e:
        if getattr(e, "status", None) == 404:
            repo.create_file(OUT_PATH, "trade reasons digest", content)
        else:
            raise


def main():
    digest = build()
    content = json.dumps(digest, indent=2)
    os.makedirs("signals", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(content)
    print(f"trade_reasons: {digest['count']} symbols → {OUT_PATH}")
    try:
        _push(content)
        print("  pushed to GitHub")
    except Exception as e:
        print(f"  push failed (non-fatal): {str(e)[:100]}", file=sys.stderr)


if __name__ == "__main__":
    main()
