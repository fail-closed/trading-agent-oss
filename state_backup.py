"""
state_backup.py — off-box backup of the real-money state files.

The order ledger and risk state live ONLY on the Railway volume, which has no
automatic backups. If the volume is lost/corrupted/detached, the audit trail for
real money is gone and there is no reconstruction path. This script pushes those
small files to the (private) GitHub repo under `${ARTIFACT_PREFIX}state/`, giving
point-in-time recovery, and can restore any that are missing locally after a
volume wipe.

Files backed up (all gitignored, volume-only):
  signals/orders.jsonl        — append-only order audit trail (system of record)
  signals/risk_state.json     — daily-loss anchors / halts / profit-lock
  signals/recon_baseline.json — reconcile baseline (else drift gets re-absorbed)
  signals/recon_snapshot.json — last position snapshot (else qty/basis drift is
                                silently re-absorbed — same reason as above)
  signals/frac_stops.json     — high-water trailing locks for FRACTIONAL positions.
                                Losing this resets every locked gain to the -8%
                                floor: the broker holds no order for these, so
                                this file IS the stop
  signals/pending_ladders.json— tranche 2s owed on a late tranche-1 fill
  signals/cash_history.json   — 90-day deployment tracking

Gated by STATE_BACKUP=true (default off → no-op, so paper is unchanged until you
opt in; the live project sets STATE_BACKUP=true). Best-effort: never raises, so
it can never break the trading pipeline.

  python3 state_backup.py             # back up present files to GitHub
  python3 state_backup.py --restore   # pull down ONLY files missing locally
"""
import os
import sys

# Volume-only state files worth protecting (relative to the app working dir).
STATE_FILES = [
    "signals/orders.jsonl",
    "signals/risk_state.json",
    "signals/recon_baseline.json",
    "signals/cash_history.json",
    "signals/entry_types.json",
    "signals/pending_ladders.json",
    "signals/trade_outcomes.jsonl",   # realized trade-outcome scorecard (memory_v2)
    # Added with the 2026-08-14 audit fixes. These carry DECISION state that no
    # broker order mirrors — the fractional locks especially: for a sub-1-share
    # position the broker holds no stop, so frac_stops.json is the stop. A volume
    # wipe without these silently un-protects every fractional gain and forgets
    # every owed tranche. stopmon_publish.json is deliberately NOT here: it only
    # paces a dashboard heartbeat, and losing it costs one extra commit.
    "signals/recon_snapshot.json",
    "signals/frac_stops.json",
    # iv_history.json accumulates ONE snapshot per day and needs MIN_HISTORY_DAYS
    # (40) before iv_rank stops returning None. Option chains are not retrievable
    # historically here, so a wipe means ~2 months of blind iv_rank — it cannot
    # be rebuilt, only re-earned.
    "signals/iv_history.json",
    # debate_outcomes.jsonl is the ONLY record of what the bull/bear gate decided
    # and what those names then did — `decisions/` is volume-only and never
    # pushed, so a wipe restarts the counterfactual from zero and the gate goes
    # back to being unmeasurable for another 20 sessions.
    "signals/debate_outcomes.jsonl",
    # Cumulative PAID-API spend against a hard $30 cap (databento_client.py).
    # Surfaced 2026-08-16 when the registry check was widened to see computed
    # constants. Losing it resets `spent_usd` to 0, silently re-arming a budget
    # we have already partly consumed — the only file here whose loss costs
    # actual money rather than information.
]

# Volume files we deliberately do NOT back up, and why. Every state-file
# constant in the codebase must appear either in STATE_FILES above or here —
# tests/test_state_registry.py enforces that, so a new feature cannot quietly
# invent state whose loss nobody has thought about. (That is exactly what
# happened on 2026-08-14: four new state files, none backed up, including the
# one that IS the stop for fractional positions.)
EPHEMERAL_STATE = {
    "corporate_actions_cache.json": "cache — refetched on demand",
    "stopmon_publish.json": "paces the dashboard heartbeat; losing it costs one extra commit",
    # ⚠ NEVER move this one to STATE_FILES. state_backup pushes to a GitHub
    # repo, and this is an OAuth credential — backing it up would publish a
    # token. It is listed here to record that the decision was made, not
    # overlooked. (The refresh token is dead anyway, OPERATIONS §4.)
    "market_calendar.json": "holiday cache — refetched from the calendar API on demand",
    "trade_reasons.json": "published GitHub artifact, regenerated from decisions/ each run",
    "earnings_cache.json": "cache — refetched on demand",
    "fundamentals_cache.json": "cache — refetched on demand",
    "insider_cache.json": "cache — refetched on demand",
    "sec_cik_map.json": "cache — refetched from SEC on demand",
}

REPO = "YOUR_GH_USER/YOUR_REPO"


def _enabled() -> bool:
    return os.getenv("STATE_BACKUP", "").strip().lower() in ("1", "true", "yes", "on")


def _repo():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set — state backup skipped.", file=sys.stderr)
        return None
    from github import Github
    return Github(token).get_repo(REPO)


def _remote_path(local_path: str) -> str:
    # ARTIFACT_PREFIX keeps the live project's backups separate from paper's.
    prefix = os.getenv("ARTIFACT_PREFIX", "")
    return f"{prefix}state/{os.path.basename(local_path)}"


def backup() -> int:
    """Push every present state file to GitHub. Returns count pushed."""
    if not _enabled():
        print("STATE_BACKUP not enabled — no-op.")
        return 0
    repo = _repo()
    if repo is None:
        return 0
    from github import GithubException
    pushed = 0
    for local_path in STATE_FILES:
        if not os.path.exists(local_path):
            continue
        try:
            with open(local_path) as f:
                content = f.read()
            remote = _remote_path(local_path)
            msg = f"state backup: {os.path.basename(local_path)}"
            try:
                existing = repo.get_contents(remote)
                # Skip the API write if content is byte-identical (avoids churn).
                if existing.decoded_content.decode("utf-8") == content:
                    continue
                repo.update_file(remote, msg, content, existing.sha)
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(remote, msg, content)
                else:
                    raise
            pushed += 1
        except Exception as e:
            print(f"  [state_backup] {local_path} failed: {e}", file=sys.stderr)
    print(f"State backup: {pushed} file(s) pushed to GitHub.")
    return pushed


def restore() -> int:
    """
    Pull state files that are MISSING locally back from GitHub. Only fills gaps —
    never overwrites a file that exists on the volume (so a healthy volume is
    never clobbered by a stale backup). Returns count restored.
    """
    repo = _repo()
    if repo is None:
        return 0
    from github import GithubException
    restored = 0
    for local_path in STATE_FILES:
        if os.path.exists(local_path):
            continue  # volume has it — never overwrite
        remote = _remote_path(local_path)
        try:
            content = repo.get_contents(remote).decoded_content.decode("utf-8")
        except GithubException as e:
            if e.status == 404:
                continue
            print(f"  [state_backup] restore {local_path} failed: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  [state_backup] restore {local_path} failed: {e}", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        from io_utils import write_text_atomic
        write_text_atomic(local_path, content)
        restored += 1
        print(f"  RESTORED {local_path} from backup ({len(content)} bytes).")
    if restored:
        print(f"State restore: {restored} file(s) recovered from GitHub backup.")
    else:
        print("State restore: nothing missing locally (volume intact) — no-op.")
    return restored


def main():
    if "--restore" in sys.argv:
        restore()
    else:
        backup()


if __name__ == "__main__":
    main()
