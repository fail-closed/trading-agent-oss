"""New state must be classified: backed up, or explicitly disposable.

On 2026-08-14 the audit fixes created four volume state files and none reached
`state_backup.STATE_FILES`. The worst was `frac_stops.json`: for a sub-1-share
position the broker holds NO stop order, so that file *is* the trailing stop —
a volume wipe plus restore would have silently reset every locked gain to the
−8% floor. Nothing in CI, the PR checklist or the /document-change skill asked
the question, and the same review found `iv_history.json` (40 days of
accumulated implied vol, not rebuildable) had been unprotected all along.

So the rule is now mechanical: every module-level state-file constant is either
in STATE_FILES or in EPHEMERAL_STATE with a reason. Both edits are cheap; what
is no longer possible is *not deciding*.
"""
import re
import subprocess

import pytest
from pathlib import Path

import state_backup

ROOT = Path(__file__).resolve().parent.parent

def _on_live_branch():
    """The baselines below describe MAIN's inventory. The live branch omits
    paper-only modules by design (and CI would fail on entries for files it
    doesn't have), so these self-skip there — same as tests/test_consistency.py.
    """
    import os
    if os.getenv("GITHUB_REF_NAME") == "live":
        return True
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() == "live"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    _on_live_branch(), reason="inventory baselines describe main; live diverges by design")


# `NAME = "something.json"` at module level is how this codebase names its state
# files (LEDGER_FILE, STATE_FILE, FRAC_STATE_FILE, HISTORY_FILE, CACHE_FILE …).
_CONST = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=\s*["\']([^"\']*\.jsonl?)["\']', re.M)

# …but not the ONLY way, and the gap was invisible until 2026-08-16. `_CONST`
# needs a quote immediately after the `=`, so every constant built by a call went
# unseen:
#
#   OUTCOMES_PATH = os.path.join(os.getenv("LEDGER_DIR", "signals"), "trade_outcomes.jsonl")
#   CACHE         = Path(os.getenv("LEDGER_DIR", "signals")) / "market_calendar.json"
#   OUT_PATH      = f"{os.getenv('ARTIFACT_PREFIX', '')}signals/trade_reasons.json"
#
# Six constants sat outside the registry's view, including BOTH memory_v2
# ledgers. Proven by probe, not by reading: deleting `signals/debate_outcomes.jsonl`
# from STATE_FILES left this suite green — the file was protected only because a
# human remembered, which is exactly what this test claims to make unnecessary.
#
# Same species as OPERATIONS §6 incident 11 — the check audited presence *in the
# form it could see*, not the actual population. This pattern therefore matches
# any module-level constant whose value ends in a quoted .json/.jsonl filename,
# however it was assembled.
# The basename is captured separately from any directory prefix INSIDE the quotes
# so an f-string path is caught too — `f"{prefix}signals/trade_reasons.json"` has
# no quote adjacent to the filename, and a basename-only pattern misses it.
_COMPUTED = re.compile(
    r'^([A-Z][A-Z0-9_]*)\s*=\s*.*?["\'][^"\']*?([^"\'/\\]+\.jsonl?)["\'][^\n]*$', re.M)

# Not state: fixed inputs that live in git, not on the volume.
IN_REPO = {"watchlist.json", "risk_limits.json", "package.json", "package-lock.json"}


def _declared_state_files():
    """{filename: [modules that declare it]} across tracked top-level modules."""
    found = {}
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    for name in files:
        if "/" in name or name.startswith("tests"):
            continue
        text = (ROOT / name).read_text()
        for pattern in (_CONST, _COMPUTED):
            for m in pattern.finditer(text):
                # Test the WHOLE assignment for a URL scheme, not the captured
                # filename. _COMPUTED captures only the basename, so a URL
                # template like SEC_SUBMISSIONS =
                # "https://data.sec.gov/submissions/CIK{cik}.json" arrives here
                # as "CIK{cik}.json" with the scheme already discarded — checking
                # the capture alone reported two SEC endpoints as unbacked state.
                if "://" in m.group(0):
                    continue
                base = Path(m.group(2)).name  # constants may carry a dir prefix
                if base not in IN_REPO:
                    found.setdefault(base, []).append(name)
    return found


def _backed_up_names():
    return {Path(p).name for p in state_backup.STATE_FILES}


def test_every_state_file_is_backed_up_or_declared_disposable():
    unclassified = {}
    for fname, modules in sorted(_declared_state_files().items()):
        if fname in _backed_up_names() or fname in state_backup.EPHEMERAL_STATE:
            continue
        unclassified[fname] = modules
    assert not unclassified, (
        "State files nobody has classified:\n  " +
        "\n  ".join(f"{f} (declared in {', '.join(m)})" for f, m in unclassified.items()) +
        "\n\nDecide which it is:\n"
        "  • carries DECISION state (a stop level, an owed order, an accumulated\n"
        "    history that cannot be refetched) → add it to state_backup.STATE_FILES\n"
        "  • a cache or a regenerable artifact → add it to EPHEMERAL_STATE with the\n"
        "    reason losing it is harmless.\n"
        "frac_stops.json is why this test exists: for fractional positions the "
        "broker holds no stop order, so the file IS the stop."
    )


def test_registry_has_no_stale_entries():
    declared = set(_declared_state_files())
    # ledger.py builds these two inline rather than as constants, so name them.
    declared |= {"recon_baseline.json", "recon_snapshot.json", "orders.jsonl"}
    stale = [f for f in state_backup.EPHEMERAL_STATE if f not in declared]
    assert not stale, (f"EPHEMERAL_STATE lists {stale}, which no module declares "
                       "any more — delete the entries so the registry keeps meaning "
                       "something.")


def test_a_file_is_never_both_backed_up_and_disposable():
    both = _backed_up_names() & set(state_backup.EPHEMERAL_STATE)
    assert not both, f"{both} is declared both backed-up and disposable"


def test_decision_state_stays_backed_up():
    """The specific files whose loss has a known cost. Named individually so the
    failure message says what breaks, not just that a list changed."""
    required = {
        "signals/frac_stops.json": "fractional trailing locks — the broker holds no stop for these",
        "signals/pending_ladders.json": "tranche 2s owed on a late fill",
        "signals/recon_snapshot.json": "drift baseline — else drift is silently re-absorbed",
        "signals/iv_history.json": "40+ days of accumulated IV; not refetchable, only re-earned",
        "signals/orders.jsonl": "the order audit trail — system of record",
        "signals/risk_state.json": "halt anchors and profit-lock state",
    }
    missing = {f: why for f, why in required.items() if f not in state_backup.STATE_FILES}
    assert not missing, f"decision state dropped from the backup list: {missing}"
