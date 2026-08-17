"""Every module is either exercised by a test or consciously exempted.

The gap this closes (2026-08-14): nothing required a new feature to ship with a
test. Proved by probe — a new module with real logic and a new env knob passed
`compileall` + the whole suite with no objection. Every test written during the
audit-fix work was discipline, not process; a future session (or an agent PR)
could ship untested trade-path code and CI would stay green.

This does NOT demand a test file per module — that would be bureaucracy. A
module counts as covered when ANY test imports it (or names its filename, for
the source-scanning tripwires). What it demands is that leaving a module
untested be a DECISION someone wrote down, not an oversight.

Adding a module to NO_TESTS is that decision. Keep the reason specific and
honest — "dormant sleeve" and "manual offline tool" are real reasons; "will add
later" is what created this problem.
"""
import re
import subprocess

import pytest
from pathlib import Path

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


# Modules with no test coverage, and why. Baselined 2026-08-14 by measurement,
# not by assumption; the four TODO-marked entries (options_trade, daytrade_flatten,
# dip_scorer, breakout_scorer) were paid off the same day — 55/82 covered now.
# Everything left is a dormant sleeve, a manual offline tool, or a vendor shim.
NO_TESTS = {
    "walk_forward": "manual walk-forward validation harness; needs sklearn + a built feature\n                     frame to mean anything, and CI deliberately omits sklearn",
    # ── Not covered in this public extraction, and why ───────────────────────
    # These modules ARE tested in the private system this was extracted from;
    # their tests depend on infrastructure (scheduler, dashboard, vendor keys)
    # that is deliberately not shipped. Listed rather than silently dropped.
    "alerts": "notification fan-out; its tests need the scheduler/bridge harness not shipped here",
    "io_utils": "atomic-write helper; covered upstream by tests requiring the scheduler harness",
    "data_client": "bar-provider fallback chain; needs live vendor keys to exercise meaningfully",
    "macro_context": "LLM macro brief; the deterministic part it gained (prediction_markets) IS tested",
    "fundamentals": "yfinance fundamentals cache; network-bound, fail-open by construction",
    "market_calendar": "holiday cache over a public API; network-bound",
    # ── Dormant sleeves: machinery kept, execution OFF ────────────────────────

    # ── Offline / manual analysis tools (never on the trade path) ─────────────

    # ── External clients: thin API wrappers, nothing to test without the API ──

    # ── Scoring inputs — advisory only, not yet in the composite score ────────
    "microstructure": "advisory modifiers only (CLAUDE.md: not in score yet)",

    # ── Thin orchestration over tested parts ─────────────────────────────────
}


# Directories whose Python is not application code we require tests for.
NOT_APP_CODE = ("tests/", "dashboard/static/", "design-system/", "bridge/",
                "tradingview-mcp/", "scripts/")


def _modules():
    """Every application module, top-level OR packaged, as an import path.

    Was `if "/" not in f` — top level only. That made the whole tripwire
    conditional on a layout choice: any module moved into a package silently
    stopped being covered, and the suite stayed green because an absent check and
    a passing check are indistinguishable (OPERATIONS §8 habit 9). Probed
    2026-08-16 before this change: a packaged module with real logic and no test
    passed 7/7. It also meant `strategies/` — which already existed — was never
    checked at all.

    Returned as dotted import paths (`strategies.catalog`) so _is_covered can
    match how a test would actually import them.
    """
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    mods = []
    for f in out:
        if f.startswith(NOT_APP_CODE) or f.endswith("__init__.py"):
            continue
        mods.append(f[:-3].replace("/", "."))
    return sorted(mods)


def _test_sources():
    return [p.read_text() for p in (ROOT / "tests").glob("*.py")]


def _is_covered(module, sources):
    """Covered = some test imports it, or names its filename (the source-scanning
    tripwires like test_stop_rule.py reference modules that way).

    `module` is a dotted path. A packaged module can be imported three ways —
    `import a.b`, `from a.b import x`, `from a import b` — and the last one names
    only the leaf, so all three are matched. Path-form references
    ("strategies/catalog.py") count too."""
    dotted = re.escape(module)
    leaf = module.rsplit(".", 1)[-1]
    pkg = module.rsplit(".", 1)[0] if "." in module else None
    path = re.escape(module.replace(".", "/"))
    for text in sources:
        if re.search(rf"^\s*(?:import|from)\s+{dotted}\b", text, re.M):
            return True
        if pkg and re.search(rf"^\s*from\s+{re.escape(pkg)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b",
                             text, re.M):
            return True
        if re.search(rf'["\']{path}\.py["\']', text):
            return True
    return False


def test_every_module_is_tested_or_consciously_exempted():
    sources = _test_sources()
    orphans = [m for m in _modules() if not _is_covered(m, sources) and m not in NO_TESTS]
    assert not orphans, (
        "These modules have no test exercising them:\n  " + "\n  ".join(orphans) +
        "\n\nAdd a test (one that would FAIL if the module broke — not an import "
        "smoke test), or add the module to NO_TESTS in this file with a specific "
        "reason. A probe on 2026-08-14 proved an untested module with real logic "
        "passes CI silently; this test is the only thing standing between that "
        "and the trade path."
    )


def test_exemptions_stay_honest():
    """An exemption for a module that now HAS tests is stale bookkeeping, and a
    stale allowlist is how an allowlist becomes a rubber stamp."""
    sources = _test_sources()
    stale = [m for m in NO_TESTS if _is_covered(m, sources)]
    assert not stale, (f"{stale} are exempted but now have tests — remove them "
                       "from NO_TESTS so the list keeps meaning something.")

    # Dotted paths, since modules may live in packages — `studies.backtest`
    # is studies/backtest.py on disk, and the old f"{m}.py" reported every
    # packaged exemption as a stale entry the moment anything moved.
    missing = [m for m in NO_TESTS
               if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
    assert not missing, (f"{missing} are exempted but no longer exist — delete "
                         "the entries.")


def test_trade_path_modules_are_never_exempt():
    """The modules that can move real money must always be tested. This list is
    deliberately hard to change: if you are adding to it, say why in the PR."""
    trade_path = {"trade", "decide", "risk_guard", "accounts", "ledger", "stops",
                  "stop_monitor", "ladder", "research"}
    exempted = trade_path & set(NO_TESTS)
    assert not exempted, (
        f"{exempted} sit on the order path and cannot be exempted from testing.")
    sources = _test_sources()
    uncovered = [m for m in sorted(trade_path) if not _is_covered(m, sources)]
    assert not uncovered, f"{uncovered} are on the order path and have no test."
