"""Global test isolation.

Tests must never touch production surfaces. This was violated on 2026-08-13:
running the suite locally (where .env supplies a real GITHUB_TOKEN) let
job_audit._publish_today push its "a"/"b" test fixtures to GitHub, OVERWRITING
status/jobs-paper-2026-08-13.json — the real record of that day's production
job runs — because the test monkeypatched AUDIT_DIR but not the publish path.
The dashboard then showed two fixture rows as the day's history.

One autouse fixture that strips the write-capable credentials is sturdier than
remembering to mock every publisher in every test: with no token, every GitHub
writer in the codebase (job_audit, journal, state_backup, push_signals,
stop_monitor, backfill_journals, ...) degrades to its no-op branch by design.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_production_side_effects(monkeypatch):
    # No GitHub writes — every publisher no-ops without a token.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # No Postgres writes — db_mirror and the singleton lock no-op without a DSN.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # No paid LLM calls from a stray unit test.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
