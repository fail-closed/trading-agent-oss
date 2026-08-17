"""The artifact repo slug has ONE owner, and deliberately NO default.

Upstream this literal appeared in 18 modules. Two consequences, and the second is
why this file ships in a public repo at all:

  A value repeated in 18 places cannot be changed safely — renaming the GitHub
  account meant editing all 18 correctly while a live system read and wrote that
  path on a schedule.

  GitHub frees a username the moment it is renamed, and anyone may claim it. A
  clone holding a valid token and a stale slug would push its trading history
  into a stranger's repo, and read instructions back out of one.

So this repo ships no default at all. An unconfigured clone publishes nothing
rather than quietly aiming at somebody else's repository.
"""
import importlib

import pytest

import artifact_repo

MODULES_THAT_PUBLISH = ["state_backup.py", "trade_reasons.py", "stop_monitor.py"]


def test_there_is_no_default_repo(monkeypatch):
    """The whole point. A built-in slug would aim a fresh clone at upstream."""
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    importlib.reload(artifact_repo)
    assert artifact_repo.name() == ""
    assert not artifact_repo.configured()


def test_configured_needs_both_token_and_repo(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "you/your-repo")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    importlib.reload(artifact_repo)
    assert not artifact_repo.configured(), "a repo without a token is not configured"
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    assert artifact_repo.configured()


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_repo_is_not_a_repo(monkeypatch, blank):
    monkeypatch.setenv("GITHUB_REPO", blank)
    importlib.reload(artifact_repo)
    assert artifact_repo.name() == ""


def test_require_explains_what_to_set(monkeypatch):
    """Failing at the check beats failing inside `get_repo("")`, three frames
    deep in the publish path and far from the misconfiguration."""
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    importlib.reload(artifact_repo)
    with pytest.raises(RuntimeError, match="GITHUB_REPO"):
        artifact_repo.require()


def test_no_module_hardcodes_a_repo_slug():
    """Including the placeholder this repo shipped with. `YOUR_GH_USER/YOUR_REPO`
    in three files was a scrubbed name, not a solved problem — it would still
    have been three places to edit, and it read as a config value rather than as
    something you MUST set."""
    import re
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    slug = re.compile(r'["\'][A-Za-z0-9_][\w.\-]*/[\w.\-]+["\']\s*(?:\)|$|,)')
    offenders = {}
    for rel in subprocess.run(["git", "ls-files", "*.py"], cwd=root,
                              capture_output=True, text=True).stdout.split():
        if rel.startswith("tests/") or Path(rel).name == "artifact_repo.py":
            continue
        hits = [h for h in slug.findall((root / rel).read_text())
                if "YOUR_GH_USER" in h or "trading-agent" in h]
        if hits:
            offenders[rel] = hits
    assert not offenders, f"hardcoded repo slugs: {offenders} — call artifact_repo.name()"


def test_publishers_route_through_the_helper():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    missing = [f for f in MODULES_THAT_PUBLISH
               if (root / f).exists() and "artifact_repo." not in (root / f).read_text()]
    assert not missing, f"these publish artifacts but never call the helper: {missing}"
