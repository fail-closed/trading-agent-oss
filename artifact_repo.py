"""
artifact_repo.py — the ONE owner of which GitHub repo receives our artifacts.

Optional feature. The engine runs fully without it; this is only for mirroring
journals, signals and status files to a repo you control, so there is an audit
trail off the machine.

  export GITHUB_TOKEN=...          # a token with write access
  export GITHUB_REPO=you/your-repo # YOUR repo

WHY THERE IS NO DEFAULT
-----------------------
Upstream, this slug was a string literal in 18 modules — everything that pushes
an artifact, and everything that reads one back. Two things follow from that,
and the second is why this file exists rather than a find-and-replace.

First, a value repeated in 18 places is a value that cannot be changed safely.
Renaming the account meant editing all 18 correctly, in one commit, while a live
system read and wrote that path on a schedule.

Second — and this is the part worth carrying into your own fork — GitHub frees a
username the moment it is renamed, and anyone may claim it. A system holding a
valid token and asking for the old slug would then be pushing its trading history
into a stranger's repository, and reading its instructions back out of one. The
upstream system feeds a file from this repo to its trading model as a directive,
so that is an injection path, not merely a data leak.

Shipping a default here would recreate exactly that: an unconfigured clone
quietly aimed at somebody else's repo. So there is no default. Publishing stays
off until you name your own repo, and `configured()` says so plainly.
"""
import os


def name() -> str:
    """`owner/repo` from GITHUB_REPO, or "" when unset."""
    return (os.getenv("GITHUB_REPO", "") or "").strip()


def configured() -> bool:
    """True when BOTH a token and a repo are set — publishing needs both, and a
    half-configured setup should fail at the check, not inside `get_repo("")`."""
    return bool(name() and os.getenv("GITHUB_TOKEN", "").strip())


def require() -> str:
    """The slug, or a clear error naming what to set. For call sites that have
    already decided they are publishing."""
    slug = name()
    if not slug:
        raise RuntimeError(
            "GITHUB_REPO is not set. Artifact publishing needs YOUR repo — e.g. "
            "GITHUB_REPO=you/your-trading-agent. There is deliberately no default: "
            "a built-in slug would aim an unconfigured clone at somebody else's repo."
        )
    return slug
