"""
io_utils.py — crash-safe file writes.

The agent's signal/state files are read by the very next job in the pipeline
(research.py → decide.py). A plain `open(path, "w"); json.dump(...)` is NOT
atomic: if the writer is killed mid-write (the scheduler's 600s timeout, an OOM,
or a redeploy), the file is left truncated and the reader crashes on invalid
JSON. These helpers write to a sibling temp file, fsync it, then os.replace()
onto the target — an atomic rename on POSIX, so a reader ever only sees the old
complete file or the new complete file, never a half-written one.

  from io_utils import write_json_atomic, write_text_atomic
  write_json_atomic("signals/2026-06-17.json", payload)
"""
import json
import os
import sys
import tempfile


def write_text_atomic(path: str, text: str) -> None:
    """Atomically write `text` to `path` (temp file + fsync + os.replace)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Temp file MUST be on the same filesystem as the target for os.replace to be
    # atomic — so create it in the target's own directory.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".swp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path: str, obj, indent: int = 2) -> None:
    """Atomically write `obj` as JSON to `path`."""
    write_text_atomic(path, json.dumps(obj, indent=indent, default=str))


def load_valid_signals(signals_path: str, today: str):
    """
    Load today's signals only if the file is present, parseable, dated TODAY, and
    carries the required blocks. Returns the dict, or None if anything is off
    (missing / truncated write / wrong date / malformed) so the caller can
    regenerate or fail safe rather than trade on bad data.

    Lives here (stdlib-only) rather than in decide.py so it stays importable in CI
    without the heavy trade-path deps (anthropic/alpaca).
    """
    if not os.path.exists(signals_path):
        return None
    try:
        with open(signals_path) as f:
            data = json.load(f)
    except (ValueError, OSError) as e:
        print(f"  signals file unreadable/corrupt: {e}", file=sys.stderr)
        return None
    if data.get("date") != today:
        print(f"  signals file date {data.get('date')!r} != today {today!r} — rejecting.", file=sys.stderr)
        return None
    if not isinstance(data.get("account"), dict) or "signals" not in data:
        print("  signals file missing 'account'/'signals' block — rejecting.", file=sys.stderr)
        return None
    return data
