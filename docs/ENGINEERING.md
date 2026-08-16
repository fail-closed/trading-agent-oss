# Engineering notes

Practices extracted from ~14 months of running this system unattended. Each one
exists because something went wrong first.

## 1. One rule, one owner, one tripwire

Any rule with more than one consumer WILL drift. The −8% stop was defined in
three files; a change to one silently disagreed with the others. The fix is
always the same shape: extract a shared helper, make every consumer import it,
and add a test that fails if a second definition appears.

`tests/test_stop_rule.py` and `tests/test_allocation_rule.py` are the templates.

**Audit consumers, not presence.** Verifying a config value is *set* proves
nothing about whether every reader applies it. Run the sweep — `grep -rln <KNOB>
*.py` — and check each hit.

## 2. You may decide anything; you may not fail to decide

Two CI tripwires enforce this:

- `test_coverage_floor.py` — every module has a test, or an entry in `NO_TESTS`
  with a specific reason.
- `test_state_registry.py` — every state file is in `state_backup.STATE_FILES`
  or in `EPHEMERAL_STATE` with a reason it is safe to lose.

Both edits are cheap. What is no longer possible is not noticing.

The registry one had a blind spot worth knowing about: its detector originally
matched only `NAME = "file.json"`, so any constant built by `os.path.join(...)`
or an f-string was invisible — six files, including one tracking spend against a
paid-API budget. Widening it surfaced four genuinely unclassified files.

## 3. Verify the verifier

A passing check and an absent check look identical from outside. Both report
success.

**A new tripwire is not done until you have broken it and watched it fail.**
Delete the entry, stub the passthrough, remove the file from the copy list —
then restore. Note the probe in the docstring.

Extend this to tooling you inherited. A deployment-verification command in the
upstream system had been trusted for months and was checking the wrong machine
the entire time; nobody had asked it to prove it could see what it claimed to.

When a check surprises you by passing, suspect the check before the code.

## 4. Measure against a benchmark, not against zero

Scoring a trade as "up after five days" counts market drift as skill. In a week
when the index rose 4%, most randomly chosen stocks are up. Every outcome in
`memory_v2.py` carries the benchmark's return over the *same* holding period,
and the headline figure is excess.

This one reclassification turned a "100% win rate" into "0% beat-market."

## 5. Fail open, fail loud, fail safe — pick per path

- **Fail open** where a third-party outage must not block a decision that already
  passed every gate (`debate.py`, `prediction_markets.py`).
- **Fail loud** on anything that reports success: check HTTP status codes, and
  never let bookkeeping sit above a critical write.
- **Fail safe** on data: `decide.py` refuses to trade on missing, corrupt or
  stale signals rather than proceeding on a partial file.

Truth about execution comes from the broker's response, never from having
intended to trade. Inferring "executed" from the presence of an order request
records rejections as fills, and those then poison the outcome scorecard.

## 6. Treat your freshest code as the least trustworthy

The natural instinct is the opposite — you just wrote it, you just checked it.
Every incident worth logging came from recently-changed code that its author had
"just verified," usually verifying the half that worked.

## 7. Keep an incident log, and never delete from it

Numbered rows, permanent identifiers, cited from the code that exists because of
them (`stops.py` cites the row explaining why it exists). Struck through when
superseded, never removed. A log that gets tidied becomes a status board, and a
status board teaches nobody anything.
