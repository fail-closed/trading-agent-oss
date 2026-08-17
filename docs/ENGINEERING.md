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

Both had blind spots worth knowing about, and both were found by accident
rather than by CI:

- The registry's detector originally matched only `NAME = "file.json"`, so any
  constant built by `os.path.join(...)` or an f-string was invisible — six
  files, including one tracking spend against a paid-API budget.
- Both tripwires scanned **top-level `*.py` only**, so an entire package could
  exist without ever being checked, and moving any module into a package would
  silently delete its coverage with no red build. Probed before the fix: a
  packaged module with untested logic *and* an unregistered state file passed
  clean.

The lesson is not "widen your regexes." It is that a tripwire's *scope* is a
silent assumption — here, a repository layout nobody had promised to keep.

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

## 8. Friction belongs where the consequences are

`setup_wizard.py` removes every obstacle it can from paper setup — it installs
dependencies, verifies broker keys against the live API before writing anything,
and runs the engine once so the user sees real output rather than a promise.

Its `--live` path does the opposite, on purpose. It refuses without recorded
paper sessions, asks whether the three files that gate a bad order have actually
been read, requires an amount the user would be untroubled to lose, warns if the
funded account exceeds that amount (position sizing reads account equity, so a
larger balance quietly ignores the stated limit), and requires a typed sentence
rather than a keypress.

None of that is enforceable — a user can lie to a wizard. It is there so the
decision is made deliberately instead of by pressing Enter four times, and every
refusal path is covered by a test asserting nothing was written.

The general rule: **optimise for the fewest steps where the worst outcome is a
wasted afternoon, and for deliberateness where the worst outcome is someone's
money.** A single "advanced setup" flow that treats both the same is the
mistake — it either patronises the paper user or waves the live one through.
