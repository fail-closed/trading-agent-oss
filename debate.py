"""
debate.py — adversarial bull/bear vetting of proposed BUYs (flag: DEBATE_BUYS).

After ask_claude() proposes trades, this runs ONE batched LLM call that argues the
strongest bull case AND bear case for each candidate buy, then judges proceed/skip.
It can only DOWNGRADE a BUY to SKIP — it never invents or upsizes a trade. Fail-open:
if the call errors, it returns no verdicts and the original buys stand (an infra
blip must never block a buy that already passed the score + ML gates).

Idea adapted from TauricResearch/TradingAgents' bull/bear researcher debate, kept
to a single cheap call (Haiku by default) to fit the cost profile.

SHADOW MODE (flag: DEBATE_SHADOW) — Phase 0 of docs/TRADINGAGENTS_PLAN.md. The
gate ran for weeks with no counterfactual: it vetoed 3/3 buys once and was
switched off, and nothing recorded what those buys would have done. With
DEBATE_SHADOW on and DEBATE_BUYS off, the call still runs and every verdict is
stamped on its decision, but NOTHING is downgraded — so both cohorts execute and
memory_v2 can score `proceed` against `skip` on real fills. That is the only
apples-to-apples version of the experiment; see `basis` in memory_v2.

  import debate
  m = debate.mode(invested_pct)                  # "live" | "shadow" | "off"
  verdicts = debate.vet_buys(buys, context)      # {symbol: {verdict, bull, bear, confidence}}
  debate.stamp_verdicts(result["decisions"], verdicts)          # always — records
  n = debate.apply_to_decisions(result["decisions"], verdicts)  # only when live — mutates
"""
import json
import os
import sys

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "bull": {"type": "string", "description": "strongest case FOR the buy"},
                    "bear": {"type": "string", "description": "strongest case AGAINST"},
                    "verdict": {"type": "string", "enum": ["proceed", "skip"]},
                    "confidence": {"type": "number", "description": "0-1 confidence in the verdict"},
                },
                "required": ["symbol", "bull", "bear", "verdict", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def enabled(invested_pct: float = None) -> bool:
    """Is the bull/bear gate active this session?

    DEBATE_BUYS accepts three forms:
      true / 1 / yes / on   always on
      false / unset         always off
      at:<fraction>         off until the book reaches that invested level,
                            then on — e.g. `at:0.80`

    The third exists because the gate was switched off to clear a deployment
    push (it had vetoed 3 of 3 buys while the account sat at 99.8% cash), and
    "turn it back on once we hit 80%" is exactly the kind of follow-up that
    depends on a human remembering. Expressing the condition here makes it a
    property of the system instead — the gate restores itself the first session
    the book is deployed, whether or not anyone is watching.

    `invested_pct` is a fraction (0.80 = 80% invested). When the threshold form
    is configured but no value is supplied, the gate stays OFF: the caller could
    not establish the condition, and silently re-enabling a filter that blocks
    buys is the more surprising failure."""
    raw = os.getenv("DEBATE_BUYS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw.startswith("at:"):
        try:
            threshold = float(raw.split(":", 1)[1])
        except ValueError:
            return False
        return invested_pct is not None and invested_pct >= threshold
    return False


def shadow_enabled() -> bool:
    """DEBATE_SHADOW: run the debate and record verdicts, but never act on them."""
    return os.getenv("DEBATE_SHADOW", "").strip().lower() in ("1", "true", "yes", "on")


def mode(invested_pct: float = None) -> str:
    """The single owner of "what is the debate gate doing this session".

    Returns "live" (vet and downgrade), "shadow" (vet and record only), or "off".
    A live gate wins over shadow — there is nothing to shadow-measure once the
    gate is already acting, and running the call twice would just double cost.

    This exists as one function because the same question was previously answered
    by `enabled()` in debate.py AND a second `os.getenv("DEBATE_BUYS")` in
    decide.py for its log line — a knob read in two files is a rule computed in
    two places (OPERATIONS §8 habit 6)."""
    if enabled(invested_pct):
        return "live"
    return "shadow" if shadow_enabled() else "off"


def mode_note(invested_pct: float = None) -> str:
    """Human-readable one-liner for the session log, or "" when there's nothing
    surprising to say. Only the `at:` form is worth narrating — it is the form
    whose state depends on the book, so "OFF" alone would look like a config bug."""
    raw = os.getenv("DEBATE_BUYS", "").strip().lower()
    if not raw.startswith("at:"):
        return ""
    pct = f"{invested_pct * 100:.1f}%" if invested_pct is not None else "unknown"
    return (f"Debate gate: {mode(invested_pct).upper()} "
            f"(threshold form {raw}; invested {pct})")


def vet_buys(buys: list, context: str) -> dict:
    """
    buys: [{"symbol","reason","signal"}]. Returns {symbol: verdict_dict}.
    Empty dict on no buys or any error (fail-open).
    """
    if not buys:
        return {}
    try:
        import anthropic
    except Exception as e:
        print(f"  [debate] anthropic unavailable ({e}) — fail-open", file=sys.stderr)
        return {}
    model = os.getenv("DEBATE_MODEL", "claude-haiku-4-5-20251001")
    payload = [{"symbol": b["symbol"], "buy_reason": b.get("reason", ""),
                "signal": b.get("signal", {})} for b in buys]
    prompt = (
        "You are a risk-focused devil's advocate on a LONG-ONLY technical trading desk. "
        "For EACH proposed buy, argue the strongest bull case and the strongest bear case "
        "from the signal data and session context, then judge: return 'proceed' only if the "
        "bull case clearly outweighs the bear after accounting for macro/regime risk; "
        "otherwise 'skip'. Bias toward 'skip' on a coin-flip — a missed trade beats a bad one. "
        "Do not consider shorting; the only options are proceed (buy) or skip.\n\n"
        f"## Session context\n{context}\n\n## Proposed buys\n{json.dumps(payload, indent=2, default=str)}"
    )
    try:
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=model, max_tokens=4000, thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in r.content if b.type == "text"), None)
        if not text:
            return {}
        out = json.loads(text)
        print(f"  [debate] vetted {len(out['verdicts'])} buy(s) via {model}", file=sys.stderr)
        return {v["symbol"]: v for v in out["verdicts"]}
    except Exception as e:
        print(f"  [debate] vetting failed ({e}) — fail-open, keeping original buys", file=sys.stderr)
        return {}


def stamp_verdicts(decisions: list, verdicts: dict, applied: bool) -> int:
    """Record each verdict on its decision WITHOUT changing any action.

    Split from apply_to_decisions so the recording path and the mutating path are
    different functions: shadow mode calls only this one, and no future edit can
    accidentally give shadow mode teeth. `applied` says whether the gate was live
    this session — memory_v2 needs it to tell a real fill from a hypothetical.

    Returns how many decisions were stamped."""
    n = 0
    for d in decisions:
        v = verdicts.get(d.get("symbol"))
        if not v:
            continue
        d["debate_verdict"] = v.get("verdict")
        d["debate_confidence"] = v.get("confidence")
        d["debate_applied"] = applied
        n += 1
    return n


def apply_to_decisions(decisions: list, verdicts: dict) -> int:
    """
    Downgrade any BUY whose verdict is 'skip' to SKIP (no trade). Pure function —
    mutates `decisions` in place. Returns how many were downgraded. A symbol with
    no verdict (e.g. call failed) is left untouched (fail-open).
    """
    n = 0
    for d in decisions:
        if d.get("action") != "BUY":
            continue
        v = verdicts.get(d.get("symbol"))
        if v and v.get("verdict") == "skip":
            d["action"] = "SKIP"
            d["trade_args"] = None
            conf = v.get("confidence", 0)
            d["reason"] = f"{d.get('reason','')} | DEBATE-SKIP: {v.get('bear','')} (conf {conf:.2f})"
            # Mark the cause so the session's decision→fill reconciliation can
            # attribute a missing trade to this gate rather than to a failure.
            # Without it, "model proposed 3 buys, 0 placed" is indistinguishable
            # from a broken order path.
            d["debate_downgraded"] = True
            n += 1
    return n
