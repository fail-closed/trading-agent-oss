"""The setup wizard — and above all, that its live path cannot be walked through.

The paper path exists to remove friction. The live path exists to add it, so the
tests that matter most here are the negative ones: every way a user might arrive
at `LIVE_TRADING=true` without having actually decided to.

`setup_wizard.py` is stdlib-only by design (it runs before `pip install`), so
these tests need no network and no broker.
"""
import json

import pytest

import setup_wizard as w


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point the wizard at a scratch directory — never the real .env."""
    monkeypatch.setattr(w, "ROOT", tmp_path)
    monkeypatch.setattr(w, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(w, "VENV", tmp_path / ".venv")
    for d in ("signals", "decisions", "journal"):
        (tmp_path / d).mkdir()
    return tmp_path


def _answers(monkeypatch, *values):
    """Feed scripted input to ask()/confirm(). Exhausting it raises, so a test
    can never silently fall through into a prompt it did not anticipate."""
    it = iter(values)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"wizard asked an unexpected question: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)


def _paper_ready(tmp_path, sessions=0):
    w.ENV_PATH.write_text("ALPACA_API_KEY=PKTEST\nALPACA_SECRET_KEY=s\n")
    for i in range(sessions):
        (tmp_path / "decisions" / f"2026-01-{i % 28 + 1:02d}_1000.json").write_text("{}")


# ── .env handling ────────────────────────────────────────────────────────────

def test_env_roundtrip_ignores_comments_and_blanks():
    w.ENV_PATH.write_text("# a comment\n\nA=1\nB=two  # trailing\n")
    assert w.read_env() == {"A": "1", "B": "two"}


def test_written_env_is_not_world_readable():
    """It holds live broker keys once live is configured."""
    w.write_env({"ALPACA_API_KEY": "PK", "ALPACA_SECRET_KEY": "s"})
    assert oct(w.ENV_PATH.stat().st_mode)[-3:] == "600"


def test_rewrite_preserves_unknown_keys():
    """A hand-added variable must survive a wizard re-run."""
    w.write_env({"ALPACA_API_KEY": "PK", "MY_OWN_FLAG": "yes"})
    assert w.read_env()["MY_OWN_FLAG"] == "yes"


def test_live_gates_are_written_closed_by_default():
    """Absent explicit live setup, the four gates must serialise as empty — not
    missing, so a reader can see they exist and are shut."""
    w.write_env({"ALPACA_API_KEY": "PK"})
    env = w.read_env()
    for k in ("I_UNDERSTAND_THIS_TRADES_REAL_MONEY", "LIVE_TRADING", "LIVE_ACCOUNTS"):
        assert env.get(k) == "", f"{k} should be present and empty"


def test_secrets_are_masked_not_echoed():
    assert w.mask("PKABCDEFGHIJKLMN") == "PKAB…MN"
    assert "CDEFGHIJKL" not in w.mask("PKABCDEFGHIJKLMN")
    assert w.mask("") == ""


# ── the live path: every refusal ─────────────────────────────────────────────

def test_live_refuses_without_a_paper_setup(monkeypatch, capsys):
    _answers(monkeypatch)                       # must not ask anything
    with pytest.raises(SystemExit):
        w.live_setup()
    assert "run `python3 setup_wizard.py` first" in capsys.readouterr().out


def test_live_refuses_with_too_few_paper_sessions(_isolate, monkeypatch, capsys):
    _paper_ready(_isolate, sessions=5)
    _answers(monkeypatch)                       # must refuse before prompting
    assert w.live_setup() == 1
    out = capsys.readouterr().out
    assert "only 5 paper sessions" in out
    assert "Nothing was changed" in out
    assert w.read_env().get("LIVE_TRADING") in (None, "")


def test_live_refuses_if_the_risk_files_were_not_read(_isolate, monkeypatch, capsys):
    _paper_ready(_isolate, sessions=25)
    _answers(monkeypatch, "n")                  # "have you read all three?" -> no
    assert w.live_setup() == 1
    assert "Read them first" in capsys.readouterr().out
    assert w.read_env().get("LIVE_TRADING") in (None, "")


def test_live_refuses_a_non_numeric_amount(_isolate, monkeypatch):
    _paper_ready(_isolate, sessions=25)
    _answers(monkeypatch, "y", "lots")
    with pytest.raises(SystemExit):
        w.live_setup()
    assert w.read_env().get("LIVE_TRADING") in (None, "")


def test_live_refuses_when_the_phrase_is_wrong(_isolate, monkeypatch, capsys):
    """The last gate. A near-miss must not open it — no fuzzy matching."""
    _paper_ready(_isolate, sessions=25)
    monkeypatch.setattr(w, "alpaca_account",
                        lambda k, s, base: (True, {"equity": "400"}))
    _answers(monkeypatch, "y", "500", "AKLIVE", "livesecret",
             "i accept i can lose this money")          # wrong case
    assert w.live_setup() == 1
    assert "did not match" in capsys.readouterr().out
    assert w.read_env().get("LIVE_TRADING") in (None, "")


def test_live_refuses_paper_keys_reused_as_live(_isolate, monkeypatch, capsys):
    _paper_ready(_isolate, sessions=25)
    monkeypatch.setattr(w, "alpaca_account", lambda k, s, base: (True, {"equity": "400"}))
    _answers(monkeypatch, "y", "500",
             "PKTEST", "s",          # same as the paper key -> rejected
             "AKLIVE", "livesecret",
             "I accept I can lose this money")
    assert w.live_setup() == 0
    assert "that is your PAPER key" in capsys.readouterr().out


def test_live_warns_when_the_account_holds_more_than_the_stated_amount(_isolate, monkeypatch, capsys):
    """Sizing comes from account equity, so a $10k account ignores a $500 answer."""
    _paper_ready(_isolate, sessions=25)
    monkeypatch.setattr(w, "alpaca_account", lambda k, s, base: (True, {"equity": "10000"}))
    _answers(monkeypatch, "y", "500", "AKLIVE", "livesecret", "n")   # decline to continue
    assert w.live_setup() == 1
    out = capsys.readouterr().out
    assert "more than the $500" in out
    assert w.read_env().get("LIVE_TRADING") in (None, "")


def test_live_succeeds_only_when_every_gate_is_satisfied(_isolate, monkeypatch):
    _paper_ready(_isolate, sessions=25)
    monkeypatch.setattr(w, "alpaca_account", lambda k, s, base: (True, {"equity": "400"}))
    _answers(monkeypatch, "y", "500", "AKLIVE", "livesecret",
             "I accept I can lose this money")
    assert w.live_setup() == 0
    env = w.read_env()
    assert env["I_UNDERSTAND_THIS_TRADES_REAL_MONEY"] == "yes"
    assert env["LIVE_TRADING"] == "true"
    assert env["LIVE_ACCOUNTS"] == "core"
    assert env["ALPACA_LIVE_API_KEY"] == "AKLIVE"
    # and the paper keys are untouched
    assert env["ALPACA_API_KEY"] == "PKTEST"


def test_live_never_writes_anything_on_any_refusal_path(_isolate, monkeypatch):
    """Belt and braces across all refusals: the file must not gain live values."""
    for answers in ([], ["n"], ["y", "nope"]):
        _paper_ready(_isolate, sessions=25)
        _answers(monkeypatch, *answers)
        try:
            w.live_setup()
        except (SystemExit, AssertionError):
            # AssertionError = the wizard asked a question this path did not
            # script, i.e. it stopped early. Either way it must not have written.
            pass
        assert w.read_env().get("LIVE_TRADING") in (None, ""), f"answers={answers}"


# ── the paper path ───────────────────────────────────────────────────────────

def test_python_version_gate(monkeypatch):
    """3.10 is the floor because the engine uses match/zoneinfo idioms below it."""
    import collections
    V = collections.namedtuple("V", "major minor micro releaselevel serial")
    monkeypatch.setattr(w.sys, "version_info", V(3, 9, 0, "final", 0))
    with pytest.raises(SystemExit):
        w.step_python()
    monkeypatch.setattr(w.sys, "version_info", V(3, 12, 1, "final", 0))
    w.step_python()          # must not raise


def test_watchlist_edit_caps_allocation(_isolate, monkeypatch):
    (_isolate / "watchlist.json").write_text(json.dumps(
        {"stocks": [{"symbol": "SPY", "name": "x", "sector": "ETF", "max_allocation": 0.1}],
         "risk": {}}))
    _answers(monkeypatch, "y", "AAPL,MSFT,SPY,QQQ,JPM,XOM,JNJ,PG,CAT,NVDA")
    w.step_watchlist({})
    wl = json.loads((_isolate / "watchlist.json").read_text())
    assert len(wl["stocks"]) == 10
    caps = {s["max_allocation"] for s in wl["stocks"]}
    assert max(caps) <= 0.10, "no single name may exceed the 10% cap"


def test_watchlist_rejects_a_dangerously_short_list(_isolate, monkeypatch, capsys):
    original = {"stocks": [{"symbol": "SPY", "name": "x", "sector": "ETF",
                            "max_allocation": 0.1}], "risk": {}}
    (_isolate / "watchlist.json").write_text(json.dumps(original))
    _answers(monkeypatch, "y", "TSLA")
    w.step_watchlist({})
    assert "very few names" in capsys.readouterr().out
    assert json.loads((_isolate / "watchlist.json").read_text()) == original


def test_alpaca_account_reports_a_readable_reason_on_401(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 401, "no", {}, None)

    monkeypatch.setattr(w.urllib.request, "urlopen", boom)
    good, msg = w.alpaca_account("k", "s", w.PAPER_BASE)
    assert not good and "key or secret is wrong" in msg


def test_alpaca_account_distinguishes_a_network_failure(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(w.urllib.request, "urlopen", boom)
    good, msg = w.alpaca_account("k", "s", w.PAPER_BASE)
    assert not good and "internet connection" in msg
