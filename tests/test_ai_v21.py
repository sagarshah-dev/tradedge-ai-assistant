# -*- coding: utf-8 -*-
"""v21 (11-11-2026): the measurement layer. evals/scorecard.py turns real-model eval runs into numbers,
evals/run_ai_evals.py --writer scores the WHY agent's AI sentence against the verifier, and /stats
(ai_stats.py) reads the live logs. Everything here runs with a FAKE model: the tests prove the counting
is right, the real numbers come from running the evals on the laptop."""
import argparse
import json
import os
import sys
from datetime import datetime

import pytest

import ai_chat
import ai_llm
import ai_stats as S
import ai_tools as T

EVALS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals")
sys.path.insert(0, EVALS)
import run_ai_evals as R      # noqa: E402
import scorecard as C         # noqa: E402


# ── quantiles ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("xs,q,want", [
    ([], 0.5, None), ([None], 0.95, None), ([7], 0.5, 7), ([7], 0.95, 7),
    ([1, 2, 3, 4], 0.5, 2), ([4, 1, 3, 2, None], 0.5, 2), (list(range(1, 21)), 0.95, 19),
    (list(range(1, 21)), 0.5, 10), (list(range(1, 101)), 0.95, 95), ([5, 5, 900], 0.95, 900)])
def test_quantile_nearest_rank(xs, q, want):
    assert C.quantile(xs, q) == want and S.quantile(xs, q) == want


# ── router scorecard ─────────────────────────────────────────────────────────
ROUTER_RECS = [
    {"q": "a", "want_tool": "live_top", "tool": "live_top", "pass": True, "pass_raw": True, "ms": 400, "tokens": 900},
    {"q": "b", "want_tool": "live_top", "tool": "live_top", "pass": True, "pass_raw": False, "ms": 600, "tokens": 1100,
     "fixes": ["x"]},                                                      # the model was wrong, code repaired it
    {"q": "c", "want_tool": "live_value", "tool": "live_top", "pass": False, "pass_raw": True, "ms": 500,
     "tokens": 1000},                                                      # the model was right, code broke it
    {"q": "d", "want_tool": "live_value", "tool": "screen", "pass": False, "pass_raw": False, "ms": 2500, "tokens": 1000},
    {"q": "e", "want_tool": "explain_market", "pass": False, "error": "quota: 429 too many"},
]


def test_router_summary_counts():
    c = C.summarize_router(ROUTER_RECS, "gemini-x", "11-11-2026 16:00")
    assert (c["cases"], c["passed"], c["accuracy"]) == (5, 2, 40.0)
    assert (c["raw_passed"], c["raw_accuracy"], c["repaired"], c["broken"]) == (2, 40.0, 1, 1)
    assert c["latency_ms"] == {"p50": 500, "p95": 2500} and c["avg_tokens"] == 1000
    assert c["model_errors"] == {"quota": 1}
    assert c["by_tool"] == {"explain_market": {"passed": 0, "cases": 1}, "live_top": {"passed": 2, "cases": 2},
                            "live_value": {"passed": 0, "cases": 2}}
    assert [f["q"] for f in c["failed"]] == ["c", "d", "e"] and c["failed"][2]["got"] == "quota: 429 too many"


def test_empty_run_does_not_divide_by_zero():
    c = C.summarize_router([], "m", "t")
    assert c["accuracy"] is None and c["latency_ms"] == {"p50": None, "p95": None} and c["avg_tokens"] is None
    w = C.summarize_writer([], "m", "t")
    assert w["first_try_pct"] is None and w["avg_attempts"] is None
    assert "—" in C.markdown({"router": c, "writer": w})


def test_writer_summary_counts():
    recs = [{"wrote": True, "first_ok": True, "attempts": 1, "dropped": 0, "ms": 900},
            {"wrote": True, "first_ok": False, "attempts": 2, "dropped": 0, "ms": 2000,
             "first_errors": ["insight uses numbers not in EVIDENCE: 99.9"]},
            {"wrote": True, "first_ok": False, "attempts": 3, "dropped": 1, "ms": 3000,
             "first_errors": ["insight has advice words ('buy')", "insight is 30 words; max 20"]},
            {"wrote": False, "error": "quota_day", "ms": 100},
            {"wrote": True, "first_ok": True, "attempts": 1, "dropped": 0, "shipped_bad": True, "ms": 800}]
    w = C.summarize_writer(recs, "gemini-x", "t")
    assert (w["scenarios"], w["drafts"], w["first_try"], w["first_try_pct"]) == (5, 4, 2, 50.0)
    assert (w["after_retry"], w["dropped"], w["clean_pct"], w["no_draft"], w["shipped_bad"]) == (1, 1, 75.0, 1, 1)
    assert w["avg_attempts"] == 1.75 and w["latency_ms"] == {"p50": 900, "p95": 3000}
    assert w["rejected_for"] == {"number not in evidence": 1, "advice word": 1, "too long": 1}


def test_scorecard_file_keeps_the_other_sections(tmp_path):
    d = str(tmp_path)
    assert C.load(d) == {}
    md = open(os.path.join(d, "SCORECARD.md"), "w")
    md.close()
    C.update(d, "router", C.summarize_router(ROUTER_RECS, "gemini-x", "11-11-2026 16:00"))
    card = C.update(d, "writer", C.summarize_writer([{"wrote": True, "first_ok": True, "attempts": 1}], "w", "t"))
    assert set(card) == {"router", "writer"} and C.load(d)["router"]["accuracy"] == 40.0
    text = open(os.path.join(d, "SCORECARD.md"), encoding="utf-8").read()
    assert "| **Accuracy (what the user gets)** | **40.0%** (2 / 5) |" in text
    assert "| Model alone, before code repair | 40.0% (2 / 5) |" in text
    assert "| Fixed by the repair layer | 1 |" in text and "| Broken by the repair layer | 1 |" in text
    assert "- `c` → wanted `live_value`, got `live_top`" in text
    assert "_Not run yet: `python evals/run_ai_evals.py --alerts`_" in text       # a part that never ran says so
    assert "| **Answers sent with an unchecked number** | **0** |" in text
    assert not os.path.exists(os.path.join(d, "scorecard.json.tmp"))
    with pytest.raises(AssertionError):
        C.update(d, "bogus", {})
    open(os.path.join(d, "scorecard.json"), "w").write("{broken")
    assert C.load(d) == {}                                                       # a corrupt file is not fatal


# ── the eval runner with a fake model ────────────────────────────────────────
CASES = [
    {"q": "top gainers now", "tool": "live_top", "args": {"metric": "pct"}},
    {"q": "fmcg sector", "tool": "explain_market", "args": {"kind": "sector_move", "target": "NIFTY FMCG"}},
    {"q": "vix now", "tool": "live_value", "args": {"symbols": ["IDX:INDIA VIX"]}},
]


class FakeRouter:
    """'top gainers' right; 'fmcg sector' wrong tool that fix_route repairs; 'vix now' wrong and stays wrong."""
    model = "fake-model"

    def __init__(self, *a, **k):
        pass

    def route(self, prompt):
        if "top gainers" in prompt:
            return ai_llm.Route("live_top", {"metric": "pct"}, "fake-model", 1000, 300)
        if "fmcg sector" in prompt:
            return ai_llm.Route("live_sector_rank", {}, "fake-model", 1200, 500)
        return ai_llm.Route("market_mood", {}, "fake-model", 800, 700)


@pytest.fixture
def runner(tmp_path, monkeypatch, mkt):
    d = tmp_path / "evals"
    d.mkdir()
    (d / "ai_router_cases.jsonl").write_text("\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8")
    monkeypatch.setattr(R, "HERE", str(d))
    monkeypatch.setattr(ai_llm, "GeminiRouter", FakeRouter)
    monkeypatch.setattr(ai_llm, "_cfg", lambda name, default="": "k" if name == "GEMINI_EVAL_API_KEY" else default)
    monkeypatch.setitem(sys.modules, "state", mkt.state)                 # the runner imports the app's state.py
    return d


def test_router_run_writes_the_scorecard(runner, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_ai_evals.py", "--sleep", "0"])
    assert R.main() == 1                                                 # one case failed -> non-zero for CI
    card = json.loads((runner / "scorecard.json").read_text(encoding="utf-8"))["router"]
    assert (card["cases"], card["passed"], card["raw_passed"], card["repaired"], card["broken"]) == (3, 2, 1, 1, 0)
    assert card["model"] == "fake-model" and card["latency_ms"] == {"p50": 500, "p95": 700}
    assert [f["q"] for f in card["failed"]] == ["vix now"]
    out = capsys.readouterr().out
    assert "model alone 33.3% -> with code repair 66.7% (repaired 1, broken 0)" in out
    assert "**66.7%** (2 / 3)" in (runner / "SCORECARD.md").read_text(encoding="utf-8")


def test_failed_only_rerun_keeps_the_full_run_numbers(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_ai_evals.py", "--sleep", "0"])
    R.main()
    before = (runner / "scorecard.json").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_ai_evals.py", "--sleep", "0", "--failed-only"])
    R.main()
    assert (runner / "scorecard.json").read_text(encoding="utf-8") == before      # 1 case is not "the accuracy"


def test_no_key_refuses_to_spend_the_bot_quota(runner, monkeypatch, capsys):
    monkeypatch.setattr(ai_llm, "_cfg", lambda name, default="": default)
    monkeypatch.setattr(sys, "argv", ["run_ai_evals.py", "--writer"])
    assert R.main() == 2 and "--use-bot-key" in capsys.readouterr().out
    assert not (runner / "scorecard.json").exists()


class Writer:
    """Scripted drafts per question: clean / wrong number then clean / always wrong / model error."""
    model = "fake-writer"

    def __init__(self):
        self.calls = {}

    def route(self, prompt):
        q = prompt.split("\n", 1)[0]
        n = self.calls[q] = self.calls.get(q, 0) + 1
        if "why is nifty down" in q:
            return ai_llm.Route("write_view", {"insight": "Large banks are pulling while smaller sectors hold."}, self.model, 500, 100)
        if "why is nifty up" in q:
            txt = "The index jumped 99.9% in one candle." if n == 1 else "Buyers are spread across most sectors."
            return ai_llm.Route("write_view", {"insight": txt}, self.model, 500, 100)
        if "why banknifty" in q:
            return ai_llm.Route("write_view", {"insight": "Banks fell 77.7% on heavy selling."}, self.model, 500, 100)
        if "strongest now" in q:
            raise ai_llm.RouterError("quota_day", "daily limit")
        return ai_llm.Route("write_view", {"insight": "Breadth and volume point the same way."}, self.model, 500, 100)


def test_writer_eval_reads_the_real_graph(mkt, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "HERE", str(tmp_path))
    w = Writer()
    code = R.run_writer_evals("k", argparse.Namespace(model=None, sleep=0), writer=w, market=mkt)
    card = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))["writer"]
    n = len(R.writer_scenarios(mkt.ctx))
    assert card["scenarios"] == n >= 20 and card["model"] == "fake-writer"
    assert card["no_draft"] == 1 and card["drafts"] == n - 1
    assert card["first_try"] == n - 3 and card["after_retry"] == 1 and card["dropped"] == 1
    assert card["shipped_bad"] == 0 and code == 0                 # the 77.7% line was removed, never sent
    assert card["rejected_for"] == {"number not in evidence": 2}
    assert w.calls["QUESTION: why is nifty up today"] == 2 and w.calls["QUESTION: why banknifty falling"] == 3
    recs = [json.loads(l) for l in open([str(p) for p in tmp_path.glob("writer_*.jsonl")][0], encoding="utf-8")]
    bank = [r for r in recs if r["q"] == "why banknifty falling"][0]
    assert bank["dropped"] == 1 and bank["insight"] is None and bank["attempts"] == 3
    out = capsys.readouterr().out
    assert "DROPPED" in out and "AFTER RETRY" in out and "NO DRAFT (quota_day)" in out and "FIRST TRY" in out


def test_writer_eval_flags_an_unchecked_number_that_got_out(mkt, tmp_path, monkeypatch):
    """If the last-resort cleaner ever stops working, the eval must say so (exit 1, shipped_bad > 0)."""
    import ai_agent as AG
    monkeypatch.setattr(R, "HERE", str(tmp_path))
    monkeypatch.setattr(AG, "clean_view", lambda view, result, question="": (view, []))
    code = R.run_writer_evals("k", argparse.Namespace(model=None, sleep=0), writer=Writer(), market=mkt)
    card = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))["writer"]
    assert card["shipped_bad"] == 1 and code == 1


def test_writer_scenarios_cover_every_kind_and_validate(mkt):
    sc = R.writer_scenarios(mkt.ctx)
    assert {a["kind"] for _q, a in sc} == {"index_move", "sector_move", "strongest_sectors", "rotation", "stock_move",
                                           "vix", "crude"}
    for _q, a in sc:
        T.validate(mkt.ctx, "explain_market", a)                 # every scenario is a legal tool call


def test_new_eval_cases_are_valid_tool_calls(mkt):
    """The 11-11 live failures are in the eval set, and each expected answer is a call the bot accepts."""
    cases = [json.loads(l) for l in open(os.path.join(EVALS, "ai_router_cases.jsonl"), encoding="utf-8") if l.strip()]
    new = [c for c in cases if c.get("v") == 20]
    assert {c["q"] for c in new} >= {"nifty oi data", "nifty future oi", "fmcg sector", "why nifty down"}
    keys = [(c["q"], json.dumps(c.get("prev"))) for c in cases]
    assert len(set(keys)) == len(keys)                           # no duplicate (question, context)
    for c in new:
        args = dict(c["args"])
        if args.get("target") == "NIFTY PSU BK":
            args["target"] = "NIFTY PSU BANK"                    # the fake market's name for the same sector
        T.validate(mkt.ctx, c["tool"], args)


# ── /stats ───────────────────────────────────────────────────────────────────
NOW = datetime(2026, 11, 11, 15, 0)


def _write(folder, prefix, day, recs, raw=()):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "%s_%s.jsonl" % (prefix, day)), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        for line in raw:
            f.write(line + "\n")


def _tr(first_errors=None, attempts=1, dropped=0, wrote=True):
    t = [{"node": "plan"}]
    for i in range(1, attempts + 1):
        t.append({"node": "write", "attempt": i, "ms": 500} if wrote else {"node": "write", "error": "quota_day"})
        if wrote:
            t.append({"node": "verify", "attempt": i, "errors": (first_errors or []) if i == 1 else
                      (["x"] if dropped else [])})
    t.append({"node": "finalize", "dropped": ["insight"] * dropped, "view": wrote and not dropped})
    return t


@pytest.fixture
def logs(tmp_path):
    chat, agent = str(tmp_path / "logs" / "ai_chat"), str(tmp_path / "logs" / "ai_agent")
    _write(chat, "ai_chat", "2026-11-11", [
        {"ts": "10:00:00", "q": "top gainers", "tool": "live_top", "model": "g", "tokens": 1000, "llm_ms": 400, "ms": 450},
        {"ts": "10:01:00", "q": "fmcg sector", "tool": "explain_market", "model": "g", "tokens": 1400, "llm_ms": 600,
         "ms": 3000, "fixes": ["one sector named -> sector card"], "agent": {"attempts": 1}},
        {"ts": "10:02:00", "q": "why nifty down", "tool": "explain_market", "model": "g", "tokens": 1200, "llm_ms": 2000,
         "ms": 2100, "agent_error": "boom"},
        {"ts": "10:03:00", "q": "weather?", "tool": "cannot_answer", "model": "g", "tokens": 800, "llm_ms": 300, "ms": 310,
         "unsupported": True},
        {"ts": "10:04:00", "q": "sbin rvol", "error": "router_quota", "ms": 90},
        {"ts": "10:05:00", "q": "odds XX", "tool": "live_value", "model": "g", "tokens": 600, "llm_ms": 500, "ms": 520,
         "error": "Unknown symbol"},
        {"ts": "10:06:00", "q": "/check", "tool": "/check", "ms": 5},
        {"ts": "10:07:00", "q": "alert me when x", "alert": True},
        {"ts": "10:08:00", "button": "ag:1:details"},
    ], raw=["{not json", "[1, 2]"])
    _write(chat, "ai_chat", "2026-11-09", [
        {"ts": "11:00:00", "q": "old question", "tool": "live_top", "model": "g", "tokens": 5000, "llm_ms": 9000, "ms": 9100}])
    _write(agent, "ai_agent", "2026-11-11", [
        {"q": "fmcg sector", "kind": "sector_move", "ms": 2400, "offer": True, "trace": _tr()},
        {"q": "why x", "kind": "index_move", "ms": 5200, "trace": _tr(["bad number"], attempts=2)},
        {"q": "why y", "kind": "index_move", "ms": 7000, "trace": _tr(["bad"], attempts=3, dropped=1)},
        {"q": "why z", "kind": "vix", "ms": 300, "trace": _tr(wrote=False)},
        {"resume": "details"}])
    return chat, agent, tmp_path


def test_stats_numbers_from_the_logs(logs):
    chat, agent, _base = logs
    c, bad, seen = S.read_days(chat, "ai_chat", 1, NOW)
    a, bad2, _s = S.read_days(agent, "ai_agent", 1, NOW)
    assert bad == 2 and bad2 == 0 and seen == ["2026-11-11"]
    s = S.summarize(c, a)
    assert (s["questions"], s["commands"], s["alert_msgs"], s["button_taps"]) == (6, 1, 1, 1)
    assert s["routed"] == 5 and s["router_errors"] == {"quota": 1}
    assert (s["repaired"], s["cannot"], s["tool_errors"], s["agent_fallbacks"]) == (1, 1, 1, 1)
    assert s["llm_ms"] == {"p50": 500, "p95": 2000} and s["total_ms"] == {"p50": 450, "p95": 3000}
    assert s["tokens"] == {"total": 5000, "avg": 1000}
    assert s["tools"][0] == ("explain_market", 2) and dict(s["tools"])["live_top"] == 1
    assert s["agent"] == {"answers": 4, "first_try": 1, "after_retry": 1, "dropped": 1, "no_draft": 1,
                          "ms": {"p50": 2400, "p95": 7000}, "offers": 1}


def test_stats_window_is_calendar_days(logs):
    chat, _agent, _b = logs
    assert len(S.read_days(chat, "ai_chat", 2, NOW)[0]) == 9                 # 10-11 has no file, 09-11 is outside
    recs, _bad, seen = S.read_days(chat, "ai_chat", 3, NOW)
    assert len(recs) == 10 and seen == ["2026-11-09", "2026-11-11"]


def test_stats_card_text(logs):
    chat, agent, base = logs
    text, btn = S.card(chat, agent, str(base / "evals"), 1, NOW)
    assert isinstance(text, T.Rich) and text.startswith("<b>📈 Assistant stats · today</b>")
    assert "<b>Use</b> 6 questions · 1 commands · 1 alert messages · 1 button taps" in text
    assert "<b>Model</b> 5 routed · 1 failed (quota × 1)" in text
    assert "<b>Speed</b> model p50 500 ms · p95 2.0s · whole answer p50 450 ms · p95 3.0s" in text
    assert "<b>Tokens</b> 5,000 total · 1,000 per question" in text
    assert "repaired the model's call 1 (20%) · said \"can't\" 1 · tool errors 1" in text
    assert "<b>WHY agent</b> 4 answers · AI line ok first try 1 · after retry 1 · removed 1 · no draft 1 · graph failed 1" in text
    assert "<i>2 unreadable log line(s) skipped</i>" in text
    assert text.endswith("Not run yet: python evals/run_ai_evals.py --use-bot-key")
    assert btn == [[("Today", "st:1"), ("7 days", "st:7"), ("30 days", "st:30")]]
    week, _b = S.card(chat, agent, None, 7, NOW)
    assert "last 7 days (2 with activity)" in week and "<b>Use</b> 7 questions" in week


def test_stats_card_shows_the_last_eval_run(logs):
    chat, agent, base = logs
    os.makedirs(str(base / "evals"))
    C.update(str(base / "evals"), "router", C.summarize_router(ROUTER_RECS, "gemini-x", "11-11-2026 16:00"))
    C.update(str(base / "evals"), "writer", C.summarize_writer(
        [{"wrote": True, "first_ok": True, "attempts": 1}], "w", "11-11-2026 16:20"))
    text, _b = S.card(chat, agent, str(base / "evals"), 1, NOW)
    assert "Router 40.0% of 5 cases (model alone 40.0%) · 11-11-2026 16:00" in text
    assert "Writer first try 100.0% · removed 0 · sent unchecked 0 · 11-11-2026 16:20" in text
    assert "Not run yet" not in text


def test_stats_with_no_logs_at_all(tmp_path):
    text, _b = S.card(str(tmp_path / "nope"), None, None, 500, NOW)          # days are capped, folders may not exist
    assert "last 30 days (0 with activity)" in text and "No questions logged in this period." in text
    assert S.summarize([], [])["agent"]["answers"] == 0


class TG:
    def __init__(self):
        self.sent = []

    def send(self, chat, text, buttons=None):
        self.sent.append((text, buttons))
        return True

    def answer_callback(self, *a):
        pass


def test_bot_stats_command_buttons_and_timing(mkt, tmp_path):
    tg = TG()
    log_dir = str(tmp_path / "logs" / "ai_chat")
    bot = ai_chat.Bot(mkt.ctx, None, tg, ["c"], log_dir)
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/check"}})
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/stats"}})
    text, btn = tg.sent[-1]
    assert text.startswith("<b>📈 Assistant stats · today</b>") and "1 commands" in text and btn[0][1] == ("7 days", "st:7")
    day = datetime.now().strftime("%Y-%m-%d")
    recs = [json.loads(l) for l in open(os.path.join(log_dir, "ai_chat_%s.jsonl" % day), encoding="utf-8")]
    assert all(isinstance(r["ms"], int) and "buttons" not in r for r in recs)   # whole-answer time is logged
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/stats 7"}})
    assert "last 7 days (1 with activity)" in tg.sent[-1][0]
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/stats lots"}})
    assert tg.sent[-1][0] == "Use /stats or /stats 7 (number of days, 1 to 30)."
    bot.handle({"callback_query": {"id": "1", "data": "st:30", "message": {"chat": {"id": "c"}}}})
    assert "last 30 days" in tg.sent[-1][0]
    n = len(tg.sent)
    bot.handle({"callback_query": {"id": "2", "data": "st:7", "message": {"chat": {"id": "x"}}}})
    assert len(tg.sent) == n and "/stats" in T.HELP_TEXT                       # a stranger gets nothing
