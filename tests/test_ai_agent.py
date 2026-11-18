# -*- coding: utf-8 -*-
"""The WHY agent (LangGraph). Trajectory tests (did it run the right checks?), the verifier loop,
human-in-the-loop, persistence, failure paths and the bot wiring. Fake writer: no network."""

import pytest

import ai_agent as AG
import ai_alerts as A
import ai_chat
import ai_evidence as E
import ai_tools as T
from ai_llm import Route, RouterError

from test_ai_v19 import PLANS


class Writer:
    """Fake LLM writer: returns the scripted views in order, records every prompt."""
    def __init__(self, *views, error=None):
        self.views, self.prompts, self.error = list(views), [], error

    def route(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return Route("write_view", self.views.pop(0) if self.views else {"summary": "Evidence above."}, "fake", 100)


def _agent(mkt, writer, tmp_path, store=True):
    st = A.AlertStore(str(tmp_path / "alerts.json")) if store else None
    return AG.MarketAgent(mkt.ctx, writer, db_dir=str(tmp_path / "db"), alerts=st, log_dir=str(tmp_path / "logs")), st


def _ask(ag, mkt, chat="c1", q="why are psu banks firing", **args):
    args = args or {"kind": "sector_move", "target": "psu bank"}
    return ag.ask(chat, T.validate(mkt.ctx, "explain_market", args), q)


GOOD = {"insight": "PSU BANK and NIFTY 50 move apart: 3 of 3 up with long build-up."}


# ── trajectory: the graph runs exactly the plan, in parallel, then the fixed workflow ──
@pytest.mark.parametrize("key", list(PLANS))
def test_trajectory_every_kind(mkt, tmp_path, key):
    kind, target, compare = key
    ag, _ = _agent(mkt, Writer(), tmp_path)
    args = {"kind": kind}
    if target and kind not in ("vix", "crude"):
        args["target"] = target
    if compare:
        args["compare"] = compare
    text, btn, rec = _ask(ag, mkt, q="why", **args)
    nodes = [t["node"] for t in rec["trace"]]
    checks = sorted(n[6:] for n in nodes if n.startswith("check:"))
    assert checks == sorted(PLANS[key]), (key, checks)
    assert nodes[0] == "plan" and nodes.index("analyze") > max(i for i, n in enumerate(nodes) if n.startswith("check:"))
    assert nodes[-1] == "finalize" and "not advice" in text and isinstance(text, T.Rich)
    assert len(text.split("\n")) <= 12 and btn[0][0][0] == "📊 Details"           # one phone screen, details on tap


def test_checks_really_run_in_parallel(mkt, tmp_path, monkeypatch):
    import threading
    import time
    seen, lock = set(), threading.Lock()
    real = E.run_check

    def slow(ctx, name, kw):
        with lock:
            seen.add(threading.get_ident())
        time.sleep(0.2)
        return real(ctx, name, kw)
    monkeypatch.setattr(E, "run_check", slow)
    ag, _ = _agent(mkt, Writer(GOOD), tmp_path)
    t0 = time.time()
    _ask(ag, mkt)
    assert time.time() - t0 < 0.2 * 5 * 0.8 and len(seen) > 1          # 5 checks, not 1 second in a row


# ── the verifier loop ────────────────────────────────────────────────────────
def test_invented_number_goes_back_with_the_reason(mkt, tmp_path):
    w = Writer({"insight": "PSU BANK +1.80% with 47% more buyers."}, GOOD)
    ag, _ = _agent(mkt, w, tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert rec["attempts"] == 2 and rec["errors_left"] == []
    assert "FIX THESE PROBLEMS" in w.prompts[1] and "numbers not in EVIDENCE: 47" in w.prompts[1]
    assert "47%" not in text and "🧠 <i>PSU BANK and NIFTY 50 move apart" in text


def test_advice_is_refused(mkt, tmp_path):
    w = Writer({"insight": "Buy PNB on dips."}, GOOD)
    ag, _ = _agent(mkt, w, tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert "advice words ('Buy')" in w.prompts[1] and "Buy PNB" not in text


def test_cause_words_are_softened_without_a_retry(mkt, tmp_path):
    w = Writer({"insight": "PSU BANK up because NIFTY 50 is flat, due to long build-up."})
    ag, _ = _agent(mkt, w, tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert rec["attempts"] == 1 and "because" not in text and "due to" not in text
    assert "while NIFTY 50 is flat, alongside long build-up" in text


def test_still_wrong_after_retries_drops_the_line(mkt, tmp_path):
    bad = {"insight": "Volume 9.99x normal across PSU BANK."}
    w = Writer(bad, bad, bad)
    ag, _ = _agent(mkt, w, tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert rec["attempts"] == 1 + AG.MAX_RETRIES and len(w.prompts) == 3
    assert "9.99x" not in text and "1 AI line(s) removed" in text and "<b>Stocks</b> PNB" in text


def test_long_insight_is_refused(mkt, tmp_path):
    w = Writer({"insight": " ".join(["word"] * 30)}, GOOD)
    ag, _ = _agent(mkt, w, tmp_path)
    _ask(ag, mkt)
    assert "insight is 30 words; max 20" in w.prompts[1]


def test_why_without_a_code_lean_is_rejected(mkt, tmp_path):
    w = Writer(dict(GOOD, why=["PSU looks best."]), GOOD)
    ag, _ = _agent(mkt, w, tmp_path)
    _ask(ag, mkt)
    assert "only for a LEAN given by code" in w.prompts[1]


def test_lean_and_confidence_come_from_code(mkt, tmp_path):
    w = Writer({"insight": "", "why": ["Breadth and rank held."], "against": ["Day move small."]})
    ag, _ = _agent(mkt, w, tmp_path)
    text, btn, rec = _ask(ag, mkt, q="strongest sector", kind="strongest_sectors")
    r = E.explain(mkt.ctx, "strongest_sectors")
    assert rec["lean"] == r["lean"] and text.startswith("<b>💪 %s strongest · confidence" % r["lean"])
    assert "LEAN (by code): %s" % r["lean"] in w.prompts[0] and "BRIEF (already shown):" in w.prompts[0]
    details = ag.resume("c1", btn[0][0][1].split(":")[1], "details")
    assert "Why: Breadth and rank held." in details and ("Lean (by code): %s" % r["lean"]) in details


# ── no model / model down: still an answer ──────────────────────────────────
def test_quota_out_gives_facts_and_says_so(mkt, tmp_path):
    ag, _ = _agent(mkt, Writer(error=RouterError("quota_day", "PerDay")), tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert "daily Gemini limit" in text and "<b>Stocks</b> PNB" in text and "🧠" not in text


def test_no_writer_at_all(mkt, tmp_path):
    ag, _ = _agent(mkt, None, tmp_path)
    text, _b, rec = _ask(ag, mkt)
    assert text.startswith("<b>📈 PSU BANK up") and rec["attempts"] == 1


def test_no_live_data_is_a_clear_message(mkt, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "run_check", lambda ctx, n, kw: E._missing("x"))
    ag, _ = _agent(mkt, Writer(GOOD), tmp_path)
    text, btn, rec = _ask(ag, mkt)
    assert text.startswith("No live data to explain") and btn is None


# ── human in the loop + persistence ─────────────────────────────────────────
def test_offer_yes_saves_alert_no_saves_nothing(mkt, tmp_path):
    ag, store = _agent(mkt, Writer(GOOD), tmp_path)
    _t, btn, rec = _ask(ag, mkt)
    assert rec["offer"] and [x[0] for x in btn[0]] == ["📊 Details", "🔔 Alert: new day high"]
    thread = btn[0][1][1].split(":")[1]
    details = ag.resume("c1", thread, "details")                       # reading details never uses up the offer
    assert "<b>PSU BANK stocks</b>" in details and "Not checked: news" in details
    reply = ag.resume("c1", thread, "yes")
    assert reply.startswith("✅ Saved alert #1") and "NIFTY PSU BANK makes a new day high" in reply
    assert [a["metric"] for a in store.active("c1")] == ["day_high_up"]
    assert ag.resume("c1", thread, "yes").startswith("That offer has already been answered")
    assert "<b>PSU BANK stocks</b>" in ag.resume("c1", thread, "details")     # still readable after the alert
    _t, btn2, _ = _ask(ag, mkt)
    assert ag.resume("c1", btn2[0][1][1].split(":")[1], "no") == "OK, no alert."
    assert len(store.active("c1")) == 1


def test_offer_belongs_to_its_chat(mkt, tmp_path):
    ag, store = _agent(mkt, Writer(GOOD), tmp_path)
    _t, btn, _ = _ask(ag, mkt, chat="c1")
    assert ag.resume("c2", btn[0][1][1].split(":")[1], "yes") == "That button belongs to another chat."
    assert store.active("c1") == []


def test_pending_offer_survives_a_restart(mkt, tmp_path):
    ag, store = _agent(mkt, Writer(GOOD), tmp_path)
    _t, btn, _ = _ask(ag, mkt)
    thread = btn[0][1][1].split(":")[1]
    ag2 = AG.MarketAgent(mkt.ctx, Writer(), db_dir=str(tmp_path / "db"), alerts=store)     # new process, same file
    assert ag2.resume("c1", thread, "yes").startswith("✅ Saved alert")


def test_stock_offer_and_down_direction(mkt, tmp_path):
    ag, store = _agent(mkt, Writer(GOOD), tmp_path)
    _t, btn, _ = _ask(ag, mkt, q="why nykaa down", kind="stock_move", target="NYKAA", direction="down")
    assert btn[0][1][0] == "🔔 Alert: new day low"
    assert ag.resume("c1", btn[0][1][1].split(":")[1], "yes").startswith("✅ Saved alert")
    assert store.active("c1")[0]["metric"] == "new_low"


def test_old_checkpoint_files_are_pruned(mkt, tmp_path):
    d = tmp_path / "db"
    d.mkdir()
    (d / "ai_agent_2020-01-01.sqlite").write_text("x")
    _agent(mkt, Writer(), tmp_path)
    assert not (d / "ai_agent_2020-01-01.sqlite").exists()


# ── the number check itself ─────────────────────────────────────────────────
def test_bad_numbers_rules():
    nums, times = AG.evidence_numbers(["NIFTY 50 25,120.4 · day -0.82% · 34 of 50 up · since 13:05",
                                       "MIDCAP 150 -1.10% · RVOL 2.40x"])
    ok_ = lambda t: AG.bad_numbers(t, nums, times)   # noqa: E731
    assert ok_("NIFTY 50 fell 0.82% (about 0.8%), 34 of 50 up since 13:05") == []
    assert ok_("NIFTY 50 near 25,120 and MIDCAP 150 down 1.1%") == []           # rounding + index names
    assert ok_("RVOL 2.4x in the last 15 min, top 3") == []                     # window lengths, small counts
    assert ok_("down 0.91%") == ["0.91"] and ok_("at 13:10") == ["13:10"] and ok_("11 stocks") == ["11"]


def test_writer_prompt_has_rules_inputs_only():
    r = {"kind": "index_move", "labels": ["GAP_LED"], "facts": ["a 1.0%"], "not_checked": ["news"],
         "lean": None, "confidence": None, "brief": ["📈 NIFTY up"]}
    p = AG.writer_prompt("why", r, ["x"])
    assert "BRIEF (already shown):\n- 📈 NIFTY up" in p and "EVIDENCE:\n- a 1.0%" in p
    assert "NOT CHECKED: news" in p and "FIX THESE PROBLEMS" in p
    w = AG.writer_system()
    assert "never claim a cause" in w and "Never advise" in w and "Empty is better than a repeat" in w


# ── the bot: router -> agent, buttons, fallback ─────────────────────────────
class TG:
    def __init__(self):
        self.sent = []

    def send(self, chat, text, buttons=None):
        self.sent.append((chat, text, buttons))
        return True

    def answer_callback(self, cq_id, text=""):
        pass


class Router:
    def __init__(self, name, args):
        self.r = (name, args)

    def route(self, prompt):
        return Route(self.r[0], self.r[1], "fake")


def test_bot_routes_why_to_agent_and_resumes_on_tap(mkt, tmp_path):
    ag, store = _agent(mkt, Writer(GOOD), tmp_path)
    tg = TG()
    bot = ai_chat.Bot(mkt.ctx, Router("explain_market", {"kind": "sector_move", "target": "psu bank"}), tg, ["c1"],
                      str(tmp_path / "logs"), alerts=store, alert_router=Router("cannot_alert", {"reason": "x"}),
                      agent=ag)
    bot.handle({"message": {"chat": {"id": "c1"}, "text": "why are psu banks firing"}})
    chat, text, btn = tg.sent[-1]
    assert text.startswith("<b>📈 PSU BANK up") and btn and btn[0][0][1].startswith("ag:c1-")
    bot.handle({"callback_query": {"id": "1", "data": btn[0][0][1], "message": {"chat": {"id": "c1"}}}})
    assert tg.sent[-1][1].startswith("<b>Why PSU BANK is up")                 # 📊 Details
    bot.handle({"callback_query": {"id": "2", "data": btn[0][1][1], "message": {"chat": {"id": "c1"}}}})
    assert tg.sent[-1][1].startswith("✅ Saved alert #1")
    assert bot.recent("c1")[0]["tool"] == "explain_market"                    # follow-ups ('and IT?') work


def test_bot_falls_back_to_facts_when_agent_breaks(mkt, tmp_path):
    class Broken:
        def ask(self, *a):
            raise RuntimeError("graph bug")
    tg = TG()
    bot = ai_chat.Bot(mkt.ctx, Router("explain_market", {"kind": "index_move"}), tg, ["c1"], str(tmp_path / "l"),
                      agent=Broken())
    bot.handle({"message": {"chat": {"id": "c1"}, "text": "why is nifty up"}})
    assert "WHY NIFTY 50 IS" in tg.sent[-1][1] and "Not advice." in tg.sent[-1][1]


def test_bot_without_agent_still_answers(mkt, tmp_path):
    tg = TG()
    bot = ai_chat.Bot(mkt.ctx, Router("explain_market", {"kind": "rotation"}), tg, ["c1"], str(tmp_path / "l"))
    bot.handle({"message": {"chat": {"id": "c1"}, "text": "which sector is picking up"}})
    assert tg.sent[-1][1].startswith("🔄") and "Day leader vs last 30 min" in tg.sent[-1][1]
