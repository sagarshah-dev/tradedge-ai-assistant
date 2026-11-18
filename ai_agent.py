# -*- coding: utf-8 -*-
"""
ai_agent.py -- the WHY agent, built with LangGraph.  v19, 09-11-2026.

    "why are psu banks firing?"
        router (ai_llm, 1 call)  -> explain_market(kind=sector_move, target=NIFTY PSU BANK)
        plan        code    which checks this kind needs (ai_evidence.plan)
        check  x N  code    run IN PARALLEL (LangGraph Send): sector, path, sectors, VIX, market ...
        analyze     code    pattern labels, candidates, confidence (rules, testable)
        write       LLM     a short view over the evidence (1 call, structured output)
        verify      code    every number must exist in the evidence; no advice words; else back
                            to write (max 2 retries); still wrong -> the bad lines are dropped
        finalize    code    facts (code) + checked view (LLM) + labels + "not checked"
        offer       human   "🔔 Alert if PSU BANK makes a new day low?" -> interrupt(); the user's
                            button resumes the SAME run (SQLite checkpointer: survives a restart)

Five LangGraph patterns, each for a real need: conditional routing (by question kind),
parallel fan-out (Send), a verifier loop (cycle with a retry limit), human-in-the-loop
(interrupt + Command(resume)), persistence (checkpointer). It is a controlled WORKFLOW,
not a free-roaming agent: in a trading context every number must be checkable.
Without a model (quota, outage) it still answers: facts + labels from code.
"""
import glob
import json
import operator
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Annotated, Optional, TypedDict

import ai_evidence as E
import ai_tools as T

MAX_RETRIES = 2                      # writer attempts after the first one
KEEP_DB_DAYS = 3                     # checkpoint files (pending alert offers) kept this long

# ── the writer: structured output only (function call with mode=ANY) ─────────
WRITE_TOOL = [{
    "name": "write_view",
    "description": "ONE insight the brief does not already say, plus reasons only when a LEAN is given.",
    "parameters": {"type": "OBJECT", "properties": {
        "insight": {"type": "STRING", "description": "Max 20 words. A link or conflict between facts that the BRIEF "
                                                     "does not state. Empty if there is nothing new."},
        "why": {"type": "ARRAY", "items": {"type": "STRING"},
                "description": "Only when LEAN is given: up to 2 reasons the evidence favours it."},
        "against": {"type": "ARRAY", "items": {"type": "STRING"},
                    "description": "Only when LEAN is given: the strongest counter-point."},
        "changes_view": {"type": "STRING", "description": "Only when LEAN is given: what in the data would change it."}},
        "required": []}}]


def writer_system(_sectors=None, _today=None):
    return (
        "You add ONE insight to a market answer for an intraday trader on NSE, read on a phone during trading.\n"
        "The trader already sees the BRIEF. Repeating it is useless.\n"
        "Rules:\n"
        "1. `insight`: max 20 words. Only a link or conflict the BRIEF does not say (e.g. 'index flat but PSU banks "
        "and IT pull opposite ways', 'price up but OI is short covering, not fresh longs'). If you have nothing new, "
        "leave it EMPTY. Empty is better than a repeat.\n"
        "2. Use ONLY numbers that appear in EVIDENCE (you may round). Prefer no numbers at all.\n"
        "3. Describe what moved together; never claim a cause ('because', 'due to', 'driven by').\n"
        "4. Never advise: no buy, sell, entry, exit, target, stop loss, 'should'.\n"
        "5. why / against / changes_view ONLY when LEAN is given; never pick a different one.\n"
        "6. Things under NOT CHECKED cannot be used or guessed. English.")


def writer_prompt(question, result, fix=None):
    c = result.get("confidence") or {}
    L = ["QUESTION: " + question, "KIND: " + result["kind"],
         "LABELS (by code): " + (", ".join(E.LABEL_TEXT.get(x, x) for x in result["labels"]) or "none")]
    if result.get("lean"):
        L.append("LEAN (by code): %s%s" % (result["lean"], (" · confidence %s, %d of %d checks agree" %
                                                            (c["level"], c["k"], c["n"])) if c else ""))
    L += ["BRIEF (already shown):"] + ["- " + x for x in result.get("brief") or []]
    L += ["EVIDENCE:"] + ["- " + f for f in result["facts"]]
    L.append("NOT CHECKED: " + ", ".join(result["not_checked"]))
    if fix:
        L.append("FIX THESE PROBLEMS in your last answer: " + "; ".join(fix))
    return "\n".join(L)


# ── the verifier: code checks the model ──────────────────────────────────────
_ADVICE_W = re.compile(r"\b(buy|buying opportunity|sell (it|now)|go long|go short|entry|exit|target\w*|stop\s*-?\s*loss|"
                       r"recommend\w*|should (you|i|we|consider)|accumulate|book profit\w*)\b", re.I)
_CAUSE_W = [(re.compile(r"\bbecause of\b", re.I), "alongside"), (re.compile(r"\bdue to\b", re.I), "alongside"),
            (re.compile(r"\bdriven by\b", re.I), "alongside"), (re.compile(r"\bcaused by\b", re.I), "alongside"),
            (re.compile(r"\bbecause\b", re.I), "while")]
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")
FREE_INTS = set(range(0, 11)) | {15, 30, 45, 60, 120}       # counts, ranks, window lengths


def _names_pattern():
    names = set(T.EXTRA_NAMES) | {"NIFTY 50", "INDIA VIX", "NIFTY BANK", "NIFTY 500"}
    with T._SAMPLE_LOCK:
        names |= set(T._ISAMPLES)
    names |= {n.replace("NIFTY ", "") for n in list(names)}
    return re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True) if n), re.I)


def evidence_numbers(facts, extra_text=""):
    nums, times = [], set()
    blob = "\n".join(facts) + "\n" + (extra_text or "")
    times |= set(_TIME.findall(blob))
    clean = _names_pattern().sub(" ", _TIME.sub(" ", blob))
    for t in _NUM.findall(clean):
        try:
            nums.append(abs(float(t.replace(",", "").replace("−", "-"))))
        except ValueError:
            pass
    return nums, times


def bad_numbers(text, nums, times):
    """Numbers / times in `text` that are not in the evidence (rounding allowed)."""
    bad = [t for t in _TIME.findall(text) if t not in times]
    clean = _names_pattern().sub(" ", _TIME.sub(" ", text))
    for t in _NUM.findall(clean):
        raw = t.replace(",", "").replace("−", "-")
        try:
            v = abs(float(raw))
        except ValueError:
            continue
        d = len(raw.split(".")[1]) if "." in raw else 0
        if d == 0 and v in FREE_INTS:
            continue
        if not any(round(e, d) == round(v, d) for e in nums):
            bad.append(t)
    return bad


def _items(view):
    """[(field, index or None, text)] for every string the model wrote."""
    out = []
    for k in ("insight", "changes_view"):
        if view.get(k):
            out.append((k, None, str(view[k])))
    for k in ("why", "against"):
        for i, x in enumerate(view.get(k) or []):
            out.append((k, i, str(x)))
    return out


def verify(view, result, question=""):
    """List of problems, empty = accepted."""
    if not view:
        return []
    nums, times = evidence_numbers(result["facts"] + (result.get("brief") or []), question)
    probs = []
    for k, i, txt in _items(view):
        b = bad_numbers(txt, nums, times)
        if b:
            probs.append("%s%s uses numbers not in EVIDENCE: %s" % (k, "" if i is None else "[%d]" % i, ", ".join(b)))
        if _ADVICE_W.search(txt):
            probs.append("%s%s has advice words ('%s')" % (k, "" if i is None else "[%d]" % i,
                                                          _ADVICE_W.search(txt).group(0)))
    if not result.get("lean") and (view.get("why") or view.get("against") or view.get("changes_view")):
        probs.append("why / against / changes_view are only for a LEAN given by code")
    ins = view.get("insight") or ""
    if len(ins.split()) > 25:
        probs.append("insight is %d words; max 20" % len(ins.split()))
    return probs


def clean_view(view, result, question=""):
    """Last resort after the retries: drop every item that still breaks a rule; soften cause words."""
    if not view:
        return None, []
    nums, times = evidence_numbers(result["facts"] + (result.get("brief") or []), question)
    out, dropped = {}, []
    for k, i, txt in _items(view):
        for rx, sub in _CAUSE_W:
            txt = rx.sub(sub, txt)
        if bad_numbers(txt, nums, times) or _ADVICE_W.search(txt) or (k != "insight" and not result.get("lean")) \
                or (k == "insight" and len(txt.split()) > 25):
            dropped.append(k)
            continue
        if i is None:
            out[k] = txt
        else:
            out.setdefault(k, []).append(txt)
    return (out or None), dropped


def normalize(view):
    """Model output -> {summary: str, points: [str], why: [str], against: [str], changes_view: str}."""
    if not isinstance(view, dict):
        return None
    out = {}
    for k in ("insight", "changes_view"):
        if view.get(k) and str(view[k]).strip():
            out[k] = str(view[k]).strip()[:300]
    for k, cap in (("why", 2), ("against", 1)):
        x = view.get(k)
        if isinstance(x, str):
            x = [x]
        x = [str(i).strip()[:300] for i in (x or []) if str(i).strip()][:cap]
        if x:
            out[k] = x
    return out or None


def soften(view):
    """Cause words are rewritten in code, not bounced back: cheaper than a retry."""
    if not view:
        return view
    v = {}
    for k, x in view.items():
        if isinstance(x, str):
            for rx, sub in _CAUSE_W:
                x = rx.sub(sub, x)
        elif isinstance(x, list):
            y = []
            for s in x:
                s = str(s)
                for rx, sub in _CAUSE_W:
                    s = rx.sub(sub, s)
                y.append(s)
            x = y
        v[k] = x
    return v


# ── the alert offer (human in the loop) ──────────────────────────────────────
def make_offer(result):
    """One alert that follows from the answer, or None. Built by code, saved only on a tap."""
    a, d = result["args"], result.get("direction")
    down = d == "down"
    idx = None
    if result["kind"] in ("index_move", "vix"):
        idx = a.get("target")
    elif result["kind"] == "sector_move":
        idx = (result["evidence"].get("sector") or {}).get("index")
    elif result["kind"] in ("strongest_sectors", "rotation") and result.get("lean"):
        cand = next((c for c in result.get("candidates") or [] if c.get("name") == result["lean"]), {})
        idx = cand.get("sector") or ("NIFTY " + result["lean"])
        with T._SAMPLE_LOCK:
            if idx not in T._ISAMPLES:
                idx = None
    if idx:
        metric = "day_low_down" if down else "day_high_up"
        return {"label": "Alert if %s makes a new day %s" % (E._label(idx), "low" if down else "high"),
                "args": {"target": "index", "metric": metric, "symbols": [idx], "window_min": 5},
                "text": "alert me when %s makes a new day %s" % (idx.lower(), "low" if down else "high")}
    if result["kind"] == "stock_move" and a.get("target"):
        return {"label": "Alert if %s makes a new day %s" % (a["target"], "low" if down else "high"),
                "args": {"target": "stocks", "metric": "new_low" if down else "new_high", "symbols": [a["target"]]},
                "text": "alert me when %s makes a new day %s" % (a["target"].lower(), "low" if down else "high")}
    return None


# ═════════════════════════════════════════════════════════════════════════════
# The graph
# ═════════════════════════════════════════════════════════════════════════════
def _merge(a, b):
    d = dict(a or {})
    d.update(b or {})
    return d


class AgentState(TypedDict, total=False):
    question: str
    args: dict
    plan: list
    evidence: Annotated[dict, _merge]          # parallel checks write their own key; merged here
    result: dict
    view: Optional[dict]
    errors: list
    attempts: int
    view_note: str
    answer: str
    final: dict
    offer: Optional[dict]
    alert_reply: str
    trace: Annotated[list, operator.add]


def build_graph(ctx, writer=None, alert_maker=None):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send, interrupt

    def plan_node(s):
        a = s["args"]
        p = E.plan(a["kind"], a.get("target"), a.get("compare"), a.get("direction"), a.get("window_min") or 30)
        return {"plan": [list(x) for x in p], "trace": [{"node": "plan", "checks": [x[0] for x in p]}]}

    def fan_out(s):
        return [Send("check", {"cid": cid, "name": name, "kw": kw}) for cid, name, kw in s["plan"]]

    def check_node(p):
        t0 = time.time()
        r = E.run_check(ctx, p["name"], p["kw"])
        return {"evidence": {p["cid"]: r},
                "trace": [{"node": "check:" + p["cid"], "ms": int((time.time() - t0) * 1000),
                           "missing": r.get("missing")}]}

    def analyze_node(s):
        if all((v or {}).get("missing") for v in s["evidence"].values()):
            raise T.ToolError("No live data to explain yet (market closed or app just started).")
        r = E.assemble(ctx, s["args"]["kind"], s["args"], s["evidence"])
        r["evidence"] = {k: {"missing": v.get("missing")} if v.get("missing") else {"ok": True}
                         for k, v in r["evidence"].items()}          # facts carry the numbers; keep state small
        r["_index"] = (s["evidence"].get("sector") or {}).get("index")
        return {"result": r, "offer": None, "trace": [{"node": "analyze", "labels": r["labels"], "lean": r["lean"]}]}

    def write_node(s):
        n = (s.get("attempts") or 0) + 1
        if writer is None:
            return {"view": None, "attempts": n, "view_note": "", "trace": [{"node": "write", "skipped": "no model"}]}
        prompt = writer_prompt(s["question"], s["result"], s.get("errors") or None)
        t0 = time.time()
        try:
            route = writer.route(prompt)
        except Exception as e:
            kind = getattr(e, "kind", type(e).__name__)
            note = {"quota_day": "AI view skipped: daily Gemini limit (resets ~13:30 IST). Facts above are complete.",
                    "quota": "AI view skipped: Gemini busy. Facts above are complete."}.get(
                        kind, "AI view skipped (%s). Facts above are complete." % kind)
            return {"view": None, "attempts": n, "view_note": note,
                    "trace": [{"node": "write", "error": kind, "ms": int((time.time() - t0) * 1000)}]}
        view = soften(normalize(dict(route.args or {})) if route.name == "write_view" else None)
        return {"view": view, "attempts": n,
                "trace": [{"node": "write", "attempt": n, "model": getattr(route, "model", ""),
                           "tokens": getattr(route, "tokens", None), "ms": int((time.time() - t0) * 1000)}]}

    def verify_node(s):
        errs = verify(s.get("view"), s["result"], s["question"])
        return {"errors": errs, "trace": [{"node": "verify", "attempt": s.get("attempts"), "errors": errs}]}

    def after_verify(s):
        if s.get("view") and s.get("errors") and (s.get("attempts") or 0) <= MAX_RETRIES:
            return "write"
        return "finalize"

    def finalize_node(s):
        r = dict(s["result"])
        view, dropped = (s.get("view"), [])
        if view and s.get("errors"):
            view, dropped = clean_view(view, r, s["question"])
        r["view"] = view
        r["view_note"] = s.get("view_note") or ("(%d AI line(s) removed: failed the number check)" % len(dropped)
                                                 if dropped else "")
        r["agent"] = True
        r_off = dict(r, evidence={"sector": {"index": r.get("_index")}})
        return {"answer": E.render_brief_html(r), "final": r, "offer": make_offer(r_off),
                "trace": [{"node": "finalize", "dropped": dropped, "view": bool(view)}]}

    def offer_node(s):
        off = s.get("offer")
        if not off:
            return {}
        choice = interrupt({"offer": off})                 # pauses here; a button tap resumes
        if choice != "yes":
            return {"alert_reply": "OK, no alert.", "trace": [{"node": "offer", "choice": "no"}]}
        reply = alert_maker(off) if alert_maker else "Alerts are off."
        return {"alert_reply": reply, "trace": [{"node": "offer", "choice": "yes"}]}

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("check", check_node)
    g.add_node("analyze", analyze_node)
    g.add_node("write", write_node)
    g.add_node("verify", verify_node)
    g.add_node("finalize", finalize_node)
    g.add_node("offer", offer_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["check"])
    g.add_edge("check", "analyze")
    g.add_edge("analyze", "write")
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", after_verify, ["write", "finalize"])
    g.add_edge("finalize", "offer")
    g.add_edge("offer", END)
    return g


class MarketAgent:
    """What ai_chat talks to. ask() -> (text, buttons, log record); resume() for the offer button."""

    def __init__(self, ctx, writer=None, db_dir=None, alerts=None, log_dir=None):
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.ctx, self.writer, self.alerts, self.log_dir = ctx, writer, alerts, log_dir
        self.db_dir = db_dir or os.path.join(getattr(ctx.state, "DATA_DIR", "data"))
        os.makedirs(self.db_dir, exist_ok=True)
        self._prune()
        # one file per day: a restart keeps today's pending offers; old days are deleted
        self.db_path = os.path.join(self.db_dir, "ai_agent_%s.sqlite" % datetime.now().strftime("%Y-%m-%d"))
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.saver = SqliteSaver(self._conn)
        self._chat = threading.local()
        self.graph = build_graph(ctx, writer, self._make_alert).compile(checkpointer=self.saver)

    def _prune(self):
        cut = (datetime.now() - timedelta(days=KEEP_DB_DAYS)).strftime("%Y-%m-%d")
        for p in glob.glob(os.path.join(self.db_dir, "ai_agent_*.sqlite")):
            if os.path.basename(p)[9:19] < cut:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _make_alert(self, off):
        if self.alerts is None:
            return "Alerts are off (no alert store)."
        import ai_alerts
        chat = getattr(self._chat, "id", "")
        try:
            rule, missing, _notes = ai_alerts.validate_rule(self.ctx, off["args"], off["text"])
            if missing:
                return "Could not make that alert (%s missing)." % ", ".join(missing)
            aid = self.alerts.add(chat, rule, off["text"], self.ctx.today)
            return "✅ Saved alert #%d\n%s\n%s\n/alerts to change, pause or delete." % (
                aid, ai_alerts.describe(rule), ai_alerts.when_text(rule))
        except T.ToolError as e:
            return "Could not save the alert: %s" % e

    def _log(self, rec):
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            p = os.path.join(self.log_dir, "ai_agent_%s.jsonl" % datetime.now().strftime("%Y-%m-%d"))
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print("[AI agent] log write failed: %s" % e)

    def ask(self, chat, args, question):
        thread = "%s-%d" % (chat, int(time.time() * 1000))
        cfg = {"configurable": {"thread_id": thread}}
        t0 = time.time()
        self._chat.id = chat
        try:
            out = self.graph.invoke({"question": question, "args": dict(args), "attempts": 0, "errors": [],
                                     "trace": []}, cfg)
        except T.ToolError as e:
            return str(e), None, {"thread": thread, "kind": args.get("kind"), "error": str(e)}
        row = [("📊 Details", "ag:%s:details" % thread)]          # the brief is short; details on tap
        intr = out.get("__interrupt__") or []
        if intr:
            off = (intr[0].value or {}).get("offer") or {}
            short = "🔔 Alert: new day %s" % ("low" if "day low" in off.get("label", "") else "high")
            row.append((short, "ag:%s:yes" % thread))          # the target is in the answer above; keep it 1 line
        buttons = [row]
        trace = out.get("trace") or []
        rec = {"thread": thread, "kind": args.get("kind"), "target": args.get("target"),
               "attempts": out.get("attempts"), "errors_left": out.get("errors") or [],
               "labels": (out.get("result") or {}).get("labels"), "lean": (out.get("result") or {}).get("lean"),
               "ms": int((time.time() - t0) * 1000), "trace": trace, "offer": bool(intr)}
        self._log(dict(rec, ts=datetime.now().strftime("%H:%M:%S"), q=question))
        return T.Rich(out.get("answer") or "No answer (bug logged)."), buttons, rec

    def resume(self, chat, thread, choice):
        from langgraph.types import Command
        if not str(thread).startswith(str(chat) + "-"):
            return "That button belongs to another chat."
        self._chat.id = chat
        cfg = {"configurable": {"thread_id": thread}}
        if choice == "details":                            # read the saved run; never consumes the alert offer
            try:
                snap = self.graph.get_state(cfg)
                fin = (snap.values or {}).get("final") if snap else None
            except Exception:
                fin = None
            return T.Rich(E.render_details(fin)) if fin else \
                "Those details are no longer kept (older than %d days)." % KEEP_DB_DAYS
        try:
            snap = self.graph.get_state(cfg)
            if not snap or not snap.next:
                return "That offer has already been answered or expired."
            out = self.graph.invoke(Command(resume=choice), cfg)
        except Exception as e:
            return "Could not finish that step: %s" % e
        self._log({"ts": datetime.now().strftime("%H:%M:%S"), "thread": thread, "resume": choice})
        return out.get("alert_reply") or "Done."
