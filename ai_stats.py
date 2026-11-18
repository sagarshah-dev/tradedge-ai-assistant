# -*- coding: utf-8 -*-
"""
ai_stats.py -- how the assistant is doing, from the logs it already writes.  v21, 11-11-2026.

    /stats        today          /stats 7      the last 7 days (max 30)

Reads   logs/ai_chat/ai_chat_YYYY-MM-DD.jsonl    one line per question (written by ai_chat.Bot)
        logs/ai_agent/ai_agent_YYYY-MM-DD.jsonl  one line per WHY answer, with the graph's trace
        evals/scorecard.json                     the last real-model eval run (optional)
Code only: no LLM, no network, read-only. A broken log line is skipped and counted, never fatal.

Why it exists: unit tests say the code is right; they cannot say how the live system behaves. These are
the numbers an on-call person asks for: how many questions, how slow (median AND the slow tail), how
many tokens, how often the model failed, how often code had to repair or reject the model's output.
"""
import json
import os
from datetime import datetime, timedelta

import ai_tools as T

MAX_DAYS = 30


def quantile(xs, q):
    """Nearest-rank quantile; None when empty. The median hides the slow answers: p95 shows them."""
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    k = max(1, int(-(-q * len(xs) // 1)))
    return xs[min(k, len(xs)) - 1]


def read_days(folder, prefix, days, today=None):
    """([records], bad_lines, [days that had a file]) for the last `days` calendar days ending today."""
    today = today or datetime.now()
    recs, bad, seen = [], 0, []
    for i in range(days - 1, -1, -1):
        day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        p = os.path.join(folder or "", "%s_%s.jsonl" % (prefix, day))
        try:
            with open(p, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        seen.append(day)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    recs.append(dict(r, _day=day))
                else:
                    bad += 1
            except ValueError:
                bad += 1
    return recs, bad, seen


def summarize(chat, agent):
    """Pure: log records -> numbers. Every count has ONE definition, written next to it."""
    q = [r for r in chat if r.get("q") is not None and not r.get("alert") and not r.get("button")]
    is_cmd = lambda r: str(r.get("q") or "").strip().startswith("/")  # noqa: E731
    cmds = [r for r in q if is_cmd(r)]
    asked = [r for r in q if not is_cmd(r)]                           # plain-language questions
    routed = [r for r in asked if r.get("model")]                     # the model answered the routing call
    router_err = {}
    for r in asked:
        e = str(r.get("error") or "")
        if e.startswith("router_"):
            router_err[e[7:]] = router_err.get(e[7:], 0) + 1
    tools = {}
    for r in routed:
        tools[r.get("tool") or "?"] = tools.get(r.get("tool") or "?", 0) + 1
    tok = [r["tokens"] for r in routed if isinstance(r.get("tokens"), (int, float))]
    llm = [r.get("llm_ms") for r in routed]
    total = [r.get("ms") for r in asked]
    out = {
        "questions": len(asked), "commands": len(cmds),
        "alert_msgs": sum(1 for r in chat if r.get("alert")), "button_taps": sum(1 for r in chat if r.get("button")),
        "routed": len(routed), "router_errors": router_err,
        "repaired": sum(1 for r in routed if r.get("fixes")),         # code changed the model's tool / arguments
        "cannot": sum(1 for r in routed if r.get("tool") == "cannot_answer" or r.get("unsupported")),
        "tool_errors": sum(1 for r in routed if r.get("error") and not str(r["error"]).startswith("router_")),
        "llm_ms": {"p50": quantile(llm, 0.5), "p95": quantile(llm, 0.95)},
        "total_ms": {"p50": quantile(total, 0.5), "p95": quantile(total, 0.95)},
        "tokens": {"total": int(sum(tok)), "avg": int(sum(tok) / len(tok)) if tok else None},
        "tools": sorted(tools.items(), key=lambda kv: (-kv[1], kv[0])),
        "agent_fallbacks": sum(1 for r in asked if r.get("agent_error")),   # graph failed -> facts-only tool answer
    }
    # the WHY agent, from its own trace
    a = [r for r in agent if r.get("kind")]
    first = retried = dropped = no_draft = 0
    for r in a:
        tr = r.get("trace") or []
        writes = [t for t in tr if t.get("node") == "write" and "attempt" in t]
        verifs = [t for t in tr if t.get("node") == "verify"]
        fin = [t for t in tr if t.get("node") == "finalize"]
        if not writes:
            no_draft += 1                                             # quota / model error / no model: facts only
            continue
        if fin and fin[-1].get("dropped"):
            dropped += 1
        elif verifs and not verifs[0].get("errors"):
            first += 1
        else:
            retried += 1
    ams = [r.get("ms") for r in a]
    out["agent"] = {"answers": len(a), "first_try": first, "after_retry": retried, "dropped": dropped,
                    "no_draft": no_draft, "ms": {"p50": quantile(ams, 0.5), "p95": quantile(ams, 0.95)},
                    "offers": sum(1 for r in a if r.get("offer"))}
    return out


def _ms(v):
    if v is None:
        return "—"
    return "%.1fs" % (v / 1000.0) if v >= 1000 else "%d ms" % v


def _share(a, b):
    return "%d%%" % round(100.0 * a / b) if b else "—"


def _eval_lines(card):
    L = []
    r, w, al = card.get("router"), card.get("writer"), card.get("alerts")
    if r:
        L.append("Router %s%% of %d cases (model alone %s%%) · %s" % (r["accuracy"], r["cases"], r["raw_accuracy"],
                                                                      r["run_at"]))
    if al:
        L.append("Alerts %s%% of %d cases · %s" % (al["accuracy"], al["cases"], al["run_at"]))
    if w:
        L.append("Writer first try %s%% · removed %d · sent unchecked %d · %s" % (
            w["first_try_pct"], w["dropped"], w["shipped_bad"], w["run_at"]))
    return L


def card(chat_dir, agent_dir, evals_dir=None, days=1, today=None):
    """(Rich text, buttons). Never raises on missing folders: an empty period says so."""
    days = max(1, min(MAX_DAYS, int(days or 1)))
    chat, bad1, seen = read_days(chat_dir, "ai_chat", days, today)
    agent, bad2, _s = read_days(agent_dir, "ai_agent", days, today)
    s = summarize(chat, agent)
    span = "today" if days == 1 else "last %d days (%d with activity)" % (days, len(seen))
    L = ["<b>📈 Assistant stats · %s</b>" % span]
    if not chat and not agent:
        L.append("No questions logged in this period.")
    else:
        L.append("<b>Use</b> %d questions · %d commands · %d alert messages · %d button taps" % (
            s["questions"], s["commands"], s["alert_msgs"], s["button_taps"]))
        fail = sum(s["router_errors"].values())
        L.append("<b>Model</b> %d routed · %d failed%s" % (s["routed"], fail, (" (%s)" % ", ".join(
            "%s × %d" % kv for kv in sorted(s["router_errors"].items()))) if fail else ""))
        L.append("<b>Speed</b> model p50 %s · p95 %s · whole answer p50 %s · p95 %s" % (
            _ms(s["llm_ms"]["p50"]), _ms(s["llm_ms"]["p95"]), _ms(s["total_ms"]["p50"]), _ms(s["total_ms"]["p95"])))
        L.append("<b>Tokens</b> %s total · %s per question" % (
            "{:,}".format(s["tokens"]["total"]), "—" if s["tokens"]["avg"] is None else "{:,}".format(s["tokens"]["avg"])))
        L.append("<b>Code checks</b> repaired the model's call %d (%s) · said \"can't\" %d · tool errors %d" % (
            s["repaired"], _share(s["repaired"], s["routed"]), s["cannot"], s["tool_errors"]))
        if s["tools"]:
            L.append("<b>Top tools</b> " + " · ".join("%s %d" % kv for kv in s["tools"][:5]))
        a = s["agent"]
        if a["answers"] or s["agent_fallbacks"]:
            L.append("<b>WHY agent</b> %d answers · AI line ok first try %d · after retry %d · removed %d · "
                     "no draft %d · graph failed %d" % (a["answers"], a["first_try"], a["after_retry"], a["dropped"],
                                                        a["no_draft"], s["agent_fallbacks"]))
            L.append("<b>WHY speed</b> p50 %s · p95 %s" % (_ms(a["ms"]["p50"]), _ms(a["ms"]["p95"])))
        if bad1 or bad2:
            L.append("<i>%d unreadable log line(s) skipped</i>" % (bad1 + bad2))
    ev = []
    if evals_dir:
        try:
            with open(os.path.join(evals_dir, "scorecard.json"), encoding="utf-8") as f:
                ev = _eval_lines(json.load(f))
        except (OSError, ValueError, KeyError, TypeError):
            ev = []
    L += ["", "<b>Last eval run (real model)</b>"] + (ev or ["Not run yet: python evals/run_ai_evals.py --use-bot-key"])
    return T.Rich("\n".join(L)), [[("Today", "st:1"), ("7 days", "st:7"), ("30 days", "st:30")]]
