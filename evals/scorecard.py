# -*- coding: utf-8 -*-
"""
scorecard.py -- turns eval run records into numbers a person (or a README) can read.  v21, 11-11-2026.

Pure functions: no model, no network, no clock except the run stamp passed in. run_ai_evals.py calls
update() after each run; the result is evals/scorecard.json (machine) and evals/SCORECARD.md (human).
Each section (router / alerts / writer) keeps the numbers of ITS last full run, with the model and time,
so running one part never erases another.

What each number means
  router.accuracy       right tool AND every expected argument, after the code repair layer (what the user gets)
  router.raw_accuracy   the same check on the model's own output, before any repair
  router.repaired       cases the model got wrong and code fixed  -> what the repair layer is worth
  router.broken         cases the model got right and code broke  -> must stay 0
  writer.first_try      AI line accepted by the verifier on the first draft
  writer.after_retry    accepted after the verifier's feedback (max 2 retries)
  writer.dropped        still wrong after retries: the line was removed, the facts were still sent
  writer.shipped_bad    answers that went out with a number that is not in the evidence -> must stay 0
"""
import json
import os

SECTIONS = ("router", "alerts", "writer")


def quantile(xs, q):
    """Nearest-rank quantile (q in 0..1) of a list; None when empty. p95 of 20 values = the 19th."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(1, int(-(-q * len(xs) // 1)))            # ceil(q * n), at least 1
    return xs[min(k, len(xs)) - 1]


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def summarize_router(recs, model, run_at):
    """recs: one dict per case from run_ai_evals (pass, pass_raw, want_tool, ms, tokens, error, fixes)."""
    n = len(recs)
    ok = [r for r in recs if r.get("pass")]
    raw = [r for r in recs if r.get("pass_raw")]
    by = {}
    for r in recs:
        b = by.setdefault(r.get("want_tool") or "?", [0, 0])
        b[0] += bool(r.get("pass"))
        b[1] += 1
    errs = {}
    for r in recs:
        if r.get("error"):
            k = str(r["error"]).split(":", 1)[0]
            errs[k] = errs.get(k, 0) + 1
    ms = [r.get("ms") for r in recs if r.get("ms")]
    tok = [r.get("tokens") for r in recs if r.get("tokens")]
    return {
        "model": model, "run_at": run_at, "cases": n, "passed": len(ok), "accuracy": _pct(len(ok), n),
        "raw_passed": len(raw), "raw_accuracy": _pct(len(raw), n),
        "repaired": sum(1 for r in recs if r.get("pass") and not r.get("pass_raw")),
        "broken": sum(1 for r in recs if r.get("pass_raw") and not r.get("pass")),
        "model_errors": errs,
        "latency_ms": {"p50": quantile(ms, 0.5), "p95": quantile(ms, 0.95)},
        "avg_tokens": int(sum(tok) / len(tok)) if tok else None,
        "by_tool": {k: {"passed": v[0], "cases": v[1]} for k, v in sorted(by.items())},
        "failed": [{"q": r.get("q"), "want": r.get("want_tool"), "got": r.get("tool") or r.get("error")}
                   for r in recs if not r.get("pass")][:40],
    }


def summarize_alerts(recs, model, run_at):
    n = len(recs)
    ok = sum(1 for r in recs if r.get("pass"))
    return {"model": model, "run_at": run_at, "cases": n, "passed": ok, "accuracy": _pct(ok, n),
            "failed": [{"q": r.get("q"), "got": r.get("got")} for r in recs if not r.get("pass")][:40]}


def summarize_writer(recs, model, run_at):
    """recs: one dict per scenario (attempts, first_ok, wrote, dropped, shipped_bad, error, ms)."""
    n = len(recs)
    wrote = [r for r in recs if r.get("wrote")]                       # the model returned a draft at all
    first = sum(1 for r in wrote if r.get("first_ok"))
    clean = sum(1 for r in wrote if not r.get("dropped"))
    ms = [r.get("ms") for r in recs if r.get("ms")]
    return {
        "model": model, "run_at": run_at, "scenarios": n, "drafts": len(wrote),
        "first_try": first, "first_try_pct": _pct(first, len(wrote)),
        "after_retry": clean - first, "clean_pct": _pct(clean, len(wrote)),
        "dropped": sum(1 for r in wrote if r.get("dropped")),
        "shipped_bad": sum(1 for r in recs if r.get("shipped_bad")),
        "no_draft": n - len(wrote),
        "avg_attempts": round(sum(r.get("attempts") or 0 for r in wrote) / float(len(wrote)), 2) if wrote else None,
        "latency_ms": {"p50": quantile(ms, 0.5), "p95": quantile(ms, 0.95)},
        "rejected_for": _reasons(recs),
    }


def _reasons(recs):
    out = {}
    for r in recs:
        for e in r.get("first_errors") or []:
            k = ("number not in evidence" if "numbers not in EVIDENCE" in e else "advice word" if "advice words" in e
                 else "too long" if "words; max" in e else "lean-only field" if "LEAN" in e else "other")
            out[k] = out.get(k, 0) + 1
    return out


def _n(v, unit=""):
    return "—" if v is None else "%s%s" % (v, unit)


def markdown(card):
    """The human page. Sections that never ran say so instead of showing zeros."""
    L = ["# Scorecard", "",
         "Real-model results, written by `evals/run_ai_evals.py`. Every number comes from the last full run of",
         "that part; nothing here is typed by hand.", ""]
    r = card.get("router")
    L += ["## Router: question → tool + arguments", ""]
    if r:
        L += ["| | |", "|---|---|",
              "| Model | `%s` |" % r["model"], "| Run | %s |" % r["run_at"], "| Cases | %d |" % r["cases"],
              "| **Accuracy (what the user gets)** | **%s%%** (%d / %d) |" % (r["accuracy"], r["passed"], r["cases"]),
              "| Model alone, before code repair | %s%% (%d / %d) |" % (r["raw_accuracy"], r["raw_passed"], r["cases"]),
              "| Fixed by the repair layer | %d |" % r["repaired"],
              "| Broken by the repair layer | %d |" % r["broken"],
              "| Latency p50 / p95 | %s / %s |" % (_n(r["latency_ms"]["p50"], " ms"), _n(r["latency_ms"]["p95"], " ms")),
              "| Avg tokens per question | %s |" % _n(r["avg_tokens"])]
        if r.get("model_errors"):
            L.append("| Model errors | %s |" % ", ".join("%s × %d" % kv for kv in sorted(r["model_errors"].items())))
        L += ["", "| Tool | Passed | Cases |", "|---|---|---|"]
        L += ["| %s | %d | %d |" % (k, v["passed"], v["cases"]) for k, v in r["by_tool"].items()]
        if r.get("failed"):
            L += ["", "Failed cases:", ""] + ["- `%s` → wanted `%s`, got `%s`" % (f["q"], f["want"], f["got"])
                                               for f in r["failed"]]
    else:
        L.append("_Not run yet: `python evals/run_ai_evals.py`_")
    a = card.get("alerts")
    L += ["", "## Alerts: sentence → alert rule", ""]
    if a:
        L += ["| | |", "|---|---|", "| Model | `%s` |" % a["model"], "| Run | %s |" % a["run_at"],
              "| **Accuracy** | **%s%%** (%d / %d) |" % (a["accuracy"], a["passed"], a["cases"])]
    else:
        L.append("_Not run yet: `python evals/run_ai_evals.py --alerts`_")
    w = card.get("writer")
    L += ["", "## Writer: the one AI sentence, checked by code", ""]
    if w:
        L += ["| | |", "|---|---|", "| Model | `%s` |" % w["model"], "| Run | %s |" % w["run_at"],
              "| Scenarios (fake market) | %d |" % w["scenarios"],
              "| Accepted on the first draft | %s%% (%d / %d) |" % (_n(w["first_try_pct"]), w["first_try"], w["drafts"]),
              "| Accepted after the verifier's feedback | %d more |" % w["after_retry"],
              "| Line removed (still wrong after retries) | %d |" % w["dropped"],
              "| **Answers sent with an unchecked number** | **%d** |" % w["shipped_bad"],
              "| No draft (model error / quota) | %d |" % w["no_draft"],
              "| Avg model calls per answer | %s |" % _n(w["avg_attempts"]),
              "| Latency p50 / p95 (whole answer) | %s / %s |" % (_n(w["latency_ms"]["p50"], " ms"),
                                                                 _n(w["latency_ms"]["p95"], " ms"))]
        if w.get("rejected_for"):
            L.append("| First drafts rejected for | %s |" % ", ".join(
                "%s × %d" % kv for kv in sorted(w["rejected_for"].items(), key=lambda kv: -kv[1])))
    else:
        L.append("_Not run yet: `python evals/run_ai_evals.py --writer`_")
    return "\n".join(L) + "\n"


def load(folder):
    try:
        with open(os.path.join(folder, "scorecard.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def update(folder, section, summary):
    """Replace ONE section, keep the others, write both files. Returns the whole card."""
    assert section in SECTIONS, section
    card = load(folder)
    card[section] = summary
    tmp = os.path.join(folder, "scorecard.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(folder, "scorecard.json"))
    with open(os.path.join(folder, "SCORECARD.md"), "w", encoding="utf-8") as f:
        f.write(markdown(card))
    return card
