# -*- coding: utf-8 -*-
"""
ai_odds.py -- "how did signals like this one do before?"  CODE ONLY, no LLM.  v19c, 11-11-2026.

    odds SBIN          SBIN's latest SANKET / SAGA / SHARK signal today -> its look-alikes in the past
    odds SAGA LONG     a stage + side without a stock
    /mlreport          the last ML study (ml_study.py): did a model beat this lookup?

Why a lookup and not a model: 11-11 data check = ~300 independent SANKET/SAGA/SHARK events
over 55 days. A model's edge on that is inside the noise, so the bot shows the honest thing:
what happened to the most similar past signals, with the sample size and a 95% range. The ML
study re-runs as data grows; only a model that beats this lookup OUTSIDE the noise would be used.

Events: one per stock per day (v_signals_1pd, the scanner's own rule), SANKET / SAGA / SHARK
only, with a direction and a scored outcome (bars_n > 0). Today's signals never match themselves.
Look-alike = same stage + side, then RVOL band, PACE band, time of day, NIFTY direction.
'Loosen' drops conditions one step at a time when too few cases match.
"""
import json
import math
import os
import re
import sqlite3

import ai_tools as T

STAGES = ("SANKET", "SAGA", "SHARK")
MIN_CASES = 12                       # fewer than this -> say so and offer 'loosen'
LOOSEN_MAX = 3
SHOW_MAX = 25
_ODDS = re.compile(r"^\s*/?odds\b\s*(.*)$", re.I)
_MLREP = re.compile(r"^\s*/?(ml\s*report|mlreport|ml study)\s*$", re.I)

# loosen level -> which conditions still apply (same stage + side always)
LEVELS = {0: ("rvol", "pace", "time", "mkt"), 1: ("rvol", "pace", "time"), 2: ("rvol", "pace"), 3: ("rvol",)}


def _f(v):
    return T._f(v)


def load_events(db, before_day=None):
    """[{day, t, sym, stage, side, rvol, pace, nifty_pct, sector, m15, m30, m60, mfe, mae, res}] (scored only)."""
    try:
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        rows = c.execute(
            "SELECT s.day, s.t, s.sym, s.stage, o.side, s.rvol, s.pace, s.nifty_pct, s.sector, o.m15, o.m30, "
            "o.m60, o.mfe, o.mae, o.res_10, s.dp15, s.day_pct FROM v_signals_1pd s JOIN outcomes o ON o.day=s.day AND o.sym=s.sym "
            "AND o.stage=s.stage AND o.t=s.t WHERE s.stage IN (%s) AND o.bars_n > 0 AND o.side IS NOT NULL "
            "%s ORDER BY s.day, s.t" % (",".join("?" * len(STAGES)), "AND s.day < ?" if before_day else ""),
            STAGES + ((before_day,) if before_day else ())).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        c.close()
    keys = ("day", "t", "sym", "stage", "side", "rvol", "pace", "nifty_pct", "sector", "m15", "m30", "m60", "mfe",
            "mae", "res", "dp15", "day_pct")
    return [dict(zip(keys, r)) for r in rows]


def latest_signal(db, sym, day):
    """The stock's latest SANKET / SAGA / SHARK row today (the snapshot the scanner logged)."""
    try:
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=10)
        r = c.execute("SELECT day, t, sym, stage, side, rvol, pace, nifty_pct, sector FROM signals WHERE day=? AND "
                      "sym=? AND stage IN (%s) ORDER BY t DESC LIMIT 1" % ",".join("?" * len(STAGES)),
                      (day, sym) + STAGES).fetchone()
        c.close()
    except sqlite3.Error:
        return None
    if not r:
        return None
    return dict(zip(("day", "t", "sym", "stage", "side", "rvol", "pace", "nifty_pct", "sector"), r))


def _bucket(t):
    hm = str(t or "")[:5]
    return "open" if hm < "10:00" else "morning" if hm < "12:00" else "midday" if hm < "14:00" else "late"


BUCKET_WORDS = {"open": "before 10:00", "morning": "10:00-12:00", "midday": "12:00-14:00", "late": "after 14:00"}


def _rvol_band(v, level):
    w = 1.5 if level < 3 else 2.0                      # +-50% (loosened: x2)
    return (v / w, v * w)


def _pace_band(v, level):
    d = 0.25 if level < 3 else 0.5
    return (v - d, v + d)


def match(sig, events, level=0):
    """Past events like `sig` at a loosen level. Returns (cases, rule words)."""
    use = LEVELS[min(level, LOOSEN_MAX)]
    st, side = str(sig.get("stage") or "").upper(), str(sig.get("side") or "").upper()
    words = ["same stage + side (%s %s)" % (st, side)]
    cases = [e for e in events if e["stage"] == st and str(e["side"]).upper() == side]
    rv, pc = _f(sig.get("rvol")), _f(sig.get("pace"))
    if "rvol" in use and rv:
        lo, hi = _rvol_band(rv, level)
        cases = [e for e in cases if _f(e["rvol"]) is not None and lo <= e["rvol"] <= hi]
        words.append("RVOL %.1f–%.1f" % (lo, hi))
    if "pace" in use and pc is not None:
        lo, hi = _pace_band(pc, level)
        cases = [e for e in cases if _f(e["pace"]) is not None and lo <= e["pace"] <= hi]
        words.append("PACE %.2f–%.2f" % (max(0.0, lo), hi))
    if "time" in use and sig.get("t"):
        b = _bucket(sig["t"])
        cases = [e for e in cases if _bucket(e["t"]) == b]
        words.append(BUCKET_WORDS[b])
    mk = _f(sig.get("nifty_pct"))
    if "mkt" in use and mk is not None:
        cases = [e for e in cases if _f(e["nifty_pct"]) is not None and (e["nifty_pct"] >= 0) == (mk >= 0)]
        words.append("NIFTY %s that day" % ("up" if mk >= 0 else "down"))
    return cases, words


def wilson(k, n, z=1.96):
    """95% range for a win rate from k of n (honest for small n, unlike ±2·sqrt(p(1-p)/n))."""
    if not n:
        return (0.0, 0.0)
    p = k / float(n)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def stats(cases):
    n = len(cases)
    w = sum(1 for e in cases if e["res"] == "WIN")
    l_ = sum(1 for e in cases if e["res"] == "LOSS")
    avg = lambda k: (sum(e[k] for e in cases if _f(e[k]) is not None) /                       # noqa: E731
                     max(1, sum(1 for e in cases if _f(e[k]) is not None))) if cases else None
    return {"n": n, "win": w, "loss": l_, "flat": n - w - l_, "rate": (w / float(n)) if n else None,
            "range": wilson(w, n), "m30": avg("m30"), "mfe": avg("mfe"), "mae": avg("mae")}


def _p(v):
    if v is None:
        return "—"
    return "%+.2f%%" % (0.0 if abs(v) < 0.005 else v)


def _pc(v):
    return "—" if v is None else "%.0f%%" % (100 * v)


def card(ctx, query, level=0):
    """(text, buttons, rec). query: 'SBIN' | 'SAGA LONG' | 'SBIN SAGA LONG'."""
    words = query.upper().split()
    stage = next((w for w in words if w in STAGES), None)
    side = next((w for w in words if w in ("LONG", "SHORT")), None)
    rest = [w for w in words if w not in STAGES and w not in ("LONG", "SHORT", "OF", "FOR", "ON", "FOR", "THE")]
    sym = None
    if rest:
        try:
            sym = T._symbol(ctx, {"symbol": rest[0]})
        except T.ToolError as e:
            return str(e), None, {"error": str(e)}
    sig = None
    if sym:
        sig = latest_signal(ctx.saga_db, sym, ctx.today)
        if sig is None and not (stage and side):
            return ("%s has no SANKET / SAGA / SHARK signal today.\nTry: odds %s SAGA LONG (any stage + side)."
                    % (sym, sym)), None, {"sym": sym, "none": True}
    if sig is None:
        if not (stage and side):
            return "Say a stock with a signal today (odds SBIN) or a stage + side (odds SAGA LONG).", None, {}
        sig = {"stage": stage, "side": side, "sym": sym, "t": None}
    if not sig.get("side"):
        return "%s's %s signal has no direction yet: nothing to compare." % (sym, sig["stage"]), None, {}
    events = load_events(ctx.saga_db, before_day=ctx.today)
    if not events:
        return "No scored SANKET / SAGA / SHARK history found in saga_obs.db.", None, {}
    base = stats([e for e in events if e["stage"] == sig["stage"] and str(e["side"]).upper() == sig["side"].upper()])
    cases, rule = match(sig, events, level)
    s = stats(cases)
    head = "🎯 %s%s %s%s" % ((sig.get("sym") + " · ") if sig.get("sym") else "", sig["stage"], sig["side"],
                            (" · %s" % str(sig["t"])[:5]) if sig.get("t") else "")
    L = ["<b>%s</b>" % head]
    if sig.get("t"):
        L.append("Signal: RVOL %s · PACE %s · NIFTY %s" % (
            "%.1fx" % sig["rvol"] if _f(sig.get("rvol")) else "—",
            "%.2f" % sig["pace"] if _f(sig.get("pace")) is not None else "—", _p(_f(sig.get("nifty_pct")))))
    if s["n"]:
        L.append("Similar past signals: <b>%d</b> · WIN %d · FLAT %d · LOSS %d" % (s["n"], s["win"], s["flat"],
                                                                                 s["loss"]))
        L.append("Win %s (likely %s–%s) · all %s %s: %s of %d" % (
            _pc(s["rate"]), _pc(s["range"][0]), _pc(s["range"][1]), sig["stage"], sig["side"], _pc(base["rate"]),
            base["n"]))
        L.append("After 30 min avg %s · best move avg %s · worst avg %s" % (_p(s["m30"]), _p(s["mfe"]), _p(s["mae"])))
    else:
        L.append("No past signal matches all of: %s" % ", ".join(rule))
    if s["n"] < MIN_CASES:
        L.append("⚠️ only %d cases: too few to read%s" % (s["n"], " — tap Loosen" if level < LOOSEN_MAX else ""))
    elif s["range"][1] - s["range"][0] > 0.3:
        L.append("⚠️ wide range: a hint, not a forecast")
    L.append("<i>Match: %s · history to %s · model not used (see /mlreport)</i>" % (
        ", ".join(rule), events[-1]["day"][8:10] + "-" + events[-1]["day"][5:7]))
    key = "%s|%s|%s" % (sig.get("sym") or "", sig["stage"], sig["side"])
    btn = []
    if s["n"]:
        btn.append(("📊 Show the %d" % s["n"], "od:%s:%d:show" % (key, level)))
    if level < LOOSEN_MAX:
        btn.append(("🔁 Loosen match", "od:%s:%d:card" % (key, level + 1)))
    return T.Rich("\n".join(L)), ([btn] if btn else None), {"sym": sig.get("sym"), "stage": sig["stage"],
                                                             "side": sig["side"], "n": s["n"], "level": level}


def show_cases(ctx, query, level):
    sym, stage, side = (query.split("|") + ["", "", ""])[:3]
    sig = latest_signal(ctx.saga_db, sym, ctx.today) if sym else None
    if not sig or sig["stage"] != stage:
        sig = {"stage": stage, "side": side, "sym": sym or None, "t": None}
    cases, rule = match(sig, load_events(ctx.saga_db, before_day=ctx.today), level)
    cases = sorted(cases, key=lambda e: (e["day"], e["t"]), reverse=True)
    L = ["%d similar %s %s signals (newest first)" % (len(cases), stage, side)]
    for e in cases[:SHOW_MAX]:
        L.append("%s-%s %s %-11s %-4s 30m %s · best %s" % (e["day"][8:10], e["day"][5:7], str(e["t"])[:5],
                                                         e["sym"], (e["res"] or "-")[:4], _p(e["m30"]),
                                                         _p(e["mfe"])))
    if len(cases) > SHOW_MAX:
        L.append("… %d more" % (len(cases) - SHOW_MAX))
    L.append("Match: " + ", ".join(rule))
    return "\n".join(L)


def on_button(ctx, data):
    """'od:<sym|stage|side>:<level>:show|card' -> (text, buttons)."""
    try:
        _p_, key, level, what = data.split(":", 3)
        level = max(0, min(LOOSEN_MAX, int(level)))
    except ValueError:
        return "That button is old.", None
    if what == "show":
        return show_cases(ctx, key, level), None
    sym, stage, side = (key.split("|") + ["", "", ""])[:3]
    text, btn, _rec = card(ctx, " ".join(x for x in (sym, stage, side) if x), level)
    return text, btn


def mlreport(ctx):
    p = os.path.join(os.path.dirname(ctx.saga_db), "ml_study.json")
    try:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return ("No ML study yet. After market hours run:  py ml_study.py\n"
                "(it reads saga_obs.db, writes data/ml_study.json; nothing in the live app changes)")
    L = ["<b>🧪 ML study · %s</b>" % r.get("run_at", "?"),
         "%d events (%s) · %d days · %s" % (r["n"], "/".join(r.get("stages", STAGES)), r["days"], r["label"]),
         "Walk-forward: %d test months · %d test events" % (r["folds"], r["n_test"]),
         "Model AUC %.3f vs lookup %.3f · gain %+.3f (95%% range %+.3f to %+.3f)" % (
             r["auc_model"], r["auc_lookup"], r["gain"], r["gain_lo"], r["gain_hi"]),
         ("✅ Model beats the lookup outside the noise" if r["verdict"] == "use" else
          "→ inside the noise: model NOT used · odds show the lookup only")]
    if r.get("top_features"):
        L.append("Most informative: " + ", ".join(r["top_features"][:4]))
    L.append("<i>Re-check when scored events > %d (now %d) · %s</i>" % (r.get("next_at", 600), r["n"],
                                                                     r.get("model", "")))
    return T.Rich("\n".join(L))
