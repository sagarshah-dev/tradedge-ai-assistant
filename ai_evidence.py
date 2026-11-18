# -*- coding: utf-8 -*-
"""
ai_evidence.py -- the checks behind every WHY question. CODE ONLY, no LLM.  v19, 09-11-2026.

    "why is nifty down"  ->  plan(kind)          which checks this kind of question needs
                         ->  CHECKS[name](...)   each reads the live caches, returns numbers + fact lines
                         ->  analyze(...)        pattern labels, candidates, confidence (rules, not the LLM)
                         ->  render(...)         a complete answer even with NO model (quota out, agent off)

The LangGraph agent (ai_agent.py) runs the same checks in parallel and adds a written view on
top; the verifier there only accepts numbers that appear in these fact lines. So this file is
the single source of truth for every number the assistant says.

Wording rule: evidence shows what MOVED TOGETHER. It never claims a cause and never advises.
"""
import os
import re
from datetime import datetime

import ai_tools as T

ALWAYS_NOT_CHECKED = ["news", "global cues (GIFT Nifty, US markets)", "FII / DII (NSE publishes after the close)"]
SECTOR_PARENT = {"NIFTY PSU BANK": "NIFTY BANK", "NIFTY PVT BANK": "NIFTY BANK", "NIFTY PSU BK": "NIFTY BANK"}
# Businesses whose costs or revenue follow crude (domain fact, not a strategy). Only F&O names present are used.
CRUDE_SENSITIVE = {"Oil marketers": ["BPCL", "HINDPETRO", "IOC"], "Upstream oil": ["ONGC", "OIL"],
                   "Paints": ["ASIANPAINT", "BERGEPAINT"], "Aviation": ["INDIGO"], "Tyres": ["MRF", "APOLLOTYRE"]}
BROADER = ("NIFTY MIDCAP 150", "NIFTY MIDCAP SELECT", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250")
FLAT_PCT = 0.15                 # |day %| below this is "flat", not up / down
# "What is it doing NOW" (display words only): how far from the day high / low, and which way the
# last 30 min went. An index up 1% on the day can still be falling: the day % alone hides that.
FADE_OFF = 0.25                 # % away from the day high (low) before "fading" ("recovering") is said
NOW_MOVE = 0.10                 # ... and the last 30 min must have moved at least this much that way
AT_EDGE = 10.0                  # within 10% of the day range of the high / low = "at the day high / low"
# Largest members by index weight, in rough order (weights change slowly; edit here when they do).
# NO weights are used: the answer only says how many of them are up, never a point contribution.
HEAVY = {"NIFTY 50": ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL", "LT", "ITC", "TCS", "SBIN",
                      "AXISBANK"],
         "NIFTY BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"]}
OI_CODE = {"long_buildup": "LB", "short_buildup": "SB", "short_covering": "SC", "long_unwinding": "LU"}
TABLE_ROWS = 12                 # a stock table longer than this shows the top and the bottom
SPLIT_INDEX_MAX, SPLIT_UP, SPLIT_DOWN = 0.3, 1.0, 0.7
FACTORS = ("day", "breadth", "vwap", "persist", "participation", "oi", "room")
FACTOR_WORDS = {"day": "day move", "breadth": "breadth", "vwap": "above VWAP", "persist": "rank held 60 min",
                "participation": "RVOL", "oi": "OI quality", "room": "range left"}


def _f(v):
    return T._f(v)


def _p(v, d=2):
    if v is None:
        return "—"
    if abs(v) < 0.5 * 10 ** -d:                        # no "-0.00%"
        v = 0.0
    return "%+.*f%%" % (d, v)


def _px(v):
    if v is None:
        return "—"
    return "{:,.1f}".format(v) if abs(v) >= 1000 else "%.2f" % v


def _x(v):
    return "—" if v is None else "%.2fx" % v


def _short(sec):
    return T._short(sec)


def _label(name):
    """'NIFTY 50' stays 'NIFTY 50' (short '50' reads badly); sectors lose the NIFTY prefix."""
    return name if name in ("NIFTY 50", "NIFTY BANK", "INDIA VIX", "NIFTY 500") else _short(name)


def _cfg(name, default=None):
    try:
        import config
        v = getattr(config, name, None)
        if v not in (None, ""):
            return v
    except Exception:
        pass
    return os.environ.get(name, default)


def _missing(why):
    return {"missing": why, "facts": []}


# ── shared readers ────────────────────────────────────────────────────────────
def _stock_results(ctx):
    return {k: v for k, v in T._live_results(ctx, required=False).items() if not T.is_index_sym(ctx, k)}


def _series(ctx, name):
    """Today's index samples [(epoch, price, high, low, vwap, pct)] (minute sampler, whole session)."""
    with T._SAMPLE_LOCK:
        xs = list(T._ISAMPLES.get(name) or ())
    day = ctx.today
    return [x for x in xs if datetime.fromtimestamp(x[0]).strftime("%Y-%m-%d") == day and x[1]]


def _hm(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _at(xs, mark):
    a = [x for x in xs if x[0] <= mark]
    return a[-1] if a else None


def _sector_names(ctx):
    return sorted({str(v).upper() for v in T._secmap(ctx).values() if v} - {"INDEX"})


def _members(ctx, sector, res):
    return [s for s in res if str(T._secmap(ctx).get(s, "")).upper() == sector]


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def _pct_change_window(xs, minutes, now_ts=None):
    """% price change over the last N min of an index series, or None."""
    if len(xs) < 2:
        return None
    now = xs[-1]
    b = _at(xs, (now_ts or now[0]) - minutes * 60)
    if b is None or not b[1]:
        return None
    return (now[1] / b[1] - 1) * 100.0


def _extreme_time(xs, high=True):
    """HH:MM when today's high (low) was set, from the minute samples. None when it was already
    set before the first sample (bot started late): the time is then not known."""
    i = 2 if high else 3
    vals = [x[i] for x in xs if x[i]]
    if not vals:
        return None
    best = max(vals) if high else min(vals)
    for k, x in enumerate(xs):
        if x[i] and abs(x[i] / best - 1) < 1e-9:
            if k == 0 and x[1] and abs(x[1] / best - 1) > 0.0005:
                return None
            return _hm(x[0])
    return None


def now_state(pos, off_high, off_low, l30):
    """at_high / at_low / fading / recovering / holding, from where price sits in the day range and
    which way the last 30 min went. None when the range is not known."""
    if pos is None:
        return None
    if pos >= 100 - AT_EDGE:
        return "at_high"
    if pos <= AT_EDGE:
        return "at_low"
    if l30 is not None and l30 <= -NOW_MOVE and off_high is not None and off_high <= -FADE_OFF:
        return "fading"
    if l30 is not None and l30 >= NOW_MOVE and off_low is not None and off_low >= FADE_OFF:
        return "recovering"
    return "holding"


def now_fields(ctx, name, f=None):
    """Where an index is NOW inside its day: % of range, % from the high / low (and when they were
    set), last 15 / 30 min, and the state word."""
    f = f or T.index_fields(ctx, name) or {}
    xs = _series(ctx, name)
    px, hi, lo = f.get("price"), f.get("high"), f.get("low")
    out = {"range_pos": None, "off_high": None, "off_low": None, "high_t": None, "low_t": None,
           "l15": _pct_change_window(xs, 15), "l30": _pct_change_window(xs, 30), "now": None}
    if px and hi and lo and hi > lo:
        out["range_pos"] = min(100.0, max(0.0, (px - lo) / (hi - lo) * 100.0))
        out["off_high"], out["off_low"] = (px / hi - 1) * 100.0, (px / lo - 1) * 100.0
        out["high_t"], out["low_t"] = _extreme_time(xs, True), _extreme_time(xs, False)
        out["now"] = now_state(out["range_pos"], out["off_high"], out["off_low"], out["l30"])
    return out


def _now_fact(nf):
    return "Now: %s from the day high%s · %s from the day low%s · last 30 min %s" % (
        _p(nf["off_high"]), (" (set %s)" % nf["high_t"]) if nf.get("high_t") else "", _p(nf["off_low"]),
        (" (set %s)" % nf["low_t"]) if nf.get("low_t") else "", _p(nf.get("l30")))


# ═════════════════════════════════════════════════════════════════════════════
# CHECKS. Each returns {"facts": [lines], ...numbers} or {"missing": why, "facts": []}.
# ═════════════════════════════════════════════════════════════════════════════
def check_market(ctx):
    res = _stock_results(ctx)
    if not res:
        return _missing("no live scan yet")
    st = ctx.state
    regime = ""
    if hasattr(st, "MARKET_REGIME_LOCK"):
        with st.MARKET_REGIME_LOCK:
            regime = str(getattr(st, "MARKET_REGIME", "") or "")
    adv = sum(1 for r in res.values() if (_f(r.get("intra_pct")) or 0) > 0)
    dec = sum(1 for r in res.values() if (_f(r.get("intra_pct")) or 0) < 0)
    nifty, vix = T.index_fields(ctx, "NIFTY 50") or {}, T.index_fields(ctx, "INDIA VIX") or {}
    above = T.index_above_vwap(ctx, "NIFTY 50")          # futures vs their VWAP (an index has no real VWAP)
    pts = T._ad_points(ctx) if hasattr(st, "AD_BREADTH_LOCK") else []
    ago = T._ad_ago(pts, 30, pts[-1]["ts"]) if pts else None
    trend = None
    if ago and pts:
        r0, r1 = _f(ago.get("ad_ratio")), _f(pts[-1].get("ad_ratio"))
        if r0 and r1:
            trend = "improving" if r1 > r0 * 1.1 else "weakening" if r1 < r0 / 1.1 else "flat"
    vix_pct = vix.get("pct")
    risk = "mixed"
    if trend == "improving" and (vix_pct is None or vix_pct <= 0):
        risk = "risk_on"
    elif trend == "weakening" or (vix_pct is not None and vix_pct >= 3):
        risk = "risk_off"
    line = "Market: %sNIFTY 50 %s%s · F&O %d up / %d down" % (
        ("regime %s · " % regime) if regime else "", _p(nifty.get("pct")),
        "" if above is None else (" (futures above VWAP)" if above else " (futures below VWAP)"), adv, dec)
    if vix:
        line += " · VIX %s (%s)" % (_px(vix.get("price")), _p(vix.get("pct"), 1))
    facts = [line]
    if ago:
        facts.append("Breadth: 30 min ago (%s) %s up / %s down → now %d / %d · %s" % (
            str(ago.get("time") or "?")[:5], ago.get("advance"), ago.get("decline"), adv, dec, trend or "—"))
    out_ago = (str(ago.get("time") or "")[:5], ago.get("advance"), ago.get("decline")) if ago else None
    return {"facts": facts, "regime": regime, "nifty_pct": nifty.get("pct"), "nifty_above_vwap": above,
            "ago": out_ago, "vix": vix.get("price"),
            "adv": adv, "dec": dec, "vix_pct": vix_pct, "breadth_trend": trend, "risk": risk}


def check_index(ctx, name):
    f = T.index_fields(ctx, name)
    if not f:
        return _missing("no live reading for %s" % name)
    xs = _series(ctx, name)
    first = xs[0] if xs else None
    line = "%s %s · day %s · high %s low %s" % (name, _px(f["price"]), _p(f.get("pct")), _px(f.get("high")),
                                               _px(f.get("low")))
    fut_above = T.index_above_vwap(ctx, name)            # never price vs the scanner's vwap_v: see there
    if fut_above is not None:
        line += " · futures %s their VWAP" % ("above" if fut_above else "below")
    facts = [line]
    gap = None
    if first:
        t0 = datetime.fromtimestamp(first[0])
        if (t0.hour, t0.minute) <= (9, 20) and first[5] is not None:
            gap = first[5]
            facts.append("First reading %s: %s vs yesterday (≈ the opening move)" % (t0.strftime("%H:%M"), _p(gap)))
        else:
            facts.append("Readings since %s only (bot started then): the opening move is not known" % _hm(first[0]))
    nf = now_fields(ctx, name, f)
    if nf["range_pos"] is not None:
        facts.append("Position in today's range: %.0f%% (0 = day low, 100 = day high)" % nf["range_pos"])
        facts.append(_now_fact(nf))
    return dict(nf, facts=facts, name=name, price=f["price"], pct=f.get("pct"), fut_above=fut_above,
                high=f.get("high"), low=f.get("low"),
                first_t=_hm(first[0]) if first else None, gap=gap, since=_hm(first[0]) if first else None)


def check_path(ctx, name):
    """When did it move: biggest 15-min fall / rise of the day, and the last 15 / 30 min."""
    xs = _series(ctx, name)
    if len(xs) < 5:
        return _missing("fewer than 5 minute readings of %s today" % name)
    best_dn, best_up = None, None
    for i, x in enumerate(xs):
        b = _at(xs[:i + 1], x[0] - 15 * 60)
        if b is None or b is x or not b[1]:
            continue
        ch = (x[1] / b[1] - 1) * 100.0
        if best_dn is None or ch < best_dn[0]:
            best_dn = (ch, b[0], x[0])
        if best_up is None or ch > best_up[0]:
            best_up = (ch, b[0], x[0])
    l15, l30 = _pct_change_window(xs, 15), _pct_change_window(xs, 30)
    bits = []
    if best_dn and best_dn[0] < 0:
        bits.append("biggest 15-min fall %s→%s %s" % (_hm(best_dn[1]), _hm(best_dn[2]), _p(best_dn[0])))
    if best_up and best_up[0] > 0:
        bits.append("biggest 15-min rise %s→%s %s" % (_hm(best_up[1]), _hm(best_up[2]), _p(best_up[0])))
    bits.append("last 15 min %s" % _p(l15))
    bits.append("last 30 min %s" % _p(l30))
    return {"facts": ["When (%s): %s (readings since %s)" % (_label(name), " · ".join(bits), _hm(xs[0][0]))],
            "fall15": best_dn, "rise15": best_up, "l15": l15, "l30": l30, "since": _hm(xs[0][0])}


def check_movers(ctx, direction=None, sector=None, index="NIFTY 50", n=5):
    res = _stock_results(ctx)
    if direction is None:                                  # follow the index being explained
        pct = (T.index_fields(ctx, index) or {}).get("pct")
        direction = "down" if (pct or 0) < 0 else "up"
    rows = [(s, _f(r.get("intra_pct")), _f(r.get("price")), str(T._secmap(ctx).get(s, "")).upper())
            for s, r in res.items()
            if _f(r.get("intra_pct")) is not None and (not sector or str(T._secmap(ctx).get(s, "")).upper() == sector)]
    if not rows:
        return _missing("no live stock moves")
    rows.sort(key=lambda x: x[1], reverse=(direction != "down"))
    top = rows[:n]
    word = "fallers" if direction == "down" else "gainers"
    return {"facts": ["Biggest F&O %s%s: %s" % (word, (" in " + _short(sector)) if sector else "", ", ".join(
        "%s %s %s (%s)" % (s, _px(px), _p(v), _short(sec)[:10]) for s, v, px, sec in top))],
        "rows": top}


def _sector_table(ctx, res):
    vals = {s: _f(r.get("intra_pct")) for s, r in res.items()}
    rows, _left = T._rank_sectors(ctx, vals)
    rows.sort(key=lambda x: x["avg"], reverse=True)
    return rows


def check_sectors(ctx):
    res = _stock_results(ctx)
    rows = _sector_table(ctx, res)
    if not rows:
        return _missing("no sector data")
    top, bot = rows[:3], rows[-3:][::-1]
    fmt = lambda x: "%s %s (%d/%d up)" % (_short(x["sector"]), _p(x["avg"]), x["up"], x["n"])  # noqa: E731
    facts = ["Sectors (avg of F&O stocks): strongest %s · weakest %s" % (", ".join(map(fmt, top)),
                                                                       ", ".join(map(fmt, bot)))]
    sudden = None
    for name in _sector_names(ctx):
        xs = _series(ctx, name)
        ch = _pct_change_window(xs, 15)
        if ch is not None and (sudden is None or abs(ch) > abs(sudden[1])):
            sudden = (name, ch)
    if sudden:
        facts.append("Sharpest sector index move last 15 min: %s %s" % (_short(sudden[0]), _p(sudden[1])))
    return {"facts": facts, "rows": rows, "sudden": sudden}


def check_breadth(ctx):
    res = _stock_results(ctx)
    if not res:
        return _missing("no live scan")
    up = sum(1 for r in res.values() if (_f(r.get("intra_pct")) or 0) > 0)
    n = len(res)
    out = {"up": up, "n": n, "share_up": up / float(n) if n else None}
    facts = []
    try:
        br = ctx.breadth() or {}
        if br.get("ok") and br.get("day"):
            out["hi_ev"], out["lo_ev"] = br["day"].get("hi"), br["day"].get("lo")
            facts.append("New-day-high events %s · new-day-low events %s (today, scanner passes)" % (
                br["day"].get("hi"), br["day"].get("lo")))
    except Exception:
        pass
    out["facts"] = facts
    return out


def check_broader(ctx, ref="NIFTY 50"):
    ref_f = T.index_fields(ctx, ref) or {}
    got = []
    for name in BROADER:
        f = T.index_fields(ctx, name)
        if f and f.get("pct") is not None:
            got.append((name, f["pct"]))
    if not got:
        return _missing("broader indices not fetched yet (NSE)")
    rp = ref_f.get("pct")
    facts = ["Broader: %s · %s %s" % (" · ".join("%s %s" % (n.replace("NIFTY ", ""), _p(v)) for n, v in got),
                                      ref.replace("NIFTY ", "NIFTY "), _p(rp))]
    gaps = {n: v - rp for n, v in got if rp is not None}
    if gaps:
        facts.append("Gap vs %s: %s" % (ref, " · ".join("%s %s" % (n.replace("NIFTY ", ""), _p(g)) for n, g in gaps.items())))
    return {"facts": facts, "vals": dict(got), "ref_pct": rp, "gaps": gaps}


def check_vix(ctx):
    f = T.index_fields(ctx, "INDIA VIX")
    if not f:
        return _missing("no VIX reading")
    xs = _series(ctx, "INDIA VIX")
    l15 = _pct_change_window(xs, 15)
    line = "VIX %s · day %s" % (_px(f["price"]), _p(f.get("pct"), 1))
    if l15 is not None:
        line += " · last 15 min %s" % _p(l15, 1)
    return {"facts": [line],
            "price": f["price"], "pct": f.get("pct"), "l15": l15}


def check_oi_mix(ctx, sector=None, window=30):
    types, skipped = T.oi_types(ctx, window)
    if sector:
        types = {s: x for s, x in types.items() if str(T._secmap(ctx).get(s, "")).upper() == sector}
    if not types:
        return _missing("no OI history for the last %d min" % window)
    c = {k: 0 for k in T.OI_TYPES + ("neutral",)}
    fresh = {k: 0 for k in T.OI_TYPES}
    for x in types.values():
        c[x["type"]] += 1
        if x["type"] in fresh and x["fresh"]:
            fresh[x["type"]] += 1
    n = len(types)
    return {"facts": ["OI last %d min (%s%d stocks): %s" % (window, (_short(sector) + ", ") if sector else "F&O, ", n,
                      " · ".join("%s %d" % (T.OI_TYPE_WORDS[k], c[k]) for k in T.OI_TYPES))],
            "counts": c, "fresh": fresh, "n": n}


def check_index_fut(ctx, name, window=30):
    c = T.index_futures(ctx, name, window)
    if not c:
        return _missing("no futures data for %s" % name)
    return dict(c, facts=[T.fut_line(c)])


def check_heavy(ctx, name):
    """The index's largest stocks: how many are up today and in the last 30 min (no weights)."""
    res = _stock_results(ctx)
    m30 = T._window_moves(ctx, 30)[0]
    oit = T.oi_types(ctx, 30)[0]
    rows = []
    for s in HEAVY.get(name) or []:
        pct = _f((res.get(s) or {}).get("intra_pct"))
        if pct is None:
            continue
        o = oit.get(s) or {}
        rows.append({"sym": s, "pct": pct, "m30": m30.get(s), "price": _f(res[s].get("price")),
                     "rvol": _f(res[s].get("rvol")), "oi": o.get("type"), "fresh": o.get("fresh")})
    if len(rows) < 3:
        return _missing("fewer than 3 of the largest %s stocks are in the scan" % _label(name))
    w30 = [r for r in rows if r["m30"] is not None]
    up, up30 = sum(1 for r in rows if r["pct"] > 0), sum(1 for r in w30 if r["m30"] > 0)
    best, worst = max(rows, key=lambda r: r["pct"]), min(rows, key=lambda r: r["pct"])
    line = "Largest %s stocks (fixed list, no weights): %d of %d up today" % (_label(name), up, len(rows))
    if w30:
        line += " · %d of %d up in the last 30 min" % (up30, len(w30))
    line += " · " + ", ".join("%s %s" % (r["sym"], _p(r["pct"])) for r in rows)
    return {"facts": [line], "rows": rows, "up": up, "n": len(rows), "up30": up30, "n30": len(w30),
            "best": best, "worst": worst, "name": name}


def check_crude(ctx):
    e = dict(T.EXTRA_INDICES.get("CRUDEOIL") or {})
    if not e.get("price"):
        return _missing("crude not available (MCX via Kite: see /check)")
    line = "Crude %s %s · day %s" % (e.get("contract") or "CRUDEOIL", _px(e["price"]), _p(e.get("pct")))
    if e.get("oi_pct") is not None:
        line += " · OI %s since first reading" % _p(e["oi_pct"], 1)
    facts = [line]
    res = _stock_results(ctx)
    groups = []
    for g, syms in CRUDE_SENSITIVE.items():
        have = [(s, _f(res[s].get("intra_pct"))) for s in syms if s in res and _f(res[s].get("intra_pct")) is not None]
        if have:
            groups.append((g, have))
    if groups:
        facts.append("Crude-linked stocks: %s" % " · ".join(
            "%s %s" % (g, ", ".join("%s %s" % (s, _p(v)) for s, v in have)) for g, have in groups))
    return {"facts": facts, "pct": e.get("pct"), "price": e["price"], "groups": groups, "oi_pct": e.get("oi_pct"),
            "contract": e.get("contract") or "CRUDEOIL"}


def _sector_index_name(ctx, sector):
    """The sector's own index in the minute samples (SECTOR_INTEL names match SECTOR_MAP)."""
    with T._SAMPLE_LOCK:
        names = set(T._ISAMPLES)
    if sector in names:
        return sector
    alt = {"NIFTY PSU BK": "NIFTY PSU BANK", "NIFTY PSU BANK": "NIFTY PSU BK"}.get(sector)
    return alt if alt in names else None


def _rank_history(ctx, minutes=(0, 15, 30, 45, 60)):
    """{sector index: [rank at now, -15, -30 ...]} by day % among the sector indices (1 = strongest)."""
    names = [n for n in _sector_names(ctx) if _series(ctx, n)]
    if len(names) < 3:
        return {}
    ser = {n: _series(ctx, n) for n in names}
    now = max(xs[-1][0] for xs in ser.values())
    out = {n: [] for n in names}
    for m in minutes:
        snap = {}
        for n, xs in ser.items():
            x = _at(xs, now - m * 60)
            if x is not None and x[5] is not None:
                snap[n] = x[5]
        order = sorted(snap, key=lambda k: snap[k], reverse=True)
        for n in names:
            out[n].append(order.index(n) + 1 if n in snap else None)
    return out


def _members_stats(ctx, sector, res, window=30, shared=None):
    """shared = {oit, fv, pace, m30} read once by a caller that loops over many sectors."""
    mem = _members(ctx, sector, res)
    if not mem:
        return None
    sh = shared or {}
    oit = sh["oit"] if "oit" in sh else T.oi_types(ctx, window)[0]
    fv = sh["fv"] if "fv" in sh else T._fut_volume(ctx, 15)[0]
    pace = sh["pace"] if "pace" in sh else T._live_pace(ctx)
    m30 = sh["m30"] if "m30" in sh else T._window_moves(ctx, 30)[0]
    pct = {s: _f(res[s].get("intra_pct")) for s in mem}
    above = [s for s in mem if _f(res[s].get("price")) and _f(res[s].get("vwap")) and
             _f(res[s]["price"]) >= _f(res[s]["vwap"])]
    c = {k: 0 for k in T.OI_TYPES + ("neutral",)}
    fresh = {k: 0 for k in T.OI_TYPES}
    for s in mem:
        if s in oit:
            c[oit[s]["type"]] += 1
            if oit[s]["type"] in fresh and oit[s]["fresh"]:
                fresh[oit[s]["type"]] += 1
    rows = sorted(({"sym": s, "pct": pct[s], "m30": m30.get(s), "price": _f(res[s].get("price")),
                    "rvol": _f(res[s].get("rvol")), "oi": (oit.get(s) or {}).get("type"),
                    "fresh": (oit.get(s) or {}).get("fresh")} for s in mem if pct[s] is not None),
                  key=lambda r: r["pct"], reverse=True)
    w30 = [r for r in rows if r["m30"] is not None]
    return {"rows": rows, "fresh": fresh, "up30": sum(1 for r in w30 if r["m30"] > 0), "n30": len(w30),
            "mem": mem, "n": len(mem), "up": sum(1 for v in pct.values() if (v or 0) > 0),
            "down": sum(1 for v in pct.values() if (v or 0) < 0), "avg": (sum(v for v in pct.values() if v is not None)
                                                                     / max(1, sum(1 for v in pct.values() if v is not None))),
            "above": len(above), "rvol": _median([_f(res[s].get("rvol")) for s in mem]),
            "fut": _median([fv.get(s) for s in mem]), "pace": _median([pace.get(s) for s in mem]),
            "oi": c, "oi_n": sum(c.values()), "pct": pct}


def check_sector(ctx, sector):
    res = _stock_results(ctx)
    m = _members_stats(ctx, sector, res)
    if not m:
        return _missing("no F&O stocks mapped to %s" % sector)
    order = sorted(m["mem"], key=lambda s: m["pct"].get(s) or 0, reverse=True)
    fmt = lambda s: "%s %s %s" % (s, _px(_f(res[s].get("price"))), _p(m["pct"].get(s)))  # noqa: E731
    lead, lag = order[:3], order[3:][-3:][::-1]       # leaders and laggards never overlap
    facts = ["%s stocks: %d of %d up · avg %s · leaders %s%s" % (
        _short(sector), m["up"], m["n"], _p(m["avg"]), ", ".join(fmt(s) for s in lead),
        (" · laggards " + ", ".join(fmt(s) for s in lag)) if lag else "")]
    facts.append("Participation: median RVOL %s · futures volume last 15 min vs 15 before %s (median) · "
                 "%d of %d above VWAP · median PACE used %s" % (_x(m["rvol"]), _x(m["fut"]), m["above"], m["n"],
                                                               "—" if m["pace"] is None else "%.2f" % m["pace"]))
    if m["oi_n"]:
        facts.append("OI last 30 min (%d with data): %s" % (m["oi_n"], " · ".join(
            "%s %d%s" % (T.OI_TYPE_WORDS[k], m["oi"][k], (" (%d fresh)" % m["fresh"][k]) if m["fresh"][k] else "")
            for k in T.OI_TYPES)))
    if m["n30"]:
        facts.append("Last 30 min: %d of %d stocks up" % (m["up30"], m["n30"]))
    idx = _sector_index_name(ctx, sector)
    nifty = T.index_fields(ctx, "NIFTY 50") or {}
    out = {"stats": m, "index": idx, "leaders": [(s, _f(res[s].get("price")), m["pct"].get(s)) for s in lead],
           "laggards": [(s, _f(res[s].get("price")), m["pct"].get(s)) for s in (lag or lead[::-1])]}
    rel = []
    if idx:
        f = T.index_fields(ctx, idx) or {}
        out["index_pct"] = f.get("pct")
        rel.append("%s index %s" % (_short(idx), _p(f.get("pct"))))
        nf = now_fields(ctx, idx, f)
        out.update(nf)
        if nf["range_pos"] is not None:
            facts.append("%s index at %.0f%% of today's range (0 = day low, 100 = day high)" % (
                _short(idx), nf["range_pos"]))
            facts.append(_now_fact(nf))
    rel.append("NIFTY 50 %s" % _p(nifty.get("pct")))
    out["nifty_pct"] = nifty.get("pct")
    sp = out.get("index_pct") if out.get("index_pct") is not None else m["avg"]
    if sp is not None and nifty.get("pct") is not None:
        out["gap"] = sp - nifty["pct"]
        rel.append("gap vs NIFTY 50 %s" % _p(out["gap"]))
    parent = SECTOR_PARENT.get(sector)
    if parent:
        pf = T.index_fields(ctx, parent) or {}
        out["parent"], out["parent_pct"] = parent, pf.get("pct")
        rel.append("%s %s" % (parent, _p(pf.get("pct"))))
    facts.append("Compare: " + " · ".join(rel))
    hist = _rank_history(ctx)
    if idx and idx in hist and hist[idx][0]:
        r = hist[idx]
        out["rank_now"], out["rank_of"] = r[0], len(hist)
        since = None
        xs = _series(ctx, idx)
        if xs:                                   # how long it has held this rank (or better)
            names = [n for n in hist]
            ser = {n: _series(ctx, n) for n in names}
            for x in reversed(xs):
                snap = {n: _at(s, x[0]) for n, s in ser.items()}
                snap = {n: v[5] for n, v in snap.items() if v is not None and v[5] is not None}
                if idx not in snap:
                    break
                rk = sorted(snap, key=lambda k: snap[k], reverse=True).index(idx) + 1
                if rk > r[0]:
                    break
                since = x[0]
        facts.append("Rank among sector indices: #%d of %d%s" % (r[0], len(hist), (" since %s" % _hm(since)) if since else ""))
        out["rank_since"] = _hm(since) if since else None
    out["facts"] = facts
    return out


def check_stock(ctx, sym):
    res = _stock_results(ctx)
    r = res.get(sym)
    if not r:
        return _missing("no live data for %s" % sym)
    px, pct, vw, op = _f(r.get("price")), _f(r.get("intra_pct")), _f(r.get("vwap")), _f(r.get("day_open"))
    moves = T._window_moves(ctx, 15)[0]
    fv = T._fut_volume(ctx, 15)[0]
    oit = T.oi_types(ctx, 30)[0].get(sym)
    pace = T._live_pace(ctx).get(sym)
    sec = str(r.get("sector") or T._secmap(ctx).get(sym, "")).upper()
    facts = ["%s %s · day %s · since open %s · %s VWAP (%s) · last 15 min %s" % (
        sym, _px(px), _p(pct), _p((px / op - 1) * 100.0 if px and op else None),
        "above" if px and vw and px >= vw else "below", _p((px / vw - 1) * 100.0 if px and vw else None),
        _p(moves.get(sym)))]
    bits = ["RVOL %s" % _x(_f(r.get("rvol"))), "futures volume last 15 min vs 15 before %s" % _x(fv.get(sym))]
    if pace is not None:
        bits.append("PACE used %.2f" % pace)
    if r.get("state"):
        bits.append("scanner state %s" % r.get("state"))
    facts.append(" · ".join(bits))
    if oit:
        facts.append("OI last 30 min: %s (OI %s, futures price %s%s)" % (
            T.OI_TYPE_WORDS[oit["type"]], _p(oit["oi"]), _p(oit["px"]),
            "" if oit["type"] == "neutral" else ", since %s%s" % (oit["started"], ", fresh" if oit["fresh"] else "")))
    out = {"facts": facts, "sym": sym, "pct": pct, "sector": sec, "oi_type": oit["type"] if oit else None,
           "price": px, "above_vwap": bool(px and vw and px >= vw), "m15": moves.get(sym),
           "started": oit["started"] if oit else None,
           "fresh": oit["fresh"] if oit else None, "fut": fv.get(sym), "rvol": _f(r.get("rvol"))}
    m = _members_stats(ctx, sec, res) if sec else None
    if m and m["n"] > 1:
        out["sector_avg"], out["sector_up"], out["sector_n"] = m["avg"], m["up"], m["n"]
        facts.append("Its sector %s: avg %s · %d of %d up" % (_short(sec), _p(m["avg"]), m["up"], m["n"]))
    return out


# ── strongest sectors: 7 factors, scored in code ─────────────────────────────
def _weights():
    w = {k: 1.0 for k in FACTORS}
    raw = _cfg("AI_STRENGTH_WEIGHTS")              # private: "breadth=2,oi=1.5" (config.py, never the repo)
    if raw:
        for part in str(raw).split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() in w and _f(v) is not None:
                    w[k.strip()] = max(0.0, _f(v))
    return w


def _percentile(vals):
    """{key: 0..1} rank of each value among the others (higher value = higher score)."""
    ks = [k for k, v in vals.items() if v is not None]
    if len(ks) < 2:
        return {k: 0.5 for k in ks}
    order = sorted(ks, key=lambda k: vals[k])
    return {k: i / (len(order) - 1.0) for i, k in enumerate(order)}


def check_strength(ctx, direction="up"):
    res = _stock_results(ctx)
    rows = _sector_table(ctx, res)
    if len(rows) < 2:
        return _missing("fewer than 2 sectors with data")
    sign = -1.0 if direction == "down" else 1.0
    hist = _rank_history(ctx)
    raw = {f: {} for f in FACTORS}
    info = {}
    for x in rows:
        sec = x["sector"]
        m = _members_stats(ctx, sec, res)
        if not m:
            continue
        idx = _sector_index_name(ctx, sec)
        ranks = [r for r in (hist.get(idx) or []) if r is not None] if idx else []
        n_sec = len(hist) or None
        persist = None
        if ranks and n_sec:
            avg_rank = sum(ranks) / float(len(ranks))
            persist = (n_sec + 1 - avg_rank) if sign > 0 else avg_rank
        good_oi, bad_oi = ("long_buildup", "short_covering") if sign > 0 else ("short_buildup", "long_unwinding")
        raw["day"][sec] = sign * x["avg"]
        raw["breadth"][sec] = (m["up"] if sign > 0 else m["down"]) / float(m["n"])
        raw["vwap"][sec] = (m["above"] if sign > 0 else m["n"] - m["above"]) / float(m["n"])
        raw["persist"][sec] = persist
        raw["participation"][sec] = m["rvol"]
        raw["oi"][sec] = (m["oi"][good_oi] - 0.5 * m["oi"][bad_oi]) / float(m["oi_n"]) if m["oi_n"] else None
        raw["room"][sec] = -m["pace"] if m["pace"] is not None else None
        info[sec] = {"m": m, "ranks": ranks, "n_sec": n_sec, "good_oi": good_oi, "bad_oi": bad_oi}
    pct = {f: _percentile(raw[f]) for f in FACTORS}
    w = _weights()
    scores = {}
    for sec in info:
        num = sum(w[f] * pct[f][sec] for f in FACTORS if sec in pct[f])
        den = sum(w[f] for f in FACTORS if sec in pct[f])
        scores[sec] = num / den if den else None
    ranked = sorted((s for s in scores if scores[s] is not None), key=lambda s: scores[s], reverse=True)
    day_rank = {x["sector"]: i + 1 for i, x in enumerate(sorted(rows, key=lambda x: sign * x["avg"], reverse=True))}
    cands, facts = [], []
    for i, sec in enumerate(ranked[:3]):
        m, inf_ = info[sec]["m"], info[sec]
        strong = [f for f in FACTORS if sec in pct[f] and pct[f][sec] >= 0.66]
        avail = [f for f in FACTORS if sec in pct[f]]
        top3 = sum(1 for r in inf_["ranks"] if r <= 3)
        line = "%d %s score %.2f · day %s (day-rank %d) · %d/%d %s · %d/%d %s VWAP" % (
            i + 1, _short(sec), scores[sec], _p(m["avg"]), day_rank.get(sec, 0), m["up"] if sign > 0 else m["down"],
            m["n"], "up" if sign > 0 else "down", m["above"] if sign > 0 else m["n"] - m["above"], m["n"],
            "above" if sign > 0 else "below")
        if inf_["ranks"]:
            line += " · top-3 in %d of %d checks (last 60 min)" % (top3, len(inf_["ranks"]))
        line += " · RVOL %s" % _x(m["rvol"])
        if m["oi_n"]:
            line += " · %s %d/%d, %s %d/%d" % (T.OI_TYPE_WORDS[inf_["good_oi"]], m["oi"][inf_["good_oi"]], m["oi_n"],
                                              T.OI_TYPE_WORDS[inf_["bad_oi"]], m["oi"][inf_["bad_oi"]], m["oi_n"])
        if m["pace"] is not None:
            line += " · PACE used %.2f" % m["pace"]
        facts.append(line)
        cands.append({"name": _short(sec), "sector": sec, "score": scores[sec], "day_rank": day_rank.get(sec),
                      "strong": strong, "avail": avail, "avg": m["avg"], "n": m["n"],
                      "moved": m["up"] if sign > 0 else m["down"], "top3": top3, "checks": len(inf_["ranks"]),
                      "good_oi": (inf_["good_oi"], m["oi"][inf_["good_oi"]], m["oi_n"]),
                      "bad_oi": (inf_["bad_oi"], m["oi"][inf_["bad_oi"]], m["oi_n"]), "rvol": m["rvol"]})
    agree = None
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        common = [f for f in FACTORS if a in pct[f] and b in pct[f]]
        k = sum(1 for f in common if pct[f][a] > pct[f][b])
        agree = (k, len(common))
    facts.insert(0, "%s sectors (7 factors scored by code: %s; weights %s)" % (
        "Strongest" if sign > 0 else "Weakest", ", ".join(FACTOR_WORDS[f] for f in FACTORS),
        "equal" if len(set(w.values())) == 1 else "yours from config"))
    if agree:
        facts.append("#1 vs #2: #1 better on %d of %d factors" % agree)
    return {"facts": facts, "candidates": cands, "agree": agree, "scores": scores, "pct": pct,
            "day_rank": day_rank, "direction": direction}


def check_rotation(ctx, window=30):
    res = _stock_results(ctx)
    hist = _rank_history(ctx, minutes=(0, window, 60))
    if not hist:
        return _missing("fewer than 3 sector indices sampled today")
    fv = T._fut_volume(ctx, 15)[0]
    oit = T.oi_types(ctx, window)[0]
    rows = []
    for idx, ranks in hist.items():
        xs = _series(ctx, idx)
        ch = _pct_change_window(xs, window)
        mem = _members(ctx, idx, res)
        lb = sum(1 for s in mem if (oit.get(s) or {}).get("type") == "long_buildup")
        oi_n = sum(1 for s in mem if s in oit)
        fut = _median([fv.get(s) for s in mem])
        up = sum(1 for s in mem if (_f(res[s].get("intra_pct")) or 0) > 0)
        now_r, then_r, hour_r = ranks
        day_pct = xs[-1][5] if xs else None
        lab = None
        if now_r and hour_r and now_r <= 3 and hour_r <= 3 and (ch or 0) >= 0:
            lab = "STEADY_LEADER"
        if now_r and then_r and then_r - now_r >= 3 and now_r <= 5 and (ch or 0) > 0 and \
                ((fut or 0) >= 1.3 or (oi_n and lb / float(oi_n) >= 0.3)):
            lab = "EMERGING"
        if hour_r and hour_r <= 3 and ((ch or 0) < 0 or (now_r and now_r - hour_r >= 3)):
            lab = "FADING"
        rows.append({"index": idx, "name": _short(idx), "rank": now_r, "rank_then": then_r, "rank_hour": hour_r,
                     "change": ch, "day_pct": day_pct, "fut": fut, "lb": lb, "oi_n": oi_n, "up": up,
                     "n": len(mem), "label": lab})
    rows.sort(key=lambda r: (r["rank"] or 99))
    facts = ["Day leader vs last %d min (sector indices, rank 1 = strongest):" % window]
    for r in rows:
        if r["label"] or (r["rank"] or 99) <= 3:
            facts.append("%s day %s · rank %s → %s (60 min ago %s) · last %d min %s · futures volume 15 min %s · "
                         "long build-up %d/%d · %d/%d up%s" % (
                             r["name"], _p(r["day_pct"]), r["rank_then"] or "—", r["rank"] or "—",
                             r["rank_hour"] or "—", window, _p(r["change"]), _x(r["fut"]), r["lb"], r["oi_n"],
                             r["up"], r["n"], (" · " + r["label"]) if r["label"] else ""))
    em = [r for r in rows if r["label"] == "EMERGING"]
    em.sort(key=lambda r: ((r["change"] or 0), r["fut"] or 0), reverse=True)
    stocks = []
    if em:
        best = em[0]
        mem = _members(ctx, best["index"], res)
        pace = T._live_pace(ctx)
        cand = [s for s in mem if (oit.get(s) or {}).get("type") == "long_buildup"] or mem
        cand.sort(key=lambda s: fv.get(s) or 0, reverse=True)
        for s in cand[:3]:
            r = res[s]
            px, vw = _f(r.get("price")), _f(r.get("vwap"))
            o = oit.get(s) or {}
            stocks.append((s, px, _f(r.get("intra_pct")), fv.get(s)))
            facts.append("  %s %s day %s · futures volume 15 min %s · OI %s%s · %s VWAP%s" % (
                s, _px(px), _p(_f(r.get("intra_pct"))), _x(fv.get(s)), _p(o.get("oi")),
                (" " + T.OI_TYPE_WORDS[o["type"]]) if o.get("type") else "",
                "above" if px and vw and px >= vw else "below",
                (" · PACE used %.2f" % pace[s]) if pace.get(s) is not None else ""))
    return {"facts": facts, "rows": rows, "emerging": [r["name"] for r in em],
            "steady": [r["name"] for r in rows if r["label"] == "STEADY_LEADER"],
            "fading": [r["name"] for r in rows if r["label"] == "FADING"], "stocks": stocks}


CHECKS = {"market": check_market, "index": check_index, "path": check_path, "movers": check_movers,
          "sectors": check_sectors, "breadth": check_breadth, "broader": check_broader, "vix": check_vix,
          "oi_mix": check_oi_mix, "crude": check_crude, "sector": check_sector, "stock": check_stock,
          "strength": check_strength, "rotation": check_rotation, "index_fut": check_index_fut,
          "heavy": check_heavy}


# ═════════════════════════════════════════════════════════════════════════════
# PLAN: which checks each question kind needs (the agent's map; also its trajectory test)
# ═════════════════════════════════════════════════════════════════════════════
def plan(kind, target=None, compare=None, direction=None, window_min=30):
    """[(check id, check name, kwargs)] — ids are unique so a plan may use one check twice."""
    return _drop_no_fut(_plan(kind, target, compare, direction, window_min), target)


def _plan(kind, target=None, compare=None, direction=None, window_min=30):
    if kind == "index_move":
        p = [("market", "market", {}), ("index", "index", {"name": target}), ("path", "path", {"name": target}),
             ("index_fut", "index_fut", {"name": target, "window": window_min}),
             ("heavy", "heavy", {"name": target}),
             ("sectors", "sectors", {}), ("breadth", "breadth", {}), ("vix", "vix", {}),
             ("oi_mix", "oi_mix", {"window": window_min}), ("crude", "crude", {})]
        if target in BROADER or compare:
            p.append(("broader", "broader", {"ref": compare or "NIFTY 50"}))
        else:
            p.append(("broader", "broader", {}))
            p.append(("movers", "movers", {"direction": None, "index": target}))     # the data's side, not the question's
        if compare and compare != "NIFTY 50":
            p.append(("index_cmp", "index", {"name": compare}))
        return p
    if kind == "sector_move":
        return [("market", "market", {}), ("sector", "sector", {"sector": target}),
                ("sector_path", "path", {"name": target}), ("sectors", "sectors", {}), ("vix", "vix", {})]
    if kind == "strongest_sectors":
        return [("market", "market", {}), ("strength", "strength", {"direction": direction or "up"}),
                ("broader", "broader", {}), ("vix", "vix", {})]
    if kind == "rotation":
        return [("market", "market", {}), ("rotation", "rotation", {"window": window_min}), ("vix", "vix", {})]
    if kind == "stock_move":
        return [("market", "market", {}), ("stock", "stock", {"sym": target}), ("sectors", "sectors", {})]
    if kind == "vix":
        return [("market", "market", {}), ("vix", "vix", {}), ("index", "index", {"name": "NIFTY 50"}),
                ("index_fut", "index_fut", {"name": "NIFTY 50", "window": window_min}),
                ("path", "path", {"name": "INDIA VIX"}), ("breadth", "breadth", {}),
                ("oi_mix", "oi_mix", {"window": window_min})]
    if kind == "crude":
        return [("market", "market", {}), ("crude", "crude", {}), ("sectors", "sectors", {}),
                ("index", "index", {"name": "NIFTY 50"})]
    raise ValueError("unknown kind %s" % kind)


def _drop_no_fut(p, target):
    """Checks that cannot apply to this index are left out of the plan (not reported as 'missing')."""
    return [x for x in p if not (x[1] == "index_fut" and x[2].get("name") not in T.INDEX_FUT)
            and not (x[1] == "heavy" and x[2].get("name") not in HEAVY)]


def run_check(ctx, name, kwargs):
    """One check, never raises (a failed check is reported as missing, the rest still answer)."""
    try:
        return CHECKS[name](ctx, **kwargs)
    except T.ToolError as e:
        return _missing(str(e))
    except Exception as e:
        return _missing("%s: %s" % (type(e).__name__, e))


# ═════════════════════════════════════════════════════════════════════════════
# ANALYZE: pattern labels + candidates + confidence, by rule (testable, repeatable)
# ═════════════════════════════════════════════════════════════════════════════
LABEL_TEXT = {
    "GAP_LED": "most of the move came at the open", "INTRADAY": "the move built during the day",
    "SUDDEN": "a big part came in one 15-min window", "BROAD": "most stocks moved the same way",
    "NARROW": "breadth is mixed: fewer names carry the move", "SECTOR_LED": "one or two sectors moved much more",
    "FEAR_RISING": "VIX rising", "FEAR_EASING": "VIX falling", "SHORTS_ADDING": "short build-up dominates OI",
    "LONGS_ADDING": "long build-up dominates OI", "LONGS_EXITING": "long unwinding dominates OI",
    "SHORTS_COVERING": "short covering dominates OI", "SMALL_WEAKER": "smaller caps weaker than NIFTY",
    "SMALL_STRONGER": "smaller caps stronger than NIFTY", "SECTOR_SPECIFIC": "sector moved apart from the market",
    "WITH_MARKET": "sector moved with the market", "STOCK_SPECIFIC": "stock moved apart from its sector",
    "WITH_SECTOR": "stock moved with its sector", "FRESH_LONGS": "fresh long build-up",
    "FRESH_SHORTS": "fresh short build-up", "ROTATION_TO_EMERGING": "new money moving to an emerging sector",
    "CRUDE_BIG_MOVE": "crude moved more than 1.5%",
    "SECTOR_SPLIT": "strong and weak sectors cancel out in the index",
    "FADING": "off the day high and falling in the last 30 min",
    "RECOVERING": "off the day low and rising in the last 30 min",
    "AT_DAY_HIGH": "at the day high", "AT_DAY_LOW": "at the day low",
    "HEAVY_AGAINST": "most of the largest stocks moved the other way in the last 30 min",
    "FUT_LONG_BUILDUP": "index futures: long build-up", "FUT_SHORT_BUILDUP": "index futures: short build-up",
    "FUT_SHORT_COVERING": "index futures: short covering", "FUT_LONG_UNWINDING": "index futures: long unwinding",
}


NOW_LABEL = {"fading": "FADING", "recovering": "RECOVERING", "at_high": "AT_DAY_HIGH", "at_low": "AT_DAY_LOW"}


def _oi_label(c, direction):
    tot = sum(c.get(k, 0) for k in T.OI_TYPES + ("neutral",))      # flat OI counts in the base: 1 of 4 is not "dominates"
    top = max(T.OI_TYPES, key=lambda k: c.get(k, 0))
    if not tot or c.get(top, 0) / float(tot) < 0.35:
        return None
    return {"short_buildup": "SHORTS_ADDING", "long_buildup": "LONGS_ADDING", "long_unwinding": "LONGS_EXITING",
            "short_covering": "SHORTS_COVERING"}[top]


def analyze(kind, args, ev):
    """(labels, candidates, confidence, lean, direction). Only rules over the evidence numbers."""
    labels, cands, conf, lean = [], [], None, None
    g = lambda k: ev.get(k) or {}  # noqa: E731
    direction = args.get("direction")
    if kind == "vix":
        vx = g("vix")
        if vx.get("pct") is not None:
            direction = "up" if vx["pct"] >= 0 else "down"
        if vx.get("pct") is not None:
            labels.append("FEAR_RISING" if vx["pct"] >= 3 else "FEAR_EASING" if vx["pct"] <= -3 else None)
        path = g("path")
        mv = (path.get("rise15") if direction == "up" else path.get("fall15")) or None
        if vx.get("pct") and mv and abs(mv[0]) >= 0.5 * abs(vx["pct"]) and abs(vx["pct"]) >= 1:
            labels.append("SUDDEN")
        nifty_dir = "up" if (g("index").get("pct") or 0) >= 0 else "down"
        br = g("breadth")
        if br.get("share_up") is not None:
            share = br["share_up"] if nifty_dir == "up" else 1 - br["share_up"]
            labels.append("BROAD" if share >= 0.7 else "NARROW" if share <= 0.55 else None)
        if g("oi_mix").get("counts"):
            labels.append(_oi_label(g("oi_mix")["counts"], nifty_dir))
    elif kind in ("index_move", "crude"):
        idx = g("index")
        if idx.get("pct") is not None:
            direction = "up" if idx["pct"] >= 0 else "down"
        pct = idx.get("pct")
        if pct and idx.get("gap") is not None and abs(pct) >= 0.15:
            labels.append("GAP_LED" if (idx["gap"] * pct > 0 and abs(idx["gap"]) >= 0.6 * abs(pct)) else "INTRADAY")
        path = g("path")
        mv = (path.get("fall15") if direction == "down" else path.get("rise15")) or None
        if pct and mv and abs(mv[0]) >= 0.5 * abs(pct) and abs(pct) >= 0.15:
            labels.append("SUDDEN")
        br = g("breadth")
        if br.get("share_up") is not None:
            share = br["share_up"] if direction == "up" else 1 - br["share_up"]
            labels.append("BROAD" if share >= 0.7 else "NARROW" if share <= 0.55 else None)
        sec = g("sectors")
        rows_ = sec.get("rows") or []
        if rows_ and pct is not None and abs(pct) < SPLIT_INDEX_MAX and rows_[0]["avg"] >= SPLIT_UP \
                and rows_[-1]["avg"] <= -SPLIT_DOWN:
            labels.append("SECTOR_SPLIT")                  # a flat index hiding a big split
        elif rows_ and pct:
            big = [x for x in rows_ if x["avg"] * pct > 0 and abs(x["avg"]) >= 2 * abs(pct)]
            if 1 <= len(big) <= 2:
                labels.append("SECTOR_LED")
        vx = g("vix")
        if vx.get("pct") is not None:
            labels.append("FEAR_RISING" if vx["pct"] >= 3 else "FEAR_EASING" if vx["pct"] <= -3 else None)
        oi = g("oi_mix")
        if oi.get("counts"):
            labels.append(_oi_label(oi["counts"], direction))
        labels.append(NOW_LABEL.get(idx.get("now")))
        fx = g("index_fut")
        if fx.get("type") in T.OI_TYPES and abs(fx.get("oi_w") or 0) >= 1.0:
            labels.append("FUT_" + fx["type"].upper())
        hv = g("heavy")
        if hv.get("n30") and direction in ("up", "down"):
            with_day = hv["up30"] if direction == "up" else hv["n30"] - hv["up30"]
            if with_day / float(hv["n30"]) <= 0.3:
                labels.append("HEAVY_AGAINST")
        bd = g("broader")
        if bd.get("gaps"):
            avg_gap = sum(bd["gaps"].values()) / len(bd["gaps"])
            labels.append("SMALL_WEAKER" if avg_gap <= -0.5 else "SMALL_STRONGER" if avg_gap >= 0.5 else None)
        cr = g("crude")
        if cr.get("pct") is not None and abs(cr["pct"]) >= 1.5:
            labels.append("CRUDE_BIG_MOVE")
    elif kind == "sector_move":
        s = g("sector")
        st = s.get("stats") or {}
        sp = s.get("index_pct", st.get("avg"))
        if sp is not None:
            direction = "up" if sp >= 0 else "down"
        if st.get("n"):
            share = (st["up"] if direction == "up" else st["down"]) / float(st["n"])
            labels.append("BROAD" if share >= 0.7 else "NARROW" if share <= 0.55 else None)
        ip, np_ = s.get("index_pct", st.get("avg")), s.get("nifty_pct")
        if ip is not None and np_ is not None:
            labels.append("SECTOR_SPECIFIC" if abs(ip - np_) >= 1.0 else "WITH_MARKET")
        if st.get("oi"):
            labels.append(_oi_label(st["oi"], direction))
        labels.append(NOW_LABEL.get(s.get("now")))
        path = g("sector_path")
        mv = (path.get("rise15") if direction == "up" else path.get("fall15")) or None
        if ip and mv and abs(mv[0]) >= 0.5 * abs(ip) and abs(ip) >= 0.3:
            labels.append("SUDDEN")
        vx = g("vix")
        if vx.get("pct") is not None:
            labels.append("FEAR_RISING" if vx["pct"] >= 3 else "FEAR_EASING" if vx["pct"] <= -3 else None)
    elif kind == "stock_move":
        s = g("stock")
        if s.get("pct") is not None:
            direction = "up" if s["pct"] >= 0 else "down"
        if s.get("pct") is not None and s.get("sector_avg") is not None:
            labels.append("STOCK_SPECIFIC" if abs(s["pct"] - s["sector_avg"]) >= 1.0 else "WITH_SECTOR")
        if s.get("oi_type") == "long_buildup" and s.get("fresh"):
            labels.append("FRESH_LONGS")
        if s.get("oi_type") == "short_buildup" and s.get("fresh"):
            labels.append("FRESH_SHORTS")
    elif kind == "strongest_sectors":
        st = g("strength")
        cands = st.get("candidates") or []
        if cands:
            lean = cands[0]["name"]
            k, n = st.get("agree") or (0, 0)
            conf = _conf(k, n)
    elif kind == "rotation":
        ro, mk = g("rotation"), g("market")
        risk = mk.get("risk", "mixed")
        rows = {r["name"]: r for r in ro.get("rows") or []}
        em, steady = ro.get("emerging") or [], ro.get("steady") or []
        if em:
            labels.append("ROTATION_TO_EMERGING")
        cands = [{"name": n, "label": "EMERGING"} for n in em[:2]] + [{"name": n, "label": "STEADY_LEADER"}
                                                                       for n in steady[:2]]
        if em or steady:
            # context decides between fresh and proven: risk-on favours fresh momentum,
            # risk-off favours the proven leader; mixed -> the stronger evidence
            if em and (risk == "risk_on" or not steady):
                lean = em[0]
            elif steady and (risk == "risk_off" or not em):
                lean = steady[0]
            else:
                e, s_ = rows[em[0]], rows[steady[0]]
                lean = em[0] if (e["change"] or 0) > (s_["change"] or 0) else steady[0]
            r = rows[lean]
            checks = [(r["change"] or 0) > 0, (r["fut"] or 0) >= 1.3,
                      bool(r["oi_n"]) and r["lb"] / float(r["oi_n"]) >= 0.3,
                      bool(r["n"]) and r["up"] / float(r["n"]) >= 0.6,
                      (risk == "risk_on") if rows[lean]["label"] == "EMERGING" else (risk != "risk_on")]
            conf = _conf(sum(checks), len(checks))
    labels = [x for x in labels if x]
    return labels, cands, conf, lean, direction


def _conf(k, n):
    if not n:
        return None
    r = k / float(n)
    return {"level": "HIGH" if r >= 0.75 else "MEDIUM" if r >= 0.55 else "LOW", "k": k, "n": n}


def not_checked(kind, target, ev):
    out = list(ALWAYS_NOT_CHECKED)
    if kind == "index_move" and target in ("NIFTY 50", "NIFTY BANK"):
        out.append("index point contribution (no index weights)")
    if kind == "index_move" and target in BROADER:
        out.append("non-F&O members of %s" % target)
    for k, v in ev.items():
        if v.get("missing"):
            out.append("%s (%s)" % (k.replace("_", " "), v["missing"]))
    return out


def _main_pct(kind, ev):
    g = lambda k: ev.get(k) or {}  # noqa: E731
    return {"index_move": g("index").get("pct"), "crude": g("crude").get("pct"), "vix": g("vix").get("pct"),
            "sector_move": (g("sector").get("index_pct") if g("sector").get("index_pct") is not None
                            else (g("sector").get("stats") or {}).get("avg")),
            "stock_move": g("stock").get("pct")}.get(kind)


def headline(kind, args, direction, as_of, flat=False):
    t, w = args.get("target"), args.get("window_min") or 30
    verb = "flat" if flat else {"up": "up", "down": "down"}.get(direction or "", "moving")
    h = {"index_move": "Why %s is %s" % (t, verb), "sector_move": "Why %s is %s" % (_short(t or ""), verb),
         "strongest_sectors": "%s sectors now, with evidence" % ("Weakest" if direction == "down" else "Strongest"),
         "rotation": "Day leader vs last %d min" % w, "stock_move": "Why %s is %s" % (t, verb),
         "vix": "Why VIX is %s" % verb, "crude": "Crude and the market"}[kind]
    if args.get("compare"):
        h += " vs %s" % args["compare"]
    return "%s · %s" % (h, str(as_of or datetime.now().strftime("%H:%M"))[:5])


# ── BRIEF: the 3-5 lines a trader reads during the market (details are one tap away) ──
def _arrow(pct, flat):
    return "➖" if flat or pct is None else ("📈" if pct >= 0 else "📉")


def _oi_phrase(c):
    if not c:
        return None
    pairs = sorted(((c.get(k, 0), k) for k in T.OI_TYPES), key=lambda x: (-x[0], T.OI_TYPES.index(x[1])))
    (n1, k1), (n2, k2) = pairs[0], pairs[1]
    if not n1:
        return None
    return "OI: %s %d vs %s %d" % (T.OI_TYPE_WORDS[k1], n1, T.OI_TYPE_WORDS[k2], n2)


FUT_READ = {"long_buildup": "futures adding longs", "short_buildup": "futures adding shorts",
            "short_covering": "futures shorts covering", "long_unwinding": "futures longs exiting"}


def _fut_short(c):
    """'Fut NIFTY: short build-up since 12:06 (fresh) · 30m OI +1.20%, price -0.25% · day OI -1.04% · DTE 12'.
    The 30-min numbers and the since-open number are named, so they never read as a contradiction."""
    bits = []
    w = c.get("window") or 30
    if c.get("type") and c["type"] != "neutral":
        bits.append("%s since %s%s" % (T.OI_TYPE_WORDS[c["type"]], c["started"], " (fresh)" if c["fresh"] else ""))
        bits.append("%dm OI %s, price %s" % (w, _p(c.get("oi_w")), _p(c.get("px_w"))))
    elif c.get("type") == "neutral":
        bits.append("%dm OI flat" % w)
    if c.get("oi_open") is not None:
        bits.append("day OI %s" % _p(c["oi_open"]))
    if c.get("iv") is not None:
        bits.append("IV %.1f" % c["iv"])
    if c.get("dte") is not None:
        bits.append("expiry day" if c["dte"] == 0 else "DTE %d" % c["dte"])
    return "Fut %s: %s" % (c.get("sym") or "", " · ".join(bits))


def _read_day_now(word, nf):
    """'Up on the day but fading from the high of 11:42': the day's direction against what price does NOW."""
    now, pos = nf.get("now"), nf.get("range_pos")
    if now is None or pos is None:
        return None
    side = {"fading": "down", "at_low": "down", "recovering": "up", "at_high": "up"}.get(now)
    if now == "fading":
        ph = "fading from the high" + ((" of %s" % nf["high_t"]) if nf.get("high_t") else "")
    elif now == "recovering":
        ph = "recovering from the low" + ((" of %s" % nf["low_t"]) if nf.get("low_t") else "")
    elif now in ("at_high", "at_low"):
        ph = "at the day %s" % now[3:]
    else:
        ph = "low in the day range" if pos < 35 else "high in the day range" if pos > 65 else "mid-range"
        side = "down" if pos < 35 else "up" if pos > 65 else None
    if word == "flat":
        return "Flat on the day, " + ph
    return "%s on the day%s%s" % (word.capitalize(), " but " if side and side != word else " and " if side else ", ", ph)


def _range_words(pos):
    """'4% of low→high range': 0 = today's low, 100 = today's high (live 11-11: '4% of day range' was not
    understood)."""
    return "%.0f%% of low→high range" % pos


def _now_line(nf, fut_above=None, above=None, n=None):
    """'Now: 19% of low→high range · -0.42% from high (11:42) · last 30m -0.21% · futures below VWAP'."""
    if nf.get("range_pos") is None:
        return None
    bits = [_range_words(nf["range_pos"])]
    if nf.get("off_high") is not None and (nf["range_pos"] < 50 or nf.get("now") == "fading"):
        bits.append("%s from high%s" % (_p(nf["off_high"]), (" (%s)" % nf["high_t"]) if nf.get("high_t") else ""))
    elif nf.get("off_low") is not None:
        bits.append("%s from low%s" % (_p(nf["off_low"]), (" (%s)" % nf["low_t"]) if nf.get("low_t") else ""))
    if nf.get("l30") is not None:
        bits.append("last 30m %s" % _p(nf["l30"]))
    if fut_above is not None:
        bits.append("futures above VWAP" if fut_above else "futures below VWAP")
    if above is not None and n:
        bits.append("%d/%d above VWAP" % (above, n))
    return "Now: " + " · ".join(bits)


def _oi_code(r):
    return (OI_CODE.get(r.get("oi")) or "—") + ("•" if r.get("fresh") and r.get("oi") in OI_CODE else "")


def stock_table(rows, cap=TABLE_ROWS):
    """A 32-character grid for a phone (shown monospace): one line per stock, sorted by day %."""
    def num(v, fmt):
        if v is None:
            return "—"
        return fmt % (0.0 if abs(v) < 0.005 else v)
    rows = list(rows)
    if len(rows) > cap:
        rows = rows[:cap // 2] + [None] + rows[-(cap - cap // 2 - 1):]
    out = ["%-10s %6s %5s %-3s %5s" % ("STOCK", "DAY%", "30m%", "OI", "RVOL")]
    for r in rows:
        out.append("…" if r is None else "%-10s %6s %5s %-3s %5s" % (
            r["sym"][:10], num(r["pct"], "%+.2f"), num(r.get("m30"), "%+.1f"), _oi_code(r),
            "—" if r.get("rvol") is None else "%.1fx" % r["rvol"]))
    out += ["LB long build-up  SB short build-up", "SC short covering LU long unwinding", "• fresh (new in the last 30 min)"]
    return out


def brief(kind, args, ev, labels, lean, conf, direction, flat):
    """[headline line, driver lines...]. Plain text; the renderer bolds line 1. Only evidence numbers."""
    g = lambda k: ev.get(k) or {}  # noqa: E731
    L = []
    mk, conf_txt = g("market"), ("confidence %s (%d/%d)" % (conf["level"], conf["k"], conf["n"]) if conf else "")
    breadth = ("Breadth: %d↑/%d↓%s" % (mk["adv"], mk["dec"], (" " + mk["breadth_trend"]) if mk.get("breadth_trend")
                                     and mk["breadth_trend"] != "flat" else "")) if mk.get("adv") is not None else None
    vixb = "VIX %s" % _p(g("vix").get("pct", mk.get("vix_pct")), 1) if (g("vix").get("pct", mk.get("vix_pct"))
                                                                      is not None) else None
    if kind == "index_move":
        ix = g("index")
        if not ix.get("price"):
            return ["%s: no live reading yet" % args.get("target")]
        word = "flat" if flat else ("up" if (ix.get("pct") or 0) >= 0 else "down")
        L.append("%s %s %s %s · %s" % (_arrow(ix.get("pct"), flat), _label(ix["name"]), word, _p(ix.get("pct")),
                                       _px(ix["price"])))
        fx, hv = g("index_fut"), g("heavy")
        # the read: day direction vs now, what futures do, breadth turning. Built by code from the checks.
        read = [_read_day_now(word, ix)]
        if fx.get("type") in FUT_READ:
            read.append("%s since %s%s" % (FUT_READ[fx["type"]], fx["started"], ", fresh" if fx["fresh"] else ""))
        if mk.get("breadth_trend") in ("weakening", "improving"):
            read.append("breadth " + mk["breadth_trend"])
        read = [x for x in read if x]
        if read:
            L.append("⚡ " + " · ".join(read))
        L.append(_now_line(ix, fut_above=ix.get("fut_above")))
        if fx.get("sym") and (fx.get("type") or fx.get("oi_open") is not None):
            L.append(_fut_short(fx))
        if hv.get("n"):
            line = "Heavy: %d/%d up" % (hv["up"], hv["n"])
            if hv.get("n30"):
                line += " · last 30m %d/%d up" % (hv["up30"], hv["n30"])
            line += " · ▲ %s %s" % (hv["best"]["sym"], _p(hv["best"]["pct"]))
            if hv["worst"]["sym"] != hv["best"]["sym"]:
                line += " ▼ %s %s" % (hv["worst"]["sym"], _p(hv["worst"]["pct"]))
            L.append(line)
        rows = g("sectors").get("rows") or []
        mv = g("movers").get("rows") or []
        fmt = lambda x: "%s %s (%d/%d up)" % (_short(x["sector"]), _p(x["avg"]), x["up"], x["n"])  # noqa: E731
        if "SECTOR_SPLIT" in labels and rows:      # a flat index hiding a split: that IS the story -> line 2
            L.insert(1, "Sectors: split underneath, %s vs %s" % (fmt(rows[0]), fmt(rows[-1])))
        elif rows:
            lead, other = (rows[0], rows[-1]) if direction != "down" else (rows[-1], rows[0])
            line = "Sectors: %s %s" % ("led by" if direction != "down" else "dragged by", fmt(lead))
            own = [x for x in mv if x[3] == lead["sector"]]          # names from THAT sector only
            if own:
                line += " · " + ", ".join("%s %s" % (x[0], _p(x[1])) for x in own[:2])
            if other is not lead:
                line += " · %s %s" % ("weakest" if direction != "down" else "strongest", fmt(other))
            L.append(line)
        oi = _oi_phrase(g("oi_mix").get("counts"))
        L.append(" · ".join(x for x in (breadth, vixb, ("Stock " + oi) if oi else None) if x))
        extra = []
        path = g("path")
        if "GAP_LED" in labels and ix.get("gap") is not None:
            extra.append("Most of it at the open (%s)" % _p(ix["gap"]))
        elif "SUDDEN" in labels:
            mvp = path.get("fall15") if direction == "down" else path.get("rise15")
            if mvp:
                extra.append("Main move %s–%s (%s)" % (_hm(mvp[1]), _hm(mvp[2]), _p(mvp[0])))
        bd = g("broader")
        if bd.get("vals") and ({"SMALL_WEAKER", "SMALL_STRONGER"} & set(labels) or args.get("compare")):
            v = bd["vals"]
            extra.append("Smaller caps: " + " · ".join("%s %s" % (n.replace("NIFTY ", ""), _p(x))
                                                      for n, x in list(v.items())[:3]))
        if "CRUDE_BIG_MOVE" in labels:
            extra.append("Crude %s" % _p(g("crude").get("pct")))
        L += extra[:2]
    elif kind == "sector_move":
        sx = g("sector")
        st = sx.get("stats") or {}
        if not st:
            return ["%s: no live data yet" % _short(args.get("target") or "")]
        pct = sx.get("index_pct", st.get("avg"))
        word = "flat" if flat else ("up" if (pct or 0) >= 0 else "down")
        h = "%s %s %s %s · %d/%d up" % (_arrow(pct, flat), _short(args["target"]), word, _p(pct), st["up"], st["n"])
        if sx.get("rank_now"):
            h += " · rank #%d of %d" % (sx["rank_now"], sx["rank_of"])
        L.append(h)
        read = [_read_day_now(word, sx)]
        if st.get("oi_n", 0) >= 2:
            top = max(T.OI_TYPES, key=lambda k: st["oi"][k])
            if st["oi"][top] / float(st["oi_n"]) >= 0.5:
                read.append("%s in %d/%d stocks" % (T.OI_TYPE_WORDS[top], st["oi"][top], st["oi_n"]))
        if sx.get("gap") is not None:
            if abs(sx["gap"]) >= 1.0:
                read.append("%s than NIFTY by %s" % ("stronger" if sx["gap"] > 0 else "weaker", _p(sx["gap"])))
            elif abs(sx["gap"]) < 0.3:
                read.append("in line with NIFTY")
        read = [x for x in read if x]
        if read:
            L.append("⚡ " + " · ".join(read))
        L.append(_now_line(sx, above=st.get("above"), n=st["n"]))
        fs = lambda rows_: ", ".join("%s %s %s" % (s, _px(px), _p(v)) for s, px, v in rows_)  # noqa: E731
        lead, lag = sx.get("leaders") or [], sx.get("laggards") or []
        if st["n"] <= 3:                                       # too few members to split leaders / laggards
            L.append("Stocks: " + fs(lead))
        else:
            L.append("Stocks: ▲ %s%s" % (fs(lead[:2]), (" ▼ " + fs(lag[:2])) if lag and lag != lead else ""))
        if st.get("oi_n"):
            bits = ["%s %d/%d%s" % (T.OI_TYPE_WORDS[k], st["oi"][k], st["oi_n"],
                                    (" (%d fresh)" % st["fresh"][k]) if st["fresh"][k] else "")
                    for k in sorted(T.OI_TYPES, key=lambda k: -st["oi"][k]) if st["oi"][k]]
            if bits:
                L.append("OI 30m: " + " · ".join(bits[:2]))
        bits = ["RVOL %s" % _x(st.get("rvol"))]
        if st.get("fut") is not None:
            bits.append("fut vol 15m %s" % _x(st["fut"]))
        path = g("sector_path")
        mvp = path.get("fall15") if direction == "down" else path.get("rise15")
        if mvp and abs(mvp[0]) > 0:
            bits.append("biggest push %s–%s (%s)" % (_hm(mvp[1]), _hm(mvp[2]), _p(mvp[0])))
        L.append("Volume: " + " · ".join(bits))
        cmp_ = ["NIFTY %s" % _p(sx.get("nifty_pct"))]
        if sx.get("parent"):
            cmp_.append("%s %s" % (_label(sx["parent"]), _p(sx.get("parent_pct"))))
        if sx.get("rank_since"):
            cmp_.append("rank held since %s" % sx["rank_since"])
        L.append("Compare: " + " · ".join(cmp_))
    elif kind == "stock_move":
        sk = g("stock")
        if not sk.get("price"):
            return ["%s: no live data yet" % args.get("target")]
        word = "flat" if flat else ("up" if (sk.get("pct") or 0) >= 0 else "down")
        L.append("%s %s %s %s %s · %s VWAP%s" % (_arrow(sk.get("pct"), flat), sk["sym"], _px(sk["price"]), word,
                                                 _p(sk.get("pct")), "above" if sk.get("above_vwap") else "below",
                                                 (" · last 15 min %s" % _p(sk["m15"])) if sk.get("m15") is not None
                                                 else ""))
        bits = []
        if sk.get("oi_type") and sk["oi_type"] != "neutral":
            bits.append("OI: %s since %s%s" % (T.OI_TYPE_WORDS[sk["oi_type"]], sk["started"],
                                                ", fresh" if sk.get("fresh") else ""))
        if sk.get("rvol") is not None:
            bits.append("RVOL %s" % _x(sk["rvol"]))
        if sk.get("fut") is not None:
            bits.append("fut vol 15m %s" % _x(sk["fut"]))
        if bits:
            L.append(" · ".join(bits))
        if sk.get("sector_avg") is not None:
            L.append("Sector %s %s (%d/%d up) → %s" % (_short(sk["sector"]), _p(sk["sector_avg"]), sk["sector_up"],
                                                       sk["sector_n"], "stock-specific move" if "STOCK_SPECIFIC" in
                                                       labels else "moving with its sector"))
    elif kind == "strongest_sectors":
        cands = g("strength").get("candidates") or []
        if not cands:
            return ["No sector ranking yet (needs live data)"]
        c = cands[0]
        L.append("%s %s %s · %s" % ("🔻" if direction == "down" else "💪", c["name"],
                                    "weakest" if direction == "down" else "strongest", conf_txt))
        bits = [_p(c["avg"]), "%d/%d %s" % (c["moved"], c["n"], "down" if direction == "down" else "up")]
        if c["checks"]:
            bits.append("top-3 in %d/%d checks (60 min)" % (c["top3"], c["checks"]))
        if c["good_oi"][2]:
            bits.append("%s %d/%d" % (T.OI_TYPE_WORDS[c["good_oi"][0]], c["good_oi"][1], c["good_oi"][2]))
        L.append(" · ".join(bits))
        for i, x in enumerate(cands[1:3], 2):
            line = "#%d %s %s (day-rank %d)" % (i, x["name"], _p(x["avg"]), x["day_rank"] or 0)
            if x["bad_oi"][1] and x["good_oi"][2]:
                line += " · %s %d/%d" % (T.OI_TYPE_WORDS[x["bad_oi"][0]], x["bad_oi"][1], x["bad_oi"][2])
            L.append(line)
    elif kind == "rotation":
        ro = g("rotation")
        rows = {r["name"]: r for r in ro.get("rows") or []}
        if not rows:
            return ["No sector history yet today (needs live samples)"]
        if lean:
            L.append("🔄 Lean: %s (%s) · %s" % (lean, "emerging" if lean in (ro.get("emerging") or [])
                                                 else "steady leader", conf_txt))
        else:
            L.append("🔄 No clear leader or new sector right now")
        for n in (ro.get("emerging") or [])[:2]:
            r = rows[n]
            L.append("Emerging: %s rank %s→%s · last %d min %s · fut vol %s" % (
                n, r["rank_then"], r["rank"], args.get("window_min") or 30, _p(r["change"]), _x(r["fut"])))
        side = ["%s (steady)" % n for n in (ro.get("steady") or [])[:2]] + \
               ["%s (fading)" % n for n in (ro.get("fading") or [])[:2]]
        if side:
            L.append("Day leaders: " + ", ".join(side))
        if ro.get("stocks"):
            L.append("In %s: %s" % ((ro.get("emerging") or ["?"])[0], ", ".join(
                "%s %s %s" % (s, _px(px), _p(v)) for s, px, v, _fv in ro["stocks"])))
    elif kind == "vix":
        vx, ix = g("vix"), g("index")
        if vx.get("price") is None:
            return ["VIX: no live reading yet"]
        word = "flat" if flat else ("up" if (vx.get("pct") or 0) >= 0 else "down")
        L.append("%s VIX %s %s %s" % (_arrow(vx.get("pct"), flat), _px(vx["price"]), word, _p(vx.get("pct"), 1)))
        L.append(" · ".join(x for x in ("NIFTY %s" % _p(ix.get("pct")) if ix.get("pct") is not None else None,
                                        breadth, _oi_phrase(g("oi_mix").get("counts"))) if x))
        mvp = g("path").get("rise15") if direction != "down" else g("path").get("fall15")
        if mvp and abs(mvp[0]) > 0:
            L.append("Biggest VIX move %s–%s (%s)" % (_hm(mvp[1]), _hm(mvp[2]), _p(mvp[0])))
    elif kind == "crude":
        cr = g("crude")
        if cr.get("price") is None:
            return ["Crude: not available (MCX via Kite: see /check)"]
        L.append("🛢 Crude %s %s %s" % (cr["contract"], _px(cr["price"]), _p(cr.get("pct"))))
        for grp, have in (cr.get("groups") or [])[:3]:
            L.append("%s: %s" % (grp, ", ".join("%s %s" % (s, _p(v)) for s, v in have)))
        if g("index").get("pct") is not None:
            L.append("NIFTY %s" % _p(g("index")["pct"]))
    asked = args.get("direction")
    if asked in ("up", "down") and direction in ("up", "down") and asked != direction and L and \
            kind in ("index_move", "sector_move", "stock_move", "vix"):
        L.insert(1, "ℹ️ You asked why it is %s: it is %s" % (asked, "flat" if flat else direction))
    return [x for x in L if x]


# ── DETAILS: one short block per check (built from numbers, not from the long fact lines) ──
def _sec_lines(cid, kind, args, v, direction):
    """(title, [lines]) for one check, or None to fall back to its fact lines."""
    if v.get("missing"):
        return None
    sfmt = lambda x: "%s %s (%d/%d)" % (_short(x["sector"]), _p(x["avg"]), x["up"], x["n"])  # noqa: E731
    if cid == "market":
        L = ["NIFTY 50 %s%s%s" % (_p(v.get("nifty_pct")), "" if v.get("nifty_above_vwap") is None else
                                  (" · futures above VWAP" if v["nifty_above_vwap"] else " · futures below VWAP"),
                                  (" · regime %s" % v["regime"]) if v.get("regime") else "")]
        b = "Breadth %d↑ / %d↓" % (v["adv"], v["dec"])
        if v.get("ago"):
            b += " (%s: %s↑/%s↓%s)" % (v["ago"][0], v["ago"][1], v["ago"][2],
                                      (", " + v["breadth_trend"]) if v.get("breadth_trend") else "")
        L.append(b)
        return "Market", L
    if cid in ("index", "index_cmp"):
        L = ["%s · %s%s" % (_px(v["price"]), _p(v.get("pct")), "" if v.get("fut_above") is None else
                            (" · futures %s their VWAP" % ("above" if v["fut_above"] else "below")))]
        if v.get("high") and v.get("low"):
            L.append("Range %s – %s%s" % (_px(v["low"]), _px(v["high"]),
                                          (" · now at %.0f%% (0 = low, 100 = high)" % v["range_pos"])
                                          if v.get("range_pos") is not None else ""))
        if v.get("off_high") is not None:
            L.append("%s from high%s · %s from low%s" % (
                _p(v["off_high"]), (" (%s)" % v["high_t"]) if v.get("high_t") else "",
                _p(v["off_low"]), (" (%s)" % v["low_t"]) if v.get("low_t") else ""))
        L.append(("Opening move %s" % _p(v["gap"])) if v.get("gap") is not None else
                 "Opening move not known (bot started %s)" % (v.get("first_t") or "late"))
        return _label(v["name"]), L
    if cid in ("path", "sector_path"):
        L = ["Last 15 min %s · last 30 min %s" % (_p(v.get("l15")), _p(v.get("l30")))]
        for key, word in (("rise15", "Biggest rise"), ("fall15", "Biggest fall")):
            x = v.get(key)
            if x and ((key == "rise15" and x[0] > 0) or (key == "fall15" and x[0] < 0)):
                L.append("%s %s–%s %s" % (word, _hm(x[1]), _hm(x[2]), _p(x[0])))
        return "When", L
    if cid == "index_fut":
        L = []
        if v.get("type"):
            L.append("Last %d min: %s%s" % (v["window"], T.OI_TYPE_WORDS[v["type"]],
                                            "" if v["type"] == "neutral" else " since %s%s" % (
                                                v["started"], ", fresh" if v["fresh"] else "")))
            if v["type"] != "neutral":
                L.append("Last %d min: OI %s · price %s" % (v["window"], _p(v["oi_w"]), _p(v["px_w"])))
        if v.get("oi_open") is not None:
            L.append("Whole day: OI %s since open" % _p(v["oi_open"]))
        iv = []
        if v.get("iv") is not None:
            iv.append("ATM IV %.1f%s" % (v["iv"], (" (%+.1f)" % v["iv_chg"]) if v.get("iv_chg") is not None else ""))
        if v.get("slope") is not None and abs(v["slope"]) >= 0.05:
            iv.append("puts dearer" if v["slope"] > 0 else "calls dearer")
        if v.get("dte") is not None:
            iv.append("expiry day" if v["dte"] == 0 else "DTE %d" % v["dte"])
        if iv:
            L.append(" · ".join(iv))
        if v.get("iv") is None:
            L.append("ATM IV: the scanner does not compute it for index futures (India VIX is NIFTY's)")
        return "Futures %s" % v["sym"], L
    if cid == "heavy":
        L = ["%d/%d up today%s" % (v["up"], v["n"], (" · %d/%d up in the last 30 min" % (v["up30"], v["n30"]))
                                    if v.get("n30") else "")]
        return [("Largest %s stocks (no weights)" % _label(v["name"]), L), ("", stock_table(v["rows"]), "mono")]
    if cid == "sectors":
        rows = v.get("rows") or []
        L = ["▲ " + " · ".join(sfmt(x) for x in rows[:3]), "▼ " + " · ".join(sfmt(x) for x in rows[-3:][::-1])]
        if v.get("sudden"):
            L.append("Sharpest 15 min: %s %s" % (_short(v["sudden"][0]), _p(v["sudden"][1])))
        return "Sectors (avg of stocks)", L
    if cid == "breadth":
        if v.get("hi_ev") is None:
            return "", []
        return "Day highs / lows", ["New-high events %s · new-low events %s" % (v["hi_ev"], v["lo_ev"])]
    if cid == "movers":
        rows = v.get("rows") or []
        up = bool(rows) and rows[0][1] >= 0
        return ("Top gainers" if up else "Top fallers"), [
            "%s %s %s · %s" % (sym, _px(px), _p(p), _short(sec)) for sym, p, px, sec in rows]
    if cid == "vix":
        return "VIX", ["%s · %s today%s" % (_px(v["price"]), _p(v.get("pct"), 1),
                                            (" · last 15 min %s" % _p(v["l15"], 1)) if v.get("l15") is not None else "")]
    if cid == "oi_mix":
        c = v.get("counts") or {}
        order = sorted(T.OI_TYPES, key=lambda k: -c.get(k, 0))
        return "OI · last 30 min (%d stocks)" % v.get("n", 0), [" · ".join(
            "%s %d" % (T.OI_TYPE_WORDS[k], c.get(k, 0)) for k in order)]
    if cid == "crude":
        L = ["%s %s · %s%s" % (v.get("contract") or "CRUDEOIL", _px(v["price"]), _p(v.get("pct")),
                               (" · OI %s" % _p(v["oi_pct"], 1)) if v.get("oi_pct") is not None else "")]
        for grp, have in v.get("groups") or []:
            L.append("%s: %s" % (grp, " · ".join("%s %s" % (s_, _p(p)) for s_, p in have)))
        return "Crude", L
    if cid == "broader":
        vals = v.get("vals") or {}
        L = [" · ".join("%s %s" % (n.replace("NIFTY ", "").title().replace("Smallcap", "Smallcap"), _p(x))
                        for n, x in vals.items())]
        if v.get("ref_pct") is not None:
            L.append("NIFTY 50 %s" % _p(v["ref_pct"]))
        return "Broader market", L
    if cid == "sector":
        st = v.get("stats") or {}
        L = ["%d/%d up · avg %s · %d/%d above VWAP" % (st["up"], st["n"], _p(st["avg"]), st["above"], st["n"])]
        if v.get("range_pos") is not None:
            L.append("Index at %.0f%% of low→high range · %s from high%s · %s from low%s" % (
                v["range_pos"], _p(v["off_high"]), (" (%s)" % v["high_t"]) if v.get("high_t") else "",
                _p(v["off_low"]), (" (%s)" % v["low_t"]) if v.get("low_t") else ""))
        if st.get("n30"):
            L.append("Last 30 min: %d/%d stocks up" % (st["up30"], st["n30"]))
        if v.get("leaders"):
            L.append("▲ " + " · ".join("%s %s %s" % (a, _px(px), _p(p)) for a, px, p in v["leaders"]))
        if v.get("laggards") and v["laggards"] != v.get("leaders"):
            L.append("▼ " + " · ".join("%s %s %s" % (a, _px(px), _p(p)) for a, px, p in v["laggards"]))
        bits = ["RVOL %s" % _x(st.get("rvol"))]
        if st.get("fut") is not None:
            bits.append("fut vol 15m %s" % _x(st["fut"]))
        if st.get("pace") is not None:
            bits.append("PACE used %.2f" % st["pace"])
        L.append(" · ".join(bits))
        if st.get("oi_n"):
            L.append("OI 30 min: " + " · ".join(
                "%s %d%s" % (T.OI_TYPE_WORDS[k], st["oi"][k], (" (%d fresh)" % st["fresh"][k]) if st["fresh"][k] else "")
                for k in T.OI_TYPES if st["oi"][k]))
        cmp_ = ["NIFTY 50 %s" % _p(v.get("nifty_pct"))]
        if v.get("gap") is not None:
            cmp_.append("gap %s" % _p(v["gap"]))
        if v.get("parent"):
            cmp_.append("%s %s" % (_label(v["parent"]), _p(v.get("parent_pct"))))
        if v.get("rank_now"):
            cmp_.append("rank #%d of %d%s" % (v["rank_now"], v["rank_of"],
                                              (" since " + v["rank_since"]) if v.get("rank_since") else ""))
        L.append(" · ".join(cmp_))
        name = _short(args.get("target") or "")
        return [("%s stocks" % name, L), ("", stock_table(st.get("rows") or []), "mono")]
    return None


def sections(kind, args, ev, direction, mono=None):
    """[(title, [lines])] in plan order. A table is a section with an empty title that follows its
    summary; its position in the list is appended to `mono` (shown monospace, columns line up)."""
    out = []
    for cid, _name, _kw in plan(kind, args.get("target"), args.get("compare"), args.get("direction"),
                                args.get("window_min") or 30):
        v = ev.get(cid) or {}
        sec = _sec_lines(cid, kind, args, v, direction)
        if sec is None:
            if v.get("facts"):
                out.append((cid.replace("_", " ").title(), list(v["facts"])))
            continue
        for one in (sec if isinstance(sec, list) else [sec]):
            if not one[1]:
                continue
            out.append((one[0], one[1]))
            if len(one) > 2 and mono is not None:
                mono.append(len(out) - 1)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY without the agent (tool call, evals, quota out) + the shared renderer
# ═════════════════════════════════════════════════════════════════════════════
def assemble(ctx, kind, args, ev):
    labels, cands, conf, lean, direction = analyze(kind, args, ev)
    mono = []
    mp = _main_pct(kind, ev)
    flat = mp is not None and abs(mp) < FLAT_PCT and kind in ("index_move", "sector_move", "stock_move", "vix")
    facts = []
    for cid, _name, _kw in plan(kind, args.get("target"), args.get("compare"), args.get("direction"),
                                args.get("window_min") or 30):
        facts += (ev.get(cid) or {}).get("facts") or []
    return {"tool": "explain_market", "kind": kind, "args": dict(args), "direction": direction,
            "evidence": ev, "facts": facts, "labels": labels, "candidates": cands, "confidence": conf,
            "lean": lean, "not_checked": not_checked(kind, args.get("target"), ev),
            "headline": headline(kind, args, direction, T._as_of(ctx), flat), "as_of": T._as_of(ctx), "view": None,
            "flat": flat, "brief": brief(kind, args, ev, labels, lean, conf, direction, flat),
            "sections": sections(kind, args, ev, direction, mono), "mono": mono}


def explain(ctx, kind, target=None, compare=None, direction=None, window_min=30):
    args = {"kind": kind, "target": target, "compare": compare, "direction": direction, "window_min": window_min}
    ev = {cid: run_check(ctx, name, kw) for cid, name, kw in plan(kind, target, compare, direction, window_min)}
    if all(v.get("missing") for v in ev.values()):
        raise T.ToolError("No live data to explain yet (market closed or app just started).")
    return assemble(ctx, kind, args, ev)


def render(r):
    """Plain-text answer (tool path, logs, evals): the brief, then the details."""
    return "\n".join(r["brief"]) + "\n\n" + render_details(r, html=False)


def render_details(r, html=True):
    """The 📊 Details view: short titled blocks, then the AI view, the pattern, what was not checked.
    html=True -> Telegram HTML (bold titles); False -> plain text."""
    import html as _h
    esc = (lambda x: _h.escape(str(x), quote=False)) if html else str  # noqa: E731
    bold = (lambda x: "<b>%s</b>" % esc(x)) if html else (lambda x: str(x).upper())  # noqa: E731
    ital = (lambda x: "<i>%s</i>" % esc(x)) if html else str  # noqa: E731
    out = [bold(r["headline"])]
    mono = set(r.get("mono") or ())
    for i, (title, lines) in enumerate(r.get("sections") or [("Evidence", r["facts"])]):
        head = ["", bold(title)] if title else []
        if html and i in mono:                     # a table: monospace, so the columns line up
            out += head + ["<pre>%s</pre>" % esc("\n".join(lines))]
        else:
            out += head + [esc(x) for x in lines]
    L = []
    v = r.get("view") or {}
    if v.get("insight") or v.get("why") or v.get("against") or v.get("changes_view"):
        L += ["", "🧠 View (AI, every number checked)"]
        if v.get("insight"):
            L.append(v["insight"])
        if v.get("why"):
            L.append("Why: " + "; ".join(v["why"]))
        if v.get("against"):
            L.append("Against: " + "; ".join(v["against"]))
        if v.get("changes_view"):
            L.append("Changes the view: " + v["changes_view"])
    if r.get("lean"):
        c = r.get("confidence") or {}
        L += ["", "Lean (by code): %s%s" % (r["lean"], (" · confidence %s (%d of %d checks agree)" %
                                                         (c["level"], c["k"], c["n"])) if c else "")]
    if r.get("view_note"):
        L.append(r["view_note"])
    out += [esc(x) if not x.startswith("🧠") else (bold(x) if html else x) for x in L]
    if r.get("labels"):
        out += ["", bold("Pattern")] + [esc("• " + LABEL_TEXT.get(x, x)) for x in r["labels"]]
    out += ["", ital("Not checked: " + ", ".join(r["not_checked"])),
            ital("Moves together, not causes. Not advice.")]
    return "\n".join(out)


_LABEL = re.compile(r"^([A-Za-z][A-Za-z0-9 /&]{1,22}):(?=\s)")


def _bold_label(line):
    """'Now: 19% of day range ...' -> '<b>Now</b> 19% ...': the eye finds the row it wants."""
    return _LABEL.sub(lambda m: "<b>%s</b>" % m.group(1), line, count=1)


def render_brief_html(r):
    """The phone answer: bold first line, 2-4 driver lines, at most ONE checked AI line."""
    import html as _h
    b = r["brief"] or [r["headline"]]
    L = ["<b>%s</b>" % _h.escape(b[0], quote=False)] + [_bold_label(_h.escape(x, quote=False)) for x in b[1:]]
    ins = (r.get("view") or {}).get("insight")
    if ins:
        L.append("🧠 <i>%s</i>" % _h.escape(ins, quote=False))
    elif r.get("view_note"):
        L.append("<i>%s</i>" % _h.escape(r["view_note"], quote=False))
    L.append("<i>%s · not advice</i>" % str(r.get("as_of") or "")[:5])
    return "\n".join(L)
