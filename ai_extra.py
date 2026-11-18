# -*- coding: utf-8 -*-
"""
ai_extra.py -- data the scanner does not keep, fetched for the AI assistant only.  v19, 09-11-2026.

    Broader indices  NIFTY MIDCAP 150 / MIDCAP SELECT / SMALLCAP 250 / MICROCAP 250 / NIFTY 500
                     -> ONE request to NSE allIndices every 2 min (the same source market_data.py
                        already uses for Mid Select and Microcap)
    Crude oil        -> MCX CRUDEOIL near-month future through the app's own Kite session
                        (rolls to the next month 2 days before expiry)

Writes ai_tools.EXTRA_INDICES; the minute sampler then treats them like any index, so
'midcap now', 'alert me when midcap falls 1%' and the WHY agent all read the same numbers.
Never raises: a failure is recorded in STATUS and shown by /check. Read-only: no order calls.
Not here (and the agent says so): FII / DII (NSE publishes them only after the close).
"""
import threading
import time
from datetime import date, datetime, timedelta

import ai_tools as T

NSE_EVERY_S = 120                 # allIndices refreshes about every minute; 2 min is plenty
CRUDE_EVERY_S = 60
ROLL_DAYS = 2                     # crude expires mid-month: use the next contract 2 days before
_NSE_URL = "https://www.nseindia.com/api/allIndices"
_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
# our name -> words that identify it in NSE's "index" / "indexSymbol" (both checked, upper case)
NSE_INDICES = {
    "NIFTY MIDCAP 150": ("NIFTY MIDCAP 150",),
    "NIFTY MIDCAP SELECT": ("NIFTY MIDCAP SELECT", "NIFTY MID SELECT"),
    "NIFTY SMALLCAP 250": ("NIFTY SMALLCAP 250", "NIFTY SMLCAP 250"),
    "NIFTY MICROCAP 250": ("NIFTY MICROCAP 250", "NIFTY MICROCAP250"),
    "NIFTY 500": ("NIFTY 500",),
}

STATUS = {"nse": None, "nse_at": None, "crude": None, "crude_at": None, "contract": None}
_lock = threading.Lock()
_last = {"nse": 0.0, "crude": 0.0}
_sess = {"s": None, "born": 0.0}
_crude = {"day": None, "inst": None, "oi_first": None}


def _f(v):
    return T._f(v)


# ── NSE broader indices ──────────────────────────────────────────────────────
def parse_nse(payload, now_ts):
    """allIndices JSON -> {our name: reading}. Pure, so tests feed it a saved sample."""
    items = payload.get("data", []) if isinstance(payload, dict) else (payload or [])
    out = {}
    for name, keys in NSE_INDICES.items():
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = {str(it.get("index") or "").upper().strip(), str(it.get("indexSymbol") or "").upper().strip()}
            if nm & set(keys):
                last = _f(it.get("last") or it.get("lastPrice"))
                pdc = _f(it.get("previousClose"))
                if not last:
                    break
                pct = _f(it.get("percentChange"))
                if pct is None and pdc:
                    pct = (last / pdc - 1) * 100.0
                out[name] = {"price": last, "pct": pct, "open": _f(it.get("open")), "pdc": pdc,
                             "day_high": _f(it.get("high")) or last, "day_low": _f(it.get("low")) or last,
                             "vwap_v": None, "ts": now_ts, "src": "NSE"}
                break
    return out


def _nse_get():
    import requests
    if _sess["s"] is None or time.time() - _sess["born"] > 30 * 60:      # fresh cookies every 30 min
        s = requests.Session()
        s.get("https://www.nseindia.com/", headers=_HDR, timeout=6)
        _sess["s"], _sess["born"] = s, time.time()
    r = _sess["s"].get(_NSE_URL, headers=_HDR, timeout=6)
    if r.status_code in (401, 403):
        _sess["s"] = None
        raise RuntimeError("NSE %d (cookies expired, retry next round)" % r.status_code)
    r.raise_for_status()
    return r.json()


def poll_nse(now_ts=None, fetch=_nse_get):
    now_ts = now_ts or time.time()
    try:
        got = parse_nse(fetch(), now_ts)
        with _lock:
            T.EXTRA_INDICES.update(got)
            STATUS["nse"] = "ok %d/%d" % (len(got), len(NSE_INDICES))
            STATUS["nse_at"] = datetime.now().strftime("%H:%M")
        return len(got)
    except Exception as e:
        with _lock:
            STATUS["nse"] = "error: %s" % str(e)[:100]
            STATUS["nse_at"] = datetime.now().strftime("%H:%M")
        return 0


# ── MCX crude (near month) ───────────────────────────────────────────────────
def pick_contract(instruments, today):
    """Nearest CRUDEOIL future that is more than ROLL_DAYS from expiry (CRUDEOILM is the mini: skipped)."""
    futs = []
    for i in instruments or []:
        if str(i.get("name") or "").upper() != "CRUDEOIL" or str(i.get("instrument_type") or "") != "FUT":
            continue
        exp = i.get("expiry")
        if isinstance(exp, datetime):
            exp = exp.date()
        if isinstance(exp, str):
            try:
                exp = datetime.strptime(exp[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
        if isinstance(exp, date) and exp - today > timedelta(days=ROLL_DAYS):
            futs.append((exp, i))
    futs.sort(key=lambda x: x[0])
    return futs[0][1] if futs else None


def parse_quote(q, contract, now_ts, oi_first=None):
    last = _f(q.get("last_price"))
    if not last:
        return None
    ohlc = q.get("ohlc") or {}
    pdc = _f(ohlc.get("close"))
    oi = _f(q.get("oi"))
    return {"price": last, "pct": (last / pdc - 1) * 100.0 if pdc else None, "open": _f(ohlc.get("open")),
            "pdc": pdc, "day_high": _f(ohlc.get("high")) or last, "day_low": _f(ohlc.get("low")) or last,
            "vwap_v": _f(q.get("average_price")), "oi": oi,
            "oi_pct": (oi / oi_first - 1) * 100.0 if oi and oi_first else None,
            "contract": contract, "ts": now_ts, "src": "Kite MCX"}


def poll_crude(state, now_ts=None):
    now_ts = now_ts or time.time()
    try:
        km = getattr(state, "kite_mgr", None)
        kite = getattr(km, "kite", None)
        if kite is None:
            raise RuntimeError("Kite not logged in")
        today = date.today()
        if _crude["day"] != today or _crude["inst"] is None:
            inst = pick_contract(kite.instruments("MCX"), today)
            if inst is None:
                raise RuntimeError("no CRUDEOIL future in the MCX list (MCX segment off?)")
            _crude.update(day=today, inst=inst, oi_first=None)
        ts = "MCX:" + _crude["inst"]["tradingsymbol"]
        q = (kite.quote([ts]) or {}).get(ts)
        if not q:
            raise RuntimeError("no quote for %s (MCX data not in this Kite plan?)" % ts)
        if _crude["oi_first"] is None and _f(q.get("oi")):
            _crude["oi_first"] = _f(q.get("oi"))           # first reading of the day = OI baseline
        r = parse_quote(q, _crude["inst"]["tradingsymbol"], now_ts, _crude["oi_first"])
        if r is None:
            raise RuntimeError("empty quote")
        with _lock:
            T.EXTRA_INDICES["CRUDEOIL"] = r
            STATUS["crude"], STATUS["contract"] = "ok", r["contract"]
            STATUS["crude_at"] = datetime.now().strftime("%H:%M")
        return True
    except Exception as e:
        with _lock:
            STATUS["crude"] = "error: %s" % str(e)[:100]
            STATUS["crude_at"] = datetime.now().strftime("%H:%M")
        return False


def poll(state):
    """Called by ai_chat's sampler every minute; each source keeps its own pace."""
    t = time.time()
    if t - _last["nse"] >= NSE_EVERY_S:
        _last["nse"] = t
        poll_nse()
    if t - _last["crude"] >= CRUDE_EVERY_S:
        _last["crude"] = t
        poll_crude(state)


def status_line():
    with _lock:
        st = dict(STATUS)
    return ("%s Broader indices %s%s\n%s Crude (MCX)     %s%s" % (
        "✅" if (st["nse"] or "").startswith("ok") else "❌", st["nse"] or "not fetched yet",
        (" · " + st["nse_at"]) if st["nse_at"] else "",
        "✅" if st["crude"] == "ok" else "❌", (st["contract"] if st["crude"] == "ok" else st["crude"]) or "not fetched yet",
        (" · " + st["crude_at"]) if st["crude_at"] else ""))
