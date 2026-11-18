# -*- coding: utf-8 -*-
"""
ai_watch.py -- five ready-made market alerts.  v20, 11-11-2026.

    fut      Futures flip     NIFTY / BANKNIFTY futures start a FRESH long or short build-up
    fade     Fade / bounce    the index is falling away from the day high (or rising off the low)
    breadth  Breadth turn     advance / decline starts weakening or improving
    sector   Sector OI        most stocks of one sector show the same build-up, some of it fresh
    vix      VIX jump         India VIX moves sharply in 15 min

HOW IT DECIDES (no LLM anywhere in this file)
    Every minute the sampler calls check(). Each detector turns live data into a STATE per subject
    ("NIFTY 50" -> "short_buildup"). A ping goes out only when the state CHANGES into a firing state:
    a level that simply stays true never pings twice. On top of that:
      * the first reading of a subject each day is only a baseline (a restart at 12:00 does not
        replay everything that is already true),
      * one subject pings at most once per COOLDOWN_S,
      * one watch pings at most MAX_PER_DAY times a day.
    Without these three, five watches on a moving market ping all day and the bot gets muted.

    On / off per watch and the last state per subject live in data/ai_watch.json (atomic write),
    so a restart keeps your switches and does not re-fire what already fired.

    /watch   shows the five with on / off buttons.   Each ping carries a "Why" button -> the WHY agent.
"""
import html
import json
import os
import threading
import time
from datetime import datetime

import ai_evidence as E
import ai_tools as T

COOLDOWN_S = 30 * 60
MAX_PER_DAY = 6
INDICES = ("NIFTY 50", "NIFTY BANK")
FADE_PCT = 0.40                  # % off the day high (low) before a fade (bounce) ping
FADE_FROM = 9 * 60 + 30          # in the first 15 min the day range is tiny: everything is "off the high"
VIX_JUMP = 3.0                   # % in 15 min
SECTOR_SHARE = 0.6               # share of the sector's stocks (with OI data) in the same build-up
SECTOR_MIN = 3                   # ... and at least this many stocks with OI data
SECTOR_FRESH = 2                 # ... and at least this many of them fresh

WATCHES = [
    ("fut", "Futures flip", "NIFTY / BANKNIFTY futures start a fresh long or short build-up"),
    ("fade", "Fade / bounce", "index falls %.1f%% off the day high, or rises that much off the low" % FADE_PCT),
    ("breadth", "Breadth turn", "advance / decline starts weakening or improving"),
    ("sector", "Sector OI", "most stocks of a sector show the same build-up, some fresh"),
    ("vix", "VIX jump", "VIX moves %.0f%% or more in 15 min" % VIX_JUMP),
]
KEYS = [k for k, _n, _d in WATCHES]
NAMES = {k: n for k, n, _d in WATCHES}


def _p(v, d=2):
    return E._p(v, d)


def _idx_line(ctx, name):
    """'NIFTY 50 25,294.2 · +1.00% today · 19% of low→high range' (the context under every index ping)."""
    f = T.index_fields(ctx, name) or {}
    if not f.get("price"):
        return None
    nf = E.now_fields(ctx, name, f)
    bits = ["%s %s" % (E._label(name), E._px(f["price"])), "%s today" % _p(f.get("pct"))]
    if nf.get("range_pos") is not None:
        bits.append(E._range_words(nf["range_pos"]))
    return " · ".join(bits)


# ── detectors: ctx -> [(subject, state, text or None, why or None)] ──────────
# state "none" (or any state with text None) never pings; it only records that the condition ended.
def detect_fut(ctx):
    out = []
    for name in INDICES:
        c = T.index_futures(ctx, name, 30)
        if not c or not c.get("type"):
            continue                                   # no 30-min OI history yet: no reading, state kept
        if c["type"] in ("long_buildup", "short_buildup") and c["fresh"]:
            text = "<b>🔔 %s futures: fresh %s since %s</b>\n30m OI %s, price %s%s\n%s" % (
                c["sym"], T.OI_TYPE_WORDS[c["type"]], c["started"], _p(c["oi_w"]), _p(c["px_w"]),
                (" · day OI %s" % _p(c["oi_open"])) if c.get("oi_open") is not None else "",
                _idx_line(ctx, name) or "")
            out.append((name, c["type"], text.rstrip(), ("index_move", name)))
        else:
            out.append((name, "none", None, None))
    return out


def detect_fade(ctx):
    now = ctx.now()
    if now.hour * 60 + now.minute < FADE_FROM:
        return []
    out = []
    for name in INDICES:
        f = T.index_fields(ctx, name) or {}
        if not f.get("price"):
            continue
        nf = E.now_fields(ctx, name, f)
        if nf["now"] == "fading" and nf["off_high"] <= -FADE_PCT:
            text = "<b>🔔 %s fading: %s from the day high%s</b>\nlast 30m %s · %s" % (
                E._label(name), _p(nf["off_high"]), (" (%s)" % nf["high_t"]) if nf.get("high_t") else "",
                _p(nf["l30"]), _idx_line(ctx, name))
            out.append((name, "fading", text, ("index_move", name)))
        elif nf["now"] == "recovering" and nf["off_low"] >= FADE_PCT:
            text = "<b>🔔 %s bouncing: %s from the day low%s</b>\nlast 30m %s · %s" % (
                E._label(name), _p(nf["off_low"]), (" (%s)" % nf["low_t"]) if nf.get("low_t") else "",
                _p(nf["l30"]), _idx_line(ctx, name))
            out.append((name, "recovering", text, ("index_move", name)))
        else:
            out.append((name, "none", None, None))
    return out


def detect_breadth(ctx):
    mk = E.check_market(ctx)
    trend = mk.get("breadth_trend")
    if trend is None:
        return []
    if trend not in ("weakening", "improving"):
        return [("market", "none", None, None)]
    ago = mk.get("ago")
    text = "<b>🔔 Breadth %s: %d↑ / %d↓ now</b>\n%sNIFTY %s%s" % (
        trend, mk["adv"], mk["dec"], ("%s: %s↑ / %s↓ · " % ago) if ago else "", _p(mk.get("nifty_pct")),
        (" · VIX %s" % _p(mk["vix_pct"], 1)) if mk.get("vix_pct") is not None else "")
    return [("market", trend, text, ("index_move", "NIFTY 50"))]


def detect_sector(ctx):
    res = E._stock_results(ctx)
    if not res:
        return []
    shared = {"oit": T.oi_types(ctx, 30)[0], "fv": {}, "pace": {}, "m30": {}}     # only OI is needed here
    nifty = (T.index_fields(ctx, "NIFTY 50") or {}).get("pct")
    out = []
    for sec in E._sector_names(ctx):
        m = E._members_stats(ctx, sec, res, shared=shared)
        if not m or m["oi_n"] < SECTOR_MIN:
            continue
        state, text = "none", None
        for k in ("long_buildup", "short_buildup"):
            if m["oi"][k] / float(m["oi_n"]) >= SECTOR_SHARE and m["fresh"][k] >= SECTOR_FRESH:
                state = k
                top = m["rows"][:2] if k == "long_buildup" else m["rows"][-2:][::-1]
                text = "<b>🔔 %s: %s in %d/%d stocks (%d fresh)</b>\n%s avg %s · %d/%d up · NIFTY %s\n%s" % (
                    E._short(sec), T.OI_TYPE_WORDS[k], m["oi"][k], m["oi_n"], m["fresh"][k], E._short(sec),
                    _p(m["avg"]), m["up"], m["n"], _p(nifty),
                    ", ".join("%s %s" % (html.escape(r["sym"], quote=False), _p(r["pct"])) for r in top))
        out.append((sec, state, text, ("sector_move", sec) if text else None))
    return out


def detect_vix(ctx):
    v = E.check_vix(ctx)
    if v.get("missing") or v.get("l15") is None:
        return []
    if abs(v["l15"]) < VIX_JUMP:
        return [("INDIA VIX", "none", None, None)]
    nifty = (T.index_fields(ctx, "NIFTY 50") or {}).get("pct")
    text = "<b>🔔 VIX %s %s in 15 min</b>\nVIX %s · %s today · NIFTY %s" % (
        "up" if v["l15"] > 0 else "down", _p(v["l15"], 1), E._px(v["price"]), _p(v.get("pct"), 1), _p(nifty))
    return [("INDIA VIX", "up" if v["l15"] > 0 else "down", text, ("vix", None))]


DETECTORS = {"fut": detect_fut, "fade": detect_fade, "breadth": detect_breadth, "sector": detect_sector,
             "vix": detect_vix}


# ── the JSON file: switches + last state per subject ─────────────────────────
class WatchStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self.d = {"on": {k: True for k in KEYS}, "day": "", "last": {}, "count": {}}
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.d["on"].update({k: bool(v) for k, v in (raw.get("on") or {}).items() if k in KEYS})
                self.d["day"] = str(raw.get("day") or "")
                self.d["last"] = dict(raw.get("last") or {})
                self.d["count"] = dict(raw.get("count") or {})
        except (OSError, ValueError):
            pass                                        # no file yet / unreadable: defaults (all on)

    def _save(self):
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f, indent=1)
            os.replace(tmp, self.path)                  # atomic: a crash never leaves half a file
        except OSError as e:
            print("[AI watch] save failed: %s" % e)

    def is_on(self, key):
        return bool(self.d["on"].get(key))

    def set_on(self, key, on):
        with self._lock:
            for k in (KEYS if key == "all" else [key]):
                if k in KEYS:
                    self.d["on"][k] = bool(on)
            self._save()

    def roll(self, day):
        """A new trading day starts clean: yesterday's states are not today's baseline."""
        with self._lock:
            if self.d["day"] != day:
                self.d.update(day=day, last={}, count={})
                self._save()

    def step(self, key, subject, state, firing, now):
        """Record the state; True when this reading should ping (see the module doc for the rules)."""
        sk = "%s|%s" % (key, subject)
        with self._lock:
            prev = self.d["last"].get(sk)
            changed = prev is None or prev.get("state") != state
            fire = bool(firing and prev is not None and changed
                        and now - float(prev.get("fired") or 0) >= COOLDOWN_S
                        and int(self.d["count"].get(key) or 0) < MAX_PER_DAY)
            if changed or fire:
                rec = {"state": state, "fired": float((prev or {}).get("fired") or 0)}
                if fire:
                    rec["fired"] = now
                    self.d["count"][key] = int(self.d["count"].get(key) or 0) + 1
                self.d["last"][sk] = rec
                self._save()
            return fire


def check(ctx, store, send, chats, now=None):
    """One pass. send(chat, Rich text, buttons) -> bool. Returns [(watch, subject, state)] that pinged.
    A detector that raises is logged and skipped: one bad watch never stops the others."""
    now = now if now is not None else time.time()
    store.roll(ctx.today)
    fired = []
    for key in KEYS:
        if not store.is_on(key):
            continue
        try:
            found = DETECTORS[key](ctx)
        except Exception as e:
            print("[AI watch] %s failed: %s: %s" % (key, type(e).__name__, e))
            continue
        for subject, state, text, why in found:
            if not store.step(key, subject, state, bool(text), now):
                continue
            body = T.Rich("%s\n<i>%s · %s · /watch to switch off</i>" % (
                text, datetime.fromtimestamp(now).strftime("%H:%M"), NAMES[key]))
            btn = [[("🧠 Why", "ww:%s:%s" % (why[0], why[1] or ""))]] if why else None
            for chat in chats:
                try:
                    send(chat, body, btn)
                except Exception as e:
                    print("[AI watch] send failed: %s" % e)
            fired.append((key, subject, state))
    return fired


# ── /watch: the card and its buttons ─────────────────────────────────────────
def card(store):
    L = ["<b>🔔 Market alerts</b>", "<i>Each fires on a change, at most once per %d min.</i>" % (COOLDOWN_S // 60), ""]
    for k, name, desc in WATCHES:
        n = int(store.d["count"].get(k) or 0)
        L.append("%s <b>%s</b> — %s%s" % ("✅" if store.is_on(k) else "⏸", name, desc,
                                          (" · %d today" % n) if n else ""))
    rows, row = [], []
    for k, name, _d in WATCHES:
        row.append(("%s %s" % ("⏸" if store.is_on(k) else "▶️", name), "wt:%s:%s" % (k, "off" if store.is_on(k) else "on")))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([("▶️ All on", "wt:all:on"), ("⏸ All off", "wt:all:off")])
    return T.Rich("\n".join(L)), rows


def on_button(store, data):
    """'wt:<key>:on|off' -> the refreshed card."""
    _p0, key, val = (str(data).split(":", 2) + ["", ""])[:3]
    if (key in KEYS or key == "all") and val in ("on", "off"):
        store.set_on(key, val == "on")
    return card(store)
