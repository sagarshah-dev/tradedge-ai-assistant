# -*- coding: utf-8 -*-
"""v20 (11-11-2026): the index / sector answer says what price does NOW (not only the day %), index futures
windows are named, the largest stocks are counted, every sector stock is shown in a table, and five
ready-made market alerts fire on a CHANGE of state (ai_watch.py)."""
import json
import os
import types
from collections import deque
from datetime import datetime

import pytest

import ai_agent as AG
import ai_chat
import ai_evidence as E
import ai_tools as T
import ai_watch as W


def ok(m, name, **args):
    r, err = T.run(m.ctx, name, args)
    assert err is None, err
    return r, T.format_reply(r)


def _nifty_series(ctx, day_open_pct, peak_pct, now_pct, peak_min=55, pdc=25000.0):
    """NIFTY 50 minute samples 09:15 -> 11:30: opens at day_open_pct, peaks at peak_min, ends at now_pct."""
    now = T._epoch(ctx)
    t0, xs, hi, lo = now - 135 * 60, [], 0.0, 1e12
    for k in range(136):
        if k <= peak_min:
            pct = day_open_pct + (peak_pct - day_open_pct) * k / float(peak_min)
        else:
            pct = peak_pct + (now_pct - peak_pct) * (k - peak_min) / float(135 - peak_min)
        px = pdc * (1 + pct / 100.0)
        hi, lo = max(hi, px), min(lo, px)
        xs.append((t0 + k * 60, px, hi, lo, pdc * 1.012, pct))
    return deque(xs, maxlen=T.ISAMPLE_KEEP)


def _fut_hist(ctx, oi_step, px_step, base=25330.0):
    """NIFTY futures OI history: flat until 30 min ago, then OI moves oi_step and price px_step per minute."""
    now = T._epoch(ctx)
    return deque([{"ts": now - 60 * k, "raw": 1e7 * (0.985 + (0 if k > 30 else (30 - k) * oi_step)),
                   "ltp": base + (0 if k > 30 else (30 - k) * px_step)} for k in range(90, -1, -3)])


@pytest.fixture
def fading(mkt, monkeypatch):
    """Live 11-11 12:32: NIFTY +1.00% on the day but sliding from the 10:10 high, futures in a FRESH short
    build-up for 30 min while their OI is down since the open. No index IV (the scanner skips indices)."""
    st, ctx = mkt.state, mkt.ctx
    monkeypatch.setitem(T._ISAMPLES, "NIFTY 50", _nifty_series(ctx, 0.9, 1.7, 1.0))
    monkeypatch.setitem(st.SECTOR_MAP, "NIFTY", "INDEX")
    monkeypatch.setitem(st.OI_HISTORY, "NIFTY", _fut_hist(ctx, 0.0004, -2.2))
    monkeypatch.setitem(st.OI_CACHE, "NIFTY", {"oi_from_open": -1.04})
    monkeypatch.setattr(st, "kite_mgr", types.SimpleNamespace(_expiry_map={"NIFTY": "2026-11-18"}), raising=False)
    return mkt


# ── what is it doing NOW ─────────────────────────────────────────────────────
@pytest.mark.parametrize("pos,off_high,off_low,l30,want", [
    (95.0, -0.02, 1.0, 0.2, "at_high"), (90.0, -0.1, 1.0, None, "at_high"), (5.0, -1.0, 0.02, -0.2, "at_low"),
    (10.0, -1.0, 0.1, None, "at_low"),
    (40.0, -0.40, 0.3, -0.20, "fading"), (40.0, -0.40, 0.3, -0.10, "fading"), (40.0, -0.25, 0.3, -0.20, "fading"),
    (40.0, -0.24, 0.3, -0.20, "holding"),            # not far enough from the high
    (40.0, -0.40, 0.3, -0.09, "holding"),            # the last 30 min did not really fall
    (40.0, -0.40, 0.3, None, "holding"),             # no 30-min history: never guessed
    (60.0, -0.3, 0.40, 0.20, "recovering"), (60.0, -0.3, 0.25, 0.10, "recovering"),
    (60.0, -0.3, 0.24, 0.20, "holding"), (60.0, -0.3, 0.40, 0.09, "holding"),
    (50.0, -0.5, 0.5, 0.0, "holding"), (None, None, None, 0.5, None)])
def test_now_state_rules(pos, off_high, off_low, l30, want):
    assert E.now_state(pos, off_high, off_low, l30) == want


def test_live_11_11_up_on_the_day_but_fading(fading):
    """Live 11-11 12:32: NIFTY +1.00% at 19% of its range with a fresh short build-up in futures; the bot led
    with 'Led by PSU BK' (the morning) and printed 'short build-up ... OI -1.04%' as if they were one window."""
    r, text = ok(fading, "explain_market", kind="index_move", direction="down")
    b = r["brief"]
    assert b[0] == "📈 NIFTY 50 up +1.00% · 25,250.0"
    assert b[1] == "ℹ️ You asked why it is down: it is up"
    assert b[2] == "⚡ Up on the day but fading from the high of 10:10 · futures adding shorts since 11:00, fresh"
    assert b[3] == "Now: 13% of low→high range · -0.69% from high (10:10) · last 30m -0.26% · futures above VWAP"
    assert b[4] == ("Fut NIFTY: short build-up since 11:00 (fresh) · 30m OI +1.22%, price -0.26% · "
                    "day OI -1.04% · DTE 12")
    assert b[5].startswith("Heavy: ") and b[6].startswith("Sectors: led by ") and b[7].startswith("Breadth: ")
    assert {"FADING", "FUT_SHORT_BUILDUP"} <= set(r["labels"])
    ix = r["evidence"]["index"]
    assert ix["now"] == "fading" and ix["high_t"] == "10:10" and ix["low_t"] == "09:15"
    assert ix["off_high"] == pytest.approx((1.010 / 1.017 - 1) * 100) and ix["l30"] == pytest.approx(-0.2594, abs=1e-3)
    nums, times = AG.evidence_numbers(r["facts"])                    # the brief invents no number
    assert AG.bad_numbers("\n".join(b), nums, times) == []


@pytest.mark.parametrize("o,peak,now,pm,read,state", [
    (0.2, 1.0, 1.0, 135, "Up on the day and at the day high", "at_high"),
    (-0.2, -1.0, -1.0, 135, "Down on the day and at the day low", "at_low"),
    (-0.9, -1.7, -1.0, 55, "Down on the day but recovering from the low of 10:10", "recovering"),
    (0.9, 1.7, 1.0, 55, "Up on the day but fading from the high of 10:10", "fading"),
    (0.3, 1.0, 0.8, 20, "Up on the day and high in the day range", "holding"),
    (0.3, 1.0, 0.65, 20, "Up on the day, mid-range", "holding"),
    (-0.05, 0.6, -0.04, 20, "Flat on the day, at the day low", "at_low")])
def test_read_line_day_against_now(mkt, monkeypatch, o, peak, now, pm, read, state):
    monkeypatch.setitem(T._ISAMPLES, "NIFTY 50", _nifty_series(mkt.ctx, o, peak, now, pm))
    r, _ = ok(mkt, "explain_market", kind="index_move")
    assert r["evidence"]["index"]["now"] == state
    assert [x for x in r["brief"] if x.startswith("⚡ ")][0].startswith("⚡ " + read)


def test_high_set_before_the_bot_started_has_no_time(mkt, monkeypatch):
    xs = list(_nifty_series(mkt.ctx, 0.9, 1.7, 1.0))
    late = deque(xs[90:], maxlen=T.ISAMPLE_KEEP)            # first sample 10:45: the 10:10 high is older
    monkeypatch.setitem(T._ISAMPLES, "NIFTY 50", late)
    nf = E.now_fields(mkt.ctx, "NIFTY 50")
    assert nf["high_t"] is None and nf["off_high"] is not None
    r, _ = ok(mkt, "explain_market", kind="index_move")
    assert r["brief"][1] == "⚡ Up on the day but fading from the high"            # no invented time
    assert r["brief"][2].startswith("Now: 13% of low→high range · -0.69% from high · last 30m")


# ── an index has no real VWAP ────────────────────────────────────────────────
def test_live_11_11_index_vwap_comes_from_the_futures_flag_only(fading, monkeypatch):
    """Live 11-11 15:02: 'above VWAP' with NIFTY at 4% of its range. The scanner's index vwap_v is the last
    price (index candles have no volume), so price vs vwap_v is noise. Only its futures-based above_vwap
    flag (NIFTY 50 / NIFTY BANK) is used; an index without that flag says nothing about VWAP."""
    data = fading.state.SECTOR_INTEL["data"]
    monkeypatch.setitem(data["NIFTY 50"], "vwap_v", data["NIFTY 50"]["price"])      # the live fallback: vwap = price
    for flag, word in ((False, "futures below VWAP"), (True, "futures above VWAP")):
        monkeypatch.setitem(data["NIFTY 50"], "above_vwap", flag)
        assert T.index_above_vwap(fading.ctx, "NIFTY 50") is flag
        r, text = ok(fading, "explain_market", kind="index_move")
        assert r["brief"][2].endswith("· " + word) and r["evidence"]["index"]["fut_above"] is flag
        assert r["evidence"]["market"]["nifty_above_vwap"] is flag
        assert ("NIFTY 50 +1.00%% (%s)" % word) in r["evidence"]["market"]["facts"][0]
        assert ("futures %s their VWAP" % word.split()[1]) in dict(r["sections"])["NIFTY 50"][0]
        assert " VWAP 2" not in text                                    # no made-up VWAP price anywhere
    monkeypatch.delitem(data["NIFTY 50"], "above_vwap")
    r, text = ok(fading, "explain_market", kind="index_move")
    assert T.index_above_vwap(fading.ctx, "NIFTY 50") is None and "VWAP" not in r["brief"][2]
    assert r["evidence"]["market"]["nifty_above_vwap"] is None and "VWAP" not in r["evidence"]["market"]["facts"][0]
    assert T.index_above_vwap(fading.ctx, "NIFTY IT") is None            # no futures map: never stated
    r, _ = ok(fading, "explain_market", kind="index_move", target="NIFTY MIDCAP 150")
    assert "VWAP" not in "\n".join(r["brief"][:3])


# ── index futures: windows named, IV gap explained ───────────────────────────
def test_futures_windows_never_read_as_one(fading):
    r, text = ok(fading, "explain_market", kind="index_move")
    sec = dict(r["sections"])["Futures NIFTY"]
    assert sec[0] == "Last 30 min: short build-up since 11:00, fresh"
    assert sec[1] == "Last 30 min: OI +1.22% · price -0.26%" and sec[2] == "Whole day: OI -1.04% since open"
    assert sec[-1] == "ATM IV: the scanner does not compute it for index futures (India VIX is NIFTY's)"
    assert "IV 1" not in "\n".join(r["brief"])


def test_flat_futures_oi_is_said_plainly(fading, monkeypatch):
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0, 0.0))
    r, _ = ok(fading, "explain_market", kind="index_move")
    assert [x for x in r["brief"] if x.startswith("Fut NIFTY")] == ["Fut NIFTY: 30m OI flat · day OI -1.04% · DTE 12"]
    assert not [x for x in r["brief"] if "futures adding" in x]


# ── the largest stocks ───────────────────────────────────────────────────────
def test_heavy_counts_match_the_scan(fading):
    r, _ = ok(fading, "explain_market", kind="index_move")
    hv, res = r["evidence"]["heavy"], fading.state.SESSION_RESULTS
    have = [s for s in E.HEAVY["NIFTY 50"] if s in res]
    assert [x["sym"] for x in hv["rows"]] == have and hv["n"] == len(have) == 8
    assert hv["up"] == sum(1 for s in have if res[s]["intra_pct"] > 0)
    m30 = T._window_moves(fading.ctx, 30)[0]
    assert hv["up30"] == sum(1 for s in have if m30[s] > 0) and hv["n30"] == 8
    best = max(have, key=lambda s: res[s]["intra_pct"])
    line = [x for x in r["brief"] if x.startswith("Heavy: ")][0]
    assert line.startswith("Heavy: %d/8 up · last 30m %d/8 up · ▲ %s " % (hv["up"], hv["up30"], best))
    assert "index point contribution (no index weights)" in r["not_checked"]      # still honest about weights


def test_heavy_only_where_the_list_exists(mkt):
    assert "heavy" not in [c[0] for c in E.plan("index_move", "NIFTY MIDCAP 150")]
    assert "heavy" in [c[0] for c in E.plan("index_move", "NIFTY BANK")]
    r, _ = ok(mkt, "explain_market", kind="index_move", target="NIFTY MIDCAP 150")
    assert "heavy" not in r["evidence"] and "heavy" not in " ".join(r["not_checked"])


def test_heavy_against_label(mkt):
    hv = {"n": 8, "up": 6, "n30": 8, "up30": 2}
    ev = {"index": {"name": "NIFTY 50", "price": 1.0, "pct": 1.0}, "heavy": hv}
    assert "HEAVY_AGAINST" in E.analyze("index_move", {"target": "NIFTY 50"}, ev)[0]
    ev["heavy"] = dict(hv, up30=3)
    assert "HEAVY_AGAINST" not in E.analyze("index_move", {"target": "NIFTY 50"}, ev)[0]
    ev["index"]["pct"] = -1.0                                   # falling index, 6 of 8 up in the last 30 min
    ev["heavy"] = dict(hv, up30=6)
    assert "HEAVY_AGAINST" in E.analyze("index_move", {"target": "NIFTY 50"}, ev)[0]


def test_too_few_heavy_stocks_is_named_missing(mkt, monkeypatch):
    monkeypatch.setitem(E.HEAVY, "NIFTY 50", ["HDFCBANK", "NOPE1", "NOPE2"])
    r, text = ok(mkt, "explain_market", kind="index_move")
    assert r["evidence"]["heavy"]["missing"] and "heavy (fewer than 3 of the largest" in text
    assert not [x for x in r["brief"] if x.startswith("Heavy")]


# ── "dominates" is honest ────────────────────────────────────────────────────
@pytest.mark.parametrize("c,want", [
    ({"long_buildup": 1, "short_buildup": 0, "short_covering": 0, "long_unwinding": 1, "neutral": 2}, None),
    ({"long_buildup": 7, "short_buildup": 1, "short_covering": 0, "long_unwinding": 4, "neutral": 13}, None),
    ({"long_buildup": 7, "short_buildup": 1, "short_covering": 0, "long_unwinding": 4, "neutral": 8}, "LONGS_ADDING"),
    ({"long_buildup": 0, "short_buildup": 3, "short_covering": 0, "long_unwinding": 0, "neutral": 1}, "SHORTS_ADDING"),
    ({"long_buildup": 0, "short_buildup": 0, "short_covering": 0, "long_unwinding": 0, "neutral": 5}, None), ({}, None)])
def test_oi_dominates_counts_flat_stocks_too(c, want):
    assert E._oi_label(c, "up") == want


# ── the sector answer ────────────────────────────────────────────────────────
def test_sector_answer_has_all_the_data(mkt):
    r, text = ok(mkt, "explain_market", kind="sector_move", target="psu bank")
    b, sx = r["brief"], r["evidence"]["sector"]
    st = sx["stats"]
    assert b[0] == "📈 PSU BANK up +1.80%% · 3/3 up · rank #%d of %d" % (sx["rank_now"], sx["rank_of"])
    assert b[1].startswith("⚡ Up on the day and at the day high · long build-up in 3/3 stocks")
    assert b[2].startswith("Now: ") and b[2].endswith("· %d/3 above VWAP" % st["above"])
    assert b[3].startswith("Stocks: PNB ") and b[4].startswith("OI 30m: long build-up 3/3")
    assert b[5].startswith("Volume: RVOL ") and "biggest push" in b[5]
    assert b[6].startswith("Compare: NIFTY ") and "NIFTY BANK" in b[6] and "rank held since" in b[6]
    assert sx["gap"] == pytest.approx(sx["index_pct"] - sx["nifty_pct"])
    assert st["fresh"]["long_buildup"] == sum(1 for x in st["rows"] if x["oi"] == "long_buildup" and x["fresh"])
    nums, times = AG.evidence_numbers(r["facts"])
    assert AG.bad_numbers("\n".join(b), nums, times) == []


@pytest.mark.parametrize("gap,want", [(2.44, "stronger than NIFTY by +2.44%"), (-1.0, "weaker than NIFTY by -1.00%"),
                                      (0.2, "in line with NIFTY"), (0.8, None), (-0.94, None)])
def test_sector_read_vs_nifty(gap, want):
    sx = {"stats": {"up": 2, "down": 1, "n": 3, "avg": 1.0, "above": 2, "oi_n": 0, "oi": {}, "fresh": {}},
          "index_pct": 1.0, "nifty_pct": 1.0 - gap, "gap": gap, "leaders": [("A", 10.0, 1.0)], "laggards": [],
          "range_pos": 50.0, "now": "holding", "off_high": -0.5, "off_low": 0.5, "l30": 0.0}
    b = E.brief("sector_move", {"target": "NIFTY IT"}, {"sector": sx}, [], None, None, "up", False)
    read = [x for x in b if x.startswith("⚡ ")][0]
    assert (want in read) if want else ("NIFTY" not in read)


def test_sector_table_lists_every_stock(mkt):
    r, _ = ok(mkt, "explain_market", kind="sector_move", target="NIFTY IT")
    secs = r["sections"]
    i = [t for t, _l in secs].index("IT stocks")
    assert secs[i + 1][0] == "" and r["mono"] == [i + 1]                        # the table follows its summary
    table = secs[i + 1][1]
    assert table[0] == "STOCK        DAY%  30m% OI   RVOL"
    rows = [x.split() for x in table[1:5]]
    res = mkt.state.SESSION_RESULTS
    assert [x[0] for x in rows] == sorted(["INFY", "TCS", "WIPRO", "HCLTECH"], key=lambda s: -res[s]["intra_pct"])
    assert all(float(x[1]) == pytest.approx(res[x[0]]["intra_pct"], abs=0.006) for x in rows)
    assert max(len(x) for x in table) <= 36                                    # fits a phone without wrapping
    html_ = E.render_details(r)
    assert "<pre>STOCK        DAY%" in html_ and "• fresh (new in the last 30 min)</pre>" in html_
    assert "<pre>" not in E.render_details(r, html=False) and "STOCK        DAY%" in E.render_details(r, html=False)


def test_stock_table_cap_and_marks():
    rows = [{"sym": "S%02d" % i, "pct": 20.0 - i, "m30": None if i == 0 else 0.001, "rvol": None if i == 0 else 1.25,
             "oi": "short_buildup" if i == 1 else "long_unwinding" if i == 2 else None, "fresh": i == 1}
            for i in range(20)]
    t = E.stock_table(rows)
    assert len(t) == 1 + 12 + 3 and t[7] == "…"                     # 6 top, a gap mark, 5 bottom, 3 legend lines
    assert [x.split()[0] for x in t[1:7]] == ["S00", "S01", "S02", "S03", "S04", "S05"]
    assert [x.split()[0] for x in t[8:13]] == ["S15", "S16", "S17", "S18", "S19"]
    assert t[1].split() == ["S00", "+20.00", "—", "—", "—"]
    assert t[2].split() == ["S01", "+19.00", "+0.0", "SB•", "1.2x"] and t[3].split()[3] == "LU"
    assert len(E.stock_table(rows[:12])) == 1 + 12 + 3 and "…" not in E.stock_table(rows[:12])


# ── design: one label per row, bold on the phone ─────────────────────────────
def test_brief_rows_have_bold_labels(fading):
    r, _ = ok(fading, "explain_market", kind="index_move")
    h = E.render_brief_html(r)
    for lab in ("Now", "Fut NIFTY", "Heavy", "Sectors", "Breadth"):
        assert "\n<b>%s</b> " % lab in h, lab
    assert h.startswith("<b>📈 NIFTY 50 up +1.00% · 25,250.0</b>\n⚡ Up on the day")
    assert "Now: " not in h and "\nNow: " in "\n".join(r["brief"])            # the colon lives in plain text only


@pytest.mark.parametrize("line,want", [
    ("Now: 19% of low→high range", "<b>Now</b> 19% of low→high range"), ("OI 30m: long build-up 3/5", "<b>OI 30m</b> long build-up 3/5"),
    ("ℹ️ You asked why it is down: it is up", None), ("Most of it at the open (+0.19%)", None),
    ("#2 IT +0.50% (day-rank 3)", None), ("12:30: nothing", None), ("⚡ Up on the day: x", None)])
def test_bold_label_only_on_label_rows(line, want):
    assert E._bold_label(line) == (want or line)


def test_agent_answer_keeps_the_new_rows(fading, tmp_path):
    from ai_llm import Route

    class Wr:
        def route(self, prompt):
            assert "BRIEF (already shown):" in prompt and "- ⚡ Up on the day but fading" in prompt
            return Route("write_view", {"insight": "Sellers are active in futures while 6 of 8 large stocks are still up."},
                         "fake", 10)
    ag = AG.MarketAgent(fading.ctx, Wr(), db_dir=str(tmp_path / "db"), alerts=None, log_dir=str(tmp_path / "l"))
    text, btn, rec = ag.ask("c", T.validate(fading.ctx, "explain_market", {"kind": "index_move"}), "why nifty down")
    assert "<b>Now</b> 13% of low→high range" in text and "🧠 <i>Sellers are active" in text and rec["attempts"] == 1
    det = ag.resume("c", btn[0][0][1].split(":")[1], "details")
    assert "<b>Futures NIFTY</b>" in det and "<pre>STOCK" in det              # the table survives the checkpoint


# ═════════════════════════════════════════════════════════════════════════════
# ai_watch: the five ready-made market alerts
# ═════════════════════════════════════════════════════════════════════════════
class Out:
    def __init__(self):
        self.sent = []

    def __call__(self, chat, text, buttons=None):
        self.sent.append((chat, text, buttons))
        return True


@pytest.fixture
def watch(tmp_path):
    return W.WatchStore(str(tmp_path / "data" / "ai_watch.json")), Out()


def _only(store, key):
    store.set_on("all", False)
    store.set_on(key, True)


def _run(m, store, out, at=0):
    return W.check(m.ctx, store, out, ["c1"], now=T._epoch(m.ctx) + at)


def test_futures_flip_fires_on_the_change_only(fading, watch, monkeypatch):
    store, out = watch
    _only(store, "fut")
    st = fading.state
    monkeypatch.setitem(st.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0, 0.0))            # flat: baseline
    assert _run(fading, store, out) == [] and out.sent == []
    monkeypatch.setitem(st.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0004, -2.2))        # fresh short build-up
    assert _run(fading, store, out, 60) == [("fut", "NIFTY 50", "short_buildup")]
    chat, text, btn = out.sent[0]
    assert chat == "c1" and isinstance(text, T.Rich)
    assert text.startswith("<b>🔔 NIFTY futures: fresh short build-up since 11:00</b>\n30m OI +1.22%, price -0.26%"
                           " · day OI -1.04%\nNIFTY 50 25,250.0 · +1.00% today · 13% of low→high range")
    assert "Futures flip · /watch to switch off" in text and btn == [[("🧠 Why", "ww:index_move:NIFTY 50")]]
    assert _run(fading, store, out, 120) == [] and len(out.sent) == 1                       # still true: no repeat


def test_old_build_up_is_not_a_flip(fading, watch, monkeypatch):
    """Short build-up that has been running for an hour is not 'fresh': nothing to announce."""
    store, out = watch
    _only(store, "fut")
    now = T._epoch(fading.ctx)
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0, 0.0))
    _run(fading, store, out)
    old = deque([{"ts": now - 60 * k, "raw": 1e7 * (1 + (90 - k) * 0.0004), "ltp": 25330 - (90 - k) * 2.2}
                 for k in range(90, -1, -3)])
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", old)
    c = T.index_futures(fading.ctx, "NIFTY 50", 30)
    assert c["type"] == "short_buildup" and c["fresh"] is False
    assert _run(fading, store, out, 60) == [] and store.d["last"]["fut|NIFTY 50"]["state"] == "none"


def test_already_true_at_start_is_a_baseline_not_a_ping(fading, watch):
    store, out = watch
    _only(store, "fut")
    assert _run(fading, store, out) == [] and out.sent == []                 # restart at 12:00 replays nothing
    assert store.d["last"]["fut|NIFTY 50"]["state"] == "short_buildup"


def test_cooldown_then_daily_cap(fading, watch, monkeypatch):
    store, out = watch
    _only(store, "fut")
    st = fading.state
    flat, short = _fut_hist(fading.ctx, 0.0, 0.0), _fut_hist(fading.ctx, 0.0004, -2.2)
    long_ = _fut_hist(fading.ctx, 0.0004, 2.2)

    def step(hist, at):
        monkeypatch.setitem(st.OI_HISTORY, "NIFTY", hist)
        return _run(fading, store, out, at)
    step(flat, 0)
    assert step(short, 60)                                          # ping 1
    assert step(long_, 120) == []                                   # a real flip, but inside the 30-min cooldown
    assert store.d["last"]["fut|NIFTY 50"]["state"] == "long_buildup"
    assert step(long_, 60 + W.COOLDOWN_S) == []                     # state did not change again: stays quiet
    assert step(short, 120 + W.COOLDOWN_S)                          # ping 2 (changed, cooldown over)
    t = 120 + W.COOLDOWN_S
    for i in range(3, W.MAX_PER_DAY + 1):                           # pings 3 .. MAX_PER_DAY
        t += W.COOLDOWN_S
        assert step(long_ if i % 2 else short, t), i
    assert store.d["count"]["fut"] == W.MAX_PER_DAY
    assert step(flat, t + 1) == [] and step(short, t + W.COOLDOWN_S + 5) == []          # daily cap reached
    assert len(out.sent) == W.MAX_PER_DAY


def test_stale_or_missing_futures_keep_the_last_state(fading, watch, monkeypatch):
    store, out = watch
    _only(store, "fut")
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0, 0.0))
    _run(fading, store, out)
    monkeypatch.delitem(fading.state.OI_HISTORY, "NIFTY")                     # no reading this minute
    monkeypatch.delitem(fading.state.OI_CACHE, "NIFTY")
    assert _run(fading, store, out, 60) == [] and store.d["last"]["fut|NIFTY 50"]["state"] == "none"


def test_fade_and_bounce(mkt, watch, monkeypatch):
    store, out = watch
    _only(store, "fade")
    put = lambda *a: monkeypatch.setitem(T._ISAMPLES, "NIFTY 50", _nifty_series(mkt.ctx, *a))  # noqa: E731
    put(0.9, 1.7, 1.65, 120)                                        # near the high: baseline
    assert _run(mkt, store, out) == []
    put(0.9, 1.7, 1.35, 55)                                         # -0.34% off the high: under the 0.40 bar
    assert _run(mkt, store, out, 60) == []
    put(0.9, 1.7, 1.0, 55)                                          # -0.69% and falling
    assert _run(mkt, store, out, 120) == [("fade", "NIFTY 50", "fading")]
    text = out.sent[-1][1]
    assert text.startswith("<b>🔔 NIFTY 50 fading: -0.69% from the day high (10:10)</b>\nlast 30m -0.26% · "
                           "NIFTY 50 25,250.0 · +1.00% today · 13% of low→high range")
    put(-0.9, -1.7, -1.0, 55)                                       # the mirror: off the low and rising
    assert _run(mkt, store, out, 180 + W.COOLDOWN_S) == [("fade", "NIFTY 50", "recovering")]
    assert out.sent[-1][1].startswith("<b>🔔 NIFTY 50 bouncing: +0.71% from the day low (10:10)</b>")


def test_fade_is_silent_in_the_first_15_minutes(mkt, watch, monkeypatch):
    store, out = watch
    monkeypatch.setitem(T._ISAMPLES, "NIFTY 50", _nifty_series(mkt.ctx, 0.9, 1.7, 1.0))
    monkeypatch.setattr(mkt.ctx, "_now", datetime(2026, 11, 6, 9, 29))
    assert W.detect_fade(mkt.ctx) == []
    monkeypatch.setattr(mkt.ctx, "_now", datetime(2026, 11, 6, 9, 30))
    assert [x[:2] for x in W.detect_fade(mkt.ctx)][0] == ("NIFTY 50", "fading")


def _breadth(m, monkeypatch, then, now_):
    """A/D history: `then` 30 min ago, `now_` at the last scan (the live counts come from the scan)."""
    ts = T._epoch(m.ctx)
    pts = deque([{"ts": ts - 31 * 60, "time": "10:59", "advance": then[0], "decline": then[1],
                  "ad_ratio": then[0] / float(then[1])},
                 {"ts": ts, "time": "11:30", "advance": now_[0], "decline": now_[1], "ad_ratio": now_[0] / float(now_[1])}])
    monkeypatch.setattr(m.state, "AD_BREADTH", pts)


def test_breadth_turn(mkt, watch, monkeypatch):
    store, out = watch
    _only(store, "breadth")
    _breadth(mkt, monkeypatch, (13, 11), (13, 11))
    assert _run(mkt, store, out) == [] and store.d["last"]["breadth|market"]["state"] == "none"
    _breadth(mkt, monkeypatch, (18, 6), (13, 11))
    assert _run(mkt, store, out, 60) == [("breadth", "market", "weakening")]
    assert out.sent[-1][1].startswith("<b>🔔 Breadth weakening: 13↑ / 11↓ now</b>\n10:59: 18↑ / 6↓ · NIFTY ")
    _breadth(mkt, monkeypatch, (13, 11), (13, 11))                                    # back to flat: quiet
    assert _run(mkt, store, out, 120 + W.COOLDOWN_S) == []
    _breadth(mkt, monkeypatch, (8, 16), (13, 11))
    assert _run(mkt, store, out, 180 + W.COOLDOWN_S) == [("breadth", "market", "improving")]


def _sector_oi(m, monkeypatch, syms, oi_step, px_step, older=0.0):
    """Give `syms` an OI history: `older` per minute before the last 30 min, then oi_step / px_step."""
    now = T._epoch(m.ctx)
    for s in syms:
        h = [{"ts": now - 60 * k,
              "raw": 1e6 * (1 + (60 - max(k, 30)) * older + (0 if k > 30 else (30 - k) * oi_step)),
              "ltp": 100.0 * (1 + (0 if k > 30 else (30 - k) * px_step))} for k in range(90, -1, -3)]
        monkeypatch.setitem(m.state.OI_HISTORY, s, deque(h))


def test_sector_oi_buildup(mkt, watch, monkeypatch):
    store, out = watch
    _only(store, "sector")
    it = ["INFY", "TCS", "WIPRO", "HCLTECH"]
    _sector_oi(mkt, monkeypatch, it, 0.0, 0.0)
    assert ("sector", "NIFTY IT", "short_buildup") not in _run(mkt, store, out)
    n0 = len(out.sent)
    _sector_oi(mkt, monkeypatch, it[:3], 0.001, -0.0005)                 # 3 of 4 fresh short build-up (75%)
    assert ("sector", "NIFTY IT", "short_buildup") in _run(mkt, store, out, 60)
    text, btn = [(t, b) for _c, t, b in out.sent[n0:] if t.startswith("<b>🔔 IT:")][0]
    assert text.startswith("<b>🔔 IT: short build-up in 3/4 stocks (3 fresh)</b>\nIT avg ")
    assert btn == [[("🧠 Why", "ww:sector_move:NIFTY IT")]]


@pytest.mark.parametrize("n_same,older,fires", [
    (2, 0.0, False),            # 2 of 4 = 50%: not "most"
    (3, 0.001, False),          # 3 of 4 but none fresh (OI was already rising before)
    (3, 0.0, True)])
def test_sector_oi_needs_most_stocks_and_fresh_ones(mkt, monkeypatch, n_same, older, fires):
    it = ["INFY", "TCS", "WIPRO", "HCLTECH"]
    _sector_oi(mkt, monkeypatch, it, 0.0, 0.0)
    _sector_oi(mkt, monkeypatch, it[:n_same], 0.001, -0.0005, older)
    got = {s: st for s, st, _t, _w in W.detect_sector(mkt.ctx)}
    assert (got["NIFTY IT"] == "short_buildup") is fires


def test_sector_names_are_html_safe(mkt, watch, monkeypatch):
    auto = ["MARUTI", "M&M", "TATAMOTORS"]
    monkeypatch.setitem(mkt.state.SESSION_RESULTS["M&M"], "intra_pct", 9.9)      # make it the sector's top name
    _sector_oi(mkt, monkeypatch, auto, 0.001, 0.0005)
    text = [t for s, _st, t, _w in W.detect_sector(mkt.ctx) if s == "NIFTY AUTO"][0]
    assert "M&amp;M" in text and "M&M" not in text


def _vix(m, monkeypatch, pct15):
    now = T._epoch(m.ctx)
    xs = deque([(now - 60 * k, 12.0 * (1 + (pct15 / 100.0) * (1 - min(k, 15) / 15.0)), 13.0, 11.5, None, 2.0)
                for k in range(40, -1, -1)], maxlen=T.ISAMPLE_KEEP)
    monkeypatch.setitem(T._ISAMPLES, "INDIA VIX", xs)


def test_vix_jump(mkt, watch, monkeypatch):
    store, out = watch
    _only(store, "vix")
    _vix(mkt, monkeypatch, 2.9)                                              # under the bar: baseline
    assert _run(mkt, store, out) == []
    _vix(mkt, monkeypatch, 3.0)
    assert _run(mkt, store, out, 60) == [("vix", "INDIA VIX", "up")]
    assert out.sent[-1][1].startswith("<b>🔔 VIX up +3.0% in 15 min</b>\nVIX 12.36 · ")
    assert out.sent[-1][2] == [[("🧠 Why", "ww:vix:")]]
    _vix(mkt, monkeypatch, -3.5)
    assert _run(mkt, store, out, 120 + W.COOLDOWN_S) == [("vix", "INDIA VIX", "down")]


def test_switched_off_watch_is_never_run(fading, watch, monkeypatch):
    store, out = watch
    store.set_on("all", False)
    monkeypatch.setitem(W.DETECTORS, "fut", lambda ctx: 1 / 0)
    assert _run(fading, store, out) == [] and store.d["last"] == {}


def test_one_broken_watch_does_not_stop_the_others(mkt, watch, monkeypatch, capsys):
    store, out = watch
    monkeypatch.setitem(W.DETECTORS, "fut", lambda ctx: 1 / 0)
    _vix(mkt, monkeypatch, 0.0)
    _run(mkt, store, out)
    _vix(mkt, monkeypatch, 4.0)
    assert ("vix", "INDIA VIX", "up") in _run(mkt, store, out, 60)
    assert "[AI watch] fut failed: ZeroDivisionError" in capsys.readouterr().out


def test_send_failure_does_not_raise(mkt, watch, monkeypatch):
    store, _out = watch
    _only(store, "vix")
    _vix(mkt, monkeypatch, 0.0)
    W.check(mkt.ctx, store, lambda *a: 1 / 0, ["c1"], now=T._epoch(mkt.ctx))
    _vix(mkt, monkeypatch, 4.0)
    assert W.check(mkt.ctx, store, lambda *a: 1 / 0, ["c1"], now=T._epoch(mkt.ctx) + 60) == [("vix", "INDIA VIX", "up")]


def test_store_survives_a_restart_and_resets_on_a_new_day(fading, tmp_path, monkeypatch):
    path = str(tmp_path / "w.json")
    a, out = W.WatchStore(path), Out()
    assert all(a.is_on(k) for k in W.KEYS)                                   # default: all five on
    a.set_on("vix", False)
    _only_keys = [k for k in W.KEYS if k != "fut"]
    for k in _only_keys:
        a.set_on(k, False)
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0, 0.0))
    _run(fading, a, out)
    monkeypatch.setitem(fading.state.OI_HISTORY, "NIFTY", _fut_hist(fading.ctx, 0.0004, -2.2))
    assert _run(fading, a, out, 60)
    b = W.WatchStore(path)                                                   # the app restarts
    assert not b.is_on("vix") and b.is_on("fut") and b.d["count"]["fut"] == 1
    assert _run(fading, b, out, 120) == [] and len(out.sent) == 1            # the same state does not re-fire
    monkeypatch.setattr(fading.ctx, "_today", "2026-11-09")
    b.roll("2026-11-09")
    assert b.d["last"] == {} and b.d["count"] == {} and not b.is_on("vix")   # states reset, switches kept
    assert json.load(open(path))["day"] == "2026-11-09" and not os.path.exists(path + ".tmp")


def test_corrupt_store_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("{not json")
    s = W.WatchStore(str(p))
    assert all(s.is_on(k) for k in W.KEYS) and s.d["last"] == {}
    p.write_text(json.dumps({"on": {"vix": False, "bogus": True}, "last": None}))
    s = W.WatchStore(str(p))
    assert not s.is_on("vix") and "bogus" not in s.d["on"] and s.d["last"] == {}


def test_card_and_buttons(watch):
    store, _ = watch
    text, rows = W.card(store)
    assert isinstance(text, T.Rich) and text.startswith("<b>🔔 Market alerts</b>")
    assert text.count("✅") == 5 and "⏸ <b>" not in text
    flat = [b for row in rows for b in row]
    assert [d for _l, d in flat] == ["wt:fut:off", "wt:fade:off", "wt:breadth:off", "wt:sector:off", "wt:vix:off",
                                     "wt:all:on", "wt:all:off"]
    assert all(len(d.encode()) <= 64 for _l, d in flat)                      # Telegram's callback_data limit
    text, rows = W.on_button(store, "wt:vix:off")
    assert "⏸ <b>VIX jump</b>" in text and rows[2][0] == ("▶️ VIX jump", "wt:vix:on") and not store.is_on("vix")
    text, _r = W.on_button(store, "wt:all:off")
    assert text.count("⏸ <b>") == 5
    W.on_button(store, "wt:bogus:on")
    W.on_button(store, "wt:vix:maybe")
    assert not any(store.is_on(k) for k in W.KEYS)                           # junk buttons change nothing


class TG:
    def __init__(self):
        self.sent = []

    def send(self, chat, text, buttons=None):
        self.sent.append((text, buttons))
        return True

    def answer_callback(self, *a):
        pass


def test_bot_watch_command_and_buttons(fading, watch, tmp_path):
    store, _ = watch
    tg = TG()
    bot = ai_chat.Bot(fading.ctx, None, tg, ["c"], str(tmp_path / "l"), watch=store)
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/watch"}})
    assert tg.sent[-1][0].startswith("<b>🔔 Market alerts</b>") and tg.sent[-1][1][-1][1][1] == "wt:all:off"
    bot.handle({"callback_query": {"id": "1", "data": "wt:fade:off", "message": {"chat": {"id": "c"}}}})
    assert "⏸ <b>Fade / bounce</b>" in tg.sent[-1][0] and not store.is_on("fade")
    bot.handle({"callback_query": {"id": "2", "data": "ww:index_move:NIFTY 50", "message": {"chat": {"id": "c"}}}})
    assert "NIFTY 50 up +1.00%" in tg.sent[-1][0] and "Fut NIFTY: short build-up" in tg.sent[-1][0]
    bot.handle({"callback_query": {"id": "3", "data": "ww:sector_move:NIFTY IT", "message": {"chat": {"id": "c"}}}})
    assert "IT " in tg.sent[-1][0].split("\n")[0]
    bot.handle({"callback_query": {"id": "4", "data": "ww:nonsense:x", "message": {"chat": {"id": "c"}}}})
    assert "kind" in tg.sent[-1][0].lower()                                  # a clear error, not a crash
    n = len(tg.sent)
    bot.handle({"callback_query": {"id": "5", "data": "wt:all:off", "message": {"chat": {"id": "stranger"}}}})
    assert len(tg.sent) == n and store.is_on("fut")                          # not on the allowlist: ignored


def test_bot_without_a_store_says_so(mkt, tmp_path):
    tg = TG()
    bot = ai_chat.Bot(mkt.ctx, None, tg, ["c"], str(tmp_path / "l"))
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/watch"}})
    assert tg.sent[-1][0] == "Market alerts are off (no store)."
    assert "/watch" in T.HELP_TEXT
