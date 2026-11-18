# -*- coding: utf-8 -*-
"""v19c: index futures (OI / IV / expiry) behind index questions, answers that follow the DATA's direction,
a lone sector name -> that sector's card, the Details layout, signal odds and the ML study."""
import json
import os
import random
import sqlite3
import sys
import types
from collections import deque

import pytest

import ai_chat
import ai_evidence as E
import ai_odds as O
import ai_tools as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAGA_SRC", ROOT))



def _public_schema():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fake_market", "schema.sql"),
              encoding="utf-8") as f:
        return f.read()

def ok(m, name, **args):
    r, err = T.run(m.ctx, name, args)
    assert err is None, err
    return r, T.format_reply(r)


@pytest.fixture
def fut(mkt, monkeypatch):
    """Give the fake market a NIFTY futures contract, like the live app: sector INDEX, OI flat then rising
    from 30 min ago with price up (fresh long build-up), expiry in 14 days. NO IV and NO skew: the live
    scanner skips indices for both (oi_engine: `if state.is_index(sym): continue`), seen live 11-11."""
    st, ctx = mkt.state, mkt.ctx
    now = T._epoch(ctx)
    monkeypatch.setitem(st.SECTOR_MAP, "NIFTY", "INDEX")
    hist = [{"ts": now - 60 * m, "raw": 1e7 * (1 + (0.0 if m > 30 else (30 - m) * 0.001)),
             "ltp": 25100 + (0 if m > 30 else (30 - m) * 3)} for m in range(90, -1, -3)]
    monkeypatch.setitem(st.OI_HISTORY, "NIFTY", deque(hist))
    monkeypatch.setitem(st.OI_CACHE, "NIFTY", {"oi_from_open": 2.4})
    monkeypatch.setattr(st, "kite_mgr", types.SimpleNamespace(_expiry_map={"NIFTY": "2026-11-20"}), raising=False)
    return mkt


@pytest.fixture
def fut_iv(fut, monkeypatch):
    """If the scanner ever does store index IV / skew, the card shows them."""
    st, now = fut.state, T._epoch(fut.ctx)
    monkeypatch.setitem(st.OI_CACHE, "NIFTY", {"oi_from_open": 2.4, "skew": {"slope": 0.4}})
    monkeypatch.setitem(st.IV_HISTORY, "NIFTY", deque([{"ts": now - 1800, "iv": 13.2}, {"ts": now - 60, "iv": 13.6}]))
    return fut


# ── index futures ───────────────────────────────────────────────────────────
def test_live_11_11_nifty_oi_reads_the_futures(fut_iv):
    """Live 11-11: 'nifty oi data' -> 'oi — (not for an index)'."""
    fut = fut_iv
    r, text = ok(fut, "live_value", symbols=["NIFTY"], fields=["oi"])
    assert "not for an index" not in text
    assert "Fut NIFTY · OI +2.40% since open · last 30 min long build-up since" in text
    assert "(fresh) · in those 30 min OI +3.00%, price +0.36%" in text            # both windows are named
    assert "ATM IV 13.6 (+0.4 in 30 min)" in text and "puts dearer" in text and "DTE 14" in text
    assert "NIFTY 50  price" in text and "day +" in text                     # the index price + day % stay


@pytest.mark.parametrize("said", ["NIFTY_FUT", "nifty fut", "Nifty Futures", "nifty-future"])
def test_live_11_11_future_words_resolve(fut, said):
    """Live 11-11: 'nifty future oi' -> 'Unknown index NIFTY_FUT'."""
    r, text = ok(fut, "live_value", symbols=[said], fields=["oi"])
    assert r["rows"][0]["sym"] == "NIFTY 50" and "Fut NIFTY" in text
    assert T.resolve_index(fut.ctx, "bn fut") == "NIFTY BANK"


def test_index_without_futures_data_says_so(mkt):
    r, text = ok(mkt, "live_value", symbols=["NIFTY"], fields=["oi"])
    assert "oi — (no futures data for this index)" in text


def test_index_futures_are_not_stocks(fut):
    r, _ = ok(fut, "oi_buildup", kind="all")
    assert "NIFTY" not in [x["sym"] for x in r["rows"]]
    assert "NIFTY" not in E._stock_results(fut.ctx)


def test_why_nifty_uses_its_futures(fut):
    r, text = ok(fut, "explain_market", kind="index_move")
    assert r["evidence"]["index_fut"]["type"] == "long_buildup" and "FUT_LONG_BUILDUP" in r["labels"]
    line = [x for x in r["brief"] if x.startswith("Fut NIFTY: ")][0]
    assert line.startswith("Fut NIFTY: long build-up since ")
    assert line.endswith("(fresh) · 30m OI +3.00%, price +0.36% · day OI +2.40% · DTE 14") and "IV" not in line
    assert ("Futures NIFTY", ) == tuple(t for t, _l in r["sections"] if t.startswith("Futures"))
    r2, _ = ok(fut, "explain_market", kind="index_move", target="NIFTY MIDCAP 150")
    assert "index_fut" not in r2["evidence"]                  # no futures on Midcap 150: check skipped, not "missing"


# ── direction follows the data ───────────────────────────────────────────────
def test_live_11_11_asked_down_but_market_up(mkt):
    """Live 11-11 09:58: 'why nifty down' while NIFTY +1.25% -> headline said down, listed fallers,
    called breadth narrow from the wrong side."""
    r, text = ok(mkt, "explain_market", kind="index_move", direction="down")
    assert r["direction"] == "up" and r["headline"].startswith("Why NIFTY 50 is up")
    assert r["brief"][1] == "ℹ️ You asked why it is down: it is up"
    assert dict(r["sections"]).get("Top gainers") and "Top fallers" not in dict(r["sections"])


# ── a lone sector name ───────────────────────────────────────────────────────
@pytest.mark.parametrize("q", ["fmcg sector", "FMCG", "how is fmcg today", "fmcg sector now"])
def test_live_11_11_lone_sector_gets_its_card(mkt, q):
    """Live 11-11: 'fmcg sector' -> all 14 sectors."""
    name, args, fixes = T.fix_route(mkt.ctx, q, "live_sector_rank", {"metric": "price"})
    assert (name, args["kind"], args["target"]) == ("explain_market", "sector_move", "NIFTY FMCG")
    r, text = ok(mkt, "explain_market", **args)
    assert r["brief"][0].startswith(("📈 FMCG", "📉 FMCG", "➖ FMCG"))


@pytest.mark.parametrize("q", ["top 5 sectors", "which sector is strongest", "fmcg vs it", "all sectors today",
                               "sector ranking"])
def test_rankings_stay_rankings(mkt, q):
    name, _a, _f = T.fix_route(mkt.ctx, q, "live_sector_rank", {"metric": "price"})
    assert name == "live_sector_rank"


# ── the Details layout ───────────────────────────────────────────────────────
def test_details_are_short_titled_blocks(fut):
    r, _ = ok(fut, "explain_market", kind="index_move")
    d = E.render_details(r)
    for title in ("<b>Market</b>", "<b>NIFTY 50</b>", "<b>When</b>", "<b>Futures NIFTY</b>",
                  "<b>Sectors (avg of stocks)</b>", "<b>VIX</b>", "<b>OI · last 30 min"):
        assert title in d, title
    body = [x for x in d.split("\n") if x and not x.startswith("<")]
    assert max(len(x) for x in body) <= 110                                   # no wall-of-text lines
    assert d.rstrip().endswith("Not advice.</i>")


# ── signal odds (lookup, not a model) ────────────────────────────────────────
def _odds_db(path, today="2026-11-06"):
    c = sqlite3.connect(path)
    c.executescript(_public_schema())
    rnd = random.Random(3)
    days = ["2026-%02d-%02d" % (mo, d) for mo in (9, 10, 11) for d in range(1, 28) if "2026-%02d-%02d" % (mo, d) < today]
    syms = ["S%02d" % i for i in range(40)]
    for day in days:
        for sym in rnd.sample(syms, 6):
            stage = rnd.choice(["SANKET", "SAGA", "SHARK", "AAHAT"])
            side = "LONG"                                  # AAHAT gets a side too: excluded by stage, not by luck
            t = "%02d:%02d:00" % (rnd.choice([9, 10, 11, 13, 14]), rnd.randint(16, 59))
            rv, pc = round(rnd.uniform(0.8, 4.0), 2), round(rnd.uniform(0.2, 1.6), 2)
            win = rv > 2.0 and rnd.random() < 0.6
            c.execute("INSERT INTO signals(day,t,sym,stage,side,rvol,pace,nifty_pct,sector) VALUES(?,?,?,?,?,?,?,?,?)",
                      (day, t, sym, stage, side, rv, pc, rnd.uniform(-1, 1), "IT"))
            c.execute("INSERT INTO outcomes(day,sym,stage,t,side,m15,m30,m60,mfe,mae,res_10,bars_n,src) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (day, sym, stage, t, side, 0.1, 0.3 if win else -0.2, 0.4,
                                                            1.0 if win else 0.3, -0.3, "WIN" if win else
                                                            rnd.choice(["LOSS", "FLAT"]), 60, "minute"))
    # today: SBIN SANKET LONG at 11:05, plus a same-day look-alike that must NOT count
    c.execute("INSERT INTO signals(day,t,sym,stage,side,rvol,pace,nifty_pct,sector) VALUES(?,?,?,?,?,?,?,?,?)",
              (today, "11:05:00", "SBIN", "SANKET", "LONG", 2.5, 0.8, 0.5, "PSU"))
    c.execute("INSERT INTO outcomes(day,sym,stage,t,side,res_10,bars_n) VALUES(?,?,?,?,?,?,?)",
              (today, "SBIN", "SANKET", "11:05:00", "LONG", "WIN", 60))
    c.commit()
    c.close()


@pytest.fixture
def odds_ctx(mkt, tmp_path, monkeypatch):
    db = str(tmp_path / "saga_obs.db")
    _odds_db(db, mkt.ctx.today)
    monkeypatch.setattr(mkt.ctx, "saga_db", db)
    return mkt.ctx


def test_wilson_range():
    lo, hi = O.wilson(6, 23)
    assert lo == pytest.approx(0.125, abs=0.01) and hi == pytest.approx(0.465, abs=0.01)
    assert O.wilson(0, 0) == (0.0, 0.0)


def test_odds_card_for_todays_signal(odds_ctx):
    text, btn, info = O.card(odds_ctx, "SBIN")
    ev = O.load_events(odds_ctx.saga_db, before_day=odds_ctx.today)
    assert all(e["day"] < odds_ctx.today and e["stage"] in O.STAGES and e["side"] for e in ev)   # no today
    assert ev and not any(e["stage"] == "AAHAT" for e in ev)                  # AAHAT is noise, even with a side
    sig = O.latest_signal(odds_ctx.saga_db, "SBIN", odds_ctx.today)
    cases, _rule = O.match(sig, ev, 0)
    s = O.stats(cases)
    assert info["n"] == s["n"] and "Similar past signals: <b>%d</b>" % s["n"] in text
    assert text.startswith("<b>🎯 SBIN · SANKET LONG · 11:05</b>") and "model not used" in text
    assert [b[0] for b in btn[0]] == ["📊 Show the %d" % s["n"], "🔁 Loosen match"]
    assert "all SANKET LONG:" in text and "likely" in text


def test_loosen_only_widens(odds_ctx):
    sig = O.latest_signal(odds_ctx.saga_db, "SBIN", odds_ctx.today)
    ev = O.load_events(odds_ctx.saga_db, before_day=odds_ctx.today)
    ns = [len(O.match(sig, ev, lv)[0]) for lv in range(O.LOOSEN_MAX + 1)]
    assert ns == sorted(ns) and ns[-1] > ns[0]
    text, btn = O.on_button(odds_ctx, "od:SBIN|SANKET|LONG:%d:card" % O.LOOSEN_MAX)
    assert "Loosen" not in str(btn)                          # last level: no further loosen button


def test_show_cases_newest_first(odds_ctx):
    text, _ = O.on_button(odds_ctx, "od:SBIN|SANKET|LONG:2:show")
    lines = [x for x in text.split("\n")[1:] if x[:2].isdigit()]
    days = [(x[3:5], x[0:2]) for x in lines]
    assert days == sorted(days, reverse=True) and "Match: same stage + side (SANKET LONG)" in text


def test_odds_without_signal_and_stage_side(odds_ctx):
    text, _b, _i = O.card(odds_ctx, "TCS")
    assert "TCS has no SANKET / SAGA / SHARK signal today" in text
    text, btn, info = O.card(odds_ctx, "SAGA LONG")
    assert text.startswith("<b>🎯 SAGA LONG</b>") and info["n"] > 0


def test_bot_odds_never_calls_the_model(odds_ctx, tmp_path):
    class R:
        def route(self, p):
            raise AssertionError("LLM called for odds")

    class TG:
        def __init__(self):
            self.sent = []

        def send(self, chat, text, buttons=None):
            self.sent.append((text, buttons))
            return True

        def answer_callback(self, *a):
            pass
    tg = TG()
    bot = ai_chat.Bot(odds_ctx, R(), tg, ["c"], str(tmp_path / "l"))
    bot.handle({"message": {"chat": {"id": "c"}, "text": "odds SBIN"}})
    assert tg.sent[-1][0].startswith("<b>🎯 SBIN")
    bot.handle({"callback_query": {"id": "1", "data": tg.sent[-1][1][0][0][1], "message": {"chat": {"id": "c"}}}})
    assert "similar SANKET LONG signals" in tg.sent[-1][0]
    bot.handle({"message": {"chat": {"id": "c"}, "text": "/mlreport"}})
    assert "No ML study yet" in tg.sent[-1][0]


# ── the ML study ─────────────────────────────────────────────────────────────
def _events(n, signal, seed=1):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        day = "2026-%02d-%02d" % (6 + i * 6 // n, 1 + (i * 37) % 27)
        rv = rnd.uniform(0.5, 4.0)
        win = (rv > 2.2 and rnd.random() < 0.85) if signal else (rnd.random() < 0.3)
        out.append({"day": day, "t": "%02d:%02d:00" % (rnd.choice([9, 10, 11, 13]), rnd.randint(16, 59)),
                    "sym": "S%d" % (i % 50), "stage": rnd.choice(O.STAGES), "side": "LONG", "rvol": rv,
                    "pace": rnd.uniform(0.2, 1.5), "nifty_pct": rnd.uniform(-1, 1), "dp15": None, "day_pct": None,
                    "res": "WIN" if win else "LOSS", "m30": 0.0, "mfe": 0.0, "mae": 0.0, "sector": "IT"})
    return out


def _study():
    sys.path.insert(0, ROOT)
    import ml_study
    return ml_study


def test_ml_study_no_signal_is_not_used():
    r = _study().study(_events(700, signal=False))
    assert r["verdict"] == "no" and r["folds"] >= 3 and r["gain_lo"] < 0 < r["gain_hi"] + 1e-9


def test_ml_study_real_signal_is_found():
    r = _study().study(_events(1500, signal=True))
    assert r["verdict"] == "use" and r["auc_model"] > r["auc_lookup"] + 0.1 and r["top_features"][0] == "rvol"


def test_ml_study_features_never_use_the_future():
    ms = _study()
    assert not {"m15", "m30", "m60", "mfe", "mae", "res"} & set(ms.FEATURES)
    e = _events(1, False)[0]
    e2 = dict(e, m30=99.0, mfe=99.0, mae=-99.0, res="WIN")
    assert ms.features(e) == ms.features(e2) or all(a == b or (a != a and b != b)
                                                    for a, b in zip(ms.features(e), ms.features(e2)))


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rnd = random.Random(5)
    y = [rnd.random() < 0.3 for _ in range(300)]
    s = [round(rnd.random(), 1) for _ in range(300)]                          # ties on purpose
    assert _study().auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_mlreport_renders(odds_ctx, tmp_path):
    rep = {"run_at": "11-11-2026 16:00", "n": 302, "days": 55, "stages": list(O.STAGES), "label": "WIN by res_10",
           "folds": 3, "n_test": 180, "auc_model": 0.55, "auc_lookup": 0.53, "gain": 0.02, "gain_lo": -0.05,
           "gain_hi": 0.09, "verdict": "no", "top_features": ["rvol", "minute"], "model": "LightGBM", "next_at": 604}
    with open(os.path.join(os.path.dirname(odds_ctx.saga_db), "ml_study.json"), "w") as f:
        json.dump(rep, f)
    t = O.mlreport(odds_ctx)
    assert "inside the noise: model NOT used" in t and "gain +0.020 (95% range -0.050 to +0.090)" in t


def test_ml_verdict_needs_the_whole_range_above_zero():
    ms = _study()
    assert ms.decide(0.60, 0.50, -0.01, 500) == "no"          # better on average, but inside the noise
    assert ms.decide(0.60, 0.50, 0.02, 200) == "no"           # too few test events
    assert ms.decide(0.60, 0.50, 0.02, 400) == "use"
