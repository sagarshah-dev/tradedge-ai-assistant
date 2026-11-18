# -*- coding: utf-8 -*-
"""v19: OI types + "fresh", text fields in the condition language, broader indices + crude,
and the evidence checks behind WHY questions. Runs on fake_market (no network, no LLM)."""
import types
from collections import deque
from datetime import date, datetime, timedelta

import pytest

import ai_alerts as A
import ai_evidence as E
import ai_extra as X
import ai_tools as T


def ok(m, name, **args):
    r, err = T.run(m.ctx, name, args)
    assert err is None, err
    return r, T.format_reply(r)


# ── OI types ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("oi,px,want", [
    (2.0, 0.5, "long_buildup"), (2.0, -0.5, "short_buildup"), (-2.0, 0.5, "short_covering"),
    (-2.0, -0.5, "long_unwinding"), (0.3, 1.0, "neutral"), (2.0, 0.05, "neutral"), (None, 1.0, None)])
def test_oi_type_truth_table(oi, px, want):
    assert T.oi_type_of(oi, px) == want


def _oi_state(mp, ctx, sym, readings):
    """readings: [(minutes ago, OI, futures price)]. monkeypatch: the market is shared by the module."""
    base = T._epoch(ctx)
    mp.setitem(ctx.state.OI_HISTORY, sym, deque([{"ts": base - 60 * m, "raw": oi, "ltp": px}
                                                for m, oi, px in readings]))


def test_oi_types_fresh_and_started(mkt, monkeypatch):
    ctx = mkt.ctx
    # flat OI for an hour, then rising from 25 min ago with price up: a FRESH long build-up
    _oi_state(monkeypatch, ctx, "SBIN", [(60, 1000, 800), (45, 1001, 800), (30, 1000, 801), (25, 999, 801), (15, 1010, 805),
                            (3, 1030, 809), (0, 1040, 810)])
    # OI was already rising in the 30 min before: build-up, but NOT fresh
    _oi_state(monkeypatch, ctx, "PNB", [(60, 900, 100), (30, 950, 101), (15, 980, 102), (0, 1000, 103)])
    x = T.oi_types(ctx, 30)[0]
    assert x["SBIN"]["type"] == "long_buildup" and x["SBIN"]["fresh"] is True
    assert x["SBIN"]["started"] == (datetime.fromtimestamp(T._epoch(ctx) - 25 * 60)).strftime("%H:%M")
    assert x["PNB"]["type"] == "long_buildup" and x["PNB"]["fresh"] is False


def test_oi_types_restart_gap_is_skipped_not_guessed(mkt, monkeypatch):
    ctx = mkt.ctx
    _oi_state(monkeypatch, ctx, "SBIN", [(90, 1000, 800), (2, 1100, 820), (0, 1110, 821)])     # hole: app was off
    types_, skipped = T.oi_types(ctx, 30)
    assert "SBIN" not in types_ and skipped >= 1


def test_oi_buildup_tool_kind_fresh_sector_and_reply(mkt):
    r, text = ok(mkt, "oi_buildup", kind="buildup")
    assert r["rows"] and all(x["type"] in ("long_buildup", "short_buildup") for x in r["rows"])
    first = r["rows"][0]
    assert first["price"] is not None and first["pct"] is not None                 # rule: price + day % on every row
    assert ("%s" % T._px2(first["price"])) in text and "day " in text
    assert "All F&O, last 30 min: long build-up" in text
    rf, _ = ok(mkt, "oi_buildup", kind="long_buildup", fresh=True)
    assert all(x["fresh"] for x in rf["rows"]) and len(rf["rows"]) < len(r["rows"])
    rs, _ = ok(mkt, "oi_buildup", kind="all", sector="PSU BANK")
    assert all(mkt.ctx.state.SECTOR_MAP[x["sym"]] == "NIFTY PSU BANK" for x in rs["rows"])
    r2, err = T.run(mkt.ctx, "oi_buildup", {"kind": "short_buildup", "sector": "IT"})
    assert r2 is None and "Counts:" in err                                          # empty answer says what exists


# ── text fields in the condition language ────────────────────────────────────
def test_clean_conditions_text_ops():
    c = T.clean_conditions([{"left": "short_state", "op": "contains", "text": "short-breaking"}])[0]
    assert c == {"left": "short_state", "op": "contains", "right": None, "value": None, "pct": 0.0,
                 "text": "SHORT BREAKING"}
    assert T.clean_conditions([{"left": "tier", "op": "=", "text": "super"}])[0]["op"] == "is"
    assert T.clean_conditions([{"left": "tier", "op": "!=", "text": "super"}])[0]["op"] == "is_not"
    assert T.clean_conditions([{"left": "stage", "op": "is", "right": "setup"}])[0]["text"] == "SETUP"
    for bad, msg in (([{"left": "tier", "op": ">", "text": "x"}], "is / is_not / contains"),
                     ([{"left": "tier", "op": "is"}], "which word"),
                     ([{"left": "rvol", "op": "contains", "value": 2}], "is a number")):
        with pytest.raises(T.ToolError) as e:
            T.clean_conditions(bad)
        assert msg in str(e.value)


@pytest.mark.parametrize("field", sorted(T.TEXT_FIELDS))
def test_every_text_field_answers_and_matches(mkt, field):
    """Coverage audit: every text field gives at least one hit on the fake session, and every
    hit really carries that word (generated from TEXT_FIELDS: a new field is tested automatically)."""
    vals = {s: f.get(field) for s, f in T.stock_fields(mkt.ctx, {field}).items()}
    words = sorted({v for v in vals.values() if v})
    assert words, "%s: no stock has a value" % field
    word = words[0]
    r, err = T.run(mkt.ctx, "live_screen", {"conditions": [{"left": field, "op": "is", "text": word}], "limit": 20})
    assert err is None, err
    assert r["n"] == sum(1 for v in vals.values() if v == word)
    assert all(r["fields"][s][field] == word for s, _ in r["rows"])
    rn, _ = T.run(mkt.ctx, "live_screen", {"conditions": [{"left": field, "op": "is_not", "text": word}], "limit": 20})
    if rn:
        assert all(rn["fields"][s].get(field) != word for s, _ in rn["rows"])


def test_short_breaking_and_numbers_together(mkt):
    r, text = ok(mkt, "live_screen", conditions=[{"left": "short_state", "op": "contains", "text": "breaking"},
                                                 {"left": "pct", "op": "<", "value": -1.5}])
    assert r["n"] >= 1 and all(r["fields"][s]["pct"] < -1.5 for s, _ in r["rows"])
    assert "short_state has BREAKING and pct < -1.5" in text


def test_text_empty_answer_lists_words_now(mkt):
    r, err = T.run(mkt.ctx, "live_screen", {"conditions": [{"left": "state", "op": "is", "text": "nonsense"}]})
    assert r is None and "state right now: BREAKING" in err


def test_alert_on_text_condition_fires_once(mkt, tmp_path):
    rule, missing, _ = A.validate_rule(mkt.ctx, {"target": "stocks", "metric": "screen", "conditions": [
        {"left": "short_state", "op": "contains", "text": "BREAKING"}]}, "alert me when any stock enters short breaking")
    assert not missing and rule["conditions"][0]["text"] == "BREAKING"
    store = A.AlertStore(str(tmp_path / "a.json"))
    store.add("c", rule, "x", mkt.ctx.today)
    sent = []
    now = mkt.now.timestamp()
    A.check_alerts(mkt.ctx, store, lambda chat, text: sent.append(text) or True, now=now, day=mkt.ctx.today)
    A.check_alerts(mkt.ctx, store, lambda chat, text: sent.append(text) or True, now=now + 60, day=mkt.ctx.today)
    assert len(sent) == 1 and "short_state has BREAKING" in sent[0]


# ── broader indices + crude ──────────────────────────────────────────────────
NSE_SAMPLE = {"data": [
    {"index": "NIFTY MIDCAP 150", "indexSymbol": "NIFTY MIDCAP 150", "last": 21000.5, "percentChange": -1.1,
     "open": 21200, "high": 21250, "low": 20980, "previousClose": 21233.0},
    {"index": "NIFTY SMALLCAP 250", "indexSymbol": "NIFTY SMLCAP 250", "last": 17000, "previousClose": 17170,
     "high": 17200, "low": 16990},
    {"index": "NIFTY MICROCAP 250", "indexSymbol": "NIFTY MICROCAP250", "last": 23000, "percentChange": 0.4},
    {"index": "NIFTY 50", "indexSymbol": "NIFTY 50", "last": 25100}]}


def test_parse_nse_names_and_pct():
    got = X.parse_nse(NSE_SAMPLE, 1000.0)
    assert set(got) == {"NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250"}
    assert got["NIFTY MIDCAP 150"]["pct"] == -1.1
    assert got["NIFTY SMALLCAP 250"]["pct"] == pytest.approx((17000 / 17170 - 1) * 100)   # computed when missing
    assert got["NIFTY MICROCAP 250"]["day_high"] == 23000                                 # no high given: price


def test_crude_contract_rolls_before_expiry():
    today = date(2026, 11, 17)
    inst = [{"name": "CRUDEOIL", "instrument_type": "FUT", "expiry": date(2026, 11, 18), "tradingsymbol": "NOV"},
            {"name": "CRUDEOIL", "instrument_type": "FUT", "expiry": date(2026, 12, 17), "tradingsymbol": "DEC"},
            {"name": "CRUDEOILM", "instrument_type": "FUT", "expiry": date(2026, 11, 25), "tradingsymbol": "MINI"},
            {"name": "CRUDEOIL", "instrument_type": "CE", "expiry": date(2026, 11, 25), "tradingsymbol": "OPT"}]
    assert X.pick_contract(inst, today)["tradingsymbol"] == "DEC"                 # 1 day left: rolled
    assert X.pick_contract(inst, date(2026, 11, 10))["tradingsymbol"] == "NOV"
    assert X.pick_contract([], today) is None


def test_crude_quote_and_poll(monkeypatch, mkt):
    q = {"last_price": 5400.0, "ohlc": {"open": 5500, "high": 5520, "low": 5380, "close": 5500.0}, "oi": 11000,
         "average_price": 5450}
    r = X.parse_quote(q, "CRUDEOIL26DECFUT", 1.0, oi_first=10000)
    assert r["pct"] == pytest.approx(-1.818, abs=1e-3) and r["oi_pct"] == pytest.approx(10.0)

    class Kite:
        def instruments(self, seg):
            assert seg == "MCX"
            return [{"name": "CRUDEOIL", "instrument_type": "FUT", "expiry": date.today() + timedelta(days=20),
                     "tradingsymbol": "CRUDEOILX"}]

        def quote(self, keys):
            return {keys[0]: q}
    st = types.SimpleNamespace(kite_mgr=types.SimpleNamespace(kite=Kite()))
    monkeypatch.setattr(X, "_crude", {"day": None, "inst": None, "oi_first": None})
    assert X.poll_crude(st) is True and T.EXTRA_INDICES["CRUDEOIL"]["price"] == 5400.0
    assert "CRUDEOILX" in X.status_line()
    assert X.poll_crude(types.SimpleNamespace()) is False and "Kite not logged in" in X.status_line()


def test_extra_indices_are_sampled_and_answerable(mkt):
    now = T._epoch(mkt.ctx)
    T.EXTRA_INDICES.update(X.parse_nse(NSE_SAMPLE, now))
    T.EXTRA_INDICES["OLD"] = {"price": 1.0, "ts": now - 3600}                  # stale: must not be sampled
    T.sample_once(mkt.ctx, now)
    assert "NIFTY MIDCAP 150" in T._ISAMPLES and "OLD" not in T._ISAMPLES
    r, text = ok(mkt, "live_value", symbols=["midcap"], fields=["price", "low"])
    assert r["rows"][0]["sym"] == "NIFTY MIDCAP 150" and "low 20980" in text.replace(",", "")
    assert T.resolve_index(mkt.ctx, "microcap") == "NIFTY MICROCAP 250"
    assert "NIFTY MIDCAP 150" in T.MARKET_INDICES                                 # 'any sector' skips it


def test_poll_nse_error_is_reported_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("NSE 403")
    assert X.poll_nse(fetch=boom) == 0 and "NSE 403" in X.status_line()


# ── evidence: plans, checks, labels ──────────────────────────────────────────
PLANS = {   # the agent's trajectory spec: which checks each question kind must run
    ("index_move", "NIFTY 50", None): ["market", "index", "path", "index_fut", "heavy", "sectors", "breadth", "vix",
                                       "oi_mix", "crude", "broader", "movers"],
    ("index_move", "NIFTY MIDCAP 150", "NIFTY 50"): ["market", "index", "path", "sectors", "breadth", "vix",
                                                     "oi_mix", "crude", "broader"],
    ("sector_move", "NIFTY PSU BANK", None): ["market", "sector", "sector_path", "sectors", "vix"],
    ("strongest_sectors", None, None): ["market", "strength", "broader", "vix"],
    ("rotation", None, None): ["market", "rotation", "vix"],
    ("stock_move", "SBIN", None): ["market", "stock", "sectors"],
    ("vix", "INDIA VIX", None): ["market", "vix", "index", "index_fut", "path", "breadth", "oi_mix"],
    ("crude", "CRUDEOIL", None): ["market", "crude", "sectors", "index"],
}


@pytest.mark.parametrize("key", list(PLANS))
def test_plan_per_kind(key):
    kind, target, compare = key
    assert [c[0] for c in E.plan(kind, target, compare)] == PLANS[key]


@pytest.mark.parametrize("args", [
    {"kind": "index_move"}, {"kind": "index_move", "target": "NIFTY BANK", "direction": "down"},
    {"kind": "sector_move", "target": "psu bank"}, {"kind": "strongest_sectors"},
    {"kind": "strongest_sectors", "direction": "down"}, {"kind": "rotation"},
    {"kind": "stock_move", "target": "NYKAA"}, {"kind": "vix"}, {"kind": "crude"}])
def test_every_kind_answers_without_a_model(mkt, args):
    r, text = ok(mkt, "explain_market", **args)
    assert text.startswith(r["brief"][0]) and r["headline"].upper() in text        # brief first, then details
    assert "Not advice." in text and "Not checked: news" in text
    assert 1 <= len(r["brief"]) <= 9 and r["facts"] and not r.get("view")          # one phone screen
    assert all(len(x) <= 125 for x in r["brief"]), [x for x in r["brief"] if len(x) > 125]
    short = E.render_brief_html(r)
    assert short.startswith("<b>") and "not advice" in short and "Not checked" not in short
    for k, v in r["evidence"].items():                       # a missing check is NAMED, never silent
        if v.get("missing"):
            assert k.replace("_", " ") in text


def test_missing_sources_are_listed(mkt):
    r, text = ok(mkt, "explain_market", kind="index_move")
    assert "crude (crude not available" in text and "broader (broader indices not fetched yet" in text
    assert "index point contribution (no index weights)" in text


def test_no_live_data_at_all_is_an_error(monkeypatch, mkt):
    monkeypatch.setattr(E, "run_check", lambda ctx, n, kw: E._missing("x"))
    r, err = T.run(mkt.ctx, "explain_market", {"kind": "index_move"})
    assert r is None and "No live data" in err


def test_labels_by_rule():
    ev = {"index": {"pct": -1.0, "gap": -0.8}, "path": {"fall15": (-0.6, 0, 0)}, "breadth": {"share_up": 0.2},
          "sectors": {"rows": [{"avg": -2.5}, {"avg": -0.3}, {"avg": 0.4}]}, "vix": {"pct": 6.0},
          "oi_mix": {"counts": {"short_buildup": 30, "long_buildup": 5, "short_covering": 5, "long_unwinding": 10}},
          "broader": {"gaps": {"MID": -0.9, "SMALL": -1.2}}, "crude": {"pct": -2.0}}
    labels, _c, _f, _l, d = E.analyze("index_move", {}, ev)
    assert d == "down"
    assert labels == ["GAP_LED", "SUDDEN", "BROAD", "SECTOR_LED", "FEAR_RISING", "SHORTS_ADDING", "SMALL_WEAKER",
                      "CRUDE_BIG_MOVE"]
    ev["index"]["gap"] = 0.1
    ev["breadth"]["share_up"] = 0.5
    ev["oi_mix"]["counts"] = {"short_buildup": 10, "long_buildup": 10, "short_covering": 10, "long_unwinding": 10}
    labels = E.analyze("index_move", {}, ev)[0]
    assert "INTRADAY" in labels and "NARROW" in labels and not any("SHORTS" in x for x in labels)   # no OI majority


def test_sector_and_stock_labels():
    ev = {"sector": {"stats": {"avg": 1.8, "n": 12, "up": 11, "down": 1,
                               "oi": {"long_buildup": 8, "short_buildup": 0, "short_covering": 2, "long_unwinding": 0}},
                     "index_pct": 1.8, "nifty_pct": 0.2}, "sector_path": {"rise15": (1.1, 0, 0)}}
    labels, *_ = E.analyze("sector_move", {}, ev)
    assert labels == ["BROAD", "SECTOR_SPECIFIC", "LONGS_ADDING", "SUDDEN"]
    labels, *_ = E.analyze("stock_move", {}, {"stock": {"pct": 3.0, "sector_avg": 0.5, "oi_type": "long_buildup",
                                                         "fresh": True}})
    assert labels == ["STOCK_SPECIFIC", "FRESH_LONGS"]


def test_strength_weights_are_yours_and_confidence(monkeypatch, mkt):
    r0 = E.check_strength(mkt.ctx)
    assert r0["candidates"] and 0 <= r0["candidates"][0]["score"] <= 1 and r0["agree"][1] >= 5
    base = [c["name"] for c in r0["candidates"]]
    monkeypatch.setattr(E, "_cfg", lambda n, d=None: "day=0,breadth=0,vwap=0,persist=0,participation=0,oi=0,room=5"
                        if n == "AI_STRENGTH_WEIGHTS" else d)
    r1 = E.check_strength(mkt.ctx)
    assert "weights yours from config" in r1["facts"][0]
    pace_order = sorted(r1["scores"], key=lambda s: r1["pct"]["room"].get(s, -1), reverse=True)
    assert r1["candidates"][0]["sector"] == pace_order[0]                    # only 'room' counts now
    assert [c["name"] for c in r1["candidates"]] != base or len(base) < 2
    assert E._conf(6, 7)["level"] == "HIGH" and E._conf(4, 7)["level"] == "MEDIUM" and E._conf(3, 7)["level"] == "LOW"


def test_rotation_lean_follows_market_context():
    rows = [{"name": "AUTO", "label": "EMERGING", "change": 0.6, "fut": 2.4, "lb": 3, "oi_n": 3, "up": 3, "n": 3},
            {"name": "METAL", "label": "STEADY_LEADER", "change": 0.1, "fut": 0.9, "lb": 2, "oi_n": 9, "up": 8, "n": 9}]
    ev = {"rotation": {"rows": rows, "emerging": ["AUTO"], "steady": ["METAL"]}, "market": {"risk": "risk_on"}}
    assert E.analyze("rotation", {}, ev)[3] == "AUTO"
    ev["market"]["risk"] = "risk_off"
    assert E.analyze("rotation", {}, ev)[3] == "METAL"
    ev["market"]["risk"] = "mixed"
    lab, cands, conf, lean, _d = E.analyze("rotation", {}, ev)
    assert lean == "AUTO" and conf["n"] == 5 and "ROTATION_TO_EMERGING" in lab


def test_explain_validation(mkt):
    v = T.validate(mkt.ctx, "explain_market", {"kind": "sector_move", "target": "psu bank", "direction": "UP"})
    assert v["target"] == "NIFTY PSU BANK" and v["direction"] == "up" and v["window_min"] == 30
    v = T.validate(mkt.ctx, "explain_market", {"kind": "index_move", "target": "midcap", "compare": "nifty"})
    assert v["target"] == "NIFTY MIDCAP 150" or v["target"] in T.index_names(mkt.ctx)
    assert T.validate(mkt.ctx, "explain_market", {"kind": "vix", "target": "x"})["target"] == "INDIA VIX"
    with pytest.raises(T.ToolError):
        T.validate(mkt.ctx, "explain_market", {"kind": "sector_move"})
    with pytest.raises(T.ToolError):
        T.validate(mkt.ctx, "explain_market", {"kind": "stock_move", "target": "NOTASTOCK"})


def test_explain_numbers_come_only_from_code(mkt):
    """No model involved: every number in the reply is in a fact line, a label or the footer."""
    import ai_agent as AG
    for args in ({"kind": "sector_move", "target": "PSU BANK"}, {"kind": "index_move"}, {"kind": "rotation"},
                 {"kind": "strongest_sectors"}, {"kind": "stock_move", "target": "SBIN"}, {"kind": "vix"}):
        r, text = ok(mkt, "explain_market", **args)
        nums, times = AG.evidence_numbers(r["facts"], r["headline"])
        assert AG.bad_numbers("\n".join(r["brief"]), nums, times) == [], args    # brief adds no number


def test_live_09_11_flat_nifty_with_sector_split():
    """09-11 15:14 live answer: NIFTY +0.08% was called 'up' and the real story (PSU BK +4.70% vs IT -1.02%)
    sat on line 7. Now: 'flat' in line 1, the split in line 2."""
    rows = [{"sector": "NIFTY PSU BK", "avg": 4.70, "up": 5, "n": 5}, {"sector": "NIFTY DEFENCE", "avg": 1.5, "up": 4, "n": 6},
            {"sector": "NIFTY METAL", "avg": -0.28, "up": 8, "n": 14}, {"sector": "NIFTY IT", "avg": -1.02, "up": 2, "n": 16}]
    ev = {"index": {"name": "NIFTY 50", "price": 25007.3, "pct": 0.08, "fut_above": True, "range_pos": 69.0, "gap": None},
          "sectors": {"rows": rows}, "market": {"adv": 119, "dec": 87, "breadth_trend": "improving", "vix_pct": -5.2},
          "vix": {"pct": -5.2}, "breadth": {"share_up": 0.57},
          "movers": {"rows": [("IREDA", 7.79, 120.0, "NIFTY NBFC"), ("PNB", 5.37, 132.29, "NIFTY PSU BK")]},
          "oi_mix": {"counts": {"long_buildup": 14, "short_buildup": 4, "short_covering": 4, "long_unwinding": 0}}}
    labels, _c, conf, lean, d = E.analyze("index_move", {"target": "NIFTY 50"}, ev)
    assert "SECTOR_SPLIT" in labels
    flat = abs(E._main_pct("index_move", ev)) < E.FLAT_PCT
    b = E.brief("index_move", {"target": "NIFTY 50"}, ev, labels, lean, conf, d, flat)
    assert b[0] == "➖ NIFTY 50 flat +0.08% · 25,007.3"
    assert b[1] == "Sectors: split underneath, PSU BK +4.70% (5/5 up) vs IT -1.02% (2/16 up)"
    assert "Now: 69% of low→high range · futures above VWAP" in b
    assert b[-1] == "Breadth: 119↑/87↓ improving · VIX -5.2% · Stock OI: long build-up 14 vs short build-up 4"
    # without the split, 'Led by' names stocks of THAT sector only (IREDA is NBFC, not PSU BK)
    ev["sectors"]["rows"] = rows[:3]
    ev["index"]["pct"] = 0.6
    labels = E.analyze("index_move", {"target": "NIFTY 50"}, ev)[0]
    b = E.brief("index_move", {"target": "NIFTY 50"}, ev, labels, None, None, "up", False)
    led = [x for x in b if x.startswith("Sectors: led by")][0]
    assert led == "Sectors: led by PSU BK +4.70% (5/5 up) · PNB +5.37% · weakest METAL -0.28% (8/14 up)"
    assert "IREDA" not in led
