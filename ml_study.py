# -*- coding: utf-8 -*-
"""
ml_study.py -- does a machine-learning model beat the simple lookup?  OFFLINE, READ-ONLY on saga_obs.db.

    py ml_study.py          (after market hours, from the TradEdge folder)

Writes data/ml_study.json, which /mlreport shows in Telegram. Nothing in the live app changes.

Method (the part that matters in an interview):
  * events  : SANKET / SAGA / SHARK, one per stock per day (v_signals_1pd), scored, with a direction
  * label   : WIN (res_10) vs not
  * features: only what the scanner logged AT the signal (rvol, pace, dp15, day_pct, nifty_pct, time of
              day, stage, side). Nothing from after the signal -> no leakage by construction.
  * test    : walk-forward by month. Train on every month BEFORE the test month, test on that month.
  * baseline: the lookup the bot already uses (win rate of the same stage + side in the training data).
  * score   : AUC (does it rank winners above losers?) on all test months pooled, model vs baseline.
  * noise   : bootstrap over test DAYS (signals on one day move together) -> 95% range of the gain.
  * verdict : 'use' only if the whole 95% range is above 0 AND there are >= 300 test events.
Needs: lightgbm (falls back to scikit-learn's gradient boosting if lightgbm is missing).
"""
import json
import os
import random
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FEATURES = ["rvol", "pace", "dp15", "day_pct", "nifty_pct", "minute", "stage_SANKET", "stage_SAGA", "stage_SHARK",
            "side_long"]
MIN_TRAIN = 80                       # events before a month can be tested
MIN_TEST_FOR_USE = 300
BOOT = 1000
SEED = 7


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def features(e):
    hm = str(e["t"] or "09:15")[:5]
    minute = (int(hm[:2]) * 60 + int(hm[3:5])) - (9 * 60 + 15)
    return [_num(e["rvol"]), _num(e["pace"]), _num(e.get("dp15")), _num(e.get("day_pct")), _num(e["nifty_pct"]),
            float(minute), float(e["stage"] == "SANKET"), float(e["stage"] == "SAGA"), float(e["stage"] == "SHARK"),
            float(str(e["side"]).upper() == "LONG")]


def auc(y, s):
    """Rank AUC with ties (Mann-Whitney). 0.5 = no better than a coin."""
    pairs = sorted(zip(s, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1
        i = j + 1
    pos = sum(1 for _, t in pairs if t)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    return (sum(r for r, (_, t) in zip(ranks, pairs) if t) - pos * (pos + 1) / 2.0) / (pos * neg)


def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                  verbose=-1), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                                              min_samples_leaf=20, random_state=SEED), "sklearn HistGradientBoosting"


def lookup_scores(train, test):
    """The bot's lookup as a score: smoothed win rate of the same stage + side in TRAIN."""
    tot = {}
    for e in train:
        k = (e["stage"], str(e["side"]).upper())
        w, n = tot.get(k, (0, 0))
        tot[k] = (w + (e["res"] == "WIN"), n + 1)
    allw = sum(w for w, _ in tot.values())
    alln = sum(n for _, n in tot.values()) or 1
    prior = allw / float(alln)
    out = []
    for e in test:
        w, n = tot.get((e["stage"], str(e["side"]).upper()), (0, 0))
        out.append((w + 5 * prior) / (n + 5.0))
    return out


def decide(auc_model, auc_lookup, gain_lo, n_test):
    """'use' only when the WHOLE 95% range of the gain is above zero and the test set is big enough.
    A model that is better on average but inside the noise is not used."""
    return "use" if gain_lo > 0 and n_test >= MIN_TEST_FOR_USE and auc_model > auc_lookup else "no"


def study(events, seed=SEED):
    import numpy as np
    events = sorted(events, key=lambda e: (e["day"], str(e["t"])))
    months = sorted({e["day"][:7] for e in events})
    y_all, m_all, b_all, d_all, folds = [], [], [], [], 0
    name = ""
    for m in months:
        train = [e for e in events if e["day"][:7] < m]
        test = [e for e in events if e["day"][:7] == m]
        if len(train) < MIN_TRAIN or not test:
            continue
        ytr = [e["res"] == "WIN" for e in train]
        if len(set(ytr)) < 2:
            continue
        model, name = make_model()
        model.fit(np.array([features(e) for e in train]), np.array(ytr, dtype=int))
        p = model.predict_proba(np.array([features(e) for e in test]))[:, 1]
        y_all += [e["res"] == "WIN" for e in test]
        m_all += list(p)
        b_all += lookup_scores(train, test)
        d_all += [e["day"] for e in test]
        folds += 1
    if not folds:
        raise SystemExit("Not enough history for one walk-forward month (need %d events before a month)." % MIN_TRAIN)
    a_m, a_b = auc(y_all, m_all), auc(y_all, b_all)
    rnd = random.Random(seed)
    days = sorted(set(d_all))
    by_day = {d: [i for i, x in enumerate(d_all) if x == d] for d in days}
    gains = []
    for _ in range(BOOT):
        idx = [i for d in (rnd.choice(days) for _ in days) for i in by_day[d]]
        y = [y_all[i] for i in idx]
        g = auc(y, [m_all[i] for i in idx]) - auc(y, [b_all[i] for i in idx])
        if g == g:                                     # not NaN
            gains.append(g)
    gains.sort()
    lo, hi = gains[int(0.025 * len(gains))], gains[int(0.975 * len(gains)) - 1]
    top = []
    try:
        model, _n = make_model()
        model.fit(np.array([features(e) for e in events]), np.array([e["res"] == "WIN" for e in events], dtype=int))
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            top = [FEATURES[i] for i in sorted(range(len(FEATURES)), key=lambda i: -imp[i]) if imp[i] > 0]
    except Exception:
        top = []
    n = len(events)
    return {"run_at": datetime.now().strftime("%d-%m-%Y %H:%M"), "n": n, "days": len({e["day"] for e in events}),
            "stages": ["SANKET", "SAGA", "SHARK"], "label": "WIN by res_10",
            "folds": folds, "n_test": len(y_all), "auc_model": a_m, "auc_lookup": a_b, "gain": a_m - a_b,
            "gain_lo": lo, "gain_hi": hi, "verdict": decide(a_m, a_b, lo, len(y_all)),
            "top_features": top, "model": name, "next_at": max(600, n * 2), "base_win": sum(y_all) / float(len(y_all))}


def main():
    import ai_odds
    db = os.path.join(HERE, "data", "saga_obs.db")
    events = ai_odds.load_events(db)
    print("events: %d" % len(events))
    r = study(events)
    out = os.path.join(HERE, "data", "ml_study.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))
    print("\nVerdict: %s -> written %s (see /mlreport in Telegram)" % (
        "model beats the lookup outside the noise" if r["verdict"] == "use" else "inside the noise, model NOT used",
        out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
