# -*- coding: utf-8 -*-
"""
ai_selfcheck.py -- offline data check for the AI chat. READ-ONLY. No LLM, no Telegram.

Run from the TradEdge folder while the app is STOPPED (or running -- it only reads):
    python ai_selfcheck.py

It prints the same HISTORY part as the /check command plus one real answer for
each history question type, so you can see the bot's replies before the market
opens. Live data (scan results, bars, OI) only exists inside the running app,
so for that part use /check in Telegram.

Exit code 0 = every history tool answered, 1 = something is missing.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ai_tools as T   # noqa: E402

# One question per history tool -- the shapes the user asks most.
SAMPLES = [
    ("top_symbols", {"metric": "gain", "days": 5, "limit": 3}),
    ("top_symbols", {"metric": "volume", "days": 3, "limit": 3}),
    ("top_symbols", {"metric": "oi", "days": 5, "limit": 3}),
    ("top_symbols", {"metric": "pace", "days": 5, "limit": 3}),
    ("level_breaks_history", {"kind": "prev_high", "limit": 3}),
    ("level_breaks_history", {"kind": "prev_low", "limit": 3}),
    ("sector_history", {"days": 5}),
]


def main():
    try:
        import state
    except Exception as e:                      # still useful without the app's state.py
        print("state.py not importable (%s) -- symbol and sector lists will be empty" % e)
        import types
        state = types.SimpleNamespace(FNO_SYMBOLS=[], SECTOR_MAP={})
    ctx = T.Ctx(state)
    print(T.check_report(ctx, live=False))
    failed = 0
    for name, args in SAMPLES:
        r, err = T.run(ctx, name, args)
        print("\n=== %s %s" % (name, args))
        if err:
            failed += 1
            print("❌ " + err)
        else:
            print(T.format_reply(r))
    print("\n%d of %d sample answers OK" % (len(SAMPLES) - failed, len(SAMPLES)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
