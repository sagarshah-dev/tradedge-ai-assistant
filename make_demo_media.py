# -*- coding: utf-8 -*-
"""
make_demo_media.py -- builds the README media: docs/demo.gif + docs/demo.png (questions and a price
alert) and docs/demo-agent.gif + docs/demo-agent.png (WHY agent, OI build-up, odds, market watch).

    pip install playwright pillow && python -m playwright install chromium
    python make_demo_media.py          (CHROME_PATH=/path/to/chrome to use an installed browser)

Nothing here is drawn by hand: it runs the real offline tour (demo.run_tour -> the same
ai_chat.Bot / ai_alerts / ai_tools code as the Telegram bot, on fake_market), records
every message, and renders them as a Telegram-style chat. Change the code, re-run this,
and the GIF shows the new replies. Not part of the bot; only for the README.
"""
import html
import io
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import demo                                    # noqa: E402

import ai_tools                                 # noqa: E402

STEPS = [("q", "what is low of nykaa"), ("q", "stocks where open = high"),
         ("q", "top 3 sectors right now"), ("q", "should I buy TATAMOTORS"),
         ("q", "alert me when nykaa goes below 348"), ("btn", 1), ("tick", 8)]

# v19-v21: the LangGraph WHY agent (and its alert offer = human-in-the-loop), fresh OI build-up,
# signal odds (code only), and a ready-made market watch whose "Why" button runs the agent.
AGENT_STEPS = [("q", "why are psu banks firing today"), ("btn", 2),
               ("q", "which stocks showed fresh OI buildup today"),
               ("q", "odds SBIN"), ("btn", 2),
               ("tick", 3), ("btn", 1)]

WIDTH, HEIGHT = 560, 860

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#ffffff;font-family:Inter,system-ui,sans-serif}
.phone{width:%dpx;height:%dpx;display:flex;flex-direction:column;background:#cfdde6;overflow:hidden}
.top{background:#517da2;color:#fff;padding:12px 16px;display:flex;gap:12px;align-items:center;flex:none}
.av{width:38px;height:38px;border-radius:50%%;background:#e7a33e;display:grid;place-items:center;
    font-weight:700;font-size:15px}
.nm{font-weight:600;font-size:15px}.st{font-size:12px;opacity:.85}
.chat{flex:1;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end;padding:10px 12px;gap:7px}
.m{max-width:92%%;padding:7px 10px 6px;border-radius:12px;box-shadow:0 1px 1px rgba(0,0,0,.12)}
.u{align-self:flex-end;background:#effdde;font-size:14px;border-bottom-right-radius:4px}
.b{align-self:flex-start;background:#fff;border-bottom-left-radius:4px}
.b pre{font-family:"DejaVu Sans Mono",monospace;font-size:11px;line-height:1.38;white-space:pre-wrap;
       word-break:break-word;color:#1b1b1b}
.b .rich{font-size:12.5px;line-height:1.42;white-space:pre-wrap;word-break:break-word;color:#1b1b1b}
.btns{display:flex;gap:6px;margin-top:5px;align-self:flex-start;width:min(92%%,330px)}
.btns span{flex:1;text-align:center;background:rgba(48,74,94,.55);color:#fff;font-size:13px;font-weight:500;
           padding:8px 6px;border-radius:9px}
.btns span.on{background:rgba(48,74,94,.9);outline:2px solid #fff}
.sys{align-self:center;background:rgba(40,60,75,.45);color:#fff;font-size:12px;padding:4px 11px;border-radius:12px}
.foot{flex:none;background:#fff;border-top:1px solid #dde;padding:10px 14px;font-size:13px;color:#8a96a0}
"""


def _items(log):
    """Chat elements in order. A press highlights the button that was pressed."""
    out = []
    for e in log:
        if e[0] == "user":
            out.append('<div class="m u">%s</div>' % html.escape(e[1]))
        elif e[0] == "bot":
            if isinstance(e[1], ai_tools.Rich):          # Telegram HTML mode: only <b> and <i> are kept
                body = html.escape(str(e[1]))
                for tag in ("b", "i"):
                    body = body.replace("&lt;%s&gt;" % tag, "<%s>" % tag).replace("&lt;/%s&gt;" % tag, "</%s>" % tag)
                out.append('<div class="m b"><div class="rich">%s</div></div>' % body)
            else:
                out.append('<div class="m b"><pre>%s</pre></div>' % html.escape(e[1]))
            if e[2]:
                out.append('<div class="btns">%s</div>' % "".join("<span>%s</span>" % html.escape(b) for b in e[2]))
        elif e[0] == "press":
            out.append(("press", e[1]))
        else:
            out.append('<div class="sys">⏱ %s</div>' % html.escape(e[1]))
    return out


def _page(items, upto):
    shown, press = [], None
    for it in items[:upto]:
        if isinstance(it, tuple):
            press = it[1]
            continue
        shown.append(it)
    if press:                                            # mark the pressed button in the last button row
        for i in range(len(shown) - 1, -1, -1):
            if shown[i].startswith('<div class="btns">'):
                shown[i] = shown[i].replace("<span>%s</span>" % html.escape(press),
                                            '<span class="on">%s</span>' % html.escape(press))
                break
    body = ('<div class="phone"><div class="top"><div class="av">TE</div><div><div class="nm">TradEdge AI</div>'
            '<div class="st">demo · simulated market · same code as the live bot</div></div></div>'
            '<div class="chat">%s</div><div class="foot">Ask anything, or start with "alert me" …</div></div>'
            % "".join(shown))
    return "<!doctype html><meta charset=utf-8><style>%s</style>%s" % (CSS % (WIDTH, HEIGHT), body)


def render(steps, name):
    from PIL import Image
    from playwright.sync_api import sync_playwright
    log = []
    demo.run_tour(steps, log=log, quiet=True)
    items = _items(log)
    out = os.path.join(ROOT, "docs")
    os.makedirs(out, exist_ok=True)
    frames, durations = [], []
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        pg = br.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1.5)   # sharp text, GIF < 1.5 MB
        for i in range(1, len(items) + 1):
            it = items[i - 1]
            if isinstance(it, str) and it.startswith('<div class="btns">'):
                continue                                 # buttons arrive with their message
            upto = i + (1 if i < len(items) and isinstance(items[i], str)
                        and items[i].startswith('<div class="btns">') else 0)
            pg.set_content(_page(items, upto))
            frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
            text = it if isinstance(it, str) else ""
            durations.append(900 if 'class="m u"' in text or isinstance(it, tuple) else
                             1200 if 'class="sys"' in text else min(4200, 1600 + 9 * len(text) // 10))
        durations[-1] = 4500                             # hold the last message
        hi = br.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2)
        hi.set_content(_page(items, len(items)))
        hi.screenshot(path=os.path.join(out, name + ".png"))
        br.close()
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(os.path.join(out, name + ".gif"), save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True)
    print("docs/%s.gif (%d frames, %.0f KB) · docs/%s.png" % (
        name, len(frames), os.path.getsize(os.path.join(out, name + ".gif")) / 1024, name))


def main():
    render(STEPS, "demo")
    render(AGENT_STEPS, "demo-agent")


if __name__ == "__main__":
    main()
