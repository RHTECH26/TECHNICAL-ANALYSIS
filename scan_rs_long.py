#!/usr/bin/env python3
"""
Full price history (since listing, back to 1990) for the Relative Strength tab, stored as WEEKLY closes
so it stays small enough for a web page. The tab uses the exact daily closes for dates inside the last
~14 months and these weekly closes for anything older.

  python scan_rs_long.py global     -> data/rs_long_global.json   (all ETFs in the Global ETF list)
  python scan_rs_long.py inetf      -> data/rs_long_inetf.json    (Indian ETFs)
  python scan_rs_long.py istock     -> data/rs_long_istock.json   (Indian stocks - same list the COT scan uses)

File layout (one shared calendar of Fridays, so no date is repeated per fund):
  { "updated": "...", "w0": "1990-01-05", "series": { "SPY": [first_week_number, [close, close, ...]], ... } }
  week number n = the Friday that is n weeks after w0. A week's value is the last close of that week
  (Mon-Fri); holiday weeks repeat the previous close (up to 2 weeks), longer gaps are null.
Prices are adjusted closes in each fund's own listing currency, the same convention as the daily files.
"""
import json, os, re, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    SESSION = None

W0 = pd.Timestamp("1990-01-05")          # a Friday
INDEX_FILE = "index.html"
TODAY = pd.Timestamp(datetime.now(timezone.utc).strftime("%Y-%m-%d"))


# ---------------- universes ----------------
def _page_list(var_name):
    html = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, html, re.S)
    if not m:
        raise SystemExit("Could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def universe_global():
    seen, out = set(), []
    for a in _page_list("GLOBAL_ETFS"):
        t = str(a.get("t", "")).strip()
        if t and t not in seen:
            seen.add(t); out.append((t, t))          # (yahoo symbol, key stored in the file)
    return out


def universe_inetf():
    seen, out = set(), []
    for a in _page_list("TREE"):
        t = str(a.get("t", "")).strip()
        if t and t not in seen:
            seen.add(t); out.append((t + ".NS", t))
    return out


def universe_istock():
    from scan_dow_trend import load_symbols            # the same stock list the COT scan reads
    return [(s + ".NS", s) for s, _ in load_symbols()]


UNIVERSES = {
    "global": dict(load=universe_global, batch=30,  threads=False, pause=2.5, session=True,  out="data/rs_long_global.json"),
    "inetf":  dict(load=universe_inetf,  batch=40,  threads=True,  pause=1.0, session=False, out="data/rs_long_inetf.json"),
    "istock": dict(load=universe_istock, batch=100, threads=True,  pause=1.0, session=False, out="data/rs_long_istock.json"),
}


# ---------------- helpers ----------------
def get_close(raw, t, multi):
    try:
        df = raw[t] if multi else raw
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(float).dropna()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return None


def to_weekly(s):
    """[first_week_number, [weekly closes]] or None."""
    s = s[s.index >= W0 - pd.Timedelta(days=6)]
    if len(s) < 5:
        return None
    w = s.resample("W-FRI").last()                    # label = that week's Friday
    w = w[w.index <= TODAY]                           # never a bar dated in the future
    w = w[w.index >= W0]
    first = w.first_valid_index()
    if first is None:
        return None
    w = w.loc[first:].ffill(limit=2)
    off = int((first - W0).days // 7)
    vals = [None if np.isnan(v) else round(float(v), 2 if v >= 10 else 4) for v in w.values]
    return [off, vals]


def load_existing(path):
    try:
        with open(path) as f:
            return json.load(f).get("series", {})
    except Exception:
        return {}


def save(path, series):
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"updated": ist.strftime("%d %b %Y, %H:%M IST"), "w0": W0.strftime("%Y-%m-%d"),
                   "series": series}, f, separators=(",", ":"))


def fetch(tickers, cfg):
    raw = None
    for attempt in range(4):
        try:
            kw = dict(period="max", interval="1d", auto_adjust=True, group_by="ticker",
                      threads=cfg["threads"], progress=False)
            if cfg["session"] and SESSION is not None:
                kw["session"] = SESSION
            raw = yf.download(tickers, **kw)
            if raw is not None and len(raw) > 0:
                break
        except Exception as e:
            print("  download attempt %d failed: %s" % (attempt + 1, e))
            raw = None
        time.sleep(10 * (attempt + 1))
    return raw


def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if name not in UNIVERSES:
        raise SystemExit("Usage: python scan_rs_long.py global | inetf | istock")
    cfg = UNIVERSES[name]
    uni = cfg["load"]()
    print("%s: %d symbols" % (name, len(uni)))
    series = load_existing(cfg["out"])        # keep earlier results for anything Yahoo fails to return today
    got_total, n_batches = 0, (len(uni) - 1) // cfg["batch"] + 1
    for b, i in enumerate(range(0, len(uni), cfg["batch"]), 1):
        part = uni[i:i + cfg["batch"]]
        raw = fetch([y for y, _ in part], cfg)
        got = 0
        if raw is not None and len(raw):
            multi = isinstance(raw.columns, pd.MultiIndex) and len(part) > 1
            for ysym, key in part:
                s = get_close(raw, ysym, multi) if multi else (get_close(raw, ysym, False) if len(part) == 1 else None)
                if s is None or s.empty:
                    continue
                wk = to_weekly(s)
                if wk:
                    series[key] = wk; got += 1
        got_total += got
        print("Batch %d/%d: %d/%d with history (running total %d, file holds %d)" % (b, n_batches, got, len(part), got_total, len(series)), flush=True)
        if b % 10 == 0:
            save(cfg["out"], series)               # progress is kept even if a later batch gets cut off
        time.sleep(cfg["pause"])
    if not series:
        raise SystemExit("Nothing downloaded - nothing written.")
    save(cfg["out"], series)
    print("Wrote %s with %d funds/stocks, %.1f MB" % (cfg["out"], len(series), os.path.getsize(cfg["out"]) / 1e6))


if __name__ == "__main__":
    main()
