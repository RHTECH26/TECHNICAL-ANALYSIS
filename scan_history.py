#!/usr/bin/env python3
"""
Daily price history for the "My portfolio" performance chart (Model Portfolio > My Portfolio).

Writes into ./out :
  cal.json         {"dates": [every Nifty 500 trading day, oldest first], "bench": [Nifty 500 closes]}    (one file)
  h/<SYMBOL>.json  {"o": index of the first date in cal.json, "c": [daily closes, null where the stock did not trade]}   (one per stock / ETF)

Every NSE stock in the stock list and every Indian ETF in index.html is included, with about 7 years of daily closes (split-adjusted by
Yahoo, not dividend-adjusted).  The page loads only the files of the stocks you actually hold or held.
The workflow publishes ./out to the "live-history" branch (one commit that is replaced every run, so the repo history does not grow).
Run locally:  python scan_history.py
"""
import json, os, re, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    from curl_cffi import requests as cffi_requests
    SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    SESSION = None

OUT_DIR = os.environ.get("OUT_DIR", "out")
INDEX_FILE = os.environ.get("INDEX_FILE", "index.html")
PERIOD = os.environ.get("HISTORY_PERIOD", "7y")
CHUNK = int(os.environ.get("HISTORY_CHUNK", 40))
BENCH = "^CRSLDX"                                   # Nifty 500
MIN_FILES = int(os.environ.get("HISTORY_MIN_FILES", 500))


def file_name(sym):
    """The file name of a symbol (the page uses the same rule): letters and digits only, everything else becomes _"""
    return re.sub(r"[^A-Za-z0-9]", "_", sym)


def page_list(var_name):
    text = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, text, re.S)
    if not m:
        raise RuntimeError("could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def members():
    from scan_dow_trend import load_symbols
    out = [(sym + ".NS", sym) for sym, _name in load_symbols()]
    for a in page_list("TREE"):
        t = str(a.get("t", "")).strip().upper()
        if t:
            out.append((t + ".NS", t))
    seen, res = set(), []
    for y, s in out:
        if s not in seen:
            seen.add(s)
            res.append((y, s))
    return res


def download(tickers, tries=3):
    kw = dict(period=PERIOD, interval="1d", group_by="ticker", auto_adjust=False, threads=(SESSION is None), progress=False)
    if SESSION is not None:
        kw["session"] = SESSION
    for attempt in range(tries):
        try:
            data = yf.download(tickers, **kw)
            if data is not None and len(data):
                return data
        except Exception as e:
            print(f"  download attempt {attempt + 1} failed: {type(e).__name__}")
        time.sleep(5 * (attempt + 1))
    return None


def clean_index(ix):
    ix = pd.DatetimeIndex(ix)
    return (ix.tz_localize(None) if ix.tz is not None else ix).normalize()


def close_of(data, ticker=None):
    """The Close column of a Yahoo download, whichever way the columns are laid out (field first, ticker first, or plain)."""
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        lv0 = list(cols.get_level_values(0))
        if ticker is not None and ticker in lv0:                  # (ticker, field)
            return data[ticker]["Close"]
        close = data["Close"]                                     # (field, ticker)
        return close.iloc[:, 0] if hasattr(close, "columns") else close
    return data["Close"]


def benchmark():
    d = None
    for attempt in range(3):
        try:
            kw = dict(period=PERIOD, interval="1d", auto_adjust=False, progress=False)
            if SESSION is not None:
                kw["session"] = SESSION
            d = yf.download(BENCH, **kw)
            if d is not None and len(d):
                break
        except Exception as e:
            print(f"  benchmark attempt {attempt + 1} failed: {type(e).__name__}")
        time.sleep(5 * (attempt + 1))
    if d is None or not len(d):
        raise RuntimeError("could not download the Nifty 500")
    ser = close_of(d, BENCH)
    ser = pd.Series(ser.to_numpy(float), index=clean_index(ser.index)).dropna()
    ser = ser[~ser.index.duplicated(keep="last")]
    if len(ser) < 500:
        raise RuntimeError("the Nifty 500 history is too short")
    return ser


def pack(series, cal):
    """closes aligned to the calendar, trimmed of leading empty days; None for gaps"""
    s = pd.Series(series.to_numpy(float), index=clean_index(series.index))
    s = s[~s.index.duplicated(keep="last")].reindex(cal)
    vals = s.to_numpy(float)
    ok = np.nonzero(~np.isnan(vals))[0]
    if len(ok) < 20:
        return None
    first, last = int(ok[0]), int(ok[-1])
    seg = vals[first:last + 1]
    return {"o": first, "c": [None if np.isnan(v) else round(float(v), 2) for v in seg]}


def main():
    if yf is None:
        raise SystemExit("yfinance is not installed")
    bench = benchmark()
    cal = bench.index
    os.makedirs(os.path.join(OUT_DIR, "h"), exist_ok=True)
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    with open(os.path.join(OUT_DIR, "cal.json"), "w") as f:
        json.dump({"updated": ist.strftime("%d %b %Y, %H:%M IST"), "dates": [str(d.date()) for d in cal], "bench": [round(float(v), 2) for v in bench.to_numpy(float)]}, f, separators=(",", ":"))
    mem = members()
    print("Universe:", len(mem), "symbols | calendar:", len(cal), "days |", cal[0].date(), "to", cal[-1].date())
    written, names = 0, {}
    for i in range(0, len(mem), CHUNK):
        part = mem[i:i + CHUNK]
        print(f"  {i + 1}-{i + len(part)} of {len(mem)}", flush=True)
        data = download([m[0] for m in part])
        if data is None:
            print("   chunk gave no data")
            continue
        for ysym, sym in part:
            try:
                close = close_of(data, ysym).dropna()
                rec = pack(close, cal)
            except Exception:
                continue
            if rec is None:
                continue
            fn = file_name(sym)
            if fn in names and names[fn] != sym:
                print("   file name clash:", sym, "and", names[fn])
                continue
            names[fn] = sym
            with open(os.path.join(OUT_DIR, "h", fn + ".json"), "w") as f:
                json.dump(rec, f, separators=(",", ":"))
            written += 1
        time.sleep(1.0)
    if written < MIN_FILES:
        raise RuntimeError(f"too few history files were written ({written})")
    size = sum(os.path.getsize(os.path.join(OUT_DIR, "h", x)) for x in os.listdir(os.path.join(OUT_DIR, "h")))
    print(f"Done: {written} history files, {size // 1024 // 1024} MB in total, newest date {cal[-1].date()}")


if __name__ == "__main__":
    main()
