#!/usr/bin/env python3
"""
Latest prices for the "My portfolio" page (Model Portfolio > My Portfolio).

Writes out/prices.json:  every NSE stock in the stock list + every Indian ETF in index.html (TREE), with
    {"p": {"SYMBOL": ["Name", last price, previous close, "YYYY-MM-DD of the last price"]}, "asof": newest trading date, ...}
The page uses it for the current value of each holding and for today's gain / loss (last price vs previous close).
During market hours Yahoo's "last price" is the live (about 15 minutes delayed) price; after the close it is the closing price.

The workflow publishes the file to the "live-prices" branch (one commit that is replaced every run, so the repo history does not grow).
Run locally:  python scan_prices.py
"""
import json, os, re, sys, time
from datetime import datetime, timezone, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    from curl_cffi import requests as cffi_requests      # makes Yahoo accept many requests from a GitHub runner
    SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    SESSION = None

OUT_DIR = os.environ.get("OUT_DIR", "out")
INDEX_FILE = os.environ.get("INDEX_FILE", "index.html")
CHUNK = int(os.environ.get("PRICES_CHUNK", 100))
MIN_ROWS = int(os.environ.get("PRICES_MIN_ROWS", 500))
MAX_AGE_DAYS = 10                    # a symbol whose last price is older than this is left out (delisted / suspended)


def page_list(var_name):
    text = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, text, re.S)
    if not m:
        raise RuntimeError("could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def members():
    """[(yahoo symbol, display symbol, name)] for all NSE stocks and Indian ETFs."""
    from scan_dow_trend import load_symbols          # the stock list file the other scans read
    out = []
    for sym, name in load_symbols():
        out.append((sym + ".NS", sym, name))
    for a in page_list("TREE"):
        t = str(a.get("t", "")).strip().upper()
        if t:
            out.append((t + ".NS", t, a.get("sub") or t))
    seen, res = set(), []
    for m in out:
        if m[1] not in seen:                          # a stock and an ETF never share a symbol; keep the first if they ever do
            seen.add(m[1])
            res.append(m)
    return res


def download(tickers, tries=3):
    kw = dict(period="10d", interval="1d", group_by="ticker", auto_adjust=False, threads=(SESSION is None), progress=False)
    if SESSION is not None:
        kw["session"] = SESSION
    for attempt in range(tries):
        try:
            data = yf.download(tickers, **kw)
            if data is not None and len(data):
                return data
        except Exception as e:
            print(f"  download attempt {attempt + 1} failed: {type(e).__name__}")
        time.sleep(4 * (attempt + 1))
    return None


def last_two(series):
    s = series.dropna()
    if len(s) < 2:
        raise ValueError("not enough rows")
    return float(s.iloc[-1]), float(s.iloc[-2]), pd.Timestamp(s.index[-1]).tz_localize(None) if pd.Timestamp(s.index[-1]).tzinfo else pd.Timestamp(s.index[-1])


def build():
    mem = members()
    print("Universe:", len(mem), "symbols")
    rows = {}
    for i in range(0, len(mem), CHUNK):
        part = mem[i:i + CHUNK]
        print(f"  {i + 1}-{i + len(part)} of {len(mem)}", flush=True)
        data = download([m[0] for m in part])
        if data is None:
            print("   chunk gave no data")
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for ysym, sym, name in part:
            try:
                close = (data[ysym] if multi else data)["Close"]
                last, prev, dt = last_two(close)
                if last > 0 and prev > 0:
                    rows[sym] = (name, last, prev, dt)
            except Exception:
                continue
        time.sleep(1.0)
    if len(rows) < MIN_ROWS:
        raise RuntimeError(f"too little price data came back ({len(rows)} symbols)")
    newest = max(r[3] for r in rows.values())
    rows = {k: v for k, v in rows.items() if (newest - v[3]).days <= MAX_AGE_DAYS}
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "updated_ist": ist.strftime("%d %b %Y, %H:%M IST"),
            "asof": str(newest.date()), "count": len(rows),
            "p": {k: [v[0], round(v[1], 2), round(v[2], 2), str(v[3].date())] for k, v in sorted(rows.items())}}


def main():
    if yf is None:
        raise SystemExit("yfinance is not installed")
    out = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "prices.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Done: {out['count']} symbols, newest price date {out['asof']} | {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
