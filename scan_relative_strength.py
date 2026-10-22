"""
Relative Strength data for the "Relative Strength" tab of index.html.

Writes two files (the page computes everything else in the browser, so the person can pick any
High date / Low date without a new scan):
  data/rs_stocks.json      daily closes of the Indian stock universe (same stock list the COT scan uses)
  data/rs_benchmarks.json  daily closes of the benchmark indices (Nasdaq Composite, S&P 500, Nifty 50, Nifty 500)

Global ETFs and Indian ETFs need no new scan: the tab reads the price history files that
fetch_global_etf_data.py / fetch_etf_data.py already write.

Run:  python scan_relative_strength.py
"""
import json, os, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from scan_dow_trend import load_symbols     # same stock-list file the COT scan reads

PERIOD     = "14mo"      # ~300 trading days: covers a 52-week high plus room for earlier High / Low dates
CHUNK      = 100
BENCH_PERIOD = "max"   # benchmarks keep their full daily history (small files), so any past date can be compared
MAX_STALE  = 10          # drop stocks whose last price is more than this many days older than the freshest one
MIN_POINTS = 60
STOCK_OUT  = "data/rs_stocks.json"
BENCH_OUT  = "data/rs_benchmarks.json"

BENCHMARKS = {           # id used by the page -> (Yahoo symbol, display name)
    "IXIC":   ("^IXIC",   "Nasdaq Composite"),
    "GSPC":   ("^GSPC",   "S&P 500"),
    "NSEI":   ("^NSEI",   "Nifty 50"),
    "CRSLDX": ("^CRSLDX", "Nifty 500"),
}


def day_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def pick_close(data, tk, multi):
    """Close column of one ticker out of a yf.download() result."""
    try:
        df = data[tk] if multi else data
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(float)
        s.index = day_index(s.index)
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return None


def clean(vals):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 2) for v in vals]


def build_stocks(frames, names, failed):
    closes = pd.concat(frames, axis=1).sort_index()
    closes = closes.dropna(how="all").ffill(limit=5)
    freshest = closes.index.max()
    stocks = []
    for sym in closes.columns:
        col = closes[sym]
        valid = col.dropna()
        if len(valid) < MIN_POINTS or (freshest - valid.index[-1]).days > MAX_STALE or np.isnan(col.iloc[-1]):
            failed += 1
            continue
        stocks.append({"s": sym, "n": names.get(sym, sym), "c": clean(col.tolist())})
    return [d.strftime("%Y-%m-%d") for d in closes.index], stocks, failed


def main():
    universe = load_symbols()
    names = {s: n for s, n in universe}
    syms = [s for s, _ in universe]
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    stamp = ist.strftime("%d %b %Y, %H:%M IST")
    os.makedirs("data", exist_ok=True)

    # ---- benchmarks (small file, written first so it is available even if the stock download struggles)
    bench = {}
    for bid, (ysym, label) in BENCHMARKS.items():
        try:
            data = yf.download(ysym, period=BENCH_PERIOD, interval="1d", auto_adjust=True, progress=False)
            s = pick_close(data, ysym, multi=isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels > 1 and ysym in data.columns.get_level_values(0))
            if s is None or s.dropna().empty:
                s = pick_close(data, ysym, multi=False)
            s = s.dropna()
            if len(s) < MIN_POINTS:
                print("benchmark skipped (too little data):", label)
                continue
            bench[bid] = {"n": label, "d": [d.strftime("%Y-%m-%d") for d in s.index], "c": clean(s.tolist())}
            print("benchmark ok:", label, len(s), "bars")
        except Exception as e:
            print("benchmark failed:", label, e)
        time.sleep(1)
    with open(BENCH_OUT, "w") as f:
        json.dump({"updated": stamp, "bench": bench}, f, separators=(",", ":"))

    # ---- stocks
    frames, failed = [], 0
    for i in range(0, len(syms), CHUNK):
        part = syms[i:i + CHUNK]
        tickers = [s + ".NS" for s in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(syms)}", flush=True)
        data = None
        for attempt in range(2):
            try:
                data = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as e:
                print("  chunk failed:", e)
            time.sleep(5)
        if data is None or not len(data):
            failed += len(part)
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for sym, tk in zip(part, tickers):
            s = pick_close(data, tk, multi)
            if s is None or s.dropna().empty:
                failed += 1
                continue
            frames.append(s.rename(sym))
        time.sleep(1)

    if not frames:
        raise SystemExit("No stock data downloaded - nothing written.")
    dates, stocks, failed = build_stocks(frames, names, failed)
    with open(STOCK_OUT, "w") as f:
        json.dump({"updated": stamp, "scanned": len(stocks), "failed": failed,
                   "dates": dates, "stocks": stocks}, f, separators=(",", ":"))
    print(f"Done. stocks={len(stocks)} failed={failed} dates={len(dates)} benchmarks={list(bench)}")


if __name__ == "__main__":
    main()
