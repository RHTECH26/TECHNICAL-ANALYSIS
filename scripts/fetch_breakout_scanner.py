#!/usr/bin/env python3
"""
Rules-based "breaking out today" scanner over the same 966-ticker global ETF universe as
fetch_global_etf_data.py (imports its TICKERS list directly, so there's one source of truth
for the universe). This is the simple, no-ML scanner from the ETF-ML-Screener project's
05_current_breakouts.py, ported to run daily in this repo's own GitHub Action instead of
locally — NOT the trained ML breakout-probability model, which is a separate, heavier
project kept local for now.

The default flag (also the one used server-side for the "asof"-day snapshot) is:
    today's close is a new 20-day high  AND  today's volume >= 1.5x its trailing 20-day
    average volume

...but the lookback window (20/50/100/200 days) and the volume multiple are both adjustable
on the site itself, live, with no extra fetch — each row also carries its own trimmed
close/volume history (last ~210 trading days, enough to cover a 200-day lookback with a
20-day volume-average tail), and the page recomputes the flag in JavaScript whenever the
person changes either control. The 52-week-high, RSI(14), and 50-day/200-day-SMA trend
columns are NOT adjustable (fixed-window indicators, shown for context) and stay
server-computed.

Writes data/breakout-scanner.json:
    {
      "asof": "YYYY-MM-DD",
      "rows": [
        {"t": "SPY", "price": 512.34, "chg1d": 0.42, "high20": true, "high52w": false,
         "volRatio": 1.87, "rsi14": 68.2, "aboveSma50": true, "aboveSma200": true,
         "breakout": true, "closes": [...], "vols": [...]},
        ...
      ]
    }
"closes"/"vols" are the trailing ~210 daily values (oldest to newest, same length, last
entry = today) used to recompute high20/high50/high100/high200 and the volume ratio
client-side for any lookback the person picks. Tickers with too little history to compute a
given metric report null for it rather than being dropped from the universe entirely (e.g. a
recently-listed ETF can still get a 20-day-high flag even without 252 days for a
52-week-high flag, and the client-side recompute falls back to "—" for a lookback longer
than the ticker's available history).

--- Why this version routes through curl_cffi ---
The original batch design here (yf.download() over 30-ticker batches, threads=False, plain
linear retry backoff) is fast when it works, but on a GitHub Actions runner it was getting
the first batch through and then coming back completely empty for every batch after that,
regardless of how much the backoff was tuned. That symptom -- works once, then silently
blocked -- is Yahoo Finance's bot detection fingerprinting the default requests/urllib3 TLS
handshake yfinance uses, not simple rate-limiting (which slowing down would actually fix).
Routing requests through curl_cffi, which impersonates a real Chrome TLS fingerprint, is the
standard workaround for this exact GitHub-Actions-specific failure mode, and it's a one-line
addition (a session object passed into yf.download) that doesn't change the batch size,
pacing, or overall runtime of the original script at all.

Run locally with:  python3 scripts/fetch_breakout_scanner.py
Runs automatically via .github/workflows/update-data.yml (as its own parallel job). Needs
curl_cffi installed (pip install curl_cffi) -- see the note above for why.
"""
import json
import os
import time

import yfinance as yf
import pandas as pd

from fetch_global_etf_data import TICKERS

try:
    from curl_cffi import requests as cffi_requests
    _SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    print("WARNING: curl_cffi not installed -- falling back to yfinance's default session, "
          "which is the thing that was getting silently blocked. Add curl_cffi to the "
          "workflow's 'pip install' step.")
    _SESSION = None

OUT_PATH = "data/breakout-scanner.json"
LOOKBACK = "1y"   # ~252 trading days — enough for the 52-week-high check and both SMAs.
BATCH_SIZE = 30
RETRIES = 4
SLEEP_BETWEEN_BATCHES = 2.5
SLEEP_ON_RETRY = 10.0

HIGH20_WINDOW = 20
HIGH52W_WINDOW = 252
VOL_AVG_WINDOW = 20
RSI_WINDOW = 14
SMA_SHORT = 50
SMA_LONG = 200
VOL_RATIO_BREAKOUT_MIN = 1.5
CLIENT_HISTORY_WINDOW = 210


def compute_rsi(closes, window=RSI_WINDOW):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def score_ticker(t, df):
    df = df.dropna(subset=["Close"])
    if len(df) < HIGH20_WINDOW + 1:
        return None
    closes = df["Close"]
    vols = df["Volume"].fillna(0) if "Volume" in df else pd.Series([0] * len(df), index=df.index)

    price = round(float(closes.iloc[-1]), 4)
    prev = float(closes.iloc[-2])
    chg1d = round((price - prev) / prev * 100, 2) if prev else None

    high20 = bool(closes.iloc[-1] >= closes.iloc[-HIGH20_WINDOW:].max())
    high52w = bool(closes.iloc[-1] >= closes.iloc[-HIGH52W_WINDOW:].max()) if len(closes) >= HIGH52W_WINDOW else None

    vol_ratio = None
    if len(vols) >= VOL_AVG_WINDOW + 1:
        avg_vol = float(vols.iloc[-(VOL_AVG_WINDOW + 1):-1].mean())
        today_vol = float(vols.iloc[-1])
        if avg_vol > 0:
            vol_ratio = round(today_vol / avg_vol, 2)

    rsi_series = compute_rsi(closes)
    rsi14 = round(float(rsi_series.iloc[-1]), 1) if pd.notna(rsi_series.iloc[-1]) else None

    above_sma50 = None
    if len(closes) >= SMA_SHORT:
        sma50 = float(closes.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean().iloc[-1])
        above_sma50 = bool(price > sma50)
    above_sma200 = None
    if len(closes) >= SMA_LONG:
        sma200 = float(closes.rolling(SMA_LONG, min_periods=SMA_LONG).mean().iloc[-1])
        above_sma200 = bool(price > sma200)

    breakout = bool(high20 and vol_ratio is not None and vol_ratio >= VOL_RATIO_BREAKOUT_MIN)

    tail_closes = closes.iloc[-CLIENT_HISTORY_WINDOW:]
    tail_vols = vols.reindex(tail_closes.index).fillna(0)

    return {
        "t": t, "price": price, "chg1d": chg1d,
        "high20": high20, "high52w": high52w, "volRatio": vol_ratio, "rsi14": rsi14,
        "aboveSma50": above_sma50, "aboveSma200": above_sma200, "breakout": breakout,
        "closes": [round(float(c), 4) for c in tail_closes],
        "vols": [int(v) for v in tail_vols],
    }


def load_existing():
    """The previously committed data/breakout-scanner.json, if any -- keyed by ticker so a
    run that still hits trouble partway through merges on top of it instead of wiping out
    everything a prior successful run already fetched (same pattern as
    fetch_global_etf_data.py's load_existing()/merge_series()). Free at runtime -- one read
    at startup, no effect on the fetch loop's pacing."""
    if not os.path.exists(OUT_PATH):
        return {}
    try:
        with open(OUT_PATH) as f:
            data = json.load(f)
        return {row["t"]: row for row in data.get("rows", []) if "t" in row}
    except Exception:
        return {}


def fetch_batch(tickers):
    raw = None
    for attempt in range(RETRIES):
        try:
            raw = yf.download(tickers, period=LOOKBACK, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=False, progress=False,
                               session=_SESSION)
            if raw is not None and len(raw) > 0:
                break
        except Exception as e:
            print(f"  batch download attempt {attempt+1} failed: {e}")
            raw = None
        time.sleep(SLEEP_ON_RETRY * (attempt + 1))
    if raw is None or len(raw) == 0:
        return {}
    single = len(tickers) == 1
    out = {}
    for t in tickers:
        try:
            df = raw if single else raw[t]
            out[t] = df
        except Exception:
            continue
    return out


def main():
    existing = load_existing()
    rows_by_ticker = dict(existing)   # start from the last good run, not from scratch
    n_batches = (len(TICKERS) - 1) // BATCH_SIZE + 1
    for i in range(0, len(TICKERS), BATCH_SIZE):
        batch = TICKERS[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        dfs = fetch_batch(batch)
        for t in batch:
            df = dfs.get(t)
            if df is None:
                continue  # keep whatever this ticker had from a previous run, if anything
            scored = score_ticker(t, df)
            if scored:
                rows_by_ticker[t] = scored

        rows = list(rows_by_ticker.values())
        flagged = sum(1 for r in rows if r["breakout"])
        print(f"Batch {batch_num}/{n_batches}: {len(rows_by_ticker)} unique tickers with data so far, {flagged} flagged")
        # Save progress after every batch — a run cut off partway keeps whatever it has.
        with open(OUT_PATH, "w") as f:
            json.dump({"asof": time.strftime("%Y-%m-%d"), "rows": rows}, f, separators=(",", ":"))
        time.sleep(SLEEP_BETWEEN_BATCHES)

    rows = list(rows_by_ticker.values())
    flagged = sum(1 for r in rows if r["breakout"])
    print(f"Wrote {OUT_PATH}: {len(rows)}/{len(TICKERS)} tickers scored, {flagged} flagged as breaking out.")


if __name__ == "__main__":
    main()
