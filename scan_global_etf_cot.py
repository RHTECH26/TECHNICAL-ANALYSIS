"""
Global ETF - Dow Theory Change of Trend scanner (daily chart).
Reads the ETF list straight from index.html (the GLOBAL_ETFS list used by the Global ETF Heatmap),
so the scan always matches the ETFs on your page.
Same logic as the TradingView indicator:
  * swing high / low = pivot with N bars on each side (confirmed N bars later)
  * uptrend   -> turns BEARISH when a daily CLOSE is below the last higher low
  * downtrend -> turns BULLISH when a daily CLOSE is above the last lower high
Timeframes: 1H, 1D, 1W, 1M (run:  python scan_global_etf_cot.py 1D 1W 1M   or   python scan_global_etf_cot.py 1H).
Output: data/global_etf_cot.json (1D) and data/global_etf_cot_1H.json / _1W.json / _1M.json
        (read by the COT tab of index.html)
"""
import json, math, os, re, time
from datetime import datetime, timezone, timedelta
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import yfinance as yf

INDEX_FILE    = "index.html"
OUT_FILE      = "data/global_etf_cot.json"
SWING_OPTIONS = [3, 5, 8]
DEFAULT_SWING = 5
CHUNK         = 100

# ---- timeframes -------------------------------------------------------------------------
# interval / history Yahoo is asked for, minimum candles needed, and how many candles back a
# signal may have fired and still be listed. "1D" keeps the original output file name so
# nothing that already reads it breaks; the others get a suffix.
TIMEFRAMES = {
    "1H": {"interval": "1h",  "period": "730d", "min_bars": 300, "max_bars": 100},
    "1D": {"interval": "1d",  "period": "3y",   "min_bars": 120, "max_bars": 30},
    "1W": {"interval": "1wk", "period": "10y",  "min_bars": 60,  "max_bars": 26},
    "1M": {"interval": "1mo", "period": "max",  "min_bars": 36,  "max_bars": 12},
}

def stamp(ts, intraday=None):
    """Break date; includes the time for hourly candles."""
    ts = pd.Timestamp(ts)
    if ts.hour or ts.minute:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def out_path(base, tf):
    return base if tf == "1D" else base.replace(".json", "_" + tf + ".json")
STALE_DAYS    = 10      # skip ETFs whose last price is older than this vs the freshest ETF


def load_etfs():
    html = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+GLOBAL_ETFS\s*=\s*(\[.*?\]);\s*\n", html, re.S)
    if not m:
        raise SystemExit("Could not find the GLOBAL_ETFS list inside index.html")
    arr = json.loads(m.group(1))
    seen, out = set(), []
    for a in arr:
        t = str(a.get("t", "")).strip()
        if t and t not in seen:
            seen.add(t)
            out.append({"t": t, "n": a.get("n", t), "cat": a.get("cat", "Other"), "vol": a.get("vol", 0)})
    print("ETFs loaded from index.html:", len(out))
    return out


def analyse(dates, high, low, close, L, max_bars):
    """Same Dow-Theory state machine as before, but the pivots are found in one vectorised pass
    (a pivot at bar p becomes known at bar p+L) so the 1H scan stays fast."""
    n = len(close)
    if n < 2 * L + 2:
        return None
    ph_val = np.full(n, np.nan)
    pl_val = np.full(n, np.nan)
    win = 2 * L + 1
    wh = sliding_window_view(high, win)
    wl = sliding_window_view(low, win)
    ch, cl = high[L: n - L], low[L: n - L]
    is_ph = (ch == wh.max(axis=1)) & ((wh == ch[:, None]).sum(axis=1) == 1)
    is_pl = (cl == wl.min(axis=1)) & ((wl == cl[:, None]).sum(axis=1) == 1)
    for k in np.nonzero(is_ph)[0]:
        ph_val[k + 2 * L] = ch[k]            # pivot bar = k+L, confirmed at k+2L
    for k in np.nonzero(is_pl)[0]:
        pl_val[k + 2 * L] = cl[k]

    lastPH = prevPH = lastPL = prevPL = np.nan
    trend = 0
    last = None   # (direction, bar_index, level_broken, stop_reference)
    for t in range(n):
        v = ph_val[t]
        if v == v:
            prevPH, lastPH = lastPH, v
        v = pl_val[t]
        if v == v:
            prevPL, lastPL = lastPL, v

        if trend == 0 and not (math.isnan(prevPH) or math.isnan(prevPL)):
            if lastPH > prevPH and lastPL > prevPL:
                trend = 1
            elif lastPH < prevPH and lastPL < prevPL:
                trend = -1

        if trend == 1 and not math.isnan(lastPL) and close[t] < lastPL:
            trend = -1
            last = ("Bearish", t, float(lastPL), float(lastPH))
        elif trend == -1 and not math.isnan(lastPH) and close[t] > lastPH:
            trend = 1
            last = ("Bullish", t, float(lastPH), float(lastPL))

    if last is None:
        return None
    direction, idx, level, ref = last
    days_since = n - 1 - idx
    if days_since > max_bars:
        return None
    after = close[idx + 1:]
    if len(after):                              # drop if price later closed back through the stop reference
        if direction == "Bullish" and after.min() < ref:
            return None
        if direction == "Bearish" and after.max() > ref:
            return None
    cmp_ = float(close[-1])
    return {
        "direction": direction,
        "cmp": round(cmp_, 2),
        "level": round(level, 2),
        "pct_from_level": round((cmp_ - level) / level * 100, 2),
        "ref": round(ref, 2),
        "pct_to_ref": round((cmp_ - ref) / ref * 100, 2),
        "trigger_date": stamp(dates[idx]),
        "days_since": int(days_since),
    }


def run_timeframe(tf, etfs):
    cfg = TIMEFRAMES[tf]
    variants = {str(L): [] for L in SWING_OPTIONS}
    loaded = []   # (etf, dates, h, l, c)
    failed = 0
    for i in range(0, len(etfs), CHUNK):
        part = etfs[i: i + CHUNK]
        tickers = [x["t"] for x in part]
        print(f"[{tf}] downloading {i + 1}-{i + len(part)} of {len(etfs)}", flush=True)
        data = None
        for attempt in range(2):
            try:
                data = yf.download(tickers, period=cfg["period"], interval=cfg["interval"],
                                   group_by="ticker", auto_adjust=True, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as ex:
                print("  chunk failed:", ex)
            time.sleep(5)
        if data is None or not len(data):
            failed += len(part)
            continue
        for x in part:
            try:
                df = data[x["t"]] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < cfg["min_bars"]:
                    failed += 1
                    continue
                loaded.append((x, df.index, *(df[k].to_numpy(float) for k in ("High", "Low", "Close"))))
            except Exception:
                failed += 1
        time.sleep(1)

    freshest = max((d[-1] for _, d, *_ in loaded), default=None)
    stale_days = STALE_DAYS if tf in ("1H", "1D") else 45     # weekly / monthly candles are dated at the period start
    scanned = 0
    for x, dates, h, l, c in loaded:
        if freshest is not None and (freshest - dates[-1]).days > stale_days:
            failed += 1
            continue
        scanned += 1
        for L in SWING_OPTIONS:
            r = analyse(dates, h, l, c, L, cfg["max_bars"])
            if r:
                variants[str(L)].append({"symbol": x["t"], "name": x["n"], "cat": x["cat"], "vol": x["vol"], **r})

    for k in variants:
        variants[k].sort(key=lambda r: (r["days_since"], r["symbol"]))
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    payload = {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"),
        "scanned": scanned,
        "failed": failed,
        "params": {"timeframe": tf, "swing_bars": DEFAULT_SWING, "options": SWING_OPTIONS, "max_days": cfg["max_bars"]},
        "results": variants[str(DEFAULT_SWING)],
        "variants": variants,
    }
    os.makedirs("data", exist_ok=True)
    path = out_path(OUT_FILE, tf)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"[{tf}] done. scanned={scanned} skipped/failed={failed} signals(default)={len(payload['results'])} -> {path}")


def main():
    tfs = [a.upper() for a in sys.argv[1:]] or ["1D"]
    bad = [t for t in tfs if t not in TIMEFRAMES]
    if bad:
        raise SystemExit("Unknown timeframe(s): %s. Use any of %s" % (bad, list(TIMEFRAMES)))
    etfs = load_etfs()
    for tf in tfs:
        run_timeframe(tf, etfs)


if __name__ == "__main__":
    main()
