"""
Global ETF - Dow Theory Change of Trend scanner (daily chart).
Reads the ETF list straight from index.html (the GLOBAL_ETFS list used by the Global ETF Heatmap),
so the scan always matches the ETFs on your page.
Same logic as the TradingView indicator:
  * swing high / low = pivot with N bars on each side (confirmed N bars later)
  * uptrend   -> turns BEARISH when a daily CLOSE is below the last higher low
  * downtrend -> turns BULLISH when a daily CLOSE is above the last lower high
Output: data/global_etf_cot.json  (read by the "Global ETF CoT" tab of index.html)
"""
import json, math, os, re, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

INDEX_FILE    = "index.html"
OUT_FILE      = "data/global_etf_cot.json"
SWING_OPTIONS = [3, 5, 8]
DEFAULT_SWING = 5
MAX_DAYS      = 30
CHUNK         = 100
PERIOD        = "3y"
MIN_BARS      = 120
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


def analyse(dates, high, low, close, L):
    """Replays the Pine state machine bar by bar. Returns the latest change of trend, or None."""
    n = len(close)
    lastPH = prevPH = lastPL = prevPL = np.nan
    trend = 0
    last = None   # (direction, bar_index, level_broken, stop_reference)
    for t in range(n):
        p = t - L                              # pivot candidate, now confirmed by L bars to its right
        if p - L >= 0:
            wh, wl = high[p - L: t + 1], low[p - L: t + 1]
            if high[p] == wh.max() and (wh == high[p]).sum() == 1:
                prevPH, lastPH = lastPH, high[p]
            if low[p] == wl.min() and (wl == low[p]).sum() == 1:
                prevPL, lastPL = lastPL, low[p]

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
    if days_since > MAX_DAYS:
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
        "trigger_date": str(dates[idx])[:10],
        "days_since": int(days_since),
    }


def main():
    etfs = load_etfs()
    variants = {str(L): [] for L in SWING_OPTIONS}
    loaded = []   # (etf, dates, h, l, c)
    failed = 0
    for i in range(0, len(etfs), CHUNK):
        part = etfs[i: i + CHUNK]
        tickers = [e["t"] for e in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(etfs)}", flush=True)
        try:
            data = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("  chunk failed:", e)
            failed += len(part)
            continue
        for e in part:
            try:
                df = data[e["t"]] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < MIN_BARS:
                    failed += 1
                    continue
                loaded.append((e, df.index, *(df[k].to_numpy(float) for k in ("High", "Low", "Close"))))
            except Exception:
                failed += 1
        time.sleep(1)

    freshest = max((d[-1] for _, d, *_ in loaded), default=None)
    scanned = 0
    for e, dates, h, l, c in loaded:
        if freshest is not None and (freshest - dates[-1]).days > STALE_DAYS:
            failed += 1
            continue
        scanned += 1
        for L in SWING_OPTIONS:
            r = analyse(dates, h, l, c, L)
            if r:
                variants[str(L)].append({"symbol": e["t"], "name": e["n"], "cat": e["cat"], "vol": e["vol"], **r})

    for k in variants:
        variants[k].sort(key=lambda x: (x["days_since"], x["symbol"]))
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    payload = {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"),
        "scanned": scanned,
        "failed": failed,
        "params": {"swing_bars": DEFAULT_SWING, "options": SWING_OPTIONS, "max_days": MAX_DAYS},
        "results": variants[str(DEFAULT_SWING)],
        "variants": variants,
    }
    os.makedirs("data", exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Done. scanned={scanned} skipped/failed={failed} signals(default)={len(payload['results'])}")


if __name__ == "__main__":
    main()
