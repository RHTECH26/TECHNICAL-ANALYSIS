"""
Dow Theory - Change of Trend scanner for NSE stocks (daily chart).
Same logic as the TradingView indicator:
  * swing high / low = pivot with N bars on each side (confirmed N bars later)
  * uptrend   -> turns BEARISH when a daily CLOSE is below the last higher low
  * downtrend -> turns BULLISH when a daily CLOSE is above the last lower high
Output: data/dow_trend.json  (read by the "Trend Change" tab of index.html)
"""
import glob, json, math, os, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

OUT_FILE      = "data/dow_trend.json"
SWING_OPTIONS = [3, 5, 8]      # selectable on the webpage
DEFAULT_SWING = 5
MAX_DAYS      = 60             # keep signals fired within the last 60 trading days
CHUNK         = 100
PERIOD        = "3y"
MIN_BARS      = 120


def find_universe():
    """Use $UNIVERSE_FILE if set, else the root .csv with a Symbol column and most rows."""
    env = os.environ.get("UNIVERSE_FILE")
    if env:
        return env
    best, best_n = None, 0
    for f in glob.glob("*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if any(c.strip().lower() in ("symbol", "ticker") for c in df.columns) and len(df) > best_n:
            best, best_n = f, len(df)
    if not best:
        raise SystemExit("No universe CSV (with a 'Symbol' column) found in the repo root.")
    return best


def load_symbols():
    path = find_universe()
    print("Universe file:", path)
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.strip().lower() in ("symbol", "ticker"))
    desc = next((c for c in df.columns if c.strip().lower() in ("description", "name", "company")), None)
    syms = df[col].astype(str).str.strip().str.upper()
    names = df[desc].astype(str).str.strip() if desc else syms
    out = [(s, n) for s, n in zip(syms, names) if s and s != "NAN"]
    seen, uniq = set(), []
    for s, n in out:
        if s not in seen:
            seen.add(s); uniq.append((s, n))
    return uniq


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
    universe = load_symbols()
    variants = {str(L): [] for L in SWING_OPTIONS}
    scanned, failed = 0, 0
    for i in range(0, len(universe), CHUNK):
        part = universe[i: i + CHUNK]
        tickers = [s + ".NS" for s, _ in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(universe)}", flush=True)
        try:
            data = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("  chunk failed:", e)
            failed += len(part)
            continue
        for (sym, name), tk in zip(part, tickers):
            try:
                df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < MIN_BARS:
                    failed += 1
                    continue
                h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
                scanned += 1
                for L in SWING_OPTIONS:
                    r = analyse(df.index, h, l, c, L)
                    if r:
                        variants[str(L)].append({"symbol": sym, "name": name, **r})
            except Exception:
                failed += 1
        time.sleep(1)

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
    print(f"Done. scanned={scanned} failed={failed} signals(default)={len(payload['results'])}")


if __name__ == "__main__":
    main()
