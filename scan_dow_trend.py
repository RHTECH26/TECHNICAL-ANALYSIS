"""
Dow Theory - Change of Trend scanner for NSE stocks (daily chart).
Same logic as the TradingView indicator:
  * swing high / low = pivot with N bars on each side (confirmed N bars later)
  * uptrend   -> turns BEARISH when a daily CLOSE is below the last higher low
  * downtrend -> turns BULLISH when a daily CLOSE is above the last lower high
Timeframes: 1H, 1D, 1W, 1M (run:  python scan_dow_trend.py 1D 1W 1M   or   python scan_dow_trend.py 1H).
Output: data/dow_trend.json (1D) and data/dow_trend_1H.json / _1W.json / _1M.json
        (read by the COT tab of index.html)
"""
import glob, json, math, os, time
from datetime import datetime, timezone, timedelta
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import yfinance as yf

OUT_FILE      = "data/dow_trend.json"
SWING_OPTIONS = [3, 5, 8]      # selectable on the webpage
DEFAULT_SWING = 5
CHUNK         = 100

# ---- timeframes -------------------------------------------------------------------------
# interval / history Yahoo is asked for, minimum candles needed, and how many candles back a
# signal may have fired and still be listed. "1D" keeps the original output file name so
# nothing that already reads it breaks; the others get a suffix.
TIMEFRAMES = {
    "1H": {"interval": "1h",  "period": "90d",  "min_bars": 200, "max_bars": 100},   # ~600 hourly bars is plenty to find swings and a fresh signal
    "1D": {"interval": "1d",  "period": "3y",   "min_bars": 120, "max_bars": 60},
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


def _read_table(path):
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception:
        return None


def find_universe():
    """Use $UNIVERSE_FILE if set, else search the whole repo for the CSV/Excel file that has a
    Symbol column and the most rows (the stock list)."""
    env = os.environ.get("UNIVERSE_FILE")
    if env:
        return env
    best, best_n, seen = None, 0, []
    for pattern in ("**/*.csv", "**/*.xlsx", "**/*.xls"):
        for f in glob.glob(pattern, recursive=True):
            if f.startswith((".git", "data/dow_trend")) or "node_modules" in f:
                continue
            seen.append(f)
            df = _read_table(f)
            if df is None:
                continue
            if any(str(c).strip().lower() in ("symbol", "ticker", "symbols") for c in df.columns) and len(df) > best_n:
                best, best_n = f, len(df)
    if not best:
        raise SystemExit("No stock-list file (CSV/Excel with a 'Symbol' column) found anywhere in the repo. "
                         "Spreadsheet-type files seen: " + (", ".join(seen) if seen else "none"))
    return best


def load_symbols():
    path = find_universe()
    print("Universe file:", path)
    df = _read_table(path)
    col = next(c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker", "symbols"))
    desc = next((c for c in df.columns if str(c).strip().lower() in ("description", "name", "company")), None)
    syms = (df[col].astype(str).str.strip().str.upper()
            .str.replace(r"^NSE:", "", regex=True).str.replace(r"\.NS$", "", regex=True))
    names = df[desc].astype(str).str.strip() if desc else syms
    out = [(s, n) for s, n in zip(syms, names) if s and s != "NAN"]
    seen, uniq = set(), []
    for s, n in out:
        if s not in seen:
            seen.add(s); uniq.append((s, n))
    print("Symbols loaded:", len(uniq))
    return uniq


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


def run_timeframe(tf, universe):
    cfg = TIMEFRAMES[tf]
    t_start = time.time()
    variants = {str(L): [] for L in SWING_OPTIONS}
    scanned, failed = 0, 0
    # was the NSE session open when this scan started? Then the latest candle is still forming and
    # every signal is based on the live price (it only becomes final after the 3:30 PM close).
    ist0 = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    mins0 = ist0.hour * 60 + ist0.minute
    market_open = ist0.weekday() < 5 and (9 * 60 + 15) <= mins0 <= (15 * 60 + 30)
    for i in range(0, len(universe), CHUNK):
        part = universe[i: i + CHUNK]
        tickers = [s + ".NS" for s, _ in part]
        print(f"[{tf}] downloading {i + 1}-{i + len(part)} of {len(universe)}", flush=True)
        data = None
        for attempt in range(2):
            try:
                data = yf.download(tickers, period=cfg["period"], interval=cfg["interval"],
                                   group_by="ticker", auto_adjust=True, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as e:
                print("  chunk failed:", e)
            time.sleep(5)
        if data is None or not len(data):
            failed += len(part)
            continue
        for (sym, name), tk in zip(part, tickers):
            try:
                df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < cfg["min_bars"]:
                    failed += 1
                    continue
                h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
                scanned += 1
                for L in SWING_OPTIONS:
                    r = analyse(df.index, h, l, c, L, cfg["max_bars"])
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
        "market_open": market_open,
        "params": {"timeframe": tf, "swing_bars": DEFAULT_SWING, "options": SWING_OPTIONS, "max_days": cfg["max_bars"]},
        "results": variants[str(DEFAULT_SWING)],
        "variants": variants,
    }
    os.makedirs("data", exist_ok=True)
    path = out_path(OUT_FILE, tf)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"[{tf}] took {(time.time() - t_start) / 60:.1f} min")
    print(f"[{tf}] done. scanned={scanned} failed={failed} signals(default)={len(payload['results'])} -> {path}")


def main():
    tfs = [a.upper() for a in sys.argv[1:]] or ["1D"]
    bad = [t for t in tfs if t not in TIMEFRAMES]
    if bad:
        raise SystemExit("Unknown timeframe(s): %s. Use any of %s" % (bad, list(TIMEFRAMES)))
    universe = load_symbols()
    for tf in tfs:
        run_timeframe(tf, universe)


if __name__ == "__main__":
    main()
