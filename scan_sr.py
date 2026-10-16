"""
S&R screener - which stocks / ETFs are sitting AT a swing high (resistance) or AT a swing low (support).

  python scan_sr.py stocks   -> data/sr_stocks.json   (Indian stocks, list = your universe CSV)
  python scan_sr.py etf      -> data/sr_etf.json      (Global ETFs, list = GLOBAL_ETFS inside index.html)

Swings are the same confirmed pivots as the Dow Theory / COT indicator (N bars each side, N = 3, 5, 8).
For every symbol and swing size we store the latest swing high (resistance) and latest swing low
(support), how far price is from each, how many other swings sit at the same level (touches),
how old the swing is, and the Dow trend. The web page filters this by "near resistance / near support".
"""
import glob, json, math, os, re, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

SWING_OPTIONS = [3, 5, 8]
PERIOD        = "2y"
CHUNK         = 100
MIN_BARS      = 120
TOUCH_LOOKBACK = 250        # bars searched for other swings at the same level
TOUCH_ATR     = 0.5         # "same level" = within 0.5 x ATR
KEEP_MIN, KEEP_MAX = -3.0, 10.0   # keep a symbol if price is within this % band of a swing level
STALE_DAYS    = 10
INDEX_FILE    = "index.html"


# ------------------------------------------------------------------ universes
def _read_table(path):
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception:
        return None


def find_universe():
    env = os.environ.get("UNIVERSE_FILE")
    if env:
        return env
    best, best_n = None, 0
    for pattern in ("**/*.csv", "**/*.xlsx", "**/*.xls"):
        for f in glob.glob(pattern, recursive=True):
            if f.startswith(".git") or "node_modules" in f or f.startswith("index_lists"):
                continue
            df = _read_table(f)
            if df is None:
                continue
            if any(str(c).strip().lower() in ("symbol", "ticker", "symbols") for c in df.columns) and len(df) > best_n:
                best, best_n = f, len(df)
    if not best:
        raise SystemExit("No stock-list file (CSV/Excel with a 'Symbol' column) found in the repo.")
    return best


def load_stocks():
    path = find_universe()
    print("Universe file:", path)
    df = _read_table(path)
    col = next(c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker", "symbols"))
    desc = next((c for c in df.columns if str(c).strip().lower() in ("description", "name", "company")), None)
    syms = (df[col].astype(str).str.strip().str.upper()
            .str.replace(r"^NSE:", "", regex=True).str.replace(r"\.NS$", "", regex=True))
    names = df[desc].astype(str).str.strip() if desc else syms
    seen, out = set(), []
    for s, n in zip(syms, names):
        if s and s != "NAN" and s not in seen:
            seen.add(s)
            out.append({"t": s + ".NS", "s": s, "n": n, "c": "", "v": 0})
    print("Symbols:", len(out))
    return out


def load_etfs():
    html = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+GLOBAL_ETFS\s*=\s*(\[.*?\]);\s*\n", html, re.S)
    if not m:
        raise SystemExit("Could not find the GLOBAL_ETFS list inside index.html")
    seen, out = set(), []
    for a in json.loads(m.group(1)):
        t = str(a.get("t", "")).strip()
        if t and t not in seen:
            seen.add(t)
            out.append({"t": t, "s": t, "n": a.get("n", t), "c": a.get("cat", "Other"), "v": a.get("vol", 0)})
    print("ETFs:", len(out))
    return out


# ------------------------------------------------------------------ engine
def pivots(high, low, L):
    n = len(high)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    if n < 2 * L + 2:
        return ph, pl
    hv, lv = sliding_window_view(high, 2 * L + 1), sliding_window_view(low, 2 * L + 1)
    ch, cl = hv[:, L], lv[:, L]
    ph[L: n - L] = (ch == hv.max(axis=1)) & ((hv == ch[:, None]).sum(axis=1) == 1)
    pl[L: n - L] = (cl == lv.min(axis=1)) & ((lv == cl[:, None]).sum(axis=1) == 1)
    return ph, pl


def dow_trend(high, low, close, ph, pl, L):
    """Dow Theory trend at the last bar: 1 up, -1 down, 0 undefined (same rule as the COT indicator)."""
    n = len(close)
    nan = float("nan")
    lastPH = prevPH = lastPL = prevPL = nan
    trend = 0
    for t in range(n):
        p = t - L
        if p >= L:
            if ph[p]:
                prevPH, lastPH = lastPH, high[p]
            if pl[p]:
                prevPL, lastPL = lastPL, low[p]
        if trend == 0 and not (math.isnan(prevPH) or math.isnan(prevPL)):
            if lastPH > prevPH and lastPL > prevPL:
                trend = 1
            elif lastPH < prevPH and lastPL < prevPL:
                trend = -1
        if trend == 1 and not math.isnan(lastPL) and close[t] < lastPL:
            trend = -1
        elif trend == -1 and not math.isnan(lastPH) and close[t] > lastPH:
            trend = 1
    return trend


def analyse(dates, high, low, close, L):
    n = len(close)
    ph, pl = pivots(high, low, L)
    ih, il = np.nonzero(ph)[0], np.nonzero(pl)[0]
    if len(ih) == 0 or len(il) == 0:
        return None
    pc = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    atr = float(tr[-14:].mean())
    c = float(close[-1])
    if c <= 0 or atr <= 0:
        return None

    hi_i, lo_i = int(ih[-1]), int(il[-1])
    SH, SL = float(high[hi_i]), float(low[lo_i])
    dr = (SH - c) / c * 100.0          # +ve: price is below the swing high
    ds = (c - SL) / c * 100.0          # +ve: price is above the swing low
    keep_r = KEEP_MIN <= dr <= KEEP_MAX
    keep_s = KEEP_MIN <= ds <= KEEP_MAX
    if not (keep_r or keep_s):
        return None

    recent_h = ih[ih >= n - TOUCH_LOOKBACK]
    recent_l = il[il >= n - TOUCH_LOOKBACK]
    ht = int((np.abs(high[recent_h] - SH) <= TOUCH_ATR * atr).sum())
    lt = int((np.abs(low[recent_l] - SL) <= TOUCH_ATR * atr).sum())
    return {
        "p": round(c, 2),
        "sh": round(SH, 2), "shd": str(dates[hi_i])[:10], "shn": n - 1 - hi_i, "sht": max(ht, 1), "dr": round(dr, 2),
        "sl": round(SL, 2), "sld": str(dates[lo_i])[:10], "sln": n - 1 - lo_i, "slt": max(lt, 1), "ds": round(ds, 2),
        "tr": dow_trend(high, low, close, ph, pl, L),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stocks"
    if mode not in ("stocks", "etf"):
        raise SystemExit("usage: python scan_sr.py stocks|etf")
    items = load_stocks() if mode == "stocks" else load_etfs()
    out_file = f"data/sr_{mode}.json"

    loaded, failed = [], 0
    for i in range(0, len(items), CHUNK):
        part = items[i: i + CHUNK]
        tickers = [x["t"] for x in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(items)}", flush=True)
        try:
            data = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("  chunk failed:", e)
            failed += len(part)
            continue
        for x in part:
            try:
                df = data[x["t"]] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < MIN_BARS:
                    failed += 1
                    continue
                loaded.append((x, df.index, *(df[k].to_numpy(float) for k in ("High", "Low", "Close"))))
            except Exception:
                failed += 1
        time.sleep(1)

    freshest = max((d[-1] for _, d, *_ in loaded), default=None)
    variants = {str(L): [] for L in SWING_OPTIONS}
    scanned = 0
    for x, dates, h, l, c in loaded:
        if freshest is not None and (freshest - dates[-1]).days > STALE_DAYS:
            failed += 1
            continue
        scanned += 1
        for L in SWING_OPTIONS:
            r = analyse(dates, h, l, c, L)
            if r:
                variants[str(L)].append({"s": x["s"], "n": x["n"], "c": x["c"], "v": x["v"], **r})

    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    payload = {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"),
        "data_date": str(freshest)[:10] if freshest is not None else "",
        "scanned": scanned, "failed": failed,
        "params": {"options": SWING_OPTIONS, "touch_atr": TOUCH_ATR, "touch_lookback": TOUCH_LOOKBACK},
        "variants": variants,
    }
    os.makedirs("data", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Done. scanned={scanned} skipped={failed} rows(5)={len(variants['5'])} size={os.path.getsize(out_file)//1024}KB")


if __name__ == "__main__":
    main()
