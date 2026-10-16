"""
S&R screener - which stocks / ETFs are sitting AT a swing high (resistance) or AT a swing low (support).

  python scan_sr.py stocks   -> data/sr_stocks.json, bo_stocks.json, rv_stocks.json   (Indian stocks, list = your universe CSV)
  python scan_sr.py etf      -> data/sr_etf.json,    bo_etf.json,    rv_etf.json      (Global ETFs, list = GLOBAL_ETFS inside index.html)

sr_*.json  = who is sitting AT a swing high / swing low (S&R tab)
bo_*.json  = who has JUST CLOSED above a swing high - long side only (Breakout tab)
rv_*.json  = who has just BOUNCED from a support level (swing low) - long side only (Reversal tab)

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
BO_WINDOW     = 20          # breakouts / breakdowns of the last 20 bars are stored
RANGE_LOOKBACK = 120        # "range high" = level is the highest high of the previous 120 bars
HI52          = 250         # 52-week = 250 bars
RV_WINDOW     = 15          # support touches of the last 15 bars are stored
RV_TOUCH_ATR  = 0.4         # low within 0.4 ATR above the level counts as a test of it
RV_PIERCE_ATR = 0.75        # a wick up to 0.75 ATR below the level is allowed if the bar closes back above it
RV_MIN_BOUNCE = 1.0         # price must have lifted at least 1 ATR off the low since the test
RV_LEFT_ATR   = 2.0         # price must have left the level by at least 2 ATR before coming back to test it


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


def analyse_bo(dates, high, low, close, vol, L):
    """Fresh closes above the latest swing high (long side only).
    A breakout = first close above the latest CONFIRMED swing high (level known before that bar)."""
    n = len(close)
    if n < 2 * L + 30:
        return None
    ph, pl = pivots(high, low, L)
    ihs, ils = np.nonzero(ph)[0], np.nonzero(pl)[0]
    pc = np.concatenate(([close[0]], close[:-1]))
    tr_ = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    atr = float(tr_[-14:].mean())
    c = float(close[-1])
    if c <= 0 or atr <= 0:
        return None

    nan = float("nan")
    lastPH = prevPH = lastPL = prevPL = nan
    trend = 0
    up = None
    start = n - 1 - BO_WINDOW
    for t in range(n):
        if t >= 1 and t >= start:
            if not math.isnan(lastPH) and close[t] > lastPH and close[t - 1] <= lastPH:
                up = (t, lastPH, trend)
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

    def vol_ratio(t):
        if t < 21:
            return None
        avg = float(vol[t - 20:t].mean())
        return round(float(vol[t]) / avg, 2) if avg > 0 and vol[t] > 0 else None

    row = {}
    if up is not None:
        t, level, tb = up
        if c > level:                                     # still holding above the level
            hi52 = close[t] > high[max(0, t - HI52):t].max()
            rng = bool(level >= 0.998 * high[max(0, t - RANGE_LOOKBACK):t].max())
            near = ihs[(ihs >= t - TOUCH_LOOKBACK) & (ihs <= t - L)]
            row.update(ud=str(dates[t])[:10], uy=n - 1 - t, ul=round(float(level), 2), ub=round(float((c - level) / level * 100), 2),
                       uv=vol_ratio(t), ut=max(1, int((np.abs(high[near] - level) <= TOUCH_ATR * atr).sum())),
                       uh=2 if hi52 else (1 if rng else 0), uk="r" if tb == -1 else ("c" if tb == 1 else ""))
    if not row:
        return None
    row["p"] = round(c, 2)
    return row


def trend_series(high, low, close, ph, pl, L):
    """Dow trend after every bar: 1 up, -1 down, 0 undefined (same rule as the COT indicator)."""
    n = len(close)
    out = np.zeros(n, np.int8)
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
        out[t] = trend
    return out


def analyse_rv(dates, high, low, close, vol, L):
    """Reversal from support (long side): price tested a confirmed swing low and turned back up.
    Support = the latest confirmed swing low, or any swing low level that has been touched at least twice.
    It must be a real retest: price had moved at least 2 ATR above the level before coming back down to it.
    Test bar t0: low within 0.4 ATR above the level (or a wick up to 0.75 ATR below it) AND the bar closes at / above the level.
    Evidence of the turn: a later close above the high of the test bar (confirmed), or the test bar itself is a
    rejection candle (closes in the upper 40% of its range). Price must still be above the level today."""
    n = len(close)
    if n < 2 * L + 40:
        return None
    ph, pl = pivots(high, low, L)
    ils, ihs = np.nonzero(pl)[0], np.nonzero(ph)[0]
    if len(ils) < 1:
        return None
    pc = np.concatenate(([close[0]], close[:-1]))
    tr_ = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    atr_arr = pd.Series(tr_).rolling(14).mean().to_numpy()
    c = float(close[-1])
    trend = trend_series(high, low, close, ph, pl, L)

    found = None
    for t0 in range(n - 1, max(n - 1 - RV_WINDOW, L + 20) - 1, -1):
        a = atr_arr[t0]
        if not (a > 0):
            continue
        conf = ils[(ils <= t0 - L) & (ils >= t0 - TOUCH_LOOKBACK)]
        if len(conf) == 0:
            continue
        best_p, best_key = None, None
        for p in conf:
            S = low[p]
            if low[t0] <= S + RV_TOUCH_ATR * a and low[t0] >= S - RV_PIERCE_ATR * a and close[t0] >= S:
                if t0 - p < 5 or high[p:t0].max() - S < RV_LEFT_ATR * a:   # a real retest: price had moved away from the level first
                    continue
                touches = int((np.abs(low[conf] - S) <= TOUCH_ATR * a).sum())
                if touches < 2 and p != conf[-1]:       # a one-touch level only counts if it is the latest swing low
                    continue
                key = (touches, S)                      # most touched level, then the one nearest the price
                if best_key is None or key > best_key:
                    best_p, best_key = p, key
        if best_p is None:
            continue
        S = float(low[best_p])
        if close[t0:].min() < S * 0.999 or c < S:       # closed below the level since the test -> not a reversal
            continue
        ll = float(low[t0:].min())
        if ll < S - 1.5 * a or (c - ll) < RV_MIN_BOUNCE * a:
            continue
        rng = high[t0] - low[t0]
        pin = bool(rng > 0 and (close[t0] - low[t0]) / rng >= 0.6)
        confirmed = bool(t0 < n - 1 and (close[t0 + 1:] > high[t0]).any())
        if not (confirmed or pin):
            continue
        found = (t0, best_p, S, ll, confirmed, best_key[0], float(atr_arr[-1]))
        break
    if found is None:
        return None

    t0, p, S, ll, confirmed, touches, atr = found
    prev = ils[ils < p]
    hl = 1 if len(prev) and S > float(low[prev[-1]]) else 0
    ctx = int(trend[t0 - 1]) if t0 >= 1 else 0
    # next resistance = lowest confirmed swing high above the price
    res = ihs[(ihs <= n - 1 - L) & (ihs >= n - 1 - TOUCH_LOOKBACK)]
    above = [float(high[i]) for i in res if high[i] > c * 1.003]
    R = min(above) if above else None
    up = round((R - c) / c * 100, 2) if R else None
    risk = c - (S - 0.5 * atr)
    rr = round((R - c) / risk, 1) if (R and risk > 0) else None
    avg = float(vol[t0 - 20:t0].mean()) if t0 >= 21 else 0.0
    return {
        "p": round(c, 2), "lv": round(S, 2), "td": str(dates[t0])[:10], "ty": n - 1 - t0,
        "ll": round(ll, 2), "bn": round((c - ll) / ll * 100, 2), "ds": round((c - S) / c * 100, 2),
        "tc": max(1, touches), "cf": 1 if confirmed else 0, "hl": hl,
        "cx": "p" if ctx == 1 else ("b" if ctx == -1 else ""),
        "sp": 1 if low[t0] < S else 0,
        "tv": round(float(vol[t0]) / avg, 2) if avg > 0 and vol[t0] > 0 else None,
        "up": up, "rr": rr, "tr": int(trend[-1]),
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
                vol = df["Volume"].fillna(0).to_numpy(float) if "Volume" in df else np.zeros(len(df))
                loaded.append((x, df.index, *(df[k].to_numpy(float) for k in ("High", "Low", "Close")), vol))
            except Exception:
                failed += 1
        time.sleep(1)

    freshest = max((d[-1] for _, d, *_ in loaded), default=None)
    variants = {str(L): [] for L in SWING_OPTIONS}
    bo_variants = {str(L): [] for L in SWING_OPTIONS}
    rv_variants = {str(L): [] for L in SWING_OPTIONS}
    scanned = 0
    for x, dates, h, l, c, vol in loaded:
        if freshest is not None and (freshest - dates[-1]).days > STALE_DAYS:
            failed += 1
            continue
        scanned += 1
        for L in SWING_OPTIONS:
            r = analyse(dates, h, l, c, L)
            if r:
                variants[str(L)].append({"s": x["s"], "n": x["n"], "c": x["c"], "v": x["v"], **r})
            b = analyse_bo(dates, h, l, c, vol, L)
            if b:
                bo_variants[str(L)].append({"s": x["s"], "n": x["n"], "c": x["c"], "v": x["v"], **b})
            rv = analyse_rv(dates, h, l, c, vol, L)
            if rv:
                rv_variants[str(L)].append({"s": x["s"], "n": x["n"], "c": x["c"], "v": x["v"], **rv})

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
    bo_payload = {k: v for k, v in payload.items() if k != "variants"}
    bo_payload["params"] = {"options": SWING_OPTIONS, "window": BO_WINDOW}
    bo_payload["variants"] = bo_variants
    bo_file = f"data/bo_{mode}.json"
    with open(bo_file, "w") as f:
        json.dump(bo_payload, f, separators=(",", ":"))
    rv_payload = {k: v for k, v in payload.items() if k != "variants"}
    rv_payload["params"] = {"options": SWING_OPTIONS, "window": RV_WINDOW}
    rv_payload["variants"] = rv_variants
    rv_file = f"data/rv_{mode}.json"
    with open(rv_file, "w") as f:
        json.dump(rv_payload, f, separators=(",", ":"))
    print(f"Done. scanned={scanned} skipped={failed} S&R rows(5)={len(variants['5'])} ({os.path.getsize(out_file)//1024}KB) | "
          f"breakout rows(5)={len(bo_variants['5'])} ({os.path.getsize(bo_file)//1024}KB) | "
          f"reversal rows(5)={len(rv_variants['5'])} ({os.path.getsize(rv_file)//1024}KB)")


if __name__ == "__main__":
    main()
