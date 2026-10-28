#!/usr/bin/env python3
"""
COTRS scanner: Dow Theory COT buy signals that are (1) ABOVE the stock's previous COT buy and (2) beating the respective index on BOTH sides.

  stocks      every stock in the repo's stock list (the same list the COT scan reads)   benchmark: Nifty 500
  india_etf   every Indian ETF in index.html (TREE)                                       benchmark: Nifty 500
  global_etf  every ETF in index.html (GLOBAL_ETFS, 4,000+ funds on many exchanges)        benchmark: Nasdaq Composite

A stock qualifies when its daily COT turned bullish (a close above the last lower high) AND
  * the level it broke is higher than the level broken by its previous COT buy signal, and
  * on that day it had gained more than the benchmark since the benchmark's latest confirmed swing high AND since its latest
    confirmed swing low (swings known on that day only).
The result also shows whether it is STILL beating the index on both sides today.

Writes data/cotrs_stocks.json, data/cotrs_india_etf.json, data/cotrs_global_etf.json (read by the COTRS tab under Scanner).
Usage:  python scan_cotrs.py            (all three)       python scan_cotrs.py stocks india_etf
"""
import json, os, re, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from model_portfolio import dow_engine, bench_anchors      # the same engine as the model portfolio

SWING     = int(os.environ.get("COTRS_SWING", 5))          # COT swing size (bars each side)
RS_LEFT   = int(os.environ.get("COTRS_RS_LEFT", 10))        # benchmark swing size (bars each side) for the relative strength anchors
RS_RIGHT  = int(os.environ.get("COTRS_RS_RIGHT", 10))
PERIOD    = os.environ.get("COTRS_PERIOD", "3y")            # daily history to download
MAX_DAYS  = int(os.environ.get("COTRS_MAX_DAYS", 60))       # keep signals that fired within this many candles
MIN_BARS  = 200
CHUNK     = 100
STALE_DAYS = 5
OUT_DIR   = os.environ.get("COTRS_OUT_DIR", "data")

UNIVERSES = {
    "stocks":     {"label": "Indian Stocks", "bench": "^CRSLDX", "bench_name": "Nifty 500"},
    "india_etf":  {"label": "Indian ETFs", "bench": "^CRSLDX", "bench_name": "Nifty 500"},
    "global_etf": {"label": "Global ETFs", "bench": "^IXIC", "bench_name": "Nasdaq Composite"},
}
MARKETS = {"TO": "Canada", "KS": "Korea", "DE": "Germany", "AX": "Australia", "T": "Japan", "L": "UK", "SA": "Brazil", "HK": "Hong Kong",
           "SW": "Switzerland", "PA": "France", "NZ": "New Zealand", "MX": "Mexico", "SI": "Singapore", "BK": "Thailand", "CO": "Denmark",
           "TW": "Taiwan", "MI": "Italy", "WA": "Poland", "MC": "Spain", "NS": "India"}


# ============================================================ universes
def page_list(var_name):
    text = open("index.html", encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, text, re.S)
    if not m:
        raise RuntimeError("could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def build_universe(uid):
    """[(yahoo ticker, display symbol, name, market)]"""
    out = []
    if uid == "stocks":
        from scan_dow_trend import load_symbols            # the stock list file the COT scan uses
        for sym, name in load_symbols():
            out.append((sym + ".NS", sym, name, "India"))
    elif uid == "india_etf":
        for a in page_list("TREE"):
            t = str(a.get("t", "")).strip().upper()
            if t:
                out.append((t + ".NS", t, a.get("sub") or t, "India"))
    elif uid == "global_etf":
        for a in page_list("GLOBAL_ETFS"):
            t = str(a.get("t", "")).strip()
            if t:
                suf = t.rsplit(".", 1)[1] if "." in t else ""
                out.append((t, t, a.get("n", t), MARKETS.get(suf, "Other") if suf else "US"))
    seen, res = set(), []
    for u in out:
        if u[0] not in seen:
            seen.add(u[0])
            res.append(u)
    return res


# ============================================================ data
def _clean_index(df):
    ix = pd.DatetimeIndex(df.index)
    df.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
    return df[~df.index.duplicated(keep="last")]


def get_benchmark(ticker):
    d = yf.download(ticker, period=PERIOD, interval="1d", auto_adjust=False, progress=False)
    if d is None or len(d) < 300:
        raise RuntimeError("could not download the benchmark %s" % ticker)
    ser = d["Close"].iloc[:, 0] if hasattr(d["Close"], "columns") else d["Close"]
    ser = pd.Series(ser.to_numpy(float), index=pd.DatetimeIndex(ser.index))
    return _clean_index(pd.DataFrame({"Close": ser}))["Close"].dropna()


def download(tickers):
    frames, failed = {}, 0
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        print(f"  downloading {i + 1}-{i + len(part)} of {len(tickers)}", flush=True)
        data = None
        for attempt in range(3):
            try:
                data = yf.download(part, period=PERIOD, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as e:
                print("   chunk failed:", type(e).__name__)
            time.sleep(4 * (attempt + 1))
        if data is None or not len(data):
            failed += len(part)
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for t in part:
            try:
                df = (data[t] if multi else data)[["High", "Low", "Close"]].dropna()
                df = _clean_index(df)
                if len(df) >= MIN_BARS:
                    frames[t] = df
                else:
                    failed += 1
            except Exception:
                failed += 1
        time.sleep(1)
    return frames, failed


# ============================================================ the scan
def scan_one(df, bench_dates, bclose, ph_anchor, pl_anchor, swing=SWING, max_days=MAX_DAYS):
    """Returns the most recent qualifying COTRS signal of one stock (a dict) or None."""
    dates = df.index.values
    H, L, C = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    n = len(C)
    _, bull, _, lvl, stop = dow_engine(H, L, C, swing)
    bi = np.nonzero(bull)[0]
    if len(bi) < 2:
        return None
    cs = np.array(pd.Series(C, index=df.index).reindex(bench_dates, method="ffill").to_numpy(), dtype=float)
    cs[: np.searchsorted(bench_dates, dates[0])] = np.nan
    bpos = np.searchsorted(bench_dates, dates, side="right") - 1

    def excess(j, anchor):
        a = anchor[j]
        if a < 0 or j < 0:
            return None
        c0, b0 = cs[a], bclose[a]
        if not (c0 == c0 and c0 > 0 and b0 > 0 and cs[j] == cs[j]):
            return None
        return ((cs[j] / c0 - 1) - (bclose[j] / b0 - 1)) * 100

    jn = len(bench_dates) - 1
    exH_now, exL_now = excess(jn, ph_anchor), excess(jn, pl_anchor)
    for k in range(len(bi) - 1, 0, -1):                       # newest first
        b = bi[k]
        if n - 1 - b > max_days:
            break
        if not lvl[b] > lvl[bi[k - 1]]:                       # the buy level must be above the previous buy level
            continue
        j = bpos[b]
        h, l = excess(j, ph_anchor), excess(j, pl_anchor)
        if h is None or l is None or not (h > 0 and l > 0):    # beating the index on BOTH sides on the signal day
            continue
        prev = float(lvl[bi[k - 1]])
        return {"date": str(pd.Timestamp(dates[b]).date()), "days_since": int(n - 1 - b), "cmp": round(float(C[-1]), 4),
                "level": round(float(lvl[b]), 4), "prev_level": round(prev, 4), "up_pct": round((float(lvl[b]) / prev - 1) * 100, 2),
                "from_level_pct": round((float(C[-1]) / float(lvl[b]) - 1) * 100, 2), "stop": round(float(stop[b]), 4),
                "exH": round(h, 2), "exL": round(l, 2),
                "exH_now": None if exH_now is None else round(exH_now, 2), "exL_now": None if exL_now is None else round(exL_now, 2),
                "beating_now": bool(exH_now is not None and exL_now is not None and exH_now > 0 and exL_now > 0)}
    return None


def market_open_now(uid):
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    if uid == "global_etf":
        return 0 <= now.hour < 21                              # Asia, Europe or the US is in session
    ist = now + timedelta(hours=5, minutes=30)
    return (ist.hour, ist.minute) >= (9, 15) and (ist.hour, ist.minute) <= (15, 45)


def build_one(uid, bench_cache):
    meta = UNIVERSES[uid]
    print(f"\n===== COTRS: {meta['label']} (benchmark {meta['bench_name']}) =====", flush=True)
    if meta["bench"] not in bench_cache:
        bench_cache[meta["bench"]] = get_benchmark(meta["bench"])
    bench = bench_cache[meta["bench"]]
    bdates, bclose = bench.index.values, bench.to_numpy(float)
    ph_a, pl_a = bench_anchors(bclose, RS_LEFT, RS_RIGHT)
    uni = build_universe(uid)
    print("Universe:", len(uni))
    frames, failed = download([u[0] for u in uni])
    freshest = max((df.index[-1] for df in frames.values()), default=None)
    results, scanned = [], 0
    for yt, sym, name, mkt in uni:
        df = frames.get(yt)
        if df is None:
            continue
        if freshest is not None and (freshest - df.index[-1]).days > STALE_DAYS:
            failed += 1
            continue
        scanned += 1
        r = scan_one(df, bdates, bclose, ph_a, pl_a)
        if r:
            results.append({"symbol": sym, "name": name, "mkt": mkt, **r})
    results.sort(key=lambda r: (r["days_since"], -r["up_pct"], r["symbol"]))
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    payload = {"updated": ist.strftime("%d %b %Y, %H:%M IST"), "asof": str(bench.index[-1].date()), "universe": meta["label"], "benchmark": meta["bench_name"],
               "scanned": scanned, "failed": failed, "market_open": market_open_now(uid),
               "params": {"swing": SWING, "rs_swing": f"{RS_LEFT}/{RS_RIGHT}", "max_days": MAX_DAYS}, "results": results}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"cotrs_{uid}.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    now_ok = sum(1 for r in results if r["beating_now"])
    print(f"Done {meta['label']}: scanned {scanned}, skipped {failed} | {len(results)} COTRS signals within {MAX_DAYS} candles, {now_ok} still beating the index on both sides | {os.path.getsize(path) // 1024} KB")


def main(argv=None):
    if yf is None:
        raise SystemExit("yfinance is not installed")
    ids = [a for a in (argv if argv is not None else sys.argv[1:]) if a in UNIVERSES] or list(UNIVERSES)
    cache, bad = {}, []
    for uid in ids:
        try:
            build_one(uid, cache)
        except Exception as e:
            bad.append(uid)
            print(f"!! {uid} failed: {type(e).__name__}: {e}")
    if len(bad) == len(ids):
        raise SystemExit("no universe could be scanned")
    if bad:
        print("Finished with problems, check the log above:", ", ".join(bad))


if __name__ == "__main__":
    main()
