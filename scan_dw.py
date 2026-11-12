#!/usr/bin/env python3
"""
Dorsey Wright relative strength (the "relative strength matrix").

Every security in a universe is compared with every other one. For a pair (A, B) a Point & Figure chart is drawn of the price RATIO A / B
(3-box reversal, 1% boxes by default). If that relative-strength chart is in a column of X's, A is outperforming B; in a column of O's, B is
outperforming A. A security's SCORE is the percentage of the other securities it is outperforming; the highest score is rank 1.

  stocks      every NSE stock in the stock list (the file the other scans read)
  india_etf   every Indian ETF in index.html (TREE)
  global_etf  every ETF in index.html (GLOBAL_ETFS, many exchanges; local-currency prices)

Writes data/dw_stocks.json, data/dw_india_etf.json, data/dw_global_etf.json:
  {"updated", "asof", "universe", "n", "box_pct", "reversal", "rows": [[symbol, name, score %, market, category], ...]}   sorted from the highest score down.
  (market and category come from the ETF lists in index.html; they are blank for stocks, which the page groups by NSE index and sector.)

The same run also writes the DW QUANT rank (data/dwq_stocks.json, dwq_india_etf.json, dwq_global_etf.json), the method of the "DW RS" sheet:
  RS      = price / benchmark price              (benchmark: Nifty 500 for Indian stocks and ETFs, Nasdaq Composite for global ETFs)
  ROC n   = RS now / RS n months ago - 1            for n = 1, 3, 6, 12  (counted back from the latest day, so the dates move forward every day;
                                                    a weekend or holiday takes the NEXT trading day)
  SCORE % = 40% x 12M ROC + 30% x 6M ROC + 20% x 3M ROC + 10% x 1M ROC
  {"updated", "asof", "universe", "benchmark", "n", "weights", "dates": {"1M": date used, ...}, "rows": [[symbol, name, score %, 1M ROC %, 3M ROC %, 6M ROC %, 12M ROC %, market, category], ...]}
Usage:  python scan_dw.py            (all three)       python scan_dw.py stocks india_etf
"""
import json, os, re, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    from curl_cffi import requests as cffi_requests
    SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    SESSION = None

BOX_PCT = float(os.environ.get("DW_BOX_PCT", 1.0))          # box size of the ratio chart, in percent
REVERSAL = int(os.environ.get("DW_REVERSAL", 3))             # boxes needed to reverse a column
PERIOD = os.environ.get("DW_PERIOD", "2y")                   # daily history to download
MIN_BARS = int(os.environ.get("DW_MIN_BARS", 250))           # a security needs this many daily prices to be ranked
STALE_DAYS = int(os.environ.get("DW_STALE_DAYS", 7))         # a security whose last price is older than this (before the newest date) is left out
BLOCK = int(os.environ.get("DW_BLOCK", 1_500_000))           # pairs processed at a time (memory)
CHUNK = int(os.environ.get("DW_CHUNK", 100))
OUT_DIR = os.environ.get("DW_OUT_DIR", "data")
INDEX_FILE = os.environ.get("INDEX_FILE", "index.html")
UNIVERSES = {"stocks": "Indian Stocks", "india_etf": "Indian ETFs", "global_etf": "Global ETFs"}
MARKETS = {"TO": "Canada", "KS": "Korea", "NS": "India", "DE": "Germany", "AX": "Australia", "T": "Japan", "L": "UK", "SA": "Brazil", "HK": "Hong Kong",
           "SW": "Switzerland", "PA": "France", "NZ": "New Zealand", "MX": "Mexico", "SI": "Singapore", "BK": "Thailand", "CO": "Denmark",
           "TW": "Taiwan", "MI": "Italy", "WA": "Poland", "MC": "Spain"}
BENCH = {"stocks": ("^CRSLDX", "Nifty 500"), "india_etf": ("^CRSLDX", "Nifty 500"), "global_etf": ("^IXIC", "Nasdaq Composite")}
PERIODS = [("1M", 1), ("3M", 3), ("6M", 6), ("12M", 12)]     # calendar months
WEIGHTS = [float(x) for x in os.environ.get("DWQ_WEIGHTS", "10,20,30,40").split(",")]      # for 1M, 3M, 6M, 12M (percent)


# ============================================================ the Dorsey Wright calculation
def pair_blocks(n, block):
    """index arrays (I, J) of every pair i < j, in blocks of at most `block` pairs"""
    i = 0
    while i < n - 1:
        cnt, i1 = 0, i
        while i1 < n - 1 and cnt + (n - 1 - i1) <= block:
            cnt += n - 1 - i1
            i1 += 1
        if i1 == i:
            i1 = i + 1
        rows = np.arange(i, i1)
        lens = n - 1 - rows
        total = int(lens.sum())
        I = np.repeat(rows, lens).astype(np.int32)
        starts = np.concatenate(([0], np.cumsum(lens)))[:-1]
        J = (np.arange(total) - np.repeat(starts, lens) + I + 1).astype(np.int32)
        yield I, J
        i = i1


def pf_state(L, I, J, box_pct, rev):
    """Final column of the P&F chart of ratio i/j for every pair: +1 = X (i outperforming), -1 = O, 0 = no column yet.
    L is (T, N) of log prices, NaN where a security has no price."""
    T = L.shape[0]
    inv = 1.0 / np.log1p(box_pct / 100.0)
    m = len(I)
    d = np.zeros(m, np.int8)
    ext = np.zeros(m, np.int64)             # top of an X column / bottom of an O column (in boxes)
    base = np.zeros(m, np.int64)            # the box where the chart started
    hasb = np.zeros(m, bool)
    for t in range(T):
        row = L[t]
        xi, xj = row[I], row[J]
        valid = np.isfinite(xi) & np.isfinite(xj)
        if not valid.any():
            continue
        fb = np.floor(np.where(valid, (xi - xj) * inv, 0.0)).astype(np.int64)
        init = valid & ~hasb
        base = np.where(init, fb, base)
        hasb |= valid
        und = valid & hasb & (d == 0) & ~init
        mx = valid & (d == 1)
        mo = valid & (d == -1)
        up = und & (fb >= base + 1)
        dn = und & (fb <= base - 1)
        x_ext = mx & (fb > ext)
        x_flip = mx & ~x_ext & (fb <= ext - rev)
        o_ext = mo & (fb < ext)
        o_flip = mo & ~o_ext & (fb >= ext + rev)
        ext = np.where(up | dn | x_ext | x_flip | o_ext | o_flip, fb, ext)
        d = np.where(up | o_flip, 1, np.where(dn | x_flip, -1, d)).astype(np.int8)
    return d


def dw_scores(L, box_pct=BOX_PCT, rev=REVERSAL, block=BLOCK):
    """Score % of every column of L: the share of the other securities it outperforms (a pair with no column yet counts half each)."""
    n = L.shape[1]
    wins = np.zeros(n)
    for I, J in pair_blocks(n, block):
        d = pf_state(L, I, J, box_pct, rev)
        r = np.where(d == 1, 1.0, np.where(d == -1, 0.0, 0.5))
        wins += np.bincount(I, weights=r, minlength=n) + np.bincount(J, weights=1.0 - r, minlength=n)
    return wins / (n - 1) * 100.0 if n > 1 else wins


# ============================================================ universes and data
def page_list(var_name):
    text = open(INDEX_FILE, encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, text, re.S)
    if not m:
        raise RuntimeError("could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def build_universe(uid):
    """([(yahoo ticker, display symbol, name)], {symbol: (market, category)}).  Stocks have no market / category here: the page groups them by NSE index and sector."""
    out, meta = [], {}
    if uid == "stocks":
        from scan_dow_trend import load_symbols
        out = [(sym + ".NS", sym, name) for sym, name in load_symbols()]
    elif uid == "india_etf":
        for a in page_list("TREE"):
            t = str(a.get("t", "")).strip().upper()
            if t:
                out.append((t + ".NS", t, a.get("sub") or t))
                meta[t] = ("India", a.get("cat") or "Other")
    elif uid == "global_etf":
        for a in page_list("GLOBAL_ETFS"):
            t = str(a.get("t", "")).strip()
            if t:
                out.append((t, t, a.get("n", t)))
                suf = t.rsplit(".", 1)[1] if "." in t else ""
                meta[t] = (MARKETS.get(suf, "Other") if suf else "US", a.get("cat") or "Other")
    seen, res = set(), []
    for u in out:
        if u[0] not in seen:
            seen.add(u[0])
            res.append(u)
    return res, meta


def close_of(data, ticker):
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if ticker in list(cols.get_level_values(0)):
            return data[ticker]["Close"]
        close = data["Close"]
        return close.iloc[:, 0] if hasattr(close, "columns") else close
    return data["Close"]


def download(tickers, tries=3):
    kw = dict(period=PERIOD, interval="1d", group_by="ticker", auto_adjust=True, threads=(SESSION is None), progress=False)
    if SESSION is not None:
        kw["session"] = SESSION
    for attempt in range(tries):
        try:
            data = yf.download(tickers, **kw)
            if data is not None and len(data):
                return data
        except Exception as e:
            print(f"  download attempt {attempt + 1} failed: {type(e).__name__}")
        time.sleep(5 * (attempt + 1))
    return None


def lookback_row(index, end, months, max_gap_days=7):
    """Row of the first trading day on or AFTER (end - months): a weekend or a holiday takes the next trading day.
    None when the data does not reach back that far."""
    target = end - pd.DateOffset(months=months)
    i = int(index.searchsorted(target, side="left"))
    if i >= len(index) or (index[i] - target).days > max_gap_days:
        return None
    return i


def roc_rows(df, bser, rows):
    """ROC of the RS line (price / benchmark) from the given rows to the last row, in %, for every column of df (rows: one per period)."""
    rs_now = df.iloc[-1] / bser.iloc[-1]
    roc = pd.DataFrame(index=df.columns, columns=[p for p, _ in PERIODS], dtype=float)
    for (name, _), i in zip(PERIODS, rows):
        if i is None or not np.isfinite(bser.iloc[i]) or bser.iloc[i] <= 0:
            roc[name] = np.nan
            continue
        roc[name] = (rs_now / (df.iloc[i] / bser.iloc[i]) - 1.0) * 100.0
    return roc


def roc_table(df, bench):
    """ROC of the relative strength line over 1, 3, 6 and 12 calendar months back from the last date (it moves forward every day)."""
    end = df.index[-1]
    bser = bench.reindex(df.index).ffill()
    rows = [lookback_row(df.index, end, m) for _, m in PERIODS]
    return roc_rows(df, bser, rows), rows


def quant_scores(roc, weights=WEIGHTS):
    """Score % = the weighted sum of the four ROCs (weights for 1M, 3M, 6M, 12M); only securities with all four are scored."""
    ok = roc.dropna()
    w = np.array(weights[:ok.shape[1]], float) / 100.0
    return (ok * w).sum(axis=1)


def load_prices(uni):
    series = {}
    t_start = time.time()
    for i in range(0, len(uni), CHUNK):
        part = uni[i:i + CHUNK]
        print(f"  downloading {i + 1}-{i + len(part)} of {len(uni)}  ({(time.time() - t_start) / 60:.1f} min so far, {len(series)} usable so far)", flush=True)
        data = download([u[0] for u in part])
        if data is None:
            print("   chunk gave no data")
            continue
        for ysym, sym, name in part:
            try:
                s = close_of(data, ysym).dropna()
            except Exception:
                continue
            if len(s) >= MIN_BARS:
                ix = pd.DatetimeIndex(s.index)
                s = pd.Series(s.to_numpy(float), index=(ix.tz_localize(None) if ix.tz is not None else ix).normalize())
                series[sym] = s[~s.index.duplicated(keep="last")]
        time.sleep(1.0)
    print(f"  download finished in {(time.time() - t_start) / 60:.1f} min: {len(series)} of {len(uni)} securities have enough history", flush=True)
    return series


def build_one(uid):
    print(f"\n===== Dorsey Wright: {UNIVERSES[uid]} =====", flush=True)
    uni, meta = build_universe(uid)
    names = {u[1]: u[2] for u in uni}
    print("Universe:", len(uni))
    series = load_prices(uni)
    if len(series) < 30:
        raise RuntimeError(f"too little price data came back ({len(series)} securities)")
    df = pd.DataFrame(series).sort_index()
    newest = df.index.max()
    last_valid = df.apply(lambda c: c.last_valid_index())
    keep = [c for c in df.columns if (newest - last_valid[c]).days <= STALE_DAYS and df[c].notna().sum() >= MIN_BARS]
    df = df[keep].ffill(limit=5)
    print("Ranked securities:", len(keep), "| days:", len(df), "|", df.index[0].date(), "to", df.index[-1].date(), flush=True)
    L = np.log(df.to_numpy(float))
    t0 = time.time()
    score = dw_scores(L)
    print(f"  scored in {time.time() - t0:.0f}s", flush=True)
    mc = lambda c: list(meta.get(c, ("", "")))                      # [market, category] (blank for stocks)
    rows = sorted(([c, names.get(c, c), round(float(sc), 2)] + mc(c) for c, sc in zip(keep, score)), key=lambda r: (-r[2], r[0]))
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    stamp = ist.strftime("%d %b %Y, %H:%M IST")
    payload = {"updated": stamp, "asof": str(newest.date()), "universe": UNIVERSES[uid], "n": len(rows),
               "box_pct": BOX_PCT, "reversal": REVERSAL, "rows": rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"dw_{uid}.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Done {UNIVERSES[uid]}: {len(rows)} ranked | top: {', '.join(r[0] + ' ' + str(r[2]) for r in rows[:3])} | {os.path.getsize(path) // 1024} KB")
    # ---- the DW quant rank (the "DW RS" sheet), from the same prices ----
    try:
        bt, bname = BENCH[uid]
        bdata = download([bt])
        if bdata is None:
            raise RuntimeError("could not download " + bname)
        bser = close_of(bdata, bt).dropna()
        bix = pd.DatetimeIndex(bser.index)
        bser = pd.Series(bser.to_numpy(float), index=(bix.tz_localize(None) if bix.tz is not None else bix).normalize())
        bser = bser[~bser.index.duplicated(keep="last")]
        roc, rows = roc_table(df, bser)
        sc = quant_scores(roc)
        used = {p: (str(df.index[i].date()) if i is not None else None) for (p, _), i in zip(PERIODS, rows)}
        q_rows = sorted(([c, names.get(c, c), round(float(sc[c]), 2)] + [round(float(roc.loc[c, p]), 2) for p, _ in PERIODS] + mc(c) for c in sc.index), key=lambda r: (-r[2], r[0]))
        q = {"updated": stamp, "asof": str(newest.date()), "universe": UNIVERSES[uid], "benchmark": bname, "n": len(q_rows), "periods": [p for p, _ in PERIODS],
             "weights": WEIGHTS, "dates": used, "rows": q_rows}
        qpath = os.path.join(OUT_DIR, f"dwq_{uid}.json")
        with open(qpath, "w") as f:
            json.dump(q, f, separators=(",", ":"))
        print(f"Done DW quant {UNIVERSES[uid]}: {len(q_rows)} ranked vs {bname} | top: {', '.join(r[0] + ' ' + str(r[2]) for r in q_rows[:3])} | {os.path.getsize(qpath) // 1024} KB")
    except Exception as e:
        print(f"!! DW quant for {UNIVERSES[uid]} failed: {type(e).__name__}: {e}")


def main(argv=None):
    if yf is None:
        raise SystemExit("yfinance is not installed")
    ids = [a for a in (argv if argv is not None else sys.argv[1:]) if a in UNIVERSES] or list(UNIVERSES)
    bad = []
    for uid in ids:
        try:
            build_one(uid)
        except Exception as e:
            bad.append(uid)
            print(f"!! {UNIVERSES[uid]} failed: {type(e).__name__}: {e}")
    if len(bad) == len(ids):
        raise SystemExit("no universe could be scored")
    if bad:
        print("Finished with problems, check the log above:", ", ".join(bad))


if __name__ == "__main__":
    main()
