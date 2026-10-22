"""
Dow Theory (COT) breadth for Nifty 50 / Nifty 500 / Nifty Midcap 150 / Nifty Smallcap 250.

For EVERY trading day in the history, each stock gets a trend state from the Dow Theory rule
(same rule as the TradingView indicator), using only information available on that day:
  * swing high / low = pivot with N bars each side (confirmed N bars later - no look-ahead)
  * uptrend   -> turns BEARISH when a daily CLOSE is below the last higher low
  * downtrend -> turns BULLISH when a daily CLOSE is above the last lower high
Then, per index and per day, we count how many member stocks are in an uptrend (bullish),
how many in a downtrend (bearish), and how many fresh bullish / bearish changes of trend
happened that day.

AD LINE: per index and per day, advances = members that closed above their own previous
close, declines = members that closed below it; net = advances - declines (stored under
indices.<key>.ad = {adv, dec, net}, aligned to the same "dates" array).

The S&P 500 NET AD LINE is stored separately under "sp500_ad" with its own US trading calendar
(sp500_ad = {dates, adv, dec, net}); the S&P 500 is not part of the Nifty COT counts above.

Output: data/cot_breadth.json  (read by the Market Breadth tab of index.html)
NOTE: uses TODAY's index members for the whole history (no point-in-time membership).
"""
import csv, io, json, math, os, time, urllib.request
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

OUT_FILE      = "data/cot_breadth.json"
LIST_DIR      = "index_lists"
SWING_OPTIONS = [3, 5, 8]
START_DATE    = "2001-01-01"   # was PERIOD = "10y" -- now pulls ~25 years of history
CHUNK         = 100
MIN_BARS      = 120

INDICES = {
    "nifty50":     ("Nifty 50",           ["ind_nifty50list.csv"]),
    "nifty500":    ("Nifty 500",          ["ind_nifty500list.csv"]),
    "midcap150":   ("Nifty Midcap 150",   ["ind_niftymidcap150list.csv"]),
    "smallcap250": ("Nifty Smallcap 250", ["ind_niftysmallcap250list.csv"]),
}
URL_BASES = ["https://niftyindices.com/IndexConstituent/",
             "https://www.niftyindices.com/IndexConstituent/",
             "https://archives.nseindia.com/content/indices/"]
SP500_URLS = ["https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/csv,*/*"}


# ---------------------------------------------------------------- index members
def _parse_symbols(text):
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
    if not rows:
        return []
    key = next((k for k in rows[0].keys() if k and k.strip().lower() == "symbol"), None)
    if key is None:
        return []
    out = []
    for r in rows:
        s = (r.get(key) or "").strip().upper()
        if s and s != "DUMMY" and not s.startswith("DUMMY"):
            out.append(s)
    return list(dict.fromkeys(out))


def load_members(key):
    fname = INDICES[key][1][0]
    local = os.path.join(LIST_DIR, f"{key}.csv")
    for base in URL_BASES:
        for attempt in range(2):
            try:
                req = urllib.request.Request(base + fname, headers=UA)
                text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
                syms = _parse_symbols(text)
                if len(syms) >= 20:
                    os.makedirs(LIST_DIR, exist_ok=True)
                    open(local, "w", encoding="utf-8").write(text)
                    print(f"{key}: {len(syms)} members downloaded from {base}")
                    return syms
            except Exception as e:
                print(f"  {key}: {base} attempt {attempt + 1} failed ({type(e).__name__})")
                time.sleep(2)
    if os.path.exists(local):
        syms = _parse_symbols(open(local, encoding="utf-8").read())
        if len(syms) >= 20:
            print(f"{key}: {len(syms)} members read from saved file {local}")
            return syms
    raise SystemExit(f"Could not get the member list for {INDICES[key][0]}. Download "
                     f"{URL_BASES[0]}{fname} in your browser and upload it to the repo as {local}")


# ---------------------------------------------------------------- Dow engine
def dow_states(high, low, close, L):
    """Per-bar trend state (+1 up, -1 down, 0 undefined) and change-of-trend events (+1 bullish, -1 bearish)."""
    n = len(close)
    state = np.zeros(n, np.int8)
    ev = np.zeros(n, np.int8)
    if n < 2 * L + 2:
        return state, ev
    hv, lv = sliding_window_view(high, 2 * L + 1), sliding_window_view(low, 2 * L + 1)
    ch, cl = hv[:, L], lv[:, L]
    piv_h = np.zeros(n, bool)
    piv_l = np.zeros(n, bool)
    piv_h[L: n - L] = (ch == hv.max(axis=1)) & ((hv == ch[:, None]).sum(axis=1) == 1)
    piv_l[L: n - L] = (cl == lv.min(axis=1)) & ((lv == cl[:, None]).sum(axis=1) == 1)
    nan = float("nan")
    lastPH = prevPH = lastPL = prevPL = nan
    trend = 0
    for t in range(n):
        p = t - L
        if p >= L:
            if piv_h[p]:
                prevPH, lastPH = lastPH, high[p]
            if piv_l[p]:
                prevPL, lastPL = lastPL, low[p]
        if trend == 0 and not (math.isnan(prevPH) or math.isnan(prevPL)):
            if lastPH > prevPH and lastPL > prevPL:
                trend = 1
            elif lastPH < prevPH and lastPL < prevPL:
                trend = -1
        if trend == 1 and not math.isnan(lastPL) and close[t] < lastPL:
            trend = -1
            ev[t] = -1
        elif trend == -1 and not math.isnan(lastPH) and close[t] > lastPH:
            trend = 1
            ev[t] = 1
        state[t] = trend
    return state, ev


# ---------------------------------------------------------------- S&P 500 NET AD LINE
def load_sp500_members():
    """Today's S&P 500 members (Yahoo style symbols, e.g. BRK-B)."""
    local = os.path.join(LIST_DIR, "sp500.csv")
    for url in SP500_URLS:
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=UA)
                text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
                syms = _parse_symbols(text)
                if len(syms) >= 400:
                    os.makedirs(LIST_DIR, exist_ok=True)
                    open(local, "w", encoding="utf-8").write(text)
                    print(f"sp500: {len(syms)} members downloaded")
                    return [x.replace(".", "-") for x in syms]
            except Exception as e:
                print(f"  sp500: download attempt {attempt + 1} failed ({type(e).__name__})")
                time.sleep(2)
    if os.path.exists(local):
        syms = _parse_symbols(open(local, encoding="utf-8").read())
        if len(syms) >= 400:
            print(f"sp500: {len(syms)} members read from saved file {local}")
            return [x.replace(".", "-") for x in syms]
    try:   # last resort: Wikipedia (needs lxml)
        req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA)
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        tbl = pd.read_html(io.StringIO(html))[0]
        syms = [str(x).strip().upper().replace(".", "-") for x in tbl["Symbol"]]
        if len(syms) >= 400:
            print(f"sp500: {len(syms)} members read from Wikipedia")
            return syms
    except Exception as e:
        print("  sp500: Wikipedia fallback failed:", type(e).__name__)
    raise RuntimeError("no S&P 500 member list available")


def compute_sp500_ad():
    """Advances / declines per day for the S&P 500, on the US trading calendar."""
    members = load_sp500_members()
    adv_cols, dec_cols = [], []
    for i in range(0, len(members), CHUNK):
        part = members[i: i + CHUNK]
        print(f"S&P 500: downloading {i + 1}-{i + len(part)} of {len(members)}", flush=True)
        try:
            data = yf.download(part, start=START_DATE, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("  chunk failed:", e)
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for sym in part:
            try:
                df = data[sym] if multi else data
                c = df["Close"].dropna()
                if len(c) < MIN_BARS:
                    continue
                chg = c.diff()          # vs the stock's own previous close
                adv_cols.append((chg > 0).astype(np.int8).rename(sym))
                dec_cols.append((chg < 0).astype(np.int8).rename(sym))
            except Exception:
                continue
        time.sleep(1)
    if len(adv_cols) < 200:
        raise RuntimeError("too little S&P 500 price data came back (%d stocks)" % len(adv_cols))
    ADVF, DECF = pd.concat(adv_cols, axis=1), pd.concat(dec_cols, axis=1)
    coverage = ADVF.notna().sum(axis=1)                 # stocks that traded that day
    adv = ADVF.fillna(0).sum(axis=1).astype(int)
    dec = DECF.fillna(0).sum(axis=1).astype(int)
    keep = (coverage >= 100) & ((adv + dec) > 0)        # skip thin early years and non-sessions
    adv, dec = adv[keep], dec[keep]
    return {"dates": [d.strftime("%Y-%m-%d") for d in adv.index],
            "adv": adv.tolist(), "dec": dec.tolist(), "net": (adv - dec).tolist(),
            "members": len(members), "with_data": len(adv_cols)}


# ---------------------------------------------------------------- main
def main():
    members = {k: load_members(k) for k in INDICES}
    universe = list(dict.fromkeys(s for v in members.values() for s in v))
    print("Universe:", len(universe), "stocks")

    frames = {}
    failed = 0
    for i in range(0, len(universe), CHUNK):
        part = universe[i: i + CHUNK]
        tickers = [s + ".NS" for s in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(universe)}", flush=True)
        try:
            data = yf.download(tickers, start=START_DATE, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("  chunk failed:", e)
            failed += len(part)
            continue
        for s, tk in zip(part, tickers):
            try:
                df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) >= MIN_BARS:
                    frames[s] = df[["High", "Low", "Close"]]
                else:
                    failed += 1
            except Exception:
                failed += 1
        time.sleep(1)
    if len(frames) < 50:
        raise SystemExit("Too little price data came back from Yahoo Finance - try again later.")

    dates = None
    for df in frames.values():
        dates = df.index if dates is None else dates.union(df.index)
    dates = dates.sort_values()
    nd = len(dates)
    print("Trading days:", nd, "| stocks with data:", len(frames))

    syms = list(frames)
    row = {s: i for i, s in enumerate(syms)}
    NODATA = np.int8(-9)
    S = {L: np.full((len(syms), nd), NODATA, np.int8) for L in SWING_OPTIONS}
    E = {L: np.zeros((len(syms), nd), np.int8) for L in SWING_OPTIONS}
    ADV = np.zeros((len(syms), nd), np.int8)   # 1 = stock closed UP vs its own previous close
    DEC = np.zeros((len(syms), nd), np.int8)   # 1 = stock closed DOWN vs its own previous close
    for s, df in frames.items():
        h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
        # Advance / Decline: compare each close with that stock's own previous close, so
        # holidays and suspended days never create fake advances or declines.
        chg = df["Close"].diff()
        ADV[row[s]] = (chg > 0).reindex(dates).fillna(False).to_numpy().astype(np.int8)
        DEC[row[s]] = (chg < 0).reindex(dates).fillna(False).to_numpy().astype(np.int8)
        for L in SWING_OPTIONS:
            st, ev = dow_states(h, l, c, L)
            ser = pd.Series(st.astype(float), index=df.index).reindex(dates).ffill(limit=5)
            S[L][row[s]] = ser.fillna(-9).to_numpy().astype(np.int8)
            E[L][row[s]] = pd.Series(ev, index=df.index).reindex(dates).fillna(0).to_numpy().astype(np.int8)

    # trim the early part of the history where Nifty 500 coverage is thin
    n500 = [row[s] for s in members["nifty500"] if s in row]
    cov = (S[SWING_OPTIONS[0]][n500] != NODATA).sum(axis=0)
    start = int(np.argmax(cov >= max(20, 0.2 * len(n500)))) if (cov >= 20).any() else 0

    out_idx = {}
    for key, (name, _) in INDICES.items():
        rows_ = [row[s] for s in members[key] if s in row]
        entry = {"name": name, "members": len(members[key]), "with_data": len(rows_)}
        adv = ADV[rows_][:, start:].sum(axis=0)
        dec = DEC[rows_][:, start:].sum(axis=0)
        entry["ad"] = {                      # AD LINE: net = advances - declines (per day)
            "adv": adv.tolist(),
            "dec": dec.tolist(),
            "net": (adv.astype(int) - dec.astype(int)).tolist(),
        }
        for L in SWING_OPTIONS:
            st = S[L][rows_][:, start:]
            ev = E[L][rows_][:, start:]
            entry[f"s{L}"] = {
                "up": (st == 1).sum(axis=0).tolist(),
                "dn": (st == -1).sum(axis=0).tolist(),
                "bu": (ev == 1).sum(axis=0).tolist(),
                "be": (ev == -1).sum(axis=0).tolist(),
            }
        out_idx[key] = entry

    sp500_ad = None
    try:
        sp500_ad = compute_sp500_ad()
        print(f"S&P 500 NET AD LINE: {len(sp500_ad['dates'])} days from {sp500_ad['dates'][0]}")
    except Exception as e:      # never let the S&P 500 part stop the Nifty results from being saved
        print("S&P 500 NET AD LINE skipped:", e)

    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    payload = {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"),
        "params": {"options": SWING_OPTIONS, "years": 25, "failed": failed, "stocks": len(frames)},
        "dates": [d.strftime("%Y-%m-%d") for d in dates[start:]],
        "indices": out_idx,
    }
    if sp500_ad:
        payload["sp500_ad"] = sp500_ad
    os.makedirs("data", exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Done. {len(payload['dates'])} days written, {len(frames)} stocks, {failed} skipped. "
          f"Size {os.path.getsize(OUT_FILE) // 1024} KB")


if __name__ == "__main__":
    main()
