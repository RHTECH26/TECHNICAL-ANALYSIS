"""
Relative Strength data for the "Relative Strength" tab of index.html.

Writes two files (the page computes everything else in the browser, so the person can pick any
High date / Low date without a new scan):
  data/rs_stocks.json      daily closes of the Indian stock universe (same stock list the COT scan uses)
  data/rs_benchmarks.json  daily closes of the benchmark indices (Nasdaq Composite, S&P 500, Nifty 50, Nifty 500)
  data/rs_groups.json      which NSE stocks belong to each NSE index (Nifty 50, 500, Midcap 150 ...) and each sector /
                           thematic index (Bank, IT, Realty, FMCG, Defence ...), for the Index and Sector filters

Global ETFs and Indian ETFs need no new scan: the tab reads the price history files that
fetch_global_etf_data.py / fetch_etf_data.py already write.

Run:  python scan_relative_strength.py
"""
import csv, io, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from scan_dow_trend import load_symbols     # same stock-list file the COT scan reads

PERIOD     = "14mo"      # ~300 trading days: covers a 52-week high plus room for earlier High / Low dates
CHUNK      = 100
BENCH_PERIOD = "max"   # benchmarks keep their full daily history (small files), so any past date can be compared
MAX_STALE  = 10          # drop stocks whose last price is more than this many days older than the freshest one
MIN_POINTS = 60
GROUPS_OUT = "data/rs_groups.json"
STOCK_OUT  = "data/rs_stocks.json"
BENCH_OUT  = "data/rs_benchmarks.json"

BENCHMARKS = {           # id used by the page -> (Yahoo symbol, display name)
    "IXIC":   ("^IXIC",   "Nasdaq Composite"),
    "GSPC":   ("^GSPC",   "S&P 500"),
    "NSEI":   ("^NSEI",   "Nifty 50"),
    "CRSLDX": ("^CRSLDX", "Nifty 500"),
}


def day_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def pick_close(data, tk, multi):
    """Close column of one ticker out of a yf.download() result."""
    try:
        df = data[tk] if multi else data
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(float)
        s.index = day_index(s.index)
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return None


def clean(vals):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 2) for v in vals]


def build_stocks(frames, names, failed):
    closes = pd.concat(frames, axis=1).sort_index()
    closes = closes.dropna(how="all").ffill(limit=5)
    freshest = closes.index.max()
    stocks = []
    for sym in closes.columns:
        col = closes[sym]
        valid = col.dropna()
        if len(valid) < MIN_POINTS or (freshest - valid.index[-1]).days > MAX_STALE or np.isnan(col.iloc[-1]):
            failed += 1
            continue
        stocks.append({"s": sym, "n": names.get(sym, sym), "c": clean(col.tolist())})
    return [d.strftime("%Y-%m-%d") for d in closes.index], stocks, failed


# ---------------------------------------------------------------- NSE index membership (Index and Sector filters)
URL_BASES = ["https://niftyindices.com/IndexConstituent/",
             "https://www.niftyindices.com/IndexConstituent/",
             "https://archives.nseindia.com/content/indices/"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/csv,*/*"}

# id, label shown on the page, kind, candidate file names on niftyindices.com (the first one that downloads is used)
INDEX_GROUPS = [
    ("nifty50",        "Nifty 50",                  "index", ["ind_nifty50list.csv"]),
    ("niftynext50",    "Nifty Next 50",             "index", ["ind_niftynext50list.csv"]),
    ("nifty100",       "Nifty 100",                 "index", ["ind_nifty100list.csv"]),
    ("nifty200",       "Nifty 200",                 "index", ["ind_nifty200list.csv"]),
    ("nifty500",       "Nifty 500",                 "index", ["ind_nifty500list.csv"]),
    ("midcap150",      "Nifty Midcap 150",          "index", ["ind_niftymidcap150list.csv"]),
    ("smallcap250",    "Nifty Smallcap 250",        "index", ["ind_niftysmallcap250list.csv"]),
    ("largemidcap250", "Nifty LargeMidcap 250",     "index", ["ind_niftylargemidcap250list.csv"]),
    ("midsmallcap400", "Nifty MidSmallcap 400",     "index", ["ind_niftymidsmallcap400list.csv"]),
    ("microcap250",    "Nifty Microcap 250",        "index", ["ind_niftymicrocap250_list.csv", "ind_niftymicrocap250list.csv"]),
    ("totalmarket",    "Nifty 750 (Total Market)",  "index", ["ind_niftytotalmarket_list.csv", "ind_niftytotalmarketlist.csv"]),
]
SECTOR_GROUPS = [
    ("bank",          "Bank",                    "sector", ["ind_niftybanklist.csv"]),
    ("privatebank",   "Private Bank",            "sector", ["ind_nifty_privatebanklist.csv", "ind_niftyprivatebanklist.csv"]),
    ("psubank",       "PSU Bank",                "sector", ["ind_niftypsubanklist.csv"]),
    ("finservice",    "Financial Services",      "sector", ["ind_niftyfinancelist.csv"]),
    ("capmarkets",    "Capital Markets",         "sector", ["ind_niftycapitalmarkets_list.csv", "ind_niftycapitalmarketslist.csv"]),
    ("it",            "IT",                      "sector", ["ind_niftyitlist.csv"]),
    ("auto",          "Auto",                    "sector", ["ind_niftyautolist.csv"]),
    ("fmcg",          "FMCG",                    "sector", ["ind_niftyfmcglist.csv"]),
    ("consdur",       "Consumer Durables",       "sector", ["ind_niftyconsumerdurableslist.csv"]),
    ("consumption",   "India Consumption",       "sector", ["ind_niftyconsumptionlist.csv"]),
    ("pharma",        "Pharma",                  "sector", ["ind_niftypharmalist.csv"]),
    ("healthcare",    "Healthcare",              "sector", ["ind_niftyhealthcarelist.csv"]),
    ("metal",         "Metal",                   "sector", ["ind_niftymetallist.csv"]),
    ("realty",        "Realty",                  "sector", ["ind_niftyrealtylist.csv"]),
    ("housing",       "Housing",                 "sector", ["ind_niftyhousing_list.csv", "ind_niftyhousinglist.csv"]),
    ("energy",        "Energy",                  "sector", ["ind_niftyenergylist.csv"]),
    ("oilgas",        "Oil & Gas",               "sector", ["ind_niftyoilgaslist.csv"]),
    ("media",         "Media",                   "sector", ["ind_niftymedialist.csv"]),
    ("infra",         "Infrastructure",          "sector", ["ind_niftyinfralist.csv"]),
    ("commodities",   "Commodities",             "sector", ["ind_niftycommoditieslist.csv"]),
    ("defence",       "India Defence",           "sector", ["ind_niftyindiadefence_list.csv", "ind_niftyindiadefencelist.csv"]),
    ("manufacturing", "India Manufacturing",     "sector", ["ind_niftyindiamanufacturing_list.csv", "ind_niftyindiamanufacturinglist.csv"]),
    ("digital",       "India Digital",           "sector", ["ind_niftyindiadigital_list.csv", "ind_niftyindiadigitallist.csv"]),
    ("logistics",     "Transport & Logistics",   "sector", ["ind_niftytransportationandlogistics_list.csv", "ind_niftytransportationlogisticslist.csv"]),
    ("chemicals",     "Chemicals",               "sector", ["ind_niftychemicals_list.csv", "ind_niftychemicalslist.csv"]),
    ("services",      "Services Sector",         "sector", ["ind_niftyservicelist.csv"]),
    ("mnc",           "MNC",                     "sector", ["ind_niftymnclist.csv"]),
    ("pse",           "PSE",                     "sector", ["ind_niftypselist.csv"]),
    ("cpse",          "CPSE",                    "sector", ["ind_niftycpse_list.csv", "ind_niftycpselist.csv"]),
]


def parse_symbols(text):
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
    if not rows:
        return []
    key = next((k for k in rows[0].keys() if k and k.strip().lower() == "symbol"), None)
    if key is None:
        return []
    out = []
    for r in rows:
        sym = (r.get(key) or "").strip().upper()
        if sym and not sym.startswith("DUMMY"):
            out.append(sym)
    return list(dict.fromkeys(out))


def fetch_group(group):
    gid, label, kind, files = group
    last_err = "no file name worked"
    for fname in files:
        for base in URL_BASES:
            try:
                req = urllib.request.Request(base + fname, headers=UA)
                text = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
                syms = parse_symbols(text)
                if len(syms) >= 5:
                    return {"id": gid, "label": label, "kind": kind, "n": len(syms), "symbols": syms, "file": fname}, None
                last_err = f"{fname}: only {len(syms)} symbols"
            except Exception as e:
                last_err = f"{fname}: {type(e).__name__}"
    return None, f"{label} ({last_err})"


def build_groups(groups=INDEX_GROUPS + SECTOR_GROUPS, fetch=fetch_group):
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(fetch, groups))
    out, failed = [], []
    for res, err in results:
        if res:
            out.append(res)
        else:
            failed.append(err)
    have = {g["id"]: g for g in out}
    # Nifty 750 = Nifty 500 + Nifty Microcap 250, in case NSE's own file name could not be read
    if "totalmarket" not in have and "nifty500" in have and "microcap250" in have:
        syms = list(dict.fromkeys(have["nifty500"]["symbols"] + have["microcap250"]["symbols"]))
        out.append({"id": "totalmarket", "label": "Nifty 750 (Total Market)", "kind": "index", "n": len(syms), "symbols": syms, "file": "built from Nifty 500 + Microcap 250"})
        failed = [f for f in failed if not f.startswith("Nifty 750")]
    order = {g[0]: i for i, g in enumerate(groups)}
    out.sort(key=lambda g: order.get(g["id"], 999))
    return out, failed


def main():
    universe = load_symbols()
    names = {s: n for s, n in universe}
    syms = [s for s, _ in universe]
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    stamp = ist.strftime("%d %b %Y, %H:%M IST")
    os.makedirs("data", exist_ok=True)

    # ---- benchmarks (small file, written first so it is available even if the stock download struggles)
    bench = {}
    for bid, (ysym, label) in BENCHMARKS.items():
        try:
            data = yf.download(ysym, period=BENCH_PERIOD, interval="1d", auto_adjust=True, progress=False)
            s = pick_close(data, ysym, multi=isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels > 1 and ysym in data.columns.get_level_values(0))
            if s is None or s.dropna().empty:
                s = pick_close(data, ysym, multi=False)
            s = s.dropna()
            if len(s) < MIN_POINTS:
                print("benchmark skipped (too little data):", label)
                continue
            bench[bid] = {"n": label, "d": [d.strftime("%Y-%m-%d") for d in s.index], "c": clean(s.tolist())}
            print("benchmark ok:", label, len(s), "bars")
        except Exception as e:
            print("benchmark failed:", label, e)
        time.sleep(1)
    with open(BENCH_OUT, "w") as f:
        json.dump({"updated": stamp, "bench": bench}, f, separators=(",", ":"))

    # ---- NSE index / sector membership (small file; a problem here never stops the stock data below)
    try:
        grp, grp_failed = build_groups()
        with open(GROUPS_OUT, "w") as f:
            json.dump({"updated": stamp, "groups": grp, "failed": grp_failed}, f, separators=(",", ":"))
        print(f"groups: {len(grp)} NSE indices / sectors saved" + (f" | could not read: {'; '.join(grp_failed)}" if grp_failed else ""))
    except Exception as e:
        print("groups skipped:", e)

    # ---- stocks
    frames, failed = [], 0
    for i in range(0, len(syms), CHUNK):
        part = syms[i:i + CHUNK]
        tickers = [s + ".NS" for s in part]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(syms)}", flush=True)
        data = None
        for attempt in range(2):
            try:
                data = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as e:
                print("  chunk failed:", e)
            time.sleep(5)
        if data is None or not len(data):
            failed += len(part)
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for sym, tk in zip(part, tickers):
            s = pick_close(data, tk, multi)
            if s is None or s.dropna().empty:
                failed += 1
                continue
            frames.append(s.rename(sym))
        time.sleep(1)

    if not frames:
        raise SystemExit("No stock data downloaded - nothing written.")
    dates, stocks, failed = build_stocks(frames, names, failed)
    with open(STOCK_OUT, "w") as f:
        json.dump({"updated": stamp, "scanned": len(stocks), "failed": failed,
                   "dates": dates, "stocks": stocks}, f, separators=(",", ":"))
    print(f"Done. stocks={len(stocks)} failed={failed} dates={len(dates)} benchmarks={list(bench)}")


if __name__ == "__main__":
    main()
