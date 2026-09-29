#!/usr/bin/env python3
"""
Refreshes everything on the site except the ETF heatmaps (that's fetch_etf_data.py and
fetch_global_etf_data.py):
  - data/live.json                       sector indices, macro dashboard, FII/DII flows
  - data/<index>-price.json              each index's own daily close (4 indices)
  - data/breadth-history[-<index>].json  % of constituents above their 200-day SMA
  - data/breadth-extra.json              new 52-week highs / new 52-week lows / net (H-L)
                                          for all 4 indices, computed in the same pass

Covers 4 indices: Nifty 50, Nifty 500, Nifty Midcap 150, Nifty Smallcap 250 — matching the
Market Breadth tab's index toggle and its 4 selectable lower-panel indicators (Stocks > 200
SMA, Net Highs (H-L), 52-Week Highs, 52-Week Lows).

Every step is wrapped so one failing source never wipes out data that's already good —
each section only gets overwritten if its own fetch succeeds. No API key anywhere:
yfinance reads Yahoo Finance's public endpoints, and the FII/DII pull tries NSE's public
JSON API with a plain browser User-Agent (best-effort — NSE sometimes blocks datacenter
IPs like GitHub Actions runners; if it fails, the flows table just keeps its last value).

Run locally with:  python3 scripts/fetch_live_data.py
Runs automatically via .github/workflows/update-data.yml on a daily schedule.
"""
import json
import time
import datetime
import urllib.request

import yfinance as yf

LIVE_PATH = "data/live.json"
EXTRA_PATH = "data/breadth-extra.json"

# Only Nifty 500 has a real historical backfill (breadth_daily.json, seeded once from a
# 20-year build) — its % > 200 SMA series is append-only and never reseeded/replaced. The
# other three indices, and the new-52W-high/low/net metric for ALL FOUR indices (nothing
# has ever backfilled that), start from nothing and get "reseeded" with a longer pull
# whenever they're still thin, same idea as this file already did for Smallcap 250.
INDEX_CONFIGS = [
    {
        "key": "nifty50", "ticker": "^NSEI",
        "constituents_url": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
        "price_path": "data/nifty50-price.json",
        "pct_path": "data/breadth-history-nifty50.json",
        "pct_has_backfill": False, "pct_default_last_date": "2024-01-01",
    },
    {
        "key": "nifty500", "ticker": "^CRSLDX",
        "constituents_url": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "price_path": "data/nifty500-price.json",
        "pct_path": "data/breadth-history.json",
        "pct_has_backfill": True, "pct_default_last_date": "2006-10-20",
    },
    {
        "key": "midcap150", "ticker": "NIFTYMIDCAP150.NS",
        "constituents_url": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "price_path": "data/niftymidcap150-price.json",
        "pct_path": "data/breadth-history-midcap150.json",
        "pct_has_backfill": False, "pct_default_last_date": "2024-01-01",
    },
    {
        "key": "smallcap250", "ticker": "NIFTYSMLCAP250.NS",
        "constituents_url": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "price_path": "data/niftysc250-price.json",
        "pct_path": "data/breadth-history-sc250.json",
        "pct_has_backfill": False, "pct_default_last_date": "2024-01-01",
    },
]

SEED_YEARS = 25           # how far back to seed on first run / while an index is still "thin".
                          # "25y" isn't a valid yfinance `period` token (only 1d/5d/1mo/3mo/
                          # 6mo/1y/2y/5y/10y/ytd/max are), so this is turned into an explicit
                          # start= date instead of a period string — see seed_start_date().
                          # Not every constituent will actually have 25 years of listed
                          # history (plenty of midcap/smallcap names are far younger), so in
                          # practice this pulls "as much as exists, up to 25 years."
INCREMENTAL_PERIOD = "500d"  # ~340 trading days — comfortably more than the 252 needed for
                              # a 52-week high/low window, once an index is already seeded
SEED_THRESHOLD = 5000     # ~25y of trading days (~6,300) minus the 252-day SMA/52W warm-up,
                          # with some margin for holidays/gaps — comfortably above what a 10y
                          # seed would produce, so a site still on the old 10y seed reseeds
                          # once more to the full 25y depth, then just appends daily after that.
HIGH_LOW_WINDOW = 252     # ~52 weeks of trading sessions
SMA_WINDOW = 200
BREADTH_BATCH_SIZE = 40  # yf.download() in small batches, not one big multi-hundred-ticker
                          # call — same lesson as fetch_etf_data.py: one huge batched call
                          # (500 tickers x 10 years for Nifty 500) is far more likely to get
                          # rate-limited or come back empty than several smaller ones.
BREADTH_RETRIES = 3
BREADTH_SLEEP_BETWEEN_BATCHES = 2.0
BREADTH_SLEEP_ON_RETRY = 6.0

SECTOR_TICKERS = [
    ("FMCG", "^CNXFMCG"),
    ("PSU Bank", "^CNXPSUBANK"),
    ("Realty", "^CNXREALTY"),
    ("Bank", "^NSEBANK"),
    ("Pvt Bank", "NIFTY_PVT_BANK.NS"),
    ("Media", "^CNXMEDIA"),
    ("Fin Service", "^CNXFIN"),
    ("Oil & Gas", "NIFTY_OIL_AND_GAS.NS"),
    ("Metal", "^CNXMETAL"),
    ("Auto", "^CNXAUTO"),
    ("Consr Durbl", "NIFTY_CONSR_DURBL.NS"),
    ("Healthcare", "NIFTY_HEALTHCARE.NS"),
    ("Pharma", "^CNXPHARMA"),
    # "Midsml Health" has no Yahoo Finance ticker — left out, keeps its last static value.
    ("IT", "^CNXIT"),
]


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def pct_change_1d(ticker):
    h = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
    if len(h) < 2:
        return None
    last, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
    return round((last - prev) / prev * 100, 2)


def fetch_sectors(existing_sectors):
    by_name = {s["n"]: s for s in existing_sectors}
    for name, ticker in SECTOR_TICKERS:
        try:
            chg = pct_change_1d(ticker)
            if chg is not None:
                by_name[name] = {"n": name, "c": chg}
                print(f"  sector {name}: {chg}%")
        except Exception as e:
            print(f"  sector {name} FAILED: {e}")
        time.sleep(0.3)
    return list(by_name.values())


def fetch_macro(existing_macro):
    m = dict(existing_macro)
    try:
        h = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")
        if len(h) >= 2:
            last, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
            m["vix"] = {"v": round(last, 2), "chg": round((last - prev) / prev * 100, 2),
                        "asof": datetime.date.today().strftime("%d %b %Y")}
    except Exception as e:
        print(f"  vix FAILED: {e}")
    try:
        h = yf.Ticker("INR=X").history(period="5d", interval="1d")
        if len(h) >= 1:
            m["usdinr"] = {"v": round(float(h["Close"].iloc[-1]), 2),
                           "asof": datetime.date.today().strftime("%d %b %Y")}
    except Exception as e:
        print(f"  usdinr FAILED: {e}")
    try:
        h = yf.Ticker("BZ=F").history(period="5d", interval="1d")
        if len(h) >= 2:
            last, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
            prevm = m.get("brent", {})
            m["brent"] = {"v": round(last, 2), "chg": round(last - prev, 2),
                          "ychg_pct": prevm.get("ychg_pct", 0),
                          "asof": datetime.date.today().strftime("%d %b %Y")}
    except Exception as e:
        print(f"  brent FAILED: {e}")
    # Nifty 50 forward PE and the 10Y G-Sec yield have no free live feed — left untouched.
    return m


def fetch_flows(existing_flows):
    # Best-effort: NSE's own provisional FII/DII JSON. Frequently blocked from datacenter
    # IPs (including GitHub Actions runners) — on any failure, keep last week's figures.
    try:
        req = urllib.request.Request(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        rows = []
        for row in data:
            d = datetime.datetime.strptime(row["date"], "%d-%b-%Y").strftime("%d %b")
            rows.append({"d": d, "fii": float(row["fiiBuyValue"]) - float(row["fiiSellValue"]),
                         "dii": float(row["diiBuyValue"]) - float(row["diiSellValue"])})
        if rows:
            print(f"  flows: {len(rows)} rows from NSE")
            return rows[-7:]
    except Exception as e:
        print(f"  flows FAILED (keeping last known week): {e}")
    return existing_flows


def fetch_index_price(ticker, existing, label):
    # yf.Ticker(...).history(period="max") has repeatedly come back badly stale for these
    # NSE index symbols (^CRSLDX capped out around 2015, NIFTYSMLCAP250.NS around 2014) —
    # a Yahoo/yfinance quirk specific to that code path for these tickers, not a real data
    # gap (the index itself trades every day). yf.download() against a shorter, explicit
    # period goes through a different Yahoo endpoint and reliably returns current data
    # instead, so that's used here, with the existing series only replaced when the new
    # pull is actually fresher (never regress to something staler than what's already saved).
    series = None

    def try_download(kwargs):
        h = yf.download(ticker, interval="1d", auto_adjust=True, threads=False, progress=False, **kwargs)
        if h is None or len(h) == 0:
            return None
        closes = h["Close"]
        if hasattr(closes, "iloc") and closes.ndim > 1:
            closes = closes.iloc[:, 0]
        s = [[d.strftime("%Y-%m-%d"), round(float(c), 2)] for d, c in closes.dropna().items()]
        return s or None

    # Try the deepest pull first (an explicit 25-year start date, same depth as the breadth
    # seed below) — retried a few times, since this is the one attempt worth not giving up on
    # early — then fall back to shorter, well-tested period tokens only if that keeps failing.
    # Whatever Yahoo actually has for a given index — some go back decades, some don't exist
    # as a ticker until a few years ago — this naturally returns "up to 25 years."
    start_date = seed_start_date()
    for attempt in range(3):
        try:
            series = try_download({"start": start_date})
            if series:
                break
        except Exception as e:
            print(f"  {label} price (start={start_date}, attempt {attempt+1}/3) FAILED: {e}")
        time.sleep(4.0 * (attempt + 1))

    if not series:
        for period in ("10y", "5y", "2y", "1y"):
            try:
                series = try_download({"period": period})
                if series:
                    break
            except Exception as e:
                print(f"  {label} price (period={period}) FAILED: {e}")
    if not series:
        print(f"  {label} price: no data fetched, keeping existing ({len(existing)} rows)")
        return existing

    new_first, new_last = series[0][0], series[-1][0]
    old_first = existing[0][0] if existing else None
    old_last = existing[-1][0] if existing else None
    days_stale = (datetime.date.today() - datetime.datetime.strptime(new_last, "%Y-%m-%d").date()).days
    if days_stale > 10:
        print(f"  {label} price: fetched series is still stale (latest {new_last}, {days_stale}d old) — Yahoo may not have fresher data for this ticker right now")
    # BUG FIX: this used to only compare the *latest* date ("is the new fetch at least as
    # fresh as what we already have"), which is true on almost every run (both series are
    # "as of today"), so it kept discarding a deeper 25-year re-fetch in favor of whatever
    # shallow series was already saved. Now: only keep the existing series if it is BOTH at
    # least as deep (more/equal rows, i.e. goes back at least as far) AND at least as fresh.
    if existing and len(existing) >= len(series) and old_first <= new_first and old_last >= new_last:
        print(f"  {label} price: existing ({len(existing)} rows, {old_first}..{old_last}) already as deep and fresh as fetch ({len(series)} rows, {new_first}..{new_last}), keeping existing")
        return existing
    print(f"  {label} price: {len(series)} rows, {new_first}..{new_last} (was {len(existing)} rows, {old_first}..{old_last})")
    return series


def load_constituents(constituents_url, label):
    try:
        req = urllib.request.Request(constituents_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("utf-8-sig")
        import csv, io
        symbols = [row["Symbol"] for row in csv.DictReader(io.StringIO(text))]
        print(f"  {label}: {len(symbols)} constituents")
        return symbols
    except Exception as e:
        print(f"  {label} constituent list FAILED: {e}")
        return None


def seed_start_date():
    return (datetime.date.today() - datetime.timedelta(days=SEED_YEARS * 365)).strftime("%Y-%m-%d")


def compute_breadth_frame(tickers, label, period=None, start=None):
    # One metric-computing pass serves both % above 200-day SMA and new 52-week highs/lows/net
    # — but the price DOWNLOAD itself is done in small batches (not one big multi-hundred-
    # ticker yf.download() call), same reasoning as fetch_etf_data.py: a single huge batch
    # (500 tickers x 25 years for Nifty 500) is far more likely to get rate-limited or come
    # back empty on GitHub's runner IPs than several smaller ones with pauses between them.
    # Pass exactly one of period (e.g. "500d", a valid yfinance token) or start (an explicit
    # "YYYY-MM-DD", used for the multi-year seed pull since "25y" isn't a real yfinance period).
    import pandas as pd
    closes = {}
    n_batches = (len(tickers) - 1) // BREADTH_BATCH_SIZE + 1
    for bi in range(0, len(tickers), BREADTH_BATCH_SIZE):
        batch = tickers[bi:bi + BREADTH_BATCH_SIZE]
        batch_num = bi // BREADTH_BATCH_SIZE + 1
        raw = None
        for attempt in range(BREADTH_RETRIES):
            try:
                kwargs = dict(interval="1d", auto_adjust=True, group_by="ticker", threads=False, progress=False)
                if start:
                    kwargs["start"] = start
                else:
                    kwargs["period"] = period
                raw = yf.download(batch, **kwargs)
                if raw is not None and len(raw) > 0:
                    break
            except Exception as e:
                print(f"  {label} breadth batch {batch_num}/{n_batches} attempt {attempt+1} failed: {e}")
                raw = None
            time.sleep(BREADTH_SLEEP_ON_RETRY * (attempt + 1))
        if raw is None or len(raw) == 0:
            print(f"  {label} breadth batch {batch_num}/{n_batches}: no data")
            continue
        single = len(batch) == 1
        for t in batch:
            try:
                s = raw["Close"] if single else raw[t]["Close"]
                s = s.dropna()
                if len(s) >= SMA_WINDOW:
                    closes[t] = s
            except Exception:
                continue
        print(f"  {label} breadth batch {batch_num}/{n_batches}: {len(closes)} usable ticker(s) so far")
        time.sleep(BREADTH_SLEEP_BETWEEN_BATCHES)

    if not closes:
        print(f"  {label} breadth: no usable price series")
        return None

    df = pd.DataFrame(closes).sort_index()
    sma = df.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    valid_sma = df.notna() & sma.notna()
    above_sma = (df > sma) & valid_sma
    pct_sma = (above_sma.sum(axis=1) / valid_sma.sum(axis=1).replace(0, pd.NA) * 100)

    roll_max = df.rolling(HIGH_LOW_WINDOW, min_periods=HIGH_LOW_WINDOW).max()
    roll_min = df.rolling(HIGH_LOW_WINDOW, min_periods=HIGH_LOW_WINDOW).min()
    valid_hl = df.notna() & roll_max.notna()
    new_high = (df >= roll_max) & valid_hl
    new_low = (df <= roll_min) & valid_hl

    idx = valid_hl.index[valid_hl.any(axis=1)]
    rows = []
    for dt in idx:
        ds = dt.strftime("%Y-%m-%d")
        pct = pct_sma.get(dt)
        pct_val = round(float(pct), 2) if pct == pct else None  # NaN check
        nh, nl = int(new_high.loc[dt].sum()), int(new_low.loc[dt].sum())
        rows.append((ds, pct_val, nh, nl, nh - nl))
    print(f"  {label} breadth: {len(rows)} valid day(s) computed")
    return rows


def merge_pct(existing, rows, has_backfill, default_last_date, label):
    reseeding = (not has_backfill) and len(existing) < SEED_THRESHOLD
    pct_rows = [[ds, pct] for ds, pct, _, _, _ in rows if pct is not None]
    if reseeding:
        if len(pct_rows) > len(existing):
            print(f"  {label} pct: reseeding with deeper history ({len(pct_rows)} rows vs {len(existing)} before)")
            return pct_rows
        return existing
    last_date = existing[-1][0] if existing else default_last_date
    new_rows = [r for r in pct_rows if r[0] > last_date]
    if new_rows:
        print(f"  {label} pct: appending {len(new_rows)} new day(s)")
        return existing + new_rows
    print(f"  {label} pct: already up to date")
    return existing


def merge_extra(existing, rows, label):
    existing = existing or {"dates": [], "newHighs": [], "newLows": [], "netHL": []}
    reseeding = len(existing["dates"]) < SEED_THRESHOLD
    if reseeding:
        if len(rows) > len(existing["dates"]):
            print(f"  {label} extra: reseeding with deeper history ({len(rows)} rows vs {len(existing['dates'])} before)")
            return {
                "dates": [r[0] for r in rows],
                "newHighs": [r[2] for r in rows],
                "newLows": [r[3] for r in rows],
                "netHL": [r[4] for r in rows],
            }
        return existing
    last_date = existing["dates"][-1] if existing["dates"] else "1900-01-01"
    new_rows = [r for r in rows if r[0] > last_date]
    if new_rows:
        print(f"  {label} extra: appending {len(new_rows)} new day(s)")
        existing["dates"] += [r[0] for r in new_rows]
        existing["newHighs"] += [r[2] for r in new_rows]
        existing["newLows"] += [r[3] for r in new_rows]
        existing["netHL"] += [r[4] for r in new_rows]
    else:
        print(f"  {label} extra: already up to date")
    return existing


def main():
    live = load_json(LIVE_PATH, {"sectors": [], "macro": {}, "flows": []})
    live["sectors"] = fetch_sectors(live.get("sectors", []))
    live["macro"] = fetch_macro(live.get("macro", {}))
    live["flows"] = fetch_flows(live.get("flows", []))
    live["asof"] = datetime.datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
    save_json(LIVE_PATH, live)

    extra_all = load_json(EXTRA_PATH, {})

    for cfg in INDEX_CONFIGS:
        key, label = cfg["key"], cfg["key"]
        print(f"--- {key} ---")

        price = load_json(cfg["price_path"], [])
        price = fetch_index_price(cfg["ticker"], price, key)
        save_json(cfg["price_path"], price)

        existing_pct = load_json(cfg["pct_path"], [])
        existing_extra = extra_all.get(key)
        reseeding = (len(existing_pct) < SEED_THRESHOLD and not cfg["pct_has_backfill"]) or \
                    (not existing_extra) or (len(existing_extra.get("dates", [])) < SEED_THRESHOLD)

        symbols = load_constituents(cfg["constituents_url"], key)
        if not symbols:
            continue
        tickers = [s + ".NS" for s in symbols]
        if reseeding:
            rows = compute_breadth_frame(tickers, key, start=seed_start_date())
        else:
            rows = compute_breadth_frame(tickers, key, period=INCREMENTAL_PERIOD)
        if rows is None:
            continue

        new_pct = merge_pct(existing_pct, rows, cfg["pct_has_backfill"], cfg["pct_default_last_date"], key)
        save_json(cfg["pct_path"], new_pct)

        extra_all[key] = merge_extra(existing_extra, rows, key)
        save_json(EXTRA_PATH, extra_all)  # save after each index so a later failure doesn't lose earlier ones

        time.sleep(1.0)

    print("Done.")


if __name__ == "__main__":
    main()
