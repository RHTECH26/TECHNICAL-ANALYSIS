#!/usr/bin/env python3
"""
Pulls daily close prices for the 98 NSE ETFs used in the Bhavcopy heatmap and writes
data/etf-history.json in the shape the heatmap's "Connect live data" loader expects:

    { "SYMBOL": [["YYYY-MM-DD", price], ...], ... }

Run locally with:  python3 scripts/fetch_etf_data.py
Runs automatically via .github/workflows/update-data.yml on a daily schedule.

No API key needed — yfinance reads Yahoo Finance's public endpoints.

Fetched with yf.download() in small batches, not one yf.Ticker(...).history() call per
symbol — every single-ticker Ticker.history() fetch in this project has ended up hitting
Yahoo's rate limit hard on GitHub's runner IPs (this same site's Nifty 500 index price,
Nifty Smallcap 250 index price, and the original version of this exact script all hit it),
while yf.download()'s batched endpoint has consistently held up. Small batches + no
threading + pauses between them is what keeps this reliable — see fetch_global_etf_data.py
for the same pattern.

Merges into the PREVIOUSLY COMMITTED data/etf-history.json instead of overwriting it from
scratch. Yahoo's batched download endpoint occasionally comes back with a NaN close for
one ticker on one specific day even though the rest of the batch (and that same ticker on
every other day) is fine -- a transient per-symbol glitch, not a real market closure.
Overwriting from scratch every run means that single bad day is gone for good the moment
it happens, with no way to recover it: the site's "1D" change then has to either fudge the
comparison across a multi-day gap, or show "no data" for that ticker until the SAME day
happens to fetch cleanly on some future run (never, if the gap is a stale, permanently-bad
date rather than "yesterday"). Merging means a transient gap in today's fetch doesn't
erase a date a previous run already captured, and a date that failed before but fetches
cleanly today gets backfilled in automatically -- self-healing across runs instead of
each run being an independent all-or-nothing snapshot.
"""
import json
import time
import yfinance as yf

# The 98 symbols from the NSE ETF master sheet (MW-ETF-17-Sep-2026.csv).
# yfinance needs the ".NS" suffix for NSE-listed tickers.
SYMBOLS = [
    "EQUAL200","HDFCBSE500","DIVIDEND","ICICIB22","ECAPINSURE","MOVALUE","MOHEALTH",
    "GROWWHOSPI","DEFENCE","MOINFRA","INSUREIETF","ELMDIV","MOMMIDCAP","MIDSELIETF",
    "MIDBANKADD","GROWWPOWER","SBIBPB","MOQUALITY","SELECTIPO","HDFCSENSEX","SNXT30BEES",
    "SNXT50BETA","BANK10BETF","MSCI360","TOP100CASE","NIFTY100EW","ESG","LOWVOLIETF",
    "HDFCQUAL","GROWWN200","ALPHAETF","MOM30IETF","NIFTYQLITY","VAL30IETF","NIFTYBEES",
    "SBINEQWETF","SHARIABEES","NV20IETF","MONIFTY500","FLEXIADD","HEALTHCARE","GROWWLOVOL",
    "MOMENTUM50","MULTICAP","EMULTIMQ","VALUEAXIS","ALPHA","ALPL30IETF","AUTOIETF",
    "BANKIETF","MOCAPITAL","CEMNTGROWW","CHEMICAL","COMMOIETF","CPSEETF","DIVOPPBEES",
    "ENERGY","GROWWEV","BFSI","FINIETF","FMCGIETF","FMCGADD","HDFCGROWTH","HEALTHY",
    "CONSUMBEES","MODEFENCE","INFRA","GROWWNET","MAKEINDIA","CONSUMER","GROWWRAIL",
    "MOTOUR","INFRAIETF","ITBEES","ELM250","METALIETF","MIDCAPETF","MOMIDMTM","MIDQ50ADD",
    "MIDCAP","MIDSMALL","MNC","NEXT50IETF","OILIETF","PHARMABEES","PVTBANIETF","ABSLPSE",
    "PSUBNKBEES","MOREALTY","MOSERVICE","HDFCSML250","SMALLCAP","SML100CASE","TOP10ADD",
    "TOP15IETF","TOP20","AONETOTAL","AONETMMQ50",
]

OUT_PATH = "data/etf-history.json"
LOOKBACK = "2y"     # how much history to keep in the JSON (trim to taste)
BATCH_SIZE = 20
RETRIES = 4
SLEEP_BETWEEN_BATCHES = 2.5
SLEEP_ON_RETRY = 8.0
MAX_ROWS_PER_TICKER = 500   # ~2 years of NSE trading days -- caps how far merging with
                            # the previous file can grow each ticker's series, since a
                            # pure merge (see main()) would otherwise keep every date
                            # forever rather than respecting LOOKBACK.


def load_existing():
    """The previously committed data/etf-history.json, if any -- merged into this run's
    fresh fetch rather than replaced by it. Missing/corrupt file just means starting from
    empty, same as the very first run ever."""
    try:
        with open(OUT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def merge_series(old_rows, new_rows):
    """old_rows/new_rows: [["YYYY-MM-DD", price], ...]. Reads old_rows by index (row[0],
    row[1]) rather than unpacking exactly 2 values -- a briefly-deployed earlier version of
    this script committed some 3-element [date, close, open] rows, and old_rows is whatever
    is already sitting in the committed JSON, so this has to tolerate that leftover shape
    (and any other stray extra fields) without crashing, discarding anything past the close.
    New rows win on a shared date (this run's data is the freshest), but a date present in
    old_rows and absent from new_rows -- this run's fetch had a transient gap on that day --
    is kept instead of silently dropped. Trimmed to the most recent MAX_ROWS_PER_TICKER
    dates afterward."""
    merged = {row[0]: row[1] for row in old_rows}
    for row in new_rows:
        merged[row[0]] = row[1]
    dates = sorted(merged.keys())[-MAX_ROWS_PER_TICKER:]
    return [[d, merged[d]] for d in dates]


def fetch_batch(tickers):
    out = {}
    raw = None
    for attempt in range(RETRIES):
        try:
            raw = yf.download([t + ".NS" for t in tickers], period=LOOKBACK, interval="1d",
                               auto_adjust=True, group_by="ticker", threads=False, progress=False)
            if raw is not None and len(raw) > 0:
                break
        except Exception as e:
            print(f"  batch download attempt {attempt+1} failed: {e}")
            raw = None
        time.sleep(SLEEP_ON_RETRY * (attempt + 1))
    if raw is None or len(raw) == 0:
        return out
    single = len(tickers) == 1
    for t in tickers:
        try:
            s = raw["Close"] if single else raw[t + ".NS"]["Close"]
            s = s.dropna()
            if len(s) > 0:
                out[t] = [[d.strftime("%Y-%m-%d"), round(float(c), 4)] for d, c in s.items()]
        except Exception:
            continue
    return out


def main():
    existing = load_existing()
    result = dict(existing)  # start from what's already committed, not from scratch --
                              # see the merge_series note above for why.
    missing = []
    gap_new = []
    n_batches = (len(SYMBOLS) - 1) // BATCH_SIZE + 1
    for i in range(0, len(SYMBOLS), BATCH_SIZE):
        batch = SYMBOLS[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        got = fetch_batch(batch)
        for t, rows in got.items():
            old_dates = {row[0] for row in existing.get(t, [])}
            merged = merge_series(existing.get(t, []), rows)
            new_dates = {row[0] for row in merged} - old_dates
            if new_dates:
                gap_new.append((t, sorted(new_dates)))
            result[t] = merged
        missing.extend([s for s in batch if s not in got])
        print(f"Batch {batch_num}/{n_batches}: got {len(got)}/{len(batch)} — running total {len(result)}")
        # Save progress after every batch so a run that gets cut off partway (timeout,
        # rate-limit) still keeps whatever it fetched instead of losing the whole thing.
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, separators=(",", ":"))
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\nWrote {OUT_PATH}: {len(result)}/{len(SYMBOLS)} symbols.")
    if missing:
        print(f"Missing entirely this run ({len(missing)}): {missing}")
    if gap_new:
        print(f"Newly added/backfilled dates this run (merged in, not overwritten):")
        for t, dates in gap_new:
            print(f"  {t}: {dates}")


if __name__ == "__main__":
    main()
