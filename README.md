# NSE Range Breakout Scanner

Static GitHub Pages site that flags NSE stocks breaking out of a chosen
N-day range — today's close above the highest High of the preceding N
trading days. A daily GitHub Action fetches live NSE data via
[`yfinance`](https://github.com/ranaroussi/yfinance), runs the scan, and
commits the results; `index.html` reads that JSON client-side and lets you
switch the range (20 / 50 / 100 / 200 days) instantly — same pattern as the
[reference screener](https://rhdano099.github.io/indoetf/).

## What counts as a signal

For each stock, using ~2 years of daily OHLCV, and for each N in
**20, 50, 100, 200**:

- Take the highest **High** over the preceding N trading days (today
  excluded).
- If today's **Close** is above that level, it's flagged as breaking out of
  the N-day range, with the level itself and % above it.

All four ranges are computed and stored per stock every run; the page's
range buttons just switch which one you're looking at — no recomputation
needed client-side. If a stock doesn't have enough history for a given N
(e.g. a recent listing, for the 200-day range), that range is left `null`
for it rather than guessed at.

Volume vs 20-day average, RSI(14) and 200-day SMA trend are included as
context columns and an optional volume filter — they aren't part of the
breakout flag itself.

## Files

| File | What it does |
|---|---|
| `index.html` | The scanner page — range buttons, volume/search filters, sortable table |
| `scripts/scan.py` | Fetches OHLCV via yfinance, computes the four range-breakout flags, writes `data/scan_results.json` |
| `data/nse_universe.csv` | Your uploaded universe — 3,168 NSE symbols + names. Edit this to add/remove symbols. |
| `data/scan_results.json` | Output of the last scan. Starts empty (`results: []`) until the Action runs once. |
| `.github/workflows/daily_scan.yml` | Runs Mon–Fri at 16:00 IST and commits the result |
| `requirements.txt` | `yfinance`, `pandas` |

## Setup

1. Create a new **public** GitHub repo and upload everything in this
   project (public repos get unlimited free GitHub Actions minutes).
2. **Settings → Actions → General → Workflow permissions** → set to
   **"Read and write permissions"**. Without this the daily commit step
   fails with a 403.
3. **Settings → Pages** → Deploy from branch → `main` / root.
4. **Actions tab** → run `Daily NSE range breakout scan` once manually
   (`workflow_dispatch`) rather than waiting for the 16:00 IST cron — this
   populates `data/scan_results.json` for the first time.
5. Your page is live at `https://<username>.github.io/<repo-name>/`.

## If a run fails

`scan.py` now catches any unexpected error, prints the full traceback to
the Action's log, and still writes out whatever results it had gathered —
plus a `last_run_error` field in `data/scan_results.json` describing what
broke. Open the failed run → the **scan** job → the **"Run scan"** step
(not "Install dependencies") and scroll to the bottom of that step's log
for the actual traceback.

## On the ~3,168-symbol universe

Downloading and scanning the full list takes real time — the script
batches yfinance calls (75 tickers per batch, small delay between
batches) to stay polite to Yahoo's free endpoint, and the workflow gives
itself a 2-hour timeout. Symbols that fail to fetch or don't have enough
history are skipped and listed in `skipped_symbols` in the output — never
backfilled with fake data. If a full daily run proves too slow or flaky,
the easiest fix is trimming `data/nse_universe.csv` down to a smaller
universe (e.g. Nifty 500) rather than changing the scan logic.

## Local test

```bash
pip install -r requirements.txt
python scripts/scan.py
python -m http.server 8000   # then open http://localhost:8000
```

## Tuning

- `RANGES` in `scripts/scan.py` controls which lookbacks are computed —
  defaults to `(20, 50, 100, 200)`. Adding a range here also needs a
  matching button added to `#rangeBar` in `index.html`.
- `MIN_BARS_REQUIRED` sets the minimum history needed before a symbol is
  scanned at all (defaults to 25 bars).
