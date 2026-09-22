# NSE Trend Reversal Scanner

Static GitHub Pages site that flags NSE stocks showing a **trend-reversal break of
structure**: after a run of consecutive lower lows, the stock closes back above
the most recent lower high (the swing high formed during the pullback between the
last two lows). A daily GitHub Action fetches live NSE data via
[`yfinance`](https://github.com/ranaroussi/yfinance), runs the scan, and commits
the results; `index.html` reads that JSON client-side — same pattern as the
[reference screener](https://rhdano099.github.io/indoetf/).

## What counts as a signal

For each stock, using ~2 years of daily OHLCV:
1. Detect swing highs/lows (a 3-bar-either-side fractal).
2. Take the most recent 2 (or 3) confirmed swing lows. If they're strictly
   decreasing → a genuine lower-low downtrend leg.
3. Take the swing high between the last two of those lows — that's the most
   recent **lower high**.
4. If price has since closed above that lower-high level, it's flagged, with
   the date of the first close that broke it.

Both a 2-lower-low and a 3-lower-low (stronger) version are computed and stored
per stock; the page lets you switch between them, plus filter by how recently
the breakout happened and by volume confirmation on the breakout day.

This is a structural/price-action signal only — it does not check volume,
RSI or trend by itself; those are shown as extra context columns so you can
judge each one yourself.

## Files

| File | What it does |
|---|---|
| `index.html` | The scanner page — filters, sortable table, reads `data/scan_results.json` |
| `scripts/scan.py` | Fetches OHLCV via yfinance, runs the swing/breakout logic, writes `data/scan_results.json` |
| `data/nse_universe.csv` | Your uploaded universe — 3,168 NSE symbols + names. Edit this to add/remove symbols. |
| `data/scan_results.json` | Output of the last scan. Starts empty (`results: []`) until the Action runs once. |
| `.github/workflows/daily_scan.yml` | Runs the scan Mon–Fri at 16:00 IST and commits the result |
| `requirements.txt` | `yfinance`, `pandas`, `numpy` |

## Setup

1. Create a new **public** GitHub repo and upload everything in this project
   (public repos get unlimited free GitHub Actions minutes; a private repo
   works too but eats into your monthly minutes quota).
2. **Settings → Actions → General → Workflow permissions** → set to
   **"Read and write permissions"**. Without this the daily commit step fails
   with a 403.
3. **Settings → Pages** → Deploy from branch → `main` / root.
4. **Actions tab** → run `Daily NSE reversal scan` once manually
   (`workflow_dispatch`) rather than waiting for the 16:00 IST cron — this
   populates `data/scan_results.json` for the first time.
5. Your page is live at `https://<username>.github.io/<repo-name>/`.

## On the ~3,168-symbol universe

Downloading and scanning the full list takes real time — the script batches
yfinance calls (100 tickers per batch, small delay between batches) to stay
polite to Yahoo's free endpoint, and the workflow gives itself a 2-hour
timeout. A run over the full universe will likely take somewhere in the
range of 20–60 minutes depending on Yahoo's response times that day; symbols
that fail to fetch or don't have enough history are skipped and listed in
`skipped_symbols` in the output — never backfilled with fake data. If a full
daily run turns out too slow or unreliable in practice, the easiest fix is
trimming `data/nse_universe.csv` down to a smaller universe (e.g. Nifty 500)
rather than changing the scan logic.

## Local test

```bash
pip install -r requirements.txt
python scripts/scan.py
python -m http.server 8000   # then open http://localhost:8000
```

## Tuning

- `SWING_LEFT` / `SWING_RIGHT` in `scripts/scan.py` control how sensitive the
  swing-point detection is (smaller = more, noisier swing points).
- `MIN_BARS_REQUIRED` sets the minimum history needed before a symbol is
  scanned at all.
- Both currently default to a 3-bar fractal and 60-bar minimum.
