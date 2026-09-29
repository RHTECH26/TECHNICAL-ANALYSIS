# Right Horizons Markets Screener

Static site (`index.html`) + one scheduled GitHub Action (split into 3 parallel jobs) that
keeps every data file fresh — ETF prices, sector indices, the macro dashboard, FII/DII
flows, 4 index prices (Nifty 50, Nifty 500, Nifty Midcap 150, Nifty Smallcap 250), and each
index's full market-breadth picture (% above 200-day SMA, new 52-week highs, new 52-week
lows, net breadth) — all via `yfinance` (no API key needed). The page never shows a data URL
to whoever's viewing it; it connects to your published site's own `/data/` files quietly in
the background.

## One-time setup

1. Create a new **public** GitHub repo (Pages needs public on the free plan, unless you
   have GitHub Pro/Team/Enterprise for private-repo Pages).
2. Push these files to it (root of the repo — don't nest them in a subfolder):
   - `index.html`
   - `data/etf-history.json`
   - `data/live.json`
   - `data/nifty50-price.json`
   - `data/breadth-history-nifty50.json`
   - `data/nifty500-price.json`
   - `data/breadth-history.json`
   - `data/niftymidcap150-price.json`
   - `data/breadth-history-midcap150.json`
   - `data/niftysc250-price.json`
   - `data/breadth-history-sc250.json`
   - `data/breadth-extra.json`
   - `data/global-etf-history.json`
   - `scripts/fetch_etf_data.py`
   - `scripts/fetch_live_data.py`
   - `scripts/fetch_global_etf_data.py`
   - `.github/workflows/update-data.yml`
3. Repo **Settings → Pages** → Source: "Deploy from a branch" → Branch: `main`, folder `/ (root)` → Save.
   GitHub gives you a URL like `https://<username>.github.io/<repo>/` within a minute or two.
4. Repo **Settings → Actions → General** → under "Workflow permissions", select
   **"Read and write permissions"** and save (the Action needs this to commit data back).
5. Repo **Actions** tab → find "Update ETF data" → **Run workflow** (the ▶ button) to fetch
   data for the first time instead of waiting for the schedule. This runs three fetches
   *at the same time* (Indian ETFs + Nifty 500/Smallcap 250 breadth · sectors/macro/flows ·
   the 966-ticker global ETF universe), so the whole thing finishes in roughly the time of
   the slowest one alone — typically **10–15 minutes** — rather than the three added up.
6. Once it finishes, open your site (`https://<username>.github.io/<repo>/`) — every panel
   (Sector Heatmap, Flows & Macro, Indian ETF Heatmap, Global ETF Heatmap, Market Breadth)
   should show a green "live" dot and start pulling from your repo's own `/data/` files
   automatically. Nothing to paste or connect — it's all wired to your published URL already.

After that, it re-fetches automatically on the schedule in `update-data.yml`
(weekdays shortly after NSE close) — no key, no manual work.

## Market Breadth tab

Choose any of 4 indices (Nifty 50, Nifty 500, Nifty Midcap 150, Nifty Smallcap 250) with the
top chip row, and any of 4 lower-chart indicators with the second chip row:

- **Stocks > 200 SMA (%)** — % of that index's constituents trading above their own 200-day
  SMA.
- **Net Highs (H-L)** — new 52-week highs minus new 52-week lows among constituents, per day.
- **52-Week Highs** — count of constituents making a new 52-week high that day.
- **52-Week Lows** — count of constituents making a new 52-week low that day.

The 5 tiles above the charts (index price, New 52W Highs, New 52W Lows, Net Breadth (H-L),
Stocks > 200 SMA) always show together regardless of which indicator is selected, matching
a scanner-style breadth screen. Both charts support scroll-to-zoom, click-drag-to-pan, and
double-click-to-reset (or use the "Reset zoom" button) — powered by the `chartjs-plugin-zoom`
plugin, inlined alongside Chart.js so nothing is fetched from an external CDN.

## What updates live vs. what stays static

- **Live**: Indian ETF prices, the global ETF universe (966 tickers, ~14 months of daily
  closes), sector index % changes, India VIX, USD/INR, Brent crude, all 4 index prices, and
  each index's full breadth picture (% above 200-day SMA, new 52-week highs/lows, net
  breadth). Only the Nifty 500's % > 200 SMA series has a historical backfill (20 years) —
  the other 3 indices' pct series, and all 4 indices' new-highs/new-lows/net-HL series, start
  populating from the first successful workflow run onward (the first run fetches ~3 years of
  history to seed them, so the charts aren't bare after just one run).
- **Best-effort live**: FII/DII cash flows — pulled from NSE's own public API, which
  occasionally blocks GitHub's servers. When that happens the page just keeps last week's
  figures until the next successful run.
- **Stays at its last compiled value**: Nifty 50 forward P/E and the India 10Y G-Sec yield —
  there's no free live feed for either, and one NSE sub-index ("Midsmall Healthcare") isn't
  tracked on Yahoo Finance at all.

## Why the workflow runs 3 jobs in parallel

`fetch_etf_data.py`, `fetch_live_data.py` and `fetch_global_etf_data.py` don't depend on
each other, so `update-data.yml` runs each as its own GitHub Actions job at the same time
instead of one after another, then a final `commit` job downloads all three results and
pushes them together. This is what actually makes the workflow faster — the global ETF
fetch (966 tickers) is by far the slowest step, so parallelizing means the other two no
longer add their time on top of it.

The global ETF fetch itself is deliberately throttled (`threads=False`, small batches,
short pauses between them) because Yahoo Finance rate-limits a fast/threaded pull across
hundreds of tickers — once that happens, every batch after the first one silently comes
back empty. If you want it to run faster and are willing to risk occasional missing
tickers on a run, you can raise `BATCH_SIZE` and lower `SLEEP_BETWEEN_BATCHES` /
`SLEEP_ON_RETRY` near the top of `scripts/fetch_global_etf_data.py` — but if tickers start
turning up mostly empty in `data/global-etf-history.json` again, that's this rate limit
again and the fix is to dial those settings back down.

## Why "Star Group Plc" used to always top the global ETF list

The original sheet (v2) had one row — `STAR.L`, labeled "Star Energy Group Plc" tracking
"Space Technologies" — that wasn't an ETF at all, just a small, volatile UK stock that had
been mixed into the list by mistake. Real stocks like that can swing far more than any
diversified ETF, so it kept showing up as the "top performer" regardless of the date range.
The current sheet (v4) removes that row along with a handful of other non-ETF entries, so
this shouldn't happen anymore; if a similar outlier ever reappears, it's almost always a
data-quality issue in the source sheet rather than a bug in the page.

## Updating the site itself

If you ask Claude for further changes to the dashboard, re-export `index.html` and
push it over the existing one — the data pipeline is unaffected.
