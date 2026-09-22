"""
NSE Range Breakout Scanner
---------------------------
Strategy: flag a stock when today's close is a new N-day high — i.e. above
the highest High of the preceding N trading days. Computed for N = 20, 50,
100 and 200 in one pass; the page lets you switch which range you're
scanning by picking from a dropdown, same as the reference screener's
lookback selector.

Run daily by .github/workflows/daily_scan.yml. Writes data/scan_results.json,
which index.html reads client-side. No synthetic data: if a symbol can't be
fetched or doesn't have enough history for a given N, that N's flag is left
null for that symbol rather than guessed at.

Any unexpected error is caught, logged with a full traceback to stderr, and
whatever results were gathered so far are still written out rather than the
whole run dying with nothing to show for it.
"""

import json
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/nse_universe.csv"
OUTPUT_PATH = "data/scan_results.json"
BATCH_SIZE = 75
HISTORY_PERIOD = "2y"
RANGES = (20, 50, 100, 200)
MIN_BARS_REQUIRED = 25  # need at least this much history to be worth scanning at all


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

def compute_breakout_flags(df, ranges=RANGES):
    """
    For each N in `ranges`: is today's close above the highest High of the
    preceding N trading days (today excluded)? Returns None for a given N
    if there isn't enough history to evaluate it honestly.
    """
    close = df["Close"]
    high = df["High"]
    last_close = float(close.iloc[-1])

    out = {}
    for n in ranges:
        key = f"high_{n}"
        if len(df) < n + 1:
            out[key] = None
            continue
        prior_high = float(high.iloc[-(n + 1) : -1].max())
        is_breakout = last_close > prior_high
        pct_above = round((last_close - prior_high) / prior_high * 100, 2) if prior_high else None
        out[key] = {
            "is_breakout": bool(is_breakout),
            "level": round(prior_high, 2),
            "pct_above": pct_above,
        }
    return out


def compute_context(df):
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    vol_20d_avg = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_20d_avg
    sma200 = close.rolling(200).mean()

    def safe(v):
        return None if pd.isna(v) else round(float(v), 2)

    return {
        "rsi14": safe(rsi.iloc[-1]),
        "vol_ratio_20d": safe(vol_ratio.iloc[-1]),
        "above_200sma": (
            None if pd.isna(sma200.iloc[-1]) else bool(close.iloc[-1] > sma200.iloc[-1])
        ),
    }


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_universe(path=UNIVERSE_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].str.strip()
    return df


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def process_one(sym, name, df):
    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(df) < MIN_BARS_REQUIRED:
        return None

    breakouts = compute_breakout_flags(df)
    ctx = compute_context(df)

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg_pct = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    return {
        "symbol": sym,
        "company": name,
        "ltp": round(last_close, 2),
        "chg_pct": chg_pct,
        **breakouts,
        **ctx,
    }


def extract_symbol_df(data, tk, single_ticker):
    """Pull one ticker's OHLCV out of a (possibly multi-ticker) yf.download result."""
    if single_ticker:
        return data
    cols = data.columns
    if hasattr(cols, "levels"):
        top_level = cols.get_level_values(0)
        if tk not in top_level:
            return None
        return data[tk]
    return None


def run_scan():
    universe = load_universe()
    names = dict(zip(universe["Symbol"], universe["Description"]))
    symbols = universe["Symbol"].tolist()
    ticker_map = {s: f"{s}.NS" for s in symbols}

    results = []
    skipped = []

    for batch in chunked(symbols, BATCH_SIZE):
        batch_tickers = [ticker_map[s] for s in batch]
        single_ticker = len(batch_tickers) == 1
        try:
            data = yf.download(
                tickers=" ".join(batch_tickers),
                period=HISTORY_PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"batch download failed ({batch[0]}..{batch[-1]}): {exc}", file=sys.stderr)
            skipped.extend(batch)
            continue

        if data is None or (hasattr(data, "empty") and data.empty):
            skipped.extend(batch)
            continue

        for sym in batch:
            tk = ticker_map[sym]
            try:
                df = extract_symbol_df(data, tk, single_ticker)
                if df is None or df.empty:
                    skipped.append(sym)
                    continue

                row = process_one(sym, names.get(sym, sym), df)
                if row is None:
                    skipped.append(sym)
                else:
                    results.append(row)
            except Exception as exc:  # noqa: BLE001
                skipped.append(sym)
                print(f"{sym} failed: {exc}", file=sys.stderr)

        time.sleep(2)  # be gentle with the free Yahoo endpoint

    return results, skipped, len(symbols)


def write_output(results, skipped, universe_count, error=None):
    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "universe_count": universe_count,
        "scanned_count": len(results),
        "skipped_count": len(skipped),
        "skipped_symbols": skipped,
        "ranges": list(RANGES),
        "results": results,
    }
    if error:
        output["last_run_error"] = error

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))


def main():
    results, skipped, universe_count = [], [], 0
    try:
        results, skipped, universe_count = run_scan()
        write_output(results, skipped, universe_count)
        print(f"Done. scanned={len(results)} skipped={len(skipped)}")
    except Exception as exc:  # noqa: BLE001
        # Never die with nothing written and nothing logged: capture the full
        # traceback, save whatever partial results exist, then exit non-zero
        # so the Action still shows failed - but the log and the JSON both
        # say why.
        tb = traceback.format_exc()
        print("FATAL: scan crashed:\n" + tb, file=sys.stderr)
        write_output(results, skipped, universe_count, error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
