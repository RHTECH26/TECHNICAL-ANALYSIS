"""
NSE Trend-Reversal Scanner
---------------------------
Strategy: standard swing-structure labeling (HH / HL / LH / LL) — each swing
high is compared only to the *previous* swing high (HH if higher, LH if
lower), each swing low only to the previous swing low (HL if higher, LL if
lower). No requirement for consecutive lower lows.

Signal: the stock's most recently confirmed swing high is a LH (lower high),
and price has since closed back above that LH level — a break of structure
back to the upside, i.e. crossing the LH.

Run daily by .github/workflows/daily_scan.yml. Writes data/scan_results.json,
which index.html reads client-side. No synthetic data: if a symbol can't be
fetched or doesn't have enough history, it's skipped and logged, never faked.
Any unexpected error is caught, logged with a full traceback to stderr, and
whatever results were gathered so far are still written out rather than the
whole run dying with nothing to show for it.
"""

import json
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/nse_universe.csv"
OUTPUT_PATH = "data/scan_results.json"
BATCH_SIZE = 75
HISTORY_PERIOD = "2y"
SWING_LEFT = 3          # bars either side of a candidate swing point
SWING_RIGHT = 3
MIN_BARS_REQUIRED = 60  # skip symbols with less history than this


# ---------------------------------------------------------------------------
# Swing / structure detection
# ---------------------------------------------------------------------------

def detect_swings(df, left=SWING_LEFT, right=SWING_RIGHT):
    """Fractal swing high/low detection. Returns two boolean arrays."""
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    n = len(df)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        h_win = highs[i - left : i + right + 1]
        l_win = lows[i - left : i + right + 1]
        h_max = h_win.max()
        l_min = l_win.min()
        if highs[i] == h_max and np.sum(h_win == h_max) == 1:
            is_swing_high[i] = True
        if lows[i] == l_min and np.sum(l_win == l_min) == 1:
            is_swing_low[i] = True

    return is_swing_high, is_swing_low


def build_swing_sequence(df, is_high, is_low):
    """Ordered list of [bar_index, 'H'|'L', price], forced to alternate
    (keeps the more extreme point when two of the same type occur in a row
    before the opposite type shows up)."""
    points = []
    for i in range(len(df)):
        if is_high[i]:
            points.append([i, "H", float(df["High"].iat[i])])
        if is_low[i]:
            points.append([i, "L", float(df["Low"].iat[i])])
    points.sort(key=lambda p: p[0])

    cleaned = []
    for p in points:
        if cleaned and cleaned[-1][1] == p[1]:
            if p[1] == "H" and p[2] > cleaned[-1][2]:
                cleaned[-1] = p
            elif p[1] == "L" and p[2] < cleaned[-1][2]:
                cleaned[-1] = p
            # else: discard, existing point is more extreme
        else:
            cleaned.append(p)
    return cleaned


def label_swings(swings):
    """HH/HL/LH/LL labeling: each swing compared only to the *previous*
    swing of the same type — not a run of consecutive lower lows."""
    labeled = []
    last_high = None
    last_low = None
    for idx, kind, price in swings:
        label = None
        if kind == "H":
            if last_high is not None:
                label = "HH" if price > last_high else "LH"
            last_high = price
        else:
            if last_low is not None:
                label = "HL" if price > last_low else "LL"
            last_low = price
        labeled.append({"index": idx, "kind": kind, "price": price, "label": label})
    return labeled


def find_lh_cross_signal(df, labeled_swings):
    """
    If the most recently confirmed swing high is a LH (lower high), and
    price has since closed above that LH's level, return the signal.
    Returns None if the last swing high isn't a LH, or price hasn't
    crossed it yet.
    """
    highs = [s for s in labeled_swings if s["kind"] == "H" and s["label"] is not None]
    if not highs:
        return None

    last_high = highs[-1]
    if last_high["label"] != "LH":
        return None

    breakout_level = last_high["price"]
    after = df.iloc[last_high["index"] + 1 :]
    closes_above = after[after["Close"] > breakout_level]
    if closes_above.empty:
        return None

    breakout_pos = df.index.get_loc(closes_above.index[0])
    breakout_date = closes_above.index[0]
    days_since = (len(df) - 1) - breakout_pos

    # context only, not a condition: the swing low right before this LH
    lows_before = [
        s for s in labeled_swings if s["kind"] == "L" and s["index"] < last_high["index"]
    ]
    prior_low = lows_before[-1] if lows_before else None

    return {
        "is_signal": True,
        "lh_level": round(breakout_level, 2),
        "lh_date": df.index[last_high["index"]].strftime("%Y-%m-%d"),
        "breakout_date": breakout_date.strftime("%Y-%m-%d"),
        "days_since_breakout": int(days_since),
        "prior_low_level": round(prior_low["price"], 2) if prior_low else None,
        "prior_low_label": prior_low["label"] if prior_low else None,
        "prior_low_date": (
            df.index[prior_low["index"]].strftime("%Y-%m-%d") if prior_low else None
        ),
    }


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

    is_h, is_l = detect_swings(df)
    swings = build_swing_sequence(df, is_h, is_l)
    labeled = label_swings(swings)
    signal = find_lh_cross_signal(df, labeled)
    ctx = compute_context(df)

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg_pct = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    return {
        "symbol": sym,
        "company": name,
        "ltp": round(last_close, 2),
        "chg_pct": chg_pct,
        "signal_lh_cross": signal,
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
