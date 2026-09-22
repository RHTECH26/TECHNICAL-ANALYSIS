"""
NSE Trend-Reversal Scanner
---------------------------
Strategy: after a downtrend of consecutive LOWER LOWS, flag a stock the first
time price closes back above the most recent LOWER HIGH (the swing high formed
during the pullback between the last two lower lows). That break of structure
is the reversal signal.

Run daily by .github/workflows/daily_scan.yml. Writes data/scan_results.json,
which index.html reads client-side. No synthetic data: if a symbol can't be
fetched or doesn't have enough history, it's skipped and logged, never faked.
"""

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/nse_universe.csv"
OUTPUT_PATH = "data/scan_results.json"
BATCH_SIZE = 100
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
    """Ordered list of (bar_index, 'H'|'L', price), forced to alternate
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


def find_reversal_signal(df, swings, min_lower_lows=2):
    """
    Looks at the most recent `min_lower_lows` confirmed swing lows. If they
    are strictly decreasing (a clean lower-low downtrend leg), take the swing
    high between the last two of them (the most recent 'lower high') as the
    resistance level, then check whether price has since closed above it.
    Returns None if no valid signal, else a dict describing it.
    """
    lows = [p for p in swings if p[1] == "L"]
    highs = [p for p in swings if p[1] == "H"]
    if len(lows) < min_lower_lows:
        return None

    recent_lows = lows[-min_lower_lows:]
    prices = [p[2] for p in recent_lows]
    if not all(prices[i] > prices[i + 1] for i in range(len(prices) - 1)):
        return None  # not a clean sequence of lower lows

    prev_low, last_low = recent_lows[-2], recent_lows[-1]
    between_highs = [h for h in highs if prev_low[0] < h[0] < last_low[0]]
    if not between_highs:
        return None

    lower_high = max(between_highs, key=lambda h: h[2])
    breakout_level = lower_high[2]

    after = df.iloc[last_low[0] + 1 :]
    closes_above = after[after["Close"] > breakout_level]
    if closes_above.empty:
        return None

    breakout_pos = df.index.get_loc(closes_above.index[0])
    breakout_date = closes_above.index[0]
    days_since = (len(df) - 1) - breakout_pos

    return {
        "is_signal": True,
        "breakout_level": round(breakout_level, 2),
        "breakout_date": breakout_date.strftime("%Y-%m-%d"),
        "days_since_breakout": int(days_since),
        "last_low_price": round(last_low[2], 2),
        "last_low_date": df.index[last_low[0]].strftime("%Y-%m-%d"),
        "prev_low_price": round(prev_low[2], 2),
        "prev_low_date": df.index[prev_low[0]].strftime("%Y-%m-%d"),
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
    sig_2 = find_reversal_signal(df, swings, min_lower_lows=2)
    sig_3 = find_reversal_signal(df, swings, min_lower_lows=3)
    ctx = compute_context(df)

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg_pct = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    return {
        "symbol": sym,
        "company": name,
        "ltp": round(last_close, 2),
        "chg_pct": chg_pct,
        "signal_2ll": sig_2,
        "signal_3ll": sig_3,
        **ctx,
    }


def main():
    universe = load_universe()
    names = dict(zip(universe["Symbol"], universe["Description"]))
    symbols = universe["Symbol"].tolist()
    ticker_map = {s: f"{s}.NS" for s in symbols}

    results = []
    skipped = []

    for batch in chunked(symbols, BATCH_SIZE):
        batch_tickers = [ticker_map[s] for s in batch]
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
            print(f"batch download failed: {exc}", file=sys.stderr)
            skipped.extend(batch)
            continue

        for sym in batch:
            tk = ticker_map[sym]
            try:
                if len(batch_tickers) == 1:
                    df = data
                elif tk in getattr(data.columns, "levels", [[]])[0]:
                    df = data[tk]
                else:
                    df = None

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

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "universe_count": len(symbols),
        "scanned_count": len(results),
        "skipped_count": len(skipped),
        "skipped_symbols": skipped,
        "results": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"Done. scanned={len(results)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
