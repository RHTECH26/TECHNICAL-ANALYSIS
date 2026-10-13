"""
NSE Support Scanner
-------------------
Finds stocks trading AT a previous swing-low support, where the swing is >= SWING_PCT (default 5%).

Swing logic (zig-zag on daily High/Low):
  * A swing LOW is confirmed only after price rallies >= SWING_PCT from it.
  * A swing HIGH is confirmed only after price falls >= SWING_PCT from it.
  So every leg between pivots is >= 5%.

"At support" = latest close is within PROXIMITY_PCT above a confirmed swing low,
and no daily close since that low has broken below it (by more than BREAK_TOL_PCT).
"""
import json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

SWING_PCT      = float(os.getenv("SWING_PCT", 5.0))      # min swing size (%)
PROXIMITY_PCT  = float(os.getenv("PROXIMITY_PCT", 3.0))  # only stocks within this % above support are kept
BREAK_TOL_PCT  = float(os.getenv("BREAK_TOL_PCT", 1.0))  # close below support by > this = support broken
LOOKBACK       = os.getenv("LOOKBACK", "2y")
TREND_SWING_PCT = float(os.getenv("TREND_SWING_PCT", 5.0))  # a swing high/low is a turning point followed by a move of at least this %
TREND_MAX_DAYS  = int(os.getenv("TREND_MAX_DAYS", 30))      # keep signals up to this many days old
TREND_OUT_FILE  = os.getenv("TREND_OUT_FILE", "data/trend_change.json")
CHUNK          = int(os.getenv("CHUNK", 100))
TICKER_FILE    = os.getenv("TICKER_FILE", "data/support_tickers.csv")  # needs a 'Symbol' column
OUT_FILE       = os.getenv("OUT_FILE", "data/support.json")


def zigzag(high, low, pct):
    """Return list of pivots: (index, price, 'H'|'L')."""
    p = pct / 100.0
    n = len(high)
    pivots, trend = [], None
    hi_i, hi = 0, high[0]
    lo_i, lo = 0, low[0]
    for i in range(1, n):
        if trend is None:
            if high[i] > hi: hi_i, hi = i, high[i]
            if low[i] < lo:  lo_i, lo = i, low[i]
            if hi >= lo * (1 + p) and hi_i > lo_i:      # first leg is up
                pivots.append((lo_i, lo, "L")); trend = "up"
            elif lo <= hi * (1 - p) and lo_i > hi_i:    # first leg is down
                pivots.append((hi_i, hi, "H")); trend = "down"
        elif trend == "up":
            if high[i] > hi:
                hi_i, hi = i, high[i]
            elif low[i] <= hi * (1 - p):
                pivots.append((hi_i, hi, "H")); trend = "down"
                lo_i, lo = i, low[i]
        else:  # down
            if low[i] < lo:
                lo_i, lo = i, low[i]
            elif high[i] >= lo * (1 + p):
                pivots.append((lo_i, lo, "L")); trend = "up"
                hi_i, hi = i, high[i]
    return pivots


def analyse(df):
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < 60:
        return None
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    dates = df.index
    cmp_ = float(close[-1])
    pivots = zigzag(high, low, SWING_PCT)
    best = None
    for k, (i, price, typ) in enumerate(pivots):
        if typ != "L":
            continue
        # swing size = rally from this low to the next confirmed high (or highest high so far)
        nxt_high = next((pv[1] for pv in pivots[k + 1:] if pv[2] == "H"), None)
        top = nxt_high if nxt_high is not None else float(high[i:].max())
        swing_up = (top / price - 1) * 100
        if swing_up < SWING_PCT:
            continue
        # support must be intact: no close below it since the pivot
        if (close[i:] < price * (1 - BREAK_TOL_PCT / 100)).any():
            continue
        dist = (cmp_ / price - 1) * 100
        if dist < -BREAK_TOL_PCT or dist > PROXIMITY_PCT:
            continue
        # prior swing: fall into this low from the preceding high
        prev_high = next((pv[1] for pv in reversed(pivots[:k]) if pv[2] == "H"), None)
        swing_down = (price / prev_high - 1) * 100 if prev_high else None
        # retests: separate visits (>=3 bars apart) back to within 1.5% of the level after the pivot
        touches, last = 0, i
        for j in range(i + 3, len(low)):
            if low[j] <= price * 1.015 and j - last >= 3:
                touches += 1; last = j
        cand = dict(
            support=round(price, 2),
            support_date=str(dates[i].date()),
            dist_pct=round(dist, 2),
            swing_up_pct=round(swing_up, 1),
            swing_down_pct=round(swing_down, 1) if swing_down is not None else None,
            retests=touches,
            days_since=int(len(close) - 1 - i),
        )
        # prefer the support the price is closest to
        if best is None or abs(cand["dist_pct"]) < abs(best["dist_pct"]):
            best = cand
    if best:
        best["cmp"] = round(cmp_, 2)
        best["date"] = str(dates[-1].date())
    return best


def analyse_trend(df):
    """Change of trend on the DAILY chart, same swing definition as the support scanner.
    Swings are the turning points of the price (a swing only counts once price has moved
    TREND_SWING_PCT away from it).   Uptrend = HH & HL.   Downtrend = LL & LH.
    BULLISH change: downtrend (LL and LH), then price closes ABOVE the last lower high.
    BEARISH change: uptrend (HH and HL), then price closes BELOW the last higher low.
    Cancelled if price later closes back through the swing that preceded the break."""
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < 60:
        return None
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    dates = df.index
    piv = zigzag(high, low, TREND_SWING_PCT)
    n = len(close)
    best = None
    for k in range(3, len(piv)):
        a, b, c, d = piv[k - 3:k + 1]
        types = a[2] + b[2] + c[2] + d[2]
        if types == "LHLH":                      # L1 H1 L2 H2
            L1, H1, L2, H2 = a[1], b[1], c[1], d[1]
            if not (L2 < L1 and H2 < H1):        # need lower low AND lower high
                continue
            hit = np.where(close[d[0] + 1:] > H2)[0]
            if not len(hit):
                continue
            j = d[0] + 1 + int(hit[0])
            ref = float(low[d[0]:j + 1].min())   # the swing low that preceded the break
            if (close[j:] < ref).any():
                continue
            sig = dict(direction="Bullish", level=H2, ref=ref)
        elif types == "HLHL":                    # H1 L1 H2 L2
            H1, L1, H2, L2 = a[1], b[1], c[1], d[1]
            if not (H2 > H1 and L2 > L1):        # need higher high AND higher low
                continue
            hit = np.where(close[d[0] + 1:] < L2)[0]
            if not len(hit):
                continue
            j = d[0] + 1 + int(hit[0])
            ref = float(high[d[0]:j + 1].max())  # the swing high that preceded the break
            if (close[j:] > ref).any():
                continue
            sig = dict(direction="Bearish", level=L2, ref=ref)
        else:
            continue
        if n - 1 - j > TREND_MAX_DAYS:
            continue
        if best is None or j > best[0]:
            best = (j, sig)
    if not best:
        return None
    j, sig = best
    cmp_ = float(close[-1])
    return dict(
        direction=sig["direction"],
        cmp=round(cmp_, 2),
        level=round(float(sig["level"]), 2),
        pct_from_level=round((cmp_ / sig["level"] - 1) * 100, 2),
        ref=round(float(sig["ref"]), 2),
        pct_to_ref=round((cmp_ / sig["ref"] - 1) * 100, 2),
        trigger_date=str(dates[j].date()),
        days_since=int(n - 1 - j),
        date=str(dates[-1].date()),
    )


def load_universe():
    t = pd.read_csv(TICKER_FILE)
    sym_col = next(c for c in t.columns if c.lower() in ("symbol", "ticker"))
    desc_col = next((c for c in t.columns if c.lower() in ("description", "name")), None)
    t["sym"] = t[sym_col].astype(str).str.strip().str.upper()
    t["name"] = t[desc_col] if desc_col else t["sym"]
    return t[["sym", "name"]].drop_duplicates("sym")


def main():
    uni = load_universe()
    syms = uni["sym"].tolist()
    names = dict(zip(uni["sym"], uni["name"]))
    results, trend, failed = [], [], 0
    for s in range(0, len(syms), CHUNK):
        batch = syms[s:s + CHUNK]
        yf_syms = [x + ".NS" for x in batch]
        try:
            data = yf.download(yf_syms, period=LOOKBACK, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("batch failed", s, e, file=sys.stderr); failed += len(batch); continue
        for sym, ys in zip(batch, yf_syms):
            try:
                df = data[ys] if len(batch) > 1 else data
                r = analyse(df)
                if r:
                    r.update(symbol=sym, name=names.get(sym, sym))
                    results.append(r)
                t = analyse_trend(df)
                if t:
                    t.update(symbol=sym, name=names.get(sym, sym))
                    trend.append(t)
            except Exception:
                failed += 1
        print(f"{min(s + CHUNK, len(syms))}/{len(syms)} scanned, {len(results)} at support, {len(trend)} trend changes", flush=True)
        time.sleep(1)
    results.sort(key=lambda r: abs(r["dist_pct"]))
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(dict(
            updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            params=dict(swing_pct=SWING_PCT, proximity_pct=PROXIMITY_PCT, break_tol_pct=BREAK_TOL_PCT),
            scanned=len(syms), failed=failed, results=results), f)
    trend.sort(key=lambda r: r["days_since"])
    with open(TREND_OUT_FILE, "w") as f:
        json.dump(dict(
            updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            params=dict(swing_pct=TREND_SWING_PCT, max_days=TREND_MAX_DAYS),
            scanned=len(syms), failed=failed, results=trend), f)
    print(f"Done. {len(results)} at support, {len(trend)} trend changes. Failed: {failed}")


if __name__ == "__main__":
    main()
