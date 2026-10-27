#!/usr/bin/env python3
"""
Model portfolios: the COT + Relative Strength strategy run as THREE separate portfolios, each compared with its own benchmark.

  stocks      Nifty 500 stocks (INR)              benchmark Nifty 500
  global_etf  US-listed ETFs from your list (USD) benchmark MSCI ACWI (the ACWI ETF), always; only funds with average volume above 200,000
  india_etf   Indian-listed ETFs (INR)            benchmark Nifty 500; only funds with average volume above 5,000

Rules (the same as the Pine Script strategy "COT + Relative Strength Backtest"):
  BUY   when the daily Dow Theory COT turns bullish (close above the last lower high) AND
        1) relative strength: since the benchmark's latest confirmed swing high AND since its latest confirmed swing low,
           it has gained more than the benchmark (swings known on that day only: no look-ahead)
        2) COT buy level: the level broken now is higher than the level broken by its previous COT buy signal
        -> filled at the next day's OPEN
  SELL  when the WEEKLY COT turns bearish (weekly close below the last weekly higher low, last CLOSED week)
        -> filled at the open of the day after the first trading day of the new week (as the Pine strategy does)
  (Optional, OFF by default: set MP_RS_EXIT=1 to also sell when the holding lags the benchmark on BOTH sides, since its swing high AND
        swing low, without waiting for a weekly COT sell, at the next day's open.)

Portfolio: every new position is POSITION_PCT % of the portfolio value on the day it is bought, paid from cash. Stocks: 5% each and no cap
(the cash decides). The two ETF portfolios: at most 10 holdings of 10% each. Costs are charged on both sides.
  - Stop-loss: a holding bought while its WEEKLY COT is already bearish gets a STOP_PCT % (25%) stop-loss from its buy price.
  - Half positions (OFF by default, MP_HALF_POSITIONS=1 turns it on): when the cash left is between 2.5% and 5% of the portfolio value, a half position is bought and topped up to 5% later
    (never while it is below its buy price or while its weekly COT is bearish).
  - Swap (OFF by default, MP_SWAP=1 turns it on): when there is no cash for a new signal, the weakest holding is sold if the new signal's relative strength leads it by at least
    SWAP_MARGIN points (holding at least SWAP_MIN_DAYS old, at most one swap every SWAP_GAP_DAYS trading days).
  - One holding per theme (the two ETF portfolios): funds that track the same index family count as one theme, so overlapping funds
    do not fill the portfolio with the same bet. The strongest signal of a theme is bought first.
If more signals appear than the cash can fund, the ones that beat the benchmark by the most are taken first.
Prices are split-adjusted, not dividend-adjusted (price returns, like a TradingView chart).

Usage:  python model_portfolio.py                    (all three)     python model_portfolio.py stocks india_etf
Writes data/mp_stocks.json, data/mp_global_etf.json, data/mp_india_etf.json (read by the Model Portfolio tab of index.html).
NOTE: stocks use TODAY's Nifty 500 members and ETFs use today's lists for the whole history (survivorship bias).
"""
import csv, json, math, os, re, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    import yfinance as yf
except ImportError:          # lets the engine be imported for testing without yfinance
    yf = None

CFG = dict(
    start=os.environ.get("MP_START", "2015-01-01"),         # the performance chart starts here
    warmup=os.environ.get("MP_WARMUP", "2011-01-01"),       # earlier data so swings and trends exist on day one
    max_positions=int(os.environ.get("MP_MAX_POSITIONS", 0)),      # 0 = no cap: the cash decides
    position_pct=float(os.environ.get("MP_POSITION_PCT", 5.0)),     # size of each new position, % of the portfolio value that day
    min_mcap_cr=float(os.environ.get("MP_MIN_MCAP_CR", 0)),         # optional market cap filter in rupees crore (0 = off, the default)
    stop_pct=float(os.environ.get("MP_STOP_PCT", 25.0)),            # stop-loss from the buy price (0 = off)
    stop_only_under_weekly_sell=os.environ.get("MP_STOP_ALWAYS", "0") != "1",
    half_positions=os.environ.get("MP_HALF_POSITIONS", "0") == "1",   # OFF by default. When on: spare cash between 2.5% and 5% buys a half position, topped up later
    half_min_frac=float(os.environ.get("MP_HALF_MIN_FRAC", 0.5)),
    swap=os.environ.get("MP_SWAP", "0") == "1",                       # OFF by default. When on: no cash for a new signal sells the weakest holding if the new one is clearly stronger
    swap_margin=float(os.environ.get("MP_SWAP_MARGIN", 10.0)),        # new signal must lead the weakest holding by this many percentage points
    swap_min_days=int(os.environ.get("MP_SWAP_MIN_DAYS", 30)),        # a holding must be at least this old (calendar days) to be swapped out
    swap_gap_days=int(os.environ.get("MP_SWAP_GAP_DAYS", 5)),         # at most one swap in this many trading days
    one_per_theme=None,                                               # ETF portfolios: one holding per theme (None = decided per portfolio)
    rs_exit=os.environ.get("MP_RS_EXIT", "0") == "1",                 # OFF by default. When on: sell at the next open if the holding lags the benchmark on BOTH sides
    rs_exit_edge=float(os.environ.get("MP_RS_EXIT_EDGE", 0.0)),       # lagging = excess return below this many percentage points, since the swing high AND the swing low
    rs_exit_days=int(os.environ.get("MP_RS_EXIT_DAYS", 1)),           # for this many closes in a row
    cost_pct=float(os.environ.get("MP_COST_PCT", 0.10)),    # per side: brokerage, taxes and slippage
    cot_swing=int(os.environ.get("MP_COT_SWING", 5)),
    wk_swing=int(os.environ.get("MP_WK_SWING", 5)),
    rs_left=int(os.environ.get("MP_RS_LEFT", 10)),
    rs_right=int(os.environ.get("MP_RS_RIGHT", 10)),
    rs_edge=float(os.environ.get("MP_RS_EDGE", 0.0)),       # percentage points
    allow_first=os.environ.get("MP_ALLOW_FIRST", "0") == "1",
    rf_pct=float(os.environ.get("MP_RF_PCT", 6.0)),         # risk-free rate for Sharpe / Sortino / alpha
    capital=float(os.environ.get("MP_CAPITAL", 10000000)),  # notional starting capital in rupees (1 crore): only used to show gains in rupees
)
SHARES_FILE = os.environ.get("MP_SHARES_FILE", "data/mp_shares.json")
BENCH_TICKERS = ["^CRSLDX", "^NSEI"]


# ====================================================================== Dow Theory engine
def _pivot_values(high, low, L, R):
    """Swing highs / lows (unique extreme of the L bars before and R bars after), reported on the bar that confirms them."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    win = L + R + 1
    if n < win:
        return ph, pl
    hv, lv = sliding_window_view(high, win), sliding_window_view(low, win)
    ch, cl = high[L:n - R], low[L:n - R]
    is_h = (ch == hv.max(axis=1)) & ((hv == ch[:, None]).sum(axis=1) == 1)
    is_l = (cl == lv.min(axis=1)) & ((lv == cl[:, None]).sum(axis=1) == 1)
    ph[np.nonzero(is_h)[0] + L + R] = ch[is_h]
    pl[np.nonzero(is_l)[0] + L + R] = cl[is_l]
    return ph, pl


def dow_engine(high, low, close, L, confirm_close=True):
    """Bar-by-bar Dow Theory state machine (the same rules as the Pine indicator).
    Returns trend (+1 up, -1 down, 0 unknown), bullish event, bearish event, and for bullish events the broken level and the swing low."""
    n = len(close)
    trend_arr = np.zeros(n, np.int8)
    bull = np.zeros(n, bool)
    bear = np.zeros(n, bool)
    lvl = np.full(n, np.nan)
    stop = np.full(n, np.nan)
    ph_val, pl_val = _pivot_values(high, low, L, L)
    nan = float("nan")
    lastPH = prevPH = lastPL = prevPL = nan
    trend = 0
    for t in range(n):
        v = ph_val[t]
        if v == v:
            prevPH, lastPH = lastPH, v
        v = pl_val[t]
        if v == v:
            prevPL, lastPL = lastPL, v
        if trend == 0 and prevPH == prevPH and prevPL == prevPL:
            if lastPH > prevPH and lastPL > prevPL:
                trend = 1
            elif lastPH < prevPH and lastPL < prevPL:
                trend = -1
        if trend == 1 and lastPL == lastPL and (close[t] < lastPL if confirm_close else low[t] < lastPL):
            trend = -1
            bear[t] = True
        elif trend == -1 and lastPH == lastPH and (close[t] > lastPH if confirm_close else high[t] > lastPH):
            lvl[t], stop[t] = lastPH, lastPL
            trend = 1
            bull[t] = True
        trend_arr[t] = trend
    return trend_arr, bull, bear, lvl, stop


def bench_anchors(bclose, L, R):
    """For every benchmark day: index of the bar of the latest CONFIRMED swing high and swing low (-1 if none yet)."""
    ph, pl = _pivot_values(bclose, bclose, L, R)
    n = len(bclose)
    out = []
    for arr in (ph, pl):
        conf = np.nonzero(~np.isnan(arr))[0]                  # days on which a swing was confirmed
        anchor = np.full(n, -1)
        anchor[conf] = conf - R                               # the swing bar itself
        idx = np.where(anchor >= 0, np.arange(n), 0)
        idx = np.maximum.accumulate(idx)
        res = anchor[idx]
        res[: conf[0] if len(conf) else n] = -1
        out.append(res)
    return out[0], out[1]


# ====================================================================== per-stock signals
class StockSignals:
    __slots__ = ("sym", "name", "dates", "open", "low", "close", "entries", "exits", "last_state", "score_b", "dwk", "theme", "rs_bad")


def stock_signals(sym, name, df, bench_dates, bclose, ph_anchor, pl_anchor, cfg, shares=None):
    """df: Open/High/Low/Close on the stock's own trading days.  Returns StockSignals."""
    dates = df.index.values
    O, H, Lw, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    n = len(C)
    _, bull, _, lvl, _ = dow_engine(H, Lw, C, cfg["cot_swing"])
    bi = np.nonzero(bull)[0]
    ok = np.zeros(len(bi), bool)
    if len(bi) > 1:
        ok[1:] = lvl[bi][1:] > lvl[bi][:-1]
    if len(bi) and cfg["allow_first"]:
        ok[0] = True

    # relative strength on the benchmark calendar
    cs = np.array(pd.Series(C, index=df.index).reindex(bench_dates, method="ffill").to_numpy(), dtype=float)     # a writable copy
    first = np.searchsorted(bench_dates, dates[0])
    cs[:first] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        def excess(anchor):
            a = np.where(anchor >= 0, anchor, 0)
            ex = ((cs / cs[a] - 1) - (bclose / bclose[a] - 1)) * 100
            ex[anchor < 0] = np.nan
            return ex
        exH, exL = excess(ph_anchor), excess(pl_anchor)
    bpos = np.searchsorted(bench_dates, dates, side="right") - 1          # benchmark index on each stock bar

    # weekly COT: the weekly trend of the last CLOSED week, seen from each daily bar (1 up, -1 down, 0 unknown)
    wk = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    dwk = np.zeros(n, np.int8)
    if len(wk) >= 2 * cfg["wk_swing"] + 2:
        w_trend, _, _, _, _ = dow_engine(wk["High"].to_numpy(float), wk["Low"].to_numpy(float), wk["Close"].to_numpy(float), cfg["wk_swing"])
        prev = pd.Series(w_trend, index=wk.index).shift(1)
        d_index = df.index
        week_label = (d_index.normalize() + pd.to_timedelta((4 - d_index.weekday).to_numpy(), unit="D"))
        dwk = prev.reindex(week_label).fillna(0).to_numpy().astype(np.int8)
    flip = (dwk == -1) & np.concatenate([[False], dwk[:-1] != -1])
    exits = [{"bar": int(t), "date": dates[t], "exec_bar": int(t + 1) if t + 1 < n else None} for t in np.nonzero(flip)[0]]
    last_wk = int(dwk[-1]) if n else 0

    # market cap (rupees crore) = price x share count; unknown share count -> not eligible
    mcap = C * float(shares) / 1e7 if shares else None

    entries = []
    for k, b in enumerate(bi):
        if not ok[k]:
            continue
        j = bpos[b]
        if j < 0:
            continue
        h, l = exH[j], exL[j]
        if h != h or l != l or not (h > cfg["rs_edge"] and l > cfg["rs_edge"]):
            continue
        if cfg["min_mcap_cr"] > 0 and (mcap is None or not mcap[b] >= cfg["min_mcap_cr"]):
            continue
        entries.append({"bar": int(b), "date": dates[b], "level": float(lvl[b]), "score": float((h + l) / 2), "exH": float(h), "exL": float(l),
                        "wk": int(dwk[b]), "mcap": None if mcap is None else float(mcap[b]),
                        "exec_bar": int(b + 1) if b + 1 < n else None})

    s = StockSignals()
    s.score_b = np.where(np.isnan(exH) | np.isnan(exL), np.nan, (exH + exL) / 2)      # relative strength score on the benchmark calendar
    s.dwk, s.theme = dwk, None
    with np.errstate(invalid="ignore"):
        s.rs_bad = (exH < cfg["rs_exit_edge"]) & (exL < cfg["rs_exit_edge"])          # lagging the benchmark since the swing high AND since the swing low
    s.sym, s.name, s.dates, s.open, s.low, s.close = sym, name, dates, O, Lw, C
    s.entries, s.exits = entries, exits
    s.last_state = {"wk": last_wk, "exH": float(exH[-1]) if len(exH) and exH[-1] == exH[-1] else None,
                    "exL": float(exL[-1]) if len(exL) and exL[-1] == exL[-1] else None}
    return s


# ====================================================================== portfolio simulation
def simulate(stocks, cal, cfg, bench_off=0):
    """cal: DatetimeIndex of benchmark trading days from the start date (bench_off = its position in the full benchmark calendar).
    Returns equity, exposure, trades, open positions, next-session signals, cash (index units) and daily flows / counters."""
    cost = cfg["cost_pct"] / 100.0
    pos_pct = cfg["position_pct"] / 100.0
    cap_n = cfg["max_positions"] if cfg["max_positions"] and cfg["max_positions"] > 0 else None
    stop_pct = cfg["stop_pct"]
    half_on, half_frac = bool(cfg.get("half_positions")), float(cfg.get("half_min_frac", 0.5))
    swap_on, swap_margin = bool(cfg.get("swap")), float(cfg.get("swap_margin", 10.0))
    swap_min_days, swap_gap = int(cfg.get("swap_min_days", 30)), int(cfg.get("swap_gap_days", 5))
    theme_on = bool(cfg.get("one_per_theme"))
    nd = len(cal)
    ns = len(stocks)
    Oc = np.full((ns, nd), np.nan)
    Lc = np.full((ns, nd), np.nan)
    Cc = np.full((ns, nd), np.nan)
    Wc = np.zeros((ns, nd), np.int8)             # weekly COT state on each day (1 up, -1 down)
    Sc = np.full((ns, nd), np.nan)               # relative strength score at each close
    Bc = np.zeros((ns, nd), bool)                # lagging the benchmark on both sides at each close
    rs_exit_on, rs_exit_days = bool(cfg.get("rs_exit")), max(1, int(cfg.get("rs_exit_days", 1)))
    for i, s in enumerate(stocks):
        idx = np.searchsorted(cal.values, s.dates)
        ok = (idx < nd) & (cal.values[np.minimum(idx, nd - 1)] == s.dates)
        Oc[i, idx[ok]] = s.open[ok]
        Lc[i, idx[ok]] = s.low[ok]
        Cc[i, idx[ok]] = s.close[ok]
        Wc[i, idx[ok]] = s.dwk[ok]
        sb = s.score_b[bench_off: bench_off + nd]
        Sc[i, : len(sb)] = sb
        bb = s.rs_bad[bench_off: bench_off + nd]
        Bc[i, : len(bb)] = bb
    Cf = pd.DataFrame(Cc.T).ffill().to_numpy().T                         # last known close
    Of = np.where(np.isnan(Oc), Cf, Oc)

    # execution schedules: calendar-day -> list of buys; stock -> weekly sells
    buys = {}
    exits = {}
    for i, s in enumerate(stocks):
        for e in s.entries:
            if e["exec_bar"] is None:
                continue
            d = s.dates[e["exec_bar"]]
            di = np.searchsorted(cal.values, d)
            if di < nd and cal.values[di] == d:
                buys.setdefault(di, []).append((i, e))
        ex = []
        for x in s.exits:
            if x["exec_bar"] is None:
                continue
            fd = np.searchsorted(cal.values, x["date"])
            ed = np.searchsorted(cal.values, s.dates[x["exec_bar"]])
            if ed < nd and cal.values[ed] == s.dates[x["exec_bar"]]:
                ex.append((int(fd), int(ed)))
        exits[i] = ex

    cash = 100.0
    pos = {}                         # stock index -> dict
    th_count = {}                    # theme -> holdings
    eq = np.zeros(nd)
    invested = np.zeros(nd)
    buy_amt = np.zeros(nd)
    sell_amt = np.zeros(nd)
    npos = np.zeros(nd, int)
    trades = []
    counters = {"swaps": 0, "partial_buys": 0, "topups": 0, "cash_skips": 0, "theme_skips": 0, "slot_skips": 0}
    last_swap = -10 ** 6

    def sell(i, d, px, why):
        nonlocal cash
        p = pos.pop(i)
        proceeds = p["shares"] * px * (1 - cost)
        cash += proceeds
        sell_amt[d] += proceeds
        if theme_on and stocks[i].theme is not None:
            th_count[stocks[i].theme] = th_count.get(stocks[i].theme, 1) - 1
        trades.append(_trade(stocks[i], p, cal[d], px, proceeds, why))
        trades[-1]["th"] = stocks[i].theme

    def equity_at_open(d):
        return cash + sum(q["shares"] * Of[j, d] for j, q in pos.items())

    for d in range(nd):
        # 1) sales at today's open: weekly COT sells, and stops that the stock gapped below
        for i in list(pos):
            p = pos[i]
            if Oc[i, d] != Oc[i, d]:
                continue
            if p["stop"] is not None and Oc[i, d] <= p["stop"]:
                sell(i, d, Oc[i, d], "Stop-loss %d%%" % stop_pct)
                continue
            if any(fd >= p["entry_d"] and ed == d for fd, ed in exits[i]):
                sell(i, d, Oc[i, d], "Weekly COT sell")
                continue
            if p.get("rs_flag"):
                sell(i, d, Oc[i, d], "Relative strength exit")
        # 2) buys at today's open, strongest relative strength first
        cands = [(i, e) for i, e in buys.get(d, []) if i not in pos and Oc[i, d] == Oc[i, d]]
        cands.sort(key=lambda t: -t[1]["score"])
        for ci, (i, e) in enumerate(cands):
            th = stocks[i].theme if theme_on else None
            if th is not None and th_count.get(th, 0) > 0:
                counters["theme_skips"] += 1                       # this theme is already held (one holding per theme)
                continue
            if cap_n is not None and len(pos) >= cap_n:
                counters["slot_skips"] += len(cands) - ci          # the portfolio is full: the rest of today's signals are skipped
                break
            eq_open = equity_at_open(d)
            target = eq_open * pos_pct
            partial = False
            if cash >= FULL_TOL * target:
                amount = min(cash, target)
            elif half_on and cash >= half_frac * target - 1e-9:
                amount, partial = cash, True                         # not enough for a full position: use what is there
            else:
                j = None                                             # try to fund it by selling the weakest holding
                if swap_on and d > 0 and d - last_swap >= swap_gap:
                    best = None
                    need = (half_frac if half_on else FULL_TOL) * target
                    for jj, q in pos.items():
                        if (cal[d] - cal[q["entry_d"]]).days < swap_min_days or Oc[jj, d] != Oc[jj, d]:
                            continue
                        sc = Sc[jj, d - 1]
                        if sc != sc or e["score"] - sc < swap_margin:
                            continue
                        if cash + q["shares"] * Oc[jj, d] * (1 - cost) < need - 1e-9:
                            continue
                        if best is None or sc < best[0]:
                            best = (sc, jj)
                    j = best[1] if best else None
                if j is None:
                    counters["cash_skips"] += 1
                    break                                            # the cash is used up: lower-ranked signals cannot be funded either
                sell(j, d, Oc[j, d], "Swapped for %s" % stocks[i].sym)
                counters["swaps"] += 1
                last_swap = d
                eq_open = equity_at_open(d)
                target = eq_open * pos_pct
                if cash >= FULL_TOL * target:
                    amount = min(cash, target)
                else:
                    amount, partial = cash, True
            px = Oc[i, d]
            shares = amount / (px * (1 + cost))
            cash -= amount
            buy_amt[d] += amount
            if partial:
                counters["partial_buys"] += 1
            under_weekly_sell = e.get("wk", 0) == -1
            use_stop = stop_pct > 0 and (under_weekly_sell or not cfg["stop_only_under_weekly_sell"])
            pos[i] = {"shares": shares, "entry_d": d, "entry_px": px, "cost_amt": amount, "level": e["level"], "w": amount / eq_open * 100,
                      "stop": px * (1 - stop_pct / 100.0) if use_stop else None, "under_weekly_sell": under_weekly_sell, "partial": partial}
            if th is not None:
                th_count[th] = th_count.get(th, 0) + 1
        # 3) top up half positions with spare cash (never a position below its buy price, never while its weekly COT is bearish)
        if half_on and cash > 0:
            partials = [j for j, q in pos.items() if q.get("partial") and q["entry_d"] < d and not q.get("rs_flag")]
            partials.sort(key=lambda j: -(Sc[j, d - 1] if d > 0 and Sc[j, d - 1] == Sc[j, d - 1] else -1e9))
            for j in partials:
                q = pos[j]
                px = Oc[j, d]
                if px != px or px < q["entry_px"] or (d > 0 and Wc[j, d - 1] == -1):
                    continue
                eq_open = equity_at_open(d)
                target = eq_open * pos_pct
                missing = target - q["shares"] * Of[j, d]
                if missing < 0.1 * target:
                    q["partial"] = False
                    continue
                amount = min(cash, missing)
                if amount < 0.1 * target:
                    continue
                new_sh = amount / (px * (1 + cost))
                q["entry_px"] = (q["entry_px"] * q["shares"] + px * new_sh) / (q["shares"] + new_sh)
                q["shares"] += new_sh
                q["cost_amt"] += amount
                q["w"] += amount / eq_open * 100
                cash -= amount
                buy_amt[d] += amount
                counters["topups"] += 1
                if q["shares"] * Of[j, d] >= 0.98 * target:
                    q["partial"] = False
        # 4) stops hit during the day: sold at the stop price
        for i in list(pos):
            p = pos[i]
            if p["stop"] is not None and Lc[i, d] == Lc[i, d] and Lc[i, d] <= p["stop"]:
                sell(i, d, p["stop"], "Stop-loss %d%%" % stop_pct)
        # 5) relative strength exit: lagging the benchmark on both sides at this close -> sold at the next open
        if rs_exit_on:
            for j, q in pos.items():
                q["bad_run"] = q.get("bad_run", 0) + 1 if Bc[j, d] else 0
                if q["bad_run"] >= rs_exit_days:
                    q["rs_flag"] = True
        # 6) mark to market at the close
        mv = sum(p["shares"] * Cf[j, d] for j, p in pos.items())
        eq[d] = cash + mv
        invested[d] = mv / eq[d] * 100 if eq[d] else 0
        npos[d] = len(pos)

    last_d = nd - 1
    opens = []
    for i, p in pos.items():
        s = stocks[i]
        pending = any(x["exec_bar"] is None for x in s.exits)          # a weekly sell signal came on the last day: it executes at the next open
        cmp_ = Cf[i, last_d]
        opens.append({"s": s.sym, "n": s.name, "th": s.theme, "ed": str(cal[p["entry_d"]].date()), "ep": round(float(p["entry_px"]), 2), "cmp": round(float(cmp_), 2),
                      "r": round(float((cmp_ / (p["entry_px"] * (1 + cost)) - 1) * 100), 2), "h": int((cal[last_d] - cal[p["entry_d"]]).days),
                      "w": round(float(p["shares"] * cmp_ / eq[last_d] * 100), 2), "lvl": round(float(p["level"]), 2),
                      "cost": float(p["cost_amt"]), "val": float(p["shares"] * cmp_), "half": bool(p.get("partial")),
                      "stop": None if p["stop"] is None else round(float(p["stop"]), 2), "uws": bool(p["under_weekly_sell"]),
                      "exH": None if s.last_state["exH"] is None else round(s.last_state["exH"], 2),
                      "exL": None if s.last_state["exL"] is None else round(s.last_state["exL"], 2),
                      "wk": "UP" if s.last_state["wk"] == 1 else "DOWN" if s.last_state["wk"] == -1 else "-", "exit_next": bool(pending or p.get("rs_flag")),
                      "exit_why": "weekly" if pending else ("rs" if p.get("rs_flag") else None)})
    opens.sort(key=lambda o: -o["w"])
    held_themes = {stocks[i].theme for i in pos if stocks[i].theme is not None}
    nxt = []
    for i, s in enumerate(stocks):
        for e in s.entries:
            if e["exec_bar"] is None and i not in pos:
                nxt.append({"s": s.sym, "n": s.name, "th": s.theme, "lvl": round(e["level"], 2), "exH": round(e["exH"], 2), "exL": round(e["exL"], 2), "score": round(e["score"], 2),
                            "mcap": None if e.get("mcap") is None else round(e["mcap"]), "uws": e.get("wk", 0) == -1,
                            "theme_held": bool(theme_on and s.theme in held_themes)})
    nxt.sort(key=lambda x: -x["score"])
    return eq, invested, trades, opens, nxt, float(cash), {"buy": buy_amt, "sell": sell_amt, "npos": npos, "counters": counters}


def theme_of(txt):
    """Index family of an ETF, so overlapping funds count as ONE theme (S&P 500, S&P 500 Growth and S&P 500 Equal Weight are all 's&p 500')."""
    desc = (r"(growth|value|equal[ -]?weight(ed)?|high dividend|dividend leaders|dividend|quality|momentum|low volatility|minimum volatility|"
            r"min(imum)? vol(atility)?|total return|net total return|pure|aristocrats|enhanced|alpha)")
    t = (txt or "").replace("[- ]?", "").replace("-", " ")
    t = re.sub(r"[^A-Za-z0-9& ]", " ", t)
    core = re.sub(r"\b" + desc + r"\b(\s+\d+)?", " ", t, flags=re.I)
    core = re.sub(r"\s+", " ", core).strip().lower()
    base = re.sub(r"\s+", " ", t).strip().lower()
    return core if len(core.split()) >= 2 else base


def _trade(s, p, d_exit, px, proceeds, why):
    ret = (proceeds / p["cost_amt"] - 1) * 100
    return {"s": s.sym, "n": s.name, "ed": None, "ep": round(float(p["entry_px"]), 2), "xd": str(pd.Timestamp(d_exit).date()), "xp": round(float(px), 2),
            "r": round(float(ret), 2), "pnl": float(proceeds - p["cost_amt"]), "w": round(float(p["w"]), 2), "why": why, "_entry_d": int(p["entry_d"]), "_days_cal": None}


# ====================================================================== statistics
def _monthly(eq, dates):
    m = pd.Series(eq, index=dates).resample("ME").last()
    first = pd.Series(eq, index=dates).iloc[0]
    r = m.pct_change()
    r.iloc[0] = m.iloc[0] / first - 1
    return r


def perf_stats(eq, dates, rf_pct):
    eq = np.asarray(eq, float)
    r = np.diff(eq) / eq[:-1]
    years = (dates[-1] - dates[0]).days / 365.25
    total = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0.0
    sd = r.std(ddof=1)
    rfd = (1 + rf_pct / 100) ** (1 / 252) - 1
    sharpe = (r.mean() - rfd) / sd * math.sqrt(252) if sd else 0.0
    dd_dev = math.sqrt(np.mean(np.minimum(r - rfd, 0) ** 2)) * math.sqrt(252)
    sortino = (r.mean() - rfd) * 252 / dd_dev if dd_dev else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    imin = int(np.argmin(dd))
    ipeak = int(np.argmax(eq[: imin + 1]))
    longest_days = 0
    start_i = 0
    for i in range(len(eq)):
        if eq[i] >= peak[i]:
            start_i = i
        else:
            longest_days = max(longest_days, (dates[i] - dates[start_i]).days)
    mo = _monthly(eq, dates)
    return {"total_return": total * 100, "cagr": cagr * 100, "volatility": sd * math.sqrt(252) * 100, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": dd[imin] * 100, "max_dd_peak": str(dates[ipeak].date()), "max_dd_trough": str(dates[imin].date()),
            "longest_drawdown_days": int(longest_days),
            "calmar": cagr / abs(dd[imin]) if dd[imin] else 0.0,
            "best_month": mo.max() * 100, "worst_month": mo.min() * 100, "positive_months": float((mo > 0).mean() * 100),
            "best_day": r.max() * 100, "worst_day": r.min() * 100, "years": years}


def relative_stats(pf, bm, dates, rf_pct):
    rp, rb = np.diff(pf) / pf[:-1], np.diff(bm) / bm[:-1]
    rfd = (1 + rf_pct / 100) ** (1 / 252) - 1
    var_b = rb.var(ddof=1)
    beta = np.cov(rp, rb, ddof=1)[0, 1] / var_b if var_b else 0.0
    alpha = ((rp - rfd).mean() - beta * (rb - rfd).mean()) * 252
    active = rp - rb
    te = active.std(ddof=1) * math.sqrt(252)
    mp, mb = _monthly(pf, dates), _monthly(bm, dates)
    up, dn = mb > 0, mb < 0
    upcap = mp[up].mean() / mb[up].mean() * 100 if up.any() and mb[up].mean() else float("nan")
    dncap = mp[dn].mean() / mb[dn].mean() * 100 if dn.any() and mb[dn].mean() else float("nan")
    years = (dates[-1] - dates[0]).days / 365.25
    cagr_p, cagr_b = (pf[-1] / pf[0]) ** (1 / years) - 1, (bm[-1] / bm[0]) ** (1 / years) - 1
    return {"beta": beta, "alpha": alpha * 100, "correlation": float(np.corrcoef(rp, rb)[0, 1]), "tracking_error": te * 100,
            "information_ratio": (cagr_p - cagr_b) * 100 / (te * 100) if te else 0.0, "excess_cagr": (cagr_p - cagr_b) * 100,
            "up_capture": upcap, "down_capture": dncap, "months_beating_benchmark": float((mp > mb).mean() * 100)}


def trade_stats(trades, invested):
    if not trades:
        return {"trades": 0}
    r = np.array([t["r"] for t in trades])
    pnl = np.array([t["pnl"] for t in trades])
    win, loss = r > 0, r <= 0
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    streak = best = 0
    for x in r <= 0:
        streak = streak + 1 if x else 0
        best = max(best, streak)
    hold = np.array([t["h"] for t in trades])
    return {"trades": len(trades), "win_rate": float(win.mean() * 100), "avg_win": float(r[win].mean()) if win.any() else 0.0,
            "avg_loss": float(r[loss].mean()) if loss.any() else 0.0,
            "payoff_ratio": float(r[win].mean() / abs(r[loss].mean())) if win.any() and loss.any() and r[loss].mean() else 0.0,
            "profit_factor": float(gp / gl) if gl else float("inf"), "expectancy": float(r.mean()), "best_trade": float(r.max()), "worst_trade": float(r.min()),
            "avg_hold_days": float(hold.mean()), "median_hold_days": float(np.median(hold)), "max_consecutive_losses": int(best),
            "avg_exposure": float(np.mean(invested)),
            "stop_loss_exits": int(sum(1 for t in trades if t["why"].startswith("Stop-loss")))}


def churn_stats(flows, eq, cal):
    """Churn ratio = portfolio turnover: the smaller of money spent on purchases and money received from sales, divided by the
    average portfolio value (standard fund turnover).  Per calendar year and annualised over the whole period."""
    s = pd.DataFrame({"buy": flows["buy"], "sell": flows["sell"], "eq": eq, "n": flows["npos"]}, index=cal)
    by_year = {}
    for y, g in s.groupby(s.index.year):
        avg = g["eq"].mean()
        by_year[str(y)] = round(float(min(g["buy"].sum(), g["sell"].sum()) / avg * 100), 1) if avg else None
    years = (cal[-1] - cal[0]).days / 365.25
    total = min(s["buy"].sum(), s["sell"].sum()) / s["eq"].mean() * 100 / years if years > 0 else 0.0
    return {"churn_ratio": float(total), "avg_positions": float(s["n"].mean()), "max_positions_held": int(s["n"].max()),
            "bought_total_pct": float(s["buy"].sum() / s["eq"].mean() * 100 / years) if years > 0 else 0.0}, by_year


# ====================================================================== data + main
def load_members():
    path = os.path.join("index_lists", "nifty500.csv")
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sym = next((v for k, v in r.items() if k and k.strip().lower() == "symbol"), "").strip().upper()
            name = next((v for k, v in r.items() if k and k.strip().lower() in ("company name", "name")), sym).strip()
            if sym and not sym.startswith("DUMMY"):
                out.append((sym, name or sym))
    return list(dict.fromkeys(out))


def download(tickers, start, chunk=100):
    frames = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"Downloading {i + 1}-{i + len(part)} of {len(tickers)}", flush=True)
        data = None
        for attempt in range(3):
            try:
                data = yf.download(part, start=start, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
                if data is not None and len(data):
                    break
            except Exception as e:
                print("  chunk failed:", type(e).__name__)
            time.sleep(5 * (attempt + 1))
        if data is None or not len(data):
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for t in part:
            try:
                df = (data[t] if multi else data)[["Open", "High", "Low", "Close"]].dropna()
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize() if pd.DatetimeIndex(df.index).tz is not None else pd.DatetimeIndex(df.index).normalize()
                df = df[~df.index.duplicated(keep="last")]
                if len(df) > 300:
                    frames[t] = df
            except Exception:
                continue
        time.sleep(1)
    return frames


FULL_TOL = 0.97                  # cash within 3% of the target size buys a full position

PORTFOLIOS = {
    # one benchmark each, with NO silent fallback to another index: if it cannot be downloaded the portfolio is not built and the log says so
    "stocks":     dict(label="Stocks (Nifty 500)", noun="Stock", currency="INR", capital=1e7, bench=["^CRSLDX"], one_per_theme=False),
    "global_etf": dict(label="Global ETFs (US-listed)", noun="ETF", currency="USD", capital=1e6, bench=["ACWI"], one_per_theme=True, min_vol=200000, position_pct=10.0, max_positions=10),
    "india_etf":  dict(label="Indian ETFs", noun="ETF", currency="INR", capital=1e7, bench=["^CRSLDX"], one_per_theme=True, min_vol=5000, position_pct=10.0, max_positions=10),
}
BENCH_NAMES = {"^CRSLDX": "Nifty 500", "ACWI": "MSCI ACWI (ACWI ETF)"}
OUT_DIR = os.environ.get("MP_OUT_DIR", "data")
VARIANTS = [("Neither cash rule", dict(half_positions=False, swap=False)),
            ("Half positions only", dict(half_positions=True, swap=False)),
            ("Swap the weakest holding only", dict(half_positions=False, swap=True)),
            ("Both cash rules", dict(half_positions=True, swap=True)),
            ("Both cash rules, without the relative strength exit", dict(half_positions=True, swap=True, rs_exit=False))]


def cfg_for(pid):
    meta = PORTFOLIOS[pid]
    c = dict(CFG)
    c["capital"] = float(os.environ["MP_CAPITAL"]) if "MP_CAPITAL" in os.environ else meta["capital"]
    c["one_per_theme"] = (os.environ["MP_ONE_PER_THEME"] != "0") if "MP_ONE_PER_THEME" in os.environ else meta["one_per_theme"]
    if "MP_POSITION_PCT" not in os.environ and "position_pct" in meta:
        c["position_pct"] = meta["position_pct"]              # ETFs: each holding 10% of the portfolio
    if "MP_MAX_POSITIONS" not in os.environ and "max_positions" in meta:
        c["max_positions"] = meta["max_positions"]            # ETFs: at most 10 holdings
    if pid != "stocks":
        c["min_mcap_cr"] = 0
    return c


def _variant_summary(label, overrides, cfg, stocks, cal, bench_off, current=None):
    c2 = dict(cfg)
    c2.update(overrides)
    eq, invested, trades, opens, nxt, cash_units, flows = simulate(stocks, cal, c2, bench_off)
    ps = perf_stats(eq, cal, cfg["rf_pct"])
    ch, _ = churn_stats(flows, eq, cal)
    k = flows["counters"]
    return {"label": label, "half": bool(overrides["half_positions"]), "swap": bool(overrides["swap"]), "rs_exit": bool(c2.get("rs_exit")),
            "rs_exits": int(sum(1 for t in trades if t["why"].startswith("Relative strength"))), "cagr": ps["cagr"], "max_drawdown": ps["max_drawdown"], "sharpe": ps["sharpe"],
            "total_return": ps["total_return"], "churn": ch["churn_ratio"], "avg_positions": ch["avg_positions"], "trades": len(trades), "swaps": k["swaps"],
            "partial_buys": k["partial_buys"], "topups": k["topups"], "cash_skips": k["cash_skips"],
            "avg_cash": float(100 - np.mean(invested))}


def run(frames, names, bench_close, cfg=CFG, shares=None, themes=None, meta=None, variants=True):
    """frames: {symbol: OHLC DataFrame}; bench_close: Series (index = dates); shares: {symbol: share count}; themes: {symbol: theme}.
    Returns the JSON payload for one portfolio."""
    shares, themes, meta = shares or {}, themes or {}, meta or {"label": "Portfolio", "noun": "Stock", "currency": "INR"}
    bench_close = bench_close.dropna()
    bdates = bench_close.index.values
    bclose = bench_close.to_numpy(float)
    ph_a, pl_a = bench_anchors(bclose, cfg["rs_left"], cfg["rs_right"])
    stocks = []
    for sym, df in frames.items():
        st = stock_signals(sym, names.get(sym, sym), df, bdates, bclose, ph_a, pl_a, cfg, shares.get(sym))
        st.theme = themes.get(sym)
        stocks.append(st)
    cal = bench_close.index[bench_close.index >= pd.Timestamp(cfg["start"])]
    bench_off = int(np.searchsorted(bdates, np.datetime64(cal[0])))
    eq, invested, trades, opens, nxt, cash_units, flows = simulate(stocks, cal, cfg, bench_off)
    ch, churn_year = churn_stats(flows, eq, cal)
    cash_pct = round(cash_units / float(eq[-1]) * 100, 2)

    # entry dates + holding days for the trades list (entry_d is an index into cal)
    for t in trades:
        ed = cal[t.pop("_entry_d")]
        t["ed"] = str(ed.date())
        t["h"] = int((pd.Timestamp(t["xd"]) - ed).days)
        t.pop("_days_cal", None)
    trades.sort(key=lambda t: t["xd"], reverse=True)
    bm = bench_close.reindex(cal).to_numpy(float)
    bm = bm / bm[0] * 100
    ps, bs = perf_stats(eq, cal, cfg["rf_pct"]), perf_stats(bm, cal, cfg["rf_pct"])
    rel = relative_stats(eq, bm, cal, cfg["rf_pct"])
    ts = trade_stats(trades, invested)
    ts.update(ch)
    ts.update({k_: v for k_, v in flows["counters"].items()})
    ts["swap_exits"] = int(sum(1 for t in trades if t["why"].startswith("Swapped")))
    ts["rs_exits"] = int(sum(1 for t in trades if t["why"].startswith("Relative strength")))

    # ---- realised and unrealised gains, in the portfolio's currency on the notional capital ----
    k = cfg["capital"] / 100.0                       # the engine works in index units that start at 100
    realised = sum(t["pnl"] for t in trades)
    unrealised = sum(o["val"] - o["cost"] for o in opens)
    total = float(eq[-1]) - 100.0
    assert abs(total - (realised + unrealised)) < 1e-6, (total, realised, unrealised)      # nothing is lost between the two
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    by_year = {}
    for t in trades:
        by_year[t["xd"][:4]] = by_year.get(t["xd"][:4], 0.0) + t["pnl"]
    cap = cfg["capital"]
    pnl = {"capital": cap, "current_value": float(eq[-1]) * k, "total": total * k, "total_pct": total,
           "realised": realised * k, "realised_pct": realised, "realised_trades": len(trades),
           "gross_profit": gp * k, "gross_loss": gl * k,
           "unrealised": unrealised * k, "unrealised_pct": unrealised, "open_positions": len(opens),
           "unrealised_winners": sum(1 for o in opens if o["val"] > o["cost"]), "unrealised_losers": sum(1 for o in opens if o["val"] <= o["cost"]),
           "invested": sum(o["cost"] for o in opens) * k, "market_value": sum(o["val"] for o in opens) * k, "cash": cash_units * k,
           "realised_by_year": {y: round(v * k) for y, v in sorted(by_year.items())}}
    for t in trades:
        t["pnl"] = round(t["pnl"] * k)               # currency units per closed trade
    for o in opens:
        o["cost"] = round(o["cost"] * k)
        o["val"] = round(o["val"] * k)
        o["pnl"] = o["val"] - o["cost"]

    # ---- what the two cash-use rules change: the same strategy run with and without them ----
    var_out = []
    if variants:
        marked = False
        for label, ov in VARIANTS:
            if "rs_exit" in ov and not cfg.get("rs_exit"):
                continue                                    # nothing to compare when the relative strength exit is off
            v = _variant_summary(label, ov, cfg, stocks, cal, bench_off)
            same = (bool(cfg.get("half_positions")) == ov["half_positions"] and bool(cfg.get("swap")) == ov["swap"]
                    and bool(cfg.get("rs_exit")) == bool(ov.get("rs_exit", cfg.get("rs_exit"))))
            v["current"] = bool(same and not marked)
            marked = marked or v["current"]
            var_out.append(v)
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    clean = lambda d: {k: (None if isinstance(v, float) and not math.isfinite(v) else (round(v, 3) if isinstance(v, float) else v)) for k, v in d.items()}
    return {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"), "asof": str(cal[-1].date()),
        "config": {"label": meta["label"], "noun": meta["noun"], "currency": meta["currency"],
                   "start": cfg["start"], "max_positions": cfg["max_positions"], "position_pct": cfg["position_pct"], "min_mcap_cr": cfg["min_mcap_cr"],
                   "stop_pct": cfg["stop_pct"], "stop_only_under_weekly_sell": cfg["stop_only_under_weekly_sell"],
                   "half_positions": bool(cfg.get("half_positions")), "swap": bool(cfg.get("swap")), "swap_margin": cfg.get("swap_margin"), "swap_min_days": cfg.get("swap_min_days"),
                   "one_per_theme": bool(cfg.get("one_per_theme")),
                   "rs_exit": bool(cfg.get("rs_exit")), "rs_exit_edge": cfg.get("rs_exit_edge"), "rs_exit_days": cfg.get("rs_exit_days"),
                   "cost_pct": cfg["cost_pct"], "rf_pct": cfg["rf_pct"], "capital": cfg["capital"],
                   "universe": meta.get("universe") or f"{len(stocks)} {meta['noun'].lower()}s checked", "cot_swing": cfg["cot_swing"], "wk_swing": cfg["wk_swing"],
                   "rs_swing": f"{cfg['rs_left']}/{cfg['rs_right']}"},
        "dates": [str(d.date()) for d in cal], "pf": [round(float(x), 3) for x in eq], "bm": [round(float(x), 3) for x in bm],
        "invested": [round(float(x), 1) for x in invested],
        "metrics": {"portfolio": clean(ps), "benchmark": clean(bs), "relative": clean(rel), "trades": clean(ts), "churn_by_year": churn_year},
        "pnl": {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in pnl.items()},
        "variants": [{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in x.items()} for x in var_out],
        "cash_pct": cash_pct, "open": opens, "signals": nxt[:30], "trades": trades,
    }


def _fetch_shares(sym):
    """Share count of one NSE stock from Yahoo (a few attempts, because Yahoo sometimes refuses)."""
    for attempt in range(3):
        try:
            t = yf.Ticker(sym + ".NS")
            sh = None
            try:
                fi = t.fast_info
                sh = fi["shares"]
                if not sh:
                    mc, px = fi["marketCap"], fi["lastPrice"]
                    sh = mc / px if mc and px else None
            except Exception:
                sh = None
            if not sh:
                sh = (t.info or {}).get("sharesOutstanding")
            if sh and float(sh) > 0:
                return sym, float(sh)
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return sym, None


def load_shares(symbols, max_age_days=30):
    """Share counts, kept in data/mp_shares.json and refreshed once a month, so a normal run needs no extra Yahoo calls."""
    cache = {}
    try:
        with open(SHARES_FILE) as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    today = datetime.now(timezone.utc).date()
    todo = []
    for sym in symbols:
        c = cache.get(sym)
        try:
            age = (today - datetime.strptime(c["d"], "%Y-%m-%d").date()).days
        except Exception:
            age = 10 ** 6
        if c is None or age > max_age_days:
            todo.append(sym)
    if todo:
        print(f"Share counts: fetching {len(todo)} of {len(symbols)} (the rest come from the saved file)", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as ex:
            for sym, sh in ex.map(_fetch_shares, todo):
                if sh:
                    cache[sym] = {"s": sh, "d": str(today)}
        os.makedirs(os.path.dirname(SHARES_FILE) or ".", exist_ok=True)
        with open(SHARES_FILE, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
    out = {sym: cache[sym]["s"] for sym in symbols if sym in cache}
    missing = len(symbols) - len(out)
    print(f"Share counts known for {len(out)} stocks" + (f" ({missing} unknown: they cannot pass the market cap filter)" if missing else ""))
    return out


def page_list(var_name):
    """The ETF lists live inside index.html (GLOBAL_ETFS and TREE)."""
    text = open("index.html", encoding="utf-8").read()
    m = re.search(r"var\s+%s\s*=\s*(\[.*?\]);\s*\n" % var_name, text, re.S)
    if not m:
        raise RuntimeError("could not find %s inside index.html" % var_name)
    return json.loads(m.group(1))


def build_universe(pid):
    """(yahoo ticker, symbol, name, theme) for every fund / stock of one portfolio."""
    meta = PORTFOLIOS[pid]
    out = []
    if pid == "stocks":
        for sym, name in load_members():
            out.append((sym + ".NS", sym, name, None))
    elif pid == "india_etf":
        for a in page_list("TREE"):
            t = str(a.get("t", "")).strip().upper()
            if t and a.get("vol", 0) >= meta["min_vol"]:
                nm = a.get("sub") or t
                out.append((t + ".NS", t, nm, theme_of(nm)))
    elif pid == "global_etf":
        for a in page_list("GLOBAL_ETFS"):
            t = str(a.get("t", "")).strip()
            if not t or "." in t or a.get("vol", 0) < meta["min_vol"]:          # US-listed (USD) funds only: no exchange suffix
                continue
            out.append((t, t, a.get("n", t), theme_of(a.get("bench") or a.get("n") or t)))
    seen, res = set(), []
    for u in out:
        if u[0] not in seen:
            seen.add(u[0])
            res.append(u)
    return res


def get_benchmark(tickers, cache):
    for bt in tickers:
        if bt in cache:
            return cache[bt]
        try:
            d = yf.download(bt, start=CFG["warmup"], interval="1d", auto_adjust=False, progress=False)
        except Exception as e:
            print("  benchmark", bt, "failed:", type(e).__name__)
            continue
        if d is not None and len(d) > 1000:
            ser = d["Close"].iloc[:, 0] if hasattr(d["Close"], "columns") else d["Close"]
            ix = pd.DatetimeIndex(ser.index)
            ser.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
            cache[bt] = (ser.dropna(), BENCH_NAMES.get(bt, bt))
            print("benchmark:", cache[bt][1], len(cache[bt][0]), "days from", cache[bt][0].index[0].date())
            return cache[bt]
    raise RuntimeError("could not download a benchmark index (%s)" % ", ".join(tickers))


def build_one(pid, bench_cache):
    meta, cfg = PORTFOLIOS[pid], cfg_for(pid)
    print(f"\n===== {meta['label']} =====", flush=True)
    bench, bname = get_benchmark(meta["bench"], bench_cache)
    uni = build_universe(pid)
    print("Universe:", len(uni), meta["noun"].lower() + "s")
    frames_y = download([u[0] for u in uni], CFG["warmup"])
    names, themes, frames = {}, {}, {}
    for yt, sym, nm, th in uni:
        if yt in frames_y:
            frames[sym], names[sym], themes[sym] = frames_y[yt], nm, th
    if len(frames) < (50 if pid == "stocks" else 15):
        raise RuntimeError("too little price data came back (%d)" % len(frames))
    shares = load_shares(list(frames)) if (pid == "stocks" and cfg["min_mcap_cr"] > 0) else {}
    if pid == "stocks":
        universe_text = f"today's Nifty 500 members ({len(frames)} stocks checked), past index changes not reflected"
    elif pid == "india_etf":
        universe_text = f"Indian-listed ETFs from your list with average volume above {meta['min_vol']:,} ({len(frames)} funds), one holding per theme"
    else:
        universe_text = f"US-listed ETFs (USD) from your list with average volume above {meta['min_vol']:,} shares ({len(frames)} funds), one holding per theme"
    payload = run(frames, names, bench, cfg, shares, themes, dict(meta, universe=universe_text))
    payload["config"]["benchmark"] = bname
    path = os.path.join(OUT_DIR, f"mp_{pid}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    m = payload["metrics"]
    print(f"Done {meta['label']}: {len(payload['trades'])} closed trades, {len(payload['open'])} open | CAGR {m['portfolio']['cagr']}% vs {m['benchmark']['cagr']}% "
          f"| max DD {m['portfolio']['max_drawdown']}% | churn {m['trades']['churn_ratio']:.0f}% a year | swaps {m['trades']['swaps']} | half positions {m['trades']['partial_buys']} "
          f"| size {os.path.getsize(path) // 1024} KB")


def main(argv=None):
    if yf is None:
        raise SystemExit("yfinance is not installed")
    pids = [a for a in (argv if argv is not None else sys.argv[1:]) if a in PORTFOLIOS] or list(PORTFOLIOS)
    cache, failed = {}, []
    for pid in pids:
        try:
            build_one(pid, cache)
        except Exception as e:
            failed.append(pid)
            print(f"!! {PORTFOLIOS[pid]['label']} failed: {type(e).__name__}: {e}")
    if len(failed) == len(pids):
        raise SystemExit("no portfolio could be built")
    if failed:
        print("Built with problems, check the log above:", ", ".join(failed))


if __name__ == "__main__":
    main()
