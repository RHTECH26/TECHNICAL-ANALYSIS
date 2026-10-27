#!/usr/bin/env python3
"""
Model portfolio: the COT + Relative Strength strategy run on the Nifty 500 stocks, compared with the Nifty 500 index.

Rules (the same as the Pine Script strategy "COT + Relative Strength Backtest"):
  BUY   a stock when its daily Dow Theory COT turns bullish (close above the last lower high) AND
        1) relative strength: since the Nifty 500's latest confirmed swing high AND since its latest confirmed swing low,
           the stock has gained more than the Nifty 500 (swings known on that day only: no look-ahead)
        2) COT buy level: the level broken now is higher than the level broken by that stock's previous COT buy signal
        -> the order is filled at the next day's OPEN
  SELL  when the stock's WEEKLY COT turns bearish (weekly close below the last weekly higher low, last CLOSED week)
        -> filled at the open of the day after the first trading day of the new week (as the Pine strategy does)

Portfolio: equal weight, at most MAX_POSITIONS positions (each = equity / MAX_POSITIONS), costs on both sides.
If more buy signals appear than there are free slots, the stocks that beat the index by the most are taken first.
Prices are split-adjusted, not dividend-adjusted (price returns, like a TradingView chart).

Writes data/model_portfolio.json (read by the Model Portfolio tab of index.html).
NOTE: uses TODAY's Nifty 500 members for the whole history, so stocks that left the index are missing (survivorship bias).
"""
import csv, json, math, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    import yfinance as yf
except ImportError:          # lets the engine be imported for testing without yfinance
    yf = None

OUT_FILE = os.environ.get("OUT_FILE", "data/model_portfolio.json")
CFG = dict(
    start=os.environ.get("MP_START", "2015-01-01"),         # the performance chart starts here
    warmup=os.environ.get("MP_WARMUP", "2011-01-01"),       # earlier data so swings and trends exist on day one
    max_positions=int(os.environ.get("MP_MAX_POSITIONS", 20)),
    cost_pct=float(os.environ.get("MP_COST_PCT", 0.10)),    # per side: brokerage, taxes and slippage
    cot_swing=int(os.environ.get("MP_COT_SWING", 5)),
    wk_swing=int(os.environ.get("MP_WK_SWING", 5)),
    rs_left=int(os.environ.get("MP_RS_LEFT", 10)),
    rs_right=int(os.environ.get("MP_RS_RIGHT", 10)),
    rs_edge=float(os.environ.get("MP_RS_EDGE", 0.0)),       # percentage points
    allow_first=os.environ.get("MP_ALLOW_FIRST", "0") == "1",
    rf_pct=float(os.environ.get("MP_RF_PCT", 6.0)),         # risk-free rate for Sharpe / Sortino / alpha
)
BENCH_TICKERS = ["^CRSLDX", "^NSEI"]
BENCH_NAMES = {"^CRSLDX": "Nifty 500", "^NSEI": "Nifty 50"}


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
    __slots__ = ("sym", "name", "dates", "open", "close", "entries", "exits", "last_state")


def stock_signals(sym, name, df, bench_dates, bclose, ph_anchor, pl_anchor, cfg):
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
        entries.append({"bar": int(b), "date": dates[b], "level": float(lvl[b]), "score": float((h + l) / 2), "exH": float(h), "exL": float(l),
                        "exec_bar": int(b + 1) if b + 1 < n else None})

    # weekly COT exit: the weekly trend of the last CLOSED week, seen from each daily bar
    wk = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    exits = []
    if len(wk) >= 2 * cfg["wk_swing"] + 2:
        w_trend, _, _, _, _ = dow_engine(wk["High"].to_numpy(float), wk["Low"].to_numpy(float), wk["Close"].to_numpy(float), cfg["wk_swing"])
        prev = pd.Series(w_trend, index=wk.index).shift(1)
        d_index = df.index
        week_label = (d_index.normalize() + pd.to_timedelta((4 - d_index.weekday).to_numpy(), unit="D"))
        dwk = prev.reindex(week_label).fillna(0).to_numpy()
        flip = (dwk == -1) & np.concatenate([[False], dwk[:-1] != -1])
        for t in np.nonzero(flip)[0]:
            exits.append({"bar": int(t), "date": dates[t], "exec_bar": int(t + 1) if t + 1 < n else None})
        last_wk = int(dwk[-1]) if n else 0
    else:
        last_wk = 0

    s = StockSignals()
    s.sym, s.name, s.dates, s.open, s.close = sym, name, dates, O, C
    s.entries, s.exits = entries, exits
    s.last_state = {"wk": last_wk, "exH": float(exH[-1]) if len(exH) and exH[-1] == exH[-1] else None,
                    "exL": float(exL[-1]) if len(exL) and exL[-1] == exL[-1] else None}
    return s


# ====================================================================== portfolio simulation
def simulate(stocks, cal, cfg):
    """cal: DatetimeIndex of benchmark trading days from the start date.  Returns equity, exposure, trades, open positions, next-session signals."""
    N = cfg["max_positions"]
    cost = cfg["cost_pct"] / 100.0
    pos_of = {s.sym: i for i, s in enumerate(stocks)}
    nd = len(cal)
    # prices aligned to the calendar
    Oc = np.full((len(stocks), nd), np.nan)
    Cc = np.full((len(stocks), nd), np.nan)
    for i, s in enumerate(stocks):
        idx = np.searchsorted(cal.values, s.dates)
        ok = (idx < nd) & (cal.values[np.minimum(idx, nd - 1)] == s.dates)
        Oc[i, idx[ok]] = s.open[ok]
        Cc[i, idx[ok]] = s.close[ok]
    Cf = pd.DataFrame(Cc.T).ffill().to_numpy().T                         # last known close
    Of = np.where(np.isnan(Oc), Cf, Oc)

    # execution schedules: calendar-day -> list of buys; stock -> sorted exits
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
    eq = np.zeros(nd)
    invested = np.zeros(nd)
    trades = []
    for d in range(nd):
        # 1) exits at today's open
        for i in list(pos):
            p = pos[i]
            hit = next((ed for fd, ed in exits[i] if fd >= p["entry_d"] and ed == d), None)
            if hit is not None and Oc[i, d] == Oc[i, d]:
                px = Oc[i, d]
                proceeds = p["shares"] * px * (1 - cost)
                cash += proceeds
                trades.append(_trade(stocks[i], p, cal[d], px, proceeds, "Weekly COT sell"))
                del pos[i]
        # 2) buys at today's open (strongest relative strength first), equal weight
        cands = [(i, e) for i, e in buys.get(d, []) if i not in pos and Oc[i, d] == Oc[i, d]]
        cands.sort(key=lambda t: -t[1]["score"])
        for i, e in cands:
            if len(pos) >= N:
                break
            eq_open = cash + sum(p["shares"] * Of[j, d] for j, p in pos.items())
            amount = min(cash, eq_open / N)
            if amount < 0.5 * eq_open / N:
                continue
            px = Oc[i, d]
            shares = amount / (px * (1 + cost))
            cash -= amount
            pos[i] = {"shares": shares, "entry_d": d, "entry_px": px, "cost_amt": amount, "level": e["level"], "w": amount / eq_open * 100}
        # 3) mark to market at the close
        mv = sum(p["shares"] * Cf[j, d] for j, p in pos.items())
        eq[d] = cash + mv
        invested[d] = mv / eq[d] * 100 if eq[d] else 0

    last_d = nd - 1
    opens = []
    for i, p in pos.items():
        s = stocks[i]
        pending = any(x["exec_bar"] is None for x in s.exits)          # a weekly sell signal came on the last day: it executes at the next open
        cmp_ = Cf[i, last_d]
        opens.append({"s": s.sym, "n": s.name, "ed": str(cal[p["entry_d"]].date()), "ep": round(float(p["entry_px"]), 2), "cmp": round(float(cmp_), 2),
                      "r": round(float((cmp_ / (p["entry_px"] * (1 + cost)) - 1) * 100), 2), "h": int((cal[last_d] - cal[p["entry_d"]]).days),
                      "w": round(float(p["shares"] * cmp_ / eq[last_d] * 100), 2), "lvl": round(float(p["level"]), 2),
                      "exH": None if s.last_state["exH"] is None else round(s.last_state["exH"], 2),
                      "exL": None if s.last_state["exL"] is None else round(s.last_state["exL"], 2),
                      "wk": "UP" if s.last_state["wk"] == 1 else "DOWN" if s.last_state["wk"] == -1 else "-", "exit_next": bool(pending)})
    opens.sort(key=lambda o: -o["w"])
    # buy signals generated on the last bar (they would be filled at the next open)
    nxt = []
    for i, s in enumerate(stocks):
        for e in s.entries:
            if e["exec_bar"] is None and i not in pos:
                nxt.append({"s": s.sym, "n": s.name, "lvl": round(e["level"], 2), "exH": round(e["exH"], 2), "exL": round(e["exL"], 2), "score": round(e["score"], 2)})
    nxt.sort(key=lambda x: -x["score"])
    cash_pct = cash / eq[last_d] * 100
    return eq, invested, trades, opens, nxt, round(cash_pct, 2)


def _trade(s, p, d_exit, px, proceeds, why):
    ret = (proceeds / p["cost_amt"] - 1) * 100
    return {"s": s.sym, "n": s.name, "ed": None, "ep": round(float(p["entry_px"]), 2), "xd": str(pd.Timestamp(d_exit).date()), "xp": round(float(px), 2),
            "r": round(float(ret), 2), "pnl": round(float(proceeds - p["cost_amt"]), 4), "w": round(float(p["w"]), 2), "why": why, "_entry_d": int(p["entry_d"]), "_days_cal": None}


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
            "avg_exposure": float(np.mean(invested))}


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


def run(frames, names, bench_close, cfg=CFG):
    """frames: {symbol: OHLC DataFrame}; bench_close: Series (index = dates). Returns the JSON payload."""
    bench_close = bench_close.dropna()
    bdates = bench_close.index.values
    bclose = bench_close.to_numpy(float)
    ph_a, pl_a = bench_anchors(bclose, cfg["rs_left"], cfg["rs_right"])
    stocks = []
    for sym, df in frames.items():
        stocks.append(stock_signals(sym, names.get(sym, sym), df, bdates, bclose, ph_a, pl_a, cfg))
    cal = bench_close.index[bench_close.index >= pd.Timestamp(cfg["start"])]
    eq, invested, trades, opens, nxt, cash_pct = simulate(stocks, cal, cfg)

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
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    clean = lambda d: {k: (None if isinstance(v, float) and not math.isfinite(v) else (round(v, 3) if isinstance(v, float) else v)) for k, v in d.items()}
    return {
        "updated": ist.strftime("%d %b %Y, %H:%M IST"), "asof": str(cal[-1].date()),
        "config": {"start": cfg["start"], "max_positions": cfg["max_positions"], "cost_pct": cfg["cost_pct"], "rf_pct": cfg["rf_pct"],
                   "universe": f"today's Nifty 500 members ({len(stocks)} stocks), past index changes not reflected", "cot_swing": cfg["cot_swing"], "wk_swing": cfg["wk_swing"],
                   "rs_swing": f"{cfg['rs_left']}/{cfg['rs_right']}"},
        "dates": [str(d.date()) for d in cal], "pf": [round(float(x), 3) for x in eq], "bm": [round(float(x), 3) for x in bm],
        "invested": [round(float(x), 1) for x in invested],
        "metrics": {"portfolio": clean(ps), "benchmark": clean(bs), "relative": clean(rel), "trades": clean(ts)},
        "cash_pct": cash_pct, "open": opens, "signals": nxt[:30], "trades": trades,
    }


def main():
    if yf is None:
        raise SystemExit("yfinance is not installed")
    members = load_members()
    names = dict(members)
    print("Universe:", len(members), "stocks")
    bench = None
    for bt in BENCH_TICKERS:
        d = yf.download(bt, start=CFG["warmup"], interval="1d", auto_adjust=False, progress=False)
        if d is not None and len(d) > 1000:
            s = d["Close"].iloc[:, 0] if hasattr(d["Close"], "columns") else d["Close"]
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize() if pd.DatetimeIndex(s.index).tz is not None else pd.DatetimeIndex(s.index).normalize()
            bench, bench_name = s.dropna(), BENCH_NAMES[bt]
            print("benchmark:", bench_name, len(bench), "days from", bench.index[0].date())
            break
    if bench is None:
        raise SystemExit("could not download the benchmark index")
    frames = download([m[0] + ".NS" for m in members], CFG["warmup"])
    frames = {k[:-3]: v for k, v in frames.items()}
    if len(frames) < 100:
        raise SystemExit("too little price data came back - try again later")
    payload = run(frames, names, bench)
    payload["config"]["benchmark"] = bench_name
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    m = payload["metrics"]
    print(f"Done. {len(payload['trades'])} closed trades, {len(payload['open'])} open | CAGR {m['portfolio']['cagr']}% vs {m['benchmark']['cagr']}% "
          f"| max DD {m['portfolio']['max_drawdown']}% | size {os.path.getsize(OUT_FILE) // 1024} KB")


if __name__ == "__main__":
    main()
