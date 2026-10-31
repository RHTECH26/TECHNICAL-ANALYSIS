#!/usr/bin/env python3
"""
Right Horizons - NSE Growth & Valuation Screener  (data builder)

What it does, once per run:
  1. Pulls the WHOLE NSE common-stock universe from TradingView's screener
     (price, TTM EPS, last-FY EPS & revenue, next-FY EPS & revenue estimates,
     sector, industry).
  2. Pulls the SECOND forward year (and a cross-check of the first) from Yahoo
     Finance for every analyst-covered stock.
  3. Computes, per stock:
        EPS growth  : FY1E vs last reported FY, FY2E vs FY1E
        Rev growth  : same two
        Industry PE : median PE of peers in the same industry
        FV          = Industry PE x TTM EPS             (+ % diff vs CMP)
        TGT FY1     = Industry PE x FY1E EPS            (+ % diff vs CMP)
        TGT FY2     = Industry PE x FY2E EPS            (+ % diff vs CMP)
  4. Writes data/growth.json, which growth.html reads.

FY1 = first unreported fiscal year, FY2 = the one after (for March year-end
companies in late 2026 that is FY27 and FY28).
"""
import json
import math
import os
import random
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ----------------------------------------------------------------- settings --
OUT_FILE = "data/growth.json"
CACHE_FILE = "data/yahoo_cache.json"

MIN_MCAP_FOR_PE_CR = 500      # peers smaller than this (Rs Cr) are ignored when computing Industry PE
MIN_PEERS = 5                 # fewer peers than this -> fall back to sector, then whole market
PE_MIN, PE_MAX = 0.0, 150.0   # peers outside this PE band are ignored (loss-makers / absurd PEs)

YAHOO_WORKERS = 6
YAHOO_CACHE_DAYS = 3          # re-fetch a stock's Yahoo estimates only after this many days
YAHOO_MAX_SYMBOLS = 1800      # safety cap per run
YAHOO_FETCH_IF_MCAP_CR = 2000 # also try Yahoo for big stocks with no TradingView estimate

LOW_BASE_GROWTH_PCT = 300     # growth above this is flagged "low base"
SOURCE_MISMATCH = 0.25        # TradingView vs Yahoo FY1 EPS differing by more than 25% is flagged

TV_COLS = [
    "name", "description", "close", "market_cap_basic", "sector", "industry",
    "earnings_per_share_diluted_ttm",
    "earnings_per_share_diluted_fy",
    "earnings_per_share_forecast_next_fy",
    "total_revenue_fy",
    "revenue_forecast_next_fy",
]


# Indices offered in the page's Index filter: (TradingView code, label, group).
# If one of these ever stops returning stocks it is skipped and noted in the log.
INDEXES = [
    ("NIFTY",             "Nifty 50",                "Broad"),
    ("NIFTYJR",           "Nifty Next 50",           "Broad"),
    ("CNX100",            "Nifty 100",               "Broad"),
    ("CNX200",            "Nifty 200",               "Broad"),
    ("CNX500",            "Nifty 500",               "Broad"),
    ("NIFTYMIDCAP150",    "Nifty Midcap 150",        "Broad"),
    ("NIFTYSMLCAP250",    "Nifty Smallcap 250",      "Broad"),
    ("NIFTY_MICROCAP250", "Nifty Microcap 250",      "Broad"),
    ("BANKNIFTY",         "Nifty Bank",              "Sectoral"),
    ("NIFTYPVTBANK",      "Nifty Private Bank",      "Sectoral"),
    ("CNXPSUBANK",        "Nifty PSU Bank",          "Sectoral"),
    ("CNXFINANCE",        "Nifty Financial Services","Sectoral"),
    ("NIFTY_CAPITAL_MKT", "Nifty Capital Markets",   "Sectoral"),
    ("CNXIT",             "Nifty IT",                "Sectoral"),
    ("CNXPHARMA",         "Nifty Pharma",            "Sectoral"),
    ("NIFTY_HEALTHCARE",  "Nifty Healthcare",        "Sectoral"),
    ("CNXAUTO",           "Nifty Auto",              "Sectoral"),
    ("CNXFMCG",           "Nifty FMCG",              "Sectoral"),
    ("NIFTY_CONSR_DURBL", "Nifty Consumer Durables", "Sectoral"),
    ("CNXMETAL",          "Nifty Metal",             "Sectoral"),
    ("CNXREALTY",         "Nifty Realty",            "Sectoral"),
    ("CNXENERGY",         "Nifty Energy",            "Sectoral"),
    ("NIFTY_OIL_AND_GAS", "Nifty Oil & Gas",         "Sectoral"),
    ("CNXINFRA",          "Nifty Infrastructure",    "Sectoral"),
    ("NIFTY_IND_DEFENCE", "Nifty India Defence",     "Sectoral"),
    ("CNXPSE",            "Nifty PSE",               "Sectoral"),
    ("CNXMEDIA",          "Nifty Media",             "Sectoral"),
    ("CNXCOMMODITIES",    "Nifty Commodities",       "Sectoral"),
]

# ------------------------------------------------------------------ helpers --
def num(x):
    """Return a clean float or None."""
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def pct(new, base):
    """% change new vs base; only meaningful when base > 0."""
    new, base = num(new), num(base)
    if new is None or base is None or base <= 0:
        return None
    return (new / base - 1.0) * 100.0


def rnd(x, n=2):
    x = num(x)
    return None if x is None else round(x, n)


# ----------------------------------------------------- 1. TradingView universe --
def fetch_universe():
    from tradingview_screener import Query, Column

    frames = []
    page, offset = 1000, 0
    while True:
        q = (
            Query()
            .set_markets("india")
            .select(*TV_COLS)
            .where(
                Column("exchange") == "NSE",
                Column("type") == "stock",
                Column("typespecs").has("common"),
            )
            .order_by("market_cap_basic", ascending=False)
            .offset(offset)
            .limit(page)
        )
        total, df = q.get_scanner_data()
        if df is None or df.empty:
            break
        frames.append(df)
        offset += page
        if offset >= total:
            break
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("TradingView returned no rows")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ticker")
    print(f"TradingView: {len(df)} NSE stocks")
    return df


def fetch_index_members():
    """Returns ({index_code: set(tickers)}, [meta dicts]) for every index in INDEXES."""
    from tradingview_screener import Query

    members, meta = {}, []
    for code, label, group in INDEXES:
        try:
            total, df = Query().select("name").set_index(f"SYML:NSE;{code}").limit(1000).get_scanner_data()
            tickers = set(df["ticker"]) if df is not None and not df.empty else set()
        except Exception as e:
            print(f"  index {code}: failed ({str(e)[:60]})")
            continue
        if not tickers:
            print(f"  index {code}: no stocks returned, skipped")
            continue
        members[code] = tickers
        meta.append({"k": code, "l": label, "g": group, "n": len(tickers)})
        time.sleep(0.3)
    print(f"Indices: {len(meta)} of {len(INDEXES)} loaded")
    return members, meta


# ---------------------------------------------------------- 2. Yahoo estimates --
def yahoo_symbol(tv_name):
    # TradingView writes BAJAJ_AUTO, Yahoo writes BAJAJ-AUTO
    return tv_name.replace("_", "-") + ".NS"


def _cell(frame, period, col):
    try:
        return num(frame.loc[period, col])
    except Exception:
        return None


def yahoo_one(tv_name):
    import yfinance as yf

    last_err = None
    for attempt in range(3):
        try:
            t = yf.Ticker(yahoo_symbol(tv_name))
            ee = t.earnings_estimate
            re_ = t.revenue_estimate
            out = {
                "eps0": _cell(ee, "0y", "avg"),
                "eps1": _cell(ee, "+1y", "avg"),
                "eps_base": _cell(ee, "0y", "yearAgoEps"),
                "n0": _cell(ee, "0y", "numberOfAnalysts"),
                "n1": _cell(ee, "+1y", "numberOfAnalysts"),
                "rev0": _cell(re_, "0y", "avg"),
                "rev1": _cell(re_, "+1y", "avg"),
                "rev_base": _cell(re_, "0y", "yearAgoRevenue"),
                "ts": dt.datetime.utcnow().isoformat(timespec="seconds"),
            }
            return tv_name, out
        except Exception as e:  # network / rate limit / missing data
            last_err = e
            time.sleep(1.5 * (attempt + 1) + random.random())
    return tv_name, {"error": str(last_err)[:80], "ts": dt.datetime.utcnow().isoformat(timespec="seconds")}


def load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def fetch_yahoo(names):
    cache = load_cache()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=YAHOO_CACHE_DAYS)
    todo = []
    for n in names:
        c = cache.get(n)
        fresh = False
        if c and c.get("ts"):
            try:
                fresh = dt.datetime.fromisoformat(c["ts"]) > cutoff and "error" not in c
            except ValueError:
                fresh = False
        if not fresh:
            todo.append(n)
    todo = todo[:YAHOO_MAX_SYMBOLS]
    print(f"Yahoo: {len(names)} wanted, {len(names) - len(todo)} cached, fetching {len(todo)}")

    done = 0
    with ThreadPoolExecutor(max_workers=YAHOO_WORKERS) as ex:
        futs = [ex.submit(yahoo_one, n) for n in todo]
        for fut in as_completed(futs):
            n, res = fut.result()
            cache[n] = res
            done += 1
            if done % 100 == 0:
                print(f"  ... {done}/{len(todo)}")
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, separators=(",", ":"))
    return cache


# ----------------------------------------------------------- 3. Computation --
def add_industry_pe(df):
    """Adds ind_pe, ind_pe_n, ind_pe_basis columns (median PE of peers)."""
    peers = df[
        (df["pe"] > PE_MIN) & (df["pe"] <= PE_MAX) & (df["mcap_cr"] >= MIN_MCAP_FOR_PE_CR)
    ]
    ind = peers.groupby("industry")["pe"].agg(["median", "count"])
    sec = peers.groupby("sector")["pe"].agg(["median", "count"])
    mkt = float(peers["pe"].median()) if len(peers) else None

    vals, ns, basis = [], [], []
    for _, r in df.iterrows():
        i, s = r["industry"], r["sector"]
        if i in ind.index and ind.loc[i, "count"] >= MIN_PEERS:
            vals.append(ind.loc[i, "median"]); ns.append(int(ind.loc[i, "count"])); basis.append("industry")
        elif s in sec.index and sec.loc[s, "count"] >= MIN_PEERS:
            vals.append(sec.loc[s, "median"]); ns.append(int(sec.loc[s, "count"])); basis.append("sector")
        else:
            vals.append(mkt); ns.append(int(len(peers))); basis.append("market")
    df["ind_pe"], df["ind_pe_n"], df["ind_pe_basis"] = vals, ns, basis
    return df


def build_rows(tv, yahoo, members=None):
    df = pd.DataFrame({
        "ticker": tv["ticker"],
        "name": tv["name"],
        "desc": tv["description"],
        "sector": tv["sector"].fillna("Other"),
        "industry": tv["industry"].fillna("Other"),
        "cmp": tv["close"].map(num),
        "mcap_cr": tv["market_cap_basic"].map(num) / 1e7,
        "eps_ttm": tv["earnings_per_share_diluted_ttm"].map(num),
        "eps_fy": tv["earnings_per_share_diluted_fy"].map(num),
        "tv_eps1": tv["earnings_per_share_forecast_next_fy"].map(num),
        "rev_fy": tv["total_revenue_fy"].map(num),
        "tv_rev1": tv["revenue_forecast_next_fy"].map(num),
    })
    # PE computed from the same EPS used for FV, so FV is internally consistent
    df["pe"] = [c / e if (c and e and e > 0) else None for c, e in zip(df["cmp"], df["eps_ttm"])]
    df["pe"] = df["pe"].astype(float)
    df = add_industry_pe(df)

    # pandas turns missing numbers into NaN; convert back to None so "is None" checks work
    df = df.astype(object).where(df.notna(), None)

    members = members or {}
    ix_of = {}
    for code, tickers in members.items():
        for t in tickers:
            ix_of.setdefault(t, []).append(code)

    rows = []
    for r in df.itertuples(index=False):
        y = yahoo.get(r.name) or {}
        flags = []

        # ---- EPS path ----
        base_eps = r.eps_fy if r.eps_fy is not None else num(y.get("eps_base"))
        eps1 = r.tv_eps1 if r.tv_eps1 is not None else num(y.get("eps0"))
        eps2 = num(y.get("eps1"))
        y_eps0 = num(y.get("eps0"))
        # base for the FY2 growth: Yahoo's own FY1 when we have it (same source, same basis)
        base_for_g2 = y_eps0 if y_eps0 is not None else eps1

        # ---- Revenue path ----
        base_rev = r.rev_fy if r.rev_fy is not None else num(y.get("rev_base"))
        rev1 = r.tv_rev1 if r.tv_rev1 is not None else num(y.get("rev0"))
        rev2 = num(y.get("rev1"))
        y_rev0 = num(y.get("rev0"))
        base_for_rg2 = y_rev0 if y_rev0 is not None else rev1

        # unit sanity: a revenue estimate 20x bigger / smaller than last FY is a currency/unit problem
        if base_rev and rev1 and not (0.05 <= rev1 / base_rev <= 20):
            flags.append("unit")
            rev1 = rev2 = None
            base_for_rg2 = None
        if base_rev and rev2 and base_for_rg2 and not (0.05 <= rev2 / base_for_rg2 <= 20):
            rev2 = None

        eps_g1 = pct(eps1, base_eps)
        eps_g2 = pct(eps2, base_for_g2)
        rev_g1 = pct(rev1, base_rev)
        rev_g2 = pct(rev2, base_for_rg2)

        # ---- flags ----
        if r.eps_ttm is None or r.eps_ttm <= 0:
            flags.append("loss")
        if base_eps is not None and base_eps <= 0 and eps1 is not None and eps1 > 0:
            flags.append("turn")        # turnaround: growth not meaningful
        if eps_g1 is not None and eps_g1 > LOW_BASE_GROWTH_PCT:
            flags.append("lowbase")
        if r.tv_eps1 and y_eps0 and r.tv_eps1 > 0 and y_eps0 > 0:
            if abs(r.tv_eps1 / y_eps0 - 1) > SOURCE_MISMATCH:
                flags.append("src")
        if r.ind_pe_basis != "industry":
            flags.append("peer" + r.ind_pe_basis[0])   # peers = sector / market fallback

        # ---- fair value & targets ----
        ipe = num(r.ind_pe)
        fv = ipe * r.eps_ttm if (ipe and r.eps_ttm and r.eps_ttm > 0) else None
        tgt1 = ipe * eps1 if (ipe and eps1 and eps1 > 0) else None
        tgt2 = ipe * eps2 if (ipe and eps2 and eps2 > 0) else None

        def diff(v):
            return pct(v, r.cmp) if (v is not None and r.cmp and r.cmp > 0) else None

        rows.append({
            "t": r.ticker, "s": r.name, "n": r.desc, "sec": r.sector, "ind": r.industry,
            "cmp": rnd(r.cmp), "mc": rnd(r.mcap_cr, 0),
            "eps_ttm": rnd(r.eps_ttm), "eps_fy": rnd(base_eps),
            "eps1": rnd(eps1), "eps2": rnd(eps2),
            "eg1": rnd(eps_g1, 1), "eg2": rnd(eps_g2, 1),
            "rg1": rnd(rev_g1, 1), "rg2": rnd(rev_g2, 1),
            "pe": rnd(r.pe, 1), "ipe": rnd(ipe, 1), "ipe_n": r.ind_pe_n, "ipe_b": r.ind_pe_basis,
            "fv": rnd(fv), "fvd": rnd(diff(fv), 1),
            "t1": rnd(tgt1), "t1d": rnd(diff(tgt1), 1),
            "t2": rnd(tgt2), "t2d": rnd(diff(tgt2), 1),
            "na": int(y["n0"]) if num(y.get("n0")) is not None else None,
            "ix": ix_of.get(r.ticker, []),
            "fl": flags,
        })
    return rows


# --------------------------------------------------------------------- main --
def main():
    tv = fetch_universe()
    tv["name"] = tv["name"].astype(str)

    wanted = []
    for r in tv.itertuples(index=False):
        covered = num(r.earnings_per_share_forecast_next_fy) is not None
        big = (num(r.market_cap_basic) or 0) / 1e7 >= YAHOO_FETCH_IF_MCAP_CR
        if covered or big:
            wanted.append(r.name)
    yahoo = fetch_yahoo(wanted)

    members, index_meta = fetch_index_members()
    rows = build_rows(tv, yahoo, members)
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    payload = {
        "updated": dt.datetime.now(ist).strftime("%d %b %Y, %H:%M IST"),
        "count": len(rows),
        "with_fy1": sum(1 for r in rows if r["eps1"] is not None),
        "with_fy2": sum(1 for r in rows if r["eps2"] is not None),
        "indices": index_meta,
        "settings": {
            "min_mcap_for_pe_cr": MIN_MCAP_FOR_PE_CR,
            "min_peers": MIN_PEERS,
            "pe_band": [PE_MIN, PE_MAX],
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    print(f"Wrote {OUT_FILE}: {len(rows)} stocks, {payload['with_fy1']} with FY1E, {payload['with_fy2']} with FY2E")


if __name__ == "__main__":
    sys.exit(main())
