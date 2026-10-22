#!/usr/bin/env python3
"""
Desk data for the Desk page of index.html.  Writes two small files into ./out :

  out/desk.json   market prices + change, top-5 gainers / losers (Nifty 500, S&P 500), Nifty 500 advances / declines
  out/news.json   market news from public RSS feeds, the last 24 hours only, tagged so the page can filter them

The workflow publishes both files to the "live-data" branch (one commit that is replaced every run, so the
repo history does not grow).  The previous news.json is read back from that branch so stories that dropped
out of a feed are still kept until they are 24 hours old.

Run locally:  python scan_desk.py
"""
import csv, html, io, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from calendar import timegm

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import feedparser
except ImportError:                      # the workflow installs it; the message below shows if it is missing
    feedparser = None

OUT_DIR    = os.environ.get("OUT_DIR", "out")
PREV_NEWS  = os.path.join(OUT_DIR, "prev_news.json")
PREV_DESK  = os.path.join(OUT_DIR, "prev_desk.json")
WINDOW_H   = 24
MAX_ITEMS  = 700
CHUNK      = 100
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/rss+xml, application/xml, text/xml, text/csv, */*"}

# ============================================================== 1. market quotes
QUOTES = [  # key, label, Yahoo symbol, kind
    ("nifty50",  "Nifty 50",         "^NSEI",     "index"),
    ("nifty500", "Nifty 500",        "^CRSLDX",   "index"),
    ("nasdaq",   "Nasdaq Composite", "^IXIC",     "index"),
    ("sp500",    "S&P 500",          "^GSPC",     "index"),
    ("usdx",     "US Dollar Index",  "DX-Y.NYB",  "index"),
    ("usdinr",   "USD / INR",        "INR=X",     "fx"),
    ("gold",     "Gold (USD/oz)",    "GC=F",      "usd"),
    ("silver",   "Silver (USD/oz)",  "SI=F",      "usd"),
    ("us10y",    "US 10Y yield",     "^TNX",      "yield"),
]


def last_two(series):
    s = series.dropna()
    if len(s) < 2:
        raise ValueError("fewer than 2 closes")
    return float(s.iloc[-1]), float(s.iloc[-2]), s.index[-1]


def fetch_quotes():
    data = yf.download([q[2] for q in QUOTES], period="10d", interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True, progress=False)
    out = []
    for key, label, tk, kind in QUOTES:
        try:
            last, prev, dt = last_two(data[tk]["Close"])
            out.append({"key": key, "label": label, "kind": kind, "price": round(last, 4), "prev": round(prev, 4),
                        "chg": round(last - prev, 4), "pct": round((last / prev - 1) * 100, 3),
                        "asof": pd.Timestamp(dt).strftime("%Y-%m-%d")})
        except Exception as e:
            print("quote failed:", tk, e)
            out.append({"key": key, "label": label, "kind": kind, "error": True})
    return out


# ============================================================== 2. movers + advance / decline
def read_csv_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col(row, *names):
    for k in row:
        if k and k.strip().lower() in names:
            return (row[k] or "").strip()
    return ""


def nifty500_members():
    rows = read_csv_rows(os.path.join("index_lists", "nifty500.csv"))
    out = []
    for r in rows:
        sym = col(r, "symbol").upper()
        if sym and not sym.startswith("DUMMY"):
            out.append((sym + ".NS", sym, col(r, "company name", "name") or sym))
    return list(dict.fromkeys(out))


def sp500_members():
    local = os.path.join("index_lists", "sp500.csv")
    text = None
    if os.path.exists(local):
        text = open(local, encoding="utf-8-sig").read()
    else:
        try:
            req = urllib.request.Request("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", headers=UA)
            text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception as e:
            print("sp500 list download failed:", e)
    if not text:
        raise RuntimeError("no S&P 500 member list")
    out = []
    for r in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        sym = col(r, "symbol").upper()
        if sym:
            out.append((sym.replace(".", "-"), sym, col(r, "security", "name") or sym))
    if len(out) < 400:
        raise RuntimeError("S&P 500 list looks incomplete")
    return list(dict.fromkeys(out))


def movers(members, label):
    rows = []
    for i in range(0, len(members), CHUNK):
        part = members[i:i + CHUNK]
        try:
            data = yf.download([m[0] for m in part], period="10d", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print(f"  {label}: chunk failed:", e)
            continue
        for ysym, sym, name in part:
            try:
                last, prev, dt = last_two(data[ysym]["Close"])
                if prev > 0:
                    rows.append((sym, name, last, last - prev, (last / prev - 1) * 100, pd.Timestamp(dt)))
            except Exception:
                continue
        time.sleep(1)
    if len(rows) < 50:
        raise RuntimeError(f"{label}: too little price data ({len(rows)} stocks)")
    newest = max(r[5] for r in rows)
    rows = [r for r in rows if (newest - r[5]).days <= 3]         # drop stale / suspended stocks
    pack = lambda r: {"s": r[0], "n": r[1], "p": round(r[2], 2), "d": round(r[3], 2), "c": round(r[4], 2)}
    ordered = sorted(rows, key=lambda r: r[4])
    return {"asof": newest.strftime("%Y-%m-%d"), "count": len(rows),
            "adv": sum(1 for r in rows if r[4] > 0), "dec": sum(1 for r in rows if r[4] < 0),
            "unch": sum(1 for r in rows if r[4] == 0),
            "gainers": [pack(r) for r in ordered[::-1][:5]], "losers": [pack(r) for r in ordered[:5]]}


# ============================================================== 3. news
# name, url, scope (india / global), tags every story of this feed gets, keep only tagged stories?
FEEDS = [
    # ---- India: business press
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "india", ["top"], False),
    ("Economic Times", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "india", [], False),
    ("Economic Times", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms", "india", ["economy"], False),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml", "india", ["top"], False),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml", "india", [], False),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/results.xml", "india", ["results"], False),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/economy.xml", "india", ["economy"], False),
    ("Mint", "https://www.livemint.com/rss/markets", "india", [], False),
    ("Mint", "https://www.livemint.com/rss/economy", "india", ["economy"], False),
    ("Mint", "https://www.livemint.com/rss/companies", "india", [], False),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss", "india", [], False),
    ("Business Standard", "https://www.business-standard.com/rss/economy-102.rss", "india", ["economy"], False),
    ("Business Standard", "https://www.business-standard.com/rss/companies-101.rss", "india", [], False),
    ("BusinessLine", "https://www.thehindubusinessline.com/markets/feeder/default.rss", "india", [], False),
    ("BusinessLine", "https://www.thehindubusinessline.com/economy/feeder/default.rss", "india", ["economy"], False),
    # ---- India: official sources
    ("RBI", "https://www.rbi.org.in/pressreleases_rss.xml", "india", ["economy"], False),
    ("SEBI", "https://www.sebi.gov.in/sebirss.xml", "india", [], False),
    ("NSE announcements", "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml", "india", [], True),
    ("BSE announcements", "https://www.bseindia.com/data/xml/announcements.xml", "india", [], True),
    # ---- Global
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "global", ["top"], False),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "global", [], False),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", "global", ["economy"], False),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "global", ["top"], False),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_marketpulse", "global", [], False),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", "global", [], False),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml", "global", ["top"], False),
    ("The Guardian", "https://www.theguardian.com/uk/business/rss", "global", [], False),
    # ---- Searches (results and management commentary): the story link opens the publisher's own page
    ("Google News", "https://news.google.com/rss/search?q=%28%22quarterly+results%22+OR+%22Q2+results%22+OR+%22Q3+results%22+OR+%22Q1+results%22+OR+%22net+profit%22%29+India+stock+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "india", ["results"], False),
    ("Google News", "https://news.google.com/rss/search?q=%28%22earnings+call%22+OR+concall+OR+%22management+commentary%22+OR+%22management+said%22+OR+guidance%29+India+company+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "india", ["mgmt"], False),
    ("Google News", "https://news.google.com/rss/search?q=%28%22earnings+results%22+OR+%22reports+quarterly%22+OR+%22earnings+beat%22+OR+%22earnings+miss%22%29+stock+when:1d&hl=en-US&gl=US&ceid=US:en", "global", ["results"], False),
    ("Google News", "https://news.google.com/rss/search?q=%28%22earnings+call%22+OR+%22management+said%22+OR+%22CEO+said%22+OR+guidance%29+company+when:1d&hl=en-US&gl=US&ceid=US:en", "global", ["mgmt"], False),
]

SECTORS = {
    "Banking & Finance": r"\b(bank|banks|banking|nbfc|lender|lenders|loan|loans|credit growth|insurer|insurance|mutual fund|hdfc|icici|sbi|axis bank|kotak|bajaj finance|jpmorgan|goldman|morgan stanley|wells fargo)\b",
    "IT & Technology": r"\b(it services|it sector|infosys|tcs|wipro|hcl ?tech|tech mahindra|software|saas|semiconductors?|chipmakers?|nvidia|microsoft|apple|alphabet|google|meta|amazon|ai\b|artificial intelligence)",
    "Pharma & Healthcare": r"\b(pharma|pharmaceutical|drugmaker|usfda|fda|hospital|healthcare|biotech|vaccine|sun pharma|cipla|dr reddy)",
    "Auto": r"\b(auto|automobile|automaker|carmaker|electric vehicles?|ev|maruti|tata motors|mahindra|bajaj auto|hero motocorp|two-wheeler|tesla|tyre)\b",
    "FMCG & Consumer": r"\b(fmcg|consumer|hindustan unilever|hul|nestle|itc|dabur|retail|retailer|apparel|walmart|costco)\b",
    "Energy & Oil": r"\b(oil|gas|crude|brent|refiner|refining|reliance|ongc|ioc|opec|renewable|solar|wind power|power sector|ntpc|energy)\b",
    "Metals & Mining": r"\b(steel|metal|metals|aluminium|aluminum|copper|zinc|iron ore|tata steel|jsw|hindalco|vedanta|mining|coal india|gold|silver)\b",
    "Real Estate & Infra": r"\b(real estate|realty|housing|dlf|infrastructure|infra|cement|construction|larsen|l&t|railways?|roads?|capex)\b",
    "Telecom & Media": r"\b(telecom|airtel|jio|vodafone idea|5g|spectrum|broadband|media|streaming)\b",
    "Defence & Industrials": r"\b(defence|defense|hal|bel|aerospace|shipbuilding|capital goods|engineering|industrial|manufacturing|boeing)\b",
}
COUNTRIES = {
    "India": r"\b(india|indian|nifty|sensex|nse|bse|rbi|sebi|rupee|mumbai|dalal street)\b",
    "United States": r"\b(united states|america|american|wall street|federal reserve|s&p 500|nasdaq|dow jones|treasur(y|ies)|white house|washington)\b",
    "China": r"\b(china|chinese|beijing|yuan|pboc|hong kong|shanghai|shenzhen)\b",
    "Japan": r"\b(japan|japanese|tokyo|boj|yen|nikkei)\b",
    "Europe": r"\b(europe|european|euro zone|eurozone|euro|ecb|germany|german|france|french|italy|dax|eu)\b",
    "United Kingdom": r"\b(uk|britain|british|bank of england|boe|london|ftse|sterling|pound)\b",
    "Middle East": r"\b(saudi|uae|iran|israel|gulf|opec|qatar|middle east|gaza)\b",
    "Russia & Ukraine": r"\b(russia|russian|ukraine|ukrainian|kremlin|moscow)\b",
}
RE_RESULTS = re.compile(r"(\bq[1-4]\s*(fy)?\s*\d{0,4}\b.*(result|earning|profit|revenue)|quarterly results?|financial results?|\bnet profit\b|\bearnings (beat|miss|report|results?|per share|season)\b|\breports? (q[1-4]|quarter|earnings|profit)|\bprofit (rises|jumps|falls|declines|surges|slumps|drops|up|down)\b|\brevenue (rises|jumps|falls|growth|up|down)\b|\bebitda\b|\bresults? (today|on|date)\b|\bboard meeting\b.*result|\bQ[1-4] ?FY)", re.I)
RE_MGMT = re.compile(r"(management (commentary|said|says|expects|sees|guidance)|\bearnings call\b|\bcon-?call\b|\bconference call\b|\bguidance\b|\boutlook (for|on|remains)|\b(ceo|cfo|md|chairman|managing director|chief executive)( & ceo)?\b.*\b(said|says|expects|sees|told)\b|\banalyst call\b|\binvestor (call|meet)\b|\btranscript\b|\border book\b|\bdemand outlook\b)", re.I)
RE_ECON = re.compile(r"(\brbi\b|reserve bank|\brepo rate\b|\binflation\b|\bcpi\b|\bwpi\b|\bgdp\b|fiscal deficit|trade deficit|current account|\biip\b|\bpmi\b|gst collection|jobs? (data|report)|unemployment|federal reserve|\bfed\b|\becb\b|bank of (england|japan)|rate (cut|hike)|monetary policy|treasury yield|bond yield|\bbudget\b|\btariffs?\b|\bimf\b|world bank|crude oil|\bbrent\b|oil prices?|\beconomy\b|\beconomic\b)", re.I)
RE_INDIA = re.compile(r"\b(india|indian|nifty|sensex|nse|bse|rbi|sebi|rupee|dalal street|mumbai)\b", re.I)
FINANCE_ONLY = re.compile(r"(stock|share|market|nifty|sensex|invest|econom|bank|profit|revenue|earnings|results|ipo|bond|yield|currency|rupee|dollar|oil|gold|inflation|rate|fund|trade|tariff|fed\b|rbi|sebi|finance|business|company|sales|deal|merger|acquisition)", re.I)


def clean_text(x, limit=None):
    x = html.unescape(re.sub(r"<[^>]+>", " ", x or ""))
    x = re.sub(r"\s+", " ", x).strip()
    return x[:limit] if limit else x


def classify(title, summary, scope, base_tags):
    text = title + " " + summary
    tags = set(base_tags)
    if RE_RESULTS.search(title) or RE_RESULTS.search(summary[:200]):
        tags.add("results")
    if RE_MGMT.search(title) or RE_MGMT.search(summary[:200]):
        tags.add("mgmt")
    if RE_ECON.search(title):
        tags.add("economy")
    sectors = [n for n, rx in SECTORS.items() if re.search(rx, title, re.I)]
    ctext = title + " " + summary[:160]
    countries = [n for n, rx in COUNTRIES.items() if re.search(rx, ctext, re.I)]
    if "United States" not in countries and re.search(r"\bU\.?S\.?A?\b|\bFed\b", ctext):   # capital letters only, so the word "us" does not count
        countries.append("United States")
    if sectors:
        tags.add("sector")
    if countries:
        tags.add("country")
    if scope == "global" and RE_INDIA.search(title):
        scope = "india"
    return sorted(tags), sectors, countries, scope


def norm_key(link, title):
    link = re.sub(r"[?#].*$", "", (link or "").strip().lower())
    t = re.sub(r"[^a-z0-9]+", "", (title or "").lower())[:90]
    return link or t, t


def entry_time(e):
    for k in ("published_parsed", "updated_parsed"):
        v = e.get(k)
        if v:
            try:
                return datetime.fromtimestamp(timegm(v), tz=timezone.utc)
            except Exception:
                pass
    return None


def fetch_feed(feed):
    name, url, scope, base_tags, keep_tagged = feed
    status = {"name": name, "url": url, "ok": False, "items": 0}
    items = []
    try:
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=20).read()
        parsed = feedparser.parse(raw)
        now = datetime.now(timezone.utc)
        for e in parsed.entries:
            title = clean_text(e.get("title"), 300)
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue
            when = entry_time(e) or now
            if when > now + timedelta(hours=2):
                when = now
            summary = clean_text(e.get("summary") or e.get("description"), 400)
            src = name
            if name == "Google News":
                s = e.get("source")
                src = (s.get("title") if isinstance(s, dict) else None) or "Google News"
                title = re.sub(r"\s+-\s+" + re.escape(src) + r"$", "", title)
            tags, sectors, countries, sc = classify(title, summary, scope, base_tags)
            if keep_tagged and not ({"results", "mgmt"} & set(tags)):
                continue
            if not FINANCE_ONLY.search(title + " " + summary[:120]) and name in ("BBC Business", "The Guardian", "Yahoo Finance"):
                continue
            items.append({"t": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "title": title, "link": link, "src": src, "scope": sc,
                          "tags": tags, "sectors": sectors, "countries": countries, "sum": summary[:240]})
        status["ok"] = True
        status["items"] = len(items)
    except Exception as ex:
        status["error"] = type(ex).__name__ + ": " + str(ex)[:120]
    return items, status


def build_news(prev_items=None, feeds=FEEDS, fetch=fetch_feed):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_H)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch, feeds))
    merged, seen_link, seen_title = [], set(), set()
    pool = [it for items, _ in results for it in items] + list(prev_items or [])
    pool.sort(key=lambda it: it["t"], reverse=True)
    for it in pool:
        if datetime.strptime(it["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) < cutoff:
            continue
        lk, tk = norm_key(it["link"], it["title"])
        if lk in seen_link or tk in seen_title:
            continue
        seen_link.add(lk); seen_title.add(tk)
        merged.append(it)
    merged = merged[:MAX_ITEMS]
    ist = now + timedelta(hours=5, minutes=30)
    return {"updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "updated_ist": ist.strftime("%d %b %Y, %H:%M IST"),
            "window_hours": WINDOW_H, "count": len(merged),
            "sources": [s for _, s in results], "items": merged}


# ============================================================== main
def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prev_desk = load_json(PREV_DESK, {})
    desk = {}
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    desk["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    desk["updated_ist"] = ist.strftime("%d %b %Y, %H:%M IST")

    try:
        desk["quotes"] = fetch_quotes()
        if all(q.get("error") for q in desk["quotes"]):
            raise RuntimeError("no quote came back")
    except Exception as e:
        print("quotes skipped:", e); desk["quotes"] = prev_desk.get("quotes", [])
    for key, label, loader in (("nifty500", "Nifty 500", nifty500_members), ("sp500", "S&P 500", sp500_members)):
        try:
            desk[key] = movers(loader(), label)
            print(f"{label}: {desk[key]['count']} stocks | adv {desk[key]['adv']} dec {desk[key]['dec']} | top gainer {desk[key]['gainers'][0]['s']} {desk[key]['gainers'][0]['c']}%")
        except Exception as e:
            print(f"{label} movers skipped:", e)
            if key in prev_desk:
                desk[key] = prev_desk[key]
    with open(os.path.join(OUT_DIR, "desk.json"), "w") as f:
        json.dump(desk, f, separators=(",", ":"))

    if feedparser is None:
        raise SystemExit("feedparser is not installed - run: pip install feedparser")
    prev = load_json(PREV_NEWS, {})
    news = build_news(prev.get("items", []))
    with open(os.path.join(OUT_DIR, "news.json"), "w") as f:
        json.dump(news, f, separators=(",", ":"))
    ok = sum(1 for s in news["sources"] if s["ok"])
    print(f"News: {news['count']} stories (last {WINDOW_H}h) | feeds OK {ok}/{len(news['sources'])}")
    for s in news["sources"]:
        print(("  ok   " if s["ok"] else "  FAIL ") + f"{s['name']:20s} {s['items']:4d}  {s.get('error','')}  {s['url'][:70]}")


if __name__ == "__main__":
    main()
