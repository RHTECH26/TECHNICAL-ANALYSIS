#!/usr/bin/env python3
"""
Rules-based "breaking out today" scanner over a ~1,026-ticker global single-stock universe
(STOCK_TICKERS below -- mapped from the user-supplied global stock list to working Yahoo
Finance tickers). Mirrors fetch_breakout_scanner.py's scoring logic and resilient-fetch
design, but scores individual stocks instead of ETFs, feeding the site's Stock Screener tab
(a duplicate of the ETF Screener tab, pointed at this data file instead).

The default flag (also the one used server-side for the "asof"-day snapshot) is:
    today's close is a new 20-day high  AND  today's volume >= 1.5x its trailing 20-day
    average volume

...but the lookback window (20/50/100/200 days) and the volume multiple are both adjustable
on the site itself, live, with no extra fetch -- each row also carries its own trimmed
close/volume history (last ~210 trading days, enough to cover a 200-day lookback with a
20-day volume-average tail), and the page recomputes the flag in JavaScript whenever the
person changes either control. The 52-week-high, RSI(14), and 50-day/200-day-SMA trend
columns are NOT adjustable (fixed-window indicators, shown for context) and stay
server-computed.

Writes data/stock-screener.json:
    {
      "asof": "YYYY-MM-DD",
      "rows": [
        {"t": "AAPL", "price": 512.34, "chg1d": 0.42, "high20": true, "high52w": false,
         "volRatio": 1.87, "rsi14": 68.2, "aboveSma50": true, "aboveSma200": true,
         "breakout": true, "closes": [...], "vols": [...]},
        ...
      ]
    }
"closes"/"vols" are the trailing ~210 daily values (oldest to newest, same length, last
entry = today) used to recompute high20/high50/high100/high200 and the volume ratio
client-side for any lookback the person picks. Tickers with too little history to compute a
given metric report null for it rather than being dropped from the universe entirely.

Run locally with:  python3 scripts/fetch_stock_screener_data.py
Runs automatically via .github/workflows/update-data.yml (as its own parallel job, alongside
fetch_breakout_scanner.py). Needs curl_cffi installed (pip install curl_cffi) -- see the note
above OUT_PATH for why.
"""
import json
import os
import random
import time

import yfinance as yf
import pandas as pd

STOCK_TICKERS = [
    "SEPL.L", "3653.TW", "2395.TW", "8334.T", "8697.T", "688082.SS", "300285.SZ", "DSG.TO", 
    "AMG", "QLYS", "MSFT", "FTNT", "2404.TW", "3293.TW", "007660.KQ", "005930.KS", "8421.T", 
    "9962.T", "0293.HK", "STVN", "DT", "WST", "NOW", "KEYS", "GATX", "PLTR", "FAST", "ZBRA", 
    "LGND", "ITT", "2360.TW", "IAG.TO", "XP", "NTAP", "ITH.L", "ADPORTS.AD", "6531.TWO", 
    "SCMN.SW", "4030.SR", "IPCALAB.NS", "1888.HK", "MTX.DE", "PETR3.SA", "WAB", "AME", "GRMN", 
    "VEEV", "BGEO.L", "DPLM.L", "2890.TW", "6805.TWO", "FPH.NZ", "4503.T", "ENEV3.SA", 
    "LOTB.BR", "TEL", "ACMR", "ROST", "2368.TW", "2914.T", "LFUS", "HBR.L", "6274.TW", 
    "3017.TW", "6525.T", "4062.T", "BMPS.MI", "MCX.NS", "301297.SZ", "600901.SS", "APH", 
    "2449.TW", "3443.TW", "443060.KS", "BS6.SI", "VAR.OL", "SONACOMS.NS", "ADANIPORTS.NS", 
    "MANKIND.NS", "ZYDUSLIFE.NS", "1093.HK", "688200.SS", "688432.SS", "LTM", "SU", "WAT", 
    "SMTC", "ETN", "KDP", "NVT", "MODON.AD", "EMAAR.DU", "4612.T", "OBEROIRLTY.NS", "BPAC3.SA", 
    "GLXY", "SHOP", "LRCX", "EBAY", "PKG", "DIVISLAB.NS", "601872.SS", "SVT.L", "SDR.L", "UU.L", 
    "RSW.L", "PRESIGHT.AD", "3665.TW", "5274.TWO", "3044.TW", "SALM.OL", "3659.T", "4684.T", 
    "8341.T", "LAURUSLABS.NS", "JINDALSTEL.NS", "1548.HK", "2269.HK", "002916.SZ", "001389.SZ", 
    "300487.SZ", "300548.SZ", "300661.SZ", "300857.SZ", "603268.SS", "301626.SZ", "MFC", 
    "DPM.TO", "B3SA3.SA", "KBCA.BR", "LITE", "SEI", "NVDA", "ANET", "SIMO", "UHS", "STX", 
    "HOOD", "2POINTZERO.AD", "ALDAR.AD", "2308.TW", "MOTILALOFS.NS", "CMBT.BR", "RAKBANK.AD", 
    "8359.T", "6139.TW", "316140.KS", "1177.HK", "ADIB.AD", "2330.TW", "5347.TW", "VTLN.SW", 
    "PE&OLES.MX", "9697.T", "6857.T", "6981.T", "6501.T", "JSWINFRA.NS", "APOLLOHOSP.NS", 
    "ENX.PA", "ORNAV.HE", "GMAB", "603893.SS", "603087.SS", "SFR.AX", "YPFD.BA", "ALAB", "SNDK", 
    "META", "TER", "NTNX", "MEDP", "AAPL", "300442.SZ", "BBDC3.SA", "IBKR", "FCIT.L", "6239.TW", 
    "ACLN.SW", "000660.KS", "9766.T", "285A.T", "TPRO.MI", "INDHOTEL.NS", "TGTX", 
    "NTPCGREEN.NS", "1801.HK", "300408.SZ", "002138.SZ", "300759.SZ", "EDU", "PCT.L", "SMT.L", 
    "3529.TWO", "PHOENIXLTD.NS", "2268.HK", "688072.SS", "603259.SS", "688025.SS", "FM.TO", 
    "PRU.AX", "HALO", "MU", "DOCN", "ARM", "ABBV", "SYDB.CO", "ALPHADHABI.AD", "UTDPLT.KL", 
    "6723.T", "6504.T", "6856.T", "RENT3.SA", "CR", "LAUR", "ROK", "SNPS", "RR.L", "EAND.AD", 
    "8299.TWO", "GMEXICOB.MX", "603156.SS", "001965.SZ", "ADI", "MPWR", "TXN", "EMAARDEV.DU", 
    "021240.KS", "AZM.MI", "DCII.JK", "MAXHEALTH.NS", "AEGISLOG.NS", "SAF.PA", "600026.SS", 
    "002463.SZ", "002821.SZ", "300811.SZ", "688313.SS", "300394.SZ", "ICE", "EGP", "WDAY", 
    "LPP.WA", "8377.T", "WELCORP.NS", "300684.SZ", "CDA.AX", "AMD", "BAP", "MRVL", "2327.TW", 
    "RED.MC", "0270.HK", "WTS", "688008.SS", "603256.SS", "SCCO", "RADICO.NS", "TORNTPHARM.NS", 
    "ASTERDM.NS", "ESI", "NPO", "MOG-A", "YKBNK.IS", "4568.T", "300502.SZ", "RMBS", "0144.HK", 
    "IGRD.QA", "AAON", "2301.TW", "PSPN.SW", "6532.T", "4507.T", "1828.HK", "300037.SZ", 
    "003031.SZ", "ENELAM.SN", "GMIN.TO", "PTC", "MMSI", "TSEM", "VIST", "EB5.SI", "INPST.AS", 
    "MFRISCOA-1.MX", "8473.T", "OIL.NS", "CRM", "2881.TW", "2884.TW", "2887.TW", "2885.TW", 
    "AUTO.OL", "HARL.TA", "600601.SS", "688498.SS", "ELD.TO", "DSV.AX", "CPLE3.SA", "6273.T", 
    "ICICIGI.NS", "CDNS", "033780.KS", "298040.KS", "WCP.TO", "KNT.TO", "SSRM", "COHR", "MTSI", 
    "VAKBN.IS", "C09.SI", "3064.T", "COALINDIA.NS", "688630.SS", "KLAC", "ADNOCLS.AD", "CPAY", 
    "016360.KS", "138040.KS", "MARICO.NS", "BHARATFORG.NS", "0267.HK", "DVN", "LNG", "VRTX", 
    "ALLE", "GLENMARK.NS", "BTO.TO", "HBM.TO", "EDV.L", "IHC.AD", "INVE-A.ST", "002945.SZ", 
    "002128.SZ", "601689.SS", "BBD-A.TO", "MOD", "CAT", "FIX", "HUBB", "HINDZINC.NS", "9698.HK", 
    "RGLD", "GLW", "9602.T", "600801.SS", "ONON", "DXCM", "3533.TW", "SECT-B.ST", "5713.T", 
    "EXLS", "MSI", "APPF", "AKBNK.IS", "NXSN.TA", "MMHD.TA", "AMMN.JK", "688195.SS", "IMG.TO", 
    "OR.TO", "FNV", "ULTA", "3081.TWO", "6954.T", "STRL", "AEIS", "ANTO.L", "AP.PS", 
    "VISTAA.MX", "KFH.KW", "MGOR.TA", "TECK-A.TO", "ISRG", "MA", "EOG", "GALD.SW", "G07.SI", 
    "CRDO", "6415.TW", "1258.HK", "688041.SS", "ARGX", "TRNO", "HALKB.IS", "PHOE.TA", 
    "GLAND.NS", "BOSCHLTD.NS", "688002.SS", "INCY", "INSW", "XOM", "FRO", "AUGO.TO", "DELTA.BK", 
    "OFSS.NS", "LTH", "SM", "PR", "RBC", "MCHP", "SN", "2383.TW", "BN4.SI", "NN.AS", "8136.T", 
    "COFORGE.NS", "2099.HK", "METSO.HE", "LLY", "DUOL", "VIAV", "SEIC", "8303.T", "DNET.JK", 
    "VTR", "ZAIN.KW", "5838.T", "5101.T", "BPE.MI", "TNE.AX", "7167.T", "DANSKE.CO", 
    "300395.SZ", "300628.SZ", "AEM", "WPM", "CMM.AX", "GMD.AX", "EVN.AX", "Q", "NEM", "4186.T", 
    "VG", "SPOT", "4250.SR", "8308.T", "8354.T", "ALKEM.NS", "LODHA.NS", "3939.HK", "300666.SZ", 
    "DOL.TO", "TFPM", "WHD", "6446.TWO", "278470.KQ", "4325.SR", "PINFRA.MX", "8306.T", 
    "8630.T", "SMMA.JK", "3692.HK", "688256.SS", "UCB.BR", "CHRD", "MCO", "MKSI", "JHX", 
    "2059.TW", "0700.HK", "300450.SZ", "KMI", "CCK", "ENKAI.IS", "OUT.JO", "5831.T", 
    "MAHABANK.NS", "688300.SS", "688617.SS", "ARTG.V", "CNQ", "SANB3.SA", "PRIO3.SA", "NST.AX", 
    "KNSA", "CDE", "2345.TW", "GFI", "600885.SS", "SQM-A.SN", "HEI", "OKE", "ESE", "SON", 
    "NXT.L", "BEAN.SW", "8331.T", "SCHAEFFLER.NS", "LLOYDSME.NS", "300308.SZ", "AGS.BR", "VCTR", 
    "BCPC", "NVMI", "VBL.NS", "PERSISTENT.NS", "CCH.L", "TECHM.NS", "ABCAPITAL.NS", "ALV.DE", 
    "603986.SS", "V", "GWRE", "AMR.AD", "5830.T", "6861.T", "ABX.TO", "EXE", "FERTIGLB.AD", 
    "9983.T", "1698.HK", "WMS", "UMBF", "IDXX", "ZWS", "UI", "8601.T", "GRASIM.NS", "600183.SS", 
    "600663.SS", "PCTY", "AU", "LB", "KRYS", "EQX", "HLMA.L", "4519.T", "7936.T", "CARL-A.CO", 
    "688017.SS", "AFL", "ROP", "IESC", "GE", "HAR.JO", "4300.SR", "WPRTS.KL", "NHPC.NS", 
    "300346.SZ", "600176.SS", "688578.SS", "ET", "IVZ", "CF", "XRO.AX", "MGY", "AGX", 
    "NESTLE.KL", "6383.T", "5334.T", "6146.T", "7741.T", "PST.MI", "GMRAIRPORT.NS", "301611.SZ", 
    "FANG", "BSY", "CS.TO", "ALNY", "1508.HK", "AIRARABIA.DU", "BESI.AS", "ULTRACEMCO.NS", 
    "HEROMOTOCO.NS", "0083.HK", "MSCI", "MLI", "WWD", "STB.OL", "7182.T", "8316.T", 
    "TVSMOTOR.NS", "SOLARINDS.NS", "688183.SS", "600988.SS", "688120.SS", "601168.SS", "COKE", 
    "PEN", "HL", "BMRN", "MUTHOOTFIN.NS", "2020.HK", "ALWN.SW", "ALK-B.CO", "601100.SS", 
    "688271.SS", "ENB", "ALQ.AX", "LYC.AX", "NXT.V", "AR", "VRT", "BZ", "FIS", "TDG", "DTM", 
    "HAS", "KGH.WA", "4543.T", "7747.T", "NAM-INDIA.NS", "600298.SS", "300604.SZ", "LUN.TO", 
    "K", "OGC.TO", "SSB", "CELH", "IRM", "002558.SZ", "SUNPHARMA.NS", "0012.HK", "301377.SZ", 
    "SAUD3.SA", "HWM", "7186.T", "JSWSTEEL.NS", "NAUKRI.NS", "CGPOWER.NS", "PIRAMALFIN.NS", 
    "BAJAJ-AUTO.NS", "TPEIR.AT", "DWS.DE", "300620.SZ", "603979.SS", "IGM.TO", "AG", "TIGO", 
    "FCFS", "TEMN.SW", "ASURB.MX", "TRN.MI", "IOB.NS", "LUPIN.NS", "IPN.PA", "CSU.TO", "DFY.TO", 
    "TOI.V", "ELI.BR", "EQT", "EPRT", "OMC", "BIRK", "MELI", "CBOE", "SPG", "RRC", "VNOM", 
    "YOU", "MISC.KL", "5016.T", "5803.T", "688012.SS", "BSX", "LMP.L", "MRL.MC", "BAJAJHFL.NS", 
    "1024.HK", "600415.SS", "603799.SS", "605499.SS", "002850.SZ", "CW", "OWL", "AS", "RIO", 
    "MNG.L", "ASTOR.IS", "6223.TW", "BOL.ST", "SOBI.ST", "CPI.JO", "ICT.PS", "SUNWAY.KL", 
    "8802.T", "SRF.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "300373.SZ", "605376.SS", "600406.SS", 
    "688019.SS", "002371.SZ", "603629.SS", "EW", "TKO", "YUM", "FN", "RTX", "BGN.MI", "PRU.L", 
    "FRES.L", "QNBTR.IS", "267250.KS", "TENAGA.KL", "2413.T", "PANI.JK", "BBRI.JK", "MORA.JK", 
    "9633.HK", "1913.HK", "601018.SS", "002602.SZ", "GMG.AX", "CART", "SBRA", "ORA", "ADBE", 
    "PDD", "APP", "JEF", "8591.T", "SCHW", "CTPNV.AS", "6181.HK", "SAN.PA", "PSSA3.SA", "FTAI", 
    "PRMB", "TW", "VIK", "LVS", "UBER", "RL", "PODD", "7769.T", "AZA.ST", "VOD.JO", "S68.SI", 
    "POLYCAB.NS", "POWERINDIA.NS", "GROWW.NS", "NEM.DE", "002409.SZ", "002484.SZ", "301200.SZ", 
    "688668.SS", "ALKS", "ASB", "WES", "GLNG", "CGNX", "AAF.L", "6515.TW", "071050.KS", 
    "196170.KQ", "SBK.JO", "QIBK.QA", "5706.T", "MAZDOCK.NS", "PRESTIGE.NS", "M&MFIN.NS", 
    "ATGL.NS", "HCLTECH.NS", "1299.HK", "1209.HK", "1530.HK", "0388.HK", "EVD.DE", "600066.SS", 
    "688099.SS", "002049.SZ", "300476.SZ", "002273.SZ", "688213.SS", "601058.SS", "MIN.AX", 
    "PME.AX", "AVGO", "AROC", "NFLX", "NXST", "CEG", "ADC", "FAF", "TRU", "AFRM", "GOOG", "RHP", 
    "1590.HK", "259960.KS", "035420.KS", "CVC.AS", "CUMMINSIND.NS", "TATACONSUM.NS", "BSE.NS", 
    "6862.HK", "9992.HK", "9961.HK", "600236.SS", "002353.SZ", "002379.SZ", "000933.SZ", 
    "603019.SS", "U-U.TO", "SOFI", "FSLR", "SGI", "LOAR", "AJG", "TPR", "RKT", "PNFP", "PIPR", 
    "ADSK", "ARES", "RDDT", "EIX", "APO", "TPG", "ALSN", "CTRE", "BLK", "SCT.L", "VAL.JO", 
    "7974.T", "4732.T", "BAJAJHLDNG.NS", "SUNDARMFIN.NS", "GVT&D.NS", "1113.HK", "2259.HK", 
    "LUG.TO", "PAAS", "AMZN", "CQP", "MNST", "VIRT", "FITB", "ONB", "CSCO", "JAZZ", "ZURN.SW", 
    "QGTS.QA", "BREN.JK", "HINDCOPPER.NS", "0966.HK", "2057.HK", "1208.HK", "603929.SS", 
    "601628.SS", "000783.SZ", "MALLPLAZA.SN", "AGI", "DLR", "CEF", "329180.KS", "ADYEN.AS", 
    "3231.T", "3003.T", "BEL.NS", "TATACAP.NS", "LINDEINDIA.NS", "JIOFIN.NS", "0016.HK", 
    "PAH3.DE", "300059.SZ", "002294.SZ", "002709.SZ", "600895.SS", "688188.SS", "688336.SS", 
    "000807.SZ", "002414.SZ", "002595.SZ", "300124.SZ", "600995.SS", "601211.SS", "601995.SS", 
    "600989.SS", "600930.SS", "ATZ.TO", "GIL", "EMA.TO", "AED.BR", "WTC.AX", "BAM", "AUB", "BX", 
    "KKR", "EVR", "VSEC", "FSS", "CPA", "BWXT", "PGR", "BIP", "GBCI", "ASELS.IS", "MS", "GEN", 
    "ABNB", "DSTKF.IS", "GARAN.IS", "4013.SR", "SHRIRAMFIN.NS", "AUBANK.NS", "NMDC.NS", 
    "600031.SS", "PAYX", "FWONA", "ORCL", "ARCC", "BLND.L", "064350.KS", "267260.KS", 
    "010140.KS", "009540.KS", "003230.KS", "SSW.JO", "1211.SR", "MLSR.TA", "NVPT.TA", 
    "CHOLAFIN.NS", "JSWENERGY.NS", "LTM.AX", "FORTIS.NS", "ASHOKLEY.NS", "1818.HK", "G24.DE", 
    "603659.SS", "605117.SS", "002517.SZ", "600292.SS", "000999.SZ", "601456.SS", "000426.SZ", 
    "000408.SZ", "300750.SZ", "688187.SS", "300033.SZ", "601336.SS", "603993.SS", "688111.SS", 
    "688629.SS", "000776.SZ", "002837.SZ", "600489.SS", "EGIE3.SA", "PLS.AX", "GGAL", "ETR", 
    "CINF", "BRO", "WMG", "SGHC", "NI", "AYI", "SPXC", "EFX", "GMED", "NU", "NFG", "RLI", "SF", 
    "O", "FUTU", "NYT", "STWD", "LPLA", "KEY", "BAJFINANCE.NS", "PIDILITIND.NS", "EICHERMOT.NS", 
    "OHI", "068270.KS", "012450.KS", "042660.KS", "NPH.JO", "IMP.JO", "MTN.JO", "1120.SR", 
    "7203.SR", "BPI.PS", "ENLT", "WAAREEENER.NS", "NESTLEIND.NS", "HDBFS.NS", "LTF.NS", 
    "MRF.NS", "TRENT.NS", "HDFCAMC.NS", "BHARTIARTL.NS", "1179.HK", "000792.SZ", "000893.SZ", 
    "002625.SZ", "002028.SZ", "601136.SS", "SGP.AX", "RYAN", "FNF", "PHYS", "TTD", "TPL", "D", 
    "AGNC", "NEE", "NLY", "ORI", "EXPE", "WAL", "COLB", "NBIX", "INTU", "LKNCY", "DTE", "KNSL", 
    "HBAN", "RITM", "AZRG.TA", "RELIANCE.NS", "SUZLON.NS", "M&M.NS", "603699.SS", "002466.SZ", 
    "601899.SS", "BKNG", "FICO", "FE", "BALL", "PEG", "BAJAJFINSV.NS", 
]
TICKERS = STOCK_TICKERS

OUT_PATH = "data/stock-screener.json"
LOOKBACK = "1y"   # ~252 trading days -- enough for the 52-week-high check and both SMAs.
# --- Why this fetch is structured the way it is ---
# The very first version of this script (single yf.download() call per batch of 30
# tickers) got the FIRST batch through fine and then came back completely empty for every
# batch after that, every run, no matter how much the inter-batch sleep/backoff was
# increased. That pattern -- works once, then hard-blocked regardless of pacing -- points to
# Yahoo Finance's bot detection fingerprinting the plain requests/urllib3 TLS handshake that
# yfinance uses by default (well documented as an issue specifically on cloud/datacenter IPs
# like GitHub Actions runners), not simple request-rate throttling. Slowing down a blocked
# client doesn't unblock it.
#
# The fix is to route requests through curl_cffi, which impersonates a real Chrome TLS
# fingerprint so Yahoo's bot detection doesn't flag the traffic in the first place. Combined
# with that: per-ticker fetches (not multi-ticker batch downloads, so one bad/delisted ticker
# can't take down a whole batch's worth of others), a short pause between tickers, and --
# still kept as a second line of defense -- exponential backoff plus merging into the
# previously-committed data/stock-screener.json instead of overwriting it, so a run that still hits
# trouble partway through keeps everything a prior successful run already fetched (same
# pattern as fetch_global_etf_data.py's load_existing()/merge_series()).
RETRIES = 4
SLEEP_BETWEEN_TICKERS = 1.2
SLEEP_ON_RETRY = 6.0             # base for exponential backoff: 6 * 2**attempt + jitter
SAVE_EVERY = 25                  # write progress to disk every N tickers processed

try:
    from curl_cffi import requests as cffi_requests
    _SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    print("WARNING: curl_cffi not installed -- falling back to yfinance's default session, "
          "which is the thing that was getting blocked. Add curl_cffi to the workflow's "
          "'pip install' step.")
    _SESSION = None

HIGH20_WINDOW = 20
HIGH52W_WINDOW = 252
VOL_AVG_WINDOW = 20
RSI_WINDOW = 14
SMA_SHORT = 50
SMA_LONG = 200
VOL_RATIO_BREAKOUT_MIN = 1.5
# Longest lookback the site's controls offer is 200 days; a 200-day volume average also
# needs 200 prior days, so 210 gives a small comfortable margin without shipping the full
# ~252-day fetch window (which would ~20% inflate the JSON for no benefit -- the fixed-window
# 52-week-high/SMA200 columns are computed here, server-side, from the full fetch instead).
CLIENT_HISTORY_WINDOW = 210


def compute_rsi(closes, window=RSI_WINDOW):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def score_ticker(t, df):
    df = df.dropna(subset=["Close"])
    if len(df) < HIGH20_WINDOW + 1:
        return None
    closes = df["Close"]
    vols = df["Volume"].fillna(0) if "Volume" in df else pd.Series([0] * len(df), index=df.index)

    price = round(float(closes.iloc[-1]), 4)
    prev = float(closes.iloc[-2])
    chg1d = round((price - prev) / prev * 100, 2) if prev else None

    high20 = bool(closes.iloc[-1] >= closes.iloc[-HIGH20_WINDOW:].max())
    high52w = bool(closes.iloc[-1] >= closes.iloc[-HIGH52W_WINDOW:].max()) if len(closes) >= HIGH52W_WINDOW else None

    vol_ratio = None
    if len(vols) >= VOL_AVG_WINDOW + 1:
        avg_vol = float(vols.iloc[-(VOL_AVG_WINDOW + 1):-1].mean())  # trailing 20 days, excluding today
        today_vol = float(vols.iloc[-1])
        if avg_vol > 0:
            vol_ratio = round(today_vol / avg_vol, 2)

    rsi_series = compute_rsi(closes)
    rsi14 = round(float(rsi_series.iloc[-1]), 1) if pd.notna(rsi_series.iloc[-1]) else None

    above_sma50 = None
    if len(closes) >= SMA_SHORT:
        sma50 = float(closes.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean().iloc[-1])
        above_sma50 = bool(price > sma50)
    above_sma200 = None
    if len(closes) >= SMA_LONG:
        sma200 = float(closes.rolling(SMA_LONG, min_periods=SMA_LONG).mean().iloc[-1])
        above_sma200 = bool(price > sma200)

    breakout = bool(high20 and vol_ratio is not None and vol_ratio >= VOL_RATIO_BREAKOUT_MIN)

    tail_closes = closes.iloc[-CLIENT_HISTORY_WINDOW:]
    tail_vols = vols.reindex(tail_closes.index).fillna(0)

    return {
        "t": t, "price": price, "chg1d": chg1d,
        "high20": high20, "high52w": high52w, "volRatio": vol_ratio, "rsi14": rsi14,
        "aboveSma50": above_sma50, "aboveSma200": above_sma200, "breakout": breakout,
        "closes": [round(float(c), 4) for c in tail_closes],
        "vols": [int(v) for v in tail_vols],
    }


def load_existing():
    """The previously committed data/stock-screener.json, if any -- keyed by ticker so this run
    can merge on top of it instead of starting from scratch (see the note above OUT_PATH)."""
    if not os.path.exists(OUT_PATH):
        return {}
    try:
        with open(OUT_PATH) as f:
            data = json.load(f)
        return {row["t"]: row for row in data.get("rows", []) if "t" in row}
    except Exception:
        return {}


def fetch_one(t):
    for attempt in range(RETRIES):
        try:
            tk = yf.Ticker(t, session=_SESSION)
            df = tk.history(period=LOOKBACK, interval="1d", auto_adjust=True)
            if df is not None and len(df) > 0:
                return df
            # An empty result with NO exception raised is yfinance's own "No data found,
            # symbol may be delisted" -- a definitive answer (wrong/delisted ticker), not a
            # transient network/rate-limit hiccup. Retrying that with exponential backoff
            # just burns ~90+ seconds per bad ticker for something that can never succeed,
            # which is what made this scanner crawl to a near-halt on a universe with even a
            # modest number of stale/incorrect tickers in it. Fail fast here instead --
            # only an actual exception below gets the backoff-and-retry treatment, since
            # that's the case that might plausibly be transient.
            print(f"  {t}: no data (symbol likely invalid/delisted) -- not retrying")
            return None
        except Exception as e:
            print(f"  {t}: attempt {attempt+1} failed: {e}")
            sleep_s = SLEEP_ON_RETRY * (2 ** attempt) + random.uniform(0, 2)
            time.sleep(sleep_s)
    return None


def main():
    existing = load_existing()
    rows_by_ticker = dict(existing)   # start from the last good run, not from scratch
    n = len(TICKERS)
    got_this_run = 0
    for i, t in enumerate(TICKERS, start=1):
        df = fetch_one(t)
        if df is not None:
            scored = score_ticker(t, df)
            if scored:
                rows_by_ticker[t] = scored
                got_this_run += 1
        else:
            print(f"  {t}: no data after {RETRIES} attempts -- keeping prior data for it if any")

        if i % SAVE_EVERY == 0 or i == n:
            rows = list(rows_by_ticker.values())
            flagged = sum(1 for r in rows if r["breakout"])
            print(f"{i}/{n} processed ({got_this_run} fetched this run, {len(rows_by_ticker)} unique tickers total, {flagged} flagged)")
            with open(OUT_PATH, "w") as f:
                json.dump({"asof": time.strftime("%Y-%m-%d"), "rows": rows}, f, separators=(",", ":"))

        time.sleep(SLEEP_BETWEEN_TICKERS)

    rows = list(rows_by_ticker.values())
    flagged = sum(1 for r in rows if r["breakout"])
    print(f"Wrote data/stock-screener.json: {len(rows)}/{n} stock tickers scored ({got_this_run} fetched this run), {flagged} flagged as breaking out.")


if __name__ == "__main__":
    main()
