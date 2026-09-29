#!/usr/bin/env python3
"""
Fetches daily USD close prices for the 4,244 tickers in the global ETF universe (sourced from
Global_Unique_Equity_ETFs_v4/v5.xlsx, filtered down from ~5,268 by removing malformed/
unresolvable tickers and option-income & covered-call funds) and writes
data/global-etf-history.json in the shape
the Global ETF Heatmap tab expects:

    { "<TICKER>": [["YYYY-MM-DD", price], ...], ... }

Tickers already carry their Yahoo Finance exchange suffix where needed (.L London,
.DE Frankfurt, .PA Paris, .T Tokyo, .SW Zurich, .MI Milan, .MC Madrid) — no suffix is
added here, unlike the NSE-only fetch_etf_data.py script.

Only ~14 months of history is kept (LOOKBACK) to keep the JSON file a reasonable size
across the current ticker count — enough for the tab's default trailing-52-week view plus some room
either side for a custom date range.

Fetched in small sequential (non-threaded) batches with pauses in between — Yahoo Finance
rate-limits (HTTP 429) hard on a big threaded/batched pull, and once that happens every
later batch in the run silently comes back empty. Progress is saved to disk after every
batch, so a run that gets cut off partway (timeout, rate-limit) still keeps whatever it
got instead of losing the whole thing.

Merges into the PREVIOUSLY COMMITTED data/global-etf-history.json instead of overwriting
it from scratch (same reasoning as fetch_etf_data.py's merge). This matters even more
here than for the NSE-only script: this universe spans US, London, Frankfurt, Paris,
Tokyo, Zurich, Milan and Madrid listings, all closing at different times relative to when
this workflow actually runs. A run that fires (scheduled or manual) before every one of
those markets has closed for the day will only have a fresh close for whichever exchanges
had already closed by then -- overwriting from scratch previously meant every OTHER
ticker's file entry got silently reset to whatever data this run happened to fetch for
it (which, for exchanges not yet closed, is still yesterday's close, same as before), and
the site's "1D" logic then found almost no ticker with data for both "yesterday" and
"today" since most tickers' most-recent date never actually advanced. Merging preserves
each ticker's own progress and lets the very next run (or the same day's later scheduled
run) top up whichever exchanges had already closed, without needing every exchange in the
world to close before a single run can produce a usable file.

--- Why this version routes through curl_cffi ---
Same failure mode already diagnosed and fixed in fetch_breakout_scanner.py: on a GitHub
Actions runner, the first batch comes through fine and then every batch after that comes
back silently empty, no matter how the backoff/pacing here is tuned. That's Yahoo Finance's
bot detection fingerprinting the default requests/urllib3 TLS handshake yfinance uses, not
rate-limiting -- so routing requests through curl_cffi (which impersonates a real Chrome TLS
fingerprint) fixes it without changing the batch size, pacing, or runtime of this script at
all. This is almost certainly the actual cause of the stale/missing tickers users see in the
Global ETF Heatmap: this script was never given the curl_cffi fix, so on any run where Yahoo
started blocking partway through, whichever tickers hadn't been reached yet just kept
whatever (increasingly old) data a previous successful run had left for them.

Run locally with:  python3 scripts/fetch_global_etf_data.py
Runs automatically via .github/workflows/update-data.yml (as its own parallel job).
No API key needed — yfinance reads Yahoo Finance's public endpoints. Needs curl_cffi
installed (pip install curl_cffi) -- see the note above for why.
"""
import json
import time
from datetime import datetime, timezone

import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    _SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    print("WARNING: curl_cffi not installed -- falling back to yfinance's default session, "
          "which is the thing that was getting silently blocked. Add curl_cffi to the "
          "workflow's 'pip install' step.")
    _SESSION = None

TICKERS = [
    'SPY', 'QQQ', 'XLF', 'XLE', 'SCHD', 'EEM', 'EWZ', 'IWM', 'FXI', 'XLU',
    'EWY', '1306.T', 'IGV', 'KRE', 'XLB', 'EFA', 'XLP', 'VEA', 'SMH', 'XLV',
    'XBI', 'SCHX', 'XLK', 'RSP', 'SCHF', 'VWO', 'XLY', 'SCHG', 'XLI', 'SCHB',
    'VUG', 'JPST', 'ICLN', 'IJH', 'ISF.L', 'XLC', 'VXUS', 'INDA', 'EWT', 'XLRE',
    'HDV', 'IWF', 'EWJ', 'XRT', 'VGT', '1545.T', 'DIA', 'VNQ', 'EFV', 'IJR',
    'SCHV', 'VTI', 'XOP', 'SCHE', 'IHI', 'VO', 'EWH', 'VTV', 'SPDW', 'FNDX',
    'VEU', 'SCHA', 'MCHI', 'ACWI', 'ILF', 'VT', 'CNYA.L', 'EWA', 'EMXC', 'XHB',
    'SPLV', 'SPMO', 'ITB', 'XME', 'IWD', 'USMV', 'VGK', 'ROBT', 'PAVE', 'SPSM',
    'FINA', 'SPEM', 'IWR', 'EWC', 'IDEV', 'KBWB', 'FXN', 'FNDF', 'CIBR', 'ITOT',
    'IVW', 'SPHQ', 'JPIE', 'MTUM', 'EUFN', 'KBE', 'SCZ', 'BBJP', 'SPMD', 'IQLT',
    'DGRO', 'XLG', 'SASU.L', 'EIDO', 'MGK', 'SCHM', 'FENY', 'EWU', 'VYM', 'KIE',
    'VLUE', 'FNDE', 'EWW', 'SHLD', 'SCHK', 'NOBL', 'IYZ', 'RDVY', 'CQQQ', 'DTCR',
    'EWS', 'QUAL', 'EWG', 'SEMI.L', 'IEUR', 'FNDA', 'WCMI', 'EZU', 'VYMI', 'FEZ',
    'IYE', 'FDL', 'IGF', 'IXC', 'IWP', 'IVLU', 'SPHD', 'FCG', 'PICK', 'FTXN',
    'XUSE.L', 'AIRR', 'IYH', 'KLMN', 'IDV', '1489.T', 'ESGE', 'EFG', 'REMX', 'WEBN.DE',
    'IVE', 'IYT', 'ITA', 'FTXH', 'VNM', 'MDY', 'MOAT', 'GRID', 'FBCG', 'IUSV',
    'VDE', 'IYW', 'IWN', 'ARTY', '586A.T', 'IHF', 'QNDX', 'FVD', 'IWB', 'URTH',
    'AAXJ', 'VFH', '1678.T', 'MIDD.L', 'EFAV', 'VB', 'IUSG', 'EWL', 'SPTM', 'JQUA',
    'KSA', 'AIA', 'IWO', 'ECH', 'IWS', 'FTCS', 'EPOL', 'IWX', 'KBWD', 'VPL',
    'IXN', 'CLOU', 'FUTY', 'NLR', 'EWI', 'SDIV', 'WSML.L', 'IYF', 'XMMO', 'ESGU',
    'IEZ', 'CRAK', 'VXF', 'EWP', 'FNDB', 'IWY', 'IUKD.L', 'IDMO', '1321.T', 'IMTM',
    'IGM', 'BNKS.L', 'VOX', 'SCHC', 'TOPT', 'JIVE', 'EWQ', 'FFSM', 'EWD', 'MOO',
    'SAEM.L', 'OIH', 'BBEU', 'EPP', 'VOE', 'PRF', 'JTEK', 'IOO', 'IHE', 'VPU',
    'XSMO', 'FDN', 'VNQI', 'FTEC', 'IJS', 'ROCY', 'PPH', 'ONEQ', 'ROCQ', 'MGV',
    'VBR', 'SMMD', 'IXJ', 'FXU', 'GRE.PA', 'EMLP', 'FHLC', 'IJK', 'POWR', 'VHT',
    'DIVB', 'FCOM', 'OEF', 'INTF', 'LTAM.L', 'XAR', 'EWN', 'WHCA.L', 'PXH', 'IJT',
    'EWM', 'VV', 'SUSM.L', 'EEMV', 'BBCA', 'VSS', 'IYM', 'UAE', 'BBAX', 'JGLO',
    'VBK', 'EMMUSC.SW', 'SPGM', 'DIV', 'QCLN', 'WENS.L', 'SDY', 'TRUT', 'VOT', 'PPA',
    'GPZ', 'EWZS', 'IYK', 'DFND.L', 'SLYV', 'BBUS', 'IGE', 'EPHE', 'BBIN', 'IJJ',
    'GXPC', 'EWJV', 'ESGV', 'VSGX', 'IWV', 'IEVL.L', 'AINF.L', 'FXL', 'SMLF', 'EZA',
    'XMHQ', 'COLO', 'IAT', 'EMGF', 'GXPT', 'TDIV', 'VDC', 'ARGT', 'IDHQ', 'EEMA',
    'LRGF', 'DVYE', 'QQQJ', 'PTF', 'JEMA', 'IDU', 'IAK', 'XPH', 'FTXR', 'DFEU.L',
    'FIDU', 'JMOM', 'JFLX', 'SMIN', 'QTOP', 'TUR', 'DEFS.PA', 'IDRV', 'FSTA', 'IYJ',
    'SLYG', 'PWV', 'FDT', 'ICOP', 'ZAP', 'IWVL.L', 'FSMD', 'PXF', 'VIS', 'FNCL',
    'DSI', 'FTGS', 'HEAL.L', 'FFLG', 'KXI', 'PWB', 'EDIV', 'SMDV', 'WQDV.L', '605A.T',
    'IEO', 'FAN', 'FEM', 'CNXT', 'LOCK.L', 'ENZL', 'IEV', 'IYC', 'PRFZ', 'ACWV',
    'IYG', 'FNDC', 'PIE', 'MISL', 'PHO', 'RWJ', 'ENOR', 'MDYV', 'WMVG.L', 'FFLC',
    'EQWL', 'FIVA', 'IDNA', 'MDYG', 'FTXO', 'PIZ', 'MGC', 'SPGP', 'JPEF', 'FDEM',
    'BLKC.L', 'RDIV', 'FDIS', 'XASX.DE', 'XES', 'IAI', 'EUSA', 'FYC', 'KOMP', 'FINX',
    'PXJ', 'PEJ', 'FHEQ', 'XTN', 'FDMO', 'FIDI', 'IKOR.L', 'FV', 'EWO', 'XTL',
    'WGDP.DE', 'IWC', 'EEMS', 'AFK', 'JVAL', 'IXG', 'PTH', 'TRUD', 'REGL', 'SPYX',
    'FXH', 'SUJP.L', 'SSAC.L', 'THD', 'EXI', 'GWX', 'VFMF', 'VAW', 'AUSF', 'EMVL.L',
    'IXP', 'ILDR', 'FMAT', 'XNTK', 'VCR', 'HYDR', 'EIS', 'IUS', 'AGED.L', 'BUZZ',
    'IDVY.L', 'WARP', 'IFV', 'IAPD.L', 'GXPD', 'XSW', 'FPXI', 'TRUC', 'XLSR', 'VEGI',
    'AIAA.L', 'FEMS', 'FDTX', 'FCA', 'FDLO', 'JXI', 'PRN', 'FXO', 'IWL', 'FVAL',
    'PNQI', '159A.T', 'XSHQ', 'PSCT', 'IPKW', 'FXZ', 'EPU', 'FPX', 'BBRE', 'IDX',
    '2513.T', 'EELV', 'SCJ', 'LGDS', 'RACK', 'MAXJ', 'IGRO', 'PJP', 'FRI', 'EFNL',
    'EIPX', 'IMVP', 'IFSU.L', 'FTRI', 'XSVM', 'IWMO.L', 'F50A.DE', 'HAP', 'PKB', 'XMAW.DE',
    'ORBX', 'SVAL', 'BBEM', 'FQAL', 'BNK.PA', 'IETC', 'DAX', 'MXI', 'SPOG.L', 'PSP',
    'DXSA.DE', 'XDWT.DE', 'FXR', 'EWK', 'EQAL', 'IDLV', 'LITM.L', 'FTA', 'IFFF.L', 'IBBQ',
    'XMVM', 'FYX', 'DGIT.L', 'TENJ', 'PBJ', 'JIG', 'IBAT', 'FDEV', 'XHS', 'QANT.L',
    'XHE', 'WEBA.DE', 'FTAW.L', 'XSHD', 'PSCH', 'SLX', 'PRAM.DE', 'CGW', 'EWX', 'WIRE.DE',
    'FREI', 'KCE', 'GXC', 'IVVM', 'JULV', 'SPEU', 'GLOF', 'GXPS', 'PDP', 'VEXC',
    'PXE', 'CAC.PA', 'IYY', 'UNIC.PA', 'XDWF.DE', 'ASEA', 'PGJ', 'PSCE', 'AASI.PA', 'FCPI',
    'SMOT', 'PXI', 'FEX', 'HSMV', 'JPSE', 'HUSV', 'FNY', 'QAT', 'DWX', 'LGLV',
    'FDNI', 'LCAS.MI', 'EFAS', 'XDWS.DE', '1311.T', '1325.T', 'RXI', 'FEP', 'QEW', 'KBWP',
    'FAB', 'DWAS', 'FMAG', 'HERO', 'QQMG', 'XMLV', 'IVVB', 'IPRV.L', 'FTHF', 'PSCI',
    'SNSR', 'JPRE', 'MADE', 'XJH', 'ISVL', 'EQRR', 'EDOW', 'FFEM', 'IFSD.L', 'FTC',
    'JPEM', 'FFGX', 'XWTS.DE', 'EMMV.L', 'ILIT', 'MATW.PA', 'WELG.DE', 'VFMV', 'FBCV', 'TOLZ',
    'LCTD', 'BNKE.PA', 'FBOT', 'FYT', 'PRUK.L', 'BBMC', '2514.T', 'MTAV.L', 'FAD', 'INDB.DE',
    'EMDM', 'INDEP.PA', 'CSD', 'PRAE.DE', 'FXG', 'FNX', 'AWHD.DE', 'ILCB', 'PSCC', 'PFI',
    'XUTC.DE', 'PR1J.DE', 'FLN', 'FICS', 'IEQU.L', '518A.T', 'FXD', 'SADA.DE', '2846.T', 'FBDC',
    'TINY', '2529.T', 'FTXG', 'XDN0.DE', 'WELD.DE', 'PUI', 'EDEN', 'DGT', 'JPME', 'MAXD.L',
    'PLAY.L', 'BBSC', 'QABA', 'PDN', 'ECNS', 'WCME', 'FPA', 'FJP', 'FDM', 'EIRL',
    'DDIV', 'XSLV', 'LCUK.DE', '2520.T', 'IEMD.L', 'XJR', 'HEAL', 'EMXU.PA', 'XSMI.DE', 'NXTG',
    '412A.T', 'IFSW.L', 'VNAM', 'TECB', 'SOCL', 'CEFA', 'MTRA', 'FDIF', 'CT2B.L', 'VDV',
    'NRJ.PA', 'KROP', 'MILN', 'FPWR', 'GMF', 'JPIN', 'VLU', 'WDIV', 'RBLD', 'EPAB.PA',
    'BLDR.L', 'EVX', 'DMAX', 'EMXF', 'SMAX', 'GLIN', 'FBUF', 'QRMI', 'XDND.DE', 'FSCS',
    'JPUS', 'VDG', 'FTCE', 'JPXN', 'AEJ.PA', 'FMED', 'JOYT', 'IAEX.L', 'MVED.L', 'MIB.PA',
    'FDCF', 'WELM.DE', 'QQQS', 'ONOF', 'IEUS', 'EMET', 'ISTM', 'EMXCN.SW', 'WOOD.L', 'AQWA',
    'PRAZ.DE', 'DVOL', 'ONLN', 'EDGX', 'MDUC.DE', 'BKF', 'TWOX', 'C007.DE', 'OPEN.L', 'MAXM.L',
    'TDV', 'DALI', 'CSTK', 'MLCC', 'XMOV.DE', 'QBIG', 'CSM', 'BRF', 'MOTI', 'RTH',
    'SIZE', 'DVLU', 'FNK', 'TOK', 'WAT.PA', 'EMC', 'F50C.DE', 'LVDS', 'NRGW.PA', 'FMTL',
    'KWT', 'CVY', 'RND', 'RNRG', 'JFLI', 'MMSC', 'TEKX', 'FDFF', 'FFLV', 'CPTL',
    'LYXIB.MC', 'USDB.L', 'CARZ', 'SDG', '1577.T', 'XPND', 'ELCR.PA', 'IDJG.L', 'ISRA', 'BRIC.L',
    'PSCF', 'RCTR', 'ENRG.PA', 'WELY.DE', 'IPXJ.L', 'FLOW', 'GXDW', 'ERTH', 'NYSX', 'FEUZ',
    'KNCT', 'EBIZ', 'SDEM', 'MMAX', 'STEN', 'POWA', 'IBRN', 'MDEV', 'XUEK.DE', 'FVC',
    'FDIQ', 'PDJE.PA', 'TELW.PA', 'WELC.DE', 'SMLV', 'GLIT.PA', 'WCMG', 'EMIF', 'WELT.DE', 'XITK',
    'FFDI', 'TRV.PA', 'XUCM.DE', 'CZA', '2518.T', 'CTEC', 'DGIN', 'VEEM', 'TEXN', '1319.T',
    'INDZ', 'QOWZ', 'QTR', 'PYZ', '10AJ.DE', 'TENM', 'DJSC.L', 'DAT', 'FTSE.PA', 'DJMC.L',
    'FCTR', 'ITAMID.MI', 'ION', 'C003.DE', 'C5E.PA', 'FKU', 'LEGR', 'FGM', 'INAA.L', 'C005.DE',
    'QQHG', 'PAWZ', 'FTDS', 'AEXX.PA', 'CUT', 'PSCM', 'FSPC', 'FTIF', 'KDOM.L', 'UPGD',
    'JLVP', 'IGSG.L', 'SMOG', 'PSCU', 'EKG', 'KLMT', 'GENZ', '1546.T', 'PSL', 'FMI.PA',
    'PMTL', 'GURU', 'LDEM', 'MEH.PA', 'JDIV', 'MAXS.L', 'CSUK.L', 'SCDS', 'LFEQ', 'XIFE.DE',
    'ISFE.L', 'QQLV', 'MRGR', 'FSGS', 'ELLE.PA', 'FSZ', 'SGQI.PA', 'C006.DE', 'FDTS', 'MMTM',
    'SUPL', 'HDMV', 'EFRA', 'MOTG', 'XEMN.DE', 'BMVP', 'SPDG', 'TRUH', 'XNZN.DE', 'SDG9.DE',
    'BNGE', 'MCDS', 'CLIX', 'DGLO', 'PEZ', 'C030.DE', '2643.T', 'XGLF.DE', 'EGUSAS.SW', 'GXPE',
    'XCNY', 'SHRY', 'XDRE.DE', 'ETEC', 'XG12.DE', 'NRAM.PA', 'PSCD', 'XDG3.DE', 'TEND.L', 'JIDE',
    'FPXE', 'TINT', 'MVAL', 'METL.L', 'RNEM', '1559.T', 'XNGI.DE', 'EBUY.PA', 'EUDIV.PA', 'TRUO',
    'TRUF', 'XZSP.DE', 'DTRE', 'JDOC', 'XFNT.DE', 'TRUN', 'SPXPW.SW', 'TRUM', 'DXSG.DE', 'OND',
    'TRUI', 'LCDS', 'MILL.PA', 'CWE.PA', '1560.T', 'ACUU.DE', 'GXLC', 'ANEW', 'SCITY.PA', 'TRUU',
    'IQSZ', 'LUXU.PA', 'TRUR', 'XNZW.DE', 'KFOR.L', 'XDGI.DE', 'ISHP', 'TMDV', 'DVVY', 'XG11.DE',
    'XZEC.DE', 'CHM.PA', 'XNNV.DE', 'XBUZ.DE', 'XDGS.DE', 'XEPA.DE', '1309.T', 'XNUS.DE', '294A.T', '1480.T',
    'CLIMA.SW', '0000H0.KS', '0000J0.KS', '0004G0.KS', '0005C0.KS', '0005D0.KS', '0007G0.KS', '0007N0.KS', '0008T0.KS', '0013P0.KS',
    '0018Z0.KS', '0023A0.KS', '0023B0.KS', '0026S0.KS', '0036R0.KS', '0036Z0.KS', '0038A0.KS', '0041D0.KS', '0046Y0.KS', '0047A0.KS',
    '0047N0.KS', '0047P0.KS', '0047R0.KS', '0048K0.KS', '0051G0.KS', '0052D0.KS', '0053L0.KS', '00634R.TW', '00651R.TW', '0065G0.KS',
    '0067V0.KS', '0067Y0.KS', '00682U.TW', '0078V0.KS', '0080G0.KS', '0082F0.KS', '0083S0.KS', '0088N0.KS', '0089D0.KS', '0090B0.KS',
    '0091P0.KS', '0092B0.KS', '0093B0.KS', '0097L0.KS', '0098F0.KS', '0098N0.KS', '0098Z0.KS', '0101N0.KS', '0102A0.KS', '0102X0.KS',
    '0103T0.KS', '0105D0.KS', '0111J0.KS', '0112X0.KS', '0114X0.KS', '0115C0.KS', '0115D0.KS', '0117V0.KS', '0120J0.KS', '0123G0.KS',
    '0127R0.KS', '0131A0.KS', '0131V0.KS', '0132H0.KS', '0138D0.KS', '0139P0.KS', '0141S0.KS', '0142D0.KS', '0144M0.KS', '0148J0.KS',
    '0153K0.KS', '0154F0.KS', '0155M0.KS', '0155N0.KS', '0167A0.KS', '0167Z0.KS', '0174J0.KS', '0174R0.KS', '0177A0.KS', '0177X0.KS',
    '0181B0.KS', '0181L0.KS', '0182R0.KS', '0183J0.KS', '0189Z0.KS', '0190C0.KS', '0190M0.KS', '0190Y0.KS', '0207G0.KS', '0209D0.KS',
    '0209Z0.KS', '0210A0.KS', '0215T0.KS', '0216Z0.KS', '0218K0.KS', '0224D0.KS', '0225V0.KS', '0227K0.KS', '0227L0.KS', '0228G0.KS',
    '0234N0.KS', '0238F0.KS', '0USE.DE', '100910.KS', '101280.KS', '102110.KS', '102780.KS', '102960.KS', '102970.KS', '104520.KS',
    '104530.KS', '105010.KS', '105190.KS', '105780.KS', '108450.KS', '108590.KS', '117460.KS', '117680.KS', '117690.KS', '117700.KS',
    '131890.KS', '1320.T', '1322.T', '1330.T', '1343.T', '1345.T', '1364.T', '138520.KS', '138530.KS', '138540.KS',
    '139220.KS', '139230.KS', '139240.KS', '139250.KS', '139260.KS', '139270.KS', '139280.KS', '139290.KS', '1397.T', '1398.T',
    '140570.KS', '140580.KS', '140700.KS', '140710.KS', '140950.KS', '143850.KS', '143860.KS', '145850.KS', '1476.T', '147970.KS',
    '148020.KS', '1481.T', '1483.T', '1485.T', '1488.T', '1495.T', '1499.T', '150460.KS', '152100.KS', '152870.KS',
    '153270.KS', '1547.T', '1551.T', '1555.T', '156080.KS', '1563.T', '157490.KS', '157500.KS', '1591.T', '1592.T',
    '1593.T', '1595.T', '1597.T', '1599.T', '161510.KS', '1658.T', '1659.T', '1660.T', '1679.T', '1680.T',
    '1681.T', '168580.KS', '1698.T', '174350.KS', '174360.KS', '178A.T', '182480.KS', '188A.T', '189400.KS', '192090.KS',
    '192720.KS', '195920.KS', '1DIV.BK', '200030.KS', '200250.KS', '200A.T', '2013.T', '2014.T', '2017.T', '2018.T',
    '203780.KS', '2080.T', '2081.T', '2082.T', '2088.T', '2096.T', '2097.T', '2098.T', '210A.T', '211560.KS',
    '211900.KS', '213610.KS', '213A.T', '215620.KS', '218420.KS', '219390.KS', '219480.KS', '220130.KS', '221A.T', '223190.KS',
    '2235.T', '2236.T', '2241.T', '2244.T', '2247.T', '2248.T', '2252.T', '2253.T', '226380.KS', '226980.KS',
    '227540.KS', '227550.KS', '227560.KS', '227570.KS', '228790.KS', '228800.KS', '228810.KS', '228820.KS', '229200.KS', '229720.KS',
    '233A.T', '234310.KS', '235A.T', '238720.KS', '241180.KS', '244580.KS', '244620.KS', '244660.KS', '244670.KS', '245340.KS',
    '245350.KS', '245360.KS', '245710.KS', '248270.KS', '250730.KS', '2515.T', '251590.KS', '2516.T', '252000.KS', '2521.T',
    '2522.T', '252650.KS', '2528.T', '2530.T', '253280.KS', '2552.T', '2555.T', '2556.T', '2558.T', '2560.T',
    '2562.T', '2564.T', '2565.T', '2566.T', '2624.T', '2626.T', '2627.T', '2628.T', '2629.T', '2636.T',
    '2637.T', '2639.T', '2640.T', '2641.T', '2642.T', '2645.T', '2646.T', '266160.KS', '266360.KS', '266370.KS',
    '266390.KS', '266410.KS', '266420.KS', '266550.KS', '269540.KS', '270800.KS', '273A.T', '275280.KS', '275300.KS', '275980.KS',
    '276000.KS', '276650.KS', '277540.KS', '278530.KS', '278540.KS', '279530.KS', '279540.KS', '2800.HK', '2803.HK', '280320.KS',
    '280920.KS', '2814.HK', '2815.HK', '281990.KS', '2823.HK', '2824.HK', '2825.HK', '2828.HK', '282A.T', '2830.HK',
    '2832.HK', '2835.HK', '283580.KS', '2836.T', '2837.HK', '2837.T', '2841.HK', '2847.T', '2849.T', '284980.KS',
    '2851.T', '2852.T', '2854.T', '2855.T', '285690.KS', '2864.T', '287180.KS', '289040.KS', '289250.KS', '289260.KS',
    '290130.KS', '291890.KS', '292050.KS', '292150.KS', '292160.KS', '292190.KS', '292500.KS', '293180.KS', '294400.KS', '295040.KS',
    '298770.KS', '3003.HK', '3004.HK', '300610.KS', '300640.KS', '300950.KS', '3012.HK', '3029.HK', '3032.HK', '3033.HK',
    '3037.HK', '3039.HK', '304760.KS', '3050.HK', '305540.KS', '3056.HK', '305720.KS', '307510.KS', '307520.KS', '3076.HK',
    '309230.KS', '3109.HK', '310960.KS', '310970.KS', '3110.HK', '3116.HK', '3121.HK', '3128.HK', '3136.HK', '3140.HK',
    '314250.KS', '3145.HK', '3147.HK', '3150.HK', '315270.KS', '315930.KS', '315960.KS', '315A.T', '316300.KS', '3169.HK',
    '316A.T', '3174.HK', '3182.HK', '3184.HK', '3185.HK', '3186.HK', '3189.HK', '3190.HK', '322400.KS', '322410.KS',
    '325010.KS', '325020.KS', '326230.KS', '326240.KS', '328A.T', '329200.KS', '332500.KS', '332930.KS', '332940.KS', '337120.KS',
    '337150.KS', '337160.KS', '3402.HK', '3403.HK', '3406.HK', '3423.HK', '3428.HK', '3431.HK', '3434.HK', '3437.HK',
    '3441.HK', '3442.HK', '3443.HK', '3444.HK', '3447.HK', '3453.HK', '3454.HK', '3456.HK', '3466.HK', '3470.HK',
    '3473.HK', '3477.HK', '3483.HK', '3486.HK', '3488.HK', '348A.T', '3499.HK', '352540.KS', '352560.KS', '354350.KS',
    '354A.T', '360200.KS', '360750.KS', '361580.KS', '363580.KS', '363A.T', '364960.KS', '364970.KS', '364980.KS', '364990.KS',
    '365000.KS', '367740.KS', '367760.KS', '367770.KS', '368190.KS', '368680.KS', '371150.KS', '371160.KS', '371460.KS', '371470.KS',
    '371870.KS', '372330.KS', '375760.KS', '375770.KS', '376410.KS', '377990.KS', '379780.KS', '379810.KS', '380340.KS', '381170.KS',
    '381180.KS', '381560.KS', '381570.KS', '383A.T', '388280.KS', '390400.KS', '391600.KS', '392A.T', '395150.KS', '395170.KS',
    '395280.KS', '395290.KS', '395A.T', '396500.KS', '396520.KS', '399110.KS', '3D3D.DE', '3DGE.DE', '3DUE.DE', '402970.KS',
    '404260.KS', '404540.KS', '404650.KS', '404A.T', '407300.KS', '407310.KS', '411540.KS', '413220.KS', '413A.T', '414780.KS',
    '415340.KS', '415920.KS', '416090.KS', '417450.KS', '417630.KS', '419430.KS', '419650.KS', '421320.KS', '426A.T', '427120.KS',
    '428560.KS', '429000.KS', '429010.KS', '429740.KS', '432840.KS', '433330.KS', '433500.KS', '434730.KS', '434960.KS', '435040.KS',
    '435A.T', '437370.KS', '438900.KS', '440910.KS', '441540.KS', '442320.KS', '443A.T', '444490.KS', '446700.KS', '446770.KS',
    '448100.KS', '448300.KS', '449180.KS', '449450.KS', '449680.KS', '449770.KS', '450180.KS', '453630.KS', '453640.KS', '453650.KS',
    '453660.KS', '453810.KS', '453950.KS', '454180.KS', '454320.KS', '455850.KS', '455860.KS', '456250.KS', '457990.KS', '458750.KS',
    '459560.KS', '459A.T', '460660.KS', '461900.KS', '461A.T', '462010.KS', '463250.KS', '463300.KS', '463640.KS', '463680.KS',
    '463690.KS', '464600.KS', '464610.KS', '464920.KS', '464930.KS', '465330.KS', '465580.KS', '465660.KS', '465670.KS', '466810.KS',
    '466920.KS', '466930.KS', '466940.KS', '466A.T', '469170.KS', '469790.KS', '472840.KS', '473460.KS', '473590.KS', '473640.KS',
    '474800.KS', '475050.KS', '475300.KS', '475310.KS', '475350.KS', '476070.KS', '476260.KS', '476310.KS', '476690.KS', '476800.KS',
    '477730.KS', '479730.KS', '479850.KS', '480030.KS', '480310.KS', '480460.KS', '481180.KS', '481190.KS', '483020.KS', '483030.KS',
    '483570.KS', '484880.KS', '485540.KS', '485690.KS', '486450.KS', '487230.KS', '487240.KS', '487750.KS', '487950.KS', '488200.KS',
    '488210.KS', '488480.KS', '488500.KS', '489250.KS', '489290.KS', '489860.KS', '489A.T', '490090.KS', '490480.KS', '491090.KS',
    '491220.KS', '491700.KS', '491820.KS', '493420.KS', '494410.KS', '494670.KS', '494840.KS', '495040.KS', '495050.KS', '495330.KS',
    '495550.KS', '495750.KS', '495850.KS', '496080.KS', '496090.KS', '496120.KS', '496770.KS', '497510.KS', '497520.KS', '498270.KS',
    '498610.KS', '498860.KS', '499150.KS', '509A.T', '512A.T', '513A.T', '520080.KS', '520089.KS', '526A.T', '530127.KS',
    '530141.KS', '530142.KS', '540A.T', '552A.T', '564A.T', '566A.T', '5ESGS.SW', '608A.T', '609A.T', '610A.T',
    '633A.T', '6PSC.DE', '6PSK.DE', '6TVL.DE', 'A200.AX', 'A300.AX', 'AADR', 'AAEQ', 'AAPY', 'AAUA',
    'AAUB', 'AAUS', 'AAVD.DE', 'AAVM', 'ABCS', 'ABEQ', 'ABFL', 'ABIG', 'ABLD', 'ABLG',
    'ABLS', 'ABOT', 'ABSL10BANK.NS', 'ABSLBANETF.NS', 'ABSLNN50ET.NS', 'ABSLPSE.NS', 'ACEP', 'ACGO', 'ACGR', 'ACIO',
    'ACKY', 'ACLC', 'ACSG', 'ACSI', 'ACSV', 'ACTS', 'ACVF', 'ACVU', 'ADDS', 'ADEF.AX',
    'ADIV', 'ADME', 'ADPV', 'AELV', 'AEMC', 'AEMG', 'AEMV', 'AESB', 'AESG', 'AESL',
    'AESR', 'AFGR', 'AFOS', 'AGIQ', 'AGIX', 'AGNG', 'AGRI11.SA', 'AGRW', 'AGSCF', 'AHBM',
    'AICH', 'AIEQ', 'AIFR', 'AIHY', 'AIMG', 'AINF', 'AINF.AX', 'AIPO', 'AIUP', 'AIVC',
    'AIX', 'AKAF', 'AKRE', 'ALAI', 'ALIL', 'ALLW', 'ALPHA.NS', 'ALPHAETF.NS', 'ALRG', 'ALTL',
    'ALUG11.SA', 'AMAX.TO', 'AMEA.DE', 'AMEI', 'AMEM', 'AMGR', 'AMID', 'AMLP', 'AMMO', 'AMZH.TO',
    'AMZP', 'ANRJ.L', 'ANTW', 'AONETMMQ50.NS', 'AONETOTAL.NS', 'AOTG', 'APA.NZ', 'APIE', 'APUE', 'AQEC',
    'AQLG', 'AQLT', 'AQLT.AX', 'ARGE11.SA', 'ARIA', 'ARKG', 'ARKK', 'ARKW', 'ARKX', 'ARMR.AX',
    'ARMY', 'ARWG', 'ASCE', 'ASCI', 'ASD', 'ASD.NZ', 'ASF.NZ', 'ASHR', 'ASHR.TO', 'ASHS',
    'ASIA.AX', 'ASLV', 'ASMH', 'ASP.NZ', 'ASR.NZ', 'ASRS.DE', 'ATEC.AX', 'ATFV', 'ATTR', 'AUE.NZ',
    'AUS.NZ', 'AUST.AX', 'AUTOBEES.NS', 'AUTOIETF.NS', 'AUVP11.SA', 'AV', 'AVDE', 'AVDS', 'AVDV', 'AVEE',
    'AVEM', 'AVES', 'AVGE', 'AVGV', 'AVIE', 'AVIV', 'AVLC', 'AVLV', 'AVMC', 'AVMV',
    'AVNM', 'AVNV', 'AVOS', 'AVRE', 'AVRY', 'AVSC', 'AVSD', 'AVSE', 'AVSU', 'AVTM',
    'AVUQ', 'AVUS', 'AVUV', 'AVXC', 'AW11.DE', 'AW12.DE', 'AW1J.DE', 'AWAY', 'AZTD', 'B1LD.DE',
    'B3BR11.SA', 'BAFE', 'BAGX', 'BAIV', 'BAMD', 'BAMG', 'BAMV', 'BANK10ADD.NS', 'BANK10BETF.NS', 'BANKADD.NS',
    'BANKBETA.NS', 'BANKBETF.NS', 'BANKETF.NS', 'BANKIETF.NS', 'BANKNIFTY1.NS', 'BANKPSU.NS', 'BAOR.AX', 'BASE.TO', 'BASG', 'BASV',
    'BAT3.DE', 'BATT', 'BAY', 'BBC', 'BBHL', 'BBHM', 'BBLS', 'BBLU', 'BBNPNBETF.NS', 'BBOV11.SA',
    'BCEM', 'BCFN', 'BCFS.DE', 'BCGD', 'BCGS', 'BCHP', 'BCIC11.SA', 'BCIL', 'BCSM', 'BCTK',
    'BCUS', 'BDEF11.SA', 'BDGS', 'BDIV', 'BDIV.TO', 'BDOM11.SA', 'BDVG', 'BEEX', 'BEEZ', 'BESF',
    'BEST11.SA', 'BETZ', 'BFIN.TO', 'BFOR', 'BFSI.NS', 'BGBL.AX', 'BGDV', 'BGEG', 'BGGG', 'BGIA',
    'BGIE.TO', 'BGIG', 'BGUS', 'BGX.DE', 'BIBL', 'BILD', 'BINT', 'BINV', 'BIVC.TO', 'BIVU.TO',
    'BIZD', 'BIZD11.SA', 'BJL7.DE', 'BKDV', 'BKEM', 'BKGI', 'BKIE', 'BKLC', 'BKLG', 'BKMC',
    'BKSE', 'BLDG', 'BLDX', 'BLES', 'BLGR', 'BLOK', 'BLOV.TO', 'BLUC', 'BLUE.SW', 'BLUX',
    'BMMPF', 'BMMT11.SA', 'BMSCG.BK', 'BMSCITH.BK', 'BNC.TO', 'BNKETFAXIS.NS', 'BNKS.AX', 'BNKS11.SA', 'BOAT', 'BOBP',
    'BOUT', 'BOVA11.SA', 'BOVB11.SA', 'BOVS11.SA', 'BOVV11.SA', 'BPAY', 'BPH', 'BQGE.DE', 'BRAX11.SA', 'BRCE',
    'BREE', 'BRES', 'BREW', 'BREW11.SA', 'BRIE', 'BRIF', 'BRNY', 'BROL', 'BRSM', 'BRSV',
    'BRXC11.SA', 'BSE500IETF.NS', 'BSET100.BK', 'BSLSENETFG.NS', 'BSMC', 'BSVO', 'BTER11.SA', 'BUIL', 'BUL', 'BULD',
    'BUSA', 'BUSM', 'BUYO', 'BUYZ', 'BVAL', 'BVBR11.SA', 'BWQG', 'BWTG', 'BXPO11.SA', 'BZWHF',
    'BZZ', 'C4RE.DE', 'C9DF.DE', 'CACE.TO', 'CACX.L', 'CADE.TO', 'CAEM.TO', 'CAFG', 'CAGE.TO', 'CAGX.TO',
    'CALF', 'CALV.TO', 'CAML', 'CAMO.TO', 'CAMX', 'CANC', 'CAPA', 'CAPE', 'CAPE11.SA', 'CAPG.TO',
    'CAPI.TO', 'CAPN.TO', 'CAPQ.TO', 'CAPU.TO', 'CARK', 'CAS', 'CASA11.SA', 'CASV.TO', 'CAUS.TO', 'CAUV.TO',
    'CBLS', 'CBOT', 'CBSE', 'CBUG.TO', 'CCFE', 'CCGP.TO', 'CCIP.TO', 'CCML', 'CCNR', 'CCOR',
    'CCSO', 'CCUL.TO', 'CCUS.TO', 'CD1.DE', 'CDEF.TO', 'CDEI', 'CDIG', 'CDIV.TO', 'CDL', 'CDZ.TO',
    'CEB4.DE', 'CEMNTGROWW.NS', 'CEQT.TO', 'CEQY.TO', 'CEW.TO', 'CFA', 'CFIN.TO', 'CFLO.AX', 'CGCV', 'CGDG',
    'CGDI.TO', 'CGDV', 'CGFS', 'CGGE', 'CGGG', 'CGGO', 'CGGR', 'CGHE.AX', 'CGIC', 'CGLO.TO',
    'CGMM', 'CGNG', 'CGPT', 'CGR.TO', 'CGRA.TO', 'CGRN.TO', 'CGRO', 'CGUN.AX', 'CGUS', 'CGV',
    'CGVV', 'CGXU', 'CHAT', 'CHDIV.SW', 'CHDIVA.SW', 'CHEMICAL.NS', 'CHGX', 'CHIP11.SA', 'CHNTRAC.MX', 'CHPS',
    'CHQQ.TO', 'CHSI.DE', 'CIEM.TO', 'CINF.TO', 'CINT.TO', 'CINV.TO', 'CLCG', 'CLCV', 'CLIM', 'CLML.TO',
    'CLR.SI', 'CLUB', 'CMAG', 'CMGG.TO', 'CMIG.TO', 'CMOM.TO', 'CMVP.TO', 'CNAV', 'CNEW.AX', 'CNQQ',
    'COAL', 'COIL.TO', 'COMM.TO', 'COMX.TO', 'CONS.NS', 'CONSUMAXIS.NS', 'CONSUMER.NS', 'CONSUMIETF.NS', 'CONY.TO', 'COOL',
    'COPA', 'COPJ', 'COPP', 'COPP.TO', 'COPX', 'COPY', 'COWG', 'COWS', 'COWZ', 'CPAI',
    'CPPR.AX', 'CPSEETF.NS', 'CQQQC.SW', 'CQTM', 'CRAM', 'CRIB', 'CRTC', 'CS1.L', 'CSB', 'CSMD',
    'CSMD.TO', 'CSSLI.SW', 'CSSMI.SW', 'CSSMIM.SW', 'CSUC.TO', 'CTEF', 'CTIF', 'CUD.TO', 'CUDV.TO', 'CUKS.L',
    'CUTE.TO', 'CVAR', 'CVGD', 'CVIE', 'CVLC', 'CVLU.TO', 'CVMC', 'CVSM', 'CWS', 'CXT.SI',
    'CYH.TO', 'CZX.DE', 'D5BK.DE', 'D6RD.DE', 'D6RU.DE', 'DACE.AX', 'DACL.TO', 'DAGL.TO', 'DAOR.AX', 'DAPP',
    'DARP', 'DBAW', 'DCOR', 'DDDD', 'DDLS', 'DDWM', 'DEEF', 'DEEP', 'DEFENCE.NS', 'DEHP',
    'DEM', 'DEMWF', 'DEMZ', 'DEPW', 'DES', 'DESK', 'DEUS', 'DEW', 'DEXC', 'DFAC',
    'DFAE', 'DFAI', 'DFAL', 'DFAR', 'DFAS', 'DFAT', 'DFAU', 'DFAW', 'DFAX', 'DFE',
    'DFEM', 'DFEV', 'DFGH.AX', 'DFGR', 'DFIC', 'DFIS', 'DFIV', 'DFJ', 'DFLV', 'DFMC',
    'DFND.AX', 'DFND.SW', 'DFNL', 'DFSE', 'DFSI', 'DFSU', 'DFSV', 'DFTT', 'DFUS', 'DFUV',
    'DFVE', 'DFVX', 'DGCE.AX', 'DGLEF', 'DGR.TO', 'DGRC.TO', 'DGRE', 'DGRS', 'DGRW', 'DGRW.MX',
    'DGS', 'DGVA.AX', 'DHS', 'DICE', 'DIEM', 'DIHP', 'DIM', 'DINT', 'DIPR', 'DISC.TO',
    'DISK', 'DISV', 'DIV.MX', 'DIV.NZ', 'DIVD', 'DIVD11.SA', 'DIVH', 'DIVI', 'DIVIDEND.NS', 'DIVL',
    'DIVN', 'DIVO11.SA', 'DIVS', 'DIVY', 'DIVZ', 'DJD', 'DJSC.SW', 'DLCU', 'DLN', 'DLS',
    'DMAD.L', 'DMEC.TO', 'DMEE.TO', 'DMEI.TO', 'DMEU.TO', 'DMID.TO', 'DMQC.TO', 'DMXU', 'DNL', 'DOCK',
    'DOCT', 'DOL', 'DOME.DE', 'DON', 'DRAM', 'DRES', 'DRFC.TO', 'DRFE.TO', 'DRFG.TO', 'DRFU.TO',
    'DRGN.AX', 'DRKY', 'DRLL', 'DRMC.TO', 'DRME.TO', 'DRMP', 'DRMU.TO', 'DRMY', 'DRNZ', 'DRUG.AX',
    'DRUP', 'DSMC', 'DSTL', 'DSTX', 'DTAN', 'DTD', 'DTEC', 'DTEC.AX', 'DTH', 'DTRE.L',
    'DUHP', 'DUKQ', 'DUKX', 'DUNK', 'DURA', 'DUSA', 'DUSG', 'DUTY', 'DVAL', 'DVDN',
    'DVGR', 'DVHG.AX', 'DVND', 'DVYA', 'DWAW', 'DWLD', 'DWM', 'DWMF', 'DWUS', 'DXIV',
    'DXJ', 'DXJJF', 'DXUV', 'DYTA', 'E200.AX', 'E908.DE', 'EAFG', 'EAFZ.AX', 'EAGL', 'EART',
    'EASY', 'EASY.TO', 'EBANK10.NS', 'EBANKNIFTY.NS', 'EBI', 'EBIT', 'ECAPINSURE.NS', 'ECDC.DE', 'ECML', 'ECOO11.SA',
    'ECOW', 'EDEF.DE', 'EDEU.DE', 'EDGE.TO', 'EDGF.TO', 'EDGI', 'EDGU', 'EDOG', 'EDOG.L', 'EEAE.DE',
    'EEGF.DE', 'EES', 'EFFE', 'EFFI', 'EFQA.DE', 'EGGQ', 'EGGS', 'EGGY', 'EGRE.DE', 'EGRP.L',
    'EHE.TO', 'EIGA.AX', 'EINC', 'EL4D.DE', 'EL4E.DE', 'ELCV', 'ELFB.DE', 'ELFY', 'ELM250.NS', 'ELMDIV.NS',
    'EMAX.TO', 'EMCR', 'EMEM', 'EMEQ', 'EMES', 'EMETAL.NS', 'EMF.NZ', 'EMG.NZ', 'EMKT', 'EMKT.AX',
    'EMKX.DE', 'EMMF', 'EMOP', 'EMQQ', 'EMSC', 'EMULTIMQ.NS', 'EMXC.AX', 'ENERGY.NS', 'ENERGYAXIS.NS', 'ENEXT50.NS',
    'ENFR', 'EPAI', 'EPEM', 'EPI', 'EPIN', 'EPMB', 'EPMV', 'EPS', 'EPSB', 'EPSV',
    'EQIN', 'EQIN.AX', 'EQL', 'EQLT.TO', 'EQTY', 'EQUAL200.NS', 'EQUAL50.NS', 'EQY.TO', 'ERET', 'ERTH.AX',
    'ES3.SI', 'ESENSEX.NS', 'ESG.NS', 'ESG.NZ', 'ESGB11.SA', 'ESGC.TO', 'ESGG', 'ESGG.TO', 'ESGI.AX', 'ESIC.L',
    'ESIF.L', 'ESIH.L', 'ESIM', 'ESLG', 'ESLV', 'ESN', 'ESNB.DE', 'ESSC', 'ESUM', 'ETFBW20TR.WA',
    'ETFPZUW20M40.WA', 'ETFT', 'ETHI.AX', 'ETHI.TO', 'ETHO', 'EUAD', 'EUAT11.SA', 'EUDG', 'EUDV.L', 'EUF.NZ',
    'EUFG.DE', 'EUG.NZ', 'EUHD.SW', 'EUN1.DE', 'EUP0.DE', 'EUPD.DE', 'EUV', 'EVIETF.NS', 'EVINDIA.NS', 'EVO.TO',
    'EWBZ11.SA', 'EX20.AX', 'EXEQ', 'EXH4.DE', 'EXIC.DE', 'EXID.DE', 'EXIE.DE', 'EXIF.DE', 'EXS2.DE', 'EXSC.DE',
    'EXSE.DE', 'EXUS', 'EXUS.AX', 'EXXU.DE', 'EXXV.DE', 'EYES', 'EYLD', 'EZM', 'FAIR.AX', 'FANG.AX',
    'FASA.L', 'FAUS.TO', 'FBT', 'FCAE.TO', 'FCBRF', 'FCCD.TO', 'FCCQ.TO', 'FCCV.TO', 'FCII.TO', 'FCIQ.TO',
    'FCQH.TO', 'FCRC.TO', 'FCRI.TO', 'FCRR.TO', 'FCRU.TO', 'FCSYZ.DE', 'FCTE', 'FCUD.TO', 'FCUH.TO', 'FCUQ.TO',
    'FCUS', 'FCUV.TO', 'FCVH.TO', 'FD5BL.DE', 'FDG', 'FDIV', 'FDL.MX', 'FDL.TO', 'FDLS', 'FDN.TO',
    'FDRS', 'FDV', 'FDVV', 'FEDM', 'FEGE', 'FEMD', 'FEMO.TO', 'FEMX.AX', 'FEOE', 'FESC',
    'FEUI.DE', 'FEUQ.DE', 'FEUS', 'FEXD.L', 'FFF', 'FFND', 'FFOG', 'FFOX', 'FFTY', 'FGBL.L',
    'FGEP.TO', 'FGQC.SW', 'FGSM', 'FGSM.TO', 'FGZJ.DE', 'FHG.TO', 'FHIL', 'FHNG.AX', 'FIND11.SA', 'FINIETF.NS',
    'FINO.TO', 'FINT', 'FINT.TO', 'FIRE.AX', 'FITZ', 'FIW', 'FIZY', 'FJ7G.DE', 'FKU.L', 'FLAU',
    'FLBR', 'FLCC', 'FLCE', 'FLCG', 'FLCV', 'FLDZ', 'FLEE', 'FLEU', 'FLEXIADD.NS', 'FLGB',
    'FLGB.MX', 'FLGR', 'FLIN', 'FLJP', 'FLKR', 'FLLA', 'FLMX', 'FLPE.DE', 'FLQAF', 'FLQL',
    'FLQM', 'FLQS', 'FLSW', 'FLTW', 'FLV', 'FMAX.TO', 'FMCE', 'FMCGADD.NS', 'FMCGIETF.NS', 'FMCX',
    'FMKT', 'FMQQ', 'FMTM', 'FNZ.NZ', 'FOOD.AX', 'FOTO', 'FOWF', 'FPAG', 'FPRO', 'FRDM',
    'FREL', 'FRGG.AX', 'FRGN', 'FRIZ', 'FRTY', 'FRUT', 'FRWD', 'FSCC', 'FSF.TO', 'FSML.AX',
    'FST.TO', 'FT1K.L', 'FTAL.SW', 'FTDPF', 'FTG', 'FTGG.DE', 'FTNQF', 'FTSRF', 'FTWO', 'FUD.TO',
    'FUEL.AX', 'FUQIF', 'FUSD', 'FUTR.AX', 'FWD', 'FWSD.L', 'FXH.MX', 'FXL.MX', 'FXM.TO', 'FXO.MX',
    'FYLD', 'G3B.SI', 'GAB.SI', 'GABF', 'GAIQ', 'GALX', 'GARA', 'GARP.AX', 'GARY', 'GASZ',
    'GBSL.TO', 'GCAD', 'GCEI.TO', 'GCFE.TO', 'GCOW', 'GCQF.AX', 'GCSC.TO', 'GDIV', 'GDIV11.SA', 'GDOC',
    'GDX', 'GEM', 'GENB11.SA', 'GEND', 'GENDES.SW', 'GENW', 'GEQ', 'GEQT.TO', 'GEW', 'GFGE.TO',
    'GFGF', 'GFLW', 'GGEP.TO', 'GGM', 'GGPY.TO', 'GGRW', 'GGTL', 'GHRP.AX', 'GIAI.TO', 'GIAX',
    'GICD.TO', 'GIDY.TO', 'GIES.TO', 'GIGD.TO', 'GIGF.TO', 'GII', 'GIND', 'GINN', 'GIUS.TO', 'GIVE.AX',
    'GK', 'GKAT', 'GLAM', 'GLCR', 'GLDX.TO', 'GLIN.AX', 'GLIX', 'GLMP11.SA', 'GLOB.AX', 'GLOW',
    'GLPR.AX', 'GLRY', 'GMOI', 'GMOV', 'GMTL.AX', 'GMVW.AX', 'GNR', 'GOAT.AX', 'GOAU', 'GOEX',
    'GOLB.L', 'GOLS', 'GOOP', 'GOP', 'GPR.NZ', 'GPT', 'GQGU', 'GQI', 'GQQQ', 'GQRE',
    'GREK', 'GRGB.L', 'GRIN', 'GRNI', 'GRNJ', 'GRNY', 'GRO.SI', 'GROWWCAPM.NS', 'GROWWCHEM.NS', 'GROWWDEFNC.NS',
    'GROWWHOSPI.NS', 'GROWWMC150.NS', 'GROWWMETAL.NS', 'GROWWN200.NS', 'GROWWNET.NS', 'GROWWNXT50.NS', 'GROWWPOWER.NS', 'GROWWPSE.NS', 'GROWWPSUBK.NS', 'GROWWRAIL.NS',
    'GROWWRLTY.NS', 'GROWWSC250.NS', 'GROZ', 'GRPA.AX', 'GRT8.SW', 'GRW', 'GRX.DE', 'GSC', 'GSEC10IETF.NS', 'GSEE',
    'GSEU', 'GSEW', 'GSGO', 'GSIB', 'GSID', 'GSIE', 'GSJY', 'GSLC', 'GSSC', 'GSUS',
    'GSUS.AX', 'GSWO', 'GTOP', 'GTR', 'GTUM.AX', 'GUNR', 'GUSA', 'GVAL', 'GVIP', 'GVLE',
    'GVLU', 'GWTH.AX', 'GXUS', 'GXUS11.SA', 'HALX', 'HAPI', 'HAPS', 'HAUS', 'HAUZ', 'HAWG',
    'HBA.TO', 'HBDV.TO', 'HBF.TO', 'HBGD.TO', 'HBNK.TO', 'HBOP.TO', 'HBTA', 'HBTA.TO', 'HCA.TO', 'HCRE.TO',
    'HDEF', 'HDFCBSE500.NS', 'HDFCGROWTH.NS', 'HDFCLOWVOL.NS', 'HDFCMID150.NS', 'HDFCMOMENT.NS', 'HDFCNEXT50.NS', 'HDFCNIF100.NS', 'HDFCNIFBAN.NS', 'HDFCNIFIT.NS',
    'HDFCNIMEG.NS', 'HDFCPSUBK.NS', 'HDFCPVTBAN.NS', 'HDFCQUAL.NS', 'HDFCSENSEX.NS', 'HDFCSML250.NS', 'HDFCVALUE.NS', 'HDUS', 'HEALTHADD.NS', 'HEALTHAXIS.NS',
    'HEALTHIETF.NS', 'HEALTHY.NS', 'HEB.TO', 'HEDG', 'HEDJ', 'HEFA', 'HEGD', 'HELS', 'HEQT', 'HEQT.TO',
    'HERD', 'HERO.TO', 'HERT11.SA', 'HEUR.AX', 'HEWB.TO', 'HFG.TO', 'HFGO', 'HFN.TO', 'HFSP', 'HFXI',
    'HGBL.AX', 'HGCQ.AX', 'HGGG.TO', 'HGR.TO', 'HGRO', 'HHIH.TO', 'HHL.TO', 'HIAI', 'HIDV', 'HIG.TO',
    'HIGH11.SA', 'HIND.TO', 'HIS', 'HJPN.AX', 'HLAL', 'HLIF.TO', 'HLIT.TO', 'HLTH', 'HLTH.AX', 'HMAX.TO',
    'HMCX.L', 'HMMJ.TO', 'HOMZ', 'HPF.TO', 'HPRC.SW', 'HPYB.TO', 'HQDG', 'HQGO', 'HRIF.TO', 'HRTS',
    'HRZSF', 'HSEP.L', 'HSUK.L', 'HTA.TO', 'HTEC', 'HTEK11.SA', 'HTTP.AX', 'HTUS', 'HUBE.DE', 'HUBL.TO',
    'HUGE.AX', 'HULC.TO', 'HULK.AX', 'HULL', 'HUM.TO', 'HUMM', 'HURA.TO', 'HUTL.TO', 'HVAC', 'HVLU.AX',
    'HVOI.TO', 'HVOL.TO', 'HWGV', 'HWIV', 'HWSM', 'HXCN.TO', 'HXDM.TO', 'HXE.TO', 'HXEM.TO', 'HXF.TO',
    'HXH.TO', 'HXT.TO', 'HXX.TO', 'HYGG.AX', 'HYLD.AX', 'HYP', 'IAINF', 'IBOB11.SA', 'IBUY', 'ICAE.TO',
    'ICAP', 'ICF', 'ICFP.DE', 'ICICIB22.NS', 'ICOW', 'ICPY', 'IDEQ', 'IDOG', 'IDVZ', 'IEFQ.L',
    'IEMIF', 'IFGL', 'IFLO', 'IFRA.AX', 'IFXAF', 'IGEO.TO', 'IGET.TO', 'IGGY', 'IGLIF', 'IHD.AX',
    'IHDG', 'IHE.MX', 'IHF.MX', 'IHOO.AX', 'IHSZF', 'IIAE.TO', 'IICE.TO', 'IIGF.AX', 'IIME.TO', 'IIMF.TO',
    'IINC', 'IIND.AX', 'IISV.AX', 'IJH.AX', 'IKO.AX', 'ILC.AX', 'ILCG', 'ILCV', 'ILOW', 'IMAX.TO',
    'IMCV', 'IMFE.TO', 'IMOM', 'INAI.TO', 'INC.SI', 'INCE', 'INCO', 'INDH', 'INDQ', 'INDS',
    'INEQ', 'INES.AX', 'INF.NZ', 'INFL', 'INFO', 'INFR.TO', 'INFRA.NS', 'INFRAIETF.NS', 'INIF.AX', 'INOC.TO',
    'INQQ', 'INSUREIETF.NS', 'INTERNET.NS', 'INTL', 'INVN', 'IOPP', 'IOZ.AX', 'IPAC', 'IPAY', 'IPO',
    'IPOS', 'IPRH.L', 'IPRP.SW', 'IQD.TO', 'IQDF', 'IQDG', 'IQDY', 'IQGR', 'IQM', 'IQQM.DE',
    'IQSI', 'IQSM', 'IQSU', 'IRSHF', 'IS07.DE', 'ISCB', 'ISLM.AX', 'ISMD', 'ISMD.AX', 'ISMWF',
    'ISO.AX', 'ISRL', 'ISUS11.SA', 'ISXAF', 'IT.NS', 'ITADD.NS', 'ITAN', 'ITAXIS.NS', 'ITBEES.NS', 'ITBETA.NS',
    'ITEQ', 'ITETF.NS', 'ITIETF.NS', 'ITIN.TO', 'IUAE.TO', 'IUKD.SW', 'IUKP.L', 'IUNSF', 'IUTCF', 'IVAL',
    'IVEP', 'IVES', 'IVSI', 'IVSS', 'IVSX', 'IVZBANKNF.NS', 'IVZSENSEX.NS', 'IWFG', 'IWLG', 'IWMY',
    'IXEDF', 'IXG.MX', 'IXLDF', 'IXSHF', 'IXUS', 'IYF.MX', 'IYG.MX', 'IYRI', 'IYT.MX', 'IZRL',
    'IZZ.AX', 'JAPN', 'JAPN.TO', 'JCEU.DE', 'JDVI', 'JDVL', 'JEDI', 'JETS', 'JEUS.DE', 'JGRW',
    'JHAC', 'JHDG', 'JHDV', 'JHEM', 'JHID', 'JHMD', 'JHML', 'JHMM', 'JHSC', 'JINT',
    'JLCO', 'JMID', 'JOET', 'JOUL', 'JPCE.DE', 'JPEQF', 'JPN.NZ', 'JPY', 'JREZ.DE', 'JSMD',
    'JSML', 'JSTC', 'JUHE.DE', 'JUIE.DE', 'JUREF', 'JUST', 'JXX', 'K0MR.SW', 'KAIT', 'KAT',
    'KBWY', 'KCAI', 'KCHP', 'KDEF', 'KDVD', 'KEMQ', 'KMCA', 'KMEM', 'KMID', 'KNGC.TO',
    'KNGG.TO', 'KNGU.TO', 'KNGX.TO', 'KNO', 'KONG', 'KOOL', 'KPHO', 'KPO', 'KQQQ', 'KRANF',
    'KSPY', 'KSTR', 'KTEC', 'KURE', 'KVLE', 'KWEB', 'KWH', 'KWIN', 'KYC', 'L1IF.AX',
    'LARGEMID50.NS', 'LATR', 'LAZR', 'LBO', 'LCAP', 'LCF', 'LCLG', 'LCTO', 'LDEG.SW', 'LEAD.TO',
    'LEEU.DE', 'LENS', 'LEVR.AX', 'LFSC', 'LGDX', 'LGH', 'LGHT', 'LGQI.DE', 'LICNETFSEN.NS', 'LICNFNHGP.NS',
    'LICNMID100.NS', 'LIT', 'LITL', 'LITP', 'LIV.NZ', 'LLYH.TO', 'LM9C.DE', 'LMAX.TO', 'LNGX', 'LNXC.DE',
    'LOEV', 'LOGO', 'LOHA', 'LONG.TO', 'LOPP', 'LOUP', 'LOWV', 'LOWVOL.NS', 'LOWVOL1.NS', 'LOWVOLIETF.NS',
    'LPGD.AX', 'LPHD.AX', 'LPRE', 'LPSV', 'LQAI', 'LRGC', 'LRGE', 'LRGG', 'LRND', 'LRNZ',
    'LSAF', 'LSAT', 'LSGE.AX', 'LSGR', 'LST', 'LSVD', 'LTCM.DE', 'LUMA', 'LVHD', 'LVHI',
    'LVOL11.SA', 'LYLD', 'LYTE', 'LYX4.DE', 'M30G.L', 'M9SV.L', 'MAEC.TO', 'MAFANG.NS', 'MAGC', 'MAGS',
    'MAGV.TO', 'MAHKTECH.NS', 'MAJATE.CO', 'MAJLO.CO', 'MAJSWN.CO', 'MAKEINDIA.NS', 'MALX.TO', 'MANUFGBEES.NS', 'MARS', 'MART.TO',
    'MATB11.SA', 'MAUG.TO', 'MAUV.TO', 'MAVF', 'MBCE', 'MBNK.TO', 'MBOX', 'MCAN.TO', 'MCLC.TO', 'MCLV.TO',
    'MCSM.TO', 'MCYC.TO', 'MDEF.TO', 'MDIF.TO', 'MDIV.TO', 'MDLV', 'MDPL', 'MDZ.NZ', 'MEDI', 'MEDX',
    'MEDX.TO', 'MEME', 'MEMY', 'MEQT.TO', 'METALIETF.NS', 'METL', 'METV', 'MEXTRAC.MX', 'MFDX', 'MFEM',
    'MFIG', 'MFMO', 'MFUN.TO', 'MFUS', 'MFVL', 'MGDV.TO', 'MGMT', 'MGNR', 'MGQE.TO', 'MHCD.TO',
    'MHOT.AX', 'MICH.AX', 'MID', 'MID150.NS', 'MID150CASE.NS', 'MIDBANKADD.NS', 'MIDCAP.NS', 'MIDCAPADD.NS', 'MIDCAPBETA.NS', 'MIDCAPETF.NS',
    'MIDCAPIETF.NS', 'MIDQ50ADD.NS', 'MIDSELIETF.NS', 'MIDSMALL.NS', 'MIGO', 'MILL11.SA', 'MINF.TO', 'MINT.TO', 'MIQE.TO', 'MIVG.TO',
    'MJ', 'MKK1.DE', 'MLPA', 'MLPI', 'MLPX', 'MLRG.TO', 'MLULF', 'MMID.TO', 'MMLP', 'MNC.NS',
    'MNVT', 'MNXT.TO', 'MNZL', 'MOALPHA50.NS', 'MOBANK10.NS', 'MOCAPITAL.NS', 'MODEFENCE.NS', 'MODL', 'MOENERGY.NS', 'MOGL.AX',
    'MOHEALTH.NS', 'MOINFRA.NS', 'MOIPO.NS', 'MOLOWVOL.NS', 'MOM100.NS', 'MOM30IETF.NS', 'MOM50.NS', 'MOMENTUM.NS', 'MOMENTUM30.NS', 'MOMENTUM50.NS',
    'MOMETAL.NS', 'MOMGF.NS', 'MOMIDMTM.NS', 'MOMMIDCAP.NS', 'MOMNC.NS', 'MOMOMENTUM.NS', 'MON100.NS', 'MONEXT50.NS', 'MONIFTY100.NS', 'MONQ50.NS',
    'MOOILGAS.NS', 'MOPSE.NS', 'MOQUALITY.NS', 'MORE.TO', 'MOREALTY.NS', 'MORT', 'MOSERVICE.NS', 'MOSMALL250.NS', 'MOTO', 'MOTOUR.NS',
    'MPLY', 'MPY.TO', 'MREL.TO', 'MRSK', 'MSCIINDIA.NS', 'MSFH.TO', 'MSFY', 'MSLC', 'MSOS', 'MSRUSB.SW',
    'MSSM', 'MSSS', 'MSST', 'MSTB', 'MSTQ', 'MSTY.TO', 'MTAW', 'MTUM.AX', 'MULC.TO', 'MULTICAP.NS',
    'MULV.TO', 'MUMC.TO', 'MUSA.TO', 'MUSQ', 'MVA.AX', 'MVB.AX', 'MVE.AX', 'MVEE.SW', 'MVPA', 'MVR.AX',
    'MVS.AX', 'MVW.AX', 'MWLV.TO', 'MWOZ.L', 'MYLD', 'MZY.NZ', 'NACP', 'NAFTRAC.MX', 'NANC', 'NANR',
    'NASA', 'NASD11.SA', 'NATO', 'NBCE', 'NBCG.TO', 'NBCR', 'NBCX.TO', 'NBDS', 'NBEM.TO', 'NBET',
    'NBGX', 'NBIE', 'NBIE.TO', 'NBIV.TO', 'NBIX.TO', 'NBJP', 'NBOV11.SA', 'NBQC.TO', 'NBSC.TO', 'NBSM',
    'NBUE.TO', 'NBUX.TO', 'NCLD', 'NDIV', 'NDIV.TO', 'NDIV11.SA', 'NDQ.AX', 'NEQT.TO', 'NETL', 'NEWZ',
    'NEXT30ADD.NS', 'NEXT50.NS', 'NEXT50ADD.NS', 'NEXT50BETA.NS', 'NEXT50IETF.NS', 'NFLP', 'NFRA', 'NGIF', 'NGPE.TO', 'NIF100IETF.NS',
    'NIFTY100EW.NS', 'NIFTYBEES.NS', 'NIFTYQLITY.NS', 'NIKL', 'NINV.TO', 'NISM', 'NITE', 'NIXT', 'NMEQ.TO', 'NMNG.TO',
    'NNRG.TO', 'NNRGF', 'NNUK.AX', 'NNWH.AX', 'NODE', 'NOEQ', 'NPBET.NS', 'NPF.NZ', 'NPFE', 'NQLT',
    'NRAM', 'NREA.TO', 'NRES', 'NRGI.TO', 'NRGY.TO', 'NRSH', 'NSCE.TO', 'NSDC.TO', 'NSDG.TO', 'NSDI.TO',
    'NSDV11.SA', 'NSGE.TO', 'NSI', 'NSIG', 'NSIV', 'NTHM.TO', 'NTSD', 'NTSX', 'NTSZ.DE', 'NUCL11.SA',
    'NUDG', 'NUDM', 'NUDV', 'NUEM', 'NUGO', 'NUKZ', 'NULC', 'NULG', 'NULV', 'NUMG',
    'NUMV', 'NURE', 'NUSC', 'NV20IETF.NS', 'NVDH.TO', 'NVIR', 'NVIT', 'NVOH', 'NVPS', 'NXTE',
    'NXTI', 'NYNY', 'NYSX.TO', 'NYYY', 'NZG.NZ', 'NZT.NZ', 'OAEM', 'OAIM', 'OAIW', 'OAKG',
    'OAKI', 'OAKM', 'OALC', 'OBOR', 'ODDS', 'ODDZ', 'OEFA', 'OEI', 'OIGS.DE', 'OILIETF.NS',
    'OILT', 'OMAH', 'ONEH', 'ONEQ.TO', 'OP2E.DE', 'OP4E.DE', 'OP5E.DE', 'OP6E.DE', 'OP7E.DE', 'OPPE',
    'OPPG', 'OPPJ', 'OPTZ', 'ORBX.TO', 'OSCV', 'OSEA', 'OTAX', 'OTGL', 'OURO11.SA', 'OVF',
    'OVL', 'OVLH', 'OVQ.SI', 'OVS', 'OWN', 'OZEM', 'OZF.AX', 'OZR.AX', 'OZXX.AX', 'OZY.NZ',
    'PALC', 'PAMC', 'PATN', 'PAYI.TO', 'PAYU.TO', 'PBDC', 'PBEU', 'PBI.TO', 'PBOG', 'PBOT',
    'PBPH', 'PCLC', 'PCLG', 'PCLN', 'PCPC', 'PCSG', 'PDC.TO', 'PDF.TO', 'PEMX', 'PEPS',
    'PEVC', 'PEVC11.SA', 'PEXL', 'PFDE', 'PFOE', 'PGA1.AX', 'PGRI', 'PGRO', 'PGRX.TO', 'PHARMABEES.NS',
    'PHE.TO', 'PHEQ', 'PHOX', 'PHR.TO', 'PIBB11.SA', 'PID.TO', 'PIEL', 'PIEQ', 'PINK', 'PINV.TO',
    'PIPE', 'PIPE11.SA', 'PIXX.AX', 'PJFG', 'PJFM', 'PJFV', 'PJIN', 'PJIO', 'PJSM', 'PJUS',
    'PLX.DE', 'POW', 'PPLN.TO', 'PPTY', 'PR1Z.DE', 'PRAJ.L', 'PRAM', 'PRAY', 'PRCS', 'PRIE.L',
    'PRMR', 'PRNT', 'PRVS', 'PRXG', 'PRXI', 'PRXV', 'PSC', 'PSET', 'PSIL', 'PSOX',
    'PSRW.L', 'PSTR', 'PSUBANK.NS', 'PSUBANKADD.NS', 'PSUBNKIETF.NS', 'PSY.TO', 'PTEU', 'PTIN', 'PTL', 'PTLC',
    'PTMC', 'PTNQ', 'PTNT', 'PVAL', 'PVTBANIETF.NS', 'PVTBANK.NS', 'PVTBANKADD.NS', 'PWER', 'PWRD', 'PWRZ',
    'PXC.TO', 'PXS.TO', 'PY', 'PZIV', 'PZLV', 'PZW.TO', 'QCE.TO', 'QCLN.TO', 'QCLR', 'QCN.TO',
    'QDEF', 'QDF', 'QDFI11.SA', 'QDPL', 'QDVO', 'QDX.TO', 'QEE.TO', 'QFF', 'QFN.AX', 'QGRO',
    'QGRW', 'QHAL.AX', 'QHSM.AX', 'QIDX', 'QINF.TO', 'QINT', 'QLBR11.SA', 'QLC', 'QLDY', 'QLTI',
    'QLTY', 'QLTY.AX', 'QLV', 'QLVD', 'QLVE', 'QMAX.AX', 'QMAX.TO', 'QMID', 'QMOM', 'QMVP.TO',
    'QNDQ.AX', 'QNTM.AX', 'QNXT', 'QOZ.AX', 'QPEU.DE', 'QQC.TO', 'QQCE.TO', 'QQCFF', 'QQEQ.TO', 'QQH',
    'QQQG', 'QQQI', 'QQQM', 'QQQQ.TO', 'QQQT', 'QQQT.TO', 'QQQY', 'QQQY.TO', 'QQWZ', 'QRE.AX',
    'QRET.TO', 'QSIX', 'QSML', 'QSML.AX', 'QTUM', 'QTUP', 'QUAL.AX', 'QUAL30IETF.NS', 'QUALITY30.NS', 'QUED.DE',
    'QUIZ', 'QUSA', 'QUU.TO', 'QVAL', 'QXM.TO', 'RAFE', 'RANK', 'RARA11.SA', 'RARI.AX', 'RAUS',
    'RBNK.TO', 'RCAN.TO', 'RCAP.AX', 'RCD.TO', 'RCGE', 'RCKT.AX', 'RDOG', 'RDTY', 'RDV.AX', 'REAI',
    'REET', 'REGS', 'REIT.AX', 'REIT.TO', 'REIX', 'REM', 'REMEF', 'REMG', 'REXC', 'REZ',
    'RFDA', 'RGEF', 'RGLO', 'RGOS.AX', 'RHRX', 'RICO11.SA', 'RID.TO', 'RIDH.TO', 'RIET', 'RIFR',
    'RIGE.TO', 'RILA', 'RING.TO', 'RINT', 'RISE', 'RIT.TO', 'RITA', 'RJCA', 'RJDI', 'RKLC.TO',
    'RKNG', 'RKSG', 'RMAX.TO', 'RMRC', 'RNIN', 'ROAM', 'ROBO', 'RODM', 'ROE', 'ROPE',
    'ROSC', 'ROUS', 'ROX.DE', 'ROYL.AX', 'RPD.TO', 'RPDH.TO', 'RQCA.TO', 'RQIN.TO', 'RQUS.TO', 'RSLGF',
    'RSMC', 'RSMV', 'RSPT', 'RUD.TO', 'RUDH.TO', 'RUDQF', 'RUNN', 'RUSC', 'RVER', 'RW',
    'RWEM', 'RWIN', 'RWLC', 'RWR', 'S3GO.AX', 'S7XE.DE', 'SAFE.TO', 'SAGP', 'SAMM', 'SAMT',
    'SAPH', 'SAUM.L', 'SAWG', 'SAWS', 'SBIBPB.NS', 'SBIETFCON.NS', 'SBIETFIT.NS', 'SBIETFPB.NS', 'SBIETFQLTY.NS', 'SBIMIDMOM.NS',
    'SBINEQWETF.NS', 'SBINMID150.NS', 'SBIO', 'SBISMLETF.NS', 'SBIVALETF.NS', 'SBLG.TO', 'SBLI.TO', 'SCAP', 'SCDV', 'SCHH',
    'SCVB11.SA', 'SDOG', 'SEA', 'SECLEDBETA.NS', 'SECT', 'SEEM', 'SEIE', 'SEIS', 'SELECTIPO.NS', 'SEMD.SW',
    'SEMG', 'SEMI', 'SENSEXADD.NS', 'SENSEXAXIS.NS', 'SENSEXBETA.NS', 'SENSEXETF.NS', 'SENSEXIETF.NS', 'SETFNIFBK.NS', 'SETM', 'SFGV',
    'SFLO', 'SFY', 'SFY.AX', 'SFYF', 'SFYI', 'SGLC', 'SGRD.TO', 'SGRT', 'SHD.SI', 'SHDG',
    'SHLD.TO', 'SHPP', 'SHTE.DE', 'SHUS', 'SIXA', 'SIXH', 'SIXL', 'SIXS', 'SK9A.DE', 'SLF.AX',
    'SLICHA.SW', 'SLQX.DE', 'SLVP', 'SMAB11.SA', 'SMAC11.SA', 'SMAL11.SA', 'SMALL250.NS', 'SMALLADD.NS', 'SMALLCAP.NS', 'SMALLGROWW.NS',
    'SMALLIETF.NS', 'SMARTRC.MX', 'SMAX.TO', 'SMCO', 'SMCP', 'SMDX', 'SMIA.SW', 'SMICHA.SW', 'SMIEX.SW', 'SMIG',
    'SMIZ', 'SML100CASE.NS', 'SMLL.AX', 'SMMCHA.SW', 'SMOM', 'SMOX', 'SMRF', 'SMRI', 'SMVP.TO', 'SNAV',
    'SNTH', 'SNTQ', 'SNXT30BEES.NS', 'SNXT50BETA.NS', 'SOVE.DE', 'SOVF', 'SOXX', 'SPCI', 'SPCK', 'SPCT',
    'SPD', 'SPDF', 'SPIA.SW', 'SPIEXT.SW', 'SPISI.SW', 'SPISID.SW', 'SPIT', 'SPPD.DE', 'SPRX', 'SPSA.DE',
    'SPTE', 'SPUB11.SA', 'SPUC', 'SPWO', 'SPXI11.SA', 'SPYA', 'SPYC', 'SPYD', 'SPYG', 'SPYG.DE',
    'SPYI11.SA', 'SPYV', 'SQLV', 'SQS', 'SRHQ', 'SROI', 'SRVR', 'SSCP', 'SSGMF', 'SSMG',
    'SSO.AX', 'SSPY', 'SSSPF', 'SSUS', 'SSXU', 'ST4R.DE', 'STHH', 'STLZ.DE', 'STNC', 'STOX',
    'STPL.TO', 'STRN', 'STTX.SW', 'STXD', 'STXE', 'STXF', 'STXG', 'STXK', 'STXV', 'STYL',
    'SUN.SI', 'SUPP', 'SURE', 'SUST.DE', 'SWCSS.SW', 'SWCSU.SW', 'SWCSW.SW', 'SWP', 'SWTZ.AX', 'SXQG',
    'SYLD', 'SYZ', 'TAOZ', 'TAX', 'TBG', 'TBNK.TO', 'TCAF', 'TCAI', 'TCCA.TO', 'TCEU.TO',
    'TCHP', 'TCLV.TO', 'TCUS.TO', 'TCV', 'TCWW.TO', 'TDAQ', 'TDEX.BK', 'TDI', 'TDNA.TO', 'TDOC.TO',
    'TDVG', 'TEC', 'TEC.TO', 'TECH.AX', 'TECH.NS', 'TECH.TO', 'TECI.TO', 'TECK11.SA', 'TECX.TO', 'TEKY',
    'TEMR', 'TEMX', 'TEQI', 'TEQT.TO', 'TESL', 'TEST', 'TEXX', 'TFGZ', 'TFNS', 'TGLB',
    'TGRT', 'TGRW', 'TGRZ', 'THE.TO', 'THEQ', 'THIR', 'THLV', 'THMZ', 'THNR', 'THU.TO',
    'TIER', 'TIH', 'TIIV', 'TILC', 'TILT', 'TILV.TO', 'TIME', 'TINS', 'TISC', 'TKCPF',
    'TLCI', 'TLF.TO', 'TLG', 'TLTD', 'TLTE', 'TLV.TO', 'TMAT', 'TMED', 'TMFC', 'TMFE',
    'TMFG', 'TMFM', 'TMFS', 'TMFX', 'TMGN', 'TMH', 'TMID', 'TMLP', 'TMSL', 'TMVE',
    'TNIDETF.NS', 'TNUK', 'TNXT', 'TNZ.NZ', 'TOGA', 'TOKN.TO', 'TOLL', 'TOLL.AX', 'TOP100CASE.NS', 'TOP10ADD.NS',
    'TOP15IETF.NS', 'TOP20.NS', 'TOS', 'TOT', 'TOUS', 'TOV', 'TPE.TO', 'TPFC', 'TPFG', 'TPHD',
    'TPIF', 'TPLC', 'TPRY', 'TPSC', 'TPU.TO', 'TPUT', 'TPUT.TO', 'TPYP', 'TQCD.TO', 'TQGD.TO',
    'TQGM.TO', 'TQID.TO', 'TQSM.TO', 'TRFK', 'TRFM', 'TRIG11.SA', 'TRND', 'TRVI.TO', 'TRVL.TO', 'TSCM',
    'TSCV', 'TSEE', 'TSEL', 'TSES', 'TSIC', 'TSLP', 'TSME', 'TSNF', 'TSPA', 'TSRS',
    'TSSD', 'TTEQ', 'TTP.TO', 'TTTX.TO', 'TULV.TO', 'TURF', 'TVAL', 'TWF.NZ', 'TWH.NZ', 'TXS',
    'TXUE', 'TXUG', 'TYYY', 'U100.AX', 'UBNK.TO', 'UCBG', 'UDA.TO', 'UDEF.TO', 'UDI', 'UDIV',
    'UDIV.TO', 'UEVM', 'UFO', 'UFOD', 'UFOX', 'UGRW.L', 'UHERO.BK', 'UIGB.L', 'UIMS.DE', 'UIQ4.DE',
    'UIVM', 'UKRE.L', 'UKSD.L', 'ULTI', 'ULVM', 'UMAX.TO', 'UMI', 'UMI.TO', 'UMMA', 'UMRT.TO',
    'UMVP.TO', 'UPGR', 'UPSD', 'URAN', 'URAN.AX', 'URND.L', 'URNJ', 'URNM', 'URNM.AX', 'US4D.DE',
    'USA.NZ', 'USAI', 'USEW', 'USF.NZ', 'USFE', 'USG.NZ', 'USH.NZ', 'USM.NZ', 'USMC', 'USMD',
    'USMF', 'USNG', 'USNZ', 'USPX', 'USS.NZ', 'USSE', 'UST.NZ', 'USV.NZ', 'USVM', 'UTES',
    'UTIL.TO', 'UTLI11.SA', 'UTLL11.SA', 'V3EA.SW', 'VA.TO', 'VAE.AX', 'VAL30IETF.NS', 'VALQ', 'VALU.DE', 'VALUEAXIS.NS',
    'VAMO', 'VAP.AX', 'VAS.AX', 'VBLD.AX', 'VCKVF', 'VCN.TO', 'VCR.MX', 'VDI', 'VDY.TO', 'VDYIF',
    'VEF.TO', 'VEGN', 'VEGX', 'VEM', 'VEQT.TO', 'VESG.AX', 'VETH.AX', 'VEU.AX', 'VFLO', 'VFVA',
    'VGAD.AX', 'VGER.SW', 'VGRO', 'VGS.AX', 'VGSR', 'VHY.AX', 'VI.TO', 'VIDI', 'VIDY.TO', 'VIG',
    'VIHY.AX', 'VISM.AX', 'VIU.TO', 'VLEU.DE', 'VLUE.AX', 'VMAX', 'VMIG.L', 'VMO.TO', 'VNCUF', 'VNGS.AX',
    'VNSE', 'VOLT', 'VOLT.AX', 'VOOY', 'VOTE', 'VOXP', 'VPX', 'VRAI', 'VRE.TO', 'VSDA',
    'VSLU', 'VSMV', 'VTEK.AX', 'VTS.AX', 'VUDV.TO', 'VUN.TO', 'VUS', 'VUSE', 'VVL.TO', 'VVO.TO',
    'VVSMF', 'VWRA11.SA', 'VXC.TO', 'VXM.TO', 'WAGN', 'WAR', 'WATS', 'WBIF', 'WBIG', 'WBIL',
    'WBIY', 'WCAP', 'WCEO', 'WCLD', 'WCMQ.AX', 'WDAF', 'WDAI', 'WDCCF', 'WDEF', 'WDGF',
    'WDNA', 'WDRN', 'WDSSF', 'WDTE', 'WDTRF', 'WEBC.DE', 'WEEUF', 'WELD', 'WGRU.DE', 'WHYP.TO',
    'WINN', 'WLDR', 'WMSE.DE', 'WNDR', 'WOMN', 'WOOD', 'WQTM', 'WR', 'WRLD', 'WRLD.AX',
    'WRLD11.SA', 'WRND', 'WSEMF', 'WSGE', 'WSMD', 'WSPC', 'WSRD.TO', 'WSRI.TO', 'WTAIF', 'WTDY.DE',
    'WTES.DE', 'WTIPF', 'WTRE', 'WTV', 'WTWE.DE', 'WUGI', 'WWJD', 'WXHG.AX', 'WXM.TO', 'WXOZ.AX',
    'WYNC.AX', 'X014.DE', 'XACTC25.CO', 'XAD.TO', 'XAIW', 'XALG.AX', 'XASG.AX', 'XASX.L', 'XB4A.DE', 'XBEE.DE',
    'XBM.TO', 'XBOV.SA', 'XC', 'XCD.TO', 'XCEM', 'XCG.TO', 'XCH.TO', 'XCHG', 'XCHP.TO', 'XCOR',
    'XCS.TO', 'XCV.TO', 'XDAT', 'XDEF', 'XDEF.DE', 'XDEV.L', 'XDIV.TO', 'XDJU.DE', 'XDJX.DE', 'XDU.TO',
    'XEG.TO', 'XEI.TO', 'XEMD', 'XEML', 'XEN.TO', 'XEQT.TO', 'XESD.DE', 'XETM.TO', 'XFN.TO', 'XGD.TO',
    'XGI.TO', 'XHC.TO', 'XHD.TO', 'XHU.TO', 'XIC.TO', 'XID.TO', 'XIDV', 'XIGV', 'XINT.TO', 'XIT.TO',
    'XIU.TO', 'XLPE.L', 'XMA.TO', 'XMA1.DE', 'XMAG', 'XMAW.SW', 'XMC.TO', 'XMD.TO', 'XMET.AX', 'XMI.TO',
    'XMM.TO', 'XMTM.TO', 'XMU.TO', 'XMV.TO', 'XMW.TO', 'XNAV', 'XNZE.DE', 'XOVR', 'XQBT', 'XQQ.TO',
    'XRE.TO', 'XSD', 'XSMH.TO', 'XSOE', 'XSPC', 'XST.TO', 'XSU.TO', 'XSXX.DE', 'XUDV', 'XUEK.L',
    'XUH.TO', 'XUSF.TO', 'XUSM', 'XUT.TO', 'XUU.TO', 'XV', 'XWNG', 'XXSC.DE', 'XZEA.DE', 'YALL',
    'YLDY', 'YMAX.AX', 'YNOT', 'YOKE', 'YUNG', 'YYY.SI', 'ZACE.TO', 'ZBK.TO', 'ZCN.TO', 'ZDH.TO',
    'ZDI.TO', 'ZDIS', 'ZDV.TO', 'ZDY.TO', 'ZEB.TO', 'ZECP', 'ZEO.TO', 'ZEQT.TO', 'ZFC.TO', 'ZFN.TO',
    'ZGD.TO', 'ZGI.TO', 'ZHU.TO', 'ZIG', 'ZIN.TO', 'ZINC', 'ZINN', 'ZIU.TO', 'ZJG.TO', 'ZJPN.TO',
    'ZLB.TO', 'ZLD.TO', 'ZLE.TO', 'ZLH.TO', 'ZLI.TO', 'ZLU.TO', 'ZMID.TO', 'ZMT.TO', 'ZNQ.TO', 'ZPAY.TO',
    'ZPD9.DE', 'ZPH.TO', 'ZPRL.DE', 'ZPW.TO', 'ZQQ.TO', 'ZRE.TO', 'ZSML.TO', 'ZUB.TO', 'ZUD.TO', 'ZUH.TO',
    'ZUT.TO', 'ZXLB.TO', 'ZXLC.TO', 'ZXLE.TO', 'ZXLF.TO', 'ZXLI.TO', 'ZXLK.TO', 'ZXLP.TO', 'ZXLR.TO', 'ZXLU.TO',
    'ZXLV.TO', 'ZXLY.TO', 'ZXM.TO', 'ZYAU.AX',
]

OUT_PATH = "data/global-etf-history.json"
LOOKBACK = "14mo"
# threads=False is the important safety knob (sequential requests, no concurrent burst that
# tends to trip Yahoo's rate limit). Batch size and sleep are tuned for speed within that
# constraint — smaller/slower is safer, bigger/faster risks 429s again.
BATCH_SIZE = 30
RETRIES = 4
SLEEP_BETWEEN_BATCHES = 2.5
SLEEP_ON_RETRY = 10.0
MAX_ROWS_PER_TICKER = 310   # ~14 months of trading days -- caps how far merging with the
                            # previous file can grow each ticker's series (see LOOKBACK).


TODAY_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _drop_future_dates(rows):
    """A handful of tickers (mostly certain non-US exchanges, e.g. APA.NZ) have come back
    from yfinance with a last bar dated a day AHEAD of this run's own UTC date -- a
    timezone-labeling quirk on Yahoo's side for those exchanges, not a real future close.
    Left in place, that bogus date wins gtmLatestDate() on the site (it's a max()), which
    skews the default 1Y view's end date for EVERY ticker to a session that doesn't really
    exist yet. Dropping anything dated past "today" (this run's own clock) at both load and
    fetch time is a cheap, safe filter -- a genuinely valid close is never dated after the
    day it's fetched on."""
    return [row for row in rows if row and row[0] <= TODAY_STR]


def load_existing():
    """The previously committed data/global-etf-history.json, if any -- merged into this
    run's fresh fetch rather than replaced by it. Missing/corrupt file just means
    starting from empty, same as the very first run ever. Also strips any bogus
    future-dated rows a past run may have already committed (see _drop_future_dates)."""
    try:
        with open(OUT_PATH) as f:
            data = json.load(f)
        return {t: _drop_future_dates(rows) for t, rows in data.items()}
    except Exception:
        return {}


def merge_series(old_rows, new_rows):
    """old_rows/new_rows: [["YYYY-MM-DD", price], ...]. Reads old_rows by index (row[0],
    row[1]) rather than unpacking exactly 2 values -- a briefly-deployed earlier version of
    this script committed some 3-element [date, close, open] rows, and old_rows is whatever
    is already sitting in the committed JSON, so this has to tolerate that leftover shape
    (and any other stray extra fields) without crashing, discarding anything past the close.
    New rows win on a shared date (this run's data is the freshest), but a date present in
    old_rows and absent from new_rows -- this ticker's exchange hasn't closed yet this run,
    or Yahoo had a transient gap -- is kept instead of silently dropped. Trimmed to the most
    recent MAX_ROWS_PER_TICKER dates afterward."""
    merged = {row[0]: row[1] for row in old_rows}
    for row in new_rows:
        merged[row[0]] = row[1]
    dates = sorted(merged.keys())[-MAX_ROWS_PER_TICKER:]
    return [[d, merged[d]] for d in dates]


def fetch_batch(tickers):
    out = {}
    raw = None
    for attempt in range(RETRIES):
        try:
            raw = yf.download(tickers, period=LOOKBACK, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=False, progress=False,
                               session=_SESSION)
            if raw is not None and len(raw) > 0:
                break
        except Exception as e:
            print(f"  batch download attempt {attempt+1} failed: {e}")
            raw = None
        time.sleep(SLEEP_ON_RETRY * (attempt + 1))
    if raw is None or len(raw) == 0:
        return out
    single = len(tickers) == 1
    for t in tickers:
        try:
            s = raw["Close"] if single else raw[t]["Close"]
            s = s.dropna()
            if len(s) > 0:
                rows = [[d.strftime("%Y-%m-%d"), round(float(c), 4)] for d, c in s.items()]
                rows = _drop_future_dates(rows)
                if rows:
                    out[t] = rows
        except Exception:
            continue
    return out


def main():
    existing = load_existing()
    result = dict(existing)  # start from what's already committed, not from scratch --
                              # see the merge_series note above for why this matters even
                              # more here than in fetch_etf_data.py.
    missing = []
    gap_new = []
    n_batches = (len(TICKERS) - 1) // BATCH_SIZE + 1
    for i in range(0, len(TICKERS), BATCH_SIZE):
        batch = TICKERS[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        got = fetch_batch(batch)
        for t, rows in got.items():
            old_dates = {row[0] for row in existing.get(t, [])}
            merged = merge_series(existing.get(t, []), rows)
            new_dates = {row[0] for row in merged} - old_dates
            if new_dates:
                gap_new.append((t, sorted(new_dates)))
            result[t] = merged
        missing.extend([t for t in batch if t not in got])
        print(f"Batch {batch_num}/{n_batches}: got {len(got)}/{len(batch)} — running total {len(result)}")
        # Save progress after every batch so a mid-run failure/timeout still leaves
        # whatever was fetched so far, instead of losing the whole run's data.
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, separators=(",", ":"))
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\nWrote {OUT_PATH}: {len(result)}/{len(TICKERS)} tickers.")
    if missing:
        print(f"Missing entirely this run ({len(missing)}): {missing[:30]}{'...' if len(missing) > 30 else ''}")
    if gap_new:
        print(f"Newly added/backfilled dates this run (merged in, not overwritten):")
        for t, dates in gap_new:
            print(f"  {t}: {dates}")


if __name__ == "__main__":
    main()
