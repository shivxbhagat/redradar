#!/usr/bin/env python3
"""
RedRadar (Discord alerts; multiline per ticker)

- Pulls NASDAQ-100 (QQQ) and/or S&P 500 (SPY) constituents from issuer files
- Computes % and $ change over 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y (adjusted closes)
- Sends a bold title; each ticker is bold on its own line, followed by
  newline-separated timeframe lines (with commas after each line except last) to 
  Discord

Env:
  DISCORD_WEBHOOK_URL=...   (required to post)
  INDICES=NDX               (NDX | SPX | BOTH; default NDX)
  THRESHOLD=-4              (trigger if any timeframe <= threshold)
  INTRADAY_1D=true          (true=30m intraday bars; false=last 2 closes)
  PREPOST=false             (include extended hours when INTRADAY_1D=true)
  TIMEFRAMES=1d,5d,1mo,3mo,6mo,1y,5y
  LIMIT_TICKERS=0           (limit for quick tests; 0 = all)
  BATCH_SIZE=80
"""

import os, io, re, time
from typing import List, Dict, Tuple
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

# ----------------- Config -----------------
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
INDICES = os.getenv("INDICES", "NDX").upper()
THRESHOLD = float(os.getenv("THRESHOLD", "-4"))
INTRADAY_1D = os.getenv("INTRADAY_1D", "true").lower() == "true"
PREPOST = os.getenv("PREPOST", "false").lower() == "true"
TIMEFRAMES = [s.strip() for s in os.getenv("TIMEFRAMES", "1d,5d,1mo,3mo,6mo,1y,5y").split(",") if s.strip()]
LIMIT_TICKERS = int(os.getenv("LIMIT_TICKERS", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "80"))

# ------------- Helpers --------------
def to_yahoo_symbol(t: str) -> str:
    return t.strip().replace(".", "-")  # BRK.B -> BRK-B

def clean_symbols(series: pd.Series) -> List[str]:
    s = series.astype(str).str.strip()
    s = s[~s.str.upper().isin({"", "NAN", "NONE", "CASH", "CASH_USD", "USD"})]
    pat = re.compile(r"^(?=.*[A-Za-z0-9])[A-Za-z0-9.\-]{1,10}$")
    s = s[s.apply(lambda x: bool(pat.fullmatch(x)))]
    syms = [to_yahoo_symbol(x) for x in s]
    return [t for t in syms if any(c.isalnum() for c in t)]

def chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# -------- Constituents (issuer files) -----
def load_ndx() -> List[str]:
    url = ("https://www.invesco.com/us/financial-products/etfs/holdings/"
           "main/holdings/0?action=download&audienceType=Investor&ticker=QQQ")
    r = requests.get(url, timeout=30); r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), header=None)

    if "Ticker" in raw.iloc[0].astype(str).values:
        hdr = raw.apply(lambda row: (row.astype(str) == "Ticker").any(), axis=1).idxmax()
        df = pd.read_csv(io.StringIO(r.text), header=hdr)
        tickers = clean_symbols(df.get("Ticker", pd.Series(dtype=str)))
    else:
        col = 2 if raw.shape[1] >= 3 else 0
        tickers = clean_symbols(raw.iloc[:, col])

    return sorted(set(tickers))

def load_spx() -> List[str]:
    url = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    r = requests.get(url, timeout=30); r.raise_for_status()
    with io.BytesIO(r.content) as f:
        raw = pd.read_excel(f, sheet_name=0, header=None, engine="openpyxl")

    header_idx = None
    for i in range(min(30, len(raw))):
        if "TICKER" in raw.iloc[i].astype(str).str.upper().tolist():
            header_idx = i; break

    with io.BytesIO(r.content) as f:
        df = pd.read_excel(f, sheet_name=0, header=header_idx, engine="openpyxl") if header_idx is not None \
             else pd.read_excel(f, sheet_name=0, engine="openpyxl")

    cols = [str(c).strip() for c in df.columns]
    tcol = None
    for cand in ("Ticker", "Ticker Symbol", "Identifier"):
        if cand in cols: tcol = cand; break
    if tcol is None:
        best, best_score = None, 0
        for c in cols:
            score = pd.Series(df[c]).astype(str).str.fullmatch(r"[A-Za-z0-9.\-]{1,10}").mean()
            if score > best_score: best, best_score = c, score
        tcol = best

    tickers = clean_symbols(df[tcol])
    tickers = [t for t in tickers if "CASH" not in t and "USD" not in t]
    return sorted(set(tickers))

def fetch_index_tickers() -> List[str]:
    tickers: List[str] = []
    if INDICES in ("NDX", "BOTH"): tickers += load_ndx()
    if INDICES in ("SPX", "BOTH"): tickers += load_spx()
    tickers = sorted(set(tickers))
    if LIMIT_TICKERS > 0:
        tickers = tickers[:LIMIT_TICKERS]
    return tickers

# -------- Percent & $ change math ---------
def series_change(s: pd.Series):
    s = s.dropna()
    if len(s) < 2: return None
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    if start == 0: return None
    pct = (end - start) / start * 100.0
    delta = end - start
    return pct, delta

def compute_changes(tickers: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Returns: { 'AAPL': {'1d': (pct, delta), '5d': (pct, delta), ...}, ... }
    """
    out: Dict[str, Dict[str, Tuple[float, float]]] = {t: {} for t in tickers}

    # 1d window
    if "1d" in TIMEFRAMES:
        for batch in chunks(tickers, BATCH_SIZE):
            if INTRADAY_1D:
                data = yf.download(batch, period="1d", interval="30m",
                                   auto_adjust=True, group_by="ticker",
                                   progress=False, threads=True, prepost=PREPOST)
            else:
                data = yf.download(batch, period="2d", interval="1d",
                                   auto_adjust=True, group_by="ticker",
                                   progress=False, threads=True)
            for t in batch:
                try:
                    close = data[t]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
                except Exception:
                    close = pd.Series(dtype=float)
                ch = series_change(close)
                if ch: out[t]["1d"] = ch

    # Multi-day windows (daily bars)
    period_map = {"5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y"}
    for tf, period in period_map.items():
        if tf not in TIMEFRAMES: continue
        for batch in chunks(tickers, BATCH_SIZE):
            data = yf.download(batch, period=period, interval="1d",
                               auto_adjust=True, group_by="ticker",
                               progress=False, threads=True)
            for t in batch:
                try:
                    close = data[t]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
                except Exception:
                    close = pd.Series(dtype=float)
                ch = series_change(close)
                if ch: out[t][tf] = ch

    return out

# --------- Message building/sending --------
def fmt_money(delta: float) -> str:
    sign = "-" if delta < 0 else "+"
    return f"({sign}${abs(delta):.2f})"

def build_blocks(results: Dict[str, Dict[str, Tuple[float, float]]], threshold: float) -> List[str]:
    """
    Per-ticker blocks with each timeframe on its own line, comma after each line except the last.

    AAPL
    6mo -7.7% (-$18.94)

    ABNB
    1mo -9.5% (-$13.11),
    6mo -14.5% (-$21.11),
    5y -13.6% (-$19.75)
    """
    blocks: List[str] = []
    for t in sorted(results.keys()):
        entries = []
        for tf in TIMEFRAMES: 
            if tf in results[t]:
                pct, delta = results[t][tf]
                if pct <= threshold:
                    entries.append(f"{tf} {pct:.1f}% {fmt_money(delta)}")
        if entries:
            lines = ""
            for i, e in enumerate(entries):
                trailing = "," if i < len(entries) - 1 else ""
                lines += f"{e}{trailing}\n"
            blocks.append(f"**{t}**\n{lines}".rstrip())  # ticker on its own line (no bold)
    return blocks


def chunk_for_discord(title: str, blocks: List[str], limit: int = 1900) -> List[str]:
    """
    Discord ~2000 char limit without splitting a ticker block.
    Keeps header/title + date like before, then a separator line.
    Inserts a blank line between ticker blocks.
    Date zone: America/Winnipeg (CST/CDT).
    """
    now_ct = datetime.now(tz=ZoneInfo("America/Winnipeg"))
    date_str = f"{now_ct.strftime('%b')} {now_ct.day}, {now_ct.year}"

    sep = "**=================================**\n"
    header_main = f"**📉 {title} ~ {date_str}**\n{sep}"
    footer = "**=================================**"

    parts: List[str] = []
    current = header_main

    for block in blocks:
        addition = block + "\n\n"  # blank line between tickers
        if len(current) + len(addition) + len(footer) > limit:
            current += footer
            parts.append(current)
            current = addition
        else:
            current += addition

    current += footer
    parts.append(current)
    return parts


def post_to_discord(content_parts: List[str]):
    if not WEBHOOK:
        print("[WARN] No DISCORD_WEBHOOK_URL set. Preview below:\n")
        for i, p in enumerate(content_parts, 1):
            print(f"--- PART {i}/{len(content_parts)} ---\n{p}\n")
        return
    for part in content_parts:
        r = requests.post(WEBHOOK, json={"content": part}, timeout=30)
        if r.status_code >= 400:
            print(f"[ERROR] Discord {r.status_code}: {r.text}")
        time.sleep(0.35)

# ------------------- Main ------------------
def main():
    tickers = fetch_index_tickers()
    ix_name = {"NDX": "NASDAQ-100", "SPX": "S&P 500", "BOTH": "NASDAQ-100 + S&P 500"}.get(INDICES, "Indices")


    results = compute_changes(tickers)
    blocks = build_blocks(results, THRESHOLD)

    now_ct = datetime.now(tz=ZoneInfo("America/Winnipeg"))
    now_str = f"{now_ct.strftime('%b')} {now_ct.day}, {now_ct.year}"

    if not blocks:
        msg = f"✅ No tickers breached the {THRESHOLD:.1f}% threshold for {now_str}"
        post_to_discord([msg])
        print(msg)
        return

    title = f"{ix_name} drop alerts (<= {THRESHOLD:.1f}%)"
    parts = chunk_for_discord(title, blocks)
    post_to_discord(parts)
    print(f"[INFO] Posted {len(parts)} message part(s).")
    print("============================================\n")
    print("Results:\n")
    print(results)
    print("\n============================================\n")



if __name__ == "__main__":
    main()
