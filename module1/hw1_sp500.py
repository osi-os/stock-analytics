# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: llmzoomcamp-2026 (3.12.12.final.0)
#     language: python
#     name: python3
# ---

# %%
print('test')

# %%
import sys
# !{sys.executable} -m ensurepip --upgrade

# %%
# !{sys.executable} -m pip install lxml

# %%
import sys
# !{sys.executable} -m pip install yfinance

# %%
import io
import requests
import pandas as pd
import numpy as np
import lxml
import yfinance as yf

# %%
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


# %%
def fetch_sp500() -> pd.DataFrame:
    """Return a DataFrame of ticker, name, and year added."""
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
 
    # The first table on the page is the constituents list.
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
 
    # Column names as they appear on the page (as of 2025/2026).
    # Symbol | Security | GICS Sector | ... | Date added | CIK | Founded
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "Date added": "date_added",
    })
 
    df = df[["ticker", "name", "date_added"]].copy()
 
    # "Date added" is usually YYYY-MM-DD but a few rows carry footnotes or
    # blanks. Parse leniently and pull out the year.
    parsed = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = parsed.dt.year
 
    return df



# %%
df = fetch_sp500()
df.head()


# %%
df["year_added"].isna().sum()   # how many dates didn't parse
 

# %%
def additions_per_year(df: pd.DataFrame, start_year: int = 2020) -> pd.Series:
    """Count additions per year, starting from `start_year`."""
    valid = df.dropna(subset=["year_added"]).copy()
    valid["year_added"] = valid["year_added"].astype(int)
    recent = valid[valid["year_added"] >= start_year]
    return recent["year_added"].value_counts().sort_index()


# %%
counts = additions_per_year(df, start_year=2020)
counts

# %% [markdown]
# 2025 had the highest number of additions to the S&P 500

# %%
# label -> Yahoo ticker
INDEXES = {
    "US - S&P 500":            "^GSPC",
    "China - Shanghai Comp":   "000001.SS",
    "Hong Kong - Hang Seng":   "^HSI",
    "Australia - ASX 200":     "^AXJO",
    "India - Nifty 50":        "^NSEI",
    "Canada - TSX Composite":  "^GSPTSE",
    "Germany - DAX":           "^GDAXI",
    "UK - FTSE 100":           "^FTSE",
    "Japan - Nikkei 225":      "^N225",
    "Mexico - IPC Mexico":     "^MXX",
    "Brazil - Ibovespa":       "^BVSP",
}
 
START = "2026-01-01"
LAST_DAY = "2026-08-21"
# yfinance's `end` is EXCLUSIVE, so to include the 21 Aug close we request
# through the 22nd, then slice to the last trading day on/before the 21st.
END = "2026-08-22"


# %%
df = yf.download("^GSPC", start="2026-01-01", end="2026-08-22",
                 progress=False, auto_adjust=True)
print(df.shape)
print("first:", df.index.min().date(), " last:", df.index.max().date())
df["Close"].tail(3)


# %%
def ytd_return(ticker):
    df = yf.download(ticker, start="2026-01-01", end="2026-08-22",
                     progress=False, auto_adjust=True)
    if df.empty:
        return float("nan")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.loc[:"2026-08-21"].dropna()
    if len(close) < 2:
        return float("nan")
    return (close.iloc[-1] / close.iloc[0] - 1) * 100


# %%
print("US:", ytd_return("^GSPC"))
print("Japan:", ytd_return("^N225"))

# %%
results = {label: ytd_return(tkr) for label, tkr in INDEXES.items()}
s = pd.Series(results, name="YTD %").sort_values(ascending=False)
s.round(2)

# %%
us = s["US - S&P 500"]
better = s[s > us].drop("US - S&P 500", errors="ignore")
print(f"US (S&P 500) YTD: {us:.2f}%")
print(f"Indexes beating the US: {len(better)} out of {len(INDEXES) - 1}")
print(list(better.index))

# %% [markdown]
# US (S&P 500) YTD: 11.90% |
# Indexes beating the US: 2 out of 10 - Japan & Canada

# %%
df = yf.download("^GSPC", start="1950-01-01", progress=False, auto_adjust=True)
print(df.shape)
print("range:", df.index.min().date(), "→", df.index.max().date())
df.tail(3)

# %%
close = df["Close"]
if isinstance(close, pd.DataFrame):   # single-ticker download can be 2-D
    close = close.iloc[:, 0]
close = close.dropna()
close.tail(3)

# %%
running_max = close.cummax()
is_ath = close == running_max          # a day at (or above) every prior close
ath_positions = np.flatnonzero(is_ath.values)
print("all-time-high days:", len(ath_positions))
close[is_ath].tail(3)

# %%
prices = close.values
dates = close.index
records = []

for a, b in zip(ath_positions[:-1], ath_positions[1:]):
    segment = prices[a:b + 1]              # peak → recovery, inclusive
    peak = prices[a]
    trough = segment.min()
    drawdown = (peak - trough) / peak * 100
    duration = (dates[b] - dates[a]).days  # calendar days, peak to recovery
    records.append({
        "peak_date": dates[a],
        "recovery_date": dates[b],
        "peak": peak,
        "trough": trough,
        "drawdown_pct": drawdown,
        "duration_days": duration,
    })

corr = pd.DataFrame(records)
print(len(corr), "segments between consecutive highs")
corr.head()

# %%
corrections = corr[corr["drawdown_pct"] >= 5].copy()
print(len(corrections), "corrections of at least 5%")
corrections.sort_values("drawdown_pct", ascending=False).head(10)

# %%
q = [0.25, 0.50, 0.75]
summary = pd.DataFrame({
    "drawdown_pct": corrections["drawdown_pct"].quantile(q),
    "duration_days": corrections["duration_days"].quantile(q),
})
summary.index = ["25th", "50th (median)", "75th"]
print(summary.round(2))
print(f"\nMedian drawdown: {corrections['drawdown_pct'].median():.2f}%")

# %% [markdown]
#                drawdown_pct  duration_days
# 25th                   6.23           55.5
# 50th (median)          7.99           92.5
# 75th                  14.02          211.5
#
# Median drawdown: 7.99%

# %%
ticker = "AMZN"
ticker_obj = yf.Ticker(ticker)

# default limit is 12; ask for more so we get the full history back to 2020
earnings = ticker_obj.get_earnings_dates(limit=60)
print(earnings.shape)
earnings.head()

# %%
print(earnings.columns.tolist())

# the surprise column name varies slightly across versions ("Surprise(%)")
surprise_col = [c for c in earnings.columns if "Surprise" in c][0]
print("surprise column:", surprise_col)

# the upcoming earnings date has no Reported EPS / Surprise yet — drop it
earnings = earnings.dropna(subset=["Reported EPS"])
print("reported events:", len(earnings))
earnings[[surprise_col, "Reported EPS", "EPS Estimate"]].head()

# %%
px = yf.download(ticker, start="2020-01-01", progress=False, auto_adjust=True)
close = px["Close"]
if isinstance(close, pd.DataFrame):     # single-ticker download can be 2-D
    close = close.iloc[:, 0]
close = close.dropna()
print(close.shape, close.index.min().date(), "→", close.index.max().date())

# %%
# For each Day2, return = Close_Day3 / Close_Day1 - 1
#   Day1 = previous trading day, Day3 = next trading day
ret2 = close.shift(-1) / close.shift(1) - 1
ret2.name = "ret2"
ret2.dropna().head()

# %%
# earnings index is tz-aware with a time-of-day; normalize to a naive date
edates = earnings.copy()
edates.index = pd.to_datetime(edates.index)
try:
    edates.index = edates.index.tz_localize(None)   # drop tz if present
except TypeError:
    pass
edates.index = edates.index.normalize()

r = ret2.copy()
r.index = r.index.normalize()

edates["ret2"] = r.reindex(edates.index).values
print("matched:", edates["ret2"].notna().sum(), "of", len(edates))
edates[[surprise_col, "ret2"]].sort_index()

# %%
# The task expects 25 entries starting 2020-10-29. Our pull went back to 2002,
# so two pre-Oct-2020 events slipped in. Restrict to the intended window.
edates = edates[edates.index >= pd.Timestamp("2020-10-29")]
print("events in scope:", len(edates))          # expect ~24 reported
print("range:", edates.index.min().date(), "→", edates.index.max().date())

# %%
pos = edates[edates[surprise_col] > 0].dropna(subset=["ret2"])
print(f"positive-surprise events: {len(pos)}")
print(f"median 2-day return after positive surprises: {pos['ret2'].median() * 100:.2f}%")
pos[[surprise_col, "ret2"]].sort_index()

# %%
both = edates.dropna(subset=["ret2", surprise_col])
corr_matrix = both[[surprise_col, "ret2"]].corr()
print(corr_matrix.round(3))
print(f"\ncorrelation (all events): {corr_matrix.loc[surprise_col, 'ret2']:.3f}")

# %%
# Broad-market regime: S&P 500 above/below its 200-day moving average.
spx = yf.download("^GSPC", start="2019-06-01", progress=False, auto_adjust=True)
spx_close = spx["Close"]
if isinstance(spx_close, pd.DataFrame):
    spx_close = spx_close.iloc[:, 0]

sma200 = spx_close.rolling(200).mean()
regime = pd.Series(np.where(spx_close > sma200, "bull", "bear"),
                   index=spx_close.index)
regime.index = regime.index.normalize()
regime.tail(3)

# %%
# most recent regime reading on or before each earnings date
reg_aligned = regime.reindex(edates.index, method="ffill")
edates = edates.assign(regime=reg_aligned.values)
edates[[surprise_col, "ret2", "regime"]]

# %%
summary = edates.dropna(subset=["ret2"]).groupby("regime").agg(
    n=("ret2", "size"),
    median_ret2=("ret2", "median"),
    mean_ret2=("ret2", "mean"),
    corr_surprise=(surprise_col, lambda x: x.corr(edates.loc[x.index, "ret2"])),
)
print(summary.round(4))

# %%
