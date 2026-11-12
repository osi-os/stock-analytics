"""Replay stock backtest trades as option trades, using Massive minute prices.

Takes the trades CSV from the stock backtester (entry/exit times decided by the
stock hitting the scanner's stop or target, or the end of day) and, for each one:

1. Buys a call for a long trade or a put for a short one, at the nearest expiry
   listed that day, 3rd listed strike out of the money from the stock entry price.
2. Entry: the option's first traded price at or after the stock entry time
   (within `max_entry_gap` minutes; otherwise the trade is skipped).
3. Exit: the option's volume-weighted price during the stock's exit bar
   (the 5 minutes in which the stock hit its level), or the last trade before
   the end of that bar if nothing traded in it.
4. Size: the picker's rules (estimated loss at the stock stop <= 10% of the
   account, premium <= 25% of the account, at least one contract), with the
   implied volatility solved from the entry price.

Limits, stated plainly: the free plan has trades only, no bid/ask quotes, so
fills are approximate. `slippage` (dollars per share, each side) is a stand-in
for the spread; results are reported at several values. Each trade is sized off
the starting account; P&L is not compounded and settled cash is not modeled.

CLI:
    python -m daytrade.option_backtest TRADES_CSV OUT_CSV
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import timedelta

import pandas as pd

from daytrade import massive
from daytrade.options import (ACCOUNT_SIZE, HOLD_MINUTES, MAX_PREMIUM_FRACTION,
                              RISK_FRACTION, STRIKES_OUT, pick_strike, plan_option_trade)

SLIPPAGES = (0.0, 0.02, 0.05)


def _entry_price(bars: pd.DataFrame, t: pd.Timestamp, max_gap: int) -> tuple[float, pd.Timestamp] | None:
    after = bars[(bars.index >= t) & (bars.index < t + timedelta(minutes=max_gap))]
    if after.empty:
        return None
    return float(after["open"].iloc[0]), after.index[0]


def _exit_price(bars: pd.DataFrame, t: pd.Timestamp, bar_minutes: int = 5) -> tuple[float, str] | None:
    end = t + timedelta(minutes=bar_minutes)
    win = bars[(bars.index >= t) & (bars.index < end)]
    if not win.empty and win["volume"].sum() > 0:
        return float((win["vwap"] * win["volume"]).sum() / win["volume"].sum()), "in_bar"
    before = bars[bars.index < end]
    if before.empty:
        return None
    return float(before["close"].iloc[-1]), "last_trade_before"


def replay_trade(tr: dict, contract_cache: dict, max_entry_gap: int = 5, **plan_kw) -> dict:
    side = tr["side"]
    kind = "call" if side == "long" else "put"
    t_in = pd.Timestamp(tr["entry_time"])
    t_out = pd.Timestamp(tr["exit_time"])
    day = t_in.date()
    entry_stock = float(tr["entry"])
    base = dict(ticker=tr["ticker"], setup=tr["setup"], side=side, kind=kind,
                entry_time=tr["entry_time"], exit_time=tr["exit_time"],
                exit_reason=tr["exit_reason"], stock_entry=entry_stock,
                stock_stop=float(tr["stop"]), stock_target=float(tr["target"]),
                stock_r=float(tr["r_multiple"]))

    ck = (tr["ticker"], kind, day)
    if ck not in contract_cache:
        contract_cache[ck] = massive.contracts(tr["ticker"], kind, day, entry_stock)
    cdf = contract_cache[ck]
    if cdf.empty:
        return {**base, "status": "no contracts listed"}
    expiry = min(e for e in cdf["expiration_date"] if e >= day)
    chain = cdf[cdf["expiration_date"] == expiry]
    try:
        strike = pick_strike(chain["strike_price"], entry_stock, kind, STRIKES_OUT)
    except ValueError as e:
        return {**base, "status": f"strike: {e}"}
    contract = chain.loc[chain["strike_price"] == strike, "ticker"].iloc[0]
    base.update(expiry=expiry.isoformat(), dte=(expiry - day).days, strike=strike, contract=contract)

    bars = massive.minute_bars(contract, day)
    if bars.empty:
        return {**base, "status": "no option trades that day"}
    got_in = _entry_price(bars, t_in, max_entry_gap)
    if got_in is None:
        return {**base, "status": f"no option trade within {max_entry_gap} min of entry"}
    opt_in, opt_in_time = got_in
    got_out = _exit_price(bars, t_out)
    if got_out is None:
        return {**base, "status": "no option trade by exit"}
    opt_out, exit_how = got_out

    # Size with the picker's rules, at the price actually paid.
    plan_chain = pd.DataFrame({"strike": chain["strike_price"].astype(float)})
    plan_chain["bid"] = plan_chain["ask"] = float("nan")
    plan_chain.loc[plan_chain["strike"] == strike, ["bid", "ask"]] = opt_in
    plan = plan_option_trade(tr["ticker"], side, entry_stock, float(tr["stop"]), float(tr["target"]),
                             plan_chain, expiry, opt_in_time.to_pydatetime(), **plan_kw)

    row = {**base, "status": "ok", "option_entry_time": opt_in_time.isoformat(),
           "option_entry": opt_in, "option_exit": opt_out, "exit_fill": exit_how,
           "iv_at_entry": plan.iv, "contracts": plan.contracts,
           "est_loss_at_stop": plan.est_loss_at_stop, "premium_paid": plan.premium_paid,
           "entry_volume_day": int(bars["volume"].sum())}
    for s in SLIPPAGES:
        buy, sell = opt_in + s, max(opt_out - s, 0.0)
        pnl = (sell - buy) * 100 * plan.contracts
        row[f"pnl_slip{s:.2f}"] = round(pnl, 2)
        row[f"pnl1_slip{s:.2f}"] = round((sell - buy) * 100, 2)  # one contract
        row[f"r_slip{s:.2f}"] = round(pnl / plan.est_loss_at_stop, 3) if plan.est_loss_at_stop else None
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    rows = []
    groups = [("all", ok)] + [(f"{k[0]} {k[1]}", g) for k, g in ok.groupby(["setup", "side"])]
    for name, g in groups:
        r = {"group": name, "trades": len(g)}
        for s in SLIPPAGES:
            col = f"pnl_slip{s:.2f}"
            r[f"win_rate@{s:.2f}"] = round(float((g[col] > 0).mean()), 3)
            r[f"total_$@{s:.2f}"] = round(float(g[col].sum()), 2)
            r[f"avg_R@{s:.2f}"] = round(float(g[f"r_slip{s:.2f}"].mean()), 3)
        r["stock_avg_R"] = round(float(g["stock_r"].mean()), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trades_csv")
    p.add_argument("out_csv")
    p.add_argument("--limit", type=int, help="only the first N trades (for a quick test)")
    a = p.parse_args(argv)

    trades = pd.read_csv(a.trades_csv)
    if a.limit:
        trades = trades.head(a.limit)
    cache: dict = {}
    rows = []
    for i, tr in enumerate(trades.to_dict("records"), 1):
        try:
            rows.append(replay_trade(tr, cache))
        except Exception as e:  # keep going; record why
            rows.append({"ticker": tr["ticker"], "entry_time": tr["entry_time"], "status": f"error: {e}"})
        print(f"{i}/{len(trades)} {rows[-1].get('ticker')} {rows[-1].get('entry_time')} {rows[-1]['status']}",
              flush=True)
        pd.DataFrame(rows).to_csv(a.out_csv, index=False)  # save as we go
    df = pd.DataFrame(rows)
    summ = summarize(df)
    summ.to_csv(a.out_csv.replace(".csv", "_summary.csv"), index=False)
    print(df["status"].value_counts().to_string())
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
