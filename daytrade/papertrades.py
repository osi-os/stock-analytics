"""Log Webull paper option trades and check them against the picker's plan.

Each row is one closed trade. Fill it in by hand (or from a Webull export once
its column format is known). Prices are per share, as Webull quotes them; the
code multiplies by 100.

Columns:
    date, ticker, kind (call/put), expiry, strike, contracts,
    entry_premium, exit_premium,            # what you actually paid / got
    stock_entry, stock_stop, stock_target,  # the scanner levels you traded
    exit_reason,                            # stop / target / time / manual
    plan_est_loss_at_stop,                  # from the picker, total $ for all contracts
    notes

The summary answers two questions:
1. How are the trades doing, in dollars and in R (P&L / planned loss at the stop)?
2. Is the picker's loss-at-stop estimate any good? For trades that exited at
   the stop, it compares the real loss with the estimate.

CLI:
    python -m daytrade.papertrades template paper_trades.csv
    python -m daytrade.papertrades summary paper_trades.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = [
    "date", "ticker", "kind", "expiry", "strike", "contracts",
    "entry_premium", "exit_premium",
    "stock_entry", "stock_stop", "stock_target",
    "exit_reason", "plan_est_loss_at_stop", "notes",
]
EXIT_REASONS = {"stop", "target", "time", "manual"}


def write_template(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; not overwriting it")
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns and c != "notes"]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["kind"] = df["kind"].str.lower()
    df["exit_reason"] = df["exit_reason"].str.lower()
    bad = set(df["exit_reason"].dropna()) - EXIT_REASONS
    if bad:
        raise ValueError(f"unknown exit_reason values: {sorted(bad)}")
    df["pnl"] = (df["exit_premium"] - df["entry_premium"]) * 100 * df["contracts"]
    df["premium_paid"] = df["entry_premium"] * 100 * df["contracts"]
    df["r"] = df["pnl"] / df["plan_est_loss_at_stop"]
    return df


def summary(df: pd.DataFrame) -> dict:
    out = {
        "trades": len(df),
        "wins": int((df["pnl"] > 0).sum()),
        "win_rate": round(float((df["pnl"] > 0).mean()), 3) if len(df) else None,
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "avg_pnl": round(float(df["pnl"].mean()), 2) if len(df) else None,
        "avg_r": round(float(df["r"].mean()), 3) if df["r"].notna().any() else None,
        "by_exit": df.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"]).round(2).to_dict("index"),
    }
    stops = df[(df["exit_reason"] == "stop") & df["plan_est_loss_at_stop"].notna()]
    if len(stops):
        real = -stops["pnl"]
        ratio = real / stops["plan_est_loss_at_stop"]
        out["stop_estimate_check"] = {
            "stopped_trades": len(stops),
            "real_loss_vs_estimate_median": round(float(ratio.median()), 2),
            "worst_ratio": round(float(ratio.max()), 2),
            "times_real_loss_over_estimate": int((ratio > 1).sum()),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["template", "summary"])
    p.add_argument("path")
    a = p.parse_args(argv)
    if a.command == "template":
        print(f"wrote {write_template(a.path)}")
        return
    df = load(a.path)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df[["date", "ticker", "kind", "strike", "contracts", "entry_premium", "exit_premium",
                  "exit_reason", "pnl", "r"]].to_string(index=False))
    print()
    for k, v in summary(df).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
