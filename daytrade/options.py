"""Turn a stock signal from the scanner into an option trade plan.

Rules (set by the trader on 2026-11-09):
- Long signal buys a call, short signal buys a put.
- Expiry: the nearest listed expiry on or after the signal date.
- Strike: the 3rd listed strike past the stock price (calls above, puts below).
- Exit on the STOCK levels: sell when the stock hits the scanner's stop or target.
- Risk: 10% of the account (default $1,074 -> $107.40) counts the estimated loss
  on the option when the stock hits the stop, not the premium paid. The 3rd-strike
  contract is kept even when its premium is above the cap; at least one contract
  is planned, and the plan is flagged when that one contract's estimated loss is
  over the cap.

The loss at the stop is an ESTIMATE: the option is repriced with Black-Scholes at
the stop price, at the same implied volatility, after `hold_minutes` of time decay,
and sold at the bid (model price minus half of today's bid-ask spread). Gaps,
volatility changes and wider spreads can make the real loss bigger, so the plan
also reports the whole premium at risk.

Chains come from Yahoo Finance (yfinance) for now. `pick_contract` and
`plan_option_trade` take a plain DataFrame, so another source (such as Massive)
can be plugged in later.

CLI:
    python -m daytrade.options NVDA long 178.50 177.70 180.10
    python -m daytrade.options --signals scanner_signals.csv --last 5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
ACCOUNT_SIZE = 1074.0
RISK_FRACTION = 0.10
# Most premium one trade may spend, as a share of the account (None = no cap).
# At least one contract is always kept. 25% chosen by the trader on 2026-11-10.
MAX_PREMIUM_FRACTION: float | None = 0.25
STRIKES_OUT = 3
# Median time from entry to stop in the Oct 15 - Nov 9 stock backtest (5 bars of 5m).
HOLD_MINUTES = 25
RISK_FREE_RATE = 0.04
MINUTES_PER_YEAR = 365 * 24 * 60


# --- Black-Scholes -------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, years: float, vol: float, kind: str,
             rate: float = RISK_FREE_RATE) -> float:
    """European option value. At or after expiry it returns intrinsic value."""
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    if years <= 0 or vol <= 0:
        return intrinsic
    sq = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / sq
    d2 = d1 - sq
    disc = math.exp(-rate * years)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, years: float, vol: float, kind: str,
             rate: float = RISK_FREE_RATE) -> float:
    if years <= 0 or vol <= 0:
        itm = spot > strike if kind == "call" else spot < strike
        return (1.0 if itm else 0.0) * (1 if kind == "call" else -1)
    sq = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / sq
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def implied_vol(price: float, spot: float, strike: float, years: float, kind: str,
                rate: float = RISK_FREE_RATE) -> float | None:
    """Solve for volatility by bisection. None if the price is outside model bounds."""
    if years <= 0 or price <= 0:
        return None
    lo, hi = 1e-4, 10.0
    if not (bs_price(spot, strike, years, lo, kind, rate) <= price <= bs_price(spot, strike, years, hi, kind, rate)):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, years, mid, kind, rate) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def years_to_expiry(now: datetime, expiry: date) -> float:
    """Calendar time from `now` to 16:00 New York on the expiry date, in years."""
    close = datetime.combine(expiry, time(16, 0), tzinfo=NY)
    return max((close - now.astimezone(NY)).total_seconds() / 60.0, 0.0) / MINUTES_PER_YEAR


# --- Picking -------------------------------------------------------------------

def pick_expiry(expiries: list[str | date], on: date) -> date:
    """Nearest listed expiry on or after `on`."""
    days = sorted(pd.Timestamp(e).date() for e in expiries)
    for d in days:
        if d >= on:
            return d
    raise ValueError(f"no expiry on or after {on}")


def pick_strike(strikes, spot: float, kind: str, n_out: int = STRIKES_OUT) -> float:
    """The n-th listed strike past the price: above it for calls, below for puts.

    A strike exactly at the price counts as at the money, not as the first one out.
    """
    s = sorted(set(float(k) for k in strikes))
    out = [k for k in s if k > spot] if kind == "call" else [k for k in reversed(s) if k < spot]
    if len(out) < n_out:
        raise ValueError(f"only {len(out)} {kind} strikes past {spot}")
    return out[n_out - 1]


@dataclass
class OptionPlan:
    ticker: str
    side: str
    kind: str
    expiry: str
    days_to_expiry: int
    strike: float
    strikes_out: int
    spot: float
    stop: float
    target: float
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    est_price_at_stop: float | None
    est_price_at_target: float | None
    est_loss_per_contract: float | None
    est_gain_per_contract: float | None
    contracts: int
    est_loss_at_stop: float | None
    premium_paid: float
    premium_pct_of_account: float
    risk_cap: float
    over_cap: bool
    move_to_strike_pct: float
    note: str


def plan_option_trade(ticker: str, side: str, spot: float, stop: float, target: float,
                      chain: pd.DataFrame, expiry: date, now: datetime,
                      account: float = ACCOUNT_SIZE, risk_fraction: float = RISK_FRACTION,
                      n_out: int = STRIKES_OUT, hold_minutes: float = HOLD_MINUTES,
                      max_premium_fraction: float | None = MAX_PREMIUM_FRACTION) -> OptionPlan:
    """Build the trade plan from one expiry's chain.

    `chain` needs columns strike, bid, ask for the right type (calls for a long
    signal, puts for a short one). `spot` is the stock price at entry.
    """
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    kind = "call" if side == "long" else "put"
    if (side == "long" and not stop < spot < target) or (side == "short" and not target < spot < stop):
        raise ValueError(f"stop/target don't bracket the price for a {side} signal")

    strike = pick_strike(chain["strike"], spot, kind, n_out)
    row = chain.loc[chain["strike"].astype(float) == strike].iloc[0]
    bid, ask = float(row["bid"]), float(row["ask"])
    cap = round(account * risk_fraction, 2)
    notes = []
    if ask <= 0:
        raise ValueError(f"no ask for {ticker} {expiry} {strike} {kind}")
    if bid <= 0:
        notes.append("no bid: you may not be able to sell this contract")

    t_now = years_to_expiry(now, expiry)
    t_exit = max(t_now - hold_minutes / MINUTES_PER_YEAR, 0.0)
    mid = (bid + ask) / 2 if bid > 0 else ask
    half_spread = (ask - bid) / 2 if bid > 0 else 0.0
    iv = implied_vol(mid, spot, strike, t_now, kind)

    est_stop = est_target = loss = gain = delta = None
    if iv is None:
        notes.append("could not solve implied volatility from the quote (stale or off-hours?); "
                      "loss at stop not estimated, sized to one contract")
    else:
        delta = bs_delta(spot, strike, t_now, iv, kind)
        est_stop = max(bs_price(stop, strike, t_exit, iv, kind) - half_spread, 0.0)
        est_target = max(bs_price(target, strike, t_exit, iv, kind) - half_spread, 0.0)
        loss = round((ask - est_stop) * 100, 2)
        gain = round((est_target - ask) * 100, 2)

    if loss is not None and loss > 0:
        contracts = max(1, math.floor(cap / loss))
    else:
        contracts = 1
    if max_premium_fraction is not None:
        by_premium = max(1, math.floor(account * max_premium_fraction / (ask * 100)))
        if by_premium < contracts:
            notes.append(f"contracts cut from {contracts} to {by_premium} by the "
                         f"{max_premium_fraction:.0%} premium cap")
            contracts = by_premium
    over_cap = loss is not None and loss * contracts > cap
    if over_cap:
        notes.append(f"one contract's estimated loss at the stop is over the ${cap:.2f} cap")
    if gain is not None and loss and gain < loss:
        notes.append("estimated gain at the target is smaller than the loss at the stop")
    if gain is not None and gain <= 0:
        notes.append("estimated to lose money even at the target: the stock move doesn't cover the premium")

    move = (strike - spot) / spot * 100 if kind == "call" else (spot - strike) / spot * 100
    return OptionPlan(
        ticker=ticker, side=side, kind=kind, expiry=expiry.isoformat(),
        days_to_expiry=(expiry - now.astimezone(NY).date()).days,
        strike=strike, strikes_out=n_out, spot=spot, stop=stop, target=target,
        bid=bid, ask=ask, iv=None if iv is None else round(iv, 4),
        delta=None if delta is None else round(delta, 3),
        est_price_at_stop=None if est_stop is None else round(est_stop, 2),
        est_price_at_target=None if est_target is None else round(est_target, 2),
        est_loss_per_contract=loss, est_gain_per_contract=gain, contracts=contracts,
        est_loss_at_stop=None if loss is None else round(loss * contracts, 2),
        premium_paid=round(ask * 100 * contracts, 2),
        premium_pct_of_account=round(ask * 100 * contracts / account * 100, 1), risk_cap=cap, over_cap=over_cap,
        move_to_strike_pct=round(move, 3), note="; ".join(notes),
    )


# --- Yahoo chain source --------------------------------------------------------

def yahoo_chain(ticker: str, on: date, side: str) -> tuple[pd.DataFrame, date]:
    """Calls or puts for the nearest expiry on or after `on`, from Yahoo Finance.

    Yahoo's chain is a live snapshot with no history, so this only works for
    today's signals.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError(f"Yahoo returned no expiries for {ticker} (rate-limited? try again in a minute)")
    expiry = pick_expiry(expiries, on)
    ch = tk.option_chain(expiry.isoformat())
    df = ch.calls if side.lower() == "long" else ch.puts
    return df[["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy(), expiry


def plan_from_signal(sig: dict, now: datetime | None = None, **kw) -> OptionPlan:
    """Plan a trade for one scanner signal dict (ticker, side, entry, stop, target)."""
    now = now or datetime.now(NY)
    local = now.astimezone(NY)
    on = local.date()
    if local.time() >= time(16, 0):  # today's contracts have expired
        on = (pd.Timestamp(on) + pd.Timedelta(days=1)).date()
    chain, expiry = yahoo_chain(sig["ticker"], on, sig["side"])
    return plan_option_trade(sig["ticker"], sig["side"], float(sig["entry"]), float(sig["stop"]),
                             float(sig["target"]), chain, expiry, now, **kw)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ticker", nargs="?")
    p.add_argument("side", nargs="?", choices=["long", "short"])
    p.add_argument("entry", nargs="?", type=float)
    p.add_argument("stop", nargs="?", type=float)
    p.add_argument("target", nargs="?", type=float)
    p.add_argument("--signals", help="scanner signals CSV (ticker, side, entry, stop, target)")
    p.add_argument("--last", type=int, default=5, help="plan the last N signals of the CSV")
    p.add_argument("--account", type=float, default=ACCOUNT_SIZE)
    p.add_argument("--risk", type=float, default=RISK_FRACTION)
    p.add_argument("--strikes-out", type=int, default=STRIKES_OUT)
    p.add_argument("--hold-minutes", type=float, default=HOLD_MINUTES)
    p.add_argument("--max-premium", type=float, default=MAX_PREMIUM_FRACTION,
                   help="most premium per trade as a share of the account; 0 = no cap")
    a = p.parse_args(argv)

    if a.signals:
        sigs = pd.read_csv(a.signals).tail(a.last).to_dict("records")
    elif a.ticker and a.side and a.entry and a.stop and a.target:
        sigs = [dict(ticker=a.ticker.upper(), side=a.side, entry=a.entry, stop=a.stop, target=a.target)]
    else:
        p.error("give TICKER SIDE ENTRY STOP TARGET or --signals CSV")

    rows = []
    for s in sigs:
        try:
            rows.append(asdict(plan_from_signal(s, account=a.account, risk_fraction=a.risk,
                                                n_out=a.strikes_out, hold_minutes=a.hold_minutes,
                                                max_premium_fraction=a.max_premium or None)))
        except Exception as e:  # keep going on the other signals
            rows.append(dict(ticker=s["ticker"], side=s["side"], note=f"error: {e}"))
    out = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
