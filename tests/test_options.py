import math
from datetime import date, datetime

import pandas as pd
import pytest

from daytrade.options import (NY, bs_price, implied_vol, pick_expiry, pick_strike,
                              plan_option_trade)


def test_pick_strike_counts_listed_strikes_past_price():
    strikes = [172.5, 175, 177.5, 180, 182.5, 185, 187.5]
    assert pick_strike(strikes, 178.67, "call") == 185      # 180, 182.5, 185
    assert pick_strike(strikes, 178.67, "put") == 172.5     # 177.5, 175, 172.5
    # a strike exactly at the price is at the money, not the first one out
    assert pick_strike(strikes, 180.0, "call") == 187.5


def test_pick_expiry_nearest_on_or_after():
    exp = ["2026-11-09", "2026-11-11", "2026-11-13"]
    assert pick_expiry(exp, date(2026, 11, 10)) == date(2026, 11, 11)
    assert pick_expiry(exp, date(2026, 11, 9)) == date(2026, 11, 9)


def test_put_call_parity_and_iv_roundtrip():
    S, K, T, v, r = 100, 103, 2 / 365, 0.35, 0.04
    c, p = bs_price(S, K, T, v, "call", r), bs_price(S, K, T, v, "put", r)
    assert math.isclose(c - p, S - K * math.exp(-r * T), abs_tol=1e-9)
    assert math.isclose(implied_vol(c, S, K, T, "call", r), v, rel_tol=1e-6)


def _chain(spot, vol, T, kind, step=2.5):
    rows = []
    for i in range(-8, 9):
        k = round(spot / step) * step + i * step
        m = bs_price(spot, k, T, vol, kind)
        rows.append(dict(strike=k, bid=max(round(m - 0.02, 2), 0.0), ask=round(m + 0.02, 2)))
    return pd.DataFrame(rows)


def test_plan_sizes_on_loss_at_stop_and_keeps_third_strike():
    now = datetime(2026, 11, 11, 10, 0, tzinfo=NY)
    exp = date(2026, 11, 13)
    T = (2 * 24 + 6) / (365 * 24)
    chain = _chain(178.7, 0.45, T, "call")
    plan = plan_option_trade("NVDA", "long", 178.7, 177.8, 180.5, chain, exp, now)
    assert plan.kind == "call" and plan.strike == 185.0
    assert plan.est_loss_per_contract > 0
    assert plan.contracts == max(1, math.floor(107.4 / plan.est_loss_per_contract))
    assert plan.est_loss_at_stop <= 107.4 or plan.contracts == 1


def test_short_signal_buys_put_and_flags_over_cap():
    now = datetime(2026, 11, 11, 10, 0, tzinfo=NY)
    exp = date(2026, 11, 13)
    T = (2 * 24 + 6) / (365 * 24)
    chain = _chain(680, 0.6, T, "put", step=1)
    plan = plan_option_trade("QQQ", "short", 680, 690, 670, chain, exp, now)  # very wide stop
    assert plan.kind == "put" and plan.strike == 677
    assert plan.contracts == 1 and plan.over_cap


def test_bad_bracket_rejected():
    now = datetime(2026, 11, 11, 10, 0, tzinfo=NY)
    with pytest.raises(ValueError):
        plan_option_trade("X", "long", 100, 101, 102, _chain(100, .3, .01, "call"), date(2026, 11, 13), now)
