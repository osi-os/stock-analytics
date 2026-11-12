from datetime import date

import pandas as pd

from daytrade import option_backtest as ob


def _bars(times, prices, vols=None):
    idx = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("America/New_York")
    vols = vols or [10] * len(prices)
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices,
                         "volume": vols, "vwap": prices, "trades": [1] * len(prices)}, index=idx)


def test_entry_and_exit_prices():
    b = _bars(["2026-10-16 09:51", "2026-10-16 09:53", "2026-10-16 10:01", "2026-10-16 10:03"],
              [1.00, 1.10, 0.80, 0.60], [10, 10, 10, 30])
    t_in = pd.Timestamp("2026-10-16 09:50", tz="America/New_York")
    price, when = ob._entry_price(b, t_in, 5)
    assert price == 1.00 and when.minute == 51
    t_out = pd.Timestamp("2026-10-16 10:00", tz="America/New_York")
    px, how = ob._exit_price(b, t_out)
    assert how == "in_bar" and abs(px - (0.8 * 10 + 0.6 * 30) / 40) < 1e-9
    # nothing in the exit bar -> last trade before its end
    px, how = ob._exit_price(b.iloc[:2], t_out)
    assert how == "last_trade_before" and px == 1.10


def test_replay_trade_with_fake_data(monkeypatch):
    cdf = pd.DataFrame({"ticker": [f"O:X{k}" for k in (100, 101, 102, 103, 104)],
                        "expiration_date": [date(2026, 10, 16)] * 5,
                        "strike_price": [100.0, 101.0, 102.0, 103.0, 104.0],
                        "contract_type": ["call"] * 5})
    monkeypatch.setattr(ob.massive, "contracts", lambda *a, **k: cdf)
    bars = _bars(["2026-10-16 10:00", "2026-10-16 10:30"], [0.20, 0.35])
    monkeypatch.setattr(ob.massive, "minute_bars", lambda *a, **k: bars)
    tr = dict(ticker="X", setup="orb", side="long", entry_time="2026-10-16 10:00:00-04:00",
              exit_time="2026-10-16 10:30:00-04:00", exit_reason="target",
              entry=100.5, stop=100.0, target=101.5, r_multiple=2.0)
    row = ob.replay_trade(tr, {})
    assert row["status"] == "ok" and row["strike"] == 103.0
    assert row["option_entry"] == 0.20 and row["option_exit"] == 0.35
    assert row["contracts"] >= 1
    assert row["pnl1_slip0.00"] == 15.0
