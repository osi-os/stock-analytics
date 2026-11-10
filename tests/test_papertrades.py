import pandas as pd
import pytest

from daytrade.papertrades import COLUMNS, load, summary, write_template


def _write(tmp_path, rows):
    p = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(p, index=False)
    return p


def test_pnl_r_and_stop_check(tmp_path):
    rows = [
        ["2026-11-10", "NVDA", "call", "2026-11-11", 177.5, 2, 1.02, 0.85, 171.29, 170.46, 172.94, "stop", 41.98, ""],
        ["2026-11-10", "QQQ", "put", "2026-11-10", 674, 1, 1.36, 2.50, 676.97, 678.56, 673.79, "target", 53.85, ""],
    ]
    df = load(_write(tmp_path, rows))
    assert df.loc[0, "pnl"] == pytest.approx(-34.0)
    assert df.loc[1, "pnl"] == pytest.approx(114.0)
    s = summary(df)
    assert s["trades"] == 2 and s["wins"] == 1
    assert s["stop_estimate_check"]["real_loss_vs_estimate_median"] == pytest.approx(34 / 41.98, abs=0.01)


def test_template_does_not_overwrite(tmp_path):
    p = write_template(tmp_path / "log.csv")
    with pytest.raises(FileExistsError):
        write_template(p)


def test_bad_exit_reason(tmp_path):
    rows = [["2026-11-10", "AAPL", "call", "2026-11-11", 352.5, 1, .5, .4, 347, 346, 349, "oops", 15, ""]]
    with pytest.raises(ValueError):
        load(_write(tmp_path, rows))
