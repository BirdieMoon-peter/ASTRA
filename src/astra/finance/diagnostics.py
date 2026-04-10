from __future__ import annotations

from typing import Any


def build_backtest_diagnostics(
    *,
    n_trade_dates: int,
    n_stocks_per_date_mean: float,
    n_total_observations: int,
    sample_adequate: bool,
    warning: str,
) -> dict[str, Any]:
    return {
        'n_trade_dates': n_trade_dates,
        'n_stocks_per_date_mean': n_stocks_per_date_mean,
        'n_total_observations': n_total_observations,
        'sample_adequate': sample_adequate,
        'warning': warning,
    }
