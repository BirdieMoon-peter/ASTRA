from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from astra.config.task_schema import BACKTEST_CURVE_PATH, CLEAN_REPORTS_PATH, MARKET_DIR

MARKET_PRICES_PATH = MARKET_DIR / "daily_prices.csv"


def load_market_prices(path: Path = MARKET_PRICES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def trading_dates(prices: list[dict[str, Any]]) -> list[str]:
    return sorted({row["trade_date"] for row in prices})


def next_trading_date(report_date: str, trading_calendar: list[str]) -> str | None:
    for trade_date in trading_calendar:
        if trade_date >= report_date:
            return trade_date
    return None
