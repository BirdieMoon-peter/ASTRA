from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REQUIRED_REPORT_COLUMNS = {"report_id", "report_date", "split", "stock_code", "title", "summary"}
REQUIRED_MARKET_COLUMNS = {"stock_code", "trade_date", "close", "pct_change"}


def _header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return {column.strip() for column in next(reader)}
        except StopIteration:
            return set()


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_experiment_package(*, package_dir: Path, gold_path: Path) -> dict[str, Any]:
    master_path = package_dir / "reports_experiment_master.csv"
    train_path = package_dir / "splits" / "reports_train.csv"
    dev_path = package_dir / "splits" / "reports_dev.csv"
    test_path = package_dir / "splits" / "reports_test.csv"
    market_path = package_dir / "market" / "daily_prices.csv"

    required_files = {
        "reports_experiment_master": master_path,
        "reports_train": train_path,
        "reports_dev": dev_path,
        "reports_test": test_path,
        "market_daily_prices": market_path,
    }
    missing_files = {name: str(path) for name, path in required_files.items() if not path.exists()}
    if missing_files:
        return {
            "status": "missing_files",
            "package_dir": str(package_dir),
            "gold_path": str(gold_path),
            "gold_available": gold_path.exists(),
            "missing_files": missing_files,
        }

    report_like_files = {
        "reports_experiment_master": master_path,
        "reports_train": train_path,
        "reports_dev": dev_path,
        "reports_test": test_path,
    }
    invalid_schema: dict[str, dict[str, Any]] = {}
    missing_split_columns: dict[str, list[str]] = {}

    for name, path in report_like_files.items():
        missing_columns = sorted(REQUIRED_REPORT_COLUMNS - _header(path))
        if missing_columns:
            invalid_schema[name] = {
                "path": str(path),
                "missing_columns": missing_columns,
            }
            if name != "reports_experiment_master":
                missing_split_columns[name] = missing_columns

    market_missing = sorted(REQUIRED_MARKET_COLUMNS - _header(market_path))
    if market_missing:
        invalid_schema["market_daily_prices"] = {
            "path": str(market_path),
            "missing_columns": market_missing,
        }
    if invalid_schema:
        return {
            "status": "invalid_schema",
            "package_dir": str(package_dir),
            "gold_path": str(gold_path),
            "gold_available": gold_path.exists(),
            "invalid_schema": invalid_schema,
            "missing_split_columns": missing_split_columns,
        }

    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "gold_path": str(gold_path),
        "gold_available": gold_path.exists(),
        "market_available": True,
        "split_counts": {
            "train": _count_rows(train_path),
            "dev": _count_rows(dev_path),
            "test": _count_rows(test_path),
        },
    }
