from __future__ import annotations

from pathlib import Path
from typing import Iterable

from astra.config.task_schema import RAW_REPORTS_PATH
from astra.data.streaming_csv import iter_csv_rows

EXPECTED_COLUMNS = (
    "report_date",
    "stock_code",
    "company_name",
    "title",
    "summary",
)


def load_reports(path: Path = RAW_REPORTS_PATH) -> list[dict[str, str]]:
    return list(iter_reports(path))


def iter_reports(path: Path = RAW_REPORTS_PATH) -> Iterable[dict[str, str]]:
    yield from iter_csv_rows(path, EXPECTED_COLUMNS)
