from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from astra.config.task_schema import resolve_history_reports_path


class HistoryRetriever:
    def __init__(self, reports_path: Path | None = None) -> None:
        if reports_path is None:
            reports_path = resolve_history_reports_path()
        self.by_stock: dict[str, list[dict[str, str]]] = defaultdict(list)
        with reports_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.by_stock[row["stock_code"]].append(row)
        for stock_rows in self.by_stock.values():
            stock_rows.sort(key=lambda item: (item["report_date"], item["report_id"]))

    def retrieve(self, stock_code: str, report_date: str, limit: int = 5) -> list[dict[str, str]]:
        history = []
        for row in self.by_stock.get(stock_code, []):
            if row["report_date"] < report_date:
                history.append(row)
        return history[-limit:]

    def format_context(self, stock_code: str, report_date: str, limit: int = 5) -> str:
        rows = self.retrieve(stock_code, report_date, limit=limit)
        if not rows:
            return "无历史研报上下文"
        parts = []
        for row in rows:
            parts.append(
                f"[{row['report_date']}] {row['title']} | {row['summary'][:180]}"
            )
        return "\n".join(parts)
