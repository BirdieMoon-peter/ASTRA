from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from astra.config.task_schema import CLEAN_REPORTS_PATH, CORPUS_STATS_PATH, SplitConfig
from astra.data.load_reports import load_reports

WHITESPACE_RE = re.compile(r"\s+")
STOCK_CODE_RE = re.compile(r"\D")


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in value.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def normalize_stock_code(value: str) -> str:
    digits = STOCK_CODE_RE.sub("", value)
    return digits.zfill(6) if digits else ""


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def build_report_id(report_date: str, stock_code: str, title: str, summary: str) -> str:
    key = "||".join((report_date, stock_code, title, summary))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def assign_split(report_year: int, split_config: SplitConfig) -> str:
    split = split_config.split_for_year(report_year)
    if split is None:
        raise ValueError(f"No split configured for year={report_year}")
    return split


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    lower_value = values[lower]
    upper_value = values[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "p25": round(_quantile(sorted_values, 0.25), 2),
        "median": float(median(sorted_values)),
        "mean": round(sum(sorted_values) / len(sorted_values), 2),
        "p75": round(_quantile(sorted_values, 0.75), 2),
        "max": float(sorted_values[-1]),
    }


def clean_reports(output_path: Path = CLEAN_REPORTS_PATH, stats_path: Path = CORPUS_STATS_PATH) -> dict[str, object]:
    raw_rows = load_reports()
    split_config = SplitConfig()

    cleaned_rows: list[dict[str, object]] = []
    dropped_empty_summary = 0
    duplicate_count = 0
    seen_keys: set[tuple[str, str, str, str]] = set()
    per_year = Counter()
    title_lengths: list[int] = []
    summary_lengths: list[int] = []
    stock_codes: set[str] = set()
    companies: set[str] = set()

    for row in raw_rows:
        report_date = row["report_date"].strip()
        parsed_date = parse_date(report_date)
        stock_code = normalize_stock_code(row["stock_code"])
        company_name = normalize_text(row["company_name"])
        title = normalize_text(row["title"])
        summary = normalize_text(row["summary"])

        if not summary:
            dropped_empty_summary += 1
            continue

        dedupe_key = (report_date, stock_code, title, summary)
        if dedupe_key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(dedupe_key)

        report_id = build_report_id(report_date, stock_code, title, summary)
        report_year = parsed_date.year
        split = assign_split(report_year, split_config)
        combined_text = f"{title}\n{summary}"

        cleaned_rows.append(
            {
                "report_id": report_id,
                "report_date": report_date,
                "report_year": report_year,
                "split": split,
                "stock_code": stock_code,
                "company_name": company_name,
                "title": title,
                "summary": summary,
                "combined_text": combined_text,
                "title_length": len(title),
                "summary_length": len(summary),
            }
        )
        per_year[str(report_year)] += 1
        title_lengths.append(len(title))
        summary_lengths.append(len(summary))
        stock_codes.add(stock_code)
        companies.add(company_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "report_id",
        "report_date",
        "report_year",
        "split",
        "stock_code",
        "company_name",
        "title",
        "summary",
        "combined_text",
        "title_length",
        "summary_length",
    ]

    import csv

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    stats: dict[str, object] = {
        "raw_report_count": len(raw_rows),
        "clean_report_count": len(cleaned_rows),
        "dropped_empty_summary_count": dropped_empty_summary,
        "duplicate_dropped_count": duplicate_count,
        "unique_stock_count": len(stock_codes),
        "unique_company_count": len(companies),
        "date_range": {
            "min": cleaned_rows[0]["report_date"] if cleaned_rows else None,
            "max": cleaned_rows[-1]["report_date"] if cleaned_rows else None,
        },
        "counts_by_year": dict(sorted(per_year.items())),
        "title_length": _length_stats(title_lengths),
        "summary_length": _length_stats(summary_lengths),
        "split_counts": dict(Counter(row["split"] for row in cleaned_rows)),
    }

    if cleaned_rows:
        sorted_dates = sorted(row["report_date"] for row in cleaned_rows)
        stats["date_range"] = {"min": sorted_dates[0], "max": sorted_dates[-1]}

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return stats
