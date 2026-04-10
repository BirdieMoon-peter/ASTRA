from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astra.config.task_schema import (
    ANNOTATIONS_DIR,
    EXPERIMENT_ANALYSTS_PATH,
    EXPERIMENT_BROKERS_PATH,
    EXPERIMENT_DEV_PATH,
    EXPERIMENT_MANIFEST_PATH,
    EXPERIMENT_MARKET_PRICES_PATH,
    EXPERIMENT_RATINGS_PATH,
    EXPERIMENT_TEST_PATH,
    EXPERIMENT_TRAIN_PATH,
    EXPERIMENT_VERSIONS_PATH,
    REPORTS_EXPERIMENT_MASTER_PATH,
    MARKET_PRICES_PATH,
    REPORT_ANALYSTS_PATH,
    REPORT_BROKERS_PATH,
    REPORT_RATINGS_PATH,
    REPORT_VERSIONS_PATH,
    REPORTS_MASTER_PATH,
    SplitConfig,
)
from astra.data.clean_reports import assign_split, build_report_id, normalize_stock_code, normalize_text, parse_date
from astra.data.streaming_csv import open_csv_writer

MASTER_FIELDNAMES = [
    "report_id",
    "report_date",
    "report_year",
    "split",
    "stock_code",
    "company_name",
    "broker_id",
    "broker_name",
    "analyst_id",
    "analyst_name",
    "title",
    "summary",
    "body_raw",
    "risk_section_raw",
    "combined_text",
    "rating",
    "target_price",
    "industry",
    "source_url",
    "version_hash",
    "is_deleted",
]
REQUIRED_MASTER_COLUMNS = {
    "report_id",
    "stock_id",
    "stock_name",
    "broker_id",
    "broker_name",
    "analyst_id",
    "analyst_name",
    "publish_time",
    "title",
    "summary",
    "body_raw",
    "risk_section_raw",
    "rating",
    "target_price",
    "industry",
    "source_url",
    "version_hash",
    "is_deleted",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _update_date_range(date_range: dict[str, str | None], value: str) -> None:
    if not value:
        return
    current_min = date_range.get("min")
    current_max = date_range.get("max")
    if current_min is None or value < current_min:
        date_range["min"] = value
    if current_max is None or value > current_max:
        date_range["max"] = value


def _combine_text(title: str, summary: str) -> str:
    if title and summary:
        return f"{title}\n{summary}"
    return title or summary


def _bool_annotation_available() -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in ANNOTATIONS_DIR.glob("*.jsonl"))


def _copy_csv_asset(
    source_path: Path,
    output_path: Path,
    *,
    primary_key: list[str] | None = None,
    join_keys: list[str] | None = None,
    date_field: str | None = None,
    unique_field: str | None = None,
) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = reader.fieldnames or []
        out_handle, writer = open_csv_writer(output_path, fieldnames)
        row_count = 0
        date_range: dict[str, str | None] = {"min": None, "max": None}
        unique_values: set[str] = set()
        try:
            for row in reader:
                clean_row = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
                writer.writerow(clean_row)
                row_count += 1
                if date_field:
                    _update_date_range(date_range, str(clean_row.get(date_field, "")))
                if unique_field:
                    value = str(clean_row.get(unique_field, "")).strip()
                    if value:
                        unique_values.add(value)
        finally:
            out_handle.close()

    table_info: dict[str, Any] = {
        "path": str(output_path),
        "source_path": str(source_path),
        "row_count": row_count,
    }
    if primary_key:
        table_info["primary_key"] = primary_key
    if join_keys:
        table_info["join_keys"] = join_keys
    if date_field:
        table_info["date_field"] = date_field
        table_info["date_range"] = date_range
    if unique_field:
        table_info[f"unique_{unique_field}_count"] = len(unique_values)
    return table_info


def _build_reports_experiment_master(
    source_path: Path = REPORTS_MASTER_PATH,
    output_path: Path = REPORTS_EXPERIMENT_MASTER_PATH,
) -> dict[str, Any]:
    split_config = SplitConfig()
    raw_report_count = 0
    packaged_report_count = 0
    skipped_deleted_count = 0
    skipped_missing_date_count = 0
    skipped_missing_key_count = 0
    duplicate_report_id_count = 0
    report_date_range: dict[str, str | None] = {"min": None, "max": None}
    split_counts: Counter[str] = Counter()
    counts_by_year: Counter[str] = Counter()
    unique_stocks: set[str] = set()
    unique_companies: set[str] = set()
    seen_report_ids: set[str] = set()

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MASTER_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"reports_master.csv missing required columns: {missing}")

        out_handle, writer = open_csv_writer(output_path, MASTER_FIELDNAMES)
        try:
            for row in reader:
                raw_report_count += 1
                is_deleted = str(row.get("is_deleted", "")).strip()
                if is_deleted == "1":
                    skipped_deleted_count += 1
                    continue

                publish_time = str(row.get("publish_time", "")).strip()
                report_date = publish_time[:10].replace("/", "-")
                if not report_date:
                    skipped_missing_date_count += 1
                    continue

                parsed_date = parse_date(report_date)
                report_year = parsed_date.year
                split = assign_split(report_year, split_config)
                stock_code = normalize_stock_code(str(row.get("stock_id", "")))
                company_name = normalize_text(str(row.get("stock_name", "")))
                broker_id = str(row.get("broker_id", "") or "").strip()
                broker_name = normalize_text(str(row.get("broker_name", "")))
                analyst_id = str(row.get("analyst_id", "") or "").strip()
                analyst_name = normalize_text(str(row.get("analyst_name", "")))
                title = normalize_text(str(row.get("title", "")))
                body_raw = normalize_text(str(row.get("body_raw", "")))
                summary = normalize_text(str(row.get("summary", ""))) or body_raw
                risk_section_raw = normalize_text(str(row.get("risk_section_raw", "")))
                rating = normalize_text(str(row.get("rating", "")))
                target_price = str(row.get("target_price", "") or "").strip()
                industry = normalize_text(str(row.get("industry", "")))
                source_url = str(row.get("source_url", "") or "").strip()
                version_hash = str(row.get("version_hash", "") or "").strip()
                report_id = str(row.get("report_id", "") or "").strip()
                if not report_id:
                    report_id = build_report_id(report_date, stock_code, title, summary)

                if not report_id or not stock_code or not (title or summary):
                    skipped_missing_key_count += 1
                    continue
                if report_id in seen_report_ids:
                    duplicate_report_id_count += 1
                    continue
                seen_report_ids.add(report_id)

                combined_text = _combine_text(title, summary)
                writer.writerow(
                    {
                        "report_id": report_id,
                        "report_date": report_date,
                        "report_year": report_year,
                        "split": split,
                        "stock_code": stock_code,
                        "company_name": company_name,
                        "broker_id": broker_id,
                        "broker_name": broker_name,
                        "analyst_id": analyst_id,
                        "analyst_name": analyst_name,
                        "title": title,
                        "summary": summary,
                        "body_raw": body_raw,
                        "risk_section_raw": risk_section_raw,
                        "combined_text": combined_text,
                        "rating": rating,
                        "target_price": target_price,
                        "industry": industry,
                        "source_url": source_url,
                        "version_hash": version_hash,
                        "is_deleted": is_deleted,
                    }
                )
                packaged_report_count += 1
                split_counts[split] += 1
                counts_by_year[str(report_year)] += 1
                unique_stocks.add(stock_code)
                unique_companies.add(company_name)
                _update_date_range(report_date_range, report_date)
        finally:
            out_handle.close()

    return {
        "path": str(output_path),
        "source_path": str(source_path),
        "row_count": packaged_report_count,
        "primary_key": ["report_id"],
        "join_keys": ["report_id", "stock_code", "report_date"],
        "date_field": "report_date",
        "date_range": report_date_range,
        "split_counts": dict(split_counts),
        "counts_by_year": dict(sorted(counts_by_year.items())),
        "unique_stock_count": len(unique_stocks),
        "unique_company_count": len(unique_companies),
        "raw_report_count": raw_report_count,
        "skipped_deleted_count": skipped_deleted_count,
        "skipped_missing_date_count": skipped_missing_date_count,
        "skipped_missing_key_count": skipped_missing_key_count,
        "duplicate_report_id_count": duplicate_report_id_count,
    }


def _build_split_exports(
    source_path: Path = REPORTS_EXPERIMENT_MASTER_PATH,
    train_path: Path = EXPERIMENT_TRAIN_PATH,
    dev_path: Path = EXPERIMENT_DEV_PATH,
    test_path: Path = EXPERIMENT_TEST_PATH,
) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = reader.fieldnames or []
        train_handle, train_writer = open_csv_writer(train_path, fieldnames)
        dev_handle, dev_writer = open_csv_writer(dev_path, fieldnames)
        test_handle, test_writer = open_csv_writer(test_path, fieldnames)
        writers = {
            "train": train_writer,
            "dev": dev_writer,
            "test": test_writer,
        }
        counts: Counter[str] = Counter()
        try:
            for row in reader:
                split = str(row.get("split", "")).strip()
                writer = writers.get(split)
                if writer is None:
                    raise ValueError(f"Unexpected split in experiment master: {split}")
                writer.writerow(row)
                counts[split] += 1
        finally:
            train_handle.close()
            dev_handle.close()
            test_handle.close()

    return {
        "reports_train": {
            "path": str(train_path),
            "row_count": counts["train"],
            "source_path": str(source_path),
            "split": "train",
        },
        "reports_dev": {
            "path": str(dev_path),
            "row_count": counts["dev"],
            "source_path": str(source_path),
            "split": "dev",
        },
        "reports_test": {
            "path": str(test_path),
            "row_count": counts["test"],
            "source_path": str(source_path),
            "split": "test",
        },
    }


def build_experiment_package() -> dict[str, Any]:
    reports_info = _build_reports_experiment_master()
    split_tables = _build_split_exports()
    ratings_info = _copy_csv_asset(
        REPORT_RATINGS_PATH,
        EXPERIMENT_RATINGS_PATH,
        join_keys=["report_id", "publish_time"],
        date_field="publish_time",
    )
    brokers_info = _copy_csv_asset(
        REPORT_BROKERS_PATH,
        EXPERIMENT_BROKERS_PATH,
        primary_key=["broker_id"],
    )
    analysts_info = _copy_csv_asset(
        REPORT_ANALYSTS_PATH,
        EXPERIMENT_ANALYSTS_PATH,
        primary_key=["analyst_id", "broker_id"],
    )
    versions_info = _copy_csv_asset(
        REPORT_VERSIONS_PATH,
        EXPERIMENT_VERSIONS_PATH,
        join_keys=["report_id", "version_hash"],
    )
    market_info = _copy_csv_asset(
        MARKET_PRICES_PATH,
        EXPERIMENT_MARKET_PRICES_PATH,
        join_keys=["stock_code", "trade_date"],
        date_field="trade_date",
        unique_field="stock_code",
    )

    manifest = {
        "created_at": _iso_now(),
        "package_dir": str(EXPERIMENT_MANIFEST_PATH.parent),
        "annotation_available": _bool_annotation_available(),
        "tables": {
            "reports_experiment_master": reports_info,
            "report_ratings": ratings_info,
            "report_brokers": brokers_info,
            "report_analysts": analysts_info,
            "report_versions": versions_info,
            "market_daily_prices": market_info,
            **split_tables,
        },
        "summary": {
            "report_count": reports_info["row_count"],
            "split_counts": reports_info["split_counts"],
            "report_date_range": reports_info["date_range"],
            "market_row_count": market_info["row_count"],
            "market_date_range": market_info.get("date_range"),
        },
    }

    EXPERIMENT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
