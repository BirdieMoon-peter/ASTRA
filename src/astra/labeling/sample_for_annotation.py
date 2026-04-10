from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from astra.config.task_schema import (
    ANNOTATION_DEV_ALIGNED_WORKSET_PATH,
    ANNOTATION_GOLD_WORKSET_PATH,
    ANNOTATION_MAIN_PATH,
    ANNOTATION_PILOT_PATH,
    ASTRA_PREDICTIONS_PATH,
    MAIN_SAMPLE_PATH,
    PILOT_SAMPLE_PATH,
)
from astra.data.clean_reports import CLEAN_REPORTS_PATH

SEED = 42
DEFAULT_GOLD_SPLIT_TARGETS = {"train": 400, "dev": 300, "test": 300}

ANNOTATION_TEMPLATE = {
    "fundamental_sentiment": None,
    "strategic_optimism": None,
    "phenomenon": None,
    "evidence_spans": [],
    "annotation_confidence": None,
    "notes": "",
}


def _read_clean_rows(path: Path = CLEAN_REPORTS_PATH) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stratified_sample(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["report_year"]].append(row)

    rng = random.Random(SEED)
    years = sorted(grouped.keys())
    if not years:
        return []

    base = sample_size // len(years)
    remainder = sample_size % len(years)
    samples: list[dict[str, str]] = []
    for index, year in enumerate(years):
        year_rows = grouped[year][:]
        rng.shuffle(year_rows)
        target = min(len(year_rows), base + (1 if index < remainder else 0))
        samples.extend(year_rows[:target])

    seen_ids = {row["report_id"] for row in samples}
    if len(samples) < sample_size:
        leftovers = [row for row in rows if row["report_id"] not in seen_ids]
        rng.shuffle(leftovers)
        samples.extend(leftovers[: sample_size - len(samples)])

    samples.sort(key=lambda item: (item["report_date"], item["stock_code"], item["report_id"]))
    return samples


def _to_annotation_record(row: dict[str, str]) -> dict[str, object]:
    annotation = dict(ANNOTATION_TEMPLATE)
    annotation["evidence_spans"] = []
    report_date = str(row["report_date"])
    report_year_value = row.get("report_year")
    report_year = int(report_year_value) if report_year_value is not None else int(report_date[:4])
    return {
        "report_id": row["report_id"],
        "report_date": report_date,
        "report_year": report_year,
        "split": row["split"],
        "stock_code": row["stock_code"],
        "company_name": row.get("company_name", ""),
        "title": row["title"],
        "summary": row["summary"],
        "annotation": annotation,
    }


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_within_split(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["report_year"]].append(row)

    rng = random.Random(SEED)
    years = sorted(grouped.keys())
    if not years:
        return []

    base = sample_size // len(years)
    remainder = sample_size % len(years)
    samples: list[dict[str, str]] = []
    for index, year in enumerate(years):
        year_rows = grouped[year][:]
        rng.shuffle(year_rows)
        target = min(len(year_rows), base + (1 if index < remainder else 0))
        samples.extend(year_rows[:target])

    seen_ids = {row["report_id"] for row in samples}
    if len(samples) < sample_size:
        leftovers = [row for row in rows if row["report_id"] not in seen_ids]
        rng.shuffle(leftovers)
        samples.extend(leftovers[: sample_size - len(samples)])

    samples.sort(key=lambda item: (item["report_date"], item["stock_code"], item["report_id"]))
    return samples


def write_gold_workset(
    *,
    clean_reports_path: Path = CLEAN_REPORTS_PATH,
    workset_path: Path = ANNOTATION_GOLD_WORKSET_PATH,
    split_targets: dict[str, int] | None = None,
) -> dict[str, object]:
    rows = _read_clean_rows(clean_reports_path)
    if split_targets is None:
        split_targets = DEFAULT_GOLD_SPLIT_TARGETS

    sampled_rows: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    year_counts: dict[str, int] = defaultdict(int)

    for split in ("train", "dev", "test"):
        target = split_targets.get(split, 0)
        split_rows = [row for row in rows if row["split"] == split]
        split_sample = _sample_within_split(split_rows, target)
        sampled_rows.extend(split_sample)
        split_counts[split] = len(split_sample)
        for row in split_sample:
            year_counts[row["report_year"]] += 1

    sampled_rows.sort(key=lambda item: (item["report_date"], item["stock_code"], item["report_id"]))
    records = [_to_annotation_record(row) for row in sampled_rows]

    workset_path.parent.mkdir(parents=True, exist_ok=True)
    with workset_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "row_count": len(records),
        "split_counts": split_counts,
        "year_counts": dict(sorted(year_counts.items())),
        "output_path": str(workset_path),
    }


def write_annotation_samples(
    pilot_size: int = 400,
    main_size: int = 2000,
    pilot_path: Path = PILOT_SAMPLE_PATH,
    main_path: Path = MAIN_SAMPLE_PATH,
    pilot_workspace_path: Path = ANNOTATION_PILOT_PATH,
    main_workspace_path: Path = ANNOTATION_MAIN_PATH,
) -> dict[str, int]:
    rows = _read_clean_rows()
    pilot_rows = _stratified_sample(rows, pilot_size)
    remaining = [row for row in rows if row["report_id"] not in {item["report_id"] for item in pilot_rows}]
    main_rows = _stratified_sample(remaining, main_size)

    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.parent.mkdir(parents=True, exist_ok=True)

    pilot_workspace_path.parent.mkdir(parents=True, exist_ok=True)
    main_workspace_path.parent.mkdir(parents=True, exist_ok=True)

    pilot_records = [_to_annotation_record(row) for row in pilot_rows]
    main_records = [_to_annotation_record(row) for row in main_rows]

    with pilot_path.open("w", encoding="utf-8") as handle:
        for row in pilot_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with main_path.open("w", encoding="utf-8") as handle:
        for row in main_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with pilot_workspace_path.open("w", encoding="utf-8") as handle:
        for row in pilot_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with main_workspace_path.open("w", encoding="utf-8") as handle:
        for row in main_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "pilot_count": len(pilot_rows),
        "main_count": len(main_rows),
    }


def write_dev_aligned_workset(
    *,
    prediction_path: Path = ASTRA_PREDICTIONS_PATH,
    output_path: Path = ANNOTATION_DEV_ALIGNED_WORKSET_PATH,
) -> dict[str, object]:
    prediction_rows = _load_prediction_rows(prediction_path)
    records = [_to_annotation_record(row) for row in prediction_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = dict(Counter(str(row["split"]) for row in prediction_rows))
    year_counts = dict(
        sorted(Counter(str(row.get("report_year") or str(row["report_date"])[:4]) for row in prediction_rows).items())
    )

    return {
        "row_count": len(records),
        "split_counts": split_counts,
        "year_counts": year_counts,
        "prediction_path": str(prediction_path),
        "output_path": str(output_path),
    }
