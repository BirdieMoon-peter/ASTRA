from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from astra.config.task_schema import CLEAN_REPORTS_PATH, DATASET_OUTPUT_DIR, SPLIT_SUMMARY_PATH


def build_split_summary(
    clean_reports_path: Path = CLEAN_REPORTS_PATH,
    output_path: Path = SPLIT_SUMMARY_PATH,
) -> dict[str, object]:
    with clean_reports_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    split_counts = Counter(row["split"] for row in rows)
    years_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
        "test": Counter(),
    }
    for row in rows:
        years_by_split[row["split"]][row["report_date"][:4]] += 1

    summary = {
        "total_reports": len(rows),
        "split_counts": dict(split_counts),
        "years_by_split": {split: dict(sorted(year_counts.items())) for split, year_counts in years_by_split.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def export_split_files(clean_reports_path: Path = CLEAN_REPORTS_PATH, output_dir: Path = DATASET_OUTPUT_DIR) -> None:
    with clean_reports_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = reader.fieldnames or []
    for split in ("train", "dev", "test"):
        split_path = output_dir / f"reports_{split}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
