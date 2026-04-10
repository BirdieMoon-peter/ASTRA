from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN_REPORTS = ROOT / "data" / "interim" / "reports_clean.csv"
ANNOTATIONS = ROOT / "data" / "annotations"
OUTPUT_PATH = ROOT / "artifacts" / "audit" / "split_leakage_report.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _annotation_id_map(paths: list[Path]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for row in _read_jsonl(path):
            report_id = row.get("report_id")
            if report_id:
                mapping[report_id].append(path.name)
    return dict(mapping)


def main() -> None:
    clean_rows = _read_csv(CLEAN_REPORTS)
    split_by_id = {row["report_id"]: row.get("split", "") for row in clean_rows if row.get("report_id")}
    year_by_split = Counter(row.get("split", "") for row in clean_rows)

    duplicate_ids = [report_id for report_id, count in Counter(split_by_id.keys()).items() if count > 1]

    annotation_files = sorted(ANNOTATIONS.glob("*.jsonl"))
    annotation_map = _annotation_id_map(annotation_files)

    multi_file_ids = {
        report_id: files
        for report_id, files in annotation_map.items()
        if len(set(files)) > 1
    }

    cross_split_annotation_ids = {}
    for report_id, files in multi_file_ids.items():
        split = split_by_id.get(report_id)
        if split:
            cross_split_annotation_ids[report_id] = {
                "split": split,
                "files": sorted(set(files)),
            }

    year_policy_violations = []
    for row in clean_rows:
        report_year = int(row["report_year"])
        split = row["split"]
        if split == "train" and report_year > 2024:
            year_policy_violations.append(row["report_id"])
        elif split == "dev" and report_year != 2025:
            year_policy_violations.append(row["report_id"])
        elif split == "test" and report_year != 2026:
            year_policy_violations.append(row["report_id"])

    # Check annotation-to-split consistency
    split_ids_dir = ANNOTATIONS / "splits"
    annotation_split_violations = []
    if split_ids_dir.exists():
        split_id_sets: dict[str, set[str]] = {}
        for split_name in ("train", "dev", "test"):
            ids_path = split_ids_dir / f"{split_name}_ids.json"
            if ids_path.exists():
                split_id_sets[split_name] = set(json.loads(ids_path.read_text(encoding="utf-8")))

        for report_id, files in annotation_map.items():
            assigned_split = split_by_id.get(report_id, "")
            for other_split, id_set in split_id_sets.items():
                if other_split != assigned_split and report_id in id_set:
                    annotation_split_violations.append({
                        "report_id": report_id,
                        "assigned_split": assigned_split,
                        "also_in_split": other_split,
                        "annotation_files": sorted(set(files)),
                    })

    report = {
        "status": "ok" if not duplicate_ids and not year_policy_violations and not annotation_split_violations else "violation_detected",
        "clean_reports_path": str(CLEAN_REPORTS),
        "annotation_files": [str(path) for path in annotation_files],
        "total_clean_reports": len(clean_rows),
        "split_counts": dict(year_by_split),
        "duplicate_report_ids_in_clean_reports": duplicate_ids[:100],
        "duplicate_report_id_count": len(duplicate_ids),
        "year_policy_violation_count": len(year_policy_violations),
        "year_policy_violations": year_policy_violations[:100],
        "annotation_multi_file_overlap_count": len(multi_file_ids),
        "annotation_multi_file_overlap_examples": dict(list(multi_file_ids.items())[:100]),
        "cross_split_annotation_overlap_count": len(cross_split_annotation_ids),
        "cross_split_annotation_overlap_examples": dict(list(cross_split_annotation_ids.items())[:100]),
        "annotation_split_violation_count": len(annotation_split_violations),
        "annotation_split_violations": annotation_split_violations[:100],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
