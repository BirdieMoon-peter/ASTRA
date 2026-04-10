from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN_REPORTS = ROOT / "data" / "interim" / "reports_clean.csv"
OUTPUT_DIR = ROOT / "data" / "annotations" / "splits"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_ids = {"train": [], "dev": [], "test": []}
    with CLEAN_REPORTS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "").strip()
            report_id = row.get("report_id", "").strip()
            if split in split_ids and report_id:
                split_ids[split].append(report_id)

    summary = {}
    for split, ids in split_ids.items():
        path = OUTPUT_DIR / f"{split}_ids.json"
        path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[split] = {"count": len(ids), "path": str(path)}

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
