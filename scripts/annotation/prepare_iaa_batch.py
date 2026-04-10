#!/usr/bin/env python3
"""Prepare a batch of reports for inter-annotator agreement (IAA) double-annotation.

Reads the annotated dev workset, randomly selects 50 reports, and exports them
in annotation-template format (same schema, empty annotation fields) to
``data/annotations/iaa_batch.jsonl``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "annotations"
    / "stratreportzh_dev_aligned_workset_annotated_normalized.jsonl"
)
OUTPUT_PATH = ROOT / "data" / "annotations" / "iaa_batch.jsonl"

IAA_SAMPLE_SIZE = 50
DEFAULT_SEED = 42

EMPTY_ANNOTATION: dict = {
    "fundamental_sentiment": None,
    "strategic_optimism": None,
    "phenomenon": None,
    "evidence_spans": [],
    "annotation_confidence": None,
    "notes": "",
}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(seed: int = DEFAULT_SEED) -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    records = _load_jsonl(INPUT_PATH)
    total = len(records)
    sample_size = min(IAA_SAMPLE_SIZE, total)

    random.seed(seed)
    sampled = random.sample(records, sample_size)

    # Build annotation-template rows (same metadata, empty annotations).
    template_rows: list[dict] = []
    for rec in sampled:
        row = {
            "report_id": rec["report_id"],
            "report_date": rec.get("report_date", ""),
            "report_year": rec.get("report_year", ""),
            "split": rec.get("split", ""),
            "stock_code": rec.get("stock_code", ""),
            "company_name": rec.get("company_name", ""),
            "title": rec.get("title", ""),
            "summary": rec.get("summary", ""),
            "annotation": dict(EMPTY_ANNOTATION),
        }
        template_rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for row in template_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print summary.
    print(f"IAA batch prepared (seed={seed})")
    print(f"  Source:  {INPUT_PATH}")
    print(f"  Output:  {OUTPUT_PATH}")
    print(f"  Total available records:  {total}")
    print(f"  Selected for IAA:         {sample_size}")
    print()
    print("Selected report_ids:")
    for row in template_rows:
        title_preview = row["title"][:60] + ("..." if len(row["title"]) > 60 else "")
        print(f"  {row['report_id']}  {title_preview}")


if __name__ == "__main__":
    seed = DEFAULT_SEED
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [seed]", file=sys.stderr)
            sys.exit(1)
    main(seed=seed)
