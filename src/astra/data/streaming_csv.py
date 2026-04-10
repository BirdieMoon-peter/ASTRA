from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def iter_csv_rows(path: Path, expected_columns: Sequence[str] | None = None) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_columns is not None and reader.fieldnames != list(expected_columns):
            raise ValueError(f"Unexpected CSV columns in {path}: {reader.fieldnames}")
        for row in reader:
            yield dict(row)


def open_csv_writer(path: Path, fieldnames: Sequence[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
    writer.writeheader()
    return handle, writer
